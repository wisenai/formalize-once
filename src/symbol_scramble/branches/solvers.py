"""The three solver branches: plain_cot, scaled, neurosymbolic (SPEC §8)."""
from __future__ import annotations

from ..grading import parse_answer
from ..llm.prompts import (
    NEUROSYM_REFINE,
    NEUROSYM_SYSTEM,
    NEUROSYM_USER,
    OPENBOOK_SYSTEM,
    OPENBOOK_USER,
    PLAIN_SYSTEM,
    PLAIN_USER,
)
from ..sandbox import run_solve
from ..schemas import BranchOutput, Variant
from .base import consensus, extract_code


class PlainCoT:
    name = "plain_cot"

    def solve(self, variant: Variant, model) -> BranchOutput:
        resp = model.complete(
            system=PLAIN_SYSTEM,
            user=PLAIN_USER.format(note=variant.note, question=variant.question),
            max_tokens=1200,
            temperature=0.0,
            reasoning="off",  # thinking OFF (plain vs scaled on identical weights)
        )
        parsed = parse_answer(resp.text, variant.output_type)
        return BranchOutput(
            variant_id=variant.variant_id,
            branch="plain_cot",
            model_id=model.model_id,
            raw_response=resp.text,
            parsed_answer=parsed,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_s=resp.latency_s,
        )


class Scaled:
    """Test-time scaling. Reasoning mode when the model supports it, else best-of-n
    self-consistency (SPEC §8) — both spend extra compute, recorded for the
    compute-matched comparison against neurosymbolic."""

    name = "scaled"

    def __init__(self, reasoning_budget: int = 3000, best_of_n: int = 5):
        self.reasoning_budget = reasoning_budget
        self.best_of_n = best_of_n

    def solve(self, variant: Variant, model) -> BranchOutput:
        user = PLAIN_USER.format(note=variant.note, question=variant.question)
        pt = ct = 0
        lat = 0.0
        if getattr(model, "reasoning_supported", False):
            resp = model.complete(
                system=PLAIN_SYSTEM,
                user=user,
                max_tokens=1500,
                reasoning=str(self.reasoning_budget),
            )
            parsed = parse_answer(resp.text, variant.output_type)
            pt, ct, lat = resp.prompt_tokens, resp.completion_tokens, resp.latency_s
            raw = resp.text
        else:
            # fire the n self-consistency samples concurrently (IO-bound)
            from concurrent.futures import ThreadPoolExecutor

            def one(_):
                return model.complete(
                    system=PLAIN_SYSTEM, user=user, max_tokens=1200, temperature=0.7
                )

            with ThreadPoolExecutor(max_workers=self.best_of_n) as ex:
                resps = list(ex.map(one, range(self.best_of_n)))
            answers = [parse_answer(r.text, variant.output_type) for r in resps]
            texts = [r.text for r in resps]
            pt = sum(r.prompt_tokens for r in resps)
            ct = sum(r.completion_tokens for r in resps)
            lat = max((r.latency_s for r in resps), default=0.0)
            parsed = consensus(answers, variant.output_type)
            raw = "\n---SAMPLE---\n".join(texts)
        return BranchOutput(
            variant_id=variant.variant_id,
            branch="scaled",
            model_id=model.model_id,
            raw_response=raw,
            parsed_answer=parsed,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=lat,
        )


class NeuroSymbolic:
    """Autoformalize the note into Python, execute in the sandbox, self-refine on
    error up to ``refine_k`` times (Logic-LM style)."""

    name = "neurosymbolic"

    def __init__(self, refine_k: int = 3):
        self.refine_k = refine_k

    def solve(self, variant: Variant, model) -> BranchOutput:
        resp = model.complete(
            system=NEUROSYM_SYSTEM,
            user=NEUROSYM_USER.format(note=variant.note, question=variant.question),
            max_tokens=1200,
            temperature=0.0,
            reasoning="off",  # isolate the code intervention from extra thinking
        )
        pt, ct, lat = resp.prompt_tokens, resp.completion_tokens, resp.latency_s
        code = extract_code(resp.text)
        exec_ok = False
        value = None
        n_refine = 0
        last_err = "no code produced"

        for attempt in range(self.refine_k + 1):
            if code is None:
                break
            sb = run_solve(code, timeout=8.0)
            if sb.ok and _is_number(sb.value):
                exec_ok, value = True, sb.value
                break
            last_err = sb.error or "non-numeric result"
            if attempt == self.refine_k:
                break
            n_refine += 1
            resp = model.complete(
                system=NEUROSYM_SYSTEM,
                user=NEUROSYM_REFINE.format(error=last_err, code=code),
                max_tokens=1200,
                temperature=0.0,
                reasoning="off",
            )
            pt += resp.prompt_tokens
            ct += resp.completion_tokens
            lat += resp.latency_s
            code = extract_code(resp.text) or code

        parsed = None
        if value is not None:
            parsed = int(round(float(value))) if variant.output_type == "integer" else float(value)

        return BranchOutput(
            variant_id=variant.variant_id,
            branch="neurosymbolic",
            model_id=model.model_id,
            raw_response=resp.text,
            parsed_answer=parsed,
            generated_code=code,
            exec_ok=exec_ok,
            n_refine=n_refine,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=lat,
        )


class OpenBook:
    """Audit intervention (Krohn-Grimberghe 2026): give the model the calculator's
    formula spec, no code. Isolates formula-recall from arithmetic-execution."""

    name = "open_book"
    _specs = None

    def _spec(self, calc_id: str) -> str:
        if OpenBook._specs is None:
            import yaml

            from .. import paths

            OpenBook._specs = yaml.safe_load(
                (paths.CONFIGS / "calculator_specs.yaml").read_text()
            )["specs"]
        return OpenBook._specs.get(str(calc_id), "")

    def solve(self, variant: Variant, model) -> BranchOutput:
        spec = self._spec(variant.calc_id)
        resp = model.complete(
            system=OPENBOOK_SYSTEM,
            user=OPENBOOK_USER.format(spec=spec, note=variant.note, question=variant.question),
            max_tokens=1200,
            temperature=0.0,
            reasoning="off",
        )
        return BranchOutput(
            variant_id=variant.variant_id,
            branch="open_book",
            model_id=model.model_id,
            raw_response=resp.text,
            parsed_answer=parse_answer(resp.text, variant.output_type),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_s=resp.latency_s,
        )


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def make_branch(name: str, config: dict):
    if name == "plain_cot":
        return PlainCoT()
    if name == "scaled":
        return Scaled(
            reasoning_budget=int(config.get("scaled_reasoning_budget", 3000)),
            best_of_n=int(config.get("scaled_best_of_n", 5)),
        )
    if name == "neurosymbolic":
        return NeuroSymbolic(refine_k=int(config.get("refine_k", 3)))
    if name == "open_book":
        return OpenBook()
    raise ValueError(f"unknown branch {name!r}")
