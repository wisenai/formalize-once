"""Orchestration: generate variants, run branches, cache, budget-guard (SPEC §10)."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from . import calc_runner, paths
from .branches.solvers import make_branch
from .config import load_config, load_models, model_spec
from .data_loader import load_seed_items
from .grading import fail_mode, is_correct
from .llm.client import make_client
from .llm.prompts import PROMPT_VERSION
from .perturb.note_rewrite import NoteRewriter
from .perturb.operators import ALL_OPERATORS
from .schemas import GradedRecord, SeedItem, Variant


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostTracker:
    budget_usd: float
    spent: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, model_id: str, pt: int, ct: int) -> None:
        spec = model_spec(model_id)
        cost = pt / 1000 * spec["cost_per_1k_in"] + ct / 1000 * spec["cost_per_1k_out"]
        with self._lock:
            self.spent += cost
            self.calls += 1

    def check(self) -> None:
        if self.spent >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.spent:.2f} >= budget ${self.budget_usd:.2f}")


# ---- variant generation ------------------------------------------------------


def generate_variants(
    config_name: str = "default",
    calc_ids: Optional[list[str]] = None,
    levels: Optional[list[str]] = None,
    n: Optional[int] = None,
    max_seeds_per_calc: int = 3,
    append: bool = False,
) -> list[Variant]:
    cfg = load_config(config_name)
    levels = levels or cfg["levels"]
    n = n or cfg["n_variants_per_item"]
    paths.ensure_dirs()

    # fall back to the config's explicit calculator list when none is passed
    if calc_ids is None:
        cc = cfg.get("calculators")
        if isinstance(cc, list):
            calc_ids = [str(c) for c in cc]

    seeds = load_seed_items(calc_ids=calc_ids, max_per_calc=max_seeds_per_calc)
    needs_llm = any(ALL_OPERATORS[l].needs_llm for l in levels if l in ALL_OPERATORS)
    rewriter = None
    if needs_llm:
        rewriter = NoteRewriter(
            make_client(cfg["note_rewrite_model"]), retries=cfg["note_rewrite_retries"]
        )
    # frontier-model perturbation generation for L3/L4/Flip (clinically realistic)
    llm_perturber = None
    if cfg.get("use_llm_perturbations"):
        from .perturb.llm_perturb import LLMPerturber

        llm_perturber = LLMPerturber(make_client(cfg.get("perturb_model", "claude_sonnet5")))

    import random

    def gen_for_seed(seed: SeedItem) -> list[Variant]:
        out: list[Variant] = []
        for level in levels:
            op = ALL_OPERATORS.get(level)
            if op is None or seed.family not in op.applies_to:
                continue
            rng = random.Random(hash((cfg["seed"], seed.row_id, level)) & 0xFFFFFFFF)
            try:
                out.extend(op.generate(seed, rng, n, rewriter=rewriter,
                                       llm_perturber=llm_perturber))
            except Exception as e:  # noqa: BLE001 - never let one seed/level crash gen
                print(f"  gen skip {seed.calc_id}:{level}: {str(e)[:80]}", flush=True)
        return out

    variants: list[Variant] = []
    # Lower concurrency here: all L1 reskins hit ONE model, so high fan-out trips its
    # upstream rate limit. The multi-model run can use full concurrency.
    gen_workers = min(cfg.get("max_workers", 8), 5)
    with ThreadPoolExecutor(max_workers=gen_workers) as ex:
        for res in ex.map(gen_for_seed, seeds):
            variants.extend(res)

    if append and (paths.VARIANTS / "variants.parquet").exists():
        existing = load_variants()
        seen = {v.variant_id for v in variants}
        variants = variants + [v for v in existing if v.variant_id not in seen]
    _write_variants(variants)
    if rewriter is not None:
        print(f"note_rewrite stats: {rewriter.stats}")
    return variants


def _write_variants(variants: list[Variant]) -> None:
    import pandas as pd

    rows = [v.model_dump() for v in variants]
    for r in rows:
        r["entities"] = json.dumps(r["entities"])
        r["op_params"] = json.dumps(r["op_params"])
    df = pd.DataFrame(rows)
    df.to_parquet(paths.VARIANTS / "variants.parquet", index=False)
    # human-readable sample per level
    with open(paths.VARIANTS / "samples.jsonl", "w") as f:
        seen: dict[str, int] = {}
        for v in variants:
            if seen.get(v.level, 0) >= 5:
                continue
            seen[v.level] = seen.get(v.level, 0) + 1
            f.write(json.dumps({
                "level": v.level, "calc": v.calc_name, "gt": v.gt_answer,
                "op": v.op_params, "note": v.note[:1200],
            }) + "\n")


def load_variants() -> list[Variant]:
    import pandas as pd

    df = pd.read_parquet(paths.VARIANTS / "variants.parquet")
    out = []
    for _, r in df.iterrows():
        d = r.to_dict()
        d["entities"] = json.loads(d["entities"])
        d["op_params"] = json.loads(d["op_params"])
        out.append(Variant(**d))
    return out


# ---- solve orchestration -----------------------------------------------------


def _cache_key(variant_id: str, branch: str, model_id: str) -> str:
    import hashlib

    h = hashlib.sha256(
        f"{variant_id}|{branch}|{model_id}|{PROMPT_VERSION}".encode()
    ).hexdigest()[:20]
    return h


def _cache_path(key: str):
    return paths.CACHE / f"{key}.json"


def _grade_one(variant: Variant, branch_obj, model) -> GradedRecord:
    out = branch_obj.solve(variant, model)
    rec = GradedRecord(**out.model_dump())
    rec.correct = is_correct(out.parsed_answer, variant)
    if branch_obj.name == "neurosymbolic":
        try:
            rec.ref_answer = calc_runner.answer(variant.calc_id, variant.entities)
        except calc_runner.CalcError:
            rec.ref_answer = None
        rec.fail_mode = fail_mode(rec, variant)
    return rec


def _stratified_subset(variants: list[Variant], n: int, seed: int) -> list[Variant]:
    """Balanced sample across (family, level) for cost-bounded frontier runs."""
    import random

    rng = random.Random(seed)
    groups: dict[tuple, list[Variant]] = {}
    for v in variants:
        groups.setdefault((v.family, v.level), []).append(v)
    per = max(1, n // max(1, len(groups)))
    out: list[Variant] = []
    for g in groups.values():
        rng.shuffle(g)
        out.extend(g[:per])
    rng.shuffle(out)
    return out[:n]


def run_experiment(
    config_name: str = "default",
    branches: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    limit: Optional[int] = None,
    subset_n: Optional[int] = None,
) -> str:
    cfg = load_config(config_name)
    branches = branches or cfg.get("branches", ["plain_cot", "scaled", "neurosymbolic"])
    models = models or [m["id"] for m in cfg["models"]]
    variants = load_variants()
    if subset_n:
        variants = _stratified_subset(variants, subset_n, cfg["seed"])
    if limit:
        variants = variants[:limit]

    clients = {mid: make_client(mid) for mid in models}
    branch_objs = {b: make_branch(b, cfg) for b in branches}
    tracker = CostTracker(budget_usd=cfg["budget_usd"])

    # build the work list, skipping cached cells
    work = []
    for v in variants:
        for mid in models:
            for b in branches:
                key = _cache_key(v.variant_id, b, mid)
                if not _cache_path(key).exists():
                    work.append((v, b, mid, key))
    total_cells = len(variants) * len(models) * len(branches)
    print(f"{len(variants)} variants x {len(models)} models x {len(branches)} branches "
          f"= {total_cells} cells; {len(work)} to run ({total_cells - len(work)} cached)",
          flush=True)

    out_path = paths.RESULTS / "graded.jsonl"
    write_lock = threading.Lock()
    done = 0
    stop = threading.Event()

    def worker(item):
        v, b, mid, key = item
        if stop.is_set():
            return None
        try:
            rec = _grade_one(v, branch_objs[b], clients[mid])
        except Exception as e:  # noqa: BLE001 - record failure, keep going
            return ("error", v.variant_id, b, mid, str(e)[:200])
        tracker.add(mid, rec.prompt_tokens, rec.completion_tokens)
        payload = rec.model_dump()
        payload.update({"level": v.level, "calc_id": v.calc_id,
                        "calc_name": v.calc_name, "family": v.family,
                        "answer_invariant": v.answer_invariant})
        _cache_path(key).write_text(json.dumps(payload))
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        try:
            tracker.check()
        except BudgetExceeded:
            stop.set()
        return ("ok", v.variant_id, b, mid)

    errors = 0
    with ThreadPoolExecutor(max_workers=cfg.get("max_workers", 8)) as ex:
        futures = [ex.submit(worker, item) for item in work]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res and res[0] == "error":
                errors += 1
            if done % 25 == 0 or stop.is_set():
                print(f"  {done}/{len(work)} cells | spent ${tracker.spent:.2f} "
                      f"| errors {errors}" + (" | BUDGET STOP" if stop.is_set() else ""),
                      flush=True)
            if stop.is_set():
                break

    print(f"DONE. ran {done} cells, {errors} errors, spent ${tracker.spent:.2f} "
          f"over {tracker.calls} model calls.")
    _consolidate_cache(out_path)
    return str(out_path)


def _consolidate_cache(out_path) -> None:
    """Rebuild graded.jsonl from the cache so it always reflects all cached cells."""
    records = []
    for p in paths.CACHE.glob("*.json"):
        try:
            records.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"consolidated {len(records)} cached records -> {out_path}")
