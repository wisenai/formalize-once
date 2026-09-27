"""LLMClient protocol + Anthropic implementation (IMPLEMENTATION_SPEC §1, §8).

Experiment code NEVER imports the vendor SDK directly — it goes through this
protocol so branches/perturbation are provider-agnostic.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


def _require_spend_opt_in(model_id):
    """Raise unless the caller has explicitly opted in to spending money."""
    if os.environ.get("SSCRAMBLE_ALLOW_SPEND"):
        return
    raise RuntimeError(
        f"refusing to make a paid API call for {model_id}: this call is not in the "
        f"local cache, so completing it would cost money.\n"
        f"Reproducing the paper's numbers needs no API calls at all -- run "
        f"`python reproduce.py`.\n"
        f"To re-run an experiment against live endpoints, set SSCRAMBLE_ALLOW_SPEND=1.")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model_id: str = ""
    finish_reason: str = ""  # "length" => completion ceiling hit (may leave text empty)


class LLMClient(Protocol):
    model_id: str

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        reasoning: Optional[str] = None,
    ) -> LLMResponse: ...


class _Retryable(Exception):
    pass


@dataclass
class AnthropicClient:
    model_id: str
    model: str
    reasoning_supported: bool = False
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=key)

    @retry(
        retry=retry_if_exception_type(_Retryable),
        wait=wait_random_exponential(min=3, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        reasoning: Optional[str] = None,
    ) -> LLMResponse:
        import time

        import anthropic

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Extended-thinking (the `scaled` branch) when a budget is requested.
        if reasoning and self.reasoning_supported:
            budget = int(reasoning)
            kwargs["max_tokens"] = max(max_tokens, budget + 1024)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["temperature"] = 1.0  # required when thinking is on
        else:
            kwargs["temperature"] = temperature

        retryable = (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.APITimeoutError,
        )
        _require_spend_opt_in(self.model_id)          # see the note on the other provider path
        t0 = time.time()
        try:
            msg = self._client.messages.create(**kwargs)
        except retryable as e:
            raise _Retryable(str(e)) from e
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) in (429, 500, 502, 503, 529):
                raise _Retryable(str(e)) from e
            raise
        dt = time.time() - t0

        text_parts = [
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ]
        text = "\n".join(text_parts)
        return LLMResponse(
            text=text,
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            latency_s=dt,
            model_id=self.model_id,
        )


@dataclass
class OpenAICompatibleClient:
    """Works for OpenRouter, Ollama (/v1), Together, vLLM — any OpenAI-style API."""

    model_id: str
    model: str
    base_url: str
    api_key_env: str
    reasoning_supported: bool = False
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        import openai

        key = os.environ.get(self.api_key_env, "ollama")  # ollama ignores the key
        self._client = openai.OpenAI(base_url=self.base_url, api_key=key)

    @retry(
        retry=retry_if_exception_type(_Retryable),
        wait=wait_random_exponential(min=3, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        reasoning: Optional[str] = None,
    ) -> LLMResponse:
        import time

        import openai

        # Qwen3 providers don't always honor reasoning:{enabled:false} (they burn the
        # whole budget thinking and return empty content); the `/no_think` directive
        # reliably disables thinking and is harmless to non-Qwen models.
        user_content = user
        if reasoning == "off" and self.reasoning_supported:
            user_content = user + "\n/no_think"

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        # reasoning: "off" forces thinking off (plain branch on toggle models);
        # a numeric budget forces thinking on (scaled branch); None = provider default.
        if self.reasoning_supported:
            if reasoning == "off":
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            elif reasoning is not None:
                kwargs["extra_body"] = {"reasoning": {"enabled": True, "max_tokens": int(reasoning)}}
                # leave generous room for the answer AFTER the reasoning budget so
                # thinking models don't exhaust the budget with an empty content field
                kwargs["max_tokens"] = max(max_tokens, int(reasoning) + 2048)

        # Every paid call in this project passes through here, so this is the one
        # place that can promise a clone costs nothing. Reproducing the paper never
        # reaches it -- tools/*.py read the stored result records, and the RuleArena
        # disk cache is consulted before this method is called, so resuming a run
        # still replays completed calls for free. What this stops is a probe script
        # run out of curiosity silently buying tokens.
        _require_spend_opt_in(self.model_id)
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, openai.APITimeoutError) as e:
            raise _Retryable(str(e)) from e
        except openai.BadRequestError as e:
            # some endpoints (GPT-5, o-series) mandate reasoning and reject enabled:false;
            # retry once letting the model reason (drop the reasoning override)
            if "reasoning is mandatory" in str(e).lower() and "extra_body" in kwargs:
                kwargs.pop("extra_body", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        except openai.APIStatusError as e:
            if getattr(e, "status_code", 0) in (429, 500, 502, 503, 529):
                raise _Retryable(str(e)) from e
            raise
        except (json.JSONDecodeError, openai.APIError) as e:
            # OpenRouter occasionally returns a truncated/HTML/non-JSON body on a
            # transient upstream hiccup; the raw JSONDecodeError (or a generic
            # APIError not matched above) escapes the specific handlers. Retry it.
            raise _Retryable(str(e)) from e
        dt = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_s=dt,
            model_id=self.model_id,
            finish_reason=resp.choices[0].finish_reason or "",
        )


def make_client(model_id: str) -> LLMClient:
    from ..config import model_spec

    spec = model_spec(model_id)
    provider = spec["provider"]
    if provider == "anthropic":
        return AnthropicClient(
            model_id=model_id,
            model=spec["model"],
            reasoning_supported=spec.get("reasoning_supported", False),
        )
    if provider in ("openrouter", "ollama", "openai_compatible"):
        base_url = spec.get("base_url") or (
            "https://openrouter.ai/api/v1" if provider == "openrouter"
            else "http://localhost:11434/v1"
        )
        return OpenAICompatibleClient(
            model_id=model_id,
            model=spec["model"],
            base_url=base_url,
            api_key_env=spec.get("api_key_env", "OPENROUTER_API_KEY"),
            reasoning_supported=spec.get("reasoning_supported", False),
        )
    raise NotImplementedError(f"provider {provider!r} not wired up")
