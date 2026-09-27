"""symbol_scramble — neuro-symbolic robustness vs. test-time scaling harness."""
from __future__ import annotations

# Load a gitignored .env (API keys) as early as possible, without failing hard.
try:
    from dotenv import load_dotenv

    from .paths import ROOT

    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001 - dotenv is best-effort
    pass
