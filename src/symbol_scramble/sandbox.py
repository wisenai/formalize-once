"""Subprocess sandbox for UNTRUSTED, model-generated Python (neuro-symbolic branch).

Runs code in a fresh interpreter with a wall-clock timeout and no inherited
environment (so no API keys / network creds leak in). The model code is expected
to define ``solve()`` and we call it; the result is returned as JSON on stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

_HARNESS = r'''
import json, sys, math
_ns = {{"__name__": "__sandbox__"}}
try:
    exec(compile({code!r}, "<model_code>", "exec"), _ns)
    if "solve" not in _ns or not callable(_ns["solve"]):
        print(json.dumps({{"ok": False, "error": "no solve() defined"}}))
        sys.exit(0)
    val = _ns["solve"]()
    if isinstance(val, bool):
        val = int(val)
    print(json.dumps({{"ok": True, "value": val}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
'''


@dataclass
class SandboxResult:
    ok: bool
    value: Optional[Any] = None
    error: Optional[str] = None


def run_solve(code: str, timeout: float = 8.0) -> SandboxResult:
    """Execute model code defining ``solve()`` and return its value."""
    script = _HARNESS.format(code=code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(script)
        f.flush()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, error="timeout")
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return SandboxResult(ok=False, error=(proc.stderr or "no output")[:300])
    try:
        payload = json.loads(out[-1])
    except json.JSONDecodeError:
        return SandboxResult(ok=False, error=("bad json: " + out[-1])[:300])
    if payload.get("ok"):
        return SandboxResult(ok=True, value=payload.get("value"))
    return SandboxResult(ok=False, error=payload.get("error", "unknown"))
