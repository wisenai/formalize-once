"""run()'s resume must not treat an abandoned cell as a finished one.

A cell cut short by a credit cap or an outage is written with its unrun cases
flagged in per_case_plain_err. Resuming past it would leave the cell permanently
incomplete while the run reports success.
"""
import json

from symbol_scramble.rulearena import core_experiment as C


def _row(model, **kw):
    r = {"model": model, "level": 0, "n": 90, "plain_fewshot": True,
         "plain_budget": "3000", "plain_sc": 1, "code_extracted": True,
         "per_case_plain_err": [0] * 90}
    r.update(kw)
    return r


def done_set(tmp_path, rows, monkeypatch):
    """Exercise run()'s resume scan against a results file, without any API."""
    out = tmp_path / "airline_core.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(C, "RESULTS", tmp_path)
    seen = {}

    def fake_client(mid):
        seen.setdefault("called", []).append(mid)
        raise RuntimeError("stop before spending")

    monkeypatch.setattr(C, "make_client", fake_client)
    try:
        C.run(["a", "b"], domain="airline", n_test=90, plain_fewshot=True,
              plain_budget="3000", plain_sc=1)
    except RuntimeError:
        pass
    return seen.get("called", [])


def test_complete_cell_is_skipped(tmp_path, monkeypatch):
    called = done_set(tmp_path, [_row("a")], monkeypatch)
    assert called == ["b"], called          # 'a' is done, 'b' still needs running


def test_abandoned_cell_is_rerun(tmp_path, monkeypatch):
    partial = _row("a", per_case_plain_err=[0] * 3 + [1] * 87)
    called = done_set(tmp_path, [partial], monkeypatch)
    assert called == ["a"], called          # must not resume past the unrun cell
