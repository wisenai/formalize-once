"""A 402 can mean two opposite things and they need opposite handling.

An empty balance is terminal: every later call fails the same way, so the run must
halt with its progress intact rather than fill each remaining cell with errors. A
concurrency-budget 402 is transient -- too many in-flight requests are each reserving
credit -- and clears as calls settle, so it must be waited out or a cell that would
have completed is thrown away.
"""
import pytest

from symbol_scramble.rulearena import core_experiment as C

EMPTY_BALANCE = ("Error code: 402 - {'error': {'message': \"This request requires "
                 "more credits, or fewer max_tokens. You requested up to 60000 "
                 "tokens, but can only afford 19657\", 'code': 402}}")
IN_FLIGHT = ("Error code: 402 - {'error': {'message': 'This request would exceed "
             "your available credits given your current in-flight requests.', "
             "'code': 402, 'metadata': {'reason': 'in_flight_budget_exhausted', "
             "'headers': {'Retry-After': '120'}}}}")


def is_terminal(msg):
    return any(s in msg.lower() for s in C.TERMINAL)


def test_empty_balance_is_terminal():
    assert is_terminal(EMPTY_BALANCE)


def test_in_flight_budget_is_not_terminal():
    """The bug this pins: halting here abandoned a run whose balance was fine."""
    assert not is_terminal(IN_FLIGHT)
    assert C.IN_FLIGHT in IN_FLIGHT


def test_connection_error_is_not_terminal():
    assert not is_terminal("_Retryable: Connection error.")


def test_in_flight_is_waited_out_then_succeeds(monkeypatch):
    calls, slept = [], []
    monkeypatch.setattr(C.time, "sleep", lambda s: slept.append(s))

    class M:
        def complete(self, system, user, **kw):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError(IN_FLIGHT)
            return "ok"

    assert C._complete_waiting_out_in_flight(M(), "s", "u", {}) == "ok"
    assert len(calls) == 3
    assert slept == [120, 240], slept       # Retry-After honoured, backing off


def test_a_real_error_still_raises_immediately(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)

    class M:
        def complete(self, system, user, **kw):
            raise RuntimeError(EMPTY_BALANCE)

    with pytest.raises(RuntimeError, match="more credits"):
        C._complete_waiting_out_in_flight(M(), "s", "u", {})
