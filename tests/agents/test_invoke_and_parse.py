"""Tests for invoke_and_parse and credits fail-fast classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.resilient_invoke import (
    ParseExhausted,
    ParsedResult,
    invoke_and_parse,
    is_retryable_error,
)
from src.utils.failure_classify import classify_failure


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if self.calls >= len(self.responses):
            raise RuntimeError("no more fake responses")
        text = self.responses[self.calls]
        self.calls += 1
        return SimpleNamespace(content=text)


def test_invoke_and_parse_success_first_attempt():
    llm = _FakeLLM(["A"])

    def parse_fn(raw: str):
        if raw.strip() == "A":
            return ParsedResult("ALL_A", "strict_first_line")
        raise ValueError("bad")

    parsed, raw, log = invoke_and_parse(llm, "prompt", parse_fn=parse_fn)
    assert parsed == "ALL_A"
    assert raw == "A"
    assert llm.calls == 1
    assert len(log) == 1
    assert log[0]["parse_ok"] is True


def test_invoke_and_parse_recovers_on_third_attempt():
    llm = _FakeLLM(["noise", "still bad", "A"])

    def parse_fn(raw: str):
        if raw.strip() == "A":
            return "ALL_A"
        raise ValueError("unparseable")

    parsed, raw, log = invoke_and_parse(
        llm,
        "prompt",
        parse_fn=parse_fn,
        max_parse_attempts=3,
        repair_hint="Return only A",
    )
    assert parsed == "ALL_A"
    assert raw == "A"
    assert llm.calls == 3
    assert len(log) == 3
    assert log[0]["parse_ok"] is False
    assert log[2]["parse_ok"] is True
    assert "Return only A" in llm.prompts[1]


def test_invoke_and_parse_exhausted():
    llm = _FakeLLM(["x", "y", "z"])

    def parse_fn(_raw: str):
        raise ValueError("nope")

    with pytest.raises(ParseExhausted) as ei:
        invoke_and_parse(llm, "prompt", parse_fn=parse_fn, max_parse_attempts=3)
    assert llm.calls == 3
    assert len(ei.value.attempt_log) == 3
    assert ei.value.raw_text == "z"


def test_is_retryable_error_credits_fail_fast():
    err = RuntimeError(
        "Error code: 402 - {'error': {'message': 'Insufficient credits. "
        "This account never purchased credits.', 'code': 402}}"
    )
    assert is_retryable_error(err) is False

    class APIStatusError(Exception):
        def __init__(self):
            super().__init__("Insufficient credits")
            self.status_code = 402

    assert is_retryable_error(APIStatusError()) is False


def test_classify_failure_credits():
    assert (
        classify_failure(
            "Error code: 402 - Insufficient credits. Purchase more at openrouter"
        )
        == "credits"
    )
    assert classify_failure(RuntimeError("payment required")) == "credits"


def test_extract_retry_after_from_attribute():
    from src.agents.resilient_invoke import extract_retry_after_seconds

    err = RuntimeError("rate limited")
    err.retry_after_s = 3.5  # type: ignore[attr-defined]
    assert extract_retry_after_seconds(err) == 3.5


def test_extract_retry_after_from_headers():
    from src.agents.resilient_invoke import extract_retry_after_seconds

    class Err(Exception):
        def __init__(self):
            super().__init__("429")
            self.status_code = 429
            self.headers = {"Retry-After": "2"}

    assert extract_retry_after_seconds(Err()) == 2.0


def test_extract_retry_after_from_message():
    from src.agents.resilient_invoke import extract_retry_after_seconds

    err = RuntimeError("Error 429: retry-after: 7")
    assert extract_retry_after_seconds(err) == 7.0


def test_resilient_invoke_uses_retry_after(monkeypatch, tmp_path):
    from src.agents import resilient_invoke as ri

    class RateLimitedThenOk:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt: str):
            self.calls += 1
            if self.calls == 1:
                err = RuntimeError("Error code: 429 - rate limit")
                err.status_code = 429  # type: ignore[attr-defined]
                err.headers = {"Retry-After": "1.25"}  # type: ignore[attr-defined]
                raise err
            return SimpleNamespace(content="ok")

    sleeps: list[float] = []

    def fake_sleep(seconds: float):
        sleeps.append(float(seconds))

    monkeypatch.setattr(ri.time, "sleep", fake_sleep)
    # Avoid writing failure traces under repo logs/
    monkeypatch.setattr(ri, "_FAILURES_DIR", tmp_path)

    llm = RateLimitedThenOk()
    result = ri.resilient_invoke(
        llm,
        "prompt",
        context={"model_name": None, "node": "test"},
        max_retries=2,
    )
    assert result.content == "ok"
    assert llm.calls == 2
    assert sleeps == [1.25]


def test_resilient_invoke_cancelled_fails_fast(monkeypatch, tmp_path):
    from src.agents import resilient_invoke as ri
    from src.utils.run_heartbeat import (
        WatchdogTimeout,
        clear_cancelled_units,
        mark_unit_cancelled,
    )

    clear_cancelled_units()
    mark_unit_cancelled("scen-1", "agent", reason="watchdog soft-skip")

    class NeverCall:
        def invoke(self, prompt: str):
            raise AssertionError("should not invoke when already cancelled")

    monkeypatch.setattr(ri, "_FAILURES_DIR", tmp_path)
    with pytest.raises(WatchdogTimeout):
        ri.resilient_invoke(
            NeverCall(),
            "prompt",
            context={
                "model_name": None,
                "node": "test",
                "scenario_id": "scen-1",
                "eval_method": "agent",
            },
            max_retries=5,
        )
    clear_cancelled_units()


def test_resilient_invoke_timeout_then_cancel_skips_retry(monkeypatch, tmp_path):
    from src.agents import resilient_invoke as ri
    from src.utils.run_heartbeat import (
        WatchdogTimeout,
        clear_cancelled_units,
        mark_unit_cancelled,
    )

    clear_cancelled_units()
    sleeps: list[float] = []
    monkeypatch.setattr(ri.time, "sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setattr(ri, "_FAILURES_DIR", tmp_path)

    class TimeoutOnce:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt: str):
            self.calls += 1
            mark_unit_cancelled("scen-2", "agent", reason="llm overtime")
            err = TimeoutError("request timed out")
            err.status_code = 408  # type: ignore[attr-defined]
            raise err

    llm = TimeoutOnce()
    with pytest.raises(WatchdogTimeout):
        ri.resilient_invoke(
            llm,
            "prompt",
            context={
                "model_name": None,
                "node": "test",
                "scenario_id": "scen-2",
                "eval_method": "agent",
            },
            max_retries=5,
        )
    assert llm.calls == 1
    assert sleeps == []
    clear_cancelled_units()


def test_resilient_invoke_timeout_without_cancel_may_retry(monkeypatch, tmp_path):
    from src.agents import resilient_invoke as ri
    from src.utils.run_heartbeat import clear_cancelled_units

    clear_cancelled_units()
    sleeps: list[float] = []
    monkeypatch.setattr(ri.time, "sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setattr(ri, "_FAILURES_DIR", tmp_path)

    class TimeoutThenOk:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt: str):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("request timed out")
            return SimpleNamespace(content="ok")

    llm = TimeoutThenOk()
    result = ri.resilient_invoke(
        llm,
        "prompt",
        context={
            "model_name": None,
            "node": "test",
            "scenario_id": "scen-3",
            "eval_method": "agent",
        },
        max_retries=2,
    )
    assert result.content == "ok"
    assert llm.calls == 2
    assert len(sleeps) == 1
    clear_cancelled_units()
