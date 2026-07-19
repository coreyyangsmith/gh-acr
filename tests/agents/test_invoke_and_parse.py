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
