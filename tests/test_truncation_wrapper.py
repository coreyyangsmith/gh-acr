"""Tests for TruncatingLLMWrapper."""

from __future__ import annotations

import pytest

from src.agents.truncation_wrapper import (
    DEFAULT_PROMPT_SAFETY_BUFFER,
    TruncatingLLMWrapper,
)
from src.utils.degradation import clear_degradations, get_degradations
from tests.helpers import FakeEncoder, RecordingLLM


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_no_truncation_when_under_limit(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "left")
    inner = RecordingLLM()
    enc = FakeEncoder()
    # gpt-4o-mini has input_limit 128_000 — short prompt passes through
    wrapper = TruncatingLLMWrapper(
        inner, encoder=enc, model_name="openai/gpt-4o-mini"
    )
    prompt = _words(10)
    wrapper.invoke(prompt)
    assert inner.prompts[-1] == prompt


def test_left_truncation_keeps_tail(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "left")
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    inner = RecordingLLM()
    enc = FakeEncoder()
    wrapper = TruncatingLLMWrapper(
        inner, encoder=enc, model_name="openai/gpt-4o-mini"
    )
    # Budget already includes any safety buffer; keep exactly this many tokens.
    wrapper._allowed_prompt_tokens = lambda prompt_tokens: 20  # type: ignore[method-assign]
    prompt = _words(100)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert isinstance(truncated, str)
    assert truncated != prompt
    # Left truncation keeps the end of the token stream (synthetic tN ids)
    assert truncated.startswith("t")
    assert len(truncated.split()) == 20


def test_right_truncation_keeps_head(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "right")
    inner = RecordingLLM()
    enc = FakeEncoder()
    wrapper = TruncatingLLMWrapper(
        inner, encoder=enc, model_name="openai/gpt-4o-mini"
    )
    wrapper._allowed_prompt_tokens = lambda prompt_tokens: 16  # type: ignore[method-assign]
    prompt = _words(200)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    # keep first 16 ids → t0 .. t15
    assert truncated.startswith("t0")
    assert "t15" in truncated.split()
    assert "t100" not in truncated.split()
    assert len(truncated.split()) == 16


def test_non_string_prompt_not_truncated():
    inner = RecordingLLM()
    wrapper = TruncatingLLMWrapper(
        inner, encoder=FakeEncoder(), model_name="openai/gpt-4o-mini"
    )
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    wrapper.invoke(payload)
    assert inner.prompts[-1] is payload


def test_model_cfg_resolves_openrouter_and_groq_alias():
    wrapper_or = TruncatingLLMWrapper(
        RecordingLLM(),
        encoder=None,
        model_name="openrouter/openai/gpt-5-nano",
    )
    cfg_or = wrapper_or._model_cfg()
    assert cfg_or.get("input_limit") == 400_000

    wrapper_groq = TruncatingLLMWrapper(
        RecordingLLM(),
        encoder=None,
        model_name="groq:llama-3.1-8b-instant",
    )
    # Key in MODEL_COSTS is "groq:..." directly
    cfg_g = wrapper_groq._model_cfg()
    assert cfg_g.get("input_limit") == 128_000


def test_fallback_to_encoder_model_max_length():
    inner = RecordingLLM()
    enc = FakeEncoder(model_max_length=512)
    wrapper = TruncatingLLMWrapper(
        inner, encoder=enc, model_name="unknown/provider-model"
    )
    allowed = wrapper._allowed_prompt_tokens(10_000)
    assert allowed == 512 - 256


def test_ainvoke_truncates_string(monkeypatch):
    import asyncio

    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "left")
    inner = RecordingLLM()
    wrapper = TruncatingLLMWrapper(
        inner, encoder=FakeEncoder(), model_name="openai/gpt-4o-mini"
    )
    wrapper._allowed_prompt_tokens = lambda prompt_tokens: 20  # type: ignore[method-assign]
    asyncio.run(wrapper.ainvoke(_words(100)))
    assert len(inner.prompts) == 1
    assert inner.prompts[0] != _words(100)


@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/qwen/qwen3-32b",
        "openrouter/meta-llama/llama-3.1-8b-instruct",
    ],
)
def test_openrouter_shared_context_budget(model_name, monkeypatch):
    """allowed = min(input, total - max_tokens) - buffer = 114624."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    wrapper = TruncatingLLMWrapper(
        RecordingLLM(), encoder=FakeEncoder(), model_name=model_name
    )
    allowed = wrapper._allowed_prompt_tokens(200_000)
    expected = min(131_072, 131_072 - 16_384) - DEFAULT_PROMPT_SAFETY_BUFFER
    assert allowed == expected
    assert allowed == 114_624


def test_openrouter_qwen_truncates_over_budget(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "right")
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()

    inner = RecordingLLM()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=FakeEncoder(),
        model_name="openrouter/qwen/qwen3-32b",
    )
    # FakeEncoder counts whitespace tokens; build a prompt over the ~114.6k cap.
    budget = 114_624
    over = budget + 1_000
    prompt = _words(over)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert len(truncated.split()) == budget
    events = get_degradations()
    assert events
    assert events[0]["category"] == "prompt_truncation"
    assert "reserved_output=16384" in events[0]["detail"]
    assert f"buffer={DEFAULT_PROMPT_SAFETY_BUFFER}" in events[0]["detail"]


def test_input_limit_only_budget(monkeypatch):
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "10")
    wrapper = TruncatingLLMWrapper(
        RecordingLLM(), encoder=FakeEncoder(), model_name="openai/gpt-4o-mini"
    )
    # Patch cfg to only expose input_limit.
    wrapper._model_cfg = lambda: {"input_limit": 1000}  # type: ignore[method-assign]
    assert wrapper._allowed_prompt_tokens(5000) == 990


def test_total_limit_only_budget(monkeypatch):
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "5")
    wrapper = TruncatingLLMWrapper(
        RecordingLLM(), encoder=FakeEncoder(), model_name="openai/gpt-4o-mini"
    )
    wrapper._model_cfg = lambda: {"total_limit": 2000}  # type: ignore[method-assign]
    assert wrapper._allowed_prompt_tokens(5000) == 1995


def test_gpt5nano_budget_uses_min_of_input_and_shared(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    wrapper = TruncatingLLMWrapper(
        RecordingLLM(),
        encoder=FakeEncoder(),
        model_name="openrouter/openai/gpt-5-nano",
    )
    # min(400000, 528000-128000) - 64 = 399936
    assert wrapper._allowed_prompt_tokens(10) == 400_000 - DEFAULT_PROMPT_SAFETY_BUFFER
