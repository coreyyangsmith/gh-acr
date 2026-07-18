"""Tests for TruncatingLLMWrapper."""

from __future__ import annotations

import pytest

from src.agents.truncation_wrapper import TruncatingLLMWrapper
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
    inner = RecordingLLM()
    enc = FakeEncoder()
    # Force a tiny budget via openrouter model entry or patch _allowed
    wrapper = TruncatingLLMWrapper(
        inner, encoder=enc, model_name="openai/gpt-4o-mini"
    )
    wrapper._allowed_prompt_tokens = lambda prompt_tokens: 20  # type: ignore[method-assign]
    prompt = _words(100)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    # target = allowed - 64 buffer → max(1, 20-64)=1 token kept from the tail
    assert isinstance(truncated, str)
    assert truncated != prompt
    # Left truncation keeps the end of the token stream (synthetic tN ids)
    assert truncated.startswith("t")


def test_right_truncation_keeps_head(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "right")
    inner = RecordingLLM()
    enc = FakeEncoder()
    wrapper = TruncatingLLMWrapper(
        inner, encoder=enc, model_name="openai/gpt-4o-mini"
    )
    wrapper._allowed_prompt_tokens = lambda prompt_tokens: 80  # type: ignore[method-assign]
    prompt = _words(200)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    # target = 80 - 64 = 16 → keep first 16 ids → t0 .. t15
    assert truncated.startswith("t0")
    assert "t15" in truncated.split()
    assert "t100" not in truncated.split()


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
        model_name="openrouter/openai/gpt-4o-mini",
    )
    cfg_or = wrapper_or._model_cfg()
    assert cfg_or.get("input_limit") == 128_000

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
