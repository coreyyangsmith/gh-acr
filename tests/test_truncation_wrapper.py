"""Tests for TruncatingLLMWrapper."""

from __future__ import annotations

import pytest

from src.agents.token_utils import estimate_prompt_tokens
from src.agents.truncation_wrapper import (
    DEFAULT_PROMPT_SAFETY_BUFFER,
    TruncatingLLMWrapper,
)
from src.utils.degradation import clear_degradations, get_degradations
from tests.helpers import FakeEncoder, RecordingLLM

OPENROUTER_SHARED_BUDGET = min(131_072, 131_072 - 16_384) - DEFAULT_PROMPT_SAFETY_BUFFER
LOCAL_QWEN_BUDGET = min(30_720, 32_768 - 2_048) - DEFAULT_PROMPT_SAFETY_BUFFER


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
    assert estimate_prompt_tokens(enc, truncated) <= 20
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
    assert estimate_prompt_tokens(enc, truncated) <= 16
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
    """allowed = min(input, total - max_tokens) - buffer = 110592."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    wrapper = TruncatingLLMWrapper(
        RecordingLLM(), encoder=FakeEncoder(), model_name=model_name
    )
    allowed = wrapper._allowed_prompt_tokens(200_000)
    expected = OPENROUTER_SHARED_BUDGET
    assert allowed == expected
    assert allowed == 110_592


def test_local_qwen32_shared_context_budget(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    wrapper = TruncatingLLMWrapper(
        RecordingLLM(),
        encoder=FakeEncoder(),
        model_name="local:Qwen/Qwen3-32B",
    )
    allowed = wrapper._allowed_prompt_tokens(200_000)
    expected = LOCAL_QWEN_BUDGET
    assert allowed == expected
    assert allowed == 26_624


def test_local_qwen32_truncates_over_budget_without_crash(monkeypatch):
    """Over-budget local prompts clip cleanly; must not raise."""
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "left")
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()

    inner = RecordingLLM()
    enc = FakeEncoder()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=enc,
        model_name="local:Qwen/Qwen3-32B",
    )
    budget = LOCAL_QWEN_BUDGET
    prompt = _words(budget + 5_000)
    # Must not raise even for large synthetic prompts.
    result = wrapper.invoke(prompt)
    assert result is not None
    truncated = inner.prompts[-1]
    assert isinstance(truncated, str)
    assert estimate_prompt_tokens(enc, truncated) <= budget
    events = get_degradations()
    assert events
    assert events[0]["category"] == "prompt_truncation"
    assert "reserved_output=2048" in events[0]["detail"]
    assert "model=local:Qwen/Qwen3-32B" in events[0]["detail"]


def test_local_qwen32_encode_failure_falls_back_to_chars(monkeypatch):
    """If the HF tokenizer blows up, char clipping still enforces the budget."""
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "right")
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    clear_degradations()

    class BoomEncoder(FakeEncoder):
        def encode(self, text: str):
            raise MemoryError("simulated tokenizer OOM")

        def decode(self, ids):
            raise MemoryError("simulated tokenizer OOM")

    inner = RecordingLLM()
    enc = BoomEncoder()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=enc,
        model_name="local:Qwen/Qwen3-32B",
    )
    # Force a tiny allowed budget so truncation path is taken.
    wrapper._allowed_prompt_tokens = lambda prompt_tokens: 12  # type: ignore[method-assign]
    prompt = _words(200)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert isinstance(truncated, str)
    assert truncated != prompt
    assert estimate_prompt_tokens(enc, truncated) <= 12
    assert get_degradations()


def test_openrouter_qwen_truncates_over_budget(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "right")
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()

    inner = RecordingLLM()
    enc = FakeEncoder()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=enc,
        model_name="openrouter/qwen/qwen3-32b",
    )
    budget = OPENROUTER_SHARED_BUDGET
    over = budget + 1_000
    prompt = _words(over)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert estimate_prompt_tokens(enc, truncated) <= budget
    events = get_degradations()
    assert events
    assert events[0]["category"] == "prompt_truncation"
    assert "reserved_output=16384" in events[0]["detail"]
    assert f"buffer={DEFAULT_PROMPT_SAFETY_BUFFER}" in events[0]["detail"]


def test_openrouter_llama_truncates_over_budget(monkeypatch):
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "right")
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()

    inner = RecordingLLM()
    enc = FakeEncoder()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=enc,
        model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    )
    budget = OPENROUTER_SHARED_BUDGET
    prompt = _words(budget + 2_000)
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert estimate_prompt_tokens(enc, truncated) <= budget
    cfg = wrapper._model_cfg()
    assert (
        estimate_prompt_tokens(enc, truncated) + int(cfg["output_limit"])
        <= int(cfg["total_limit"])
    )
    assert get_degradations()


def test_undercounting_encoder_still_truncates_via_chars4(monkeypatch):
    """When HF reports far fewer tokens than chars/4, still clip to budget."""
    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "left")
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()

    class UndercountEncoder:
        """Reports ~1 token per 20 chars (severe undercount vs chars/4)."""

        def encode(self, text: str):
            n = max(1, len(text) // 20) if text else 0
            return list(range(n))

        def decode(self, ids):
            # Cannot reconstruct; force char-clip path after id attempt.
            return "x" * max(1, len(ids) * 20)

    inner = RecordingLLM()
    enc = UndercountEncoder()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=enc,
        model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    )
    budget = OPENROUTER_SHARED_BUDGET
    # Dense text: chars/4 >> undercount encode length; must still truncate.
    prompt = "a" * ((budget + 5_000) * 4)
    assert estimate_prompt_tokens(enc, prompt) > budget
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert isinstance(truncated, str)
    assert estimate_prompt_tokens(enc, truncated) <= budget
    events = get_degradations()
    assert events
    assert events[0]["category"] == "prompt_truncation"


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
    # min(400000, 528000-128000) - 4096 = 395904
    assert wrapper._allowed_prompt_tokens(10) == 400_000 - DEFAULT_PROMPT_SAFETY_BUFFER
