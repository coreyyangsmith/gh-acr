"""Tests for shared prompt token budget helper."""

from __future__ import annotations

from src.agents.prompt_budget import (
    DEFAULT_PROMPT_SAFETY_BUFFER,
    allowed_prompt_tokens,
)
from tests.helpers import FakeEncoder

# OpenRouter shared-context models: total 131072, output 16384, buffer 4096.
OPENROUTER_SHARED_BUDGET = min(131_072, 131_072 - 16_384) - DEFAULT_PROMPT_SAFETY_BUFFER
LOCAL_QWEN_BUDGET = min(30_720, 32_768 - 2_048) - DEFAULT_PROMPT_SAFETY_BUFFER


def test_openrouter_qwen_shared_context_budget(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("openrouter/qwen/qwen3-32b")
    assert allowed == OPENROUTER_SHARED_BUDGET
    assert allowed == 110_592


def test_openrouter_llama_shared_context_budget(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("openrouter/meta-llama/llama-3.1-8b-instruct")
    assert allowed == OPENROUTER_SHARED_BUDGET
    assert allowed == 110_592


def test_local_qwen32_shared_context_budget(monkeypatch):
    """Local native window reserves 2048 for generation — not the whole context."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("local:Qwen/Qwen3-32B")
    # min(30720, 32768 - 2048) - 4096 = 26624
    assert allowed == LOCAL_QWEN_BUDGET
    assert allowed == 26_624
    # Guard against the historical misconfig that collapsed the budget to ~704.
    assert allowed > 10_000


def test_local_qwen8_shared_context_budget(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("local:Qwen/Qwen3-8B")
    assert allowed == LOCAL_QWEN_BUDGET


def test_local_llama_shared_context_budget_does_not_collapse(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    for model in (
        "local:meta-llama/Llama-3.1-8B-Instruct",
        "local:meta-llama/Llama-3.1-8B",
        "local:meta-llama/Llama-3.2-1B",
    ):
        allowed = allowed_prompt_tokens(model)
        assert allowed > 10_000, f"{model} collapsed to {allowed}"


def test_groq_llama_sliding_window_budget_does_not_collapse(monkeypatch):
    """Groq advertises output_limit ≈ total; sliding_window must not zero the budget."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("groq:llama-3.1-8b-instant")
    assert allowed == 128_000 - DEFAULT_PROMPT_SAFETY_BUFFER
    assert allowed > 10_000


def test_encoder_fallback_budget():
    enc = FakeEncoder(model_max_length=512)
    assert allowed_prompt_tokens("unknown/no-limits", encoder=enc) == 512 - 256
