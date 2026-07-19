"""Tests for shared prompt token budget helper."""

from __future__ import annotations

from src.agents.prompt_budget import (
    DEFAULT_PROMPT_SAFETY_BUFFER,
    allowed_prompt_tokens,
)
from tests.helpers import FakeEncoder


def test_openrouter_qwen_shared_context_budget(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("openrouter/qwen/qwen3-32b")
    assert allowed == min(131_072, 131_072 - 16_384) - DEFAULT_PROMPT_SAFETY_BUFFER
    assert allowed == 114_624


def test_local_qwen32_shared_context_budget(monkeypatch):
    """Local native window reserves 2048 for generation — not the whole context."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("local:Qwen/Qwen3-32B")
    # min(30720, 32768 - 2048) - 64 = 30656
    assert allowed == min(30_720, 32_768 - 2_048) - DEFAULT_PROMPT_SAFETY_BUFFER
    assert allowed == 30_656
    # Guard against the historical misconfig that collapsed the budget to ~704.
    assert allowed > 10_000


def test_local_qwen8_shared_context_budget(monkeypatch):
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    allowed = allowed_prompt_tokens("local:Qwen/Qwen3-8B")
    assert allowed == 30_656


def test_encoder_fallback_budget():
    enc = FakeEncoder(model_max_length=512)
    assert allowed_prompt_tokens("unknown/no-limits", encoder=enc) == 512 - 256
