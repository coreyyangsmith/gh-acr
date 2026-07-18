"""Tests for model cost / limit config helpers."""

from __future__ import annotations

from src.config.model_costs import MODEL_COSTS, get_model_config


def test_openai_entry_present():
    cfg = get_model_config("openai/gpt-4o-mini")
    assert cfg["input_limit"] == 128_000
    assert "input_cost_per_1k" in cfg


def test_openrouter_entries_present():
    assert "openrouter/openai/gpt-4o-mini" in MODEL_COSTS
    assert "openrouter/anthropic/claude-sonnet-4.5" in MODEL_COSTS
    cfg = get_model_config("openrouter/anthropic/claude-sonnet-4.5")
    assert cfg.get("input_limit") == 200_000


def test_groq_colon_key():
    cfg = get_model_config("groq:llama-3.1-8b-instant")
    assert cfg.get("sliding_window") is True


def test_unknown_model_returns_empty():
    assert get_model_config("totally/unknown-model") == {}
