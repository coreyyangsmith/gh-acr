"""Tests for model cost / limit config helpers."""

from __future__ import annotations

from src.config.model_costs import (
    MODEL_COSTS,
    estimate_usd_cost,
    get_model_config,
    price_key,
)


def test_openai_entry_present():
    cfg = get_model_config("openai/gpt-4o-mini")
    assert cfg["input_limit"] == 128_000
    assert "input_cost_per_1k" in cfg


def test_openrouter_entries_present():
    assert "openrouter/openai/gpt-4o-mini" in MODEL_COSTS
    assert "openrouter/anthropic/claude-sonnet-4.5" in MODEL_COSTS
    assert "openrouter/meta-llama/llama-3.1-8b-instruct" in MODEL_COSTS
    cfg = get_model_config("openrouter/anthropic/claude-sonnet-4.5")
    assert cfg.get("input_limit") == 200_000
    llama = get_model_config("openrouter/meta-llama/llama-3.1-8b-instruct")
    assert llama.get("tokenizer") == "llama"
    assert llama.get("input_cost_per_1k") == 0.00002
    assert llama.get("output_cost_per_1k") == 0.00003
    assert llama.get("input_limit") == 131_072
    assert llama.get("total_limit") == 131_072
    # $0.02 / $0.03 per 1M tokens
    assert abs(llama["input_cost_per_1k"] * 1000 - 0.02) < 1e-12
    assert abs(llama["output_cost_per_1k"] * 1000 - 0.03) < 1e-12


def test_estimate_usd_cost_openrouter_llama():
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    cost_in, cost_out, total = estimate_usd_cost(model, 1481, 759)
    assert abs(cost_in - 1481 / 1000 * 0.00002) < 1e-12
    assert abs(cost_out - 759 / 1000 * 0.00003) < 1e-12
    assert abs(total - (cost_in + cost_out)) < 1e-12


def test_price_key_preserves_openrouter_and_groq():
    assert (
        price_key("openrouter/meta-llama/llama-3.1-8b-instruct")
        == "openrouter/meta-llama/llama-3.1-8b-instruct"
    )
    assert price_key("groq:llama-3.1-8b-instant") == "groq:llama-3.1-8b-instant"
    assert price_key("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert price_key("local:foo") == "local:foo"
    assert price_key("gpt-4o-mini") == "openai/gpt-4o-mini"


def test_groq_colon_key():
    cfg = get_model_config("groq:llama-3.1-8b-instant")
    assert cfg.get("sliding_window") is True


def test_unknown_model_returns_empty():
    assert get_model_config("totally/unknown-model") == {}
    assert estimate_usd_cost("totally/unknown-model", 100, 50) == (0.0, 0.0, 0.0)
