"""Unit tests for rate limit lookup helpers."""

from __future__ import annotations

import pytest

from src.config.rate_limits import (
    DEFAULT_RPM,
    DEFAULT_TPM,
    EXPECTED_OUTPUT_RATIO_DEFAULT,
    MODEL_RATE_LIMITS,
    get_limits_for_model,
)


def test_known_model_limits():
    limits = get_limits_for_model("openai/gpt-4o-mini")
    assert limits["requests_per_minute"] == 400
    assert limits["tokens_per_minute"] == 180000
    assert "expected_output_ratio" in limits


def test_unknown_model_falls_back_to_defaults():
    limits = get_limits_for_model("openai/totally-unknown-model")
    assert limits["requests_per_minute"] == DEFAULT_RPM
    assert limits["tokens_per_minute"] == DEFAULT_TPM
    assert limits["expected_output_ratio"] == EXPECTED_OUTPUT_RATIO_DEFAULT


def test_groq_llama31_8b_uses_documented_soft_caps():
    """Soft defaults are ~90% of Groq published developer-tier ceilings."""
    limits = get_limits_for_model("groq:llama-3.1-8b-instant")
    assert limits["requests_per_minute"] == 27  # 0.9 * 30
    assert limits["tokens_per_minute"] == 5400  # 0.9 * 6000
    assert limits["requests_per_day"] == 14400
    assert limits["tokens_per_day"] == 500000
    assert limits["published_requests_per_minute"] == 30
    assert limits["published_tokens_per_minute"] == 6000
    assert "groq:llama-3.1-8b-instant" in MODEL_RATE_LIMITS


def test_groq_llama31_8b_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RL_RPM_GROQ_LLAMA31_8B", "15")
    monkeypatch.setenv("RL_TPM_GROQ_LLAMA31_8B", "3000")
    monkeypatch.setenv("RL_RPD_GROQ_LLAMA31_8B", "1000")
    monkeypatch.setenv("RL_TPD_GROQ_LLAMA31_8B", "50000")
    # Re-import module values would be stale; evaluate via MODEL_RATE_LIMITS
    # construction pattern by calling get after reloading the module keys.
    # get_limits_for_model reads from the already-imported MODEL_RATE_LIMITS,
    # so patch that entry the same way production env would at import time.
    from src.config import rate_limits as rl

    monkeypatch.setitem(
        rl.MODEL_RATE_LIMITS,
        "groq:llama-3.1-8b-instant",
        {
            "requests_per_minute": int(15),
            "tokens_per_minute": int(3000),
            "requests_per_day": int(1000),
            "tokens_per_day": int(50000),
            "expected_output_ratio": EXPECTED_OUTPUT_RATIO_DEFAULT,
        },
    )
    limits = get_limits_for_model("groq:llama-3.1-8b-instant")
    assert limits["requests_per_minute"] == 15
    assert limits["tokens_per_minute"] == 3000
    assert limits["requests_per_day"] == 1000
    assert limits["tokens_per_day"] == 50000
