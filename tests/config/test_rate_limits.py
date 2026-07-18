"""Unit tests for rate limit lookup helpers."""

from __future__ import annotations

from src.config.rate_limits import (
    DEFAULT_RPM,
    DEFAULT_TPM,
    EXPECTED_OUTPUT_RATIO_DEFAULT,
    get_limits_for_model,
)


def test_known_model_limits():
    limits = get_limits_for_model("openai/gpt-4o-mini")
    assert limits["requests_per_minute"] == 100
    assert limits["tokens_per_minute"] == 200000
    assert "expected_output_ratio" in limits


def test_unknown_model_falls_back_to_defaults():
    limits = get_limits_for_model("openai/totally-unknown-model")
    assert limits["requests_per_minute"] == DEFAULT_RPM
    assert limits["tokens_per_minute"] == DEFAULT_TPM
    assert limits["expected_output_ratio"] == EXPECTED_OUTPUT_RATIO_DEFAULT


def test_scheme_prefixed_unknown_still_defaults():
    limits = get_limits_for_model("groq:llama-3.1-8b-instant")
    assert limits["requests_per_minute"] == DEFAULT_RPM
