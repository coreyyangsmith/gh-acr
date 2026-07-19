"""Tests for classify_failure."""

from __future__ import annotations

from src.utils.failure_classify import classify_failure


def test_classify_token_limit():
    assert (
        classify_failure(RuntimeError("maximum context length exceeded"))
        == "token_limit"
    )
    assert classify_failure("This model's token limit was hit") == "token_limit"


def test_classify_rate_limit_timeout_connection_auth():
    assert classify_failure(RuntimeError("429 rate limit")) == "rate_limit"
    assert classify_failure(TimeoutError("request timed out")) == "timeout"
    assert classify_failure(ConnectionError("connection reset by peer")) == "connection"
    assert classify_failure(PermissionError("invalid api key")) == "auth"


def test_classify_prep_flag():
    assert classify_failure(RuntimeError("something odd"), prep=True) == "prep"
    # More specific patterns still win even when prep=True
    assert (
        classify_failure(ConnectionError("connection refused"), prep=True)
        == "connection"
    )


def test_classify_other():
    assert classify_failure(ValueError("unexpected shape")) == "other"


def test_classify_credits():
    msg = (
        "Error code: 402 - {'error': {'message': 'Insufficient credits. "
        "This account never purchased credits. Make sure your key is on the "
        "correct account or org, and if so, purchase more at "
        "https://openrouter.ai/settings/credits', 'code': 402}}"
    )
    assert classify_failure(msg) == "credits"
    assert classify_failure(RuntimeError(msg)) == "credits"
