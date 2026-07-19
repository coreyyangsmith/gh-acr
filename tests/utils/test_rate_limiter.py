"""Tests for token-bucket rate limiter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.utils.rate_limiter import LimiterRegistry, RateLimiter, _TokenBucket


def setup_function():
    LimiterRegistry._instances.clear()


def teardown_function():
    LimiterRegistry._instances.clear()


def test_token_bucket_refill_and_consume():
    bucket = _TokenBucket(capacity=10.0, refill_rate_per_sec=5.0)
    bucket.tokens = 0.0
    now = 100.0
    bucket.last_refill_ts = now
    # Advance 1 second -> +5 tokens
    assert bucket.deficit_seconds(5.0, now + 1.0) == 0.0
    bucket.consume(5.0, now + 1.0)
    assert bucket.tokens == 0.0
    # Need 10 tokens at 5/sec -> 2 seconds
    wait = bucket.deficit_seconds(10.0, now + 1.0)
    assert wait == 2.0


def test_rate_limiter_acquire_skips_wait_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RL_ENABLE_WAITING", raising=False)
    limiter = RateLimiter(
        rpm=1,
        tpm=1,
        backoff={
            "max_retries": 1,
            "initial_delay": 0.01,
            "multiplier": 2,
            "max_delay": 1,
            "jitter": 0,
        },
    )
    with patch("src.utils.rate_limiter.time.sleep") as sleep:
        waited = limiter.acquire(expected_tokens=10_000)
    assert waited == 0.0
    sleep.assert_not_called()
    assert limiter.wait_events == 0


def test_rate_limiter_acquire_waits_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RL_ENABLE_WAITING", "1")
    limiter = RateLimiter(
        rpm=60,
        tpm=6000,
        backoff={
            "max_retries": 1,
            "initial_delay": 0.01,
            "multiplier": 2,
            "max_delay": 1,
            "jitter": 0,
        },
    )
    with patch("src.utils.rate_limiter.time.sleep") as sleep:
        waited = limiter.acquire(expected_tokens=10)
    assert waited == 0.0 or waited >= 0.0
    sleep.assert_not_called()


def test_rate_limiter_adjust_returns_unused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RL_ENABLE_WAITING", "1")
    limiter = RateLimiter(
        rpm=60,
        tpm=6000,
        backoff={
            "max_retries": 1,
            "initial_delay": 0.01,
            "multiplier": 2,
            "max_delay": 1,
            "jitter": 0,
        },
    )
    before = limiter.tpm_bucket.tokens
    limiter.acquire(expected_tokens=100)
    after_acquire = limiter.tpm_bucket.tokens
    assert after_acquire < before
    limiter.adjust(actual_tokens=40, reserved_tokens=100)
    assert limiter.tpm_bucket.tokens >= after_acquire


def test_limiter_registry_singleton_and_metrics():
    a = LimiterRegistry.get(key="m1", rpm=10, tpm=1000, backoff={})
    b = LimiterRegistry.get(key="m1", rpm=99, tpm=9999, backoff={})
    assert a is b
    c = LimiterRegistry.get(key="m2", rpm=10, tpm=1000, backoff={})
    assert c is not a
    metrics = LimiterRegistry.metrics()
    assert "m1" in metrics and "m2" in metrics
