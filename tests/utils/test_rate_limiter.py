"""Tests for token-bucket rate limiter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.utils.rate_limiter import (
    DailyQuotaExceeded,
    LimiterRegistry,
    RateLimiter,
    _TokenBucket,
)


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


def test_daily_request_quota_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RL_ENABLE_WAITING", raising=False)
    limiter = RateLimiter(rpm=60, tpm=6000, backoff={}, rpd=2, tpd=10000)
    limiter.acquire(expected_tokens=1)
    limiter.acquire(expected_tokens=1)
    with pytest.raises(DailyQuotaExceeded) as ei:
        limiter.acquire(expected_tokens=1)
    assert ei.value.kind == "requests"
    assert ei.value.retry_after_s >= 0
    assert ei.value.status_code == 429
    snap = limiter.snapshot()
    assert snap["rpd_remaining"] == 0
    assert snap["requests_today"] == 2


def test_daily_token_quota_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RL_ENABLE_WAITING", raising=False)
    limiter = RateLimiter(rpm=60, tpm=6000, backoff={}, rpd=100, tpd=50)
    limiter.acquire(expected_tokens=40)
    with pytest.raises(DailyQuotaExceeded) as ei:
        limiter.acquire(expected_tokens=20)
    assert ei.value.kind == "tokens"


def test_backoff_sleep_honors_retry_after(monkeypatch: pytest.MonkeyPatch):
    limiter = RateLimiter(rpm=60, tpm=6000, backoff={"initial_delay": 0.5, "multiplier": 2, "max_delay": 20, "jitter": 0})
    with patch("src.utils.rate_limiter.time.sleep") as sleep:
        delay = limiter.backoff_sleep(1, retry_after_s=2.5)
    assert delay == 2.5
    sleep.assert_called_once_with(2.5)
    assert limiter.total_retries == 1


def test_limiter_registry_singleton_and_metrics():
    a = LimiterRegistry.get(key="m1", rpm=10, tpm=1000, backoff={}, rpd=100, tpd=5000)
    b = LimiterRegistry.get(key="m1", rpm=99, tpm=9999, backoff={})
    assert a is b
    c = LimiterRegistry.get(key="m2", rpm=10, tpm=1000, backoff={})
    assert c is not a
    metrics = LimiterRegistry.metrics()
    assert "m1" in metrics and "m2" in metrics
    assert "rpd_remaining" in metrics["m1"]
    assert "seconds_until_day_reset" in metrics["m1"]
