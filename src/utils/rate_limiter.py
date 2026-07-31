"""Thread-safe rate limiting for LLM API calls.

This module provides a token-bucket based rate limiter that can enforce both
requests-per-minute (RPM) and tokens-per-minute (TPM) limits, plus optional
daily request/token quotas (RPD/TPD). Client-side waiting is **disabled by
default** (``RL_ENABLE_WAITING`` unset/false); set ``RL_ENABLE_WAITING=1`` to
restore blocking acquire behavior for minute buckets.

When waiting is enabled, the limiter uses two token buckets:
1. **RPM bucket**: Refills at requests/minute rate
2. **TPM bucket**: Refills at tokens/minute rate

Optional daily counters (UTC calendar day) refuse new acquires when exhausted
by raising :class:`DailyQuotaExceeded` rather than sleeping until midnight.

Classes
-------
- **RateLimiter**: Main rate limiter with dual bucket enforcement
- **LimiterRegistry**: Global registry for sharing limiters across threads

Usage Pattern
-------------
Typically used via the callback system in llm_base.py, but can be used directly::

    from src.utils.rate_limiter import LimiterRegistry
    from src.config.rate_limits import BACKOFF_SETTINGS

    # Get or create a limiter for a model
    limiter = LimiterRegistry.get(
        key="openai/gpt-4o-mini",
        rpm=100,
        tpm=200000,
        backoff=BACKOFF_SETTINGS
    )

    # Block until capacity available
    wait_time = limiter.acquire(expected_tokens=1000)

    # After the API call, adjust for actual usage
    limiter.adjust(actual_tokens=800, reserved_tokens=1000)
"""

from __future__ import annotations

import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def client_waiting_enabled() -> bool:
    """Return True when client-side RPM/TPM pacing sleeps are enabled.

    Disabled by default: hosted APIs (e.g. OpenRouter) are assumed not to need
    local token-bucket waiting. Set ``RL_ENABLE_WAITING=1`` to restore pacing.
    """
    return os.getenv("RL_ENABLE_WAITING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DailyQuotaExceeded(RuntimeError):
    """Raised when a daily RPD/TPD quota would be exceeded.

    Attributes
    ----------
    retry_after_s
        Approximate seconds until the UTC calendar-day counter resets.
    kind
        ``"requests"`` or ``"tokens"``.
    """

    def __init__(self, message: str, *, retry_after_s: float, kind: str):
        super().__init__(message)
        self.retry_after_s = float(retry_after_s)
        self.kind = kind
        self.status_code = 429


def _utc_day_key(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y-%m-%d")


def _seconds_until_utc_midnight(now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    tomorrow = current.replace(hour=0, minute=0, second=0, microsecond=0)
    # advance to next midnight
    from datetime import timedelta

    tomorrow = tomorrow + timedelta(days=1)
    return max(0.0, (tomorrow - current).total_seconds())


class _TokenBucket:
    """A single token bucket with continuous refill.

    Implements a leaky bucket that refills at a continuous rate up to
    a maximum capacity. Thread-safety is provided by the parent
    RateLimiter class.
    """

    def __init__(self, capacity: float, refill_rate_per_sec: float):
        """Initialize the token bucket.

        Parameters
        ----------
        capacity
            Maximum number of tokens the bucket can hold.
        refill_rate_per_sec
            Rate at which tokens are added (tokens per second).
        """
        self.capacity = float(capacity)
        self.refill_rate_per_sec = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill_ts = time.perf_counter()

    def _refill(self, now: float) -> None:
        """Add tokens based on elapsed time since last refill."""
        elapsed = max(0.0, now - self.last_refill_ts)
        if elapsed <= 0.0:
            return
        added = elapsed * self.refill_rate_per_sec
        if added <= 0.0:
            self.last_refill_ts = now
            return
        self.tokens = min(self.capacity, self.tokens + added)
        self.last_refill_ts = now

    def deficit_seconds(self, need: float, now: float) -> float:
        """Calculate seconds until `need` tokens will be available.

        Returns
        -------
        float
            Seconds to wait, or 0.0 if tokens available now.
            Returns infinity if refill rate is zero.
        """
        self._refill(now)
        shortfall = max(0.0, need - self.tokens)
        if shortfall <= 0.0:
            return 0.0
        if self.refill_rate_per_sec <= 0:
            return float("inf")
        return shortfall / self.refill_rate_per_sec

    def consume(self, need: float, now: float) -> None:
        """Remove tokens from the bucket."""
        self._refill(now)
        self.tokens = max(0.0, self.tokens - need)

    def add_back(self, gain: float, now: float) -> None:
        """Return tokens to the bucket (up to capacity)."""
        self._refill(now)
        self.tokens = min(self.capacity, self.tokens + max(0.0, gain))


class RateLimiter:
    """Dual bucket (RPM/TPM) limiter with optional daily quotas and backoff.

    This class enforces both request-per-minute and token-per-minute limits
    using two token buckets. Optional RPD/TPD counters refuse acquires when
    exhausted. Thread-safe; collects wait/retry metrics.

    Attributes
    ----------
    rpm_bucket : _TokenBucket
        Bucket tracking request rate.
    tpm_bucket : _TokenBucket
        Bucket tracking token rate.
    total_wait_time_s : float
        Cumulative time spent waiting for capacity.
    wait_events : int
        Number of times a request had to wait.
    total_retries : int
        Number of backoff retries performed.
    """

    def __init__(
        self,
        *,
        rpm: int,
        tpm: int,
        backoff: Dict[str, Any],
        rpd: int | None = None,
        tpd: int | None = None,
    ):
        """Initialize the rate limiter.

        Parameters
        ----------
        rpm
            Requests per minute limit.
        tpm
            Tokens per minute limit.
        backoff
            Backoff configuration dict with keys:
            - initial_delay: float
            - multiplier: float
            - max_delay: float
            - jitter: float (0-1)
        rpd
            Optional requests-per-day hard quota (UTC calendar day).
        tpd
            Optional tokens-per-day hard quota (UTC calendar day).
        """
        self.lock = threading.Lock()
        self.rpm_bucket = _TokenBucket(
            capacity=float(rpm),
            refill_rate_per_sec=float(rpm) / 60.0,
        )
        self.tpm_bucket = _TokenBucket(
            capacity=float(tpm),
            refill_rate_per_sec=float(tpm) / 60.0,
        )
        self.backoff = backoff
        self.rpd = int(rpd) if rpd is not None and int(rpd) > 0 else None
        self.tpd = int(tpd) if tpd is not None and int(tpd) > 0 else None
        self._day_key = _utc_day_key()
        self._requests_today = 0
        self._tokens_today = 0.0

        # Metrics
        self.total_wait_time_s: float = 0.0
        self.wait_events: int = 0
        self.total_retries: int = 0
        self.last_retry_delay_s: float = 0.0
        self.last_error: Optional[str] = None

    def _roll_day_unlocked(self) -> None:
        key = _utc_day_key()
        if key != self._day_key:
            self._day_key = key
            self._requests_today = 0
            self._tokens_today = 0.0

    def _check_daily_unlocked(self, expected_tokens: int) -> None:
        """Raise DailyQuotaExceeded if this acquire would exceed RPD/TPD."""
        if self.rpd is None and self.tpd is None:
            return
        self._roll_day_unlocked()
        reset_s = _seconds_until_utc_midnight()
        if self.rpd is not None and self._requests_today + 1 > self.rpd:
            raise DailyQuotaExceeded(
                f"Daily request quota exceeded ({self._requests_today}/{self.rpd} RPD); "
                f"resets in ~{reset_s:.0f}s",
                retry_after_s=reset_s,
                kind="requests",
            )
        if self.tpd is not None and self._tokens_today + float(expected_tokens) > self.tpd:
            raise DailyQuotaExceeded(
                f"Daily token quota exceeded "
                f"({self._tokens_today:.0f}+{expected_tokens}/{self.tpd} TPD); "
                f"resets in ~{reset_s:.0f}s",
                retry_after_s=reset_s,
                kind="tokens",
            )

    def _record_daily_unlocked(self, expected_tokens: int) -> None:
        if self.rpd is None and self.tpd is None:
            return
        self._roll_day_unlocked()
        self._requests_today += 1
        self._tokens_today += float(expected_tokens)

    def acquire(self, *, expected_tokens: int) -> float:
        """Reserve capacity for one request (optionally blocking on RPM/TPM).

        When ``RL_ENABLE_WAITING`` is unset/false (default), returns immediately
        without sleeping, but still checks/records daily quotas when configured.
        When enabled, blocks until both RPM and TPM buckets can satisfy the
        request. Daily RPD/TPD exhaustion raises :class:`DailyQuotaExceeded`.

        Parameters
        ----------
        expected_tokens
            Estimated total tokens for this request.

        Returns
        -------
        float
            Total seconds spent waiting for capacity.
        """
        # Always enforce daily quotas (fail fast rather than sleep until midnight).
        with self.lock:
            self._check_daily_unlocked(expected_tokens)

        if not client_waiting_enabled():
            with self.lock:
                self._record_daily_unlocked(expected_tokens)
            return 0.0

        waited = 0.0
        while True:
            now = time.perf_counter()
            with self.lock:
                self._check_daily_unlocked(expected_tokens)
                deficit_r = self.rpm_bucket.deficit_seconds(1.0, now)
                deficit_t = self.tpm_bucket.deficit_seconds(float(expected_tokens), now)

                delay = max(deficit_r, deficit_t)
                if delay <= 0.0:
                    # Capacity available - consume and proceed
                    self.rpm_bucket.consume(1.0, now)
                    self.tpm_bucket.consume(float(expected_tokens), now)
                    self._record_daily_unlocked(expected_tokens)
                    return waited

            # Sleep outside the lock to allow other threads to proceed
            to_sleep = max(0.001, delay)
            time.sleep(to_sleep)
            waited += to_sleep

            # Track wait metrics
            with self.lock:
                self.total_wait_time_s += to_sleep
                self.wait_events += 1

    def adjust(self, *, actual_tokens: int, reserved_tokens: int) -> None:
        """Adjust token accounting after a request completes.

        Call this after an API response is received to correct the token
        count. If fewer tokens were used than reserved, the excess is
        returned to the bucket (and daily token counter when applicable).

        Parameters
        ----------
        actual_tokens
            Actual token count from the API response.
        reserved_tokens
            Token count that was reserved in acquire().
        """
        now = time.perf_counter()
        delta = float(reserved_tokens - actual_tokens)
        if abs(delta) < 1e-9:
            return

        with self.lock:
            if delta > 0:
                # Return unused tokens
                self.tpm_bucket.add_back(delta, now)
                if self.tpd is not None:
                    self._roll_day_unlocked()
                    self._tokens_today = max(0.0, self._tokens_today - delta)
            else:
                # Consume extra tokens that weren't reserved
                self.tpm_bucket.consume(-delta, now)
                if self.tpd is not None:
                    self._roll_day_unlocked()
                    self._tokens_today += -delta

    def backoff_sleep(self, attempt: int, *, retry_after_s: float | None = None) -> float:
        """Sleep with exponential backoff and jitter, or an explicit Retry-After.

        Used for error recovery with increasing delays between retries.
        When ``retry_after_s`` is provided (e.g. from a 429 ``Retry-After``
        header), that delay is used instead of the exponential schedule.

        Parameters
        ----------
        attempt
            Current attempt number (1-indexed).
        retry_after_s
            Optional provider-requested delay in seconds.

        Returns
        -------
        float
            Actual sleep duration in seconds.
        """
        if retry_after_s is not None and retry_after_s > 0:
            delay = float(retry_after_s)
        else:
            base = float(self.backoff.get("initial_delay", 0.5))
            mult = float(self.backoff.get("multiplier", 2.0))
            max_delay = float(self.backoff.get("max_delay", 20.0))
            jitter = float(self.backoff.get("jitter", 0.3))

            delay = min(max_delay, base * (mult ** max(0, attempt - 1)))
            # Apply jitter to prevent thundering herd
            delay = delay * (1.0 - jitter * random.random())

        time.sleep(delay)
        self.last_retry_delay_s = delay
        self.total_retries += 1
        return delay

    def snapshot(self) -> Dict[str, Any]:
        """Return a metrics snapshot including remaining quota estimates."""
        with self.lock:
            self._roll_day_unlocked()
            now = time.perf_counter()
            self.rpm_bucket._refill(now)
            self.tpm_bucket._refill(now)
            reset_s = _seconds_until_utc_midnight()
            return {
                "total_wait_time_s": self.total_wait_time_s,
                "wait_events": self.wait_events,
                "total_retries": self.total_retries,
                "last_retry_delay_s": self.last_retry_delay_s,
                "last_error": self.last_error,
                "rpm_tokens_remaining": self.rpm_bucket.tokens,
                "tpm_tokens_remaining": self.tpm_bucket.tokens,
                "rpm_capacity": self.rpm_bucket.capacity,
                "tpm_capacity": self.tpm_bucket.capacity,
                "rpd": self.rpd,
                "tpd": self.tpd,
                "requests_today": self._requests_today,
                "tokens_today": self._tokens_today,
                "rpd_remaining": (
                    None if self.rpd is None else max(0, self.rpd - self._requests_today)
                ),
                "tpd_remaining": (
                    None
                    if self.tpd is None
                    else max(0.0, float(self.tpd) - self._tokens_today)
                ),
                "day_key_utc": self._day_key,
                "seconds_until_day_reset": reset_s,
            }


class LimiterRegistry:
    """Global registry for rate limiters.

    Provides a singleton-like access pattern for rate limiters, ensuring
    that all threads share the same limiter instance for a given key.
    This is important for accurate rate limiting across concurrent requests.
    """

    _instances: Dict[str, RateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        key: str,
        *,
        rpm: int,
        tpm: int,
        backoff: Dict[str, Any],
        rpd: int | None = None,
        tpd: int | None = None,
    ) -> RateLimiter:
        """Get or create a rate limiter for the given key.

        Parameters
        ----------
        key
            Unique identifier for the limiter (typically model name).
        rpm
            Requests per minute limit.
        tpm
            Tokens per minute limit.
        backoff
            Backoff configuration dict.
        rpd
            Optional requests-per-day quota.
        tpd
            Optional tokens-per-day quota.

        Returns
        -------
        RateLimiter
            The rate limiter instance for this key.
        """
        with cls._lock:
            rl = cls._instances.get(key)
            if rl is None:
                rl = RateLimiter(rpm=rpm, tpm=tpm, backoff=backoff, rpd=rpd, tpd=tpd)
                cls._instances[key] = rl
            return rl

    @classmethod
    def metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get aggregate metrics for all registered limiters."""
        with cls._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for key, rl in cls._instances.items():
                out[key] = rl.snapshot()
            return out


__all__ = [
    "RateLimiter",
    "LimiterRegistry",
    "DailyQuotaExceeded",
    "client_waiting_enabled",
]
