"""Thread-safe rate limiting for LLM API calls.

This module provides a token-bucket based rate limiter that can enforce both
requests-per-minute (RPM) and tokens-per-minute (TPM) limits. Client-side
waiting is **disabled by default** (``RL_ENABLE_WAITING`` unset/false); set
``RL_ENABLE_WAITING=1`` to restore blocking acquire behavior.

When waiting is enabled, the limiter uses two token buckets:
1. **RPM bucket**: Refills at requests/minute rate
2. **TPM bucket**: Refills at tokens/minute rate

Requests then block until both buckets have sufficient capacity.

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

Token Estimation
----------------
Before an API call, estimate the total tokens (input + expected output).
After the call, use adjust() to correct for actual usage. This allows
the limiter to release unused capacity back to the pool.

Metrics
-------
The LimiterRegistry provides aggregate metrics for monitoring::

    metrics = LimiterRegistry.metrics()
    for model, stats in metrics.items():
        print(f"{model}: {stats['wait_events']} waits, {stats['total_wait_time_s']:.1f}s")
"""

from __future__ import annotations

import os
import random
import threading
import time
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


class _TokenBucket:
    """A single token bucket with continuous refill.

    Implements a leaky bucket that refills at a constant rate up to
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
    """Dual bucket (RPM/TPM) limiter with jittered exponential backoff.

    This class enforces both request-per-minute and token-per-minute limits
    using two token buckets. It provides thread-safe access and collects
    metrics on wait times and retries.

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

    def __init__(self, *, rpm: int, tpm: int, backoff: Dict[str, Any]):
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
        """
        self.lock = threading.Lock()
        self.rpm_bucket = _TokenBucket(
            capacity=float(rpm), 
            refill_rate_per_sec=float(rpm) / 60.0
        )
        self.tpm_bucket = _TokenBucket(
            capacity=float(tpm), 
            refill_rate_per_sec=float(tpm) / 60.0
        )
        self.backoff = backoff

        # Metrics
        self.total_wait_time_s: float = 0.0
        self.wait_events: int = 0
        self.total_retries: int = 0
        self.last_retry_delay_s: float = 0.0
        self.last_error: Optional[str] = None

    def acquire(self, *, expected_tokens: int) -> float:
        """Reserve capacity for one request (optionally blocking on RPM/TPM).

        When ``RL_ENABLE_WAITING`` is unset/false (default), returns immediately
        without sleeping. When enabled, blocks until both RPM and TPM buckets
        can satisfy the request.

        Parameters
        ----------
        expected_tokens
            Estimated total tokens for this request.

        Returns
        -------
        float
            Total seconds spent waiting for capacity.
        """
        if not client_waiting_enabled():
            return 0.0

        waited = 0.0
        while True:
            now = time.perf_counter()
            with self.lock:
                deficit_r = self.rpm_bucket.deficit_seconds(1.0, now)
                deficit_t = self.tpm_bucket.deficit_seconds(float(expected_tokens), now)

                delay = max(deficit_r, deficit_t)
                if delay <= 0.0:
                    # Capacity available - consume and proceed
                    self.rpm_bucket.consume(1.0, now)
                    self.tpm_bucket.consume(float(expected_tokens), now)
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
        returned to the bucket.

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
            else:
                # Consume extra tokens that weren't reserved
                self.tpm_bucket.consume(-delta, now)

    def backoff_sleep(self, attempt: int) -> float:
        """Sleep with exponential backoff and jitter.

        Used for error recovery with increasing delays between retries.

        Parameters
        ----------
        attempt
            Current attempt number (1-indexed).

        Returns
        -------
        float
            Actual sleep duration in seconds.
        """
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


class LimiterRegistry:
    """Global registry for rate limiters.

    Provides a singleton-like access pattern for rate limiters, ensuring
    that all threads share the same limiter instance for a given key.
    This is important for accurate rate limiting across concurrent requests.

    Examples
    --------
    >>> limiter = LimiterRegistry.get("openai/gpt-4o-mini", rpm=100, tpm=200000, backoff={})
    >>> limiter.acquire(expected_tokens=500)
    >>> # ... make API call ...
    >>> limiter.adjust(actual_tokens=400, reserved_tokens=500)

    >>> # Get metrics for all limiters
    >>> for key, stats in LimiterRegistry.metrics().items():
    ...     print(f"{key}: {stats['wait_events']} waits")
    """

    _instances: Dict[str, RateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key: str, *, rpm: int, tpm: int, backoff: Dict[str, Any]) -> RateLimiter:
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

        Returns
        -------
        RateLimiter
            The rate limiter instance for this key.
        """
        with cls._lock:
            rl = cls._instances.get(key)
            if rl is None:
                rl = RateLimiter(rpm=rpm, tpm=tpm, backoff=backoff)
                cls._instances[key] = rl
            return rl

    @classmethod
    def metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get aggregate metrics for all registered limiters.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping of limiter keys to their metrics:
            - total_wait_time_s: float
            - wait_events: int
            - total_retries: int
            - last_retry_delay_s: float
            - last_error: Optional[str]
        """
        with cls._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for key, rl in cls._instances.items():
                out[key] = {
                    "total_wait_time_s": rl.total_wait_time_s,
                    "wait_events": rl.wait_events,
                    "total_retries": rl.total_retries,
                    "last_retry_delay_s": rl.last_retry_delay_s,
                    "last_error": rl.last_error,
                }
            return out


__all__ = [
    "RateLimiter",
    "LimiterRegistry",
    "client_waiting_enabled",
]
