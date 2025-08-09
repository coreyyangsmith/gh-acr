from __future__ import annotations

import threading
import time
import random
from typing import Callable, Dict, Any, Optional


class _TokenBucket:
    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.refill_rate_per_sec = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill_ts = time.perf_counter()

    def _refill(self, now: float) -> None:
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
        self._refill(now)
        shortfall = max(0.0, need - self.tokens)
        if shortfall <= 0.0:
            return 0.0
        if self.refill_rate_per_sec <= 0:
            # Cannot ever refill
            return float("inf")
        return shortfall / self.refill_rate_per_sec

    def consume(self, need: float, now: float) -> None:
        self._refill(now)
        self.tokens = max(0.0, self.tokens - need)

    def add_back(self, gain: float, now: float) -> None:
        self._refill(now)
        self.tokens = min(self.capacity, self.tokens + max(0.0, gain))


class RateLimiter:
    """Dual bucket (RPM/TPM) limiter with jittered exponential backoff.

    Use acquire() before an API call with your expected token usage to throttle.
    After the call, call adjust(actual_tokens, reserved_tokens) to correct the
    token bucket if output tokens differed from the estimate.
    """

    def __init__(self, *, rpm: int, tpm: int, backoff: Dict[str, Any]):
        self.lock = threading.Lock()
        self.rpm_bucket = _TokenBucket(capacity=float(rpm), refill_rate_per_sec=float(rpm) / 60.0)
        self.tpm_bucket = _TokenBucket(capacity=float(tpm), refill_rate_per_sec=float(tpm) / 60.0)
        self.backoff = backoff

        # Metrics
        self.total_wait_time_s = 0.0
        self.wait_events = 0
        self.total_retries = 0
        self.last_retry_delay_s = 0.0
        self.last_error: Optional[str] = None

    def acquire(self, *, expected_tokens: int) -> float:
        """Block until both RPM and TPM buckets can satisfy the request.

        Returns the time spent waiting (seconds).
        """
        waited = 0.0
        while True:
            now = time.perf_counter()
            with self.lock:
                deficit_r = self.rpm_bucket.deficit_seconds(1.0, now)
                deficit_t = self.tpm_bucket.deficit_seconds(float(expected_tokens), now)

                delay = max(deficit_r, deficit_t)
                if delay <= 0.0:
                    # Consume allowances and proceed
                    self.rpm_bucket.consume(1.0, now)
                    self.tpm_bucket.consume(float(expected_tokens), now)
                    return waited

            # Sleep outside the lock
            to_sleep = max(0.001, delay)
            time.sleep(to_sleep)
            waited += to_sleep
            # track metrics
            with self.lock:
                self.total_wait_time_s += to_sleep
                self.wait_events += 1

    def adjust(self, *, actual_tokens: int, reserved_tokens: int) -> None:
        now = time.perf_counter()
        delta = float(reserved_tokens - actual_tokens)
        if abs(delta) < 1e-9:
            return
        with self.lock:
            if delta > 0:
                # Return unused tokens
                self.tpm_bucket.add_back(delta, now)
            else:
                # Spend extra tokens that were not reserved
                self.tpm_bucket.consume(-delta, now)

    def backoff_sleep(self, attempt: int) -> float:
        base = float(self.backoff.get("initial_delay", 0.5))
        mult = float(self.backoff.get("multiplier", 2.0))
        max_delay = float(self.backoff.get("max_delay", 20.0))
        jitter = float(self.backoff.get("jitter", 0.3))
        delay = min(max_delay, base * (mult ** max(0, attempt - 1)))
        # Full jitter
        delay = delay * (1.0 - jitter * random.random())
        time.sleep(delay)
        self.last_retry_delay_s = delay
        self.total_retries += 1
        return delay


class LimiterRegistry:
    _instances: Dict[str, RateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key: str, *, rpm: int, tpm: int, backoff: Dict[str, Any]) -> RateLimiter:
        with cls._lock:
            rl = cls._instances.get(key)
            if rl is None:
                rl = RateLimiter(rpm=rpm, tpm=tpm, backoff=backoff)
                cls._instances[key] = rl
            return rl

    @classmethod
    def metrics(cls) -> Dict[str, Dict[str, Any]]:
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

