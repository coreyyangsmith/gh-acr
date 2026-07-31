"""Thread-safe run progress tracker with tqdm bar and ETA.

Used by ``src.cli.run_all`` to report global progress across
``scenarios × methods`` work units, plus concise per-worker stage updates.
Optionally writes a durable heartbeat JSON for stalled-run detection.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Per-thread/async worker identity for stage hooks that lack an explicit worker id
_current_worker_id: ContextVar[Optional[str]] = ContextVar("run_progress_worker_id", default=None)

# Process-wide active tracker (set for the duration of one _run_all)
_active: Optional["RunProgress"] = None
_active_lock = threading.Lock()


def get_active_progress() -> Optional["RunProgress"]:
    """Return the active :class:`RunProgress` for this process, if any."""
    return _active


def get_current_worker_id() -> Optional[str]:
    """Return the ContextVar worker id for the current thread/task, if set."""
    return _current_worker_id.get()


def set_current_worker_id(worker_id: Optional[str]):
    """Set (or clear) the ContextVar worker id; returns the token for reset."""
    return _current_worker_id.set(worker_id)


def reset_current_worker_id(token) -> None:
    """Reset the ContextVar worker id using a token from :func:`set_current_worker_id`."""
    _current_worker_id.reset(token)


def set_stage(stage: str, *, detail: str | None = None, log: bool = True) -> None:
    """Update the current worker's stage via ContextVar (no-op if no active run).

    Parameters
    ----------
    stage
        Short stage label (e.g. ``clone``, ``patch``).
    detail
        Optional extra context shown in parentheses.
    log
        If False, update tqdm/in-flight state without emitting an INFO line
        (useful for high-frequency clone percent ticks).
    """
    prog = _active
    if prog is None:
        return
    wid = _current_worker_id.get()
    if not wid:
        return
    prog.set_stage(wid, stage, detail=detail, log=log)


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human duration (e.g. ``1h23m``, ``45s``)."""
    if seconds < 0 or seconds != seconds:  # NaN
        return "--"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


@dataclass
class _InFlight:
    scenario: str
    method: str
    stage: str = "start"
    detail: str = ""
    started_at: float = field(default_factory=time.perf_counter)


class RunProgress:
    """Global progress for one ``run_all`` invocation.

    Parameters
    ----------
    total
        Total work units (typically ``n_scenarios * n_methods``).
    disable_tqdm
        Force-disable the tqdm bar (also disabled when not a TTY or
        ``GHACR_NO_PROGRESS=1``).
    heartbeat_path
        Optional path for a durable JSON heartbeat written on progress updates.
    """

    def __init__(
        self,
        total: int,
        *,
        disable_tqdm: bool | None = None,
        heartbeat_path: Path | str | None = None,
        heartbeat_min_interval_s: float = 5.0,
    ):
        self.total = max(0, int(total))
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._next_worker_num = 1
        self._free_workers: list[str] = []
        self._in_flight: dict[str, _InFlight] = {}
        self._pbar = None
        self._heartbeat_path = Path(heartbeat_path) if heartbeat_path else None
        self._heartbeat_min_interval_s = max(0.5, float(heartbeat_min_interval_s))
        self._last_heartbeat_write = 0.0
        self._heartbeat_ticker_stop = threading.Event()
        self._heartbeat_ticker: Optional[threading.Thread] = None

        env_disable = (os.getenv("GHACR_NO_PROGRESS") or "").strip() in ("1", "true", "True", "yes")
        is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        if disable_tqdm is None:
            disable_tqdm = env_disable or not is_tty
        self._tqdm_disabled = bool(disable_tqdm)

        if self.total > 0 and not self._tqdm_disabled:
            try:
                from tqdm import tqdm

                self._pbar = tqdm(
                    total=self.total,
                    desc="run_all",
                    unit="task",
                    dynamic_ncols=True,
                    file=sys.stderr,
                    mininterval=0.5,
                )
            except Exception:
                self._pbar = None
                self._tqdm_disabled = True

    # ------------------------------------------------------------------
    # Lifecycle helpers (module-level active pointer)
    # ------------------------------------------------------------------

    def activate(self) -> "RunProgress":
        """Register this tracker as the process-wide active progress."""
        global _active
        with _active_lock:
            _active = self
        self.write_heartbeat(force=True)
        self._start_heartbeat_ticker()
        return self

    def deactivate(self) -> None:
        """Clear the process-wide active pointer if it is this instance."""
        global _active
        self._stop_heartbeat_ticker()
        with _active_lock:
            if _active is self:
                _active = None
        self.write_heartbeat(force=True)
        self.close()

    def _start_heartbeat_ticker(self) -> None:
        """Refresh heartbeat periodically so long stages do not look dead."""
        if self._heartbeat_path is None or self._heartbeat_ticker is not None:
            return
        self._heartbeat_ticker_stop.clear()
        interval = max(5.0, self._heartbeat_min_interval_s)

        def _tick() -> None:
            while not self._heartbeat_ticker_stop.wait(interval):
                self.write_heartbeat(force=True)

        self._heartbeat_ticker = threading.Thread(
            target=_tick, name="ghacr-heartbeat-ticker", daemon=True
        )
        self._heartbeat_ticker.start()

    def _stop_heartbeat_ticker(self) -> None:
        self._heartbeat_ticker_stop.set()
        if self._heartbeat_ticker is not None:
            self._heartbeat_ticker.join(timeout=self._heartbeat_min_interval_s + 2.0)
            self._heartbeat_ticker = None

    def abandon_in_flight(
        self,
        *,
        scenario: str,
        method: str,
        reason: str,
    ) -> str | None:
        """Mark a matching in-flight unit failed and free its worker slot.

        Returns the worker id if one was abandoned, else None.
        """
        with self._lock:
            match_id: str | None = None
            for wid, entry in self._in_flight.items():
                if entry.scenario == scenario and entry.method == method:
                    match_id = wid
                    break
            if match_id is None:
                return None
            self._in_flight.pop(match_id, None)
            self.done += 1
            self.failed += 1
            if self._pbar is not None:
                try:
                    self._pbar.update(1)
                except Exception:
                    pass
            if match_id not in self._free_workers:
                self._free_workers.append(match_id)
            self._refresh_tqdm_postfix_unlocked()
            snap = self._snapshot_unlocked()
        logger.warning(
            "[%s] scenario=%s method=%s abandoned (%s) | %s",
            match_id,
            scenario,
            method,
            reason,
            snap,
        )
        self.write_heartbeat(force=True)
        return match_id

    def close(self) -> None:
        """Close the tqdm bar if open."""
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass
            self._pbar = None

    # ------------------------------------------------------------------
    # Worker slot management
    # ------------------------------------------------------------------

    def acquire_worker(self) -> str:
        """Allocate a short worker id (``W1``, ``W2``, …)."""
        with self._lock:
            if self._free_workers:
                return self._free_workers.pop()
            wid = f"W{self._next_worker_num}"
            self._next_worker_num += 1
            return wid

    def release_worker(self, worker_id: str) -> None:
        """Return a worker id to the free pool and drop in-flight state."""
        with self._lock:
            self._in_flight.pop(worker_id, None)
            if worker_id not in self._free_workers:
                self._free_workers.append(worker_id)
            self._refresh_tqdm_postfix_unlocked()
        self.write_heartbeat()

    # ------------------------------------------------------------------
    # Progress updates
    # ------------------------------------------------------------------

    def mark_started(self, worker_id: str, scenario: Any, method: str) -> None:
        """Record that a worker began a scenario/method unit."""
        with self._lock:
            self._in_flight[worker_id] = _InFlight(
                scenario=str(scenario),
                method=str(method),
                stage="start",
            )
            self._refresh_tqdm_postfix_unlocked()
        logger.info(
            "[%s] scenario=%s method=%s stage=start",
            worker_id,
            scenario,
            method,
        )
        self.write_heartbeat()

    def set_stage(
        self,
        worker_id: str,
        stage: str,
        *,
        detail: str | None = None,
        log: bool = True,
    ) -> None:
        """Update the stage for an in-flight worker."""
        with self._lock:
            entry = self._in_flight.get(worker_id)
            if entry is None:
                entry = _InFlight(scenario="?", method="?", stage="")
                self._in_flight[worker_id] = entry
            # Skip identical updates to avoid duplicate INFO spam
            new_detail = entry.detail if detail is None else detail
            same = entry.stage == stage and entry.detail == new_detail
            entry.stage = stage
            if detail is not None:
                entry.detail = detail
            scenario = entry.scenario
            method = entry.method
            detail_s = entry.detail
            self._refresh_tqdm_postfix_unlocked()
        if log and not same:
            extra = f" ({detail_s})" if detail_s else ""
            logger.info(
                "[%s] scenario=%s method=%s stage=%s%s",
                worker_id,
                scenario,
                method,
                stage,
                extra,
            )
        if not same:
            self.write_heartbeat()

    def mark_stage_done(
        self,
        worker_id: str,
        stage: str,
        *,
        elapsed_s: float | None = None,
    ) -> None:
        """Log completion of a stage (does not advance the global counter)."""
        with self._lock:
            entry = self._in_flight.get(worker_id)
            scenario = entry.scenario if entry else "?"
            method = entry.method if entry else "?"
        if elapsed_s is not None:
            logger.info(
                "[%s] scenario=%s method=%s stage=%s done in %.1fs",
                worker_id,
                scenario,
                method,
                stage,
                elapsed_s,
            )
        else:
            logger.info(
                "[%s] scenario=%s method=%s stage=%s done",
                worker_id,
                scenario,
                method,
                stage,
            )

    def mark_done(
        self,
        worker_id: str,
        *,
        ok: bool = True,
        elapsed_s: float | None = None,
    ) -> None:
        """Advance the global counter after a scenario/method unit finishes."""
        with self._lock:
            entry = self._in_flight.pop(worker_id, None)
            self.done += 1
            if ok:
                self.ok += 1
            else:
                self.failed += 1
            if self._pbar is not None:
                try:
                    self._pbar.update(1)
                except Exception:
                    pass
            self._refresh_tqdm_postfix_unlocked()
            snap = self._snapshot_unlocked()

        scenario = entry.scenario if entry else "?"
        method = entry.method if entry else "?"
        status = "ok" if ok else "FAIL"
        elapsed_part = f" in {elapsed_s:.1f}s" if elapsed_s is not None else ""
        logger.info(
            "[%s] scenario=%s method=%s finished %s%s | %s",
            worker_id,
            scenario,
            method,
            status,
            elapsed_part,
            snap,
        )
        logger.info("%s", self.snapshot_line())
        self.write_heartbeat(force=True)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def write_heartbeat(self, *, force: bool = False) -> None:
        """Persist a durable heartbeat snapshot (throttled unless ``force``)."""
        if self._heartbeat_path is None:
            return
        now = time.perf_counter()
        if (
            not force
            and (now - self._last_heartbeat_write) < self._heartbeat_min_interval_s
        ):
            return
        try:
            from src.utils.rate_limiter import LimiterRegistry
            from src.utils.run_heartbeat import list_active_llm_calls, write_heartbeat

            with self._lock:
                in_flight = []
                for wid, entry in self._in_flight.items():
                    in_flight.append(
                        {
                            "worker": wid,
                            "scenario": entry.scenario,
                            "method": entry.method,
                            "stage": entry.stage,
                            "detail": entry.detail,
                            "age_s": round(now - entry.started_at, 3),
                        }
                    )
                payload = {
                    "done": self.done,
                    "ok": self.ok,
                    "failed": self.failed,
                    "total": self.total,
                    "elapsed_s": round(now - self.t0, 3),
                    "in_flight_count": len(in_flight),
                    "in_flight": in_flight,
                    "snapshot": self._snapshot_unlocked(),
                }
            metrics = LimiterRegistry.metrics()
            total_retries = sum(int(m.get("total_retries") or 0) for m in metrics.values())
            wait_events = sum(int(m.get("wait_events") or 0) for m in metrics.values())
            payload["total_retries"] = total_retries
            payload["wait_events"] = wait_events
            payload["limiter_metrics"] = metrics
            payload["active_llm_calls"] = list_active_llm_calls()
            write_heartbeat(self._heartbeat_path, payload)
            self._last_heartbeat_write = now
        except Exception:
            logger.debug("heartbeat write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def format_eta(self) -> str:
        with self._lock:
            return self._format_eta_unlocked()

    def snapshot_line(self) -> str:
        with self._lock:
            return self._snapshot_unlocked()

    def _format_eta_unlocked(self) -> str:
        if self.done < 1 or self.total <= 0:
            return "ETA --"
        elapsed = time.perf_counter() - self.t0
        if elapsed <= 0:
            return "ETA --"
        rate = self.done / elapsed
        remaining = max(0, self.total - self.done)
        if rate <= 0:
            return "ETA --"
        return f"ETA {format_duration(remaining / rate)}"

    def _rate_per_min_unlocked(self) -> float:
        elapsed = time.perf_counter() - self.t0
        if elapsed <= 0 or self.done < 1:
            return 0.0
        return (self.done / elapsed) * 60.0

    def _snapshot_unlocked(self) -> str:
        pct = (100.0 * self.done / self.total) if self.total else 0.0
        rate = self._rate_per_min_unlocked()
        eta = self._format_eta_unlocked()
        in_flight = len(self._in_flight)
        return (
            f"[progress] {self.done}/{self.total} ({pct:.1f}%) | "
            f"in_flight={in_flight} | rate={rate:.2f}/min | {eta} | "
            f"ok={self.ok} fail={self.failed}"
        )

    def _refresh_tqdm_postfix_unlocked(self) -> None:
        if self._pbar is None:
            return
        parts: list[str] = []
        for wid, entry in list(self._in_flight.items())[:4]:
            sid = entry.scenario
            if len(sid) > 12:
                sid = sid[:10] + "…"
            parts.append(f"{wid}:{entry.method}/{sid}:{entry.stage}")
        extra = len(self._in_flight) - len(parts)
        postfix = ", ".join(parts)
        if extra > 0:
            postfix = f"{postfix} +{extra}" if postfix else f"+{extra}"
        try:
            self._pbar.set_postfix_str(postfix or self._format_eta_unlocked(), refresh=False)
        except Exception:
            pass

    def summary_line(self) -> str:
        """Final one-line summary after the run."""
        with self._lock:
            elapsed = time.perf_counter() - self.t0
            avg = (elapsed / self.done) if self.done else 0.0
            return (
                f"[progress] complete {self.done}/{self.total} in {format_duration(elapsed)} | "
                f"ok={self.ok} fail={self.failed} | avg={avg:.1f}s/task"
            )


__all__ = [
    "RunProgress",
    "format_duration",
    "get_active_progress",
    "get_current_worker_id",
    "set_current_worker_id",
    "reset_current_worker_id",
    "set_stage",
]
