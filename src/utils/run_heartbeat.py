"""Durable run heartbeat / health-state helpers.

``RunProgress`` writes a JSON heartbeat beside the results CSV so operators can
tell whether a long-running ``run_all`` is still advancing or stalled. A
background watchdog can abort the process when the heartbeat goes stale or an
in-flight unit exceeds a deadline, leaving the ledger resumable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Active LLM call registry (updated by resilient_invoke)
_active_calls_lock = threading.Lock()
_active_calls: dict[str, dict[str, Any]] = {}

# Soft-skip cancel registry: scenario×method units the watchdog abandoned so
# resilient_invoke can fail fast instead of retrying through another timeout.
_cancelled_units_lock = threading.Lock()
_cancelled_units: dict[tuple[str, str], str] = {}

# Serialize heartbeat IO per path (ticker + progress + watchdog race otherwise)
_heartbeat_io_locks: dict[str, threading.Lock] = {}
_heartbeat_io_locks_guard = threading.Lock()


def _heartbeat_io_lock(path: Path) -> threading.Lock:
    key = str(Path(path).absolute())
    with _heartbeat_io_locks_guard:
        lock = _heartbeat_io_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _heartbeat_io_locks[key] = lock
        return lock


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def heartbeat_path_for_results(results_path: Path | str) -> Path:
    """Return ``<stem>_heartbeat.json`` next to the results CSV."""
    path = Path(results_path)
    return path.with_name(f"{path.stem}_heartbeat.json")


def register_active_llm_call(
    call_id: str,
    *,
    node: Any = None,
    scenario_id: Any = None,
    file_path: Any = None,
    model_name: Any = None,
    attempt: int | None = None,
) -> None:
    """Record that an LLM invoke started (for heartbeat / watchdog)."""
    with _active_calls_lock:
        _active_calls[str(call_id)] = {
            "node": node,
            "scenario_id": scenario_id,
            "file_path": file_path,
            "model_name": model_name,
            "attempt": attempt,
            "started_at": time.perf_counter(),
            "started_at_utc": _utc_now_iso(),
        }


def unregister_active_llm_call(call_id: str) -> None:
    """Clear an active LLM invoke record."""
    with _active_calls_lock:
        _active_calls.pop(str(call_id), None)


def _unit_key(scenario: Any, method: Any) -> tuple[str, str] | None:
    scenario_s = str(scenario or "").strip()
    method_s = str(method or "").strip()
    if not scenario_s or not method_s or scenario_s == "?" or method_s == "?":
        return None
    return (scenario_s, method_s)


def mark_unit_cancelled(
    scenario: Any,
    method: Any,
    *,
    reason: str | None = None,
) -> bool:
    """Mark a scenario×method unit as soft-skipped / cancelled.

    Returns True if the unit was newly marked, False if already cancelled or
    the key is invalid.
    """
    key = _unit_key(scenario, method)
    if key is None:
        return False
    with _cancelled_units_lock:
        if key in _cancelled_units:
            return False
        _cancelled_units[key] = str(reason or "watchdog soft-skip")
        return True


def is_unit_cancelled(scenario: Any, method: Any) -> bool:
    """Return True if the scenario×method unit was soft-skipped."""
    key = _unit_key(scenario, method)
    if key is None:
        return False
    with _cancelled_units_lock:
        return key in _cancelled_units


def cancelled_unit_reason(scenario: Any, method: Any) -> str | None:
    """Return the cancel reason for a unit, or None if not cancelled."""
    key = _unit_key(scenario, method)
    if key is None:
        return None
    with _cancelled_units_lock:
        return _cancelled_units.get(key)


def clear_cancelled_units() -> None:
    """Clear the cancel registry (tests / process reset)."""
    with _cancelled_units_lock:
        _cancelled_units.clear()


def list_active_llm_calls() -> list[dict[str, Any]]:
    """Return a snapshot of in-flight LLM calls with age_s."""
    now = time.perf_counter()
    with _active_calls_lock:
        out: list[dict[str, Any]] = []
        for cid, info in _active_calls.items():
            row = dict(info)
            row["call_id"] = cid
            started = float(info.get("started_at") or now)
            row["age_s"] = round(now - started, 3)
            out.append(row)
        return out


def _parse_heartbeat_text(text: str) -> dict[str, Any]:
    """Parse heartbeat JSON, tolerating truncated/concatenated writes."""
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("empty heartbeat", text, 0)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        raise json.JSONDecodeError("heartbeat root must be object", text, 0)
    except json.JSONDecodeError:
        # Concurrent writers can concatenate two documents; take the first object.
        decoder = json.JSONDecoder()
        data, _end = decoder.raw_decode(text)
        if isinstance(data, dict):
            return data
        raise


def write_heartbeat(path: Path | str, payload: dict[str, Any]) -> Path:
    """Atomically write a heartbeat JSON document (process-wide locked)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["updated_at"] = _utc_now_iso()
    body["pid"] = os.getpid()
    encoded = json.dumps(body, ensure_ascii=False, indent=2, default=str)
    # Unique tmp avoids two threads clobbering the same .tmp on Windows.
    tmp = dest.with_name(
        f"{dest.stem}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    lock = _heartbeat_io_lock(dest)
    with lock:
        try:
            tmp.write_text(encoded, encoding="utf-8")
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return dest


def read_heartbeat(path: Path | str) -> dict[str, Any] | None:
    """Load a heartbeat file, or None if missing/unreadable."""
    dest = Path(path)
    if not dest.exists():
        return None
    lock = _heartbeat_io_lock(dest)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with lock:
                text = dest.read_text(encoding="utf-8")
            return _parse_heartbeat_text(text)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.02 * (attempt + 1))
    logger.warning("Failed to read heartbeat %s: %s", dest, last_exc)
    return None


def heartbeat_age_s(payload: dict[str, Any] | None) -> float | None:
    """Seconds since ``updated_at``, or None if unavailable."""
    if not payload:
        return None
    raw = payload.get("updated_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
    except Exception:
        return None


def dump_thread_stacks(path: Path | str) -> Path:
    """Write all thread stacks to ``path`` for stalled-run diagnostics."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames = sys._current_frames()
    lines: list[str] = [
        f"# thread dump pid={os.getpid()} at {_utc_now_iso()}",
        f"# threads={len(frames)}",
        "",
    ]
    for thread in threading.enumerate():
        frame = frames.get(thread.ident) if thread.ident is not None else None
        lines.append(f"--- Thread {thread.name} id={thread.ident} ---")
        if frame is None:
            lines.append("(no frame)")
        else:
            lines.extend(traceback.format_stack(frame))
        lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def format_status_report(
    *,
    heartbeat: dict[str, Any] | None,
    ledger_path: Path | str | None = None,
    failures_path: Path | str | None = None,
) -> str:
    """Human-readable status for ``--status`` / operators."""
    lines: list[str] = []
    if heartbeat is None:
        lines.append("heartbeat: missing")
    else:
        age = heartbeat_age_s(heartbeat)
        age_s = f"{age:.1f}s" if age is not None else "?"
        lines.append(
            f"heartbeat: pid={heartbeat.get('pid')} age={age_s} "
            f"updated_at={heartbeat.get('updated_at')}"
        )
        lines.append(
            f"progress: done={heartbeat.get('done')}/{heartbeat.get('total')} "
            f"ok={heartbeat.get('ok')} fail={heartbeat.get('failed')} "
            f"in_flight={heartbeat.get('in_flight_count')}"
        )
        for unit in heartbeat.get("in_flight") or []:
            lines.append(
                f"  - {unit.get('worker')}: {unit.get('method')}/"
                f"{unit.get('scenario')} stage={unit.get('stage')} "
                f"age={unit.get('age_s')}s"
            )
        for call in heartbeat.get("active_llm_calls") or []:
            lines.append(
                f"  llm: node={call.get('node')} scenario={call.get('scenario_id')} "
                f"attempt={call.get('attempt')} age={call.get('age_s')}s"
            )
        retries = heartbeat.get("total_retries")
        waits = heartbeat.get("wait_events")
        if retries is not None or waits is not None:
            lines.append(f"rate_limit: waits={waits} retries={retries}")

    if ledger_path:
        lp = Path(ledger_path)
        lines.append(f"ledger: {lp} exists={lp.exists()}")
        if lp.exists():
            try:
                last = None
                with lp.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            last = line
                if last:
                    ev = json.loads(last)
                    lines.append(
                        f"  latest: status={ev.get('status')} "
                        f"method={ev.get('eval_method')} "
                        f"scenario={ev.get('scenario_id')} "
                        f"ts={ev.get('timestamp')}"
                    )
            except Exception as exc:
                lines.append(f"  latest: (unreadable: {exc})")

    if failures_path:
        fp = Path(failures_path)
        lines.append(f"failures: {fp} exists={fp.exists()}")
    return "\n".join(lines)


class WatchdogTimeout(TimeoutError):
    """Raised / recorded when the watchdog soft-skips a stalled unit."""

    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 408


class RunWatchdog:
    """Background monitor for stalled work units.

    Modes
    -----
    skip (default)
        Soft-skip overdue scenario×method units (and hung LLM calls) via
        ``on_skip``, then keep the run going. Heartbeat staleness alone only
        warns — a keepalive ticker should refresh the heartbeat during long
        healthy stages.
    abort
        Legacy behavior: dump diagnostics and ``os._exit(2)``.
    """

    def __init__(
        self,
        *,
        heartbeat_path: Path,
        stale_s: float,
        max_unit_age_s: float | None = None,
        max_llm_call_age_s: float | None = None,
        poll_s: float = 15.0,
        diagnostics_dir: Path | None = None,
        mode: str = "skip",
        on_skip: Callable[[dict[str, Any]], None] | None = None,
        on_refresh: Callable[[], None] | None = None,
    ):
        self.heartbeat_path = Path(heartbeat_path)
        self.stale_s = float(stale_s)
        self.max_unit_age_s = (
            float(max_unit_age_s) if max_unit_age_s is not None else None
        )
        self.max_llm_call_age_s = (
            float(max_llm_call_age_s) if max_llm_call_age_s is not None else None
        )
        self.poll_s = max(1.0, float(poll_s))
        self.diagnostics_dir = Path(diagnostics_dir or self.heartbeat_path.parent)
        self.mode = (mode or "skip").strip().lower()
        if self.mode not in {"skip", "abort"}:
            self.mode = "skip"
        self.on_skip = on_skip
        self.on_refresh = on_refresh
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.triggered = False
        self.trigger_reason: str | None = None
        self._skipped_keys: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="ghacr-run-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_s + 1.0)
            self._thread = None

    def _run(self) -> None:
        # Allow the first heartbeat to appear before checking.
        self._stop.wait(self.poll_s)
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("[watchdog] poll failed")
            self._stop.wait(self.poll_s)

    def _write_diagnostics(self, reason: str) -> None:
        try:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            dump_thread_stacks(
                self.diagnostics_dir
                / f"{self.heartbeat_path.stem}_watchdog_stacks.txt"
            )
            hb = read_heartbeat(self.heartbeat_path)
            if hb is not None:
                write_heartbeat(
                    self.diagnostics_dir
                    / f"{self.heartbeat_path.stem}_watchdog_snapshot.json",
                    {**hb, "watchdog_reason": reason},
                )
        except Exception:
            logger.exception("[watchdog] Failed to write diagnostics")

    def _poll_once(self) -> None:
        if self.on_refresh is not None:
            try:
                self.on_refresh()
            except Exception:
                logger.debug("[watchdog] on_refresh failed", exc_info=True)

        overdue = self._find_overdue()
        if not overdue:
            hb = read_heartbeat(self.heartbeat_path)
            age = heartbeat_age_s(hb)
            if age is not None and age > self.stale_s:
                # Keepalive should prevent this; warn only in skip mode.
                msg = f"heartbeat stale for {age:.1f}s (limit {self.stale_s:.1f}s)"
                if self.mode == "abort":
                    self._abort(msg)
                else:
                    logger.warning(
                        "[watchdog] %s — not aborting (mode=skip); "
                        "waiting for unit/LLM age limits to soft-skip",
                        msg,
                    )
            return

        if self.mode == "abort":
            reason = overdue[0]["reason"]
            self._abort(reason)
            return

        for item in overdue:
            key = (str(item["scenario"]), str(item["method"]))
            with self._lock:
                if key in self._skipped_keys:
                    continue
                self._skipped_keys.add(key)
            self.triggered = True
            self.trigger_reason = item["reason"]
            logger.error(
                "[watchdog] Soft-skipping stalled unit scenario=%s method=%s: %s",
                key[0],
                key[1],
                item["reason"],
            )
            self._write_diagnostics(item["reason"])
            if self.on_skip is not None:
                try:
                    self.on_skip(item)
                except Exception:
                    logger.exception(
                        "[watchdog] on_skip failed for %s/%s", key[0], key[1]
                    )

    def _abort(self, reason: str) -> None:
        self.triggered = True
        self.trigger_reason = reason
        logger.error("[watchdog] Aborting run: %s", reason)
        self._write_diagnostics(reason)
        os._exit(2)

    def _find_overdue(self) -> list[dict[str, Any]]:
        hb = read_heartbeat(self.heartbeat_path)
        if hb is None:
            return []
        found: list[dict[str, Any]] = []
        unit_limit = self.max_unit_age_s
        if unit_limit is not None:
            for unit in hb.get("in_flight") or []:
                unit_age = float(unit.get("age_s") or 0.0)
                if unit_age > unit_limit:
                    found.append(
                        {
                            "kind": "unit",
                            "scenario": unit.get("scenario"),
                            "method": unit.get("method"),
                            "worker": unit.get("worker"),
                            "age_s": unit_age,
                            "reason": (
                                f"in-flight unit {unit.get('worker')} "
                                f"{unit.get('method')}/{unit.get('scenario')} "
                                f"age={unit_age:.1f}s exceeds {unit_limit:.1f}s"
                            ),
                        }
                    )
        if self.max_llm_call_age_s is not None:
            for call in hb.get("active_llm_calls") or list_active_llm_calls():
                call_age = float(call.get("age_s") or 0.0)
                if call_age > self.max_llm_call_age_s:
                    scenario = call.get("scenario_id") or "?"
                    # Prefer matching in-flight method for this scenario if unique
                    method = "?"
                    matches = [
                        u
                        for u in (hb.get("in_flight") or [])
                        if str(u.get("scenario")) == str(scenario)
                    ]
                    if len(matches) == 1:
                        method = str(matches[0].get("method") or "?")
                    elif matches:
                        # Fall back to oldest matching in-flight for this scenario
                        matches = sorted(
                            matches, key=lambda u: float(u.get("age_s") or 0.0), reverse=True
                        )
                        method = str(matches[0].get("method") or "?")
                    found.append(
                        {
                            "kind": "llm",
                            "scenario": scenario,
                            "method": method,
                            "node": call.get("node"),
                            "age_s": call_age,
                            "reason": (
                                f"LLM call node={call.get('node')} "
                                f"scenario={scenario} "
                                f"age={call_age:.1f}s exceeds "
                                f"{self.max_llm_call_age_s:.1f}s"
                            ),
                        }
                    )
        # Deduplicate by scenario/method keeping first reason
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for item in found:
            key = (str(item.get("scenario")), str(item.get("method")))
            if key[0] == "?" or key[1] == "?":
                continue
            if key not in dedup:
                dedup[key] = item
        return list(dedup.values())


__all__ = [
    "WatchdogTimeout",
    "RunWatchdog",
    "cancelled_unit_reason",
    "clear_cancelled_units",
    "dump_thread_stacks",
    "format_status_report",
    "heartbeat_age_s",
    "heartbeat_path_for_results",
    "is_unit_cancelled",
    "list_active_llm_calls",
    "mark_unit_cancelled",
    "read_heartbeat",
    "register_active_llm_call",
    "unregister_active_llm_call",
    "write_heartbeat",
]
