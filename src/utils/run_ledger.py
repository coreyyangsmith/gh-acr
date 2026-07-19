"""Append-only JSONL run ledger for pipeline success/failure/degradation tracking.

Persists one JSON object per line so a crashed run remains diagnosable.
Each write is flushed immediately to disk.

Non-success records (``failure`` / ``degraded``) are dual-written to an optional
failures-only JSONL when ``failures_path`` is configured.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

from src.utils.failure_classify import classify_failure


class _MemoryLogHandler(logging.Handler):
    """Collect formatted log records into an in-memory list."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # pragma: no cover – never break the pipeline for logging
            self.handleError(record)


@contextmanager
def capture_logs(
    *,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
) -> Generator[list[str], None, None]:
    """Temporarily capture log lines from the root (or given) logger.

    Yields a live list that accumulates formatted log messages for the
    duration of the ``with`` block. Used to embed logs into failure ledger
    records. Temporarily lowers the target logger's level so INFO/DEBUG
    messages are captured even if the process default is WARNING.
    """
    target = logger or logging.getLogger()
    handler = _MemoryLogHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    previous_level = target.level
    target.addHandler(handler)
    # Ensure the logger itself emits records at the requested level
    if previous_level == logging.NOTSET or previous_level > level:
        target.setLevel(level)
    try:
        yield handler.records
    finally:
        target.removeHandler(handler)
        handler.close()
        target.setLevel(previous_level)


def load_successful_units(ledger_path: str | Path) -> set[tuple[str, str]]:
    """Return ``{(scenario_id, eval_method), ...}`` for completed ledger units.

    Includes ``success`` and ``degraded`` (soft failures that already wrote
    flagged CSV rows). Ignores hard ``failure`` / summary lines and any
    ``eval_method`` of ``prep``. Missing or unreadable files yield an empty set.
    """
    path = Path(ledger_path)
    done: set[tuple[str, str]] = set()
    if not path.is_file():
        return done
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") not in {"success", "degraded"}:
                    continue
                method = str(rec.get("eval_method") or "")
                if not method or method == "prep":
                    continue
                scenario_id = rec.get("scenario_id")
                if scenario_id is None:
                    continue
                done.add((str(scenario_id), method))
    except OSError:
        return set()
    return done


def _count_status_lines(ledger_path: Path) -> tuple[int, int, int]:
    """Count success/failure/degraded status lines in an existing ledger file."""
    ok = 0
    fail = 0
    degraded = 0
    if not ledger_path.is_file():
        return ok, fail, degraded
    try:
        with ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = rec.get("status")
                if status == "success":
                    ok += 1
                elif status == "failure":
                    fail += 1
                elif status == "degraded":
                    degraded += 1
    except OSError:
        return 0, 0, 0
    return ok, fail, degraded


class RunLedger:
    """Append-only JSONL ledger for per-scenario pipeline outcomes."""

    def __init__(
        self,
        path: str | Path,
        *,
        failures_path: str | Path | None = None,
        seed_counts: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.failures_path = Path(failures_path) if failures_path is not None else None
        if self.failures_path is not None:
            self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        self.success_count = 0
        self.failure_count = 0
        self.degraded_count = 0
        self._lock = threading.Lock()
        if seed_counts and self.path.is_file():
            self.success_count, self.failure_count, self.degraded_count = (
                _count_status_lines(self.path)
            )

    @classmethod
    def from_existing(
        cls,
        path: str | Path,
        *,
        failures_path: str | Path | None = None,
    ) -> "RunLedger":
        """Open a ledger for append, re-seeding success/failure counters from disk."""
        return cls(path, failures_path=failures_path, seed_counts=True)

    def append(self, record: dict[str, Any]) -> None:
        """Append one JSON record and flush immediately.

        Thread-safe so concurrent scenario workers can share one ledger.
        Non-success statuses are also dual-written to ``failures_path`` when set.
        """
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self._lock:
            status = payload.get("status")
            if status == "success":
                self.success_count += 1
            elif status == "failure":
                self.failure_count += 1
            elif status == "degraded":
                self.degraded_count += 1

            line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

            if (
                self.failures_path is not None
                and status in {"failure", "degraded"}
            ):
                with self.failures_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()

    def record_success(
        self,
        *,
        scenario_id: Any,
        eval_method: str,
        model_name: str | None = None,
        df_index: Any = None,
        repo: str | None = None,
        num_files: int | None = None,
        exact_match_overall: Any = None,
        processing_time_s: float | None = None,
        **extra: Any,
    ) -> None:
        self.append(
            {
                "scenario_id": scenario_id,
                "df_index": df_index,
                "repo": repo,
                "eval_method": eval_method,
                "model_name": model_name,
                "status": "success",
                "num_files": num_files,
                "exact_match_overall": exact_match_overall,
                "processing_time_s": processing_time_s,
                **extra,
            }
        )

    def record_failure(
        self,
        *,
        scenario_id: Any,
        eval_method: str,
        error: BaseException,
        model_name: str | None = None,
        df_index: Any = None,
        repo: str | None = None,
        traceback_text: str | None = None,
        captured_logs: Iterable[str] | None = None,
        processing_time_s: float | None = None,
        failure_category: str | None = None,
        prep: bool = False,
        **extra: Any,
    ) -> None:
        logs_list = list(captured_logs) if captured_logs is not None else None
        category = failure_category or classify_failure(error, prep=prep)
        self.append(
            {
                "scenario_id": scenario_id,
                "df_index": df_index,
                "repo": repo,
                "eval_method": eval_method,
                "model_name": model_name,
                "status": "failure",
                "failure_category": category,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback_text,
                "captured_logs": logs_list,
                "processing_time_s": processing_time_s,
                **extra,
            }
        )

    def record_degraded(
        self,
        *,
        scenario_id: Any,
        eval_method: str,
        degradation_events: list[dict[str, Any]],
        model_name: str | None = None,
        df_index: Any = None,
        repo: str | None = None,
        processing_time_s: float | None = None,
        failure_category: str | None = None,
        **extra: Any,
    ) -> None:
        """Record a soft-degradation outcome (no exception, but not a clean success)."""
        primary = failure_category
        if primary is None and degradation_events:
            primary = str(degradation_events[0].get("category") or "other")
        self.append(
            {
                "scenario_id": scenario_id,
                "df_index": df_index,
                "repo": repo,
                "eval_method": eval_method,
                "model_name": model_name,
                "status": "degraded",
                "failure_category": primary or "other",
                "degradation_events": list(degradation_events),
                "processing_time_s": processing_time_s,
                **extra,
            }
        )

    def record_summary(self, **extra: Any) -> None:
        """Append a final summary record with success/failure/degraded counts."""
        self.append(
            {
                "status": "summary",
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "degraded_count": self.degraded_count,
                **extra,
            }
        )


__all__ = [
    "RunLedger",
    "capture_logs",
    "load_successful_units",
]
