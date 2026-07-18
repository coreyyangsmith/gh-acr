"""Append-only JSONL run ledger for pipeline success/failure tracking.

Persists one JSON object per line so a crashed run remains diagnosable.
Each write is flushed immediately to disk.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable


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


class RunLedger:
    """Append-only JSONL ledger for per-scenario pipeline outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.success_count = 0
        self.failure_count = 0

    def append(self, record: dict[str, Any]) -> None:
        """Append one JSON record and flush immediately."""
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        status = payload.get("status")
        if status == "success":
            self.success_count += 1
        elif status == "failure":
            self.failure_count += 1

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
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
        **extra: Any,
    ) -> None:
        logs_list = list(captured_logs) if captured_logs is not None else None
        self.append(
            {
                "scenario_id": scenario_id,
                "df_index": df_index,
                "repo": repo,
                "eval_method": eval_method,
                "model_name": model_name,
                "status": "failure",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback_text,
                "captured_logs": logs_list,
                "processing_time_s": processing_time_s,
                **extra,
            }
        )

    def record_summary(self, **extra: Any) -> None:
        """Append a final summary record with success/failure counts."""
        self.append(
            {
                "status": "summary",
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                **extra,
            }
        )


__all__ = [
    "RunLedger",
    "capture_logs",
]
