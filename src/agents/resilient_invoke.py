"""Resilient LLM invocation with retries, failure traces, and optional I/O dumps.

Wraps ``llm.invoke`` so transient API failures (rate limits, timeouts, 5xx,
connection errors) are retried with jittered exponential backoff. Permanent
failures (auth, validation) fail fast. On final failure a JSON trace is written
under ``logs/llm_failures/`` for later diagnosis.

Optional full prompt/response tracing is enabled via ``GHACR_TRACE_IO=1``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config.rate_limits import BACKOFF_SETTINGS, get_limits_for_model
from src.utils.rate_limiter import LimiterRegistry

logger = logging.getLogger(__name__)

# Prefer dedicated LLM_MAX_RETRIES; fall back to RL_MAX_RETRIES / BACKOFF_SETTINGS
_LLM_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", str(BACKOFF_SETTINGS.get("max_retries", 5)))
)

_FAILURES_DIR = Path("logs") / "llm_failures"
_IO_TRACE_DIR = Path("logs") / "llm_io"

# Substrings / patterns that indicate a retryable transient error
_RETRYABLE_PATTERNS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
    "timeout",
    "timed out",
    "connection",
    "connect",
    "temporarily unavailable",
    "service unavailable",
    "503",
    "502",
    "504",
    "overloaded",
    "server error",
    "internal server error",
    "500",
    "bad gateway",
    "gateway timeout",
    "api connection",
    "remote end closed",
    "broken pipe",
    "reset by peer",
)

# Fail-fast (non-retryable) patterns
_FATAL_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "incorrect api key",
    "permission denied",
    "forbidden",
    "401",
    "403",
    "invalid_request",
    "invalid request",
    "context_length",
    "maximum context length",
    "token limit",
    "model_not_found",
    "does not exist",
)


def _trace_io_enabled() -> bool:
    return os.getenv("GHACR_TRACE_IO", "0").strip().lower() in {"1", "true", "yes", "on"}


def _safe_slug(value: Any, *, max_len: int = 80) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(value if value is not None else "unknown"))
    return (text[:max_len] or "unknown").strip("_") or "unknown"


def is_retryable_error(error: BaseException) -> bool:
    """Classify whether an LLM exception should be retried."""
    name = type(error).__name__.lower()
    msg = str(error).lower()
    combined = f"{name} {msg}"

    if any(p in combined for p in _FATAL_PATTERNS):
        return False

    # Common exception class name heuristics
    if any(
        tok in name
        for tok in (
            "ratelimit",
            "timeout",
            "connection",
            "apiconnection",
            "serviceunavailable",
            "internalserver",
        )
    ):
        return True

    status = getattr(error, "status_code", None) or getattr(error, "http_status", None)
    try:
        if status is not None and int(status) in {408, 429, 500, 502, 503, 504}:
            return True
        if status is not None and int(status) in {400, 401, 403, 404}:
            return False
    except (TypeError, ValueError):
        pass

    return any(p in combined for p in _RETRYABLE_PATTERNS)


def _extract_response_text(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "content"):
        content = result.content
        return content if isinstance(content, str) else str(content)
    return str(result)


def _get_limiter(model_name: str | None):
    if not model_name:
        return None
    try:
        limits = get_limits_for_model(model_name)
        return LimiterRegistry.get(
            key=model_name,
            rpm=int(limits.get("requests_per_minute", 60)),
            tpm=int(limits.get("tokens_per_minute", 150000)),
            backoff=BACKOFF_SETTINGS,
        )
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def write_failure_trace(
    *,
    prompt_text: str,
    error: BaseException,
    context: dict[str, Any],
    attempts: int,
    last_response: str | None = None,
    attempt_log: list[dict[str, Any]] | None = None,
) -> Path:
    """Persist a failure trace JSON under ``logs/llm_failures/``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    scenario = _safe_slug(context.get("scenario_id", "unknown"))
    node = _safe_slug(context.get("node", "unknown"))
    file_part = _safe_slug(context.get("file_path", "nofil"))
    filename = f"{scenario}_{node}_{file_part}_{ts}.json"
    path = _FAILURES_DIR / filename

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "attempts": attempts,
        "attempt_log": attempt_log or [],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "prompt": prompt_text,
        "last_partial_response": last_response,
    }
    _write_json(path, payload)
    logger.error("LLM failure trace written to %s", path)
    return path


def _maybe_write_io_trace(
    *,
    prompt_text: str,
    response_text: str | None,
    context: dict[str, Any],
    status: str,
    error: BaseException | None = None,
    latency_s: float | None = None,
) -> Optional[Path]:
    if not _trace_io_enabled():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    scenario = _safe_slug(context.get("scenario_id", "unknown"))
    node = _safe_slug(context.get("node", "unknown"))
    file_part = _safe_slug(context.get("file_path", "nofil"))
    path = _IO_TRACE_DIR / f"{scenario}_{node}_{file_part}_{ts}_{status}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "context": context,
        "latency_s": latency_s,
        "prompt": prompt_text,
        "response": response_text,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
    }
    return _write_json(path, payload)


def resilient_invoke(
    llm: Any,
    prompt_text: str,
    *,
    context: dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> Any:
    """Invoke an LLM with retry/backoff and durable failure logging.

    Parameters
    ----------
    llm
        LangChain-compatible chat model with ``.invoke``.
    prompt_text
        Fully rendered prompt string.
    context
        Metadata for logging/traces: scenario_id, eval_method, node,
        file_path, model_name, etc.
    max_retries
        Override for ``LLM_MAX_RETRIES`` / ``BACKOFF_SETTINGS['max_retries']``.

    Returns
    -------
    Any
        The raw LLM result (same as ``llm.invoke``).

    Raises
    ------
    BaseException
        Re-raises the last error after retries are exhausted (or immediately
        for non-retryable errors). A failure trace JSON is written first.
    """
    ctx = dict(context or {})
    model_name = ctx.get("model_name")
    retries = int(max_retries if max_retries is not None else _LLM_MAX_RETRIES)
    retries = max(0, retries)
    limiter = _get_limiter(str(model_name) if model_name else None)

    # Label LangFuse generations with the multi-agent node name
    _clear_node = None
    try:
        from src.agents.observability import clear_llm_node, set_llm_node

        set_llm_node(ctx.get("node"))
        _clear_node = clear_llm_node
    except Exception:
        pass

    attempt_log: list[dict[str, Any]] = []
    last_error: BaseException | None = None
    last_response: str | None = None

    try:
        # attempts = initial try + retries
        for attempt in range(1, retries + 2):
            t0 = time.perf_counter()
            try:
                logger.info(
                    "[resilient_invoke] attempt=%d/%d node=%s scenario=%s file=%s model=%s",
                    attempt,
                    retries + 1,
                    ctx.get("node"),
                    ctx.get("scenario_id"),
                    ctx.get("file_path"),
                    model_name,
                )
                result = llm.invoke(prompt_text)
                latency = time.perf_counter() - t0
                response_text = _extract_response_text(result)
                last_response = response_text
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "success",
                        "latency_s": round(latency, 4),
                    }
                )
                _maybe_write_io_trace(
                    prompt_text=prompt_text,
                    response_text=response_text,
                    context=ctx,
                    status="success",
                    latency_s=latency,
                )
                return result
            except Exception as exc:
                latency = time.perf_counter() - t0
                last_error = exc
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "latency_s": round(latency, 4),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "retryable": is_retryable_error(exc),
                    }
                )
                logger.warning(
                    "[resilient_invoke] attempt=%d failed (%s): %s",
                    attempt,
                    type(exc).__name__,
                    exc,
                )

                if not is_retryable_error(exc) or attempt > retries:
                    break

                # Back off before the next attempt
                if limiter is not None:
                    delay = limiter.backoff_sleep(attempt)
                else:
                    base = float(BACKOFF_SETTINGS.get("initial_delay", 0.5))
                    mult = float(BACKOFF_SETTINGS.get("multiplier", 2.0))
                    max_delay = float(BACKOFF_SETTINGS.get("max_delay", 20.0))
                    delay = min(max_delay, base * (mult ** max(0, attempt - 1)))
                    time.sleep(delay)
                logger.info("[resilient_invoke] backing off %.2fs before retry", delay)

        assert last_error is not None
        trace_path = write_failure_trace(
            prompt_text=prompt_text,
            error=last_error,
            context=ctx,
            attempts=len(attempt_log),
            last_response=last_response,
            attempt_log=attempt_log,
        )
        _maybe_write_io_trace(
            prompt_text=prompt_text,
            response_text=last_response,
            context={**ctx, "failure_trace": str(trace_path)},
            status="failure",
            error=last_error,
        )
        # Attach trace path for upstream ledger consumers if useful
        try:
            last_error.failure_trace_path = str(trace_path)  # type: ignore[attr-defined]
        except Exception:
            pass
        raise last_error
    finally:
        if _clear_node is not None:
            try:
                _clear_node()
            except Exception:
                pass


__all__ = [
    "resilient_invoke",
    "is_retryable_error",
    "write_failure_trace",
]
