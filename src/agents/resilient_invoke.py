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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.config.rate_limits import BACKOFF_SETTINGS, get_limits_for_model
from src.utils.rate_limiter import LimiterRegistry

logger = logging.getLogger(__name__)

# Prefer dedicated LLM_MAX_RETRIES; fall back to RL_MAX_RETRIES / BACKOFF_SETTINGS
_LLM_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", str(BACKOFF_SETTINGS.get("max_retries", 5)))
)
_LLM_MAX_PARSE_ATTEMPTS = int(os.getenv("LLM_MAX_PARSE_ATTEMPTS", "3"))

_FAILURES_DIR = Path("logs") / "llm_failures"
_IO_TRACE_DIR = Path("logs") / "llm_io"


@dataclass(frozen=True)
class ParsedResult:
    """Optional wrapper so ``parse_fn`` can report a recovery strategy."""

    value: Any
    strategy: str | None = None

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
    "402",
    "insufficient credits",
    "payment required",
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
        if status is not None and int(status) in {400, 401, 402, 403, 404}:
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


def _latest_llm_call_record(node: str | None) -> dict[str, Any]:
    """Pull the most recent ledger record for this node (tokens/cost)."""
    try:
        from src.agents.observability import get_llm_calls

        calls = get_llm_calls() or []
    except Exception:
        return {}
    if not calls:
        return {}
    if node:
        for record in reversed(calls):
            if not isinstance(record, dict):
                continue
            if record.get("node") == node:
                return record
    last = calls[-1]
    return last if isinstance(last, dict) else {}


def _write_call_artifacts(
    *,
    artifact_dir: Path | str | None,
    prompt_text: str,
    response_text: str | None,
    artifacts: dict[str, str] | None,
    context: dict[str, Any],
    elapsed_s: float | None,
    status: str,
    error: BaseException | None = None,
) -> None:
    if artifact_dir is None:
        return
    try:
        from src.agents.artifact_io import write_agent_call

        node = context.get("node")
        usage = _latest_llm_call_record(str(node) if node else None)
        metadata: dict[str, Any] = {
            "agent": context.get("agent") or node,
            "node": node,
            "model_name": context.get("model_name") or usage.get("model_name"),
            "elapsed_s": round(float(elapsed_s), 4) if elapsed_s is not None else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_in": usage.get("cost_in"),
            "cost_out": usage.get("cost_out"),
            "total_cost": usage.get("total_cost"),
            "usage_from_api": usage.get("usage_from_api"),
            "scenario_id": context.get("scenario_id"),
            "eval_method": context.get("eval_method"),
            "file_path": context.get("file_path"),
            "call_id": context.get("call_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_used": True,
            "status": status,
        }
        if error is not None:
            metadata["error_type"] = type(error).__name__
            metadata["error_message"] = str(error)
        extra = context.get("metadata_extra")
        if isinstance(extra, dict) and extra:
            metadata.update(extra)
        write_agent_call(
            artifact_dir,
            input_text=prompt_text,
            output_text=response_text or "",
            artifacts=artifacts,
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("Failed to persist agent call artifacts at %s: %s", artifact_dir, exc)


def _patch_call_metadata(artifact_dir: Path | str | None, updates: dict[str, Any]) -> None:
    """Merge *updates* into an existing ``metadata.json`` under *artifact_dir*."""
    if artifact_dir is None or not updates:
        return
    try:
        meta_path = Path(artifact_dir) / "metadata.json"
        if not meta_path.is_file():
            return
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        data.update(updates)
        meta_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to patch call metadata at %s: %s", artifact_dir, exc)


class ParseExhausted(ValueError):
    """Raised when ``invoke_and_parse`` cannot recover a structured value."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        attempt_log: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.attempt_log = list(attempt_log or [])


def resilient_invoke(
    llm: Any,
    prompt_text: str,
    *,
    context: dict[str, Any] | None = None,
    max_retries: int | None = None,
    artifact_dir: Path | str | None = None,
    artifacts: dict[str, str] | None = None,
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
        file_path, model_name, agent, call_id, etc.
    max_retries
        Override for ``LLM_MAX_RETRIES`` / ``BACKOFF_SETTINGS['max_retries']``.
    artifact_dir
        Optional directory for per-call ``input.txt`` / ``output.txt`` /
        ``artifacts/`` / ``metadata.json``.
    artifacts
        Optional supporting files (e.g. diffs) written under ``artifacts/``.

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
    last_latency: float | None = None

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
                last_latency = latency
                response_text = _extract_response_text(result)
                last_response = response_text
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "success",
                        "latency_s": round(latency, 4),
                    }
                )
                logger.info(
                    "[resilient_invoke] success in %.3fs node=%s scenario=%s file=%s",
                    latency,
                    ctx.get("node"),
                    ctx.get("scenario_id"),
                    ctx.get("file_path"),
                )
                _maybe_write_io_trace(
                    prompt_text=prompt_text,
                    response_text=response_text,
                    context=ctx,
                    status="success",
                    latency_s=latency,
                )
                _write_call_artifacts(
                    artifact_dir=artifact_dir,
                    prompt_text=prompt_text,
                    response_text=response_text,
                    artifacts=artifacts,
                    context=ctx,
                    elapsed_s=latency,
                    status="success",
                )
                return result
            except Exception as exc:
                latency = time.perf_counter() - t0
                last_latency = latency
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
        _write_call_artifacts(
            artifact_dir=artifact_dir,
            prompt_text=prompt_text,
            response_text=last_response,
            artifacts=artifacts,
            context=ctx,
            elapsed_s=last_latency,
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


def invoke_and_parse(
    llm: Any,
    prompt_text: str,
    *,
    parse_fn: Callable[[str], Any],
    max_parse_attempts: int | None = None,
    repair_hint: str | None = None,
    context: dict[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    artifacts: dict[str, str] | None = None,
    max_retries: int | None = None,
) -> tuple[Any, str, list[dict[str, Any]]]:
    """Invoke an LLM and parse the result, retrying only when local parse fails.

    ``parse_fn`` should apply local recovery first and either return a value or
    raise / return ``None`` when unrecoverable. Successful local recovery on
    attempt 1 does not trigger further LLM calls.

    Returns
    -------
    tuple
        ``(parsed, raw_text, attempt_log)``

    Raises
    ------
    ParseExhausted
        After ``max_parse_attempts`` failed local parses (API errors still
        propagate from ``resilient_invoke``).
    """
    attempts = int(
        max_parse_attempts if max_parse_attempts is not None else _LLM_MAX_PARSE_ATTEMPTS
    )
    attempts = max(1, attempts)
    ctx = dict(context or {})
    attempt_log: list[dict[str, Any]] = []
    last_raw = ""
    current_prompt = prompt_text

    for attempt in range(1, attempts + 1):
        result = resilient_invoke(
            llm,
            current_prompt,
            context=ctx,
            max_retries=max_retries,
            artifact_dir=artifact_dir,
            artifacts=artifacts,
        )
        raw = _extract_response_text(result)
        last_raw = raw
        parsed: Any = None
        parse_error: str | None = None
        strategy: str | None = None
        try:
            parsed = parse_fn(raw)
            if isinstance(parsed, ParsedResult):
                strategy = parsed.strategy
                parsed = parsed.value
            if parsed is None:
                parse_error = "parse_fn returned None"
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            parsed = None

        entry: dict[str, Any] = {
            "attempt": attempt,
            "parse_ok": parsed is not None and parse_error is None,
            "raw_snippet": (raw or "")[:200],
        }
        if strategy:
            entry["parse_strategy"] = strategy
        if parse_error:
            entry["parse_error"] = parse_error
        attempt_log.append(entry)

        if parsed is not None and parse_error is None:
            meta = {
                "parse_ok": True,
                "parse_attempts": attempt_log,
                "parse_attempt": attempt,
            }
            if strategy:
                meta["parse_strategy"] = strategy
            _patch_call_metadata(artifact_dir, meta)
            return parsed, raw, attempt_log

        logger.warning(
            "[invoke_and_parse] parse failed attempt=%d/%d node=%s: %s",
            attempt,
            attempts,
            ctx.get("node"),
            parse_error,
        )
        _patch_call_metadata(
            artifact_dir,
            {
                "parse_ok": False,
                "parse_attempts": attempt_log,
                "parse_attempt": attempt,
                "parse_error": parse_error,
            },
        )

        if attempt < attempts and repair_hint:
            current_prompt = (
                f"{prompt_text.rstrip()}\n\n"
                f"---\nPrevious output was not parseable. {repair_hint}\n"
            )

    raise ParseExhausted(
        f"failed to parse LLM output after {attempts} attempt(s)",
        raw_text=last_raw,
        attempt_log=attempt_log,
    )


__all__ = [
    "resilient_invoke",
    "invoke_and_parse",
    "is_retryable_error",
    "write_failure_trace",
    "ParseExhausted",
    "ParsedResult",
]
