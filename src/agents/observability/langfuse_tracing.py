"""LangFuse observability helpers for LLM inference.

Soft-disabled when ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are unset
(or ``LANGFUSE_TRACING_ENABLED`` is falsy). When enabled, each scenario reuses
one shared CallbackHandler and wraps the run in a parent observation so all
multi-agent LLM calls nest under a single method-named trace
(``bypass7``, ``force_mix``, ``agent``, ``base_a``, ``base_b``).

Tracing is optional and non-blocking: export/flush/observation failures are
swallowed. After ``LANGFUSE_FAILURE_THRESHOLD`` (default 3) failures in a
process, tracing is disabled for the remainder of the run so evaluation
continues without further telemetry attempts.

Trace attributes (Langfuse v4 / CallbackHandler metadata keys):
- ``langfuse_trace_name`` = eval method (low-cardinality operation id)
- ``langfuse_session_id`` = scenario id (Sessions view / per-scenario grouping)
- ``langfuse_tags`` = ``[method, "scenario:<id>"]``
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_eval_method: ContextVar[str] = ContextVar("ghacr_eval_method", default="")
_scenario_id: ContextVar[str] = ContextVar("ghacr_scenario_id", default="")
_model_name: ContextVar[str] = ContextVar("ghacr_model_name", default="")
_shared_handler: ContextVar[Any | None] = ContextVar("ghacr_langfuse_handler", default=None)
_llm_node: ContextVar[str] = ContextVar("ghacr_llm_node", default="")
_llm_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("ghacr_llm_calls", default=None)
# Snapshot of the last completed scenario's calls (survives clear_run_context)
_last_llm_calls: ContextVar[list[dict[str, Any]]] = ContextVar(
    "ghacr_last_llm_calls", default=[]
)

_FALSEY = {"0", "false", "no", "off", ""}
_MAX_OTEL_ATTR_CHARS = 4096
_DEFAULT_FAILURE_THRESHOLD = 3
_export_guard_installed = False

# Process-level circuit breaker: repeated telemetry failures disable tracing
# for the rest of the evaluation run without aborting work.
_circuit_lock = threading.Lock()
_failure_count = 0
_disabled_due_to_failures = False


def _failure_threshold() -> int:
    raw = (os.getenv("LANGFUSE_FAILURE_THRESHOLD") or "").strip()
    if not raw:
        return _DEFAULT_FAILURE_THRESHOLD
    try:
        return max(1, int(raw))
    except Exception:
        return _DEFAULT_FAILURE_THRESHOLD


def reset_langfuse_circuit_breaker() -> None:
    """Clear the failure circuit breaker (primarily for tests)."""
    global _failure_count, _disabled_due_to_failures
    with _circuit_lock:
        _failure_count = 0
        _disabled_due_to_failures = False


def get_langfuse_failure_count() -> int:
    """Return how many LangFuse failures have been recorded this process."""
    with _circuit_lock:
        return _failure_count


def is_langfuse_circuit_open() -> bool:
    """Return True when repeated failures have disabled tracing for this run."""
    with _circuit_lock:
        return _disabled_due_to_failures


def _record_langfuse_failure(where: str, exc: BaseException | str | None = None) -> None:
    """Count a non-fatal LangFuse failure; open the circuit after the threshold."""
    global _failure_count, _disabled_due_to_failures
    detail = ""
    if isinstance(exc, BaseException):
        detail = f"{type(exc).__name__}: {exc}"
    elif exc is not None:
        detail = str(exc)

    with _circuit_lock:
        _failure_count += 1
        count = _failure_count
        threshold = _failure_threshold()
        newly_opened = False
        if not _disabled_due_to_failures and count >= threshold:
            _disabled_due_to_failures = True
            newly_opened = True
        disabled = _disabled_due_to_failures

    if newly_opened:
        logger.warning(
            "[langfuse] disabling tracing for the rest of this run after %d "
            "failure(s) (threshold=%d); evaluation continues without telemetry. "
            "last_error_at=%s %s",
            count,
            threshold,
            where,
            detail,
        )
        try:
            _shared_handler.set(None)
        except Exception:
            pass
        return

    if disabled:
        # Already open: keep logs quiet beyond the first disable message.
        logger.debug("[langfuse] %s failed while circuit open: %s", where, detail)
        return

    logger.warning(
        "[langfuse] %s failed (non-fatal, %d/%d): %s",
        where,
        count,
        threshold,
        detail,
    )


def is_langfuse_enabled() -> bool:
    """Return True when LangFuse credentials are present and tracing is not disabled.

    Returns False when credentials are missing, ``LANGFUSE_TRACING_ENABLED`` is
    falsy, or the failure circuit breaker has opened for this process.
    """
    if is_langfuse_circuit_open():
        return False
    flag = os.getenv("LANGFUSE_TRACING_ENABLED", "1").strip().lower()
    if flag in _FALSEY:
        return False
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public_key and secret_key)


def _import_callback_handler() -> Any | None:
    """Lazily import LangFuse CallbackHandler; return None on failure."""
    try:
        from langfuse.langchain import CallbackHandler  # type: ignore

        return CallbackHandler
    except Exception:
        try:
            from langfuse.callback import CallbackHandler  # type: ignore

            return CallbackHandler
        except Exception as exc:  # pragma: no cover – optional dependency path
            logger.warning("[langfuse] CallbackHandler unavailable: %s", exc)
            return None


def _usage_token_counts(usage: Any) -> tuple[int, int]:
    """Extract (input, output) token counts from LangFuse / OpenAI usage dicts."""
    if not isinstance(usage, dict):
        return 0, 0
    prompt = usage.get("input")
    if prompt is None:
        prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens")
    completion = usage.get("output")
    if completion is None:
        completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens")
    try:
        in_tok = int(prompt or 0)
    except Exception:
        in_tok = 0
    try:
        out_tok = int(completion or 0)
    except Exception:
        out_tok = 0
    return in_tok, out_tok


class _CostInjectingObservation:
    """Proxy that injects MODEL_COSTS-derived ``cost_details`` on generation.update()."""

    def __init__(self, inner: Any):
        self._inner = inner

    def update(self, *args: Any, **kwargs: Any) -> Any:
        if "cost_details" not in kwargs:
            usage = kwargs.get("usage_details")
            if usage is None:
                usage = kwargs.get("usage")
            in_tok, out_tok = _usage_token_counts(usage)
            model = _model_name.get() or str(kwargs.get("model") or "")
            if model or in_tok or out_tok:
                try:
                    from src.config.model_costs import estimate_usd_cost

                    cost_in, cost_out, total = estimate_usd_cost(model, in_tok, out_tok)
                    if cost_in or cost_out or total or model:
                        kwargs["cost_details"] = {
                            "input": float(cost_in),
                            "output": float(cost_out),
                            "total": float(total),
                        }
                except Exception:
                    pass
        result = self._inner.update(*args, **kwargs)
        # generation.update(...).end() chaining — keep proxy if update returns self-like
        if result is self._inner:
            return self
        return result

    def end(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.end(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _wrap_handler_with_cost_details(handler: Any) -> Any:
    """Wrap CallbackHandler._detach_observation so generation updates include costs."""
    original = getattr(handler, "_detach_observation", None)
    if not callable(original):
        return handler

    def _detach(run_id: Any) -> Any:
        obs = original(run_id)
        if obs is None:
            return None
        return _CostInjectingObservation(obs)

    try:
        handler._detach_observation = _detach  # type: ignore[method-assign]
    except Exception:
        return handler
    return handler


def _sanitize_otel_attribute_value(value: Any) -> Any:
    """Coerce attribute values to OTEL-safe primitives and bound string size."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(value, str):
        text = value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if len(text) > _MAX_OTEL_ATTR_CHARS:
            return text[: _MAX_OTEL_ATTR_CHARS - 3] + "..."
        return text
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value[:64]:
            if isinstance(item, bool):
                out.append(item)
            elif isinstance(item, (int, float)):
                out.append(item)
            else:
                out.append(_sanitize_otel_attribute_value(str(item)))
        return out
    # Nested / unsupported types become short strings.
    return _sanitize_otel_attribute_value(str(value))


def _sanitize_span_attributes(attributes: Any) -> dict[str, Any]:
    if not isinstance(attributes, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        try:
            key_s = str(key)
        except Exception:
            continue
        if not key_s or key_s.lower() in {"proto", "constructor", "prototype"}:
            continue
        cleaned[key_s] = _sanitize_otel_attribute_value(value)
    return cleaned


def _install_resilient_span_export_guard() -> None:
    """Make Langfuse/OTLP span export failures non-fatal to evaluation runs.

    Large LLM inputs/outputs can trigger ``google.protobuf.message.EncodeError``
    inside the OTEL exporter. That must never abort scenario processing.
    """
    global _export_guard_installed
    if _export_guard_installed:
        return
    try:
        from langfuse._client.span_exporter import (  # type: ignore
            LangfuseTransformingSpanExporter,
        )
        from opentelemetry.sdk.trace.export import SpanExportResult  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.debug("[langfuse] export guard unavailable: %s", exc)
        _export_guard_installed = True
        return

    if getattr(LangfuseTransformingSpanExporter.export, "_ghacr_guarded", False):
        _export_guard_installed = True
        return

    original_export = LangfuseTransformingSpanExporter.export

    def _safe_export(self: Any, spans: Any) -> Any:
        if is_langfuse_circuit_open():
            try:
                return SpanExportResult.FAILURE
            except Exception:  # pragma: no cover
                return None
        try:
            # Best-effort truncation of oversized attributes before protobuf encode.
            safe_spans = []
            for span in spans or []:
                try:
                    attrs = getattr(span, "attributes", None)
                    if attrs:
                        cleaned = _sanitize_span_attributes(dict(attrs))
                        clone = getattr(self, "_clone_span", None)
                        if callable(clone):
                            span = clone(span=span, attributes=cleaned)
                except Exception:
                    pass
                safe_spans.append(span)
            result = original_export(self, safe_spans)
            # Treat explicit FAILURE returns as a counted telemetry fault.
            try:
                if result == SpanExportResult.FAILURE:
                    _record_langfuse_failure("span_export", "SpanExportResult.FAILURE")
            except Exception:
                pass
            return result
        except Exception as exc:
            _record_langfuse_failure("span_export", exc)
            try:
                return SpanExportResult.FAILURE
            except Exception:  # pragma: no cover
                return None

    _safe_export._ghacr_guarded = True  # type: ignore[attr-defined]
    LangfuseTransformingSpanExporter.export = _safe_export  # type: ignore[method-assign]
    _export_guard_installed = True


def _ensure_shared_handler() -> Any | None:
    """Create or return the per-scenario shared CallbackHandler."""
    existing = _shared_handler.get()
    if existing is not None:
        return existing
    if not is_langfuse_enabled():
        return None
    _install_resilient_span_export_guard()
    # Bound OTEL attribute payloads when the SDK honors the env var.
    os.environ.setdefault(
        "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", str(_MAX_OTEL_ATTR_CHARS)
    )
    handler_cls = _import_callback_handler()
    if handler_cls is None:
        return None
    try:
        handler = _wrap_handler_with_cost_details(handler_cls())
    except Exception as exc:
        _record_langfuse_failure("callback_handler_init", exc)
        return None
    _shared_handler.set(handler)
    return handler


def set_run_context(
    *,
    eval_method: str,
    scenario_id: str,
    model_name: str | None = None,
) -> None:
    """Bind scenario metadata and reset per-scenario LangFuse / token state."""
    _eval_method.set(str(eval_method or ""))
    _scenario_id.set(str(scenario_id or ""))
    _model_name.set(str(model_name or ""))
    _llm_node.set("")
    _llm_calls.set([])
    _last_llm_calls.set([])
    _shared_handler.set(None)
    if is_langfuse_enabled():
        _ensure_shared_handler()


def clear_run_context() -> None:
    """Reset scenario metadata and shared LangFuse state after a run completes.

    Preserves a snapshot of ``llm_calls`` so the run ledger can still read
    token records after the scenario finishes.
    """
    calls = _llm_calls.get()
    if calls is not None:
        _last_llm_calls.set(list(calls))
    _eval_method.set("")
    _scenario_id.set("")
    _model_name.set("")
    _llm_node.set("")
    _llm_calls.set(None)
    _shared_handler.set(None)


def get_run_context() -> dict[str, str]:
    """Return the current run context as a plain dict (for tests/diagnostics)."""
    return {
        "eval_method": _eval_method.get(),
        "scenario_id": _scenario_id.get(),
        "model_name": _model_name.get(),
    }


def trace_name_for_method(eval_method: str | None = None) -> str:
    """Return the LangFuse trace name for an eval method (method only).

    Low-cardinality operation identifier so traces group by agent method
    in the LangFuse UI: ``base_a``, ``base_b``, ``agent``, ``bypass7``,
    ``force_mix``.
    """
    method = (eval_method if eval_method is not None else _eval_method.get()) or ""
    method = str(method).strip()
    return method or "unknown"


def make_trace_name(eval_method: str | None = None, scenario_id: str | None = None) -> str:
    """Build the canonical LangFuse / LangChain run name (= eval method).

    The scenario id argument is accepted for backward compatibility but is
    ignored; scenario identity is carried via ``langfuse_session_id`` and
    tags instead of the trace name.
    """
    _ = scenario_id  # kept for call-site compatibility; not part of the name
    return trace_name_for_method(eval_method)


def set_llm_node(node: str | None) -> None:
    """Label the next LLM generation with a multi-agent node name."""
    _llm_node.set(str(node or "").strip())


def get_llm_node() -> str:
    return _llm_node.get() or ""


def clear_llm_node() -> None:
    _llm_node.set("")


def get_llm_calls() -> list[dict[str, Any]]:
    """Return structured per-LLM token records for this (or last) scenario."""
    calls = _llm_calls.get()
    if calls:
        return list(calls)
    last = _last_llm_calls.get()
    return list(last) if last else []


def append_llm_call(record: dict[str, Any]) -> None:
    """Append one LLM call record (prompt/completion tokens, node, etc.)."""
    calls = _llm_calls.get()
    if calls is None:
        calls = []
        _llm_calls.set(calls)
    calls.append(dict(record))


def get_shared_handler() -> Any | None:
    """Return the per-scenario shared CallbackHandler, if any."""
    return _shared_handler.get()


@contextmanager
def scenario_observation(name: str) -> Iterator[Any | None]:
    """Wrap a scenario run so all LLM invokes nest under one LangFuse observation.

    Yields the observation/span object when available (else ``None``) so callers
    can attach scenario-level cost metadata before the context exits.

    No-op when LangFuse is disabled or the SDK observation API is unavailable.
    """
    if not is_langfuse_enabled():
        yield None
        return

    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
    except Exception:
        try:
            from langfuse import Langfuse  # type: ignore

            client = Langfuse()
        except Exception as exc:  # pragma: no cover
            _record_langfuse_failure("scenario_observation_client", exc)
            yield None
            return

    start = getattr(client, "start_as_current_observation", None)
    if start is None:
        # Older SDKs may only expose start_as_current_span
        start = getattr(client, "start_as_current_span", None)
    if start is None:
        yield None
        return

    try:
        cm = start(as_type="span", name=name)
    except TypeError:
        try:
            cm = start(name=name)
        except Exception as exc:  # pragma: no cover
            _record_langfuse_failure("scenario_observation_start", exc)
            yield None
            return
    except Exception as exc:  # pragma: no cover
        _record_langfuse_failure("scenario_observation_start", exc)
        yield None
        return

    with cm as obs:
        yield obs

def update_observation_cost_metadata(
    observation: Any | None,
    *,
    tokens_in: int,
    tokens_out: int,
    cost_in: float,
    cost_out: float,
    total_cost: float,
    model_name: str | None = None,
) -> None:
    """Attach scenario-level token/cost totals to a LangFuse parent observation."""
    if observation is None or not is_langfuse_enabled():
        return
    metadata = _sanitize_span_attributes(
        {
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "cost_in": float(cost_in),
            "cost_out": float(cost_out),
            "total_cost": float(total_cost),
            **({"model_name": str(model_name)} if model_name else {}),
        }
    )
    try:
        update = getattr(observation, "update", None)
        if callable(update):
            update(metadata=metadata)
            return
    except Exception as exc:
        _record_langfuse_failure("observation_cost_metadata", exc)
    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
        fn = getattr(client, "update_current_span", None) or getattr(
            client, "update_current_observation", None
        )
        if callable(fn):
            fn(metadata=metadata)
    except Exception as exc:
        _record_langfuse_failure("current_span_cost_metadata", exc)


def build_langfuse_invoke_config(
    existing: Any | None = None,
    *,
    model_name: str | None = None,
    observation_name: str | None = None,
) -> Any | None:
    """Merge LangFuse callbacks/metadata into a LangChain invoke config.

    Reuses the per-scenario shared CallbackHandler so multi-agent LLM calls
    nest under one parent observation. When tracing is disabled, returns
    ``existing`` unchanged.

    Trace attributes (Langfuse v4 CallbackHandler metadata keys):
    - ``langfuse_trace_name`` = eval method
    - ``langfuse_session_id`` = scenario id
    - ``langfuse_tags`` = ``[method, "scenario:<id>"]``
    """
    if not is_langfuse_enabled():
        return existing

    handler = _ensure_shared_handler()
    if handler is None:
        return existing

    eval_method = _eval_method.get() or "unknown"
    scenario_id = _scenario_id.get() or "unknown"
    resolved_model = model_name or _model_name.get() or ""
    trace_name = trace_name_for_method(eval_method)
    node = get_llm_node()
    run_name = observation_name or node or trace_name

    config: dict[str, Any]
    if existing is None:
        config = {}
    elif isinstance(existing, dict):
        config = dict(existing)
    else:
        try:
            config = dict(existing)  # type: ignore[arg-type]
        except Exception:
            return existing

    callbacks = list(config.get("callbacks") or [])
    if handler not in callbacks:
        callbacks.append(handler)
    config["callbacks"] = callbacks
    config["run_name"] = run_name

    metadata = dict(config.get("metadata") or {})
    metadata.update(
        {
            "langfuse_trace_name": trace_name,
            "langfuse_session_id": scenario_id,
            "eval_method": eval_method,
            "scenario_id": scenario_id,
            "model_name": resolved_model,
        }
    )
    if node:
        metadata["llm_node"] = node

    # Trace-level tags via CallbackHandler (langfuse_tags); also keep LangChain tags
    desired_tags: list[str] = []
    if eval_method:
        desired_tags.append(eval_method)
    if scenario_id and scenario_id != "unknown":
        desired_tags.append(f"scenario:{scenario_id}")
    if node:
        desired_tags.append(node)

    existing_lf_tags = metadata.get("langfuse_tags")
    lf_tags: list[str] = []
    if isinstance(existing_lf_tags, list):
        lf_tags = [str(t) for t in existing_lf_tags]
    for t in desired_tags:
        if t not in lf_tags:
            lf_tags.append(t)
    metadata["langfuse_tags"] = lf_tags
    config["metadata"] = metadata

    tags = list(config.get("tags") or [])
    for t in desired_tags:
        if t and t not in tags:
            tags.append(t)
    config["tags"] = tags

    return config


def flush_langfuse() -> None:
    """Best-effort flush of pending LangFuse events.

    Export/serialization failures (e.g. protobuf EncodeError on oversized
    attributes) are logged and swallowed so evaluation continues. Repeated
    failures open the circuit breaker and disable further tracing attempts.
    """
    if not is_langfuse_enabled():
        return
    _install_resilient_span_export_guard()
    try:
        from langfuse import get_client  # type: ignore

        get_client().flush()
    except Exception as first_exc:
        try:
            from langfuse import Langfuse  # type: ignore

            Langfuse().flush()
        except Exception as exc:  # pragma: no cover
            _record_langfuse_failure("flush", exc or first_exc)


class LangfuseLLMWrapper:
    """Wrap an LLM runnable and inject LangFuse callbacks on every invoke."""

    def __init__(self, inner: Any, *, model_name: str):
        self._inner = inner
        self._model_name = model_name

    def with_config(self, config: dict[str, Any] | None = None) -> "LangfuseLLMWrapper":
        if hasattr(self._inner, "with_config"):
            self._inner = self._inner.with_config(config)
        return self

    def invoke(self, prompt_input: Any, config: Any | None = None) -> Any:
        # Prefer node-labeled run_name for generations; keep method as langfuse_trace_name.
        node = get_llm_node()
        merged = build_langfuse_invoke_config(
            config,
            model_name=self._model_name,
            observation_name=node or None,
        )
        return self._inner.invoke(prompt_input, config=merged)

    async def ainvoke(self, prompt_input: Any, config: Any | None = None) -> Any:
        node = get_llm_node()
        merged = build_langfuse_invoke_config(
            config,
            model_name=self._model_name,
            observation_name=node or None,
        )
        if hasattr(self._inner, "ainvoke"):
            return await self._inner.ainvoke(prompt_input, config=merged)
        return self._inner.invoke(prompt_input, config=merged)


__all__ = [
    "LangfuseLLMWrapper",
    "append_llm_call",
    "build_langfuse_invoke_config",
    "clear_llm_node",
    "clear_run_context",
    "flush_langfuse",
    "get_langfuse_failure_count",
    "get_llm_calls",
    "get_llm_node",
    "get_run_context",
    "get_shared_handler",
    "is_langfuse_circuit_open",
    "is_langfuse_enabled",
    "make_trace_name",
    "reset_langfuse_circuit_breaker",
    "scenario_observation",
    "set_llm_node",
    "set_run_context",
    "trace_name_for_method",
    "update_observation_cost_metadata",
]
