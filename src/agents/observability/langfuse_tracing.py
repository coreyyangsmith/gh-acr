"""LangFuse observability helpers for LLM inference.

Soft-disabled when ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are unset
(or ``LANGFUSE_TRACING_ENABLED`` is falsy). When enabled, every LLM invoke gets
a LangChain ``CallbackHandler`` with a trace name that includes the eval method.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

_eval_method: ContextVar[str] = ContextVar("ghacr_eval_method", default="")
_scenario_id: ContextVar[str] = ContextVar("ghacr_scenario_id", default="")
_model_name: ContextVar[str] = ContextVar("ghacr_model_name", default="")

_FALSEY = {"0", "false", "no", "off", ""}


def is_langfuse_enabled() -> bool:
    """Return True when LangFuse credentials are present and tracing is not disabled."""
    flag = os.getenv("LANGFUSE_TRACING_ENABLED", "1").strip().lower()
    if flag in _FALSEY:
        return False
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public_key and secret_key)


def set_run_context(
    *,
    eval_method: str,
    scenario_id: str,
    model_name: str | None = None,
) -> None:
    """Bind scenario metadata used to name LangFuse traces."""
    _eval_method.set(str(eval_method or ""))
    _scenario_id.set(str(scenario_id or ""))
    _model_name.set(str(model_name or ""))


def clear_run_context() -> None:
    """Reset scenario metadata after a run completes."""
    _eval_method.set("")
    _scenario_id.set("")
    _model_name.set("")


def get_run_context() -> dict[str, str]:
    """Return the current run context as a plain dict (for tests/diagnostics)."""
    return {
        "eval_method": _eval_method.get(),
        "scenario_id": _scenario_id.get(),
        "model_name": _model_name.get(),
    }


def make_trace_name(eval_method: str | None = None, scenario_id: str | None = None) -> str:
    """Build the canonical LangFuse / LangChain run name."""
    method = (eval_method if eval_method is not None else _eval_method.get()) or "unknown"
    sid = (scenario_id if scenario_id is not None else _scenario_id.get()) or "unknown"
    return f"{method}-scenario-{sid}"


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


def build_langfuse_invoke_config(
    existing: Any | None = None,
    *,
    model_name: str | None = None,
) -> Any | None:
    """Merge LangFuse callbacks/metadata into a LangChain invoke config.

    When tracing is disabled, returns ``existing`` unchanged.
    """
    if not is_langfuse_enabled():
        return existing

    handler_cls = _import_callback_handler()
    if handler_cls is None:
        return existing

    eval_method = _eval_method.get() or "unknown"
    scenario_id = _scenario_id.get() or "unknown"
    resolved_model = model_name or _model_name.get() or ""
    trace_name = make_trace_name(eval_method, scenario_id)

    config: dict[str, Any]
    if existing is None:
        config = {}
    elif isinstance(existing, dict):
        config = dict(existing)
    else:
        # Preserve non-dict configs by wrapping only what we can
        try:
            config = dict(existing)  # type: ignore[arg-type]
        except Exception:
            return existing

    callbacks = list(config.get("callbacks") or [])
    try:
        callbacks.append(handler_cls())
    except Exception as exc:
        logger.warning("[langfuse] Failed to construct CallbackHandler: %s", exc)
        return existing
    config["callbacks"] = callbacks
    config["run_name"] = trace_name

    metadata = dict(config.get("metadata") or {})
    metadata.update(
        {
            "langfuse_trace_name": trace_name,
            "eval_method": eval_method,
            "scenario_id": scenario_id,
            "model_name": resolved_model,
        }
    )
    config["metadata"] = metadata

    tags = list(config.get("tags") or [])
    if eval_method and eval_method not in tags:
        tags.append(eval_method)
    config["tags"] = tags

    return config


def flush_langfuse() -> None:
    """Best-effort flush of pending LangFuse events."""
    if not is_langfuse_enabled():
        return
    try:
        from langfuse import get_client  # type: ignore

        get_client().flush()
    except Exception:
        try:
            from langfuse import Langfuse  # type: ignore

            Langfuse().flush()
        except Exception as exc:  # pragma: no cover
            logger.debug("[langfuse] flush failed: %s", exc)


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
        merged = build_langfuse_invoke_config(config, model_name=self._model_name)
        return self._inner.invoke(prompt_input, config=merged)

    async def ainvoke(self, prompt_input: Any, config: Any | None = None) -> Any:
        merged = build_langfuse_invoke_config(config, model_name=self._model_name)
        if hasattr(self._inner, "ainvoke"):
            return await self._inner.ainvoke(prompt_input, config=merged)
        return self._inner.invoke(prompt_input, config=merged)


__all__ = [
    "LangfuseLLMWrapper",
    "build_langfuse_invoke_config",
    "clear_run_context",
    "flush_langfuse",
    "get_run_context",
    "is_langfuse_enabled",
    "make_trace_name",
    "set_run_context",
]
