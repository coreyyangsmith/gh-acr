"""LangFuse observability helpers for LLM inference.

Soft-disabled when ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are unset
(or ``LANGFUSE_TRACING_ENABLED`` is falsy). When enabled, each scenario reuses
one shared CallbackHandler and wraps the run in a parent observation so all
multi-agent LLM calls nest under ``{eval_method}-scenario-{id}``.
"""

from __future__ import annotations

import logging
import os
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


def is_langfuse_enabled() -> bool:
    """Return True when LangFuse credentials are present and tracing is not disabled."""
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


def _ensure_shared_handler() -> Any | None:
    """Create or return the per-scenario shared CallbackHandler."""
    existing = _shared_handler.get()
    if existing is not None:
        return existing
    if not is_langfuse_enabled():
        return None
    handler_cls = _import_callback_handler()
    if handler_cls is None:
        return None
    try:
        handler = handler_cls()
    except Exception as exc:
        logger.warning("[langfuse] Failed to construct CallbackHandler: %s", exc)
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


def make_trace_name(eval_method: str | None = None, scenario_id: str | None = None) -> str:
    """Build the canonical LangFuse / LangChain run name."""
    method = (eval_method if eval_method is not None else _eval_method.get()) or "unknown"
    sid = (scenario_id if scenario_id is not None else _scenario_id.get()) or "unknown"
    return f"{method}-scenario-{sid}"


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
def scenario_observation(name: str) -> Iterator[None]:
    """Wrap a scenario run so all LLM invokes nest under one LangFuse observation.

    No-op when LangFuse is disabled or the SDK observation API is unavailable.
    """
    if not is_langfuse_enabled():
        yield
        return

    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
    except Exception:
        try:
            from langfuse import Langfuse  # type: ignore

            client = Langfuse()
        except Exception as exc:  # pragma: no cover
            logger.debug("[langfuse] scenario_observation unavailable: %s", exc)
            yield
            return

    start = getattr(client, "start_as_current_observation", None)
    if start is None:
        # Older SDKs may only expose start_as_current_span
        start = getattr(client, "start_as_current_span", None)
    if start is None:
        yield
        return

    try:
        cm = start(as_type="span", name=name)
    except TypeError:
        try:
            cm = start(name=name)
        except Exception as exc:  # pragma: no cover
            logger.debug("[langfuse] start observation failed: %s", exc)
            yield
            return
    except Exception as exc:  # pragma: no cover
        logger.debug("[langfuse] start observation failed: %s", exc)
        yield
        return

    with cm:
        yield


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
    """
    if not is_langfuse_enabled():
        return existing

    handler = _ensure_shared_handler()
    if handler is None:
        return existing

    eval_method = _eval_method.get() or "unknown"
    scenario_id = _scenario_id.get() or "unknown"
    resolved_model = model_name or _model_name.get() or ""
    trace_name = make_trace_name(eval_method, scenario_id)
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
            "eval_method": eval_method,
            "scenario_id": scenario_id,
            "model_name": resolved_model,
        }
    )
    if node:
        metadata["llm_node"] = node
    config["metadata"] = metadata

    tags = list(config.get("tags") or [])
    if eval_method and eval_method not in tags:
        tags.append(eval_method)
    if node and node not in tags:
        tags.append(node)
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
        # Prefer node-labeled run_name for generations; keep scenario trace_name in metadata.
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
    "get_llm_calls",
    "get_llm_node",
    "get_run_context",
    "get_shared_handler",
    "is_langfuse_enabled",
    "make_trace_name",
    "scenario_observation",
    "set_llm_node",
    "set_run_context",
]
