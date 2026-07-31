"""Groq API LLM handler (``groq:<model>``).

Groq's Chat Completions API does **not** expose a client-selectable weight
precision / quantization parameter (no FP8 vs BF16 switch). Numerics are
fixed by Groq's hosted model revision. If you need to avoid OpenRouter
endpoints advertised as FP8, use ``openrouter/...`` with the quantization
filter in ``openrouter_handler`` instead of this direct Groq path.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from ...config.model_costs import MODEL_COSTS
from .base import BaseLLMHandler

logger = logging.getLogger(__name__)

# Bounded HTTP timeout so hung Groq calls become retryable instead of freezing
# the worker forever. Override via GROQ_REQUEST_TIMEOUT (seconds).
_DEFAULT_GROQ_REQUEST_TIMEOUT_S = 120.0


def resolve_groq_request_timeout() -> float:
    """Return the Groq ``request_timeout`` in seconds (env-overridable)."""
    raw = (os.getenv("GROQ_REQUEST_TIMEOUT") or "").strip()
    if not raw:
        return _DEFAULT_GROQ_REQUEST_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "[groq] Invalid GROQ_REQUEST_TIMEOUT=%r; using default %.1fs",
            raw,
            _DEFAULT_GROQ_REQUEST_TIMEOUT_S,
        )
        return _DEFAULT_GROQ_REQUEST_TIMEOUT_S
    if value <= 0:
        logger.warning(
            "[groq] Non-positive GROQ_REQUEST_TIMEOUT=%r; using default %.1fs",
            raw,
            _DEFAULT_GROQ_REQUEST_TIMEOUT_S,
        )
        return _DEFAULT_GROQ_REQUEST_TIMEOUT_S
    return value


class GroqHandler(BaseLLMHandler):
    """ChatGroq backend using ``GROQ_API_KEY``.

    Precision/quantization is not configurable via the Groq API; this handler
    only sets model id, temperature, max tokens, and request timeout.
    """

    scheme = "groq"
    separator = ":"
    api_key_env = "GROQ_API_KEY"

    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        backend_name = self.parse_model_id(model_name)
        api_key = self.require_api_key()

        try:
            from langchain_groq import ChatGroq  # type: ignore
        except ImportError as exc:  # pragma: no cover
            logger.error("langchain-groq not installed: %s", exc)
            raise RuntimeError(
                "Please install 'langchain-groq' to use groq: models"
            ) from exc

        model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(
            f"groq/{backend_name}", {}
        )
        max_out = int(model_cfg.get("output_limit", 0))
        request_timeout = resolve_groq_request_timeout()
        # No dtype / quantization kwargs: Groq API has none to set.
        common_kwargs: dict[str, Any] = dict(
            groq_api_key=api_key,
            model=backend_name,
            temperature=0,
            request_timeout=request_timeout,
        )
        if max_out > 0:
            raw_llm = ChatGroq(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
        else:
            raw_llm = ChatGroq(**common_kwargs)  # type: ignore[call-arg]
        logger.info(
            "[groq] Initialized model=%s request_timeout=%.1fs "
            "(precision not API-configurable)",
            backend_name,
            request_timeout,
        )
        return None, raw_llm


def create_groq_backend(model_name: str) -> Tuple[Optional[Any], Any]:
    """Initialize Groq chat backend for model_name (groq:<model>)."""
    return GroqHandler().create(model_name)


__all__ = [
    "GroqHandler",
    "create_groq_backend",
    "resolve_groq_request_timeout",
]
