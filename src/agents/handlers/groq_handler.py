"""Groq API LLM handler (``groq:<model>``).

Groq's Chat Completions API does **not** expose a client-selectable weight
precision / quantization parameter (no FP8 vs BF16 switch). Numerics are
fixed by Groq's hosted model revision. If you need to avoid OpenRouter
endpoints advertised as FP8, use ``openrouter/...`` with the quantization
filter in ``openrouter_handler`` instead of this direct Groq path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from ...config.model_costs import MODEL_COSTS
from .base import BaseLLMHandler

logger = logging.getLogger(__name__)


class GroqHandler(BaseLLMHandler):
    """ChatGroq backend using ``GROQ_API_KEY``.

    Precision/quantization is not configurable via the Groq API; this handler
    only sets model id, temperature, and max tokens.
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
        # No dtype / quantization kwargs: Groq API has none to set.
        common_kwargs = dict(
            groq_api_key=api_key, model=backend_name, temperature=0
        )
        if max_out > 0:
            raw_llm = ChatGroq(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
        else:
            raw_llm = ChatGroq(**common_kwargs)  # type: ignore[call-arg]
        logger.info(
            "[groq] Initialized model=%s (precision not API-configurable)",
            backend_name,
        )
        return None, raw_llm


def create_groq_backend(model_name: str) -> Tuple[Optional[Any], Any]:
    """Initialize Groq chat backend for model_name (groq:<model>)."""
    return GroqHandler().create(model_name)


__all__ = ["GroqHandler", "create_groq_backend"]
