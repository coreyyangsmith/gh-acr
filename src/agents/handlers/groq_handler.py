"""Groq API LLM handler (``groq:<model>``)."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from ...config.model_costs import MODEL_COSTS
from .base import BaseLLMHandler

logger = logging.getLogger(__name__)


class GroqHandler(BaseLLMHandler):
    """ChatGroq backend using ``GROQ_API_KEY``."""

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
        common_kwargs = dict(
            groq_api_key=api_key, model=backend_name, temperature=0
        )
        if max_out > 0:
            raw_llm = ChatGroq(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
        else:
            raw_llm = ChatGroq(**common_kwargs)  # type: ignore[call-arg]
        return None, raw_llm


__all__ = ["GroqHandler"]
