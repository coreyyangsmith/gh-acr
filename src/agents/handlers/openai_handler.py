"""OpenAI API LLM handler (``openai/<model>``)."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from ...config.model_costs import MODEL_COSTS
from ..token_utils import tiktoken_encoder
from .base import BaseLLMHandler

logger = logging.getLogger(__name__)


class OpenAIHandler(BaseLLMHandler):
    """ChatOpenAI backend using ``OPENAI_API_KEY``."""

    scheme = "openai"
    separator = "/"
    api_key_env = "OPENAI_API_KEY"

    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        backend_name = self.parse_model_id(model_name)
        api_key = self.require_api_key()

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI  # type: ignore

        model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(
            f"openai/{backend_name}", {}
        )
        max_out = int(model_cfg.get("output_limit", 0))
        is_gpt5 = backend_name.startswith("gpt-5")

        common_kwargs: dict[str, Any] = dict(api_key=api_key, model=backend_name)
        if not is_gpt5:
            common_kwargs["temperature"] = 0

        if max_out > 0:
            raw_llm = ChatOpenAI(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
        else:
            raw_llm = ChatOpenAI(**common_kwargs)  # type: ignore[call-arg]

        enc = tiktoken_encoder(backend_name)
        return enc, raw_llm


def create_openai_backend(model_name: str) -> Tuple[Optional[Any], Any]:
    """Initialize OpenAI chat backend for model_name (openai/<model>)."""
    return OpenAIHandler().create(model_name)


__all__ = ["OpenAIHandler", "create_openai_backend"]
