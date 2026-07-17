"""OpenRouter API LLM handler (``openrouter/<provider>/<model>``).

Uses the OpenAI-compatible Chat Completions endpoint documented at
https://openrouter.ai/docs/quickstart via ``langchain_openai.ChatOpenAI``
with ``base_url`` pointed at OpenRouter.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from ...config.model_costs import MODEL_COSTS
from ..token_utils import tiktoken_encoder
from .base import BaseLLMHandler

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterHandler(BaseLLMHandler):
    """ChatOpenAI pointed at OpenRouter using ``OPENROUTER_API_KEY``."""

    scheme = "openrouter"
    separator = "/"
    api_key_env = "OPENROUTER_API_KEY"

    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        backend_name = self.parse_model_id(model_name)
        api_key = self.require_api_key()

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI  # type: ignore

        base_url = (
            os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
            or DEFAULT_OPENROUTER_BASE_URL
        )

        default_headers: dict[str, str] = {}
        referer = (
            os.getenv("OPENROUTER_HTTP_REFERER")
            or os.getenv("HTTP_REFERER")
            or ""
        ).strip()
        app_title = (os.getenv("OPENROUTER_APP_TITLE") or "").strip()
        if referer:
            default_headers["HTTP-Referer"] = referer
        if app_title:
            default_headers["X-OpenRouter-Title"] = app_title

        model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(
            f"openrouter/{backend_name}", {}
        )
        max_out = int(model_cfg.get("output_limit", 0))

        common_kwargs: dict[str, Any] = dict(
            api_key=api_key,
            base_url=base_url,
            model=backend_name,
            temperature=0,
        )
        if default_headers:
            common_kwargs["default_headers"] = default_headers

        if max_out > 0:
            raw_llm = ChatOpenAI(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
        else:
            raw_llm = ChatOpenAI(**common_kwargs)  # type: ignore[call-arg]

        # Prefer tiktoken for OpenAI-served models under OpenRouter; else cl100k.
        enc_name = backend_name.split("/", 1)[-1] if "/" in backend_name else backend_name
        enc = tiktoken_encoder(enc_name)
        logger.info(
            "[openrouter] Initialized model=%s base_url=%s",
            backend_name,
            base_url,
        )
        return enc, raw_llm


__all__ = ["OpenRouterHandler", "DEFAULT_OPENROUTER_BASE_URL"]
