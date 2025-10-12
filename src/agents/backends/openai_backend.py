from __future__ import annotations

from typing import Any, Optional, Tuple
import os
import logging

from ..token_utils import tiktoken_encoder
from ...config.model_costs import MODEL_COSTS


logger = logging.getLogger(__name__)


def create_openai_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:  # noqa: D401
    """Initialize OpenAI chat backend and encoder for model_name (openai/<model>)."""
    backend_name = model_name.split("/", 1)[1]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = f"OPENAI_API_KEY missing – cannot load OpenAI backend for model: {backend_name}"
        logger.error(msg)
        raise RuntimeError(msg)
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError:
        from langchain_community.chat_models import ChatOpenAI  # type: ignore

    model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(f"openai/{backend_name}", {})
    max_out = int(model_cfg.get("output_limit", 0))
    is_gpt5 = backend_name.startswith("gpt-5")

    common_kwargs = dict(api_key=api_key, model=backend_name)
    if not is_gpt5:
        common_kwargs["temperature"] = 0  # type: ignore[assignment]

    if max_out > 0:
        raw_llm = ChatOpenAI(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
    else:
        raw_llm = ChatOpenAI(**common_kwargs)  # type: ignore[call-arg]

    enc = tiktoken_encoder(backend_name)
    return enc, raw_llm


__all__ = ["create_openai_backend"]


