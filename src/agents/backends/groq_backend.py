from __future__ import annotations

from typing import Any, Optional, Tuple
import os
import logging

from ...config.model_costs import MODEL_COSTS


logger = logging.getLogger(__name__)


def create_groq_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:  # noqa: D401
    """Initialize Groq chat backend for model_name (groq:<model>)."""
    backend_name = model_name.split(":", 1)[1]
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        msg = f"GROQ_API_KEY missing – cannot load Groq backend for model: {backend_name}"
        logger.error(msg)
        raise RuntimeError(msg)
    try:
        from langchain_groq import ChatGroq  # type: ignore
    except ImportError as exc:  # pragma: no cover
        logger.error("langchain-groq not installed: %s", exc)
        raise RuntimeError("Please install 'langchain-groq' to use groq: models")

    model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(f"groq/{backend_name}", {})
    max_out = int(model_cfg.get("output_limit", 0))
    common_kwargs = dict(groq_api_key=api_key, model=backend_name, temperature=0)
    raw_llm = ChatGroq(max_tokens=max_out, **common_kwargs) if max_out > 0 else ChatGroq(**common_kwargs)  # type: ignore[call-arg]
    enc = None
    return enc, raw_llm


__all__ = ["create_groq_backend"]


