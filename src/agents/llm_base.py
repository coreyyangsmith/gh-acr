"""Centralized LLM backend registry and initialization.

This module provides the primary interface for obtaining LLM backends throughout
the application. It abstracts away the differences between OpenAI, Groq, and
local Hugging Face models behind a unified interface.

Main Functions
--------------
- `get_backend(model_name)`: Returns an (encoder, llm) tuple for any supported model

Supported Model Schemes
-----------------------
1. **openai/<model>**: OpenAI API via `langchain_openai.ChatOpenAI`
   - Requires OPENAI_API_KEY environment variable
   - Examples: "openai/gpt-4o-mini", "openai/gpt-4.1-nano-2025-04-14"

2. **groq:<model>**: Groq API via `langchain_groq.ChatGroq`
   - Requires GROQ_API_KEY environment variable
   - Examples: "groq:llama-3.1-8b-instant", "groq:qwen/qwen3-32b"

3. **local:<path>**: Locally loaded transformers model (CPU/GPU)
   - No API key required, runs inference locally
   - Examples: "local:meta-llama/Llama-3.2-1B", "local:Qwen/Qwen3-8B"

Architecture
------------
The module uses a caching strategy (@lru_cache) to avoid redundant model loading.
Each backend is wrapped with:
1. **TruncatingLLMWrapper**: Clips over-long prompts to model limits
2. **RateLimitAndCostHandler**: Enforces rate limits and logs token costs
3. **_ThreadSafeLLMWrapper**: Prevents tokenizer concurrency issues
4. **Langfuse callback** (optional): For observability when LANGFUSE_ENABLED=1

Example Usage
-------------
>>> from src.agents.llm_base import get_backend, count_tokens
>>> encoder, llm = get_backend("openai/gpt-4o-mini")
>>> result = llm.invoke("Hello, world!")
>>> token_count = count_tokens(encoder, result.content)
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from .backends import create_groq_backend, create_local_backend, create_openai_backend
from .callbacks import RateLimitAndCostHandler
from .token_utils import count_tokens, tiktoken_encoder
from .truncation_wrapper import TruncatingLLMWrapper
from ..config.model_costs import MODEL_COSTS

# Optional Langfuse callback integration (best-effort)
try:  # pragma: no cover
    from langfuse.langchain import CallbackHandler as LangfuseCallback  # type: ignore
except Exception:  # pragma: no cover
    LangfuseCallback = None  # type: ignore


logger = logging.getLogger(__name__)


class _ThreadSafeLLMWrapper:
    """Thread-safe wrapper to prevent tokenizer concurrency crashes.

    Some tokenizers (especially HuggingFace-based) are not thread-safe.
    This wrapper serializes calls to prevent race conditions.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._lock = threading.Lock()

    def with_config(self, config: dict[str, Any] | None = None) -> "_ThreadSafeLLMWrapper":
        """Chain configuration to the inner LLM."""
        with self._lock:
            if hasattr(self._inner, "with_config"):
                self._inner = self._inner.with_config(config)
        return self

    def invoke(self, prompt_input: Any, config: Any | None = None) -> Any:
        """Invoke the LLM with thread-safety."""
        with self._lock:
            return self._inner.invoke(prompt_input, config=config)

    async def ainvoke(self, prompt_input: Any, config: Any | None = None) -> Any:
        """Async invoke - delegates to inner implementation."""
        # Note: async calls still need lock since tokenization may not be async-safe
        with self._lock:
            if hasattr(self._inner, "ainvoke"):
                return await self._inner.ainvoke(prompt_input, config=config)
            return self._inner.invoke(prompt_input, config=config)


@lru_cache(maxsize=None)
def get_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:
    """Return an (encoder, llm) tuple for the specified model.

    This is the primary entry point for obtaining LLM backends. The function
    caches results so multiple calls with the same model name are efficient.

    Parameters
    ----------
    model_name
        A URI-like model identifier:
        - "openai/<model>" for OpenAI models
        - "groq:<model>" for Groq models
        - "local:<path>" for local HuggingFace models

    Returns
    -------
    Tuple[Optional[Any], Optional[Any]]
        A tuple of (encoder, llm) where:
        - encoder: A tiktoken Encoding or HuggingFace tokenizer with .encode()
        - llm: A LangChain chat model wrapped with rate limiting and truncation

    Raises
    ------
    ValueError
        If the model_name scheme is not recognized.
    RuntimeError
        If the backend cannot be initialized (e.g., missing API key).

    Examples
    --------
    >>> encoder, llm = get_backend("openai/gpt-4o-mini")
    >>> result = llm.invoke("What is 2+2?")
    >>> print(result.content)
    """
    raw_llm: Optional[Any] = None
    enc: Optional[Any] = None

    # Route to appropriate backend based on model scheme
    if model_name.startswith("openai/"):
        enc, raw_llm = create_openai_backend(model_name)

    elif model_name.startswith("groq:"):
        logger.info("[groq] Using Groq backend: model=%s", model_name)
        enc, raw_llm = create_groq_backend(model_name)

    elif model_name.startswith("local:"):
        logger.info("[local] Using local Transformers backend: model=%s", model_name)
        enc, raw_llm = create_local_backend(model_name)

    else:
        msg = f"Unknown model_name scheme: {model_name!r}. Expected openai/, groq:, or local:"
        logger.error(msg)
        raise ValueError(msg)

    if raw_llm is None:
        msg = f"Failed to initialize LLM backend for model_name={model_name}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Wrap with truncation to clip over-long prompts
    try:
        raw_llm = TruncatingLLMWrapper(raw_llm, encoder=enc, model_name=model_name)
    except Exception:
        pass

    # Attach rate limiting and cost logging callback
    handler = RateLimitAndCostHandler(encoder=enc, model_name=model_name)
    try:
        raw_llm = raw_llm.with_config({"callbacks": [handler]})
    except Exception:
        pass

    # Optional: Langfuse callback for observability
    if os.getenv("LANGFUSE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on") and \
       os.getenv("LANGFUSE_READY", "0").strip() in ("1", "true", "TRUE"):
        try:
            if LangfuseCallback is not None:
                lf_handler = LangfuseCallback()
                raw_llm = raw_llm.with_config({"callbacks": [lf_handler]})
        except Exception:
            pass

    # Final wrapper: thread-safe wrapper to prevent tokenizer concurrency issues
    try:
        raw_llm = _ThreadSafeLLMWrapper(raw_llm)
    except Exception:
        pass

    return enc, raw_llm


# Re-export commonly used functions for convenience
__all__ = [
    "get_backend",
    "count_tokens",
    "tiktoken_encoder",
]
