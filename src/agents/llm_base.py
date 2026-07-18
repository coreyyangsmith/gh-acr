"""Centralized LLM backend registry and initialization.

This module provides the primary interface for obtaining LLM backends throughout
the application. It abstracts away the differences between OpenAI, Groq, and
local Hugging Face models behind a unified interface.

Main Functions
--------------
- `get_backend(model_name)`: Returns an (encoder, llm) tuple for any supported model

Supported Model Schemes
-----------------------
1. **openai/<model>**: OpenAI API via `OpenAIHandler`
   - Requires OPENAI_API_KEY environment variable
   - Examples: "openai/gpt-4o-mini", "openai/gpt-4.1-nano-2025-04-14"

2. **openrouter/<provider>/<model>**: OpenRouter (OpenAI-compatible API)
   - Requires OPENROUTER_API_KEY environment variable
   - Examples: "openrouter/anthropic/claude-sonnet-4.5"

3. **groq:<model>**: Groq API via `GroqHandler`
   - Requires GROQ_API_KEY environment variable
   - Examples: "groq:llama-3.1-8b-instant", "groq:qwen/qwen3-32b"

4. **local:<path>**: Locally loaded transformers model (CPU/GPU)
   - No API key required, runs inference locally
   - Examples: "local:meta-llama/Llama-3.2-1B", "local:Qwen/Qwen3-8B"

Architecture
------------
Provider-specific logic lives in ``src.agents.handlers``. This module
resolves a handler via the registry, then wraps the result with:
1. **TruncatingLLMWrapper**: Clips over-long prompts to model limits
2. **RateLimitAndCostHandler**: Enforces rate limits and logs token costs
3. **LangfuseLLMWrapper**: Injects LangFuse callbacks when credentials are set
4. **_ThreadSafeLLMWrapper**: Prevents tokenizer concurrency issues
Results are cached with @lru_cache to avoid redundant model loading.

Example Usage
-------------
>>> from src.agents.llm_base import get_backend, count_tokens
>>> encoder, llm = get_backend("openai/gpt-4o-mini")
>>> result = llm.invoke("Hello, world!")
>>> token_count = count_tokens(encoder, result.content)
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Optional, Tuple

from .callbacks import RateLimitAndCostHandler
from .handlers import create_backend
from .observability import LangfuseLLMWrapper
from .token_utils import count_tokens, tiktoken_encoder
from .truncation_wrapper import TruncatingLLMWrapper


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
        - "openrouter/<provider>/<model>" for OpenRouter models
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
    logger.info("[get_backend] Resolving handler for model=%s", model_name)
    enc, raw_llm = create_backend(model_name)

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

    # Inject LangFuse callbacks on every invoke (soft-disabled without credentials)
    try:
        raw_llm = LangfuseLLMWrapper(raw_llm, model_name=model_name)
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
