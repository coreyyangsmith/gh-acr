"""Centralised LLM backend registry.

`get_backend(model_name)` returns a `(encoder, llm)` tuple where

* **encoder** – a tiktoken Encoding (or Hugging-Face tokenizer) implementing
  `.encode(text)` so we can count tokens; may be `None` if unavailable.
* **llm** – a LangChain **chat model** (or pipeline) that can be piped into
  prompts.  `None` indicates that the requested backend is not usable (e.g.
  missing credentials), so callers should fall back to a non-LLM strategy.

The function supports these URI-like schemes:

1. `openai/<model>` – via `langchain_openai.ChatOpenAI`
2. `local:<path>` – locally loaded transformers model (CPU / GPU)
3. `groq:<model>` – via Groq API using `langchain_groq.ChatGroq`
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Optional
import os
import logging
from contextlib import nullcontext
import threading
import asyncio

# Optional Langfuse callback integration (best-effort)
try:  # pragma: no cover
    from langfuse.langchain import CallbackHandler as LangfuseCallback  # type: ignore
except Exception:  # pragma: no cover
    LangfuseCallback = None  # type: ignore

from ..config.model_costs import MODEL_COSTS
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage

from .token_utils import tiktoken_encoder, count_tokens
from .callbacks import RateLimitAndCostHandler
from .truncation_wrapper import TruncatingLLMWrapper
from .local_textgen import build_local_text_generator, generate_local_text
from .backends import create_openai_backend, create_groq_backend, create_local_backend

logger = logging.getLogger(__name__)


# Local backend helpers moved to dedicated modules


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to backends/hf_utils.py)


# (extracted to local_textgen.py)


# (extracted to local_textgen.py)


# ---------------------------------------------------------------------------
# Lazy imports (so that users without transformers / openai can still run base)
# ---------------------------------------------------------------------------

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError: # pragma: no cover
    get_openai_callback = None


## Callback moved to src/agents/callbacks.py


## Truncation wrapper moved to src/agents/truncation_wrapper.py

@lru_cache(maxsize=None)
def get_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:  # noqa: D401
    """Return *(encoder, llm)* for *model_name*.

    Raises a RuntimeError/ValueError if the requested backend cannot be
    initialized. Backends are cached so multiple calls with the same name
    are cheap.
    """
    raw_llm: Optional[Any] = None
    enc: Optional[Any] = None

    # OpenAI -----------------------------------------------------------------
    if model_name.startswith("openai/"):
        enc, raw_llm = create_openai_backend(model_name)

    # Groq API ---------------------------------------------------------------
    elif model_name.startswith("groq:"):
        logger.info("[groq] Using Groq backend: model=%s", model_name)
        enc, raw_llm = create_groq_backend(model_name)

    # Local transformers model ----------------------------------------------
    elif model_name.startswith("local:"):
        logger.info("[local] Using local Transformers backend: model=%s", model_name)
        enc, raw_llm = create_local_backend(model_name)

    else:
        msg = f"Unknown model_name scheme {model_name}"
        logger.error(msg)
        raise ValueError(msg)

    if raw_llm is None:
        msg = f"Failed to initialize LLM backend for model_name={model_name}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Always wrap with truncation so over-long prompts are clipped to limits
    try:
        raw_llm = TruncatingLLMWrapper(raw_llm, encoder=enc, model_name=model_name)
    except Exception:
        pass

    # Attach a LangChain callback for rate limiting and cost logging
    handler = RateLimitAndCostHandler(encoder=enc, model_name=model_name)
    try:
        raw_llm = raw_llm.with_config({"callbacks": [handler]})  # type: ignore[attr-defined]
    except Exception:
        pass

    # Optional: Langfuse callback handler
    if os.getenv("LANGFUSE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on") and os.getenv("LANGFUSE_READY", "0").strip() in ("1", "true", "TRUE"):
        try:
            if LangfuseCallback is not None:
                handler = LangfuseCallback()
                raw_llm = raw_llm.with_config({"callbacks": [handler]})  # type: ignore[attr-defined]
        except Exception:
            pass

    # OUTERMOST: thread-safe wrapper to prevent tokenizer concurrency crashes
    try:
        raw_llm = _ThreadSafeLLMWrapper(raw_llm)
    except Exception:
        pass

    return enc, raw_llm
## Convenience helpers moved to src/agents/token_utils.py
