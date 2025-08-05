"""Centralised LLM backend registry.

`get_backend(model_name)` returns a `(encoder, llm)` tuple where

* **encoder** – a tiktoken Encoding (or Hugging-Face tokenizer) implementing
  `.encode(text)` so we can count tokens; may be `None` if unavailable.
* **llm** – a LangChain **chat model** (or pipeline) that can be piped into
  prompts.  `None` indicates that the requested backend is not usable (e.g.
  missing credentials), so callers should fall back to a non-LLM strategy.

The function supports three URI-like schemes:

1. `openai/<model>` – via `langchain_openai.ChatOpenAI`
2. `hf_hub:<repo_id>` – HuggingFace Inference API (community)
3. `local:<path>` – locally loaded transformers model (CPU / GPU)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Optional
import os
import logging

from ..config.model_costs import MODEL_COSTS
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (so that users without transformers / openai can still run base)
# ---------------------------------------------------------------------------

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError: # pragma: no cover
    get_openai_callback = None


@lru_cache(maxsize=None)
def _tiktoken_encoder(model_name: str):  # noqa: D401
    """Return a *tiktoken* encoder for *model_name* or `None` if not available."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover
            return None


# ---------------------------------------------------------------------------
# Cost-logging wrapper
# ---------------------------------------------------------------------------

class CostLoggingWrapper(Runnable):
    """Wrapper to log token usage and cost for an LLM."""

    def __init__(self, llm: Any, encoder: Any, model_name: str):
        self.llm = llm
        self.encoder = encoder
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def invoke(self, prompt: Any, config: Optional[RunnableConfig] = None) -> AIMessage:
        """Invoke the LLM, logging token usage and cost."""
        # Note: this is a simplified cost-logger. For production, you will
        # likely want to use a more robust callback-based solution.
        
        backend_name = self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name

        if self.model_name.startswith("openai/") and get_openai_callback is not None:
            with get_openai_callback() as cb:
                result = self.llm.invoke(prompt, config)
                self.logger.info(
                    "LLM call to %s completed.\n  *  Tokens: %d prompt, %d completion (%d total)\n  *  Cost:   $%.4f",
                    backend_name,
                    cb.prompt_tokens,
                    cb.completion_tokens,
                    cb.total_tokens,
                    cb.total_cost,
                )
        else:
            # Fallback for non-OpenAI models
            prompt_text = str(prompt)
            prompt_tokens = count_tokens(self.encoder, prompt_text)
            
            result = self.llm.invoke(prompt, config)
            
            content = result.content if hasattr(result, 'content') else str(result)
            completion_tokens = count_tokens(self.encoder, content)
            
            total_tokens = prompt_tokens + completion_tokens
            
            cost_info = MODEL_COSTS.get(self.model_name, {})
            if not cost_info:
                 cost_info = MODEL_COSTS.get(f"openai/{backend_name}", {})


            input_cost_per_1k = cost_info.get("input_cost_per_1k", 0)
            output_cost_per_1k = cost_info.get("output_cost_per_1k", 0)
            
            total_cost = ((prompt_tokens / 1000) * input_cost_per_1k) + ((completion_tokens / 1000) * output_cost_per_1k)

            self.logger.info(
                "LLM call to %s completed.\n  *  Tokens: %d prompt, %d completion (%d total)\n  *  Cost:   $%.4f (estimated)",
                backend_name,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_cost,
            )
            
        return result


@lru_cache(maxsize=None)
def get_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:  # noqa: D401
    """Return *(encoder, llm)* for *model_name*.

    • If no suitable backend/credentials, returns (None, None).
    • Backends are cached so multiple calls with the same name are cheap.
    """
    raw_llm: Optional[Any] = None
    enc: Optional[Any] = None

    # OpenAI -----------------------------------------------------------------
    if model_name.startswith("openai/"):
        backend_name = model_name.split("/", 1)[1]
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY missing – cannot load OpenAI backend.")
            return None, None
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI  # type: ignore
        raw_llm = ChatOpenAI(api_key=api_key, model=backend_name, temperature=0)  # type: ignore[call-arg]
        enc = _tiktoken_encoder(backend_name)

    # HuggingFace Hosted model ----------------------------------------------
    elif model_name.startswith("hf_hub:"):
        repo_id = model_name.split(":", 1)[1]
        try:
            from langchain_community.chat_models import HuggingFaceHub  # type: ignore
        except ImportError as exc:  # pragma: no cover
            logger.warning("HuggingFaceHub import failed: %s", exc)
            return None, None
        hf_token = os.getenv("HF_API_TOKEN")
        raw_llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=hf_token, model_kwargs={"temperature": 0})  # type: ignore[call-arg]
        enc = None  # transformers tokeniser not used for counting here

    # Local transformers model ----------------------------------------------
    elif model_name.startswith("local:"):
        model_path = model_name.split(":", 1)[1]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
            from langchain_community.chat_models import HuggingFacePipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            logger.warning("Transformers pipeline unavailable: %s", exc)
            return None, None
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        hf_pipe = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=1024)
        raw_llm = HuggingFacePipeline(pipeline=hf_pipe)  # type: ignore[call-arg]
        enc = tok
    
    else:
        logger.warning("Unknown model_name scheme %s", model_name)
        return None, None

    if raw_llm is None:
        return None, None

    llm = CostLoggingWrapper(llm=raw_llm, encoder=enc, model_name=model_name)
    return enc, llm


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def count_tokens(encoder: Optional[Any], text: str) -> int:  # noqa: D401
    """Return token count using *encoder* if available, else fallback to words."""
    if encoder is None:
        return len(text.split())
    if hasattr(encoder, "encode"):
        return len(encoder.encode(text))  # type: ignore[attr-defined]
    return len(text.split())
