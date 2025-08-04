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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (so that users without transformers / openai can still run base)
# ---------------------------------------------------------------------------

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore


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


@lru_cache(maxsize=None)
def get_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:  # noqa: D401
    """Return *(encoder, llm)* for *model_name*.

    • If no suitable backend/credentials, returns (None, None).
    • Backends are cached so multiple calls with the same name are cheap.
    """

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
        llm = ChatOpenAI(api_key=api_key, model=backend_name, temperature=0)  # type: ignore[call-arg]
        enc = _tiktoken_encoder(backend_name)
        return enc, llm

    # HuggingFace Hosted model ----------------------------------------------
    if model_name.startswith("hf_hub:"):
        repo_id = model_name.split(":", 1)[1]
        try:
            from langchain_community.chat_models import HuggingFaceHub  # type: ignore
        except ImportError as exc:  # pragma: no cover
            logger.warning("HuggingFaceHub import failed: %s", exc)
            return None, None
        hf_token = os.getenv("HF_API_TOKEN")
        llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=hf_token, model_kwargs={"temperature": 0})  # type: ignore[call-arg]
        enc = None  # transformers tokeniser not used for counting here
        return enc, llm

    # Local transformers model ----------------------------------------------
    if model_name.startswith("local:"):
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
        llm = HuggingFacePipeline(pipeline=hf_pipe)  # type: ignore[call-arg]
        enc = tok
        return enc, llm

    logger.warning("Unknown model_name scheme %s", model_name)
    return None, None


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
