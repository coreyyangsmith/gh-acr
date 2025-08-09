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
from ..config.rate_limits import get_limits_for_model, BACKOFF_SETTINGS
from ..utils.rate_limiter import LimiterRegistry
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
        limits = get_limits_for_model(model_name)
        self.expected_output_ratio: float = float(limits.get("expected_output_ratio", 0.25))
        rpm = int(limits.get("requests_per_minute", 60))
        tpm = int(limits.get("tokens_per_minute", 150000))
        self._limiter = LimiterRegistry.get(
            key=f"{model_name}",
            rpm=rpm,
            tpm=tpm,
            backoff=BACKOFF_SETTINGS,
        )

    def _is_transient_error(self, exc: Exception) -> bool:
        text = str(getattr(exc, "message", "")) or str(exc)
        code = getattr(exc, "status_code", None)
        retry_after = getattr(exc, "retry_after", None)
        if code in (429, 500, 502, 503, 504):
            return True
        lowered = text.lower()
        if any(s in lowered for s in [
            "rate limit",
            "too many requests",
            "overloaded",
            "timeout",
            "temporarily unavailable",
            "try again",
        ]):
            return True
        return False

    def invoke(self, prompt: Any, config: Optional[RunnableConfig] = None) -> AIMessage:
        """Invoke the LLM, logging token usage and cost."""
        # Note: this is a simplified cost-logger. For production, you will
        # likely want to use a more robust callback-based solution.
        
        backend_name = self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name

        # Pre-flight estimated token usage and throttle before calling the backend
        prompt_text = str(prompt)
        prompt_tokens = count_tokens(self.encoder, prompt_text)

        # Cap prompt tokens against configured input_limit if known, by truncation
        model_cfg = MODEL_COSTS.get(self.model_name, {}) or MODEL_COSTS.get(f"openai/{backend_name}", {})
        input_limit = int(model_cfg.get("input_limit", 0))
        if input_limit and prompt_tokens > input_limit:
            # naive truncation to stay under limit
            if hasattr(self.encoder, "encode") and hasattr(self.encoder, "decode"):
                encoded = self.encoder.encode(prompt_text)
                prompt_text = self.encoder.decode(encoded[: max(0, input_limit - 1)])
                prompt_tokens = count_tokens(self.encoder, prompt_text)
            else:
                # fallback word-based trimming
                words = prompt_text.split()
                prompt_text = " ".join(words[: max(1, input_limit - 1)])
                prompt_tokens = len(prompt_text.split())

        # Reserve expected total tokens (prompt + expected output)
        output_limit = int(model_cfg.get("output_limit", 0))
        expected_output_tokens = int(self.expected_output_ratio * output_limit) if output_limit else int(0.25 * prompt_tokens)
        expected_total_tokens = prompt_tokens + expected_output_tokens

        attempts = int(BACKOFF_SETTINGS.get("max_retries", 5))
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            # Acquire reservation for this attempt
            self._limiter.acquire(expected_tokens=expected_total_tokens)
            try:
                if self.model_name.startswith("openai/") and get_openai_callback is not None:
                    with get_openai_callback() as cb:
                        # Rebuild prompt if we truncated
                        result = self.llm.invoke(prompt_text, config)
                        self.logger.info(
                            "LLM call to %s completed.\n  *  Tokens: %d prompt, %d completion (%d total)\n  *  Cost:   $%.4f",
                            backend_name,
                            cb.prompt_tokens,
                            cb.completion_tokens,
                            cb.total_tokens,
                            cb.total_cost,
                        )
                        # Adjust token bucket using actuals
                        try:
                            self._limiter.adjust(
                                actual_tokens=int(cb.total_tokens),
                                reserved_tokens=int(expected_total_tokens),
                            )
                        except Exception:
                            pass
                        return result
                else:
                    # Fallback for non-OpenAI models
                    result = self.llm.invoke(prompt_text, config)
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
                    try:
                        self._limiter.adjust(
                            actual_tokens=int(total_tokens),
                            reserved_tokens=int(expected_total_tokens),
                        )
                    except Exception:
                        pass
                    return result
            except Exception as exc:  # pragma: no cover - network/env specific
                last_exc = exc
                try:
                    # Return reserved tokens for this failed attempt
                    self._limiter.adjust(actual_tokens=0, reserved_tokens=int(expected_total_tokens))
                except Exception:
                    pass
                self._limiter.last_error = str(exc)
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                # Jittered exponential backoff
                self._limiter.backoff_sleep(attempt)

        # Should not reach here; raise last exception
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM invocation failed without exception; unexpected state")


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
        # Respect per-request output token limits using max_tokens (Chat Completions)
        model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(f"openai/{backend_name}", {})
        max_out = int(model_cfg.get("output_limit", 0))
        if max_out > 0:
            raw_llm = ChatOpenAI(api_key=api_key, model=backend_name, temperature=0, max_tokens=max_out)  # type: ignore[call-arg]
        else:
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
        # Try to bound output length if possible
        model_cfg = MODEL_COSTS.get(model_name, {})
        max_out = int(model_cfg.get("output_limit", 0)) or 1024
        raw_llm = HuggingFaceHub(
            repo_id=repo_id,
            huggingfacehub_api_token=hf_token,
            model_kwargs={"temperature": 0, "max_new_tokens": max_out},
        )  # type: ignore[call-arg]
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
