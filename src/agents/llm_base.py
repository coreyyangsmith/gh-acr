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
from contextlib import nullcontext

# Optional Phoenix / OpenTelemetry imports (best-effort)
try:  # pragma: no cover
    from phoenix.otel import register as phoenix_register  # type: ignore
    from openinference.instrumentation.openai import OpenAIInstrumentor  # type: ignore
    from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore
    from opentelemetry import trace as trace_api  # type: ignore
    from opentelemetry.trace import Status, StatusCode  # type: ignore
    try:
        from openinference.semconv.trace import SpanAttributes as OIAttrs  # type: ignore
    except Exception:  # pragma: no cover
        class OIAttrs:  # type: ignore
            INPUT_VALUE = "input.value"
            OUTPUT_VALUE = "output.value"
            LLM_MODEL_NAME = "llm.model_name"
except Exception:  # pragma: no cover
    phoenix_register = None  # type: ignore
    OpenAIInstrumentor = None  # type: ignore
    LangChainInstrumentor = None  # type: ignore
    trace_api = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    class OIAttrs:  # type: ignore
        INPUT_VALUE = "input.value"
        OUTPUT_VALUE = "output.value"
        LLM_MODEL_NAME = "llm.model_name"

from ..config.model_costs import MODEL_COSTS
from ..config.rate_limits import get_limits_for_model, BACKOFF_SETTINGS
from ..utils.rate_limiter import LimiterRegistry
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler

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

## Tracer is now configured in src/startup.py; here we only enrich current spans.

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


class RateLimitAndCostHandler(BaseCallbackHandler):
    """LangChain callback that enforces rate limits and logs cost/tokens.

    This preserves ChatOpenAI (or other LC chat model) as the terminal node so
    LangChain/OpenInference instrumentation can trace the chain.
    """

    def __init__(self, *, encoder: Optional[Any], model_name: str):
        self.encoder = encoder
        self.model_name = model_name
        limits = get_limits_for_model(model_name)
        self.expected_output_ratio: float = float(limits.get("expected_output_ratio", 0.25))
        rpm = int(limits.get("requests_per_minute", 60))
        tpm = int(limits.get("tokens_per_minute", 150000))
        self._limiter = LimiterRegistry.get(
            key=f"{model_name}", rpm=rpm, tpm=tpm, backoff=BACKOFF_SETTINGS
        )
        self._reservations: dict[Any, dict[str, int]] = {}

    def _backend_name(self) -> str:
        return self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:  # type: ignore[override]
        prompt_text = "\n\n".join(prompts or [])
        prompt_tokens = count_tokens(self.encoder, prompt_text)
        model_cfg = MODEL_COSTS.get(self.model_name, {}) or MODEL_COSTS.get(f"openai/{self._backend_name()}", {})

        # Cap prompt tokens against configured input_limit if known, by truncation
        input_limit = int(model_cfg.get("input_limit", 0))
        if input_limit and prompt_tokens > input_limit:
            if hasattr(self.encoder, "encode") and hasattr(self.encoder, "decode"):
                encoded = self.encoder.encode(prompt_text)
                prompt_text = self.encoder.decode(encoded[: max(0, input_limit - 1)])
                prompt_tokens = count_tokens(self.encoder, prompt_text)
            else:
                words = prompt_text.split()
                prompt_text = " ".join(words[: max(1, input_limit - 1)])
                prompt_tokens = len(prompt_text.split())

        output_limit = int(model_cfg.get("output_limit", 0))
        expected_output_tokens = int(self.expected_output_ratio * output_limit) if output_limit else int(0.25 * prompt_tokens)
        expected_total_tokens = prompt_tokens + expected_output_tokens
        self._limiter.acquire(expected_tokens=int(expected_total_tokens))
        self._reservations[run_id] = {
            "prompt_tokens": int(prompt_tokens),
            "reserved": int(expected_total_tokens),
        }

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:  # type: ignore[override]
        info = self._reservations.pop(run_id, None)
        if info is None:
            return
        prompt_tokens = int(info.get("prompt_tokens", 0))

        # Extract output text from generations
        try:
            texts: list[str] = []
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    if hasattr(gen, "text") and gen.text:
                        texts.append(str(gen.text))
                    elif hasattr(gen, "message") and getattr(gen.message, "content", None):
                        content = gen.message.content
                        texts.append(content if isinstance(content, str) else str(content))
            output_text = "\n".join(texts)
        except Exception:
            output_text = ""

        completion_tokens = count_tokens(self.encoder, output_text)
        total_tokens = prompt_tokens + completion_tokens
        cost_info = MODEL_COSTS.get(self.model_name, {}) or MODEL_COSTS.get(f"openai/{self._backend_name()}", {})
        input_cost_per_1k = float(cost_info.get("input_cost_per_1k", 0))
        output_cost_per_1k = float(cost_info.get("output_cost_per_1k", 0))
        total_cost = ((prompt_tokens / 1000.0) * input_cost_per_1k) + ((completion_tokens / 1000.0) * output_cost_per_1k)

        # Adjust limiter using actuals
        try:
            self._limiter.adjust(actual_tokens=int(total_tokens), reserved_tokens=int(info.get("reserved", 0)))
        except Exception:
            pass

        logger.info(
            "LLM call to %s completed.\n  *  Tokens: %d prompt, %d completion (%d total)\n  *  Cost:   $%.4f%s",
            self._backend_name(),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            total_cost,
            " (estimated)" if not (input_cost_per_1k or output_cost_per_1k) else "",
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:  # type: ignore[override]
        info = self._reservations.pop(run_id, None)
        if info is None:
            return
        try:
            self._limiter.adjust(actual_tokens=0, reserved_tokens=int(info.get("reserved", 0)))
        except Exception:
            pass


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

    # Attach a LangChain callback so that rate limiting and cost logging happen
    # while preserving LangChain/OpenInference spans for the terminal LLM call.
    handler = RateLimitAndCostHandler(encoder=enc, model_name=model_name)
    try:
        # Newer LC runnables support .with_config and callbacks on invoke
        raw_llm = raw_llm.with_config({"callbacks": [handler]})  # type: ignore[attr-defined]
    except Exception:
        # Fallback: rely on per-call callbacks by callers; we still return handler via config
        pass
    return enc, raw_llm


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
