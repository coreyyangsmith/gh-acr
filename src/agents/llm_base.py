"""Centralised LLM backend registry.

`get_backend(model_name)` returns a `(encoder, llm)` tuple where

* **encoder** – a tiktoken Encoding (or Hugging-Face tokenizer) implementing
  `.encode(text)` so we can count tokens; may be `None` if unavailable.
* **llm** – a LangChain **chat model** (or pipeline) that can be piped into
  prompts.  `None` indicates that the requested backend is not usable (e.g.
  missing credentials), so callers should fall back to a non-LLM strategy.

The function supports these URI-like schemes:

1. `openai/<model>` – via `langchain_openai.ChatOpenAI`
2. `hf_hub:<repo_id>` – HuggingFace Inference API (community)
3. `local:<path>` – locally loaded transformers model (CPU / GPU)
4. `groq:<model>` – via Groq API using `langchain_groq.ChatGroq`
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Optional
import os
import logging
from contextlib import nullcontext

# Optional Langfuse callback integration (best-effort)
try:  # pragma: no cover
    from langfuse.langchain import CallbackHandler as LangfuseCallback  # type: ignore
except Exception:  # pragma: no cover
    LangfuseCallback = None  # type: ignore

from ..config.model_costs import MODEL_COSTS
from ..config.rate_limits import get_limits_for_model, BACKOFF_SETTINGS
from ..utils.rate_limiter import LimiterRegistry
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults and helpers for local Transformers backends (tiny test model)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

DEFAULT_LOCAL_MODEL_ID = (
    os.getenv("HF_MODEL_ID") or os.getenv("MODEL_ID") or "gpt2"
)
HF_LOCAL_ONLY = os.getenv("HF_LOCAL_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")
LOCAL_SEED = int(os.getenv("SEED", "42"))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_HF_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, "data", "models")
HF_TRUST_REMOTE_CODE = os.getenv("HF_TRUST_REMOTE_CODE", "0").strip().lower() in ("1", "true", "yes", "on")
HF_REVISION = os.getenv("HF_REVISION", "").strip() or None


def _get_hf_token() -> Optional[str]:
    for var in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.getenv(var)
        if tok and tok.strip():
            return tok.strip()
    return None

def _should_use_accelerate() -> bool:
    # Respect HF_DEVICE_MAP if the user explicitly asked
    v = os.getenv("HF_DEVICE_MAP", "").strip().lower()
    if v in {"auto", "cpu"} or v.isdigit() or v.startswith("cuda"):
        return True
    # Otherwise: single-GPU H100 can hold 8B comfortably -> no need to shard
    return False

def _weights_exist_locally(model_id: str) -> bool:
    try:
        # Very cheap heuristic: look for any *.safetensors under the cache subdir
        root = os.path.join(DEFAULT_HF_CACHE_DIR, model_id.replace("/", os.sep))
        for base, _, files in os.walk(root):
            if any(f.endswith(".safetensors") for f in files):
                return True
    except Exception:
        pass
    return False


def _pick_torch_dtype() -> Any:
    # prefer env override first
    dt = _parse_torch_dtype_env()
    if dt == "auto":
        dt = None
    if dt is not None:
        return dt
    # default: bf16 on CUDA, else fp32
    try:
        if torch is not None and torch.cuda.is_available():
            # Hopper/Ampere -> bf16 is ideal
            return torch.bfloat16
    except Exception:
        pass
    return torch.float32


def _parse_torch_dtype_env() -> Optional[Any]:  # type: ignore[override]
    """Return torch dtype from HF_TORCH_DTYPE env or 'auto' string, else None.

    HF_TORCH_DTYPE can be one of: auto, float16, bfloat16, float32
    """
    val = os.getenv("HF_TORCH_DTYPE", "").strip().lower()
    if not val:
        return None
    if val == "auto":
        return "auto"
    try:
        if torch is not None:
            if val in ("fp16", "float16", "half"):
                return torch.float16
            if val in ("bf16", "bfloat16"):
                return torch.bfloat16
            if val in ("fp32", "float32"):
                return torch.float32
    except Exception:
        pass
    return None


def _get_device_map_from_env() -> Any:  # type: ignore[override]
    """Derive device_map from HF_DEVICE_MAP env.

    Examples:
    - not set → "auto"
    - "auto" → "auto"
    - "cpu" → {"": "cpu"}
    - "0" or "cuda:0" → {"": 0}
    - any other → "auto"
    """
    v = os.getenv("HF_DEVICE_MAP", "").strip().lower()
    if not v:
        # Default to automatic device placement/sharding across available GPUs
        return "auto"
    if v == "auto":
        # Respect explicit request for auto-sharding; do NOT force cuda:0
        return "auto"
    if v == "cpu":
        return {"": "cpu"}
    if v.isdigit():
        return {"": int(v)}
    if v.startswith("cuda"):
        idx = int(v.split(":", 1)[1]) if ":" in v else 0
        return {"": idx}
    return "auto"


def _collect_model_devices(model: Any) -> list[str]:  # type: ignore[override]
    """Best-effort summary of device placements for the given model."""
    devices: set[str] = set()
    # From Accelerate/hf_device_map
    try:
        dm = getattr(model, "hf_device_map", None)
        if isinstance(dm, dict):
            for v in dm.values():
                if isinstance(v, int):
                    devices.add(f"cuda:{v}")
                else:
                    devices.add(str(v))
    except Exception:
        pass
    # Fallback: single .device attribute
    try:
        dev = getattr(model, "device", None)
        if dev is not None:
            devices.add(str(dev))
    except Exception:
        pass
    # If still unknown, infer from CUDA availability
    if not devices:
        try:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                devices.add("cuda:0")
            else:
                devices.add("cpu")
        except Exception:
            devices.add("cpu")
    return sorted(devices)


def _hf_device_index() -> int:
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return 0
    except Exception:  # pragma: no cover
        pass
    return -1


def _format_bytes(num: int) -> str:
    try:
        size = float(num)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
            if size < 1024.0:
                return f"{size:.2f}{unit}"
            size /= 1024.0
        return f"{size:.2f}EiB"
    except Exception:
        return str(num)


def _log_gpu_overview(context: str, model: Any | None = None) -> None:
    """Log a concise overview of CUDA devices and memory, plus model device map."""
    try:
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "") or "<unset>"
        alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "") or "<unset>"
        hf_dev_map_env = os.getenv("HF_DEVICE_MAP", "") or "<unset>"

        if torch is None:
            logger.info("%s CUDA overview: torch not available. CUDA_VISIBLE_DEVICES=%s", context, cuda_visible)
            return
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            logger.info("%s CUDA overview: CUDA not available. CUDA_VISIBLE_DEVICES=%s", context, cuda_visible)
            return

        num = int(torch.cuda.device_count())
        cuda_ver = getattr(torch.version, "cuda", None)
        header = (
            f"{context} CUDA overview: torch={getattr(torch, '__version__', '?')}, "
            f"cuda={cuda_ver}, devices={num}, CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"HF_DEVICE_MAP={hf_dev_map_env}, PYTORCH_CUDA_ALLOC_CONF={alloc_conf}"
        )
        lines: list[str] = [header]

        # If Accelerate placed modules, show a sample of the device map
        try:
            dm = getattr(model, "hf_device_map", None)
            if isinstance(dm, dict) and dm:
                sample = {k: str(v) for k, v in list(dm.items())[:8]}
                extra = "" if len(dm) <= 8 else f" ... +{len(dm) - 8} more"
                lines.append(f"{context} hf_device_map: {sample}{extra}")
        except Exception:
            pass

        for idx in range(num):
            name = f"cuda:{idx}"
            total_prop = 0
            try:
                props = torch.cuda.get_device_properties(idx)
                name = props.name
                total_prop = int(getattr(props, "total_memory", 0))
            except Exception:
                pass
            free_b = None
            total_b = None
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
            except Exception:
                pass
            reserved_b = None
            allocated_b = None
            try:
                reserved_b = torch.cuda.memory_reserved(idx)
                allocated_b = torch.cuda.memory_allocated(idx)
            except Exception:
                pass

            parts = [f"cuda:{idx} {name}"]
            parts.append(f"total={_format_bytes((total_b or total_prop) or 0)}")
            if free_b is not None:
                parts.append(f"free={_format_bytes(int(free_b))}")
            if reserved_b is not None:
                parts.append(f"reserved={_format_bytes(int(reserved_b))}")
            if allocated_b is not None:
                parts.append(f"allocated={_format_bytes(int(allocated_b))}")
            lines.append(f"{context} " + ", ".join(parts))

        logger.info("\n".join(lines))
    except Exception as e:  # pragma: no cover
        try:
            logger.info("%s CUDA overview logging failed: %s", context, e)
        except Exception:
            pass


def build_local_text_generator(model_id: str | None = None, *, local_only: bool | None = None):
    """Return (hf_pipeline, tokenizer) for a small local model suitable for tests."""
    if model_id is None:
        model_id = DEFAULT_LOCAL_MODEL_ID
    if local_only is None:
        local_only = HF_LOCAL_ONLY
    # Lazy import to avoid hard dependency when not needed
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed  # type: ignore

    set_seed(LOCAL_SEED)
    # Ensure cache dir exists and try offline-first load from cache, then fallback to download
    os.makedirs(DEFAULT_HF_CACHE_DIR, exist_ok=True)
    logger.info(
        "[local] Initializing HF generator: model_id=%s, local_only=%s, cache_dir=%s",
        model_id,
        bool(local_only),
        DEFAULT_HF_CACHE_DIR,
    )
    hf_token = _get_hf_token()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            use_fast=False,
            cache_dir=DEFAULT_HF_CACHE_DIR,
            token=hf_token,  # new arg
            revision=HF_REVISION,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
        )
        use_accel = _should_use_accelerate()
        dtype = _pick_torch_dtype()
        local_ok = _weights_exist_locally(model_id)

        if use_accel:
            # Use Accelerate only if explicitly requested
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                device_map=_get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        else:
            # Use device_map placement and avoid manual .to(...)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                local_files_only=bool(local_ok),
                device_map=_get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        logger.info("[local] Model devices: %s", ", ".join(_collect_model_devices(model)))
        logger.info("[local] Loaded model+tokenizer from cache (local_files_only=True): %s", model_id)
    except Exception:
        if bool(local_only):
            logger.error("[local] Cache-only load failed and HF_LOCAL_ONLY is set; re-raising for %s", model_id)
            raise
        logger.info("[local] Cache miss; downloading model to cache: %s -> %s", model_id, DEFAULT_HF_CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            cache_dir=DEFAULT_HF_CACHE_DIR,
            token=hf_token,
            revision=HF_REVISION,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
        )
        use_accel = _should_use_accelerate()
        dtype = _pick_torch_dtype()
        local_ok = _weights_exist_locally(model_id)

        if use_accel:
            # Use Accelerate only if explicitly requested
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                device_map=_get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        else:
            # Use device_map placement and avoid manual .to(...)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                local_files_only=bool(local_ok),
                device_map=_get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        logger.info("[local] Model devices (download): %s", ", ".join(_collect_model_devices(model)))
        logger.info("[local] Downloaded model+tokenizer and cached: %s", model_id)
    # --- Safety against position-id overflow ---
    npos = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024)))
    reserve_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
    reserve_new = max(1, min(reserve_new, npos - 32))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.config.pad_token_id = pad_id

    _log_gpu_overview("[local.build_local_text_generator]", model)

    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        truncation=True,
        max_new_tokens=reserve_new,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    logger.info(
        "[local] Created text-generation pipeline for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, truncation=%s)",
        model_id,
        npos,
        tokenizer.model_max_length,
        reserve_new,
        True,
    )
    return generator, tokenizer


def generate_local_text(prompt: str, *, max_new_tokens: int = 64, model_id: str | None = None) -> str:
    """Quick local generation helper using the tiny test model by default."""
    generator, tokenizer = build_local_text_generator(model_id)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    outputs = generator(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
        num_return_sequences=1,
    )
    generated = outputs[0]["generated_text"]
    return generated[len(prompt):].lstrip() if generated.startswith(prompt) else generated


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

        input_limit = int(model_cfg.get("input_limit", 0))
        output_limit = int(model_cfg.get("output_limit", 0))
        sliding_window = bool(model_cfg.get("sliding_window", False))
        total_limit = int(model_cfg.get("total_limit", 0))

        # Base expected output size
        expected_output_tokens = int(self.expected_output_ratio * output_limit) if output_limit else int(0.25 * prompt_tokens)
        if output_limit:
            expected_output_tokens = min(expected_output_tokens, output_limit)

        if sliding_window and total_limit:
            # Enforce prompt + expected_output <= total_limit
            allowed_prompt = max(1, total_limit - expected_output_tokens)
            if prompt_tokens > allowed_prompt:
                if hasattr(self.encoder, "encode") and hasattr(self.encoder, "decode"):
                    encoded = self.encoder.encode(prompt_text)
                    prompt_text = self.encoder.decode(encoded[: allowed_prompt])
                    prompt_tokens = count_tokens(self.encoder, prompt_text)
                else:
                    words = prompt_text.split()
                    prompt_text = " ".join(words[: allowed_prompt])
                    prompt_tokens = len(prompt_text.split())
            expected_total_tokens = min(total_limit, prompt_tokens + expected_output_tokens)
        else:
            # Separate caps: bound prompt by input_limit if provided
            if input_limit and prompt_tokens > input_limit:
                if hasattr(self.encoder, "encode") and hasattr(self.encoder, "decode"):
                    encoded = self.encoder.encode(prompt_text)
                    prompt_text = self.encoder.decode(encoded[: max(0, input_limit - 1)])
                    prompt_tokens = count_tokens(self.encoder, prompt_text)
                else:
                    words = prompt_text.split()
                    prompt_text = " ".join(words[: max(1, input_limit - 1)])
                    prompt_tokens = len(prompt_text.split())
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

    Raises a RuntimeError/ValueError if the requested backend cannot be
    initialized. Backends are cached so multiple calls with the same name
    are cheap.
    """
    raw_llm: Optional[Any] = None
    enc: Optional[Any] = None

    # OpenAI -----------------------------------------------------------------
    if model_name.startswith("openai/"):
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
        # Respect per-request output token limits using max_tokens (Chat Completions)
        model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(f"openai/{backend_name}", {})
        max_out = int(model_cfg.get("output_limit", 0))
        # Some GPT-5 variants do not accept a temperature parameter; omit when backend_name startswith "gpt-5"
        is_gpt5 = backend_name.startswith("gpt-5")
        if max_out > 0:
            if is_gpt5:
                raw_llm = ChatOpenAI(api_key=api_key, model=backend_name, max_tokens=max_out)  # type: ignore[call-arg]
            else:
                raw_llm = ChatOpenAI(api_key=api_key, model=backend_name, temperature=0, max_tokens=max_out)  # type: ignore[call-arg]
        else:
            if is_gpt5:
                raw_llm = ChatOpenAI(api_key=api_key, model=backend_name)  # type: ignore[call-arg]
            else:
                raw_llm = ChatOpenAI(api_key=api_key, model=backend_name, temperature=0)  # type: ignore[call-arg]
        enc = _tiktoken_encoder(backend_name)
    # Groq API ---------------------------------------------------------------
    elif model_name.startswith("groq:"):
        logger.info("[groq] Using Groq backend: model=%s", model_name)
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
        if max_out > 0:
            raw_llm = ChatGroq(groq_api_key=api_key, model=backend_name, temperature=0, max_tokens=max_out)  # type: ignore[call-arg]
        else:
            raw_llm = ChatGroq(groq_api_key=api_key, model=backend_name, temperature=0)  # type: ignore[call-arg]
        enc = None  # use word-count fallback for token estimates unless configured
    # Local transformers model ----------------------------------------------
    elif model_name.startswith("local:"):
        logger.info("[local] Using local Transformers backend: model=%s", model_name)
        model_path = model_name.split(":", 1)[1]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
            from langchain_community.llms import HuggingFacePipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            logger.error("Transformers pipeline unavailable: %s", exc)
            raise RuntimeError(f"Transformers pipeline unavailable: {exc}")
        requested = (model_path or "").strip()
        if not requested:
            msg = "local: backend requires a model id or path, e.g. local:gpt2"
            logger.error(msg)
            raise ValueError(msg)
        # Load user-specified local path or HF repo id (offline-first)
        os.makedirs(DEFAULT_HF_CACHE_DIR, exist_ok=True)
        logger.info(
            "[local] Preparing local backend: model=%s, cache_dir=%s, HF_LOCAL_ONLY=%s",
            requested,
            DEFAULT_HF_CACHE_DIR,
            HF_LOCAL_ONLY,
        )
        hf_token = _get_hf_token()
        try:
            # Prefer explicit GPT-2 tokenizer for distilgpt2
            tok = AutoTokenizer.from_pretrained(
                requested,
                local_files_only=True,
                use_fast=False,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=HF_TRUST_REMOTE_CODE,
            )
            logger.info("[local] Loaded tokenizer from cache for %s via AutoTokenizer", requested)
            use_accel = _should_use_accelerate()
            dtype = _pick_torch_dtype()
            local_ok = _weights_exist_locally(requested)

            if use_accel:
                # Use Accelerate only if explicitly requested
                model = AutoModelForCausalLM.from_pretrained(
                    requested,
                    cache_dir=DEFAULT_HF_CACHE_DIR,
                    token=hf_token,
                    revision=HF_REVISION,
                    trust_remote_code=True,
                    device_map=_get_device_map_from_env(),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            else:
                # Use device_map placement and avoid manual .to(...)
                model = AutoModelForCausalLM.from_pretrained(
                    requested,
                    cache_dir=DEFAULT_HF_CACHE_DIR,
                    token=hf_token,
                    revision=HF_REVISION,
                    trust_remote_code=True,
                    local_files_only=bool(local_ok),  # only lock to local if weights exist
                    device_map=_get_device_map_from_env(),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            logger.info("[local] Model devices (cached): %s", ", ".join(_collect_model_devices(model)))
            logger.info("[local] Loaded model from cache for %s (device_map=auto)", requested)
        except Exception:
            if HF_LOCAL_ONLY:
                logger.error("[local] Cache-only load failed for %s and HF_LOCAL_ONLY=1; re-raising", requested)
                raise
            if "distilgpt2" in requested:
                try:
                    from transformers import GPT2Tokenizer  # type: ignore
                    tok = GPT2Tokenizer.from_pretrained(
                        requested,
                        cache_dir=DEFAULT_HF_CACHE_DIR,
                        token=hf_token,
                        revision=HF_REVISION,
                        trust_remote_code=HF_TRUST_REMOTE_CODE,
                    )
                    logger.info("[local] Downloaded GPT2Tokenizer for %s to %s", requested, DEFAULT_HF_CACHE_DIR)
                except Exception:
                    tok = AutoTokenizer.from_pretrained(
                        requested,
                        use_fast=False,
                        cache_dir=DEFAULT_HF_CACHE_DIR,
                        token=hf_token,
                        revision=HF_REVISION,
                        trust_remote_code=HF_TRUST_REMOTE_CODE,
                    )
                    logger.info("[local] Downloaded AutoTokenizer for %s to %s", requested, DEFAULT_HF_CACHE_DIR)
            else:
                tok = AutoTokenizer.from_pretrained(
                    requested,
                    use_fast=False,
                    cache_dir=DEFAULT_HF_CACHE_DIR,
                    token=hf_token,
                    revision=HF_REVISION,
                    trust_remote_code=HF_TRUST_REMOTE_CODE,
                )
                logger.info("[local] Downloaded AutoTokenizer for %s to %s", requested, DEFAULT_HF_CACHE_DIR)
            use_accel = _should_use_accelerate()
            dtype = _pick_torch_dtype()
            local_ok = _weights_exist_locally(requested)

            if use_accel:
                # Use Accelerate only if explicitly requested
                model = AutoModelForCausalLM.from_pretrained(
                    requested,
                    cache_dir=DEFAULT_HF_CACHE_DIR,
                    token=hf_token,
                    revision=HF_REVISION,
                    trust_remote_code=True,
                    device_map=_get_device_map_from_env(),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            else:
                # Use device_map placement and avoid manual .to(...)
                model = AutoModelForCausalLM.from_pretrained(
                    requested,
                    cache_dir=DEFAULT_HF_CACHE_DIR,
                    token=hf_token,
                    revision=HF_REVISION,
                    trust_remote_code=True,
                    local_files_only=bool(local_ok),  # only lock to local if weights exist
                    device_map=_get_device_map_from_env(),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            logger.info("[local] Model devices (downloaded): %s", ", ".join(_collect_model_devices(model)))
            logger.info("[local] Downloaded model %s to %s (device_map=auto)", requested, DEFAULT_HF_CACHE_DIR)
        # --- Safety against position-id overflow (same as tiny helper) ---
        npos = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024)))
        reserve_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
        reserve_new = max(1, min(reserve_new, npos - 32))

        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model.config.pad_token_id = tok.pad_token_id

        tok.truncation_side = os.getenv("LOCAL_TRUNCATION_SIDE", "left")
        tok.model_max_length = npos - reserve_new

        # If this is a Qwen3 model, wrap with chat template and recommended settings
        requested_lower = requested.lower()
        is_qwen3 = "qwen3" in requested_lower or "/qwen3-" in requested_lower
        is_gptoss = ("gpt-oss" in requested_lower) or ("/gpt-oss-" in requested_lower) or ("openai/gpt-oss" in requested_lower)
        if is_qwen3:
            # Create a lightweight wrapper that formats a single-string prompt
            # into Qwen3's chat template and applies recommended decoding params.
            class _FakeGen:
                def __init__(self, text: str):
                    self.text = text

            class _FakeResponse:
                def __init__(self, text: str):
                    self.generations = [[_FakeGen(text)]]

            class Qwen3ChatWrapper:
                def __init__(self, model_obj, tokenizer_obj):
                    self._model = model_obj
                    self._tok = tokenizer_obj
                    self._callbacks: list[Any] = []
                    # Thinking switch (default on per Qwen3 best-practice)
                    self._enable_thinking = os.getenv("QWEN3_ENABLE_THINKING", "1").strip().lower() in ("1", "true", "yes", "on")
                    # Sampling presets
                    if self._enable_thinking:
                        self._temperature = float(os.getenv("QWEN3_TEMPERATURE", "0.6"))
                        self._top_p = float(os.getenv("QWEN3_TOP_P", "0.95"))
                        self._top_k = int(os.getenv("QWEN3_TOP_K", "20"))
                    else:
                        self._temperature = float(os.getenv("QWEN3_TEMPERATURE", "0.7"))
                        self._top_p = float(os.getenv("QWEN3_TOP_P", "0.8"))
                        self._top_k = int(os.getenv("QWEN3_TOP_K", "20"))
                    # Default to a larger output window for Qwen3
                    default_qwen_max = max(reserve_new, 2048)
                    self._max_new = int(os.getenv("QWEN3_MAX_NEW_TOKENS", str(default_qwen_max)))
                    # Optional YaRN scaling via tokenizer config override (Transformers >= 4.51)
                    try:
                        if os.getenv("QWEN3_ENABLE_YARN", "0").strip().lower() in ("1", "true", "yes", "on"):
                            factor = float(os.getenv("QWEN3_YARN_FACTOR", "4.0"))
                            orig = int(os.getenv("QWEN3_YARN_ORIG_CTX", str(getattr(self._model.config, "max_position_embeddings", 32768))))
                            # Some tokenizers support dynamic json override; best-effort here
                            if hasattr(self._model, "config") and hasattr(self._model.config, "rope_scaling"):
                                self._model.config.rope_scaling = {
                                    "rope_type": "yarn",
                                    "factor": float(factor),
                                    "original_max_position_embeddings": int(orig),
                                }
                    except Exception:
                        pass

                def with_config(self, config: dict[str, Any] | None = None):  # type: ignore[override]
                    try:
                        if config and "callbacks" in config:
                            self._callbacks = list(config.get("callbacks") or [])
                    except Exception:
                        pass
                    return self

                def invoke(self, prompt_text: str, config: RunnableConfig | None = None):  # type: ignore[override]
                    try:
                        # Resolve callbacks from config as well
                        callbacks = list(self._callbacks)
                        try:
                            if config and isinstance(config, dict):
                                callbacks.extend(list(config.get("callbacks") or []))  # type: ignore[arg-type]
                        except Exception:
                            pass

                        # Prepare chat-formatted input
                        messages = [{"role": "user", "content": str(prompt_text)}]
                        chat_text = self._tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False, #Hardcode to False for Qwen3, for now
                        )

                        # Notify start
                        run_id = id(self) ^ hash(chat_text)
                        for cb in callbacks:
                            try:
                                cb.on_llm_start({"name": "qwen3-local"}, [chat_text], run_id=run_id)
                            except Exception:
                                pass

                        # Generation
                        pad_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else self._tok.pad_token_id
                        local_tok = self._tok.__class__.from_pretrained(self._tok.name_or_path, use_fast=False, trust_remote_code=True)
                        _log_gpu_overview("[local.Qwen3ChatWrapper.invoke]", self._model)
                        generator = pipeline(
                            "text-generation",
                            model=self._model,
                            tokenizer=local_tok,
                            batch_size=1,
                            truncation=True,
                        )
                        outputs = generator(
                            chat_text,
                            truncation=True,
                            padding=True,
                            do_sample=True,
                            temperature=self._temperature,
                            top_p=self._top_p,
                            top_k=self._top_k,
                            max_new_tokens=self._max_new,
                            eos_token_id=self._tok.eos_token_id,
                            pad_token_id=pad_id,
                            num_return_sequences=1,
                        )
                        full_text = outputs[0].get("generated_text", "")
                        # Strip prompt prefix if present
                        if full_text.startswith(chat_text):
                            answer_text = full_text[len(chat_text):].lstrip()
                        else:
                            answer_text = full_text

                        # Notify end (best-effort for token accounting)
                        fake_resp = _FakeResponse(answer_text)
                        for cb in callbacks:
                            try:
                                cb.on_llm_end(fake_resp, run_id=run_id)
                            except Exception:
                                pass

                        return AIMessage(content=answer_text)
                    except Exception as e:  # pragma: no cover
                        for cb in self._callbacks:
                            try:
                                cb.on_llm_error(e, run_id=id(self))
                            except Exception:
                                pass
                        raise

            raw_llm = Qwen3ChatWrapper(model, tok)
            enc = tok
            logger.info(
                "[local] Initialized Qwen3 chat wrapper for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, thinking=%s)",
                requested,
                npos,
                tok.model_max_length,
                reserve_new,
                True,
            )
        elif is_gptoss:
            # Create a lightweight wrapper that formats input via the model's chat template
            # using the Harmony response format and allows setting reasoning level.
            class _FakeGen:
                def __init__(self, text: str):
                    self.text = text

            class _FakeResponse:
                def __init__(self, text: str):
                    self.generations = [[_FakeGen(text)]]

            class GPTOSSChatWrapper:
                def __init__(self, model_obj, tokenizer_obj):
                    self._model = model_obj
                    self._tok = tokenizer_obj
                    self._callbacks: list[Any] = []
                    # Reasoning level can be low|medium|high (default: medium)
                    self._reasoning_level = os.getenv("GPT_OSS_REASONING_LEVEL", "medium").strip().lower()
                    # Default to a larger output window for chat-style models
                    default_max = max(reserve_new, 2048)
                    self._max_new = int(os.getenv("GPT_OSS_MAX_NEW_TOKENS", str(default_max)))
                    # Sampling presets (tune as needed)
                    self._temperature = float(os.getenv("GPT_OSS_TEMPERATURE", "0.7"))
                    self._top_p = float(os.getenv("GPT_OSS_TOP_P", "0.9"))

                def with_config(self, config: dict[str, Any] | None = None):  # type: ignore[override]
                    try:
                        if config and "callbacks" in config:
                            self._callbacks = list(config.get("callbacks") or [])
                    except Exception:
                        pass
                    return self

                def invoke(self, prompt_text: str, config: RunnableConfig | None = None):  # type: ignore[override]
                    try:
                        callbacks = list(self._callbacks)
                        try:
                            if config and isinstance(config, dict):
                                callbacks.extend(list(config.get("callbacks") or []))  # type: ignore[arg-type]
                        except Exception:
                            pass

                        # Prepare Harmony-formatted chat via tokenizer template
                        sys_msg = {"role": "system", "content": f"Reasoning: {self._reasoning_level}"}
                        messages = [sys_msg, {"role": "user", "content": str(prompt_text)}]
                        chat_text = self._tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                        # Notify start
                        run_id = id(self) ^ hash(chat_text)
                        for cb in callbacks:
                            try:
                                cb.on_llm_start({"name": "gpt-oss-local"}, [chat_text], run_id=run_id)
                            except Exception:
                                pass

                        # Generation
                        pad_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else self._tok.pad_token_id
                        _log_gpu_overview("[local.GPTOSSChatWrapper.invoke]", self._model)
                        generator = pipeline(
                            "text-generation",
                            model=self._model,
                            tokenizer=self._tok,
                            batch_size=1,
                            truncation=True,
                        )
                        outputs = generator(
                            chat_text,
                            truncation=True,
                            padding=True,
                            do_sample=True,
                            temperature=self._temperature,
                            top_p=self._top_p,
                            max_new_tokens=self._max_new,
                            eos_token_id=self._tok.eos_token_id,
                            pad_token_id=pad_id,
                            num_return_sequences=1,
                        )
                        full_text = outputs[0].get("generated_text", "")
                        answer_text = full_text[len(chat_text):].lstrip() if full_text.startswith(chat_text) else full_text

                        # Notify end (best-effort)
                        fake_resp = _FakeResponse(answer_text)
                        for cb in callbacks:
                            try:
                                cb.on_llm_end(fake_resp, run_id=run_id)
                            except Exception:
                                pass

                        return AIMessage(content=answer_text)
                    except Exception as e:  # pragma: no cover
                        for cb in self._callbacks:
                            try:
                                cb.on_llm_error(e, run_id=id(self))
                            except Exception:
                                pass
                        raise

            raw_llm = GPTOSSChatWrapper(model, tok)
            enc = tok
            logger.info(
                "[local] Initialized GPT-OSS chat wrapper for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, reasoning=%s)",
                requested,
                npos,
                tok.model_max_length,
                reserve_new,
                os.getenv("GPT_OSS_REASONING_LEVEL", "medium").strip().lower(),
            )
        else:
            _log_gpu_overview("[local.HuggingFacePipeline]", model)
            hf_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                batch_size=1,
                truncation=True,
                max_new_tokens=reserve_new,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            logger.info(
                "[local] Initialized HuggingFacePipeline for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, truncation=%s)",
                requested,
                npos,
                tok.model_max_length,
                reserve_new,
                True,
            )
            raw_llm = HuggingFacePipeline(pipeline=hf_pipe)  # type: ignore[call-arg]
            enc = tok
    
    else:
        msg = f"Unknown model_name scheme {model_name}"
        logger.error(msg)
        raise ValueError(msg)

    if raw_llm is None:
        msg = f"Failed to initialize LLM backend for model_name={model_name}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Attach a LangChain callback so that rate limiting and cost logging happen
    # while preserving LangChain/OpenInference spans for the terminal LLM call.
    handler = RateLimitAndCostHandler(encoder=enc, model_name=model_name)
    try:
        # Newer LC runnables support .with_config and callbacks on invoke
        raw_llm = raw_llm.with_config({"callbacks": [handler]})  # type: ignore[attr-defined]
    except Exception:
        # Fallback: rely on per-call callbacks by callers; we still return handler via config
        pass

    # Attach Langfuse callback handler only if startup marked it ready
    if os.getenv("LANGFUSE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on") and os.getenv("LANGFUSE_READY", "0").strip() in ("1", "true", "TRUE"):
        try:
            if LangfuseCallback is not None:
                handler = LangfuseCallback()
                raw_llm = raw_llm.with_config({"callbacks": [handler]})  # type: ignore[attr-defined]
        except Exception:
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
