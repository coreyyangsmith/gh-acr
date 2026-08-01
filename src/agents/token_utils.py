from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from ..config.model_costs import get_model_config

# Lazy import so consumers without tiktoken can still use word-count fallback
try:  # pragma: no cover
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

# Hugging Face tokenizer families that must not fall back to cl100k_base.
_NATIVE_TOKENIZER_FAMILIES = frozenset({"qwen", "llama"})

# Explicit HF repos for configured provider models (canonical MODEL_COSTS keys).
_NATIVE_HF_REPOS: dict[str, str] = {
    "openrouter/qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "groq:qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "groq:llama-3.1-8b-instant": "meta-llama/Llama-3.1-8B-Instruct",
    "local:Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
    "local:Qwen/Qwen3-32B": "Qwen/Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "local:meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "local:meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
}

_TIKTOKEN_ENCODINGS = frozenset(
    {
        "o200k_base",
        "o200k_base_encoding",
        "cl100k_base",
        "p50k_base",
        "r50k_base",
        "gpt2",
    }
)


@lru_cache(maxsize=None)
def tiktoken_encoder(model_name: str):  # noqa: D401
    """Return a tiktoken encoder for model_name, or None if unavailable."""
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
def _tiktoken_encoding(encoding_name: str):
    """Return a tiktoken Encoding by name, or None."""
    if tiktoken is None:
        return None
    name = encoding_name
    if name == "o200k_base_encoding":
        name = "o200k_base"
    if name == "gpt2":
        name = "r50k_base"
    try:
        return tiktoken.get_encoding(name)
    except Exception:  # pragma: no cover
        return None


def _strip_provider_prefix(model_name: str) -> str:
    if model_name.startswith("openrouter/"):
        return model_name[len("openrouter/") :]
    if model_name.startswith("openai/"):
        return model_name[len("openai/") :]
    if model_name.startswith("local:"):
        return model_name[len("local:") :]
    if model_name.startswith("groq:"):
        return model_name[len("groq:") :]
    if model_name.startswith("groq/"):
        return model_name[len("groq/") :]
    return model_name


def hf_repo_for_model(model_name: str, tokenizer_family: str | None = None) -> str:
    """Map a configured model id to a Hugging Face tokenizer repo id."""
    if model_name in _NATIVE_HF_REPOS:
        return _NATIVE_HF_REPOS[model_name]

    key = model_name
    if key.startswith("openrouter/") and key[len("openrouter/") :] in {
        k[len("openrouter/") :] for k in _NATIVE_HF_REPOS if k.startswith("openrouter/")
    }:
        return _NATIVE_HF_REPOS.get(f"openrouter/{key[len('openrouter/') :]}", key)

    # Exact local-style ids already look like HF repos.
    stripped = _strip_provider_prefix(model_name)
    lower = stripped.lower()
    family = (tokenizer_family or "").strip().lower()

    if "qwen3-32b" in lower:
        return "Qwen/Qwen3-32B"
    if "qwen3-8b" in lower:
        return "Qwen/Qwen3-8B"
    if "llama-3.1-8b-instruct" in lower or "llama-3.1-8b" in lower:
        if "instruct" in lower:
            return "meta-llama/Llama-3.1-8B-Instruct"
        return "meta-llama/Llama-3.1-8B"
    if "llama-3.2-1b" in lower:
        return "meta-llama/Llama-3.2-1B"

    if family == "qwen" and "/" in stripped:
        # Preserve org/model casing from common Qwen ids when possible.
        org, name = stripped.split("/", 1)
        return f"{org}/{name}" if org.lower() != "qwen" else f"Qwen/{name}"
    if family == "llama" and "/" in stripped:
        return stripped

    raise RuntimeError(
        f"No Hugging Face tokenizer repo mapping for model={model_name!r} "
        f"(tokenizer_family={tokenizer_family!r}). Add an entry to "
        "token_utils._NATIVE_HF_REPOS or MODEL_COSTS."
    )


class _TokenizersEncoderAdapter:
    """Minimal encode()-compatible wrapper around ``tokenizers.Tokenizer``."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)


def _load_hf_tokenizer_via_tokenizers(repo_id: str, hf_token: str | None) -> Any:
    """Load tokenizer.json via huggingface_hub + tokenizers (no Torch needed)."""
    from huggingface_hub import hf_hub_download  # type: ignore
    from tokenizers import Tokenizer  # type: ignore

    path = hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        token=hf_token,
    )
    return _TokenizersEncoderAdapter(Tokenizer.from_file(path))


def _load_hf_tokenizer_via_transformers(repo_id: str, hf_token: str | None) -> Any:
    """Fallback loader using transformers.AutoTokenizer (may import Torch)."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional / broken env path
        raise RuntimeError(
            f"Native tokenizer fallback for {repo_id!r} could not import "
            f"'transformers.AutoTokenizer' ({type(exc).__name__}: {exc}). "
            "Prefer the lightweight path (tokenizers + huggingface_hub via "
            "`uv sync`), or install a matching local-llm stack with: "
            "uv sync --extra local-llm"
        ) from exc
    return AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=True,
        token=hf_token,
    )


@lru_cache(maxsize=None)
def _load_hf_tokenizer(repo_id: str) -> Any:
    """Load and cache a Hugging Face tokenizer; never silently fall back.

    Prefers ``tokenizers`` + ``huggingface_hub`` so OpenRouter/Groq token
    counting does not require Torch. Falls back to ``transformers`` only when
    tokenizer.json cannot be loaded that way.
    """
    # Lazy import avoids circular deps at module load; parity with local_backend.
    from .handlers.hf_utils import get_hf_token

    hf_token = get_hf_token()
    tokenizers_error: Exception | None = None
    try:
        return _load_hf_tokenizer_via_tokenizers(repo_id, hf_token)
    except Exception as exc:
        tokenizers_error = exc

    try:
        return _load_hf_tokenizer_via_transformers(repo_id, hf_token)
    except Exception as exc:
        details = f"Underlying error: {exc}"
        if tokenizers_error is not None:
            details = (
                f"tokenizers path failed ({type(tokenizers_error).__name__}: "
                f"{tokenizers_error}); transformers fallback failed "
                f"({type(exc).__name__}: {exc})"
            )
        raise RuntimeError(
            f"Failed to load Hugging Face tokenizer {repo_id!r} for token counting. "
            "Ensure tokenizers + huggingface_hub are installed (`uv sync`), the "
            "model is cached locally or HF_TOKEN is set for gated repos "
            "(e.g. meta-llama/*), and the model license is accepted on Hugging Face "
            f"if required. {details}. "
            "For local model inference also run: uv sync --extra local-llm"
        ) from exc


def resolve_encoder(model_name: str) -> Optional[Any]:
    """Resolve an encode/decode-capable encoder for *model_name*.

    - ``qwen`` / ``llama`` families use the native Hugging Face tokenizer.
      Missing transformers or a failed download raises ``RuntimeError``
      (no silent cl100k undercount).
    - OpenAI / tiktoken encodings use tiktoken.
    - Unknown families return ``None`` so callers fall back to word counts.
    """
    cfg = get_model_config(model_name)
    tokenizer_name = str(cfg.get("tokenizer") or "").strip().lower()

    if tokenizer_name in _NATIVE_TOKENIZER_FAMILIES:
        repo = hf_repo_for_model(model_name, tokenizer_name)
        return _load_hf_tokenizer(repo)

    if tokenizer_name in _TIKTOKEN_ENCODINGS:
        enc = _tiktoken_encoding(tokenizer_name)
        if enc is not None:
            return enc

    # OpenAI-compatible model ids: prefer encoding_for_model.
    bare = _strip_provider_prefix(model_name)
    if (
        model_name.startswith("openai/")
        or model_name.startswith("openrouter/openai/")
        or bare.startswith("gpt-")
        or bare.startswith("o1")
        or bare.startswith("o3")
        or bare.startswith("o4")
    ):
        return tiktoken_encoder(bare)

    # Unknown / unconfigured family: do not invent a cl100k undercount.
    return None


def count_tokens(encoder: Optional[Any], text: str) -> int:  # noqa: D401
    """Return token count using encoder if available; fallback to words."""
    if encoder is None:
        return len(text.split())
    if hasattr(encoder, "encode"):
        try:
            ids = encoder.encode(text)  # type: ignore[attr-defined]
            # HF tokenizers may return Encoding objects; prefer length of ids list.
            try:
                return len(ids)
            except TypeError:  # pragma: no cover
                return len(list(ids))
        except Exception:
            # Local HF tokenizers can OOM or reject pathological inputs; word
            # count keeps TruncatingLLMWrapper / prompt_budget from crashing.
            return len(text.split())
    return len(text.split())


def chars_per_token_estimate(text: str) -> int:
    """Conservative chars/4 token estimate (ceil)."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def estimate_prompt_tokens(encoder: Optional[Any], text: str) -> int:
    """Conservative prompt size for budgeting / truncation decisions.

    Takes the max of the encoder count and a chars/4 heuristic so provider
    overcounts (seen on OpenRouter Llama) still trigger clipping when the
    native HF tokenizer underestimates.
    """
    return max(count_tokens(encoder, text), chars_per_token_estimate(text))


__all__ = [
    "tiktoken_encoder",
    "resolve_encoder",
    "hf_repo_for_model",
    "count_tokens",
    "chars_per_token_estimate",
    "estimate_prompt_tokens",
]
