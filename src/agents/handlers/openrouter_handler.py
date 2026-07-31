"""OpenRouter API LLM handler (``openrouter/<org>/<model>``).

Uses the OpenAI-compatible Chat Completions endpoint documented at
https://openrouter.ai/docs/quickstart via ``langchain_openai.ChatOpenAI``
with ``base_url`` pointed at OpenRouter.

Optional **inference-provider** routing is configured per model family via
env vars — see ``resolve_provider_routing``. When unset, Groq is preferred
with fallbacks allowed.

By default, endpoints that *advertise* a non–full-precision quantization
(fp8, int4, …) are filtered out; ``unknown`` (unspecified) stays allowed.
See ``resolve_quantizations`` and
https://openrouter.ai/docs/guides/routing/provider-selection#quantization
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from ...config.model_costs import MODEL_COSTS
from ..token_utils import resolve_encoder
from .base import BaseLLMHandler
from .request_timeout import resolve_llm_request_timeout

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_openrouter_request_timeout() -> float:
    """Return OpenRouter HTTP timeout seconds (env-overridable)."""
    return resolve_llm_request_timeout(specific_env="OPENROUTER_REQUEST_TIMEOUT")

# Prefer Groq when OPENROUTER_PROVIDER* is unset; allow_fallbacks=True so
# other full-precision / unknown hosts remain usable if Groq is unavailable.
DEFAULT_OPENROUTER_PROVIDER_ORDER: tuple[str, ...] = ("groq",)

# Blacklist advertised low-bit quantizations by allowlisting only full
# precision + unspecified. OpenRouter has no exclude-list for quantizations.
DEFAULT_OPENROUTER_QUANTIZATIONS: tuple[str, ...] = (
    "fp16",
    "bf16",
    "fp32",
    "unknown",
)

# Canonical family keys used in OPENROUTER_PROVIDER_<FAMILY> env vars.
_MODEL_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (family_key, substrings matched against the OpenRouter model id)
    ("gpt5nano", ("gpt-5-nano", "gpt5nano", "gpt-5.1-nano")),
    ("llama3", ("llama-3", "llama3")),
    ("qwen3", ("qwen3", "qwen-3")),
)


def openrouter_model_family(model_name: str) -> str | None:
    """Map an OpenRouter model id / full name to a family key, or ``None``.

    Examples
    --------
    >>> openrouter_model_family("openrouter/meta-llama/llama-3.1-8b-instruct")
    'llama3'
    >>> openrouter_model_family("openrouter/openai/gpt-5-nano")
    'gpt5nano'
    >>> openrouter_model_family("openrouter/qwen/qwen3-32b")
    'qwen3'
    """
    text = model_name.strip().lower()
    if text.startswith("openrouter/"):
        text = text[len("openrouter/") :]
    for family, needles in _MODEL_FAMILIES:
        if any(n in text for n in needles):
            return family
    return None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_provider_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _provider_order_disabled(raw: str) -> bool:
    return raw.strip().lower() in {"off", "none", "any", "*"}


def resolve_provider_routing(model_name: str) -> dict[str, Any] | None:
    """Build OpenRouter ``provider`` order/fallback prefs.

    Env vars (family = ``gpt5nano`` | ``llama3`` | ``qwen3``)::

        OPENROUTER_PROVIDER_GPT5NANO=openai
        OPENROUTER_PROVIDER_LLAMA3=together,fireworks
        OPENROUTER_PROVIDER_QWEN3=deepinfra

    Comma-separated values become ``provider.order``. When a provider is
    **explicitly** set, ``allow_fallbacks`` defaults to ``false`` (hard pin).
    Override with::

        OPENROUTER_ALLOW_FALLBACKS_LLAMA3=1
        # or global:
        OPENROUTER_ALLOW_FALLBACKS=0

    A global ``OPENROUTER_PROVIDER`` applies when the model family has no
    specific override.

    When neither is set, defaults to preferring Groq with
    ``allow_fallbacks=True``. Set ``OPENROUTER_PROVIDER=off`` (or
    ``none`` / ``any`` / ``*``) to disable order preference entirely.
    """
    family = openrouter_model_family(model_name)
    raw: str | None = None
    if family is not None:
        raw = (os.getenv(f"OPENROUTER_PROVIDER_{family.upper()}") or "").strip()
    if not raw:
        raw = (os.getenv("OPENROUTER_PROVIDER") or "").strip()

    if raw and _provider_order_disabled(raw):
        return None

    if not raw:
        # Default preference: Groq first, then other eligible hosts.
        allow_raw = None
        if family is not None:
            allow_raw = os.getenv(f"OPENROUTER_ALLOW_FALLBACKS_{family.upper()}")
        if allow_raw is None or not str(allow_raw).strip():
            allow_raw = os.getenv("OPENROUTER_ALLOW_FALLBACKS")
        return {
            "order": list(DEFAULT_OPENROUTER_PROVIDER_ORDER),
            "allow_fallbacks": _parse_bool(allow_raw, default=True),
        }

    order = _parse_provider_list(raw)
    if not order:
        return None

    # Explicit pin: no fallbacks unless overridden.
    allow_default = False
    allow_raw = None
    if family is not None:
        allow_raw = os.getenv(f"OPENROUTER_ALLOW_FALLBACKS_{family.upper()}")
    if allow_raw is None or not str(allow_raw).strip():
        allow_raw = os.getenv("OPENROUTER_ALLOW_FALLBACKS")
    allow_fallbacks = _parse_bool(allow_raw, default=allow_default)

    return {
        "order": order,
        "allow_fallbacks": allow_fallbacks,
    }


def resolve_quantizations() -> list[str] | None:
    """Return OpenRouter ``provider.quantizations`` allowlist, or ``None``.

    Defaults to :data:`DEFAULT_OPENROUTER_QUANTIZATIONS` so hosts that
    *advertise* fp8 / int4 / etc. are blacklisted, while full-precision
    (fp16/bf16/fp32) and unspecified (``unknown``, e.g. Groq) remain.

    Override with ``OPENROUTER_QUANTIZATIONS`` (comma-separated). Set to
    ``off`` / ``none`` / ``any`` / ``*`` / empty to disable the filter.
    """
    raw = os.getenv("OPENROUTER_QUANTIZATIONS")
    if raw is None:
        return list(DEFAULT_OPENROUTER_QUANTIZATIONS)
    stripped = raw.strip()
    if not stripped or stripped.lower() in {"off", "none", "any", "*"}:
        return None
    return _parse_provider_list(stripped)


def resolve_provider_preferences(model_name: str) -> dict[str, Any] | None:
    """Merge provider order + quantization filters into one ``provider`` object."""
    provider: dict[str, Any] = {}
    routing = resolve_provider_routing(model_name)
    if routing:
        provider.update(routing)
    quantizations = resolve_quantizations()
    if quantizations:
        provider["quantizations"] = quantizations
    return provider or None


class OpenRouterHandler(BaseLLMHandler):
    """ChatOpenAI pointed at OpenRouter using ``OPENROUTER_API_KEY``."""

    scheme = "openrouter"
    separator = "/"
    api_key_env = "OPENROUTER_API_KEY"

    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        backend_name = self.parse_model_id(model_name)
        api_key = self.require_api_key()

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI  # type: ignore

        base_url = (
            os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
            or DEFAULT_OPENROUTER_BASE_URL
        )

        default_headers: dict[str, str] = {}
        referer = (
            os.getenv("OPENROUTER_HTTP_REFERER")
            or os.getenv("HTTP_REFERER")
            or ""
        ).strip()
        app_title = (os.getenv("OPENROUTER_APP_TITLE") or "").strip()
        if referer:
            default_headers["HTTP-Referer"] = referer
        if app_title:
            default_headers["X-OpenRouter-Title"] = app_title

        model_cfg = MODEL_COSTS.get(model_name, {}) or MODEL_COSTS.get(
            f"openrouter/{backend_name}", {}
        )
        max_out = int(model_cfg.get("output_limit", 0))

        request_timeout = resolve_openrouter_request_timeout()
        common_kwargs: dict[str, Any] = dict(
            api_key=api_key,
            base_url=base_url,
            model=backend_name,
            temperature=0,
            timeout=request_timeout,
        )
        if default_headers:
            common_kwargs["default_headers"] = default_headers

        provider = resolve_provider_preferences(model_name)
        if provider is not None:
            common_kwargs["extra_body"] = {"provider": provider}
            logger.info(
                "[openrouter] provider prefs family=%s order=%s "
                "allow_fallbacks=%s quantizations=%s",
                openrouter_model_family(model_name),
                provider.get("order"),
                provider.get("allow_fallbacks"),
                provider.get("quantizations"),
            )

        if max_out > 0:
            raw_llm = ChatOpenAI(max_tokens=max_out, **common_kwargs)  # type: ignore[call-arg]
        else:
            raw_llm = ChatOpenAI(**common_kwargs)  # type: ignore[call-arg]

        # Native HF tokenizers for qwen/llama; tiktoken for OpenAI-family models.
        enc = resolve_encoder(model_name)
        logger.info(
            "[openrouter] Initialized model=%s base_url=%s tokenizer=%s "
            "timeout=%.1fs",
            backend_name,
            base_url,
            type(enc).__name__ if enc is not None else None,
            request_timeout,
        )
        return enc, raw_llm


__all__ = [
    "OpenRouterHandler",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_OPENROUTER_PROVIDER_ORDER",
    "DEFAULT_OPENROUTER_QUANTIZATIONS",
    "openrouter_model_family",
    "resolve_openrouter_request_timeout",
    "resolve_provider_routing",
    "resolve_provider_preferences",
    "resolve_quantizations",
]
