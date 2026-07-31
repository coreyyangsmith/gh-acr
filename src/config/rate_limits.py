"""Centralized model/API rate limits configuration.

This module defines per-model request/token throughput caps and generic
backoff settings used by the LLM invocation layer. These limits are enforced
client-side to minimize server-side 429 (rate limit) errors.

When running with ``--concurrency > 1`` (scenario worker pool), these soft
caps only apply if ``RL_ENABLE_WAITING=1``. Waiting is **disabled by default**
so concurrent workers are not paced by the local token bucket.

Configuration Levels
--------------------
1. **BACKOFF_SETTINGS**: Retry behavior for transient errors
2. **MODEL_RATE_LIMITS**: Per-model RPM/TPM (and optional RPD/TPD) limits
3. **Default limits**: Fallback for unlisted models

Environment Variables
---------------------
Backoff Settings:
- RL_MAX_RETRIES: Maximum retry attempts (default: 5)
- RL_BACKOFF_INITIAL: Initial delay in seconds (default: 0.5)
- RL_BACKOFF_MULTIPLIER: Exponential multiplier (default: 2.0)
- RL_BACKOFF_MAX: Maximum delay in seconds (default: 20.0)
- RL_BACKOFF_JITTER: Jitter factor 0-1 (default: 0.3)

Default Limits:
- RL_DEFAULT_RPM: Default requests per minute (default: 60)
- RL_DEFAULT_TPM: Default tokens per minute (default: 150000)
- RL_EXPECTED_OUTPUT_RATIO: Expected output/input ratio (default: 0.25)

Model-Specific Overrides:
- RL_RPM_GPT41_NANO, RL_TPM_GPT41_NANO, etc.
- RL_RPM_GROQ_LLAMA31_8B, RL_TPM_GROQ_LLAMA31_8B,
  RL_RPD_GROQ_LLAMA31_8B, RL_TPD_GROQ_LLAMA31_8B

Recommended values for concurrent hosted-API runs (tune to your account tier):
- OpenAI Tier 1 gpt-4o-mini: ~500 RPM / ~200k TPM → set RL_RPM_GPT4O_MINI=400, RL_TPM_GPT4O_MINI=180000
- OpenAI Tier 3+ gpt-4o-mini: often 5000+ RPM / 2M+ TPM → raise accordingly
- Groq llama-3.1-8b-instant (published): 30 RPM / 6K TPM / 14.4K RPD / 500K TPD
  → defaults below are ~90% of published ceilings; enable ``RL_ENABLE_WAITING=1``
- Unlisted models: set RL_DEFAULT_RPM / RL_DEFAULT_TPM above your planned concurrency

Example Usage
-------------
>>> from src.config.rate_limits import get_limits_for_model, BACKOFF_SETTINGS
>>> limits = get_limits_for_model("openai/gpt-4o-mini")
>>> print(f"RPM: {limits['requests_per_minute']}, TPM: {limits['tokens_per_minute']}")
"""

from __future__ import annotations

import os
from typing import Any, Dict


# -----------------------------------------------------------------------------
# Exponential Backoff Settings
# -----------------------------------------------------------------------------

BACKOFF_SETTINGS: Dict[str, Any] = {
    "max_retries": int(os.getenv("RL_MAX_RETRIES", "5")),
    "initial_delay": float(os.getenv("RL_BACKOFF_INITIAL", "0.5")),
    "multiplier": float(os.getenv("RL_BACKOFF_MULTIPLIER", "2.0")),
    "max_delay": float(os.getenv("RL_BACKOFF_MAX", "20.0")),
    "jitter": float(os.getenv("RL_BACKOFF_JITTER", "0.3")),
}
"""Exponential backoff configuration for handling transient errors.

Keys:
- max_retries: Maximum number of retry attempts before giving up
- initial_delay: First retry delay in seconds
- multiplier: Factor applied to delay on each retry (exponential growth)
- max_delay: Cap on retry delay to prevent excessive waits
- jitter: Random factor (0-1) to prevent thundering herd
"""


# -----------------------------------------------------------------------------
# Default Rate Limits
# -----------------------------------------------------------------------------

DEFAULT_RPM: int = int(os.getenv("RL_DEFAULT_RPM", "60"))
"""Default requests per minute for unlisted models."""

DEFAULT_TPM: int = int(os.getenv("RL_DEFAULT_TPM", "150000"))
"""Default tokens per minute for unlisted models."""

EXPECTED_OUTPUT_RATIO_DEFAULT: float = float(os.getenv("RL_EXPECTED_OUTPUT_RATIO", "0.25"))
"""Expected ratio of output tokens to output limit.

Used to reserve token budget for the response when calculating
how many input tokens can be used.
"""


# -----------------------------------------------------------------------------
# Per-Model Rate Limits
# -----------------------------------------------------------------------------

# Groq published ceilings for llama-3.1-8b-instant (developer tier docs):
#   RPM=30, RPD=14400, TPM=6000, TPD=500000
# Soft defaults keep ~10% headroom to reduce 429s under concurrency.
_GROQ_LLAMA31_8B_PUBLISHED_RPM = 30
_GROQ_LLAMA31_8B_PUBLISHED_TPM = 6000
_GROQ_LLAMA31_8B_PUBLISHED_RPD = 14400
_GROQ_LLAMA31_8B_PUBLISHED_TPD = 500000

MODEL_RATE_LIMITS: Dict[str, Dict[str, Any]] = {
    "openai/gpt-4.1-nano-2025-04-14": {
        "requests_per_minute": int(os.getenv("RL_RPM_GPT41_NANO", "120")),
        "tokens_per_minute": int(os.getenv("RL_TPM_GPT41_NANO", "300000")),
        "expected_output_ratio": float(os.getenv("RL_OUTRATIO_GPT41_NANO", str(EXPECTED_OUTPUT_RATIO_DEFAULT))),
    },
    "openai/gpt-4o-mini": {
        # Soft client caps (below typical Tier-1 OpenAI limits) so concurrent
        # workers have headroom without immediately tripping 429s. Override via
        # RL_RPM_GPT4O_MINI / RL_TPM_GPT4O_MINI to match your account tier.
        "requests_per_minute": int(os.getenv("RL_RPM_GPT4O_MINI", "400")),
        "tokens_per_minute": int(os.getenv("RL_TPM_GPT4O_MINI", "180000")),
        "expected_output_ratio": float(os.getenv("RL_OUTRATIO_GPT4O_MINI", str(EXPECTED_OUTPUT_RATIO_DEFAULT))),
    },
    "groq:llama-3.1-8b-instant": {
        # Soft caps ~90% of Groq published developer-tier ceilings.
        # Source: https://console.groq.com/docs/rate-limits
        "requests_per_minute": int(
            os.getenv("RL_RPM_GROQ_LLAMA31_8B", str(int(_GROQ_LLAMA31_8B_PUBLISHED_RPM * 0.9)))
        ),
        "tokens_per_minute": int(
            os.getenv("RL_TPM_GROQ_LLAMA31_8B", str(int(_GROQ_LLAMA31_8B_PUBLISHED_TPM * 0.9)))
        ),
        "requests_per_day": int(
            os.getenv("RL_RPD_GROQ_LLAMA31_8B", str(_GROQ_LLAMA31_8B_PUBLISHED_RPD))
        ),
        "tokens_per_day": int(
            os.getenv("RL_TPD_GROQ_LLAMA31_8B", str(_GROQ_LLAMA31_8B_PUBLISHED_TPD))
        ),
        "expected_output_ratio": float(
            os.getenv("RL_OUTRATIO_GROQ_LLAMA31_8B", str(EXPECTED_OUTPUT_RATIO_DEFAULT))
        ),
        "published_requests_per_minute": _GROQ_LLAMA31_8B_PUBLISHED_RPM,
        "published_tokens_per_minute": _GROQ_LLAMA31_8B_PUBLISHED_TPM,
        "published_requests_per_day": _GROQ_LLAMA31_8B_PUBLISHED_RPD,
        "published_tokens_per_day": _GROQ_LLAMA31_8B_PUBLISHED_TPD,
    },
}
"""Per-model rate limit configurations.

These represent soft caps enforced by the client-side rate limiter
to minimize server-side 429 errors. Values should be set below the
actual API limits to provide headroom.
"""


def get_limits_for_model(model_name: str) -> Dict[str, Any]:
    """Return per-minute RPM/TPM limits for a model, with sane defaults.

    The lookup is performed on the fully qualified model key used elsewhere
    in the codebase (e.g., "openai/<name>" or "groq:<model>").

    Parameters
    ----------
    model_name
        The model identifier (e.g., "openai/gpt-4o-mini")

    Returns
    -------
    Dict[str, Any]
        A dict containing at least:
        - requests_per_minute: int
        - tokens_per_minute: int
        - expected_output_ratio: float
        Optional daily quotas when configured:
        - requests_per_day: int
        - tokens_per_day: int

    Examples
    --------
    >>> limits = get_limits_for_model("openai/gpt-4o-mini")
    >>> print(limits["tokens_per_minute"])
    200000
    """
    limits = dict(MODEL_RATE_LIMITS.get(model_name, {}))
    if limits:
        # Ensure all required keys exist
        limits.setdefault("requests_per_minute", DEFAULT_RPM)
        limits.setdefault("tokens_per_minute", DEFAULT_TPM)
        limits.setdefault("expected_output_ratio", EXPECTED_OUTPUT_RATIO_DEFAULT)
        return limits

    # Fallback to defaults for unlisted models
    return {
        "requests_per_minute": DEFAULT_RPM,
        "tokens_per_minute": DEFAULT_TPM,
        "expected_output_ratio": EXPECTED_OUTPUT_RATIO_DEFAULT,
    }


__all__ = [
    "BACKOFF_SETTINGS",
    "DEFAULT_RPM",
    "DEFAULT_TPM",
    "EXPECTED_OUTPUT_RATIO_DEFAULT",
    "MODEL_RATE_LIMITS",
    "get_limits_for_model",
]
