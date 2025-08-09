"""Centralized model/API rate limits configuration.

This module defines per-model request/token throughput caps and generic
backoff settings used by the LLM invocation layer. Values can be overridden
via environment variables for quick tuning without code changes.
"""

from __future__ import annotations

import os
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Backoff settings (override via environment variables)
# ---------------------------------------------------------------------------

BACKOFF_SETTINGS: Dict[str, Any] = {
    # Maximum retry attempts when encountering transient/rate-limit errors
    "max_retries": int(os.getenv("RL_MAX_RETRIES", "5")),
    # Initial backoff delay in seconds
    "initial_delay": float(os.getenv("RL_BACKOFF_INITIAL", "0.5")),
    # Exponential multiplier applied per attempt
    "multiplier": float(os.getenv("RL_BACKOFF_MULTIPLIER", "2.0")),
    # Maximum backoff delay in seconds
    "max_delay": float(os.getenv("RL_BACKOFF_MAX", "20.0")),
    # Full-jitter scale factor (0..1) applied to computed delay
    "jitter": float(os.getenv("RL_BACKOFF_JITTER", "0.3")),
}


# ---------------------------------------------------------------------------
# Per-model rate limits
#
# Requests/minute (RPM) and Tokens/minute (TPM) here represent a *soft cap*
# enforced by our client-side rate limiter to minimize server-side 429s.
# If a model is not listed, defaults are used.
# ---------------------------------------------------------------------------

DEFAULT_RPM = int(os.getenv("RL_DEFAULT_RPM", "60"))
DEFAULT_TPM = int(os.getenv("RL_DEFAULT_TPM", "150000"))

# Default expected output ratio (portion of output_limit to reserve)
EXPECTED_OUTPUT_RATIO_DEFAULT = float(os.getenv("RL_EXPECTED_OUTPUT_RATIO", "0.25"))

MODEL_RATE_LIMITS: Dict[str, Dict[str, Any]] = {
    # OpenAI models (tune as per published guidance / organization quotas)
    "openai/gpt-4.1-nano-2025-04-14": {
        "requests_per_minute": int(os.getenv("RL_RPM_GPT41_NANO", "120")),
        "tokens_per_minute": int(os.getenv("RL_TPM_GPT41_NANO", "300000")),
        "expected_output_ratio": float(os.getenv("RL_OUTRATIO_GPT41_NANO", str(EXPECTED_OUTPUT_RATIO_DEFAULT))),
    },
    "openai/gpt-4o-mini": {
        "requests_per_minute": int(os.getenv("RL_RPM_GPT4O_MINI", "100")),
        "tokens_per_minute": int(os.getenv("RL_TPM_GPT4O_MINI", "200000")),
        "expected_output_ratio": float(os.getenv("RL_OUTRATIO_GPT4O_MINI", str(EXPECTED_OUTPUT_RATIO_DEFAULT))),
    },
}


def get_limits_for_model(model_name: str) -> Dict[str, Any]:
    """Return per-minute RPM/TPM limits for a model, with sane defaults.

    The lookup is performed on the fully qualified model key used elsewhere
    in the codebase (e.g. "openai/<name>").
    """
    limits = dict(MODEL_RATE_LIMITS.get(model_name, {}))
    if limits:
        # ensure required keys exist
        limits.setdefault("requests_per_minute", DEFAULT_RPM)
        limits.setdefault("tokens_per_minute", DEFAULT_TPM)
        limits.setdefault("expected_output_ratio", EXPECTED_OUTPUT_RATIO_DEFAULT)
        return limits
    # Fallback defaults
    return {
        "requests_per_minute": DEFAULT_RPM,
        "tokens_per_minute": DEFAULT_TPM,
        "expected_output_ratio": EXPECTED_OUTPUT_RATIO_DEFAULT,
    }

