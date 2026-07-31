"""Shared HTTP request-timeout resolution for OpenAI-compatible LLM clients."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Under the default watchdog max_llm (900s) so hung calls fail and free
# ThreadPoolExecutor slots near soft-skip rather than blocking forever.
DEFAULT_LLM_REQUEST_TIMEOUT_S = 600.0


def resolve_llm_request_timeout(
    *,
    specific_env: str,
    default: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
) -> float:
    """Resolve a positive HTTP timeout from env.

    Checks ``specific_env`` first, then ``GHACR_LLM_REQUEST_TIMEOUT``, then
    ``default``. Invalid / non-positive values fall back to ``default``.
    """
    for name in (specific_env, "GHACR_LLM_REQUEST_TIMEOUT"):
        raw = (os.getenv(name) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "[request_timeout] Invalid %s=%r; trying next / default %.1fs",
                name,
                raw,
                default,
            )
            continue
        if value <= 0:
            logger.warning(
                "[request_timeout] Non-positive %s=%r; trying next / default %.1fs",
                name,
                raw,
                default,
            )
            continue
        return value
    return float(default)


__all__ = [
    "DEFAULT_LLM_REQUEST_TIMEOUT_S",
    "resolve_llm_request_timeout",
]
