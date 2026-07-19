"""Shared prompt-token budget calculation.

Used by structured evidence fitting and the TruncatingLLMWrapper safety net
so both agree on ``min(input_limit, total_limit - output_limit) - buffer``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from ...config.model_costs import MODEL_COSTS, get_model_config

DEFAULT_PROMPT_SAFETY_BUFFER = 64


def prompt_safety_buffer() -> int:
    try:
        return max(
            0,
            int(os.getenv("PROMPT_TRUNCATION_BUFFER", str(DEFAULT_PROMPT_SAFETY_BUFFER))),
        )
    except Exception:
        return DEFAULT_PROMPT_SAFETY_BUFFER


def model_cfg_for_name(model_name: str) -> dict[str, Any]:
    cfg = get_model_config(model_name)
    if cfg:
        return dict(cfg)
    # Fallback for keys that only live under exact MODEL_COSTS entries.
    return dict(MODEL_COSTS.get(model_name, {}) or {})


def allowed_prompt_tokens(
    model_name: str,
    *,
    encoder: Optional[Any] = None,
    cfg: Mapping[str, Any] | None = None,
) -> int:
    """Return max prompt tokens allowed (safety buffer already applied)."""
    model_cfg = dict(cfg) if cfg is not None else model_cfg_for_name(model_name)
    try:
        input_limit = int(model_cfg.get("input_limit", 0) or 0)
    except Exception:
        input_limit = 0
    try:
        output_limit = int(model_cfg.get("output_limit", 0) or 0)
    except Exception:
        output_limit = 0
    try:
        total_limit = int(model_cfg.get("total_limit", 0) or 0)
    except Exception:
        total_limit = 0

    buffer = prompt_safety_buffer()
    candidates: list[int] = []
    if input_limit > 0:
        candidates.append(input_limit)
    if total_limit > 0 and output_limit > 0:
        candidates.append(total_limit - output_limit)
    elif total_limit > 0:
        candidates.append(total_limit)

    if candidates:
        return max(1, min(candidates) - buffer)

    try:
        enc_max = int(getattr(encoder, "model_max_length", 0) or 0)
    except Exception:
        enc_max = 0
    if enc_max > 0:
        return max(1, enc_max - 256)
    return 0


__all__ = [
    "DEFAULT_PROMPT_SAFETY_BUFFER",
    "allowed_prompt_tokens",
    "model_cfg_for_name",
    "prompt_safety_buffer",
]
