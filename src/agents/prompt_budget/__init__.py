"""Structure-aware prompt budgeting for merge-resolution agents."""

from __future__ import annotations

from .budget import (
    DEFAULT_PROMPT_SAFETY_BUFFER,
    allowed_prompt_tokens,
    model_cfg_for_name,
    prompt_safety_buffer,
)
from .clip import OMIT_MARKER_FMT, TRUNC_MARKER_FMT, head_tail_clip
from .fit import (
    BlockFitAction,
    EvidenceBlock,
    FitReport,
    REPAIR_HEADROOM_TOKENS,
    fit_global_ab_prompt,
    fit_variable_blocks,
)

__all__ = [
    "DEFAULT_PROMPT_SAFETY_BUFFER",
    "OMIT_MARKER_FMT",
    "TRUNC_MARKER_FMT",
    "BlockFitAction",
    "EvidenceBlock",
    "FitReport",
    "REPAIR_HEADROOM_TOKENS",
    "allowed_prompt_tokens",
    "fit_global_ab_prompt",
    "fit_variable_blocks",
    "head_tail_clip",
    "model_cfg_for_name",
    "prompt_safety_buffer",
]
