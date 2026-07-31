from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .token_utils import estimate_prompt_tokens
from .prompt_budget.budget import (
    DEFAULT_PROMPT_SAFETY_BUFFER,
    allowed_prompt_tokens as shared_allowed_prompt_tokens,
    model_cfg_for_name,
    prompt_safety_buffer,
)
from ..utils.degradation import record_degradation


logger = logging.getLogger(__name__)


class TruncatingLLMWrapper:
    """Last-resort wrapper that truncates over-long prompts before invocation.

    Prefer structure-aware fitting in ``prompt_budget`` at prompt construction.
    This wrapper remains as a transport safeguard when a caller still exceeds
    the shared context budget. The allowed prompt budget is
    ``min(input_limit, total_limit - max_tokens) - buffer``. Default truncation
    side is left (keep tail).

    Budget decisions use ``estimate_prompt_tokens`` (max of encoder count and
    chars/4) so provider overcounts still trigger clipping.
    """

    def __init__(self, inner: Any, *, encoder: Optional[Any], model_name: str):
        self._inner = inner
        self._encoder = encoder
        self._model_name = model_name

    def _model_cfg(self) -> dict[str, Any]:
        return model_cfg_for_name(self._model_name)

    def _allowed_prompt_tokens(self, prompt_tokens: int) -> int:
        """Return the max prompt tokens allowed (buffer already applied)."""
        allowed = shared_allowed_prompt_tokens(
            self._model_name, encoder=self._encoder, cfg=self._model_cfg()
        )
        if allowed > 0:
            logger.debug(
                "[TruncatingLLMWrapper] Budget model=%s allowed=%d prompt_tokens=%d",
                self._model_name,
                allowed,
                prompt_tokens,
            )
            return allowed
        logger.warning(
            "[TruncatingLLMWrapper] Could not determine allowed prompt tokens "
            "for model=%s. No truncation will be applied, which may cause errors.",
            self._model_name,
        )
        return 0

    def _clip_by_estimate(self, text: str, target_tokens: int, side: str) -> str:
        """Shrink *text* until ``estimate_prompt_tokens`` <= *target_tokens*."""
        enc = self._encoder
        if estimate_prompt_tokens(enc, text) <= target_tokens:
            return text

        # Prefer encoder id clipping when it actually reduces the estimate.
        if enc is not None and hasattr(enc, "encode") and hasattr(enc, "decode"):
            try:
                ids = enc.encode(text)  # type: ignore[attr-defined]
                try:
                    id_list = list(ids)
                except TypeError:  # pragma: no cover
                    id_list = ids
                if len(id_list) > target_tokens:
                    keep = (
                        id_list[:target_tokens]
                        if side == "right"
                        else id_list[-target_tokens:]
                    )
                    truncated = enc.decode(keep)  # type: ignore[attr-defined]
                    if estimate_prompt_tokens(enc, truncated) <= target_tokens:
                        return truncated
            except Exception as e:
                logger.warning(
                    "[TruncatingLLMWrapper] Encoder truncation failed: %s, "
                    "falling back to character-based",
                    e,
                )

        # Binary search on character length (chars/4 drives the estimate when
        # the encoder undercounts).
        lo, hi = 1, len(text)
        best = text[:1] if side == "right" else text[-1:]
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid] if side == "right" else text[-mid:]
            est = estimate_prompt_tokens(enc, candidate)
            if est <= target_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _truncate_text(self, text: str) -> str:
        try:
            if not text:
                return text
            enc = self._encoder
            prompt_tokens = estimate_prompt_tokens(enc, text)
            allowed = self._allowed_prompt_tokens(prompt_tokens)
            if allowed <= 0 or prompt_tokens <= allowed:
                return text
            side = os.getenv(
                "LOCAL_TRUNCATION_SIDE", os.getenv("TRUNCATION_SIDE", "left")
            ).strip().lower()

            target_tokens = max(1, allowed)
            buffer = prompt_safety_buffer()
            cfg = self._model_cfg()
            reserved_output = int(cfg.get("output_limit", 0) or 0)
            total_limit = int(cfg.get("total_limit", 0) or 0)
            input_limit = int(cfg.get("input_limit", 0) or 0)

            logger.warning(
                "[TruncatingLLMWrapper] Unexpected wrapper fallback truncation: "
                "tokens=%d -> allowed=%d (input_limit=%d total_limit=%d "
                "reserved_output=%d buffer=%d), side=%s, model=%s. "
                "Structured prompt budgeting should have prevented this.",
                prompt_tokens,
                target_tokens,
                input_limit,
                total_limit,
                reserved_output,
                buffer,
                side,
                self._model_name,
            )
            record_degradation(
                "prompt_truncation",
                "wrapper fallback: prompt exceeded model context window",
                detail=(
                    f"truncation_mode=wrapper_fallback tokens={prompt_tokens} "
                    f"allowed={target_tokens} input_limit={input_limit} "
                    f"total_limit={total_limit} reserved_output={reserved_output} "
                    f"buffer={buffer} side={side} model={self._model_name}"
                ),
                node="truncating_llm_wrapper",
            )

            truncated = self._clip_by_estimate(text, target_tokens, side)
            logger.info(
                "[TruncatingLLMWrapper] Truncation complete: "
                "original_chars=%d, truncated_chars=%d, estimate_after=%d",
                len(text),
                len(truncated),
                estimate_prompt_tokens(enc, truncated),
            )
            return truncated
        except Exception as e:
            logger.error("[TruncatingLLMWrapper] Truncation failed entirely: %s", e)
            return text

    def with_config(self, config: dict[str, Any] | None = None):  # type: ignore[override]
        try:
            if hasattr(self._inner, "with_config"):
                self._inner = self._inner.with_config(config)  # type: ignore[attr-defined]
        except Exception:
            pass
        return self

    def invoke(self, prompt_input: Any, config: Any | None = None):  # type: ignore[override]
        try:
            if isinstance(prompt_input, str):
                prompt_input = self._truncate_text(prompt_input)
        except Exception:
            pass
        return self._inner.invoke(prompt_input, config=config)

    async def ainvoke(self, prompt_input: Any, config: Any | None = None):  # type: ignore[override]
        try:
            if isinstance(prompt_input, str):
                prompt_input = self._truncate_text(prompt_input)
        except Exception:
            pass
        if hasattr(self._inner, "ainvoke"):
            return await self._inner.ainvoke(prompt_input, config=config)  # type: ignore[attr-defined]
        return self._inner.invoke(prompt_input, config=config)


__all__ = ["TruncatingLLMWrapper", "DEFAULT_PROMPT_SAFETY_BUFFER"]
