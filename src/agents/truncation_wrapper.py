from __future__ import annotations

import os
from typing import Any, Optional

from .token_utils import count_tokens
from ..config.model_costs import MODEL_COSTS
from ..config.rate_limits import get_limits_for_model


class TruncatingLLMWrapper:
    """Lightweight wrapper that truncates over-long prompts before invocation.

    Truncation respects per-model limits from MODEL_COSTS and the expected
    output ratio from rate_limits.get_limits_for_model. If the input exceeds
    the allowed prompt window, it is clipped to the limit instead of letting
    the downstream backend error. Default truncation side is left (keep tail).
    """

    def __init__(self, inner: Any, *, encoder: Optional[Any], model_name: str):
        self._inner = inner
        self._encoder = encoder
        self._model_name = model_name
        try:
            self._expected_output_ratio = float(
                get_limits_for_model(model_name).get("expected_output_ratio", 0.25)
            )
        except Exception:
            self._expected_output_ratio = 0.25

    def _model_cfg(self) -> dict[str, Any]:
        try:
            cfg = MODEL_COSTS.get(self._model_name, {})
            if cfg:
                return dict(cfg)
            if self._model_name.startswith("openai/"):
                return dict(MODEL_COSTS.get(self._model_name, {}) or {})
            if self._model_name.startswith("groq:"):
                alias = "groq/" + self._model_name.split(":", 1)[1]
                return dict(MODEL_COSTS.get(alias, {}) or {})
            return {}
        except Exception:
            return {}

    def _allowed_prompt_tokens(self, prompt_tokens: int) -> int:
        cfg = self._model_cfg()
        try:
            input_limit = int(cfg.get("input_limit", 0))
        except Exception:
            input_limit = 0
        try:
            output_limit = int(cfg.get("output_limit", 0))
        except Exception:
            output_limit = 0
        try:
            total_limit = int(cfg.get("total_limit", 0))
        except Exception:
            total_limit = 0
        sliding_window = bool(cfg.get("sliding_window", False))

        try:
            expected_output_tokens = (
                int(self._expected_output_ratio * output_limit)
                if output_limit
                else int(0.25 * max(1, prompt_tokens))
            )
            if output_limit:
                expected_output_tokens = min(expected_output_tokens, output_limit)
        except Exception:
            expected_output_tokens = max(1, prompt_tokens // 4)

        # Compute allowed prompt tokens from config if available
        if sliding_window and total_limit:
            allowed = max(1, total_limit - expected_output_tokens)
        elif input_limit:
            allowed = max(1, input_limit)
        else:
            allowed = 0

        # Fallback: if no config limits are known (unknown model key), derive
        # a safe allowance from the encoder's declared model_max_length and an
        # estimated new-token reserve. This ensures we still truncate even when
        # MODEL_COSTS lacks an entry for the provided model name.
        if allowed <= 0:
            try:
                enc_max = int(getattr(self._encoder, "model_max_length", 0))
            except Exception:
                enc_max = 0
            if enc_max and enc_max > 0:
                try:
                    reserve_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
                except Exception:
                    reserve_new = 256
                reserve_new = max(1, min(reserve_new, max(1, enc_max - 32)))
                allowed = max(1, enc_max - reserve_new)

        try:
            buffer_tokens = int(os.getenv("TOKENIZER_BUFFER_TOKENS", os.getenv("LOCAL_TOKENIZER_BUFFER_TOKENS", "512")))
        except Exception:
            buffer_tokens = 0

        if allowed > 0 and buffer_tokens > 0:
            allowed = max(1, allowed - buffer_tokens)
        return allowed

    def _truncate_text(self, text: str) -> str:
        try:
            if not text:
                return text
            enc = self._encoder
            prompt_tokens = count_tokens(enc, text)
            allowed = self._allowed_prompt_tokens(prompt_tokens)
            if allowed <= 0 or prompt_tokens <= allowed:
                return text
            side = os.getenv("TRUNCATION_SIDE", "left").strip().lower()
            if hasattr(enc, "encode") and hasattr(enc, "decode"):
                try:
                    ids = enc.encode(text)  # type: ignore[attr-defined]
                    keep = ids[:allowed] if side == "right" else ids[-allowed:]
                    return enc.decode(keep)  # type: ignore[attr-defined]
                except Exception:
                    pass
            words = text.split()
            return " ".join(words[:allowed]) if side == "right" else " ".join(words[-allowed:])
        except Exception:
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


__all__ = ["TruncatingLLMWrapper"]


