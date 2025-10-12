from __future__ import annotations

from typing import Any, Optional
import logging

from langchain_core.callbacks import BaseCallbackHandler

from ..config.model_costs import MODEL_COSTS
from ..config.rate_limits import get_limits_for_model, BACKOFF_SETTINGS
from ..utils.rate_limiter import LimiterRegistry
from .token_utils import count_tokens


logger = logging.getLogger(__name__)


class RateLimitAndCostHandler(BaseCallbackHandler):
    """LangChain callback that enforces rate limits and logs cost/tokens."""

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

        expected_output_tokens = int(self.expected_output_ratio * output_limit) if output_limit else int(0.25 * prompt_tokens)
        if output_limit:
            expected_output_tokens = min(expected_output_tokens, output_limit)

        if sliding_window and total_limit:
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


__all__ = ["RateLimitAndCostHandler"]


