from __future__ import annotations

from typing import Any, Optional
import logging

from langchain_core.callbacks import BaseCallbackHandler

from ..config.model_costs import estimate_usd_cost, get_model_config
from ..config.rate_limits import get_limits_for_model, BACKOFF_SETTINGS
from ..utils.rate_limiter import LimiterRegistry
from .token_utils import count_tokens


logger = logging.getLogger(__name__)


def _extract_usage_tokens(response: Any) -> tuple[int | None, int | None]:
    """Prefer provider-reported usage from an LLMResult when available."""
    # LangChain LLMResult.llm_output["token_usage"]
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            if prompt is None:
                prompt = usage.get("input_tokens")
            completion = usage.get("completion_tokens")
            if completion is None:
                completion = usage.get("output_tokens")
            if prompt is not None and completion is not None:
                return int(prompt), int(completion)
    except Exception:
        pass

    # Chat generations may carry usage_metadata on the message
    try:
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg is not None else None
                if isinstance(meta, dict):
                    prompt = meta.get("input_tokens")
                    if prompt is None:
                        prompt = meta.get("prompt_tokens")
                    completion = meta.get("output_tokens")
                    if completion is None:
                        completion = meta.get("completion_tokens")
                    if prompt is not None and completion is not None:
                        return int(prompt), int(completion)
                resp_meta = getattr(msg, "response_metadata", None) if msg is not None else None
                if isinstance(resp_meta, dict):
                    usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
                    if isinstance(usage, dict):
                        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
                        completion = usage.get("completion_tokens", usage.get("output_tokens"))
                        if prompt is not None and completion is not None:
                            return int(prompt), int(completion)
    except Exception:
        pass

    return None, None


class RateLimitAndCostHandler(BaseCallbackHandler):
    """LangChain callback that enforces rate limits and logs cost/tokens."""

    def __init__(self, *, encoder: Optional[Any], model_name: str):
        self.encoder = encoder
        self.model_name = model_name
        limits = get_limits_for_model(model_name)
        self.expected_output_ratio: float = float(limits.get("expected_output_ratio", 0.25))
        rpm = int(limits.get("requests_per_minute", 60))
        tpm = int(limits.get("tokens_per_minute", 150000))
        rpd = limits.get("requests_per_day")
        tpd = limits.get("tokens_per_day")
        self._limiter = LimiterRegistry.get(
            key=f"{model_name}",
            rpm=rpm,
            tpm=tpm,
            backoff=BACKOFF_SETTINGS,
            rpd=int(rpd) if rpd is not None else None,
            tpd=int(tpd) if tpd is not None else None,
        )
        self._reservations: dict[Any, dict[str, int]] = {}

    def _backend_name(self) -> str:
        return self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:  # type: ignore[override]
        prompt_text = "\n\n".join(prompts or [])
        prompt_tokens = count_tokens(self.encoder, prompt_text)
        model_cfg = get_model_config(self.model_name)

        input_limit = int(model_cfg.get("input_limit", 0))
        output_limit = int(model_cfg.get("output_limit", 0))
        sliding_window = bool(model_cfg.get("sliding_window", False))
        total_limit = int(model_cfg.get("total_limit", 0))

        expected_output_tokens = int(self.expected_output_ratio * output_limit) if output_limit else int(0.25 * prompt_tokens)
        if output_limit:
            expected_output_tokens = min(expected_output_tokens, output_limit)

        # NOTE: We only LOG if prompts exceed limits here, but do NOT truncate.
        # Truncation is handled by TruncatingLLMWrapper BEFORE this callback fires.
        # Double-truncation caused issues on Compute Canada.
        if sliding_window and total_limit:
            allowed_prompt = max(1, total_limit - expected_output_tokens)
            if prompt_tokens > allowed_prompt:
                logger.warning(
                    "[RateLimitAndCostHandler] Prompt exceeds sliding window limit: "
                    "tokens=%d, allowed=%d, total_limit=%d, model=%s. "
                    "TruncatingLLMWrapper should have already handled this.",
                    prompt_tokens, allowed_prompt, total_limit, self.model_name
                )
            expected_total_tokens = min(total_limit, prompt_tokens + expected_output_tokens)
        else:
            if input_limit and prompt_tokens > input_limit:
                logger.warning(
                    "[RateLimitAndCostHandler] Prompt exceeds input limit: "
                    "tokens=%d, limit=%d, model=%s. "
                    "TruncatingLLMWrapper should have already handled this.",
                    prompt_tokens, input_limit, self.model_name
                )
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

        api_prompt, api_completion = _extract_usage_tokens(response)
        usage_from_api = api_prompt is not None and api_completion is not None

        if usage_from_api:
            prompt_tokens = int(api_prompt)
            completion_tokens = int(api_completion)
        else:
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
        cost_in, cost_out, total_cost = estimate_usd_cost(
            self.model_name, prompt_tokens, completion_tokens
        )
        has_rates = "input_cost_per_1k" in get_model_config(self.model_name)

        try:
            self._limiter.adjust(actual_tokens=int(total_tokens), reserved_tokens=int(info.get("reserved", 0)))
        except Exception:
            pass

        # Structured per-call token record for the run ledger
        node = ""
        try:
            from .observability import append_llm_call, get_llm_node, get_run_context

            ctx = get_run_context()
            node = get_llm_node()
            append_llm_call(
                {
                    "node": node or None,
                    "scenario_id": ctx.get("scenario_id") or None,
                    "eval_method": ctx.get("eval_method") or None,
                    "model_name": self.model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_in": cost_in,
                    "cost_out": cost_out,
                    "total_cost": total_cost,
                    "usage_from_api": usage_from_api,
                }
            )
        except Exception:
            pass

        # Best-effort: push the same MODEL_COSTS dollars into the active LangFuse generation
        try:
            from .observability.langfuse_tracing import is_langfuse_enabled

            if is_langfuse_enabled() and (cost_in or cost_out or total_cost):
                from langfuse import get_client  # type: ignore

                get_client().update_current_generation(
                    usage_details={
                        "input": int(prompt_tokens),
                        "output": int(completion_tokens),
                    },
                    cost_details={
                        "input": float(cost_in),
                        "output": float(cost_out),
                        "total": float(total_cost),
                    },
                )
        except Exception:
            pass

        estimated_suffix = " (estimated)" if not has_rates else ("" if usage_from_api else " (tiktoken)")
        if node:
            logger.debug(
                "LLM call to %s completed (node=%s).\n  *  Tokens: %d prompt, %d completion (%d total)\n  *  Cost:   $%.4f%s",
                self._backend_name(),
                node,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_cost,
                estimated_suffix,
            )
        else:
            logger.debug(
                "LLM call to %s completed.\n  *  Tokens: %d prompt, %d completion (%d total)\n  *  Cost:   $%.4f%s",
                self._backend_name(),
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_cost,
                estimated_suffix,
            )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:  # type: ignore[override]
        info = self._reservations.pop(run_id, None)
        logger.error(
            "LLM call to %s failed: %s: %s",
            self._backend_name(),
            type(error).__name__,
            error,
        )
        if info is None:
            return
        try:
            self._limiter.adjust(actual_tokens=0, reserved_tokens=int(info.get("reserved", 0)))
            self._limiter.last_error = f"{type(error).__name__}: {error}"
        except Exception:
            pass


__all__ = ["RateLimitAndCostHandler", "_extract_usage_tokens"]
