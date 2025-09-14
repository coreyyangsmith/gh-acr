from __future__ import annotations

from typing import Any, Dict
import os
from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ...utils.logger import logger

__all__ = ["conflict_analyzer_node"]

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "new_bypass2"
    / "conflict_judge_prompt.txt"
)
_PROMPT_STR = _PROMPT_PATH.read_text(encoding="utf-8")


def _normalize_decision(text: str) -> str:
    t = (text or "").strip().lower()
    if "a" in t:
        return "ALL_A"
    if "b" in t:
        return "ALL_B"
    return "MIX"


def conflict_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Conflict analyzer started.")
    summaries: Dict[str, Dict[str, str]] = state.get("summaries", {}) or {}
    diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
    diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}

    model_name = state.get("model_name") or os.getenv(
        "OPENAI_MODEL", "openai/gpt-4o-mini"
    )
    encoder, llm = get_backend(model_name)

    a_sum = "\n\n".join(f"{p}: {s.get('summary_a', '')}" for p, s in summaries.items())
    b_sum = "\n\n".join(f"{p}: {s.get('summary_b', '')}" for p, s in summaries.items())
    a_diff = "\n\n".join(f"{p}: {diffs_a.get(p, '')}" for p in summaries.keys())
    b_diff = "\n\n".join(f"{p}: {diffs_b.get(p, '')}" for p in summaries.keys())

    if llm is None:
        logger.warning("No LLM backend available for analyzer; using heuristic.")
        len_a = len(a_sum)
        len_b = len(b_sum)
        if abs(len_a - len_b) < 0.05 * max(1, (len_a + len_b) // 2):
            decision = "MIX"
        else:
            decision = "ALL_A" if len_a <= len_b else "ALL_B"
        raw_output = f"Heuristic decision based on summary lengths (A={len_a}, B={len_b}): {decision}"
    else:
        prompt_text = _PROMPT_STR.format(
            a_summary=a_sum, b_summary=b_sum, a_diff=a_diff, b_diff=b_diff
        )
        result = llm.invoke(prompt_text)
        content = result.content if hasattr(result, "content") else str(result)
        raw_output = content.strip()
        decision = _normalize_decision(raw_output)
        for path in summaries.keys():
            counts = state.setdefault("token_counts", {}).setdefault(
                path,
                {
                    "system_prompt": 0,
                    "original": 0,
                    "diff_a": 0,
                    "diff_b": 0,
                    "output": 0,
                },
            )
            counts["system_prompt"] += count_tokens(encoder, prompt_text)
            counts["output"] += count_tokens(encoder, raw_output)

    state["bypass_decision"] = decision
    state["bypass_method"] = (
        "A" if decision == "ALL_A" else ("B" if decision == "ALL_B" else "MIX")
    )
    state["bypass_analyzer_output"] = raw_output
    state["status"] = "analyzed"
    logger.info(f"Conflict analyzer decision: {decision}")
    return state



