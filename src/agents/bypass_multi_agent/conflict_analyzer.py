from __future__ import annotations

"""Conflict analyzer – quick global judgement to optionally bypass the pipeline.

This node inspects the A/B summaries (and diffs) and asks an LLM to decide one of:

- "All A"   → select Parent A for all files
- "All B"   → select Parent B for all files
- "Mix A/B" → proceed with the standard multi-agent pipeline

It stores a normalized decision in ``state["bypass_decision"]`` with values
"ALL_A", "ALL_B", or "MIX". A raw textual rationale is saved to
``state["bypass_analyzer_output"]`` for auditability.
"""

from typing import Any, Dict
import os

from ..llm_base import get_backend, count_tokens
from ...utils.logger import logger

__all__ = ["conflict_analyzer_node"]


_PROMPT = (
    "You are a senior merge reviewer. Based on the following file-by-file summaries "
    "of changes for Parent A and Parent B, decide if we should: \n\n"
    "- All A   → choose Parent A for all files\n"
    "- All B   → choose Parent B for all files\n"
    "- Mix A/B → use a mix and perform per-file merging later\n\n"
    "Provide your judgement STRICTLY as one of these exact strings: \n"
    "All A, All B, Mix A/B.\n\n"
    "Consider overall risk, coherence, and whether one side clearly dominates across files.\n\n"
    "Summaries Parent A:\n{a_summary}\n\nSummaries Parent B:\n{b_summary}\n\n"
    "(Optional context) Diffs A:\n{a_diff}\n\nDiffs B:\n{b_diff}\n\n"
    "Answer:"
)


def _normalize_decision(text: str) -> str:
    t = (text or "").strip().lower()
    if "all a" in t:
        return "ALL_A"
    if "all b" in t:
        return "ALL_B"
    return "MIX"


def conflict_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Conflict analyzer started.")
    summaries: Dict[str, Dict[str, str]] = state.get("summaries", {}) or {}
    diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
    diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}

    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    encoder, llm = get_backend(model_name)

    # Build concatenated context across files to form a global judgement
    a_sum = "\n\n".join(f"{p}: {s.get('summary_a','')}" for p, s in summaries.items())
    b_sum = "\n\n".join(f"{p}: {s.get('summary_b','')}" for p, s in summaries.items())
    a_diff = "\n\n".join(f"{p}: {diffs_a.get(p, '')}" for p in summaries.keys())
    b_diff = "\n\n".join(f"{p}: {diffs_b.get(p, '')}" for p in summaries.keys())

    if llm is None:
        logger.warning("No LLM backend available for analyzer; using heuristic.")
        # Heuristic: if A and B summaries are very similar lengths overall, choose MIX; else prefer shorter
        len_a = len(a_sum)
        len_b = len(b_sum)
        if abs(len_a - len_b) < 0.05 * max(1, (len_a + len_b) // 2):
            decision = "MIX"
        else:
            decision = "ALL_A" if len_a <= len_b else "ALL_B"
        raw_output = f"Heuristic decision based on summary lengths (A={len_a}, B={len_b}): {decision}"
    else:
        prompt_text = _PROMPT.format(a_summary=a_sum, b_summary=b_sum, a_diff=a_diff, b_diff=b_diff)
        result = llm.invoke(prompt_text)
        content = result.content if hasattr(result, "content") else str(result)
        raw_output = content.strip()
        decision = _normalize_decision(raw_output)
        # Attribute prompt/output tokens to all files to keep accounting consistent with planner
        for path in summaries.keys():
            counts = state.setdefault("token_counts", {}).setdefault(
                path, {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0}
            )
            counts["system_prompt"] += count_tokens(encoder, _PROMPT)
            counts["output"] += count_tokens(encoder, raw_output)

    state["bypass_decision"] = decision
    state["bypass_analyzer_output"] = raw_output
    state["status"] = "analyzed"
    logger.info(f"Conflict analyzer decision: {decision}")
    return state


