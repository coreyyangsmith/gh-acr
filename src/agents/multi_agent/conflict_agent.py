"""Conflict agent – asks an LLM to produce a merge *plan* given summaries.

Fallback behaviour (when no backend) keeps the previous heuristic.
"""
from __future__ import annotations

from typing import Any, Dict
import os
from langchain_core.prompts import PromptTemplate

from ..llm_base import get_backend
from ...utils.logger import logger

__all__ = ["conflict_agent_node"]

_PLAN_PROMPT_STR = (
    "You are an experienced software engineer tasked with planning how to merge two "
    "sets of changes.  For each file, you are given concise summaries of what Parent A "
    "and Parent B did.  Decide which parent's changes should dominate, or output 'merge' "
    "if both should be combined manually.  Return a JSON object mapping file paths to "
    "one of 'A', 'B', or 'merge'.\n\n"\
    "{summaries}\n\nPlan:"\
)

_prompt = PromptTemplate.from_template(_PLAN_PROMPT_STR)


def conflict_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Conflict agent started.")
    summaries: Dict[str, Dict[str, str]] = state["summaries"]
    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    _, llm = get_backend(model_name)

    if llm is None:
        logger.warning("No LLM backend available, falling back to heuristic.")
        # --- fallback heuristic based on summary length ----------------------------------
        plan = {}
        for path, s in summaries.items():
            a_len = len(s["summary_a"])
            b_len = len(s["summary_b"])
            plan[path] = "A" if a_len <= b_len else "B"
    else:
        import json
        logger.info("Generating merge plan using LLM.")
        formatted = "\n".join(f"{p}:\n  - A: {s['summary_a']}\n  - B: {s['summary_b']}" for p, s in summaries.items())
        result = (_prompt | llm).invoke({"summaries": formatted})
        content = result.content if hasattr(result, "content") else str(result)
        try:
            plan = json.loads(content)
        except Exception:
            logger.error("Failed to parse LLM output as JSON, falling back to heuristic.")
            # Fallback to heuristic if parsing fails
            plan = {p: "merge" for p in summaries}

    logger.info(f"Generated merge plan: {plan}")
    state["conflict_plan"] = plan
    state["status"] = "planned"
    logger.info("Conflict agent finished.")
    return state
