"""Resolution agent – calls an LLM to produce the merged content.

If the *plan* says 'A' or 'B' we can short-circuit without an LLM, otherwise we
ask the model to merge.
"""
from __future__ import annotations

from typing import Any, Dict
import os
from langchain_core.prompts import PromptTemplate
from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ...utils.logger import logger

__all__ = ["resolution_agent_node"]

_MERGE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "multi" / "resolver_prompt.txt"
_MERGE_PROMPT_STR = _MERGE_PROMPT_PATH.read_text(encoding="utf-8")

_prompt = PromptTemplate.from_template(_MERGE_PROMPT_STR)


def resolution_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Resolution agent started.")
    plan: Dict[str, str] = state["conflict_plan"]
    parent_a = state["parent_a_contents"]
    parent_b = state["parent_b_contents"]

    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    encoder, llm = get_backend(model_name)

    resolved: Dict[str, str] = {}

    ancestor_contents = state.get("ancestor_contents", {})
    diffs_a = state.get("diffs_a", {})
    diffs_b = state.get("diffs_b", {})

    for path, choice in plan.items():
        # Precompute token usage comparable to simple agent for cost accounting (accumulate)
        counts = state.setdefault("token_counts", {}).setdefault(
            path, {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0}
        )
        counts["system_prompt"] += count_tokens(encoder, _MERGE_PROMPT_STR)
        counts["original"] += count_tokens(encoder, ancestor_contents.get(path, ""))
        counts["diff_a"] += count_tokens(encoder, diffs_a.get(path, ""))
        counts["diff_b"] += count_tokens(encoder, diffs_b.get(path, ""))
        if choice in ("A", "B") or llm is None:
            if llm is None and choice not in ("A", "B"):
                logger.warning(f"No LLM backend available to merge {path}, falling back to parent A.")
                choice = "A"
            logger.info(f"Resolving conflict for {path} by selecting parent {choice}.")
            merged_text = parent_a.get(path, "") if choice == "A" else parent_b.get(path, "")
            resolved[path] = merged_text
            # Record output tokens accumulatively
            counts["output"] += count_tokens(encoder, merged_text)
            continue
        
        logger.info(f"Resolving conflict for {path} using LLM.")
        runnable = _prompt | llm  # type: ignore[operator]
        # New resolver prompt expects a JSON plan and original_code/patches.
        # For now, pass the single-file slice of the plan plus parents as patches.
        single_plan = {path: plan.get(path, "merge")}
        result = runnable.invoke(
            {
                "plan": single_plan,
                "original_code": ancestor_contents.get(path, ""),
                "patch_a": diffs_a.get(path, ""),
                "patch_b": diffs_b.get(path, ""),
            }
        )
        content = result.content if hasattr(result, "content") else str(result)
        merged_text = content.strip("\n")
        resolved[path] = merged_text
        counts["output"] += count_tokens(encoder, merged_text)

    state["resolved_contents"] = resolved
    state["status"] = "resolved_multi"
    logger.info("Resolution agent finished.")
    return state
