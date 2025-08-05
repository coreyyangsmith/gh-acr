"""Resolution agent – calls an LLM to produce the merged content.

If the *plan* says 'A' or 'B' we can short-circuit without an LLM, otherwise we
ask the model to merge.
"""
from __future__ import annotations

from typing import Any, Dict
import os
from langchain_core.prompts import PromptTemplate

from ..llm_base import get_backend
from ...utils.logger import logger

__all__ = ["resolution_agent_node"]

_MERGE_PROMPT_STR = (
    "You are a merge-conflict resolver.  The file below has conflicting versions "
    "from two parents.  Produce the merged file that incorporates both sets of "
    "changes correctly.  Do NOT include any explanations—only the merged file.\n\n"\
    "File path: {file_path}\n\n"\
    "--- Parent A version ---\n{version_a}\n--- End A ---\n\n"\
    "--- Parent B version ---\n{version_b}\n--- End B ---\n\n"\
    "[Merged file below]"\
)

_prompt = PromptTemplate.from_template(_MERGE_PROMPT_STR)


def resolution_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Resolution agent started.")
    plan: Dict[str, str] = state["conflict_plan"]
    parent_a = state["parent_a_contents"]
    parent_b = state["parent_b_contents"]

    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    _, llm = get_backend(model_name)

    resolved: Dict[str, str] = {}

    for path, choice in plan.items():
        if choice in ("A", "B") or llm is None:
            if llm is None and choice not in ("A", "B"):
                logger.warning(f"No LLM backend available to merge {path}, falling back to parent A.")
                choice = "A"
            logger.info(f"Resolving conflict for {path} by selecting parent {choice}.")
            resolved[path] = parent_a.get(path, "") if choice == "A" else parent_b.get(path, "")
            continue
        
        logger.info(f"Resolving conflict for {path} using LLM.")
        runnable = _prompt | llm  # type: ignore[operator]
        result = runnable.invoke(
            {
                "file_path": path,
                "version_a": parent_a.get(path, ""),
                "version_b": parent_b.get(path, ""),
            }
        )
        content = result.content if hasattr(result, "content") else str(result)
        resolved[path] = content.strip("\n")

    state["resolved_contents"] = resolved
    state["status"] = "resolved_multi"
    logger.info("Resolution agent finished.")
    return state
