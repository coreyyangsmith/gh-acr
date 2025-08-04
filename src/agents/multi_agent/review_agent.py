"""Review agent – uses an LLM to provide feedback on the merged patch."""
from __future__ import annotations

from typing import Any, Dict
import os
from langchain_core.prompts import PromptTemplate

from ..llm_base import get_backend

__all__ = ["review_agent_node"]

_REVIEW_PROMPT_STR = (
    "You are a senior code reviewer.  Review the merged file below for correctness "
    "and style.  Provide a short verdict (ACCEPT / REJECT) followed by one bullet-point "
    "summary.\n\n"\
    "File path: {file_path}\n\n"\
    "--- MERGED CONTENT ---\n{merged}\n--- END ---\n\nReview:"\
)

_prompt = PromptTemplate.from_template(_REVIEW_PROMPT_STR)


def review_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    resolved: Dict[str, str] = state["resolved_contents"]
    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    _, llm = get_backend(model_name)

    reviews: Dict[str, str] = {}

    for path, content in resolved.items():
        if llm is None:
            reviews[path] = "ACCEPT – heuristic stub (no LLM)."
            continue
        res = (_prompt | llm).invoke({"file_path": path, "merged": content})
        reviews[path] = res.content if hasattr(res, "content") else str(res)

    state["reviews"] = reviews
    state["status"] = "reviewed"
    return state
