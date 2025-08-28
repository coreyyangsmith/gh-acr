"""Resolution agent – calls an LLM to produce the merged content.

If the *plan* says 'A' or 'B' we can short-circuit without an LLM, otherwise we
ask the model to merge.
"""
from __future__ import annotations

from typing import Any, Dict
import os
from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ..utils import render_template, extract_text_content, scenario_file_list
from ...utils.logger import logger

__all__ = ["resolution_agent_node"]

_MERGE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "bypass4" / "resolver_prompt.txt"
_MERGE_PROMPT_STR = _MERGE_PROMPT_PATH.read_text(encoding="utf-8")

def _render_template(template: str, variables: Dict[str, str]) -> str:
    return render_template(template, variables)


def resolution_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Resolution agent started.")
    plan: Dict[str, str] = state.get("conflict_plan", {}) or {}
    parent_a = state.get("parent_a_contents", {}) or {}
    parent_b = state.get("parent_b_contents", {}) or {}

    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    encoder, llm = get_backend(model_name)

    resolved: Dict[str, str] = {}
    final_diffs: Dict[str, str] = {}

    ancestor_contents = state.get("ancestor_contents", {})
    diffs_a = state.get("diffs_a", {})
    diffs_b = state.get("diffs_b", {})

    scenario_files = scenario_file_list(state, fallback_paths=list(parent_a.keys()) + list(parent_b.keys()) + list(diffs_a.keys()) + list(diffs_b.keys()))

    for path in scenario_files:
        choice = plan.get(path, "merge")
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
            counts["output"] += count_tokens(encoder, merged_text)
            continue
        import json
        single_plan = {path: plan.get(path, "merge")}
        feedback_map = state.get("review_feedback", {}) or {}
        feedback_text = str(feedback_map.get(path, "")).strip()
        prompt_text = _render_template(
            _MERGE_PROMPT_STR,
            {
                "plan": json.dumps(single_plan, ensure_ascii=False),
                "original_code": ancestor_contents.get(path, ""),
                "patch_a": diffs_a.get(path, ""),
                "patch_b": diffs_b.get(path, ""),
                "review_feedback": feedback_text,
            },
        )
        result = llm.invoke(prompt_text)
        content = extract_text_content(result)
        merged_text = content.strip("\n")
        resolved[path] = merged_text
        counts["output"] += count_tokens(encoder, merged_text)
        try:
            import difflib
            a_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
            m_lines = merged_text.splitlines(keepends=True)
            final_diffs[path] = "".join(
                difflib.unified_diff(a_lines, m_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
            )
        except Exception:
            final_diffs[path] = ""

    state["resolved_contents"] = resolved
    state["final_diffs"] = final_diffs
    state["status"] = "resolved_multi"
    logger.info("Resolution agent finished.")
    return state


