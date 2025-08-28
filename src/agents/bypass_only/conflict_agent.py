"""Conflict agent – asks an LLM to produce a merge plan given summaries."""
from __future__ import annotations

from typing import Any, Dict
import os
from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ..utils import render_template, extract_text_content
from ...utils.logger import logger

__all__ = ["conflict_agent_node"]

_PLAN_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "bypass_only" / "plan_prompt.txt"
_PLAN_PROMPT_STR = _PLAN_PROMPT_PATH.read_text(encoding="utf-8")

def _render_template(template: str, variables: Dict[str, str]) -> str:
    return render_template(template, variables)


def conflict_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Conflict agent started.")
    summaries: Dict[str, Dict[str, str]] = state.get("summaries", {}) or {}
    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    encoder, llm = get_backend(model_name)

    if llm is None:
        logger.warning("No LLM backend available, falling back to heuristic.")
        plan = {}
        for path, s in summaries.items():
            a_len = len(s.get("summary_a", ""))
            b_len = len(s.get("summary_b", ""))
            plan[path] = "A" if a_len <= b_len else "B"
    else:
        import json
        a_sum = "\n\n".join(f"{p}: {s.get('summary_a','')}" for p, s in summaries.items())
        b_sum = "\n\n".join(f"{p}: {s.get('summary_b','')}" for p, s in summaries.items())
        a_diff = "\n\n".join(f"{p}: {state.get('diffs_a', {}).get(p, '')}" for p in summaries.keys())
        b_diff = "\n\n".join(f"{p}: {state.get('diffs_b', {}).get(p, '')}" for p in summaries.keys())
        prompt_text = _render_template(
            _PLAN_PROMPT_STR,
            {"a_diff": a_diff, "a_summary": a_sum, "b_diff": b_diff, "b_summary": b_sum},
        )
        result = llm.invoke(prompt_text)
        content = extract_text_content(result)
        try:
            plan = json.loads(content)
        except Exception:
            logger.error("Failed to parse LLM output as JSON; defaulting to merge.")
            plan = {p: "merge" for p in summaries}
        for path in summaries.keys():
            counts = state.setdefault("token_counts", {}).setdefault(
                path, {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0}
            )
            counts["system_prompt"] += count_tokens(encoder, _PLAN_PROMPT_STR)
            counts["output"] += count_tokens(encoder, content)

    if not plan:
        files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
        plan = {p: "merge" for p in files}
    state["conflict_plan"] = plan
    state["status"] = "planned"
    logger.info("Conflict agent finished.")
    return state


