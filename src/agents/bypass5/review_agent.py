"""Review agent – uses an LLM to provide feedback on the merged patch."""

from __future__ import annotations

from typing import Any, Dict
import os
from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ..utils import render_template, extract_text_content
from ...utils.logger import logger

__all__ = ["review_agent_node"]

_REVIEW_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "bypass5" / "review_prompt.txt"
)
_REVIEW_PROMPT_STR = _REVIEW_PROMPT_PATH.read_text(encoding="utf-8")


def _render_template(template: str, variables: Dict[str, str]) -> str:
    return render_template(template, variables)


def review_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Review agent started.")
    resolved: Dict[str, str] = state.get("resolved_contents", {}) or {}
    model_name = state.get("model_name") or os.getenv(
        "OPENAI_MODEL", "openai/gpt-4o-mini"
    )
    encoder, llm = get_backend(model_name)

    import json

    reviews: Dict[str, str] = {}
    review_results: Dict[str, Dict[str, str]] = {}

    # Ensure we iterate all scenario files even if resolved map is sparse
    files = (
        state.get("sample_row", {})
        .get("scenario_json", {})
        .get("files_in_merge_conflict", [])
    )
    file_iter = files or list(resolved.keys())
    for path in file_iter:
        content = resolved.get(path, "")
        logger.info(f"Reviewing {path}.")
        if llm is None:
            reviews[path] = "ACCEPT – heuristic stub (no LLM)."
            review_results[path] = {"outcome": "ACCEPT", "rationale": ""}
            logger.warning(f"No LLM backend available, using heuristic for {path}.")
            continue
        prompt_text = _render_template(_REVIEW_PROMPT_STR, {"generated_code": content})
        res = llm.invoke(prompt_text)
        text = extract_text_content(res)
        reviews[path] = text
        # Parse structured outcome for control flow
        outcome = "REJECT"
        rationale = ""
        try:
            data = json.loads(text)
            # Support either object with keys or array containing one object
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                raw_outcome = str(data.get("outcome", "")).strip().upper()
                if raw_outcome in {"ACCEPT", "REJECT"}:
                    outcome = raw_outcome
                rationale = str(data.get("rationale", "")).strip()
        except Exception:
            rationale = text.strip()
        review_results[path] = {"outcome": outcome, "rationale": rationale}
        # Token accounting: accumulate review prompt and output tokens
        counts = state.setdefault("token_counts", {}).setdefault(
            path,
            {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0},
        )
        counts["system_prompt"] += count_tokens(encoder, _REVIEW_PROMPT_STR)
        counts["output"] += count_tokens(encoder, text)

    state["reviews"] = reviews
    state["review_results"] = review_results
    state["status"] = "reviewed"
    logger.info("Review agent finished.")
    return state
