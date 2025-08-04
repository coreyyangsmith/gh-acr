"""Summariser agent – describes how each parent diff changes the file using an LLM.

If an LLM backend is not available (e.g. missing API key) it falls back to a
crude line-count heuristic so the pipeline never crashes.
"""
from __future__ import annotations

from typing import Any, Dict
import os

from langchain_core.prompts import PromptTemplate

from ..llm_base import get_backend, count_tokens

__all__ = ["summarizer_agent_node"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT_STR = (
    "You are a senior software engineer. Analyse the diff below relative to the "
    "original file and provide a concise English summary of what the changes do.\n"\
    "\n"\
    "File path: {file_path}\n"\
    "--- ORIGINAL ---\n{original}\n--- END ORIGINAL ---\n\n"\
    "--- DIFF ---\n{diff}\n--- END DIFF ---\n\n"\
    "Summary:"\
)

_prompt = PromptTemplate.from_template(_SUMMARY_PROMPT_STR)

# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def _fallback_summary(diff_text: str) -> str:  # noqa: D401
    """Return a simple (non-LLM) summary based on line counts."""
    adds = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return f"Adds {adds} lines, removes {dels} lines."


def summarizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    diffs_a: Dict[str, str] = state["diffs_a"]
    diffs_b: Dict[str, str] = state["diffs_b"]
    ancestor_contents: Dict[str, str] = state["ancestor_contents"]

    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    encoder, llm = get_backend(model_name)

    summaries: Dict[str, Dict[str, str]] = {}

    for path in set(diffs_a) | set(diffs_b):
        original_text = ancestor_contents.get(path, "")
        summary_pair: Dict[str, str] = {}

        for parent_label, diff_text in (("A", diffs_a.get(path, "")), ("B", diffs_b.get(path, ""))):
            if not diff_text:
                summary_pair[f"summary_{parent_label.lower()}"] = "(no changes)"
                continue

            if llm is None:
                summary_pair[f"summary_{parent_label.lower()}"] = _fallback_summary(diff_text)
            else:
                prompt_vars = {
                    "file_path": path,
                    "original": original_text,
                    "diff": diff_text,
                }
                result = (_prompt | llm).invoke(prompt_vars)
                content = result.content if hasattr(result, "content") else str(result)
                summary_pair[f"summary_{parent_label.lower()}"] = content.strip()

            # token accounting (optional)
            token_stats = {
                "prompt_tokens": count_tokens(encoder, _SUMMARY_PROMPT_STR.format(**{k: "" for k in _prompt.input_variables})),
                "diff_tokens": count_tokens(encoder, diff_text),
                "output_tokens": count_tokens(encoder, summary_pair[f"summary_{parent_label.lower()}"]),
            }
            state.setdefault("token_counts", {}).setdefault(path, {}).setdefault(f"summary_{parent_label.lower()}", token_stats)

        summaries[path] = summary_pair

    state["summaries"] = summaries
    state["status"] = "summarised"
    return state
