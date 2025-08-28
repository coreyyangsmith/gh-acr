"""Summariser agent – describes how each parent diff changes the file using an LLM.

If an LLM backend is not available (e.g. missing API key) it falls back to a
crude line-count heuristic so the pipeline never crashes.
"""
from __future__ import annotations

from typing import Any, Dict
import os

from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ...utils.logger import logger

__all__ = ["summarizer_agent_node"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "bypass3" / "summarizer_prompt.txt"
_SUMMARY_PROMPT_STR = _SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")

def _render_template(template: str, variables: Dict[str, str]) -> str:
    """Render a very small subset of the template by replacing {{ var }} tokens.

    We intentionally avoid external templating engines; only direct replacements
    of the exact pattern with single spaces are supported.
    """
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", value)
    return rendered

# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def _fallback_summary(diff_text: str) -> str:  # noqa: D401
    """Return a simple (non-LLM) summary based on line counts."""
    adds = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return f"Adds {adds} lines, removes {dels} lines."


def summarizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Summarizer agent started.")
    diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
    diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}
    ancestor_contents: Dict[str, str] = state.get("ancestor_contents", {}) or {}

    model_name = state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    encoder, llm = get_backend(model_name)

    summaries: Dict[str, Dict[str, str]] = {}

    files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
    file_iter = files or (set(diffs_a) | set(diffs_b))
    for path in file_iter:
        original_text = ancestor_contents.get(path, "")
        summary_pair: Dict[str, str] = {}
        logger.info(f"Summarizing changes for {path}.")

        for parent_label, diff_text in (("A", diffs_a.get(path, "")), ("B", diffs_b.get(path, ""))):
            if not diff_text:
                summary_pair[f"summary_{parent_label.lower()}"] = "(no changes)"
                continue

            if llm is None:
                logger.warning(f"No LLM backend available, using heuristic for {path}.")
                summary_pair[f"summary_{parent_label.lower()}"] = _fallback_summary(diff_text)
            else:
                prompt_text = _render_template(
                    _SUMMARY_PROMPT_STR,
                    {
                        "original_code": original_text,
                        "patch": diff_text,
                    },
                )
                result = llm.invoke(prompt_text)
                content = result.content if hasattr(result, "content") else str(result)
                summary_pair[f"summary_{parent_label.lower()}"] = content.strip()
                # Track token usage per file/parent label for overall accounting (accumulate)
                counts = state.setdefault("token_counts", {}).setdefault(
                    path, {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0}
                )
                counts["system_prompt"] += count_tokens(encoder, _SUMMARY_PROMPT_STR)
                counts["original"] += count_tokens(encoder, original_text)
                if parent_label == "A":
                    counts["diff_a"] += count_tokens(encoder, diff_text)
                else:
                    counts["diff_b"] += count_tokens(encoder, diff_text)
                counts["output"] += count_tokens(encoder, summary_pair[f"summary_{parent_label.lower()}"])

        summaries[path] = summary_pair

    state["summaries"] = summaries
    state["status"] = "summarised"
    logger.info("Summarizer agent finished.")
    return state


