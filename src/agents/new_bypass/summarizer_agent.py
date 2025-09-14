from __future__ import annotations

from typing import Any, Dict
import json
import os
from pathlib import Path

from ..llm_base import get_backend, count_tokens
from ...utils.logger import logger

__all__ = ["summarizer_agent_node"]

_SUMMARY_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "new_bypass"
    / "summarizer_prompt.txt"
)
_SUMMARY_PROMPT_STR = _SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")


def _render_template(template: str, variables: Dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", value)
    return rendered


def _fallback_summary(diff_text: str) -> str:  # noqa: D401
    adds = sum(
        1
        for ln in diff_text.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    )
    dels = sum(
        1
        for ln in diff_text.splitlines()
        if ln.startswith("-") and not ln.startswith("---")
    )
    return f"Adds {adds} lines, removes {dels} lines."


def summarizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    logger.info("Summarizer agent started.")
    diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
    diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}
    ancestor_contents: Dict[str, str] = state.get("ancestor_contents", {}) or {}
    # Optional (clone mode): aggregated commit messages for A and B ranges
    commit_messages_a: str = str(state.get("commit_messages_a", "") or "")
    commit_messages_b: str = str(state.get("commit_messages_b", "") or "")

    model_name = state.get("model_name") or os.getenv(
        "OPENAI_MODEL", "openai/gpt-4o-mini"
    )
    encoder, llm = get_backend(model_name)

    summaries: Dict[str, Dict[str, str]] = {}

    files = (
        state.get("sample_row", {})
        .get("scenario_json", {})
        .get("files_in_merge_conflict", [])
    )
    file_iter = files or (set(diffs_a) | set(diffs_b))
    for path in file_iter:
        original_text = ancestor_contents.get(path, "")
        summary_pair: Dict[str, str] = {}
        logger.info(f"Summarizing changes for {path}.")

        for parent_label, diff_text in (
            ("A", diffs_a.get(path, "")),
            ("B", diffs_b.get(path, "")),
        ):
            if not diff_text:
                # Even if no changes, still provide a structured JSON with commit message context
                commit_text = commit_messages_a if parent_label == "A" else commit_messages_b
                commit_lines = commit_text.splitlines() if commit_text else []
                subject = commit_lines[0] if commit_lines else ""
                body = "\n".join(commit_lines[1:]) if len(commit_lines) > 1 else ""
                summary_pair[f"summary_{parent_label.lower()}"] = json.dumps(
                    {
                        "parent": parent_label,
                        "file_path": path,
                        "summary": "(no changes)",
                        "commit_message": {
                            "subject": subject,
                            "body": body,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                continue

            if llm is None:
                logger.warning(f"No LLM backend available, using heuristic for {path}.")
                commit_text = commit_messages_a if parent_label == "A" else commit_messages_b
                heuristic = _fallback_summary(diff_text)
                commit_lines = commit_text.splitlines() if commit_text else []
                subject = commit_lines[0] if commit_lines else ""
                body = "\n".join(commit_lines[1:]) if len(commit_lines) > 1 else ""
                summary_pair[f"summary_{parent_label.lower()}"] = json.dumps(
                    {
                        "parent": parent_label,
                        "file_path": path,
                        "summary": heuristic,
                        "commit_message": {
                            "subject": subject,
                            "body": body,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            else:
                prompt_text = _render_template(
                    _SUMMARY_PROMPT_STR,
                    {
                        "original_code": original_text,
                        "patch": diff_text,
                    },
                )
                # Append commit messages as additional context if available
                commit_text = commit_messages_a if parent_label == "A" else commit_messages_b
                if commit_text:
                    prompt_text = f"{prompt_text}\n\n[COMMIT_MESSAGES]\n{commit_text}"
                result = llm.invoke(prompt_text)
                content = result.content if hasattr(result, "content") else str(result)
                commit_lines = commit_text.splitlines() if commit_text else []
                subject = commit_lines[0] if commit_lines else ""
                body = "\n".join(commit_lines[1:]) if len(commit_lines) > 1 else ""
                summary_pair[f"summary_{parent_label.lower()}"] = json.dumps(
                    {
                        "parent": parent_label,
                        "file_path": path,
                        "summary": content.strip(),
                        "commit_message": {
                            "subject": subject,
                            "body": body,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                counts = state.setdefault("token_counts", {}).setdefault(
                    path,
                    {
                        "system_prompt": 0,
                        "original": 0,
                        "diff_a": 0,
                        "diff_b": 0,
                        "output": 0,
                    },
                )
                counts["system_prompt"] += count_tokens(encoder, _SUMMARY_PROMPT_STR)
                counts["original"] += count_tokens(encoder, original_text)
                if commit_text:
                    # Attribute commit message tokens to input as additional context
                    counts["original"] += count_tokens(encoder, commit_text)
                if parent_label == "A":
                    counts["diff_a"] += count_tokens(encoder, diff_text)
                else:
                    counts["diff_b"] += count_tokens(encoder, diff_text)
                counts["output"] += count_tokens(
                    encoder, summary_pair[f"summary_{parent_label.lower()}"]
                )

        summaries[path] = summary_pair

    state["summaries"] = summaries
    state["status"] = "summarised"
    logger.info("Summarizer agent finished.")
    return state


