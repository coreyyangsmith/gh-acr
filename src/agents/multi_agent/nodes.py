"""Unified node implementations for multi-agent merge resolution.

This module provides factory functions that create LangGraph nodes for the
multi-agent merge resolution pipeline. Each factory accepts a `prompt_variant`
parameter to specify which prompt templates to use.

Node Types
----------
- **summarizer**: Generates summaries describing how each parent's diff changes files
- **conflict_analyzer**: Makes a global judgement (All A / All B / Mix)
- **conflict_agent**: Creates a per-file merge plan based on summaries
- **resolution_agent**: Produces the actual merged file contents
- **review_agent**: Reviews merged output and provides feedback for iterations

All nodes follow the same pattern:
1. Read state and extract relevant data
2. Load the prompt template for the specified variant
3. Call the LLM (with fallback heuristics if unavailable)
4. Update state with results and token accounting
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Literal

from ..llm_base import get_backend, count_tokens
from ..utils import render_template, extract_text_content, scenario_file_list
from ...utils.logger import logger


def _log_node_start(node_name: str, variant: str, state: Dict[str, Any]) -> float:
    """Log node start with state diagnostics, return start time."""
    scenario_id = state.get("scenario_id", "unknown")
    file_count = len(state.get("diffs_a", {}) or {})
    logger.info(
        "[%s] Starting (variant=%s, scenario=%s, files=%d)",
        node_name, variant, scenario_id, file_count
    )
    return time.perf_counter()


def _log_node_end(node_name: str, start_time: float, state: Dict[str, Any]) -> None:
    """Log node completion with timing and diagnostics."""
    elapsed = time.perf_counter() - start_time
    status = state.get("status", "unknown")
    logger.info(
        "[%s] Completed in %.3fs (status=%s)",
        node_name, elapsed, status
    )
    
    # Log memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(
            "[%s] Memory: rss=%.1fMB, vms=%.1fMB",
            node_name, mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024
        )
    except Exception:
        pass


# Type alias for prompt variants
PromptVariant = Literal["bypass7", "force_mix"]

# Type alias for node functions
NodeFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def _get_prompts_dir() -> Path:
    """Return the base prompts directory path."""
    return Path(__file__).resolve().parents[2] / "prompts"


def _load_prompt(variant: PromptVariant, prompt_name: str) -> str:
    """Load a prompt template for the given variant.

    Parameters
    ----------
    variant
        The prompt variant directory name (e.g., "bypass7")
    prompt_name
        The prompt file name (e.g., "summarizer_prompt.txt")

    Returns
    -------
    str
        The prompt template content.

    Raises
    ------
    FileNotFoundError
        If the prompt file doesn't exist.
    """
    prompt_path = _get_prompts_dir() / variant / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _get_model_name(state: Dict[str, Any]) -> str:
    """Extract the model name from state or environment."""
    return state.get("model_name") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")


def _init_token_counts(state: Dict[str, Any], path: str) -> Dict[str, int]:
    """Initialize or retrieve token count dict for a file path."""
    return state.setdefault("token_counts", {}).setdefault(
        path, {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0}
    )


# =============================================================================
# Summarizer Node
# =============================================================================

def _fallback_summary(diff_text: str) -> str:
    """Return a simple (non-LLM) summary based on line counts."""
    adds = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return f"Adds {adds} lines, removes {dels} lines."


def create_summarizer_node(prompt_variant: PromptVariant) -> NodeFunc:
    """Create a summarizer node that describes how each parent diff changes files.

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use (currently "bypass7")

    Returns
    -------
    NodeFunc
        A LangGraph node function that summarizes diffs.
    """
    # Load prompt at creation time for efficiency
    prompt_str = _load_prompt(prompt_variant, "summarizer_prompt.txt")

    def summarizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summaries describing how each parent's diff modifies files."""
        node_start = _log_node_start("summarizer_agent", prompt_variant, state)

        diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
        diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}
        ancestor_contents: Dict[str, str] = state.get("ancestor_contents", {}) or {}

        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        summaries: Dict[str, Dict[str, str]] = {}
        files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
        file_iter = files or (set(diffs_a) | set(diffs_b))

        for path in file_iter:
            original_text = ancestor_contents.get(path, "")
            summary_pair: Dict[str, str] = {}
            logger.info("Summarizing changes for %s.", path)

            for parent_label, diff_text in (("A", diffs_a.get(path, "")), ("B", diffs_b.get(path, ""))):
                if not diff_text:
                    summary_pair[f"summary_{parent_label.lower()}"] = "(no changes)"
                    continue

                if llm is None:
                    logger.warning("No LLM backend available, using heuristic for %s.", path)
                    summary_pair[f"summary_{parent_label.lower()}"] = _fallback_summary(diff_text)
                else:
                    prompt_text = render_template(
                        prompt_str,
                        {"original_code": original_text, "patch": diff_text},
                    )
                    result = llm.invoke(prompt_text)
                    content = extract_text_content(result)
                    summary_pair[f"summary_{parent_label.lower()}"] = content.strip()

                    # Track token usage
                    counts = _init_token_counts(state, path)
                    counts["system_prompt"] += count_tokens(encoder, prompt_str)
                    counts["original"] += count_tokens(encoder, original_text)
                    if parent_label == "A":
                        counts["diff_a"] += count_tokens(encoder, diff_text)
                    else:
                        counts["diff_b"] += count_tokens(encoder, diff_text)
                    counts["output"] += count_tokens(encoder, summary_pair[f"summary_{parent_label.lower()}"])

            summaries[path] = summary_pair

        state["summaries"] = summaries
        state["status"] = "summarised"
        _log_node_end("summarizer_agent", node_start, state)
        return state

    return summarizer_agent_node


# =============================================================================
# Conflict Analyzer Node
# =============================================================================

def _normalize_decision_standard(text: str) -> str:
    """Normalize decision text to ALL_A, ALL_B, or MIX."""
    t = (text or "").strip().lower()
    if "a" in t:
        return "ALL_A"
    if "b" in t:
        return "ALL_B"
    return "MIX"


def create_conflict_analyzer_node(prompt_variant: PromptVariant) -> NodeFunc:
    """Create a conflict analyzer node for global judgement (All A / All B / Mix).

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use

    Returns
    -------
    NodeFunc
        A LangGraph node function that analyzes conflicts.
    """
    prompt_str = _load_prompt(prompt_variant, "conflict_judge_prompt.txt")

    def conflict_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze summaries and diffs to make a global resolution decision."""
        node_start = _log_node_start("conflict_analyzer", prompt_variant, state)

        summaries: Dict[str, Dict[str, str]] = state.get("summaries", {}) or {}
        diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
        diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}

        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        # Build concatenated context for global judgement
        a_sum = "\n\n".join(f"{p}: {s.get('summary_a', '')}" for p, s in summaries.items())
        b_sum = "\n\n".join(f"{p}: {s.get('summary_b', '')}" for p, s in summaries.items())
        a_diff = "\n\n".join(f"{p}: {diffs_a.get(p, '')}" for p in summaries.keys())
        b_diff = "\n\n".join(f"{p}: {diffs_b.get(p, '')}" for p in summaries.keys())

        if llm is None:
            logger.warning("No LLM backend available for analyzer; using heuristic.")
            len_a, len_b = len(a_sum), len(b_sum)
            if abs(len_a - len_b) < 0.05 * max(1, (len_a + len_b) // 2):
                decision = "MIX"
            else:
                decision = "ALL_A" if len_a <= len_b else "ALL_B"
            raw_output = f"Heuristic decision based on summary lengths (A={len_a}, B={len_b}): {decision}"
        else:
            prompt_text = prompt_str.format(
                a_summary=a_sum, b_summary=b_sum, a_diff=a_diff, b_diff=b_diff
            )

            result = llm.invoke(prompt_text)
            content = extract_text_content(result)
            raw_output = content.strip()
            decision = _normalize_decision_standard(raw_output)
            for path in summaries.keys():
                counts = _init_token_counts(state, path)
                counts["system_prompt"] += count_tokens(encoder, prompt_text)
                counts["output"] += count_tokens(encoder, raw_output)

        state["bypass_decision"] = decision
        state["bypass_method"] = "A" if decision == "ALL_A" else ("B" if decision == "ALL_B" else "MIX")
        state["bypass_analyzer_output"] = raw_output
        state["status"] = "analyzed"
        logger.info("Conflict analyzer decision: %s", decision)
        _log_node_end("conflict_analyzer", node_start, state)
        return state

    return conflict_analyzer_node


# =============================================================================
# Conflict Agent (Planner) Node
# =============================================================================

def create_conflict_agent_node(prompt_variant: PromptVariant) -> NodeFunc:
    """Create a conflict agent node that generates merge plans.

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use

    Returns
    -------
    NodeFunc
        A LangGraph node function that creates merge plans.
    """
    prompt_str = _load_prompt(prompt_variant, "plan_prompt.txt")

    def conflict_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a per-file merge plan based on summaries."""
        node_start = _log_node_start("conflict_agent", prompt_variant, state)

        summaries: Dict[str, Dict[str, str]] = state.get("summaries", {}) or {}
        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        if llm is None:
            logger.warning("No LLM backend available, falling back to heuristic.")
            plan = {}
            for path, s in summaries.items():
                a_len = len(s.get("summary_a", ""))
                b_len = len(s.get("summary_b", ""))
                plan[path] = "A" if a_len <= b_len else "B"
        else:
            logger.info("Generating merge plan using LLM.")
            a_sum = "\n\n".join(f"{p}: {s.get('summary_a', '')}" for p, s in summaries.items())
            b_sum = "\n\n".join(f"{p}: {s.get('summary_b', '')}" for p, s in summaries.items())
            a_diff = "\n\n".join(f"{p}: {state.get('diffs_a', {}).get(p, '')}" for p in summaries.keys())
            b_diff = "\n\n".join(f"{p}: {state.get('diffs_b', {}).get(p, '')}" for p in summaries.keys())

            prompt_text = render_template(
                prompt_str,
                {"a_diff": a_diff, "a_summary": a_sum, "b_diff": b_diff, "b_summary": b_sum},
            )
            result = llm.invoke(prompt_text)
            content = extract_text_content(result)
            try:
                plan = json.loads(content)
            except Exception:
                logger.error("Failed to parse LLM output as JSON, falling back to merge.")
                plan = {p: "merge" for p in summaries}

            for path in summaries.keys():
                counts = _init_token_counts(state, path)
                counts["system_prompt"] += count_tokens(encoder, prompt_str)
                counts["output"] += count_tokens(encoder, content)

        # Ensure plan contains all files
        if not plan:
            files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
            plan = {p: "merge" for p in files}

        state["conflict_plan"] = plan
        state["status"] = "planned"
        logger.info("Conflict agent finished. Plan: %s", plan)
        _log_node_end("conflict_agent", node_start, state)
        return state

    return conflict_agent_node


# =============================================================================
# Resolution Agent Node
# =============================================================================

def create_resolution_agent_node(prompt_variant: PromptVariant) -> NodeFunc:
    """Create a resolution agent node that produces merged file contents.

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use

    Returns
    -------
    NodeFunc
        A LangGraph node function that resolves conflicts.
    """
    prompt_str = _load_prompt(prompt_variant, "resolver_prompt.txt")

    def resolution_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Produce merged file contents based on the plan."""
        node_start = _log_node_start("resolution_agent", prompt_variant, state)

        plan: Dict[str, str] = state.get("conflict_plan", {}) or {}
        parent_a = state.get("parent_a_contents", {}) or {}
        parent_b = state.get("parent_b_contents", {}) or {}
        ancestor_contents = state.get("ancestor_contents", {})
        diffs_a = state.get("diffs_a", {})
        diffs_b = state.get("diffs_b", {})

        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        resolved: Dict[str, str] = {}
        final_diffs: Dict[str, str] = {}
        resolution_history: Dict[str, list] = state.get("resolution_history", {}) or {}

        scenario_files = scenario_file_list(
            state,
            fallback_paths=list(parent_a.keys()) + list(parent_b.keys()) + list(diffs_a.keys()) + list(diffs_b.keys()),
        )

        for path in scenario_files:
            choice = plan.get(path, "merge")
            counts = _init_token_counts(state, path)
            counts["system_prompt"] += count_tokens(encoder, prompt_str)
            counts["original"] += count_tokens(encoder, ancestor_contents.get(path, ""))
            counts["diff_a"] += count_tokens(encoder, diffs_a.get(path, ""))
            counts["diff_b"] += count_tokens(encoder, diffs_b.get(path, ""))

            if choice in ("A", "B") or llm is None:
                if llm is None and choice not in ("A", "B"):
                    logger.warning("No LLM backend available to merge %s, falling back to parent A.", path)
                    choice = "A"
                logger.info("Resolving conflict for %s by selecting parent %s.", path, choice)
                merged_text = parent_a.get(path, "") if choice == "A" else parent_b.get(path, "")
                resolved[path] = merged_text
                resolution_history.setdefault(path, []).append(merged_text)
                counts["output"] += count_tokens(encoder, merged_text)
                continue

            logger.info("Resolving conflict for %s using LLM.", path)
            single_plan = {path: plan.get(path, "merge")}
            feedback_map = state.get("review_feedback", {}) or {}
            feedback_text = str(feedback_map.get(path, "")).strip()

            if feedback_text:
                logger.info("Applying review feedback for %s (length=%d chars)", path, len(feedback_text))

            prompt_text = render_template(
                prompt_str,
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
            resolution_history.setdefault(path, []).append(merged_text)
            counts["output"] += count_tokens(encoder, merged_text)

            # Compute unified diff vs original
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
        state["resolution_history"] = resolution_history
        state["status"] = "resolved_multi"
        _log_node_end("resolution_agent", node_start, state)
        return state

    return resolution_agent_node


# =============================================================================
# Review Agent Node
# =============================================================================

def create_review_agent_node(prompt_variant: PromptVariant) -> NodeFunc:
    """Create a review agent node that provides feedback on merged output.

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use

    Returns
    -------
    NodeFunc
        A LangGraph node function that reviews resolutions.
    """
    prompt_str = _load_prompt(prompt_variant, "review_prompt.txt")

    def review_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Review merged output and provide feedback for potential iterations."""
        node_start = _log_node_start("review_agent", prompt_variant, state)

        resolved: Dict[str, str] = state.get("resolved_contents", {}) or {}
        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        reviews: Dict[str, str] = {}
        review_results: Dict[str, Dict[str, str]] = {}
        review_history: Dict[str, list] = state.get("review_history", {}) or {}

        files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
        file_iter = files or list(resolved.keys())

        for path in file_iter:
            content = resolved.get(path, "")
            logger.info("Reviewing %s.", path)

            if llm is None:
                reviews[path] = "ACCEPT – heuristic stub (no LLM)."
                review_results[path] = {"outcome": "ACCEPT", "rationale": ""}
                logger.warning("No LLM backend available, using heuristic for %s.", path)
                continue

            prompt_text = render_template(prompt_str, {"generated_code": content})
            res = llm.invoke(prompt_text)
            text = extract_text_content(res)
            reviews[path] = text
            review_history.setdefault(path, []).append(text)

            # Parse structured outcome
            outcome = "REJECT"
            rationale = ""
            try:
                data = json.loads(text)
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

            counts = _init_token_counts(state, path)
            counts["system_prompt"] += count_tokens(encoder, prompt_str)
            counts["output"] += count_tokens(encoder, text)

        state["reviews"] = reviews
        state["review_results"] = review_results
        state["review_history"] = review_history
        state["status"] = "reviewed"
        _log_node_end("review_agent", node_start, state)
        return state

    return review_agent_node


__all__ = [
    "create_summarizer_node",
    "create_conflict_analyzer_node",
    "create_conflict_agent_node",
    "create_resolution_agent_node",
    "create_review_agent_node",
    "PromptVariant",
    "NodeFunc",
]

