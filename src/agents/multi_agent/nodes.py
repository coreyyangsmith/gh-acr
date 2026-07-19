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

from ..artifact_io import (
    agent_call_dir,
    base_metadata,
    file_path_to_slug,
    get_artifact_root,
    write_agent_call,
)
from ..llm_base import get_backend, count_tokens
from ..parse_utils import (
    extract_analyzer_verdict,
    parse_plan_json,
    parse_review_outcome,
)
from ..prompt_budget import (
    EvidenceBlock,
    FitReport,
    fit_global_ab_prompt,
    fit_variable_blocks,
)
from ..resilient_invoke import (
    ParseExhausted,
    ParsedResult,
    invoke_and_parse,
    resilient_invoke,
)
from ..utils import extract_text_content, scenario_file_list
from ...utils.degradation import record_degradation
from ...utils.logger import logger
from ...utils.run_progress import get_active_progress, get_current_worker_id, set_stage


# Map internal node names to short progress stage labels
_NODE_STAGE_NAMES: dict[str, str] = {
    "summarizer_agent": "summarise",
    "conflict_analyzer": "analyze",
    "conflict_agent": "plan",
    "resolution_agent": "patch",
    "review_agent": "review",
}


def _log_node_start(node_name: str, variant: str, state: Dict[str, Any]) -> float:
    """Log node start with state diagnostics, return start time."""
    scenario_id = state.get("scenario_id", "unknown")
    file_count = len(state.get("diffs_a", {}) or {})
    stage = _NODE_STAGE_NAMES.get(node_name, node_name)
    set_stage(stage, detail=f"{file_count} files")
    logger.debug(
        "[%s] Starting (variant=%s, scenario=%s, files=%d)",
        node_name, variant, scenario_id, file_count
    )
    return time.perf_counter()


def _log_node_end(node_name: str, start_time: float, state: Dict[str, Any]) -> None:
    """Log node completion with timing and diagnostics."""
    elapsed = time.perf_counter() - start_time
    status = state.get("status", "unknown")
    scenario_id = state.get("scenario_id", "unknown")
    stage = _NODE_STAGE_NAMES.get(node_name, node_name)
    wid = get_current_worker_id()
    prog = get_active_progress()
    if wid and prog is not None:
        prog.mark_stage_done(wid, stage, elapsed_s=elapsed)
    else:
        logger.info(
            "[%s] scenario=%s Completed in %.3fs (status=%s)",
            node_name, scenario_id, elapsed, status
        )

    # Log memory usage if psutil is available (debug only — noisy under concurrency)
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug(
            "[%s] Memory: rss=%.1fMB, vms=%.1fMB",
            node_name, mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024
        )
    except Exception:
        pass


# Type alias for prompt variants
PromptVariant = Literal["bypass7", "better_judge", "force_mix"]

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


def _invoke_context(
    state: Dict[str, Any],
    *,
    node: str,
    agent: str,
    file_path: str | None = None,
    call_id: str | None = None,
) -> Dict[str, Any]:
    """Build shared metadata for resilient_invoke failure traces and artifacts."""
    return {
        "scenario_id": state.get("scenario_id"),
        "eval_method": state.get("eval_method"),
        "node": node,
        "agent": agent,
        "file_path": file_path,
        "call_id": call_id,
        "model_name": _get_model_name(state),
    }


def _init_token_counts(state: Dict[str, Any], path: str) -> Dict[str, int]:
    """Initialize or retrieve token count dict for a file path."""
    return state.setdefault("token_counts", {}).setdefault(
        path, {"system_prompt": 0, "original": 0, "diff_a": 0, "diff_b": 0, "output": 0}
    )


def _with_budget_artifacts(
    supporting: Dict[str, str] | None,
    report: FitReport | None,
) -> Dict[str, str]:
    """Copy supporting artifacts and attach structured budget metadata when clipped."""
    artifacts = dict(supporting or {})
    if report is not None and report.was_clipped:
        artifacts["truncation_report.json"] = report.artifact_json()
    return artifacts


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
        artifact_root = get_artifact_root(state)

        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        summaries: Dict[str, Dict[str, str]] = {}
        files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
        file_iter = files or (set(diffs_a) | set(diffs_b))

        for path in file_iter:
            original_text = ancestor_contents.get(path, "")
            summary_pair: Dict[str, str] = {}
            file_slug = file_path_to_slug(path)
            logger.debug("Summarizing changes for %s.", path)

            for parent_label, diff_text in (("A", diffs_a.get(path, "")), ("B", diffs_b.get(path, ""))):
                call_id = parent_label.lower()
                call_dir = agent_call_dir(
                    artifact_root, agent="summarizer", file_slug=file_slug, call_id=call_id
                )
                supporting = {
                    "original.txt": original_text,
                    f"{call_id}.diff": diff_text or "",
                }
                if not diff_text:
                    summary_pair[f"summary_{call_id}"] = "(no changes)"
                    write_agent_call(
                        call_dir,
                        input_text="",
                        output_text="(no changes)",
                        artifacts=supporting,
                        metadata=base_metadata(
                            agent="summarizer",
                            node="summarizer_agent",
                            state=state,
                            file_path=path,
                            call_id=call_id,
                            llm_used=False,
                            extra={"reason": "empty_diff"},
                        ),
                    )
                    continue

                if llm is None:
                    logger.warning("No LLM backend available, using heuristic for %s.", path)
                    record_degradation(
                        "llm_unavailable_heuristic",
                        "summarizer used heuristic (no LLM)",
                        node="summarizer_agent",
                        file=path,
                    )
                    heuristic = _fallback_summary(diff_text)
                    summary_pair[f"summary_{call_id}"] = heuristic
                    write_agent_call(
                        call_dir,
                        input_text="",
                        output_text=heuristic,
                        artifacts=supporting,
                        metadata=base_metadata(
                            agent="summarizer",
                            node="summarizer_agent",
                            state=state,
                            file_path=path,
                            call_id=call_id,
                            llm_used=False,
                            extra={"reason": "no_llm"},
                        ),
                    )
                else:
                    fit = fit_variable_blocks(
                        template=prompt_str,
                        render="mustache",
                        fixed_variables={},
                        blocks=[
                            EvidenceBlock(
                                block_id="original_code",
                                text=original_text,
                                kind="context",
                                file_path=path,
                                priority=20,
                            ),
                            EvidenceBlock(
                                block_id="patch",
                                text=diff_text,
                                kind="primary",
                                side=parent_label,
                                file_path=path,
                                priority=10,
                            ),
                        ],
                        encoder=encoder,
                        model_name=model_name,
                        node="summarizer_agent",
                        file_path=path,
                    )
                    prompt_text = fit.prompt
                    call_artifacts = _with_budget_artifacts(supporting, fit)
                    result = resilient_invoke(
                        llm,
                        prompt_text,
                        context=_invoke_context(
                            state,
                            node="summarizer_agent",
                            agent="summarizer",
                            file_path=path,
                            call_id=call_id,
                        ),
                        artifact_dir=call_dir,
                        artifacts=call_artifacts,
                    )
                    content = extract_text_content(result)
                    summary_pair[f"summary_{call_id}"] = content.strip()

                    # Track token usage
                    counts = _init_token_counts(state, path)
                    counts["system_prompt"] += count_tokens(encoder, prompt_str)
                    counts["original"] += count_tokens(encoder, original_text)
                    if parent_label == "A":
                        counts["diff_a"] += count_tokens(encoder, diff_text)
                    else:
                        counts["diff_b"] += count_tokens(encoder, diff_text)
                    counts["output"] += count_tokens(encoder, summary_pair[f"summary_{call_id}"])

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
    """Normalize decision text to ALL_A, ALL_B, or MIX.

    Uses multi-strategy local recovery (first-line, standalone verdict line,
    choose-phrase). Returns MIX when unrecoverable; does **not** record a
    soft degradation — callers that soft-default after retries should call
    ``record_degradation`` themselves.
    """
    decision, _strategy = extract_analyzer_verdict(text)
    return decision if decision is not None else "MIX"


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
        artifact_root = get_artifact_root(state)
        call_dir = agent_call_dir(artifact_root, agent="analyzer")

        model_name = _get_model_name(state)
        encoder, llm = get_backend(model_name)

        # Build concatenated context for global judgement
        a_sum = "\n\n".join(f"{p}: {s.get('summary_a', '')}" for p, s in summaries.items())
        b_sum = "\n\n".join(f"{p}: {s.get('summary_b', '')}" for p, s in summaries.items())
        a_diff = "\n\n".join(f"{p}: {diffs_a.get(p, '')}" for p in summaries.keys())
        b_diff = "\n\n".join(f"{p}: {diffs_b.get(p, '')}" for p in summaries.keys())
        supporting = {
            "a_summaries.txt": a_sum,
            "b_summaries.txt": b_sum,
            "a_diffs.txt": a_diff,
            "b_diffs.txt": b_diff,
        }

        if llm is None:
            logger.warning("No LLM backend available for analyzer; using heuristic.")
            record_degradation(
                "llm_unavailable_heuristic",
                "conflict analyzer used heuristic (no LLM)",
                node="conflict_analyzer",
            )
            len_a, len_b = len(a_sum), len(b_sum)
            if abs(len_a - len_b) < 0.05 * max(1, (len_a + len_b) // 2):
                decision = "MIX"
            else:
                decision = "ALL_A" if len_a <= len_b else "ALL_B"
            raw_output = f"Heuristic decision based on summary lengths (A={len_a}, B={len_b}): {decision}"
            write_agent_call(
                call_dir,
                input_text="",
                output_text=raw_output,
                artifacts=supporting,
                metadata=base_metadata(
                    agent="analyzer",
                    node="conflict_analyzer",
                    state=state,
                    llm_used=False,
                    extra={"reason": "no_llm", "decision": decision},
                ),
            )
        else:
            fit = fit_global_ab_prompt(
                template=prompt_str,
                render="format",
                paths=list(summaries.keys()),
                summaries=summaries,
                diffs_a=diffs_a,
                diffs_b=diffs_b,
                encoder=encoder,
                model_name=model_name,
                node="conflict_analyzer",
            )
            prompt_text = fit.prompt
            # Prefer fitted aggregates in artifacts so audits match the model input.
            supporting = {
                "a_summaries.txt": fit.variables.get("a_summary", a_sum),
                "b_summaries.txt": fit.variables.get("b_summary", b_sum),
                "a_diffs.txt": fit.variables.get("a_diff", a_diff),
                "b_diffs.txt": fit.variables.get("b_diff", b_diff),
            }
            call_artifacts = _with_budget_artifacts(supporting, fit)

            try:
                def _parse_verdict(raw: str) -> ParsedResult:
                    decision, strategy = extract_analyzer_verdict(raw)
                    if decision is None:
                        raise ValueError("unrecognized analyzer verdict")
                    return ParsedResult(decision, strategy)

                parsed, raw_output, _attempt_log = invoke_and_parse(
                    llm,
                    prompt_text,
                    parse_fn=_parse_verdict,
                    repair_hint=(
                        "Return exactly one of the following tokens on its own first line, "
                        "with no other text before it: A, B, or Mix."
                    ),
                    context=_invoke_context(
                        state, node="conflict_analyzer", agent="analyzer"
                    ),
                    artifact_dir=call_dir,
                    artifacts=call_artifacts,
                )
                decision = parsed
            except ParseExhausted as exc:
                raw_output = (exc.raw_text or "").strip()
                logger.error(
                    "Analyzer verdict unparseable after retries; defaulting to MIX."
                )
                if not raw_output:
                    record_degradation(
                        "unclear_verdict_fallback",
                        "empty analyzer output; defaulting to MIX",
                        node="conflict_analyzer",
                    )
                else:
                    record_degradation(
                        "unclear_verdict_fallback",
                        "unrecognized analyzer verdict; defaulting to MIX",
                        detail=raw_output[:200],
                        node="conflict_analyzer",
                    )
                decision = "MIX"
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
        artifact_root = get_artifact_root(state)
        # One global planner call (like analyzer) lives at method/planner/
        call_dir = agent_call_dir(artifact_root, agent="planner")

        if llm is None:
            logger.warning("No LLM backend available, falling back to heuristic.")
            record_degradation(
                "llm_unavailable_heuristic",
                "conflict planner used heuristic (no LLM)",
                node="conflict_agent",
            )
            plan = {}
            for path, s in summaries.items():
                a_len = len(s.get("summary_a", ""))
                b_len = len(s.get("summary_b", ""))
                plan[path] = "A" if a_len <= b_len else "B"
            write_agent_call(
                call_dir,
                input_text="",
                output_text=json.dumps(plan, indent=2, ensure_ascii=False),
                artifacts={},
                metadata=base_metadata(
                    agent="planner",
                    node="conflict_agent",
                    state=state,
                    llm_used=False,
                    extra={"reason": "no_llm"},
                ),
            )
        else:
            logger.debug("Generating merge plan using LLM.")
            a_sum = "\n\n".join(f"{p}: {s.get('summary_a', '')}" for p, s in summaries.items())
            b_sum = "\n\n".join(f"{p}: {s.get('summary_b', '')}" for p, s in summaries.items())
            a_diff = "\n\n".join(f"{p}: {state.get('diffs_a', {}).get(p, '')}" for p in summaries.keys())
            b_diff = "\n\n".join(f"{p}: {state.get('diffs_b', {}).get(p, '')}" for p in summaries.keys())
            supporting = {
                "a_summaries.txt": a_sum,
                "b_summaries.txt": b_sum,
                "a_diffs.txt": a_diff,
                "b_diffs.txt": b_diff,
            }

            fit = fit_global_ab_prompt(
                template=prompt_str,
                render="mustache",
                paths=list(summaries.keys()),
                summaries=summaries,
                diffs_a=state.get("diffs_a", {}) or {},
                diffs_b=state.get("diffs_b", {}) or {},
                encoder=encoder,
                model_name=model_name,
                node="conflict_agent",
            )
            prompt_text = fit.prompt
            supporting = {
                "a_summaries.txt": fit.variables.get("a_summary", a_sum),
                "b_summaries.txt": fit.variables.get("b_summary", b_sum),
                "a_diffs.txt": fit.variables.get("a_diff", a_diff),
                "b_diffs.txt": fit.variables.get("b_diff", b_diff),
            }
            call_artifacts = _with_budget_artifacts(supporting, fit)
            expected_paths = set(summaries.keys())

            def _parse_plan(raw: str) -> dict:
                return parse_plan_json(raw, expected_paths=expected_paths)

            try:
                plan, content, _attempt_log = invoke_and_parse(
                    llm,
                    prompt_text,
                    parse_fn=_parse_plan,
                    repair_hint=(
                        "Return a single JSON object whose keys are the conflicted file "
                        "paths and whose values are one of: A, B, or merge."
                    ),
                    context=_invoke_context(state, node="conflict_agent", agent="planner"),
                    artifact_dir=call_dir,
                    artifacts=call_artifacts,
                )
            except ParseExhausted as exc:
                content = exc.raw_text or ""
                logger.error("Failed to parse LLM plan JSON after retries; merge-all fallback.")
                # Distinguish JSON vs schema failure from the last attempt error.
                last_err = ""
                if exc.attempt_log:
                    last_err = str(exc.attempt_log[-1].get("parse_error") or "")
                if "schema mismatch" in last_err:
                    record_degradation(
                        "plan_schema_fallback",
                        "plan JSON schema mismatch; merge-all fallback",
                        detail=last_err[:200],
                        node="conflict_agent",
                    )
                else:
                    record_degradation(
                        "json_parse_fallback",
                        "plan JSON unparseable; merge-all fallback",
                        detail=(content or "")[:200],
                        node="conflict_agent",
                    )
                plan = {p: "merge" for p in summaries}

            for path in summaries.keys():
                counts = _init_token_counts(state, path)
                counts["system_prompt"] += count_tokens(encoder, prompt_str)
                counts["output"] += count_tokens(encoder, content)

        # Ensure plan contains all files
        if not plan:
            files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
            record_degradation(
                "plan_schema_fallback",
                "empty plan; merge-all fallback",
                node="conflict_agent",
            )
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
        artifact_root = get_artifact_root(state)
        attempt_no = int(state.get("_review_iter", 0)) + 1
        call_id = f"attempt_{attempt_no}"

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
            file_slug = file_path_to_slug(path)
            call_dir = agent_call_dir(
                artifact_root, agent="resolver", file_slug=file_slug, call_id=call_id
            )
            feedback_map = state.get("review_feedback", {}) or {}
            feedback_text = str(feedback_map.get(path, "")).strip()
            supporting = {
                "plan.json": json.dumps({path: plan.get(path, "merge")}, indent=2, ensure_ascii=False),
                "original.txt": ancestor_contents.get(path, ""),
                "a.diff": diffs_a.get(path, ""),
                "b.diff": diffs_b.get(path, ""),
                "review_feedback.txt": feedback_text,
            }

            if choice in ("A", "B") or llm is None:
                if llm is None and choice not in ("A", "B"):
                    logger.warning("No LLM backend available to merge %s, falling back to parent A.", path)
                    record_degradation(
                        "llm_unavailable_heuristic",
                        "resolver used parent A (no LLM)",
                        node="resolution_agent",
                        file=path,
                    )
                    choice = "A"
                logger.debug("Resolving conflict for %s by selecting parent %s.", path, choice)
                merged_text = parent_a.get(path, "") if choice == "A" else parent_b.get(path, "")
                resolved[path] = merged_text
                resolution_history.setdefault(path, []).append(merged_text)
                counts["output"] += count_tokens(encoder, merged_text)
                write_agent_call(
                    call_dir,
                    input_text="",
                    output_text=merged_text,
                    artifacts=supporting,
                    metadata=base_metadata(
                        agent="resolver",
                        node="resolution_agent",
                        state=state,
                        file_path=path,
                        call_id=call_id,
                        llm_used=False,
                        extra={"reason": "parent_select" if choice in ("A", "B") else "no_llm", "choice": choice},
                    ),
                )
                continue

            logger.debug("Resolving conflict for %s using LLM.", path)
            single_plan = {path: plan.get(path, "merge")}

            if feedback_text:
                logger.debug("Applying review feedback for %s (length=%d chars)", path, len(feedback_text))

            plan_json = json.dumps(single_plan, ensure_ascii=False)
            fit = fit_variable_blocks(
                template=prompt_str,
                render="mustache",
                fixed_variables={"plan": plan_json},
                blocks=[
                    EvidenceBlock(
                        block_id="original_code",
                        text=ancestor_contents.get(path, ""),
                        kind="context",
                        file_path=path,
                        priority=20,
                    ),
                    EvidenceBlock(
                        block_id="patch_a",
                        text=diffs_a.get(path, ""),
                        side="A",
                        kind="diff",
                        file_path=path,
                        priority=30,
                    ),
                    EvidenceBlock(
                        block_id="patch_b",
                        text=diffs_b.get(path, ""),
                        side="B",
                        kind="diff",
                        file_path=path,
                        priority=30,
                    ),
                    EvidenceBlock(
                        block_id="review_feedback",
                        text=feedback_text,
                        kind="secondary",
                        file_path=path,
                        priority=40,
                    ),
                ],
                encoder=encoder,
                model_name=model_name,
                node="resolution_agent",
                file_path=path,
            )
            prompt_text = fit.prompt
            call_artifacts = _with_budget_artifacts(supporting, fit)
            result = resilient_invoke(
                llm,
                prompt_text,
                context=_invoke_context(
                    state,
                    node="resolution_agent",
                    agent="resolver",
                    file_path=path,
                    call_id=call_id,
                ),
                artifact_dir=call_dir,
                artifacts=call_artifacts,
            )
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
        artifact_root = get_artifact_root(state)
        attempt_no = int(state.get("_review_iter", 0)) + 1
        call_id = f"attempt_{attempt_no}"
        plan = state.get("conflict_plan", {}) or {}
        diffs_a = state.get("diffs_a", {}) or {}
        diffs_b = state.get("diffs_b", {}) or {}

        reviews: Dict[str, str] = {}
        review_results: Dict[str, Dict[str, str]] = {}
        review_history: Dict[str, list] = state.get("review_history", {}) or {}

        files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
        file_iter = files or list(resolved.keys())

        for path in file_iter:
            content = resolved.get(path, "")
            logger.debug("Reviewing %s.", path)
            file_slug = file_path_to_slug(path)
            call_dir = agent_call_dir(
                artifact_root, agent="reviewer", file_slug=file_slug, call_id=call_id
            )
            supporting = {
                "resolved.txt": content,
                "plan.json": json.dumps({path: plan.get(path, "merge")}, indent=2, ensure_ascii=False),
                "a.diff": diffs_a.get(path, ""),
                "b.diff": diffs_b.get(path, ""),
            }

            if llm is None:
                reviews[path] = "ACCEPT – heuristic stub (no LLM)."
                review_results[path] = {"outcome": "ACCEPT", "rationale": ""}
                logger.warning("No LLM backend available, using heuristic for %s.", path)
                record_degradation(
                    "llm_unavailable_heuristic",
                    "reviewer used ACCEPT stub (no LLM)",
                    node="review_agent",
                    file=path,
                )
                write_agent_call(
                    call_dir,
                    input_text="",
                    output_text=reviews[path],
                    artifacts=supporting,
                    metadata=base_metadata(
                        agent="reviewer",
                        node="review_agent",
                        state=state,
                        file_path=path,
                        call_id=call_id,
                        llm_used=False,
                        extra={"reason": "no_llm", "outcome": "ACCEPT"},
                    ),
                )
                continue

            prompt_fit = fit_variable_blocks(
                template=prompt_str,
                render="mustache",
                fixed_variables={},
                blocks=[
                    EvidenceBlock(
                        block_id="generated_code",
                        text=content,
                        kind="primary",
                        file_path=path,
                        priority=10,
                    ),
                ],
                encoder=encoder,
                model_name=model_name,
                node="review_agent",
                file_path=path,
            )
            prompt_text = prompt_fit.prompt
            call_artifacts = _with_budget_artifacts(supporting, prompt_fit)

            def _parse_review(raw: str) -> dict[str, str]:
                outcome, rationale = parse_review_outcome(raw)
                if outcome is None:
                    raise ValueError("review JSON missing ACCEPT/REJECT outcome")
                return {"outcome": outcome, "rationale": rationale}

            try:
                parsed_review, text, _attempt_log = invoke_and_parse(
                    llm,
                    prompt_text,
                    parse_fn=_parse_review,
                    repair_hint=(
                        'Return a single JSON object like '
                        '{"outcome":"ACCEPT"|"REJECT","rationale":"..."} with no other text.'
                    ),
                    context=_invoke_context(
                        state,
                        node="review_agent",
                        agent="reviewer",
                        file_path=path,
                        call_id=call_id,
                    ),
                    artifact_dir=call_dir,
                    artifacts=call_artifacts,
                )
                outcome = parsed_review["outcome"]
                rationale = parsed_review.get("rationale", "")
            except ParseExhausted as exc:
                text = exc.raw_text or ""
                outcome = "REJECT"
                rationale = text.strip()
                record_degradation(
                    "json_parse_fallback",
                    "review JSON unparseable; defaulting to REJECT",
                    detail=(text or "")[:200],
                    node="review_agent",
                    file=path,
                )

            reviews[path] = text
            review_history.setdefault(path, []).append(text)
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

