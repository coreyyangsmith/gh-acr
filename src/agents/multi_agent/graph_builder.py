"""Graph builder for the bypass7 multi-agent merge resolution workflow.

This module provides the factory function that creates the complete LangGraph
workflow. The workflow uses the node factories from `nodes.py` and wires
them together appropriately.

Workflow
--------
- **build_bypass_graph**: Full multi-agent with summary → analyze → plan → patch → review loop
"""

from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, Literal

from langgraph.graph import END, StateGraph

from ..observability import build_langfuse_invoke_config, make_trace_name
from .nodes import (
    create_summarizer_node,
    create_conflict_analyzer_node,
    create_conflict_agent_node,
    create_resolution_agent_node,
    create_review_agent_node,
    PromptVariant,
)


# Type alias for resolver functions
ResolverFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def _nested_langfuse_config() -> Dict[str, Any] | None:
    """Build invoke config so nested multi-agent roots are method-named in LangFuse."""
    return build_langfuse_invoke_config({"run_name": make_trace_name()})


def _route_after_review(state: Dict[str, Any]) -> str:
    """Route based on review outcomes and retry budget.

    Returns "finish" if:
    - All files have ACCEPT outcome, or
    - Maximum retry iterations (2) have been reached

    Otherwise returns "retry" to trigger another resolution attempt.
    """
    max_iters = 2
    iter_no = int(state.get("_review_iter", 0))
    results = state.get("review_results", {}) or {}
    if results and all((v or {}).get("outcome") == "ACCEPT" for v in results.values()):
        return "finish"
    if iter_no >= max_iters:
        return "finish"
    return "retry"


def _prepare_feedback_for_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect reviewer comments for REJECT files and increment retry counter.

    Aggregates feedback across iterations so the resolution agent sees the full
    history of reviewer guidance per file.
    """
    iter_no = int(state.get("_review_iter", 0))
    review_results = state.get("review_results", {}) or {}
    raw_reviews = state.get("reviews", {}) or {}

    # Maintain history of comments per file across iterations
    history: Dict[str, list] = state.setdefault("review_feedback_history", {})

    for path, data in review_results.items():
        if not isinstance(data, dict):
            continue
        if str(data.get("outcome", "")).upper() == "ACCEPT":
            continue

        rationale = str(data.get("rationale", "")).strip()
        full_text = str(raw_reviews.get(path, "")).strip()

        entry_parts = []
        if rationale:
            entry_parts.append(f"Rationale: {rationale}")
        if full_text and full_text != rationale:
            entry_parts.append(f"Raw: {full_text}")
        if not entry_parts:
            entry_parts.append("(no review comments provided)")

        entry = f"Iteration {iter_no}:\n" + "\n".join(entry_parts)
        history.setdefault(path, []).append(entry)

    # Build feedback for next resolution pass
    feedback_map: Dict[str, str] = {}
    for path, entries in history.items():
        feedback_map[path] = "\n\n---\n\n".join(entries)

    state["review_feedback"] = feedback_map
    state["_review_iter"] = iter_no + 1
    state["review_feedback_history"] = history
    return state


def _create_bypass_select_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a node that selects parent contents based on bypass decision."""
    from ...utils.logger import logger
    
    def _bypass_select(state: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(state.get("bypass_decision", "MIX")).upper()
        scenario_id = state.get("scenario_id", "unknown")
        
        if decision == "ALL_A":
            parent_a = state.get("parent_a_contents", {}) or {}
            state["resolved_contents"] = parent_a
            # Log diagnostic info for ALL_A bypass
            logger.info(
                "[bypass_select] scenario=%s decision=ALL_A, parent_a_contents has %d files",
                scenario_id, len(parent_a)
            )
            for fpath, content in parent_a.items():
                content_len = len(content) if content else 0
                if content_len == 0:
                    logger.warning("[bypass_select] scenario=%s EMPTY parent_a_contents for file: %s", scenario_id, fpath)
                else:
                    logger.debug("[bypass_select] scenario=%s parent_a_contents[%s] = %d chars", scenario_id, fpath, content_len)
        elif decision == "ALL_B":
            parent_b = state.get("parent_b_contents", {}) or {}
            state["resolved_contents"] = parent_b
            # Log diagnostic info for ALL_B bypass
            logger.info(
                "[bypass_select] scenario=%s decision=ALL_B, parent_b_contents has %d files",
                scenario_id, len(parent_b)
            )
            for fpath, content in parent_b.items():
                content_len = len(content) if content else 0
                if content_len == 0:
                    logger.warning("[bypass_select] scenario=%s EMPTY parent_b_contents for file: %s", scenario_id, fpath)
                else:
                    logger.debug("[bypass_select] scenario=%s parent_b_contents[%s] = %d chars", scenario_id, fpath, content_len)
        else:
            logger.info("[bypass_select] scenario=%s decision=%s (not ALL_A/ALL_B, will use resolution agent)", scenario_id, decision)
        
        state["status"] = "bypassed" if decision in ("ALL_A", "ALL_B") else state.get("status", "")
        return state
    return _bypass_select


def _create_finalize_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a node that finalizes diffs for all resolved files."""
    from ...utils.logger import logger
    
    def _finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
        resolved = state.get("resolved_contents", {}) or {}
        final_diffs = dict(state.get("final_diffs", {}) or {})
        ancestor_contents = state.get("ancestor_contents", {}) or {}
        scenario_id = state.get("scenario_id", "unknown")
        bypass_decision = state.get("bypass_decision", "")
        
        logger.info(
            "[finalize] scenario=%s bypass_decision=%s, resolved_contents has %d files, existing final_diffs=%d",
            scenario_id, bypass_decision, len(resolved), len(final_diffs)
        )
        
        # Check if resolved_contents matches expected files
        expected_files = state.get("sample_row", {}).get("scenario_json", {}).get("files_in_merge_conflict", [])
        if expected_files:
            missing_in_resolved = set(expected_files) - set(resolved.keys())
            if missing_in_resolved:
                logger.warning(
                    "[finalize] scenario=%s Missing files in resolved_contents: %s",
                    scenario_id, list(missing_in_resolved)
                )

        for path, merged_text in resolved.items():
            merged_len = len(merged_text) if merged_text else 0
            if merged_len == 0:
                logger.warning("[finalize] scenario=%s EMPTY resolved_contents for file: %s", scenario_id, path)
            
            if path not in final_diffs:
                a_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
                m_lines = str(merged_text).splitlines(keepends=True)
                final_diffs[path] = "".join(
                    difflib.unified_diff(a_lines, m_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
                )
                logger.debug("[finalize] scenario=%s computed final_diff for %s (%d chars)", scenario_id, path, len(final_diffs[path]))

        state["final_diffs"] = final_diffs
        state["status"] = "review_finalized"
        logger.info("[finalize] scenario=%s completed with %d final_diffs", scenario_id, len(final_diffs))
        return state
    return _finalize_node


def _route_after_analyze(state: Dict[str, Any]) -> str:
    """Route based on analyzer decision: all_a, all_b, or mix."""
    decision = str(state.get("bypass_decision", "MIX")).upper()
    if decision == "ALL_A":
        return "all_a"
    if decision == "ALL_B":
        return "all_b"
    return "mix"


def build_bypass_graph(prompt_variant: PromptVariant = "bypass") -> ResolverFunc:
    """Build a full bypass multi-agent resolver with review loop.

    This is the complete workflow used by both "bypass" and "bypass7" variants:

    ```
    summarise → analyze ─┬─ [all_a/all_b] → bypass → finalize → END
                         └─ [mix] → plan → patch → review ─┬─ [finish] → finalize → END
                                                   ↑       └─ [retry] → feedback ─┘
                                                   └───────────────────────────────┘
    ```

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use ("bypass" or "bypass7")

    Returns
    -------
    ResolverFunc
        A callable that executes the complete workflow on a state dict.
    """
    # Create nodes with the specified prompt variant
    summarizer = create_summarizer_node(prompt_variant)
    analyzer = create_conflict_analyzer_node(prompt_variant)
    planner = create_conflict_agent_node(prompt_variant)
    resolver = create_resolution_agent_node(prompt_variant)
    reviewer = create_review_agent_node(prompt_variant)
    bypass_select = _create_bypass_select_node()
    finalize = _create_finalize_node()

    def feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return _prepare_feedback_for_retry(state)

    def resolver_function(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full bypass multi-agent workflow."""
        sg = StateGraph(dict)

        # Add all nodes
        sg.add_node("summarise", summarizer)
        sg.add_node("analyze", analyzer)
        sg.add_node("plan", planner)
        sg.add_node("patch", resolver)
        sg.add_node("review", reviewer)
        sg.add_node("bypass", bypass_select)
        sg.add_node("finalize", finalize)
        sg.add_node("feedback", feedback_node)

        # Set entry point
        sg.set_entry_point("summarise")

        # Wire up the graph
        sg.add_edge("summarise", "analyze")
        sg.add_conditional_edges(
            "analyze",
            _route_after_analyze,
            {"all_a": "bypass", "all_b": "bypass", "mix": "plan"},
        )
        sg.add_edge("bypass", "finalize")
        sg.add_edge("plan", "patch")
        sg.add_edge("patch", "review")
        sg.add_conditional_edges(
            "review",
            _route_after_review,
            {"finish": "finalize", "retry": "feedback"},
        )
        sg.add_edge("feedback", "patch")
        sg.add_edge("finalize", END)

        # Compile and run
        sub_app = sg.compile()

        if "_review_iter" not in state:
            state["_review_iter"] = 0

        return sub_app.invoke(state, config=_nested_langfuse_config())

    return resolver_function


def build_force_mix_graph(prompt_variant: PromptVariant = "force_mix") -> ResolverFunc:
    """Build a force-mix multi-agent resolver that always takes the mix path.

    Unlike :func:`build_bypass_graph`, this workflow **skips** the conflict
    analyzer (LLM-as-judge) step entirely. ``bypass_decision`` and
    ``bypass_method`` are hard-set to ``"MIX"`` in state before the planner
    runs, preserving downstream reporting compatibility.

    Workflow::

        summarise → force_mix_marker → plan → patch → review ─┬─ [finish] → finalize → END
                                                      ↑        └─ [retry] → feedback ─┘
                                                      └────────────────────────────────┘

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use (defaults to "force_mix").

    Returns
    -------
    ResolverFunc
        A callable that executes the force-mix workflow on a state dict.
    """
    summarizer = create_summarizer_node(prompt_variant)
    planner = create_conflict_agent_node(prompt_variant)
    resolver = create_resolution_agent_node(prompt_variant)
    reviewer = create_review_agent_node(prompt_variant)
    finalize = _create_finalize_node()

    def _force_mix_marker(state: Dict[str, Any]) -> Dict[str, Any]:
        """Set bypass_decision/method to MIX without calling the LLM analyzer."""
        state["bypass_decision"] = "MIX"
        state["bypass_method"] = "MIX"
        state["bypass_analyzer_output"] = "[force_mix] conflict analyzer skipped"
        state["status"] = "analyzed"
        return state

    def feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return _prepare_feedback_for_retry(state)

    def resolver_function(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the force-mix workflow (no analyzer, always mix path)."""
        sg = StateGraph(dict)

        sg.add_node("summarise", summarizer)
        sg.add_node("force_mix_marker", _force_mix_marker)
        sg.add_node("plan", planner)
        sg.add_node("patch", resolver)
        sg.add_node("review", reviewer)
        sg.add_node("finalize", finalize)
        sg.add_node("feedback", feedback_node)

        sg.set_entry_point("summarise")

        sg.add_edge("summarise", "force_mix_marker")
        sg.add_edge("force_mix_marker", "plan")
        sg.add_edge("plan", "patch")
        sg.add_edge("patch", "review")
        sg.add_conditional_edges(
            "review",
            _route_after_review,
            {"finish": "finalize", "retry": "feedback"},
        )
        sg.add_edge("feedback", "patch")
        sg.add_edge("finalize", END)

        sub_app = sg.compile()

        if "_review_iter" not in state:
            state["_review_iter"] = 0

        return sub_app.invoke(state, config=_nested_langfuse_config())

    return resolver_function


__all__ = [
    "build_bypass_graph",
    "build_force_mix_graph",
    "ResolverFunc",
]



