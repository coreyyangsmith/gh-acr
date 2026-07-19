"""Graph builder for the multi-agent merge resolution workflow.

Provides a configurable LangGraph factory used by bypass7, better_judge,
force_mix, and the better_judge ablation methods (bj_*).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Callable, Dict

from langgraph.graph import END, StateGraph

from ..observability import build_langfuse_invoke_config, make_trace_name
from ...utils.run_progress import set_stage
from .nodes import (
    create_summarizer_node,
    create_conflict_analyzer_node,
    create_conflict_agent_node,
    create_resolution_agent_node,
    create_review_agent_node,
    PromptVariant,
)
from ..trace_replay import (
    NODE_ANALYZE,
    NODE_PATCH,
    NODE_PLAN,
    NODE_REVIEW,
    NODE_SUMMARISE,
    apply_trace_replay,
    wrap_node_with_replay_skip,
    write_method_replay_metadata,
)


# Type alias for resolver functions
ResolverFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class BypassGraphConfig:
    """Topology flags for the multi-agent merge graph.

    Parameters
    ----------
    prompt_variant
        Which prompt directory to load (``bypass7``, ``better_judge``, ``force_mix``).
    include_summarizer
        If False, seed ``summaries`` from raw diffs (no LLM summarizer).
    include_analyzer
        If False, hard-set ``bypass_decision=MIX`` (force-mix path).
    include_planner
        If False, seed ``conflict_plan`` with ``"merge"`` for every file.
    include_reviewer
        If False, skip the review/feedback loop (patch goes straight to finalize).
    """

    prompt_variant: PromptVariant
    include_summarizer: bool = True
    include_analyzer: bool = True
    include_planner: bool = True
    include_reviewer: bool = True


def _nested_langfuse_config() -> Dict[str, Any] | None:
    """Build invoke config so nested multi-agent roots are method-named in LangFuse.

    Trace name = eval method (from run context); scenario id goes to
    ``langfuse_session_id`` / tags via ``build_langfuse_invoke_config``.
    """
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
    from ..artifact_io import (
        agent_call_dir,
        base_metadata,
        file_path_to_slug,
        get_artifact_root,
        write_agent_call,
    )

    def _bypass_select(state: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(state.get("bypass_decision", "MIX")).upper()
        scenario_id = state.get("scenario_id", "unknown")
        artifact_root = get_artifact_root(state)
        set_stage("bypass", detail=decision)

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
                write_agent_call(
                    agent_call_dir(
                        artifact_root,
                        agent="bypass",
                        file_slug=file_path_to_slug(fpath),
                    ),
                    input_text="",
                    output_text=content or "",
                    artifacts={"decision.txt": decision},
                    metadata=base_metadata(
                        agent="bypass",
                        node="bypass_select",
                        state=state,
                        file_path=fpath,
                        llm_used=False,
                        extra={"decision": decision},
                    ),
                )
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
                write_agent_call(
                    agent_call_dir(
                        artifact_root,
                        agent="bypass",
                        file_slug=file_path_to_slug(fpath),
                    ),
                    input_text="",
                    output_text=content or "",
                    artifacts={"decision.txt": decision},
                    metadata=base_metadata(
                        agent="bypass",
                        node="bypass_select",
                        state=state,
                        file_path=fpath,
                        llm_used=False,
                        extra={"decision": decision},
                    ),
                )
        else:
            logger.info("[bypass_select] scenario=%s decision=%s (not ALL_A/ALL_B, will use resolution agent)", scenario_id, decision)

        state["status"] = "bypassed" if decision in ("ALL_A", "ALL_B") else state.get("status", "")
        return state
    return _bypass_select


def _create_finalize_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a node that finalizes diffs for all resolved files."""
    from ...utils.logger import logger
    from ..artifact_io import get_artifact_root, write_final_artifacts

    def _finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
        resolved = state.get("resolved_contents", {}) or {}
        final_diffs = dict(state.get("final_diffs", {}) or {})
        ancestor_contents = state.get("ancestor_contents", {}) or {}
        scenario_id = state.get("scenario_id", "unknown")
        bypass_decision = state.get("bypass_decision", "")
        artifact_root = get_artifact_root(state)
        set_stage("finalize", detail=f"{len(resolved)} files")

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

            write_final_artifacts(
                artifact_root,
                file_path=path,
                resolved_text=str(merged_text) if merged_text is not None else "",
                final_diff=final_diffs.get(path, ""),
            )

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


def _create_seed_raw_summaries_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Seed summaries from raw diffs (no LLM summarizer)."""
    from ...utils.logger import logger
    from ..artifact_io import (
        agent_call_dir,
        base_metadata,
        file_path_to_slug,
        get_artifact_root,
        write_agent_call,
    )
    from ..utils import scenario_file_list

    def _seed_raw_summaries(state: Dict[str, Any]) -> Dict[str, Any]:
        diffs_a: Dict[str, str] = state.get("diffs_a", {}) or {}
        diffs_b: Dict[str, str] = state.get("diffs_b", {}) or {}
        artifact_root = get_artifact_root(state)
        scenario_id = state.get("scenario_id", "unknown")
        set_stage("seed_raw_summaries")

        files = scenario_file_list(
            state,
            fallback_paths=list(set(diffs_a) | set(diffs_b)),
        )

        summaries: Dict[str, Dict[str, str]] = {}
        for path in files:
            raw_a = diffs_a.get(path, "") or ""
            raw_b = diffs_b.get(path, "") or ""
            summary_a = raw_a if raw_a.strip() else "(no changes)"
            summary_b = raw_b if raw_b.strip() else "(no changes)"
            summaries[path] = {"summary_a": summary_a, "summary_b": summary_b}

            file_slug = file_path_to_slug(path)
            for call_id, text in (("a", summary_a), ("b", summary_b)):
                write_agent_call(
                    agent_call_dir(
                        artifact_root,
                        agent="summarizer",
                        file_slug=file_slug,
                        call_id=call_id,
                    ),
                    input_text="",
                    output_text=text,
                    artifacts={"note.txt": "summarization skipped; raw diff used as summary"},
                    metadata=base_metadata(
                        agent="summarizer",
                        node="seed_raw_summaries",
                        state=state,
                        file_path=path,
                        call_id=call_id,
                        llm_used=False,
                        extra={"reason": "no_summary_ablation"},
                    ),
                )

        state["summaries"] = summaries
        state["status"] = "summarised"
        logger.info(
            "[seed_raw_summaries] scenario=%s seeded %d file summaries from raw diffs",
            scenario_id,
            len(summaries),
        )
        return state

    return _seed_raw_summaries


def _create_force_mix_marker_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Hard-set bypass_decision/method to MIX without calling the LLM analyzer."""

    def _force_mix_marker(state: Dict[str, Any]) -> Dict[str, Any]:
        set_stage("force_mix")
        state["bypass_decision"] = "MIX"
        state["bypass_method"] = "MIX"
        state["bypass_analyzer_output"] = "[force_mix] conflict analyzer skipped"
        state["status"] = "analyzed"
        return state

    return _force_mix_marker


def _create_all_merge_plan_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Seed conflict_plan with merge for every conflicted file (no LLM planner)."""
    from ...utils.logger import logger
    from ..artifact_io import (
        agent_call_dir,
        base_metadata,
        get_artifact_root,
        write_agent_call,
    )
    from ..utils import scenario_file_list
    import json

    def _all_merge_plan(state: Dict[str, Any]) -> Dict[str, Any]:
        set_stage("all_merge_plan")
        diffs_a = state.get("diffs_a", {}) or {}
        diffs_b = state.get("diffs_b", {}) or {}
        parent_a = state.get("parent_a_contents", {}) or {}
        parent_b = state.get("parent_b_contents", {}) or {}
        files = scenario_file_list(
            state,
            fallback_paths=list(parent_a.keys())
            + list(parent_b.keys())
            + list(diffs_a.keys())
            + list(diffs_b.keys()),
        )
        plan = {path: "merge" for path in files}
        state["conflict_plan"] = plan
        state["status"] = "planned"

        artifact_root = get_artifact_root(state)
        write_agent_call(
            agent_call_dir(artifact_root, agent="planner"),
            input_text="",
            output_text=json.dumps(plan, indent=2, ensure_ascii=False),
            artifacts={"note.txt": "planner skipped; all files set to merge"},
            metadata=base_metadata(
                agent="planner",
                node="all_merge_plan",
                state=state,
                llm_used=False,
                extra={"reason": "no_plan_ablation"},
            ),
        )
        logger.info(
            "[all_merge_plan] scenario=%s seeded merge plan for %d files",
            state.get("scenario_id", "unknown"),
            len(plan),
        )
        return state

    return _all_merge_plan


def build_configured_graph(config: BypassGraphConfig) -> ResolverFunc:
    """Build a multi-agent resolver from topology flags.

    See :class:`BypassGraphConfig` for the meaning of each flag.

    When ``state["trace_replay"]["enabled"]`` is set, hydrated stages from a
    canonical ``better_judge`` snapshot are skipped (see ``src.agents.trace_replay``).
    """
    prompt_variant = config.prompt_variant

    summarizer_raw = (
        create_summarizer_node(prompt_variant)
        if config.include_summarizer
        else _create_seed_raw_summaries_node()
    )
    summarizer = wrap_node_with_replay_skip(
        summarizer_raw,
        NODE_SUMMARISE if config.include_summarizer else "seed_raw_summaries",
    )
    analyzer = (
        wrap_node_with_replay_skip(
            create_conflict_analyzer_node(prompt_variant), NODE_ANALYZE
        )
        if config.include_analyzer
        else None
    )
    force_mix_marker = (
        wrap_node_with_replay_skip(
            _create_force_mix_marker_node(), "force_mix_marker"
        )
        if not config.include_analyzer
        else None
    )
    planner_raw = (
        create_conflict_agent_node(prompt_variant)
        if config.include_planner
        else _create_all_merge_plan_node()
    )
    planner = wrap_node_with_replay_skip(
        planner_raw,
        NODE_PLAN if config.include_planner else "all_merge_plan",
    )
    resolver = wrap_node_with_replay_skip(
        create_resolution_agent_node(prompt_variant), NODE_PATCH
    )
    reviewer = (
        wrap_node_with_replay_skip(
            create_review_agent_node(prompt_variant), NODE_REVIEW
        )
        if config.include_reviewer
        else None
    )
    bypass_select = (
        wrap_node_with_replay_skip(_create_bypass_select_node(), "bypass")
        if config.include_analyzer
        else None
    )
    finalize = _create_finalize_node()

    def feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
        if "feedback" in (state.get("_replay_skip_nodes") or set()):
            return state
        return _prepare_feedback_for_retry(state)

    def resolver_function(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the configured multi-agent workflow."""
        # Conditional trace replay: hydrate prefix / skip markers before invoke.
        state, provenance = apply_trace_replay(state)
        write_method_replay_metadata(state.get("artifact_root"), provenance)

        sg = StateGraph(dict)

        # Entry: summarise (LLM) or seed_raw_summaries
        entry_name = "summarise" if config.include_summarizer else "seed_raw_summaries"
        sg.add_node(entry_name, summarizer)
        sg.set_entry_point(entry_name)

        sg.add_node("plan", planner)
        sg.add_node("patch", resolver)
        sg.add_node("finalize", finalize)

        if config.include_analyzer:
            assert analyzer is not None and bypass_select is not None
            sg.add_node("analyze", analyzer)
            sg.add_node("bypass", bypass_select)
            sg.add_edge(entry_name, "analyze")
            sg.add_conditional_edges(
                "analyze",
                _route_after_analyze,
                {"all_a": "bypass", "all_b": "bypass", "mix": "plan"},
            )
            sg.add_edge("bypass", "finalize")
        else:
            assert force_mix_marker is not None
            sg.add_node("force_mix_marker", force_mix_marker)
            sg.add_edge(entry_name, "force_mix_marker")
            sg.add_edge("force_mix_marker", "plan")

        sg.add_edge("plan", "patch")

        if config.include_reviewer:
            assert reviewer is not None
            sg.add_node("review", reviewer)
            sg.add_node("feedback", feedback_node)
            sg.add_edge("patch", "review")
            sg.add_conditional_edges(
                "review",
                _route_after_review,
                {"finish": "finalize", "retry": "feedback"},
            )
            sg.add_edge("feedback", "patch")
        else:
            sg.add_edge("patch", "finalize")

        sg.add_edge("finalize", END)

        sub_app = sg.compile()

        if "_review_iter" not in state:
            state["_review_iter"] = 0

        return sub_app.invoke(state, config=_nested_langfuse_config())

    return resolver_function


def build_bypass_graph(prompt_variant: PromptVariant = "bypass7") -> ResolverFunc:
    """Build a full bypass multi-agent resolver with review loop.

    This is the complete workflow used by both "bypass7" and "better_judge":

    ```
    summarise → analyze ─┬─ [all_a/all_b] → bypass → finalize → END
                         └─ [mix] → plan → patch → review ─┬─ [finish] → finalize → END
                                                   ↑       └─ [retry] → feedback ─┘
                                                   └───────────────────────────────┘
    ```

    Parameters
    ----------
    prompt_variant
        Which prompt templates to use ("bypass7" or "better_judge")

    Returns
    -------
    ResolverFunc
        A callable that executes the complete workflow on a state dict.
    """
    return build_configured_graph(
        BypassGraphConfig(
            prompt_variant=prompt_variant,
            include_summarizer=True,
            include_analyzer=True,
            include_planner=True,
            include_reviewer=True,
        )
    )


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
    return build_configured_graph(
        BypassGraphConfig(
            prompt_variant=prompt_variant,
            include_summarizer=True,
            include_analyzer=False,
            include_planner=True,
            include_reviewer=True,
        )
    )


__all__ = [
    "BypassGraphConfig",
    "build_configured_graph",
    "build_bypass_graph",
    "build_force_mix_graph",
    "ResolverFunc",
]
