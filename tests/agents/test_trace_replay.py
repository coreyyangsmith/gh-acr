"""Tests for conditional better_judge trace replay."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.artifact_io import (
    base_metadata,
    list_resolver_attempts,
    read_agent_output,
    write_agent_call,
    write_final_artifacts,
)
from src.agents.trace_replay import (
    METHOD_REPLAY_META_FILENAME,
    SNAPSHOT_FILENAME,
    SNAPSHOT_VERSION,
    adapt_legacy_artifacts,
    apply_trace_replay,
    build_snapshot_from_state,
    hydrate_state_from_snapshot,
    load_snapshot,
    plan_ablation_replay,
    save_snapshot_from_state,
    should_skip_node,
    wrap_node_with_replay_skip,
    write_method_replay_metadata,
    write_snapshot,
)


def _mix_snapshot(**overrides) -> dict:
    base = {
        "version": SNAPSHOT_VERSION,
        "source_method": "better_judge",
        "scenario_id": "scn-1",
        "model_name": "openrouter/qwen/qwen3-32b",
        "timestamp": "2026-07-18T00:00:00+00:00",
        "bypass_decision": "MIX",
        "bypass_method": "MIX",
        "bypass_analyzer_output": "Mix\nBecause...",
        "summaries": {
            "a.py": {"summary_a": "sumA", "summary_b": "sumB"},
        },
        "conflict_plan": {"a.py": "merge"},
        "resolver_attempt_1": {"a.py": "FIRST_ATTEMPT\n"},
        "resolved_contents": {"a.py": "FINAL_AFTER_REVIEW\n"},
        "final_diffs": {"a.py": "diff\n"},
        "review_iters": 1,
        "resolver_attempts": 2,
        "files": ["a.py"],
    }
    base.update(overrides)
    return base


def _ab_snapshot(decision: str = "ALL_A") -> dict:
    return _mix_snapshot(
        bypass_decision=decision,
        bypass_method="A" if decision == "ALL_A" else "B",
        conflict_plan={},
        resolver_attempt_1={"a.py": "PARENT_A\n"},
        resolved_contents={"a.py": "PARENT_A\n"},
        review_iters=0,
        resolver_attempts=1,
    )


def test_build_and_write_snapshot(tmp_path: Path):
    state = {
        "scenario_id": "scn-1",
        "model_name": "m1",
        "bypass_decision": "MIX",
        "bypass_method": "MIX",
        "bypass_analyzer_output": "Mix",
        "summaries": {"a.py": {"summary_a": "A", "summary_b": "B"}},
        "conflict_plan": {"a.py": "merge"},
        "resolved_contents": {"a.py": "final"},
        "final_diffs": {"a.py": "d"},
        "resolution_history": {"a.py": ["first", "final"]},
        "_review_iter": 1,
        "sample_row": {"scenario_json": {"files_in_merge_conflict": ["a.py"]}},
        "artifact_root": str(tmp_path / "better_judge"),
    }
    path = save_snapshot_from_state(state)
    assert path is not None
    assert path.name == SNAPSHOT_FILENAME
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["version"] == SNAPSHOT_VERSION
    assert snap["resolver_attempt_1"]["a.py"] == "first"
    assert snap["resolved_contents"]["a.py"] == "final"


def test_plan_bj_no_summary_never_reuses():
    plan = plan_ablation_replay("bj_no_summary", _mix_snapshot())
    assert plan.strategy == "no_reuse"
    assert plan.reused_nodes == []
    assert "analyze" in plan.executed_nodes


def test_plan_bj_no_judge_mix_full_reuse():
    plan = plan_ablation_replay("bj_no_judge", _mix_snapshot())
    assert plan.strategy == "full_suffix_reuse"
    assert "patch" in plan.reused_nodes
    assert plan.executed_nodes == ["finalize"]


def test_plan_bj_no_judge_ab_reuses_summaries_only():
    plan = plan_ablation_replay("bj_no_judge", _ab_snapshot())
    assert plan.strategy == "reuse_summaries_run_mix"
    assert plan.reused_nodes == ["summarise"]
    assert "force_mix_marker" in plan.executed_nodes


def test_plan_bj_no_plan_mix_runs_resolver():
    plan = plan_ablation_replay("bj_no_plan", _mix_snapshot())
    assert plan.strategy == "reuse_prefix_run_resolver"
    assert "all_merge_plan" in plan.executed_nodes
    assert "patch" in plan.executed_nodes


def test_plan_bj_no_plan_ab_reuses_bypass():
    plan = plan_ablation_replay("bj_no_plan", _ab_snapshot())
    assert plan.strategy == "reuse_bypass"


def test_plan_bj_no_review_uses_attempt_1_not_final():
    snap = _mix_snapshot()
    plan = plan_ablation_replay("bj_no_review", snap)
    assert plan.strategy == "reuse_first_resolver"
    assert "resolver_attempt_1" in plan.hydrate_keys
    assert "resolved_contents" not in plan.hydrate_keys

    state: dict = {"eval_method": "bj_no_review"}
    hydrate_state_from_snapshot(state, snap, plan)
    assert state["resolved_contents"]["a.py"] == "FIRST_ATTEMPT\n"
    assert state["resolved_contents"]["a.py"] != snap["resolved_contents"]["a.py"]


def test_plan_missing_snapshot_falls_back():
    plan = plan_ablation_replay("bj_no_review", None, load_error="missing_source_root")
    assert plan.strategy == "live_fallback"
    assert plan.fallback_reason == "missing_source_root"


def test_legacy_artifact_adaptation(tmp_path: Path):
    root = tmp_path / "better_judge"
    write_agent_call(
        root / "foo_a.py" / "summarizer" / "a",
        output_text="sumA",
        metadata={"file_path": "foo/a.py", "agent": "summarizer"},
    )
    write_agent_call(
        root / "foo_a.py" / "summarizer" / "b",
        output_text="sumB",
        metadata={"file_path": "foo/a.py", "agent": "summarizer"},
    )
    write_agent_call(
        root / "analyzer",
        output_text="Mix\nReasoning...",
        metadata={"agent": "analyzer"},
    )
    write_agent_call(
        root / "planner",
        output_text=json.dumps({"foo/a.py": "merge"}),
        metadata={"agent": "planner"},
    )
    write_agent_call(
        root / "foo_a.py" / "resolver" / "attempt_1",
        output_text="attempt1",
        metadata={"agent": "resolver"},
    )
    write_agent_call(
        root / "foo_a.py" / "resolver" / "attempt_2",
        output_text="attempt2",
        metadata={"agent": "resolver"},
    )
    write_final_artifacts(
        root, file_path="foo/a.py", resolved_text="final", final_diff="d"
    )

    adapted = adapt_legacy_artifacts(
        root, files=["foo/a.py"], model_name="m", scenario_id="s"
    )
    assert adapted is not None
    assert adapted["legacy_adapted"] is True
    assert adapted["bypass_decision"] == "MIX"
    assert adapted["summaries"]["foo/a.py"]["summary_a"] == "sumA"
    assert adapted["resolver_attempt_1"]["foo/a.py"] == "attempt1"
    assert adapted["resolved_contents"]["foo/a.py"] == "final"
    assert list_resolver_attempts(root, file_slug="foo_a.py") == [1, 2]


def test_load_snapshot_prefers_json_over_legacy(tmp_path: Path):
    root = tmp_path / "better_judge"
    snap = _mix_snapshot(model_name="m1")
    write_snapshot(root, snap)
    loaded, err, legacy = load_snapshot(root, expected_model="m1")
    assert err is None
    assert legacy is False
    assert loaded["resolver_attempt_1"]["a.py"] == "FIRST_ATTEMPT\n"


def test_load_snapshot_model_mismatch(tmp_path: Path):
    root = tmp_path / "better_judge"
    write_snapshot(root, _mix_snapshot(model_name="other-model"))
    loaded, err, _ = load_snapshot(root, expected_model="m1")
    assert loaded is None
    assert err == "model_mismatch"


def test_apply_trace_replay_hydrates_and_sets_provenance(tmp_path: Path):
    root = tmp_path / "better_judge"
    write_snapshot(root, _mix_snapshot(model_name="m1"))
    state = {
        "scenario_id": "scn-1",
        "eval_method": "bj_no_review",
        "model_name": "m1",
        "artifact_root": str(tmp_path / "bj_no_review"),
        "sample_row": {"scenario_json": {"files_in_merge_conflict": ["a.py"]}},
        "trace_replay": {
            "enabled": True,
            "source_method": "better_judge",
            "source_root": str(root),
        },
    }
    out, prov = apply_trace_replay(state)
    assert prov.enabled is True
    assert prov.strategy == "reuse_first_resolver"
    assert out["resolved_contents"]["a.py"] == "FIRST_ATTEMPT\n"
    assert should_skip_node(out, "patch")
    assert out["trace_replay_provenance"]["strategy"] == "reuse_first_resolver"

    meta_path = write_method_replay_metadata(tmp_path / "bj_no_review", prov)
    assert meta_path is not None
    assert meta_path.name == METHOD_REPLAY_META_FILENAME
    disk = json.loads(meta_path.read_text(encoding="utf-8"))
    assert disk["strategy"] == "reuse_first_resolver"


def test_apply_trace_replay_disabled():
    state = {
        "eval_method": "bj_no_review",
        "trace_replay": {"enabled": False},
    }
    out, prov = apply_trace_replay(state)
    assert prov.enabled is False
    assert out is state


def test_wrap_node_skip_review_finishes_route():
    calls = {"n": 0}

    def live(state):
        calls["n"] += 1
        return state

    wrapped = wrap_node_with_replay_skip(live, "review")
    state = {
        "_replay_skip_nodes": {"review"},
        "resolved_contents": {"a.py": "x"},
        "_review_iter": 0,
    }
    out = wrapped(state)
    assert calls["n"] == 0
    assert out["review_results"]["a.py"]["outcome"] == "ACCEPT"
    assert out["_review_iter"] >= 2


def test_base_metadata_includes_replay_fields():
    state = {
        "scenario_id": "s",
        "eval_method": "bj_no_plan",
        "model_name": "m",
        "trace_replay_provenance": {
            "enabled": True,
            "strategy": "reuse_prefix_run_resolver",
            "source_path": "/tmp/better_judge",
            "source_method": "better_judge",
            "reused_nodes": ["summarise", "analyze"],
            "executed_nodes": ["all_merge_plan", "patch", "finalize"],
            "fallback_reason": None,
        },
    }
    meta = base_metadata(agent="planner", node="all_merge_plan", state=state, llm_used=False)
    assert meta["trace_replay_enabled"] is True
    assert meta["trace_replay_strategy"] == "reuse_prefix_run_resolver"
    assert meta["trace_replay_reused_nodes"] == ["summarise", "analyze"]


def test_read_agent_output_helper(tmp_path: Path):
    write_agent_call(tmp_path / "analyzer", output_text="Mix")
    assert read_agent_output(tmp_path, agent="analyzer") == "Mix"
    assert read_agent_output(tmp_path, agent="missing") is None


def test_configured_graph_replay_skips_llm_for_no_review(tmp_path: Path):
    """End-to-end: bj_no_review with MIX snapshot runs finalize without LLM."""
    from src.agents.multi_agent import create_resolver

    source = tmp_path / "better_judge"
    write_snapshot(
        source,
        _mix_snapshot(model_name="test-model"),
    )
    artifact_root = tmp_path / "bj_no_review"
    artifact_root.mkdir()

    state = {
        "scenario_id": "scn-1",
        "eval_method": "bj_no_review",
        "model_name": "test-model",
        "artifact_root": str(artifact_root),
        "ancestor_contents": {"a.py": "base\n"},
        "parent_a_contents": {"a.py": "a\n"},
        "parent_b_contents": {"a.py": "b\n"},
        "diffs_a": {"a.py": "+a\n"},
        "diffs_b": {"a.py": "+b\n"},
        "sample_row": {
            "scenario_json": {"files_in_merge_conflict": ["a.py"]},
            "df_index": "scn-1",
            "name": "o/r",
        },
        "trace_replay": {
            "enabled": True,
            "source_method": "better_judge",
            "source_root": str(source),
        },
    }

    with patch("src.agents.multi_agent.nodes.get_backend") as mock_backend:
        mock_backend.return_value = (MagicMock(), MagicMock())
        resolver = create_resolver("bj_no_review")
        result = resolver(state)

    # No LLM should have been invoked (all stages hydrated / skipped)
    mock_backend.assert_not_called()
    assert result["resolved_contents"]["a.py"] == "FIRST_ATTEMPT\n"
    assert (artifact_root / METHOD_REPLAY_META_FILENAME).is_file()
    assert (artifact_root / "a.py" / "final" / "resolved.txt").read_text(
        encoding="utf-8"
    ) == "FIRST_ATTEMPT\n"
    prov = json.loads(
        (artifact_root / METHOD_REPLAY_META_FILENAME).read_text(encoding="utf-8")
    )
    assert prov["strategy"] == "reuse_first_resolver"


def test_configured_graph_no_summary_does_not_reuse_downstream(tmp_path: Path):
    """bj_no_summary must not hydrate analyzer/plan from the canonical MIX trace."""
    from src.agents.multi_agent import create_resolver

    source = tmp_path / "better_judge"
    write_snapshot(source, _mix_snapshot(model_name="test-model"))
    artifact_root = tmp_path / "bj_no_summary"
    artifact_root.mkdir()

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mix\n")

    state = {
        "scenario_id": "scn-1",
        "eval_method": "bj_no_summary",
        "model_name": "test-model",
        "artifact_root": str(artifact_root),
        "ancestor_contents": {"a.py": "base\n"},
        "parent_a_contents": {"a.py": "PARENT_A\n"},
        "parent_b_contents": {"a.py": "PARENT_B\n"},
        "diffs_a": {"a.py": "--- a\n+++ b\n+a\n"},
        "diffs_b": {"a.py": "--- a\n+++ b\n+b\n"},
        "sample_row": {
            "scenario_json": {"files_in_merge_conflict": ["a.py"]},
            "df_index": "scn-1",
            "name": "o/r",
        },
        "trace_replay": {
            "enabled": True,
            "source_method": "better_judge",
            "source_root": str(source),
        },
    }

    # Force analyzer to ALL_A so we take bypass (no planner/resolver LLM needed).
    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(MagicMock(encode=lambda t: list(range(5))), llm),
    ), patch(
        "src.agents.multi_agent.nodes.invoke_and_parse",
        return_value=("ALL_A", "A", []),
    ), patch(
        "src.agents.multi_agent.nodes.fit_global_ab_prompt",
    ) as fit_mock:
        fit_mock.return_value = MagicMock(
            prompt="p",
            variables={
                "a_summary": "x",
                "b_summary": "y",
                "a_diff": "d",
                "b_diff": "d",
            },
            was_clipped=False,
        )
        resolver = create_resolver("bj_no_summary")
        result = resolver(state)

    assert result["bypass_decision"] == "ALL_A"
    assert result["resolved_contents"]["a.py"] == "PARENT_A\n"
    # Must not have reused canonical FINAL_AFTER_REVIEW
    assert result["resolved_contents"]["a.py"] != "FINAL_AFTER_REVIEW\n"
    prov = json.loads(
        (artifact_root / METHOD_REPLAY_META_FILENAME).read_text(encoding="utf-8")
    )
    assert prov["strategy"] == "no_reuse"


def test_run_all_kwargs_accept_trace_replay():
    """_run_kwargs helper / signature smoke: main accepts the flag via tyro."""
    import inspect

    from src.cli.run_all import main

    params = inspect.signature(main).parameters
    assert "trace_replay" in params
    assert params["trace_replay"].default is False
