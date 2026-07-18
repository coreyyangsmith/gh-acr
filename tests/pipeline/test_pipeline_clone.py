"""Tests for merge_pipeline.pipeline_clone pure + mocked pieces."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.merge_pipeline import pipeline_clone as pc


def test_diff_ratio_identical_and_different():
    assert pc._diff_ratio("hello", "hello") == pytest.approx(1.0)
    assert pc._diff_ratio("hello", "world") < 1.0
    assert pc._diff_ratio("", "") == pytest.approx(1.0)


def test_checkout_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("GHACR_CLONE_DIR", str(tmp_path / "checkouts"))
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    root = pc._checkout_root()
    assert root == tmp_path / "checkouts"


@pytest.mark.slow
def test_read_files_at_commit_tiny_repo(tiny_local_repo: Path):
    from git import Repo

    repo = Repo(tiny_local_repo)
    head = repo.head.commit.hexsha
    contents = pc._read_files_at_commit(repo, head, ["main.py", "missing.py"])
    assert "main.py" in contents
    assert contents["main.py"]
    assert contents["missing.py"] == ""


def test_load_sample_node_by_id_column(tiny_benchmark_df: pd.DataFrame):
    with patch.object(pc, "load_benchmark", return_value=tiny_benchmark_df):
        state = {"scenario_id": "s2"}
        out = pc.load_sample_node(state)
    assert out["status"] == "sample_loaded"
    assert out["sample_row"]["id"] == "s2"
    assert out["sample_row"]["scenario_json"]["files_in_merge_conflict"] == ["b.py"]


def test_load_sample_node_missing_raises(tiny_benchmark_df: pd.DataFrame):
    with patch.object(pc, "load_benchmark", return_value=tiny_benchmark_df):
        with pytest.raises(ValueError, match="not found"):
            pc.load_sample_node({"scenario_id": "does-not-exist"})


def test_evaluate_node_computes_metrics(tiny_local_repo: Path):
    from git import Repo

    repo = Repo(tiny_local_repo)
    head = repo.head.commit.hexsha
    truth = pc._read_files_at_commit(repo, head, ["main.py"])

    state = {
        "sample_row": {
            "scenario_json": {
                "files_in_merge_conflict": ["main.py"],
                "parents": [head, head],
                "merge_commit_hash": head,
            }
        },
        "repo_path": str(tiny_local_repo),
        "ancestor_contents": {"main.py": "print('base')\n"},
        "resolved_contents": dict(truth),
    }
    out = pc.evaluate_node(state)
    assert out["status"] == "evaluated"
    assert out["evaluation"]["overall_exact_match"] is True
    assert out["evaluation"]["exact_match"]["main.py"] is True
    assert out["evaluation"]["similarity"]["main.py"] == pytest.approx(1.0)


def test_build_graph_dispatch_builds_for_all_methods():
    def _make_sentinel(name):
        def _node(state):
            state["resolved_contents"] = state.get("parent_a_contents", {})
            state["status"] = f"resolved_{name}"
            return state

        return _node

    with patch.object(pc, "resolve_conflict_agent_node", _make_sentinel("agent")), patch.object(
        pc, "resolve_conflict_base_a_node", _make_sentinel("base_a")
    ), patch.object(pc, "resolve_conflict_base_b_node", _make_sentinel("base_b")), patch.object(
        pc, "resolve_conflict_bypass7_multi_agent_node", _make_sentinel("bypass7")
    ), patch.object(pc, "resolve_conflict_force_mix_node", _make_sentinel("force_mix")):
        for method in ("agent", "base_a", "base_b", "bypass7", "force_mix"):
            app = pc.build_graph(eval_method=method)
            assert app is not None


def test_build_graph_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown eval_method"):
        pc.build_graph(eval_method="not_a_real_method")  # type: ignore[arg-type]
