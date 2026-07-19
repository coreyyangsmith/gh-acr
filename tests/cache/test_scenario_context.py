"""Tests for persistent scenario context cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.cache import scenario_context as sc


def _sample(scenario_id: str = "own/repo@abc", merge: str = "merge1") -> dict:
    return {
        "id": scenario_id,
        "name": "own/repo",
        "df_index": 0,
        "scenario_json": {
            "merge_commit_hash": merge,
            "parents": ["p0", "p1"],
            "files_in_merge_conflict": ["src/a.py", "b.py"],
        },
    }


def _prepared_state(sample: dict) -> dict:
    files = sample["scenario_json"]["files_in_merge_conflict"]
    return {
        "scenario_id": sample["id"],
        "sample_row": sample,
        "repo_path": "/tmp/repos/own___repo",
        "ancestor_contents": {f: f"anc-{f}" for f in files},
        "parent_a_contents": {f: f"a-{f}" for f in files},
        "parent_b_contents": {f: f"b-{f}" for f in files},
        "diffs_a": {f: f"da-{f}" for f in files},
        "diffs_b": {f: f"db-{f}" for f in files},
        "truth_contents": {f: f"t-{f}" for f in files},
        "diffs_truth": {f: f"dt-{f}" for f in files},
        "commit_messages_a": "msg a",
        "commit_messages_b": "msg b",
        "status": "context_prepared",
    }


def test_save_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GHACR_CONTEXT_CACHE_DIR", str(tmp_path / "cache"))
    sample = _sample()
    state = _prepared_state(sample)
    root = sc.save_context(sample["id"], state)
    assert (root / "meta.json").is_file()

    loaded = sc.load_context(sample["id"], sample)
    assert loaded is not None
    assert loaded["ancestor_contents"]["src/a.py"] == "anc-src/a.py"
    assert loaded["truth_contents"]["b.py"] == "t-b.py"
    assert loaded["commit_messages_a"] == "msg a"
    assert loaded["sample_row"]["name"] == "own/repo"


def test_load_invalidates_on_merge_sha_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GHACR_CONTEXT_CACHE_DIR", str(tmp_path / "cache"))
    sample = _sample(merge="merge1")
    sc.save_context(sample["id"], _prepared_state(sample))

    changed = _sample(merge="merge2")
    assert sc.load_context(sample["id"], changed) is None


def test_ensure_prepared_uses_cache_on_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GHACR_CONTEXT_CACHE_DIR", str(tmp_path / "cache"))
    sample = _sample()
    sc.save_context(sample["id"], _prepared_state(sample))

    with patch(
        "src.merge_pipeline.pipeline_clone.prepare_context_node"
    ) as prep_node, patch("src.merge_pipeline.pipeline_clone._clone_repo") as clone:
        out = sc.ensure_prepared(sample["id"], sample)
        prep_node.assert_not_called()
        clone.assert_not_called()
        assert out["truth_contents"]["src/a.py"] == "t-src/a.py"
        assert out["from_context_cache"] is True
