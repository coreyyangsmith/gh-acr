"""Tests for dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.dataset.loader import load_benchmark


def test_load_benchmark_parses_scenario_and_ids(tiny_benchmark_csv: Path):
    df = load_benchmark(tiny_benchmark_csv)
    assert len(df) == 3
    assert "scenario_json" in df.columns
    assert isinstance(df.iloc[0]["scenario_json"], dict)
    assert df.iloc[0]["scenario_json"]["files_in_merge_conflict"] == ["a.py"]
    assert df["id"].dtype == object or str(df["id"].dtype).startswith("str")
    assert list(df["id"]) == ["s1", "s2", "s3"]


def test_load_benchmark_missing_id_uses_index(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "name": "owner/r",
                "scenario": "{'files_in_merge_conflict': ['x.py'], 'parents': ['a','b'], 'merge_commit_hash': 'm'}",
                "difficulty": "easy",
            }
        ]
    )
    path = tmp_path / "no_id.csv"
    df.to_csv(path, index=True)
    loaded = load_benchmark(path)
    assert "id" in loaded.columns
    assert loaded.iloc[0]["id"] == "0"


def test_load_benchmark_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_benchmark("/nonexistent/path/does_not_exist.csv")
