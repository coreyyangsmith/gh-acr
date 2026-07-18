"""Tests for dataset processing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.dataset.processing.get_subset import _normalize_percent_tag, get_subset
from src.dataset.processing.split_dataset_by_difficulty import split_by_difficulty
from src.dataset.processing.extract_merge_scenario_from_ggb import (
    _resolve_csv_path,
    process_ggb,
)
from src.dataset.processing.extract_samples_from_subset import (
    _derive_default_output_path,
    _normalize_ids,
    extract_rows_by_ids,
)


def test_normalize_percent_tag():
    assert _normalize_percent_tag(10.0) == "10"
    assert _normalize_percent_tag(12.5) == "12_5"


def test_get_subset_deterministic(tiny_benchmark_csv: Path, tmp_path: Path):
    out1 = get_subset(
        tiny_benchmark_csv, percent=100, seed=42, output_csv=tmp_path / "s1.csv"
    )
    out2 = get_subset(
        tiny_benchmark_csv, percent=100, seed=42, output_csv=tmp_path / "s2.csv"
    )
    df1 = pd.read_csv(out1, index_col=0)
    df2 = pd.read_csv(out2, index_col=0)
    assert len(df1) == 3
    assert list(df1["id"]) == list(df2["id"])


def test_get_subset_invalid_percent(tiny_benchmark_csv: Path):
    with pytest.raises(ValueError, match="percent"):
        get_subset(tiny_benchmark_csv, percent=0)
    with pytest.raises(ValueError, match="percent"):
        get_subset(tiny_benchmark_csv, percent=101)


def test_split_by_difficulty(tiny_benchmark_csv: Path):
    easy, medium, hard = split_by_difficulty(tiny_benchmark_csv, write_files=False)
    assert list(easy["id"]) == ["s1"]
    assert list(medium["id"]) == ["s2"]
    assert list(hard["id"]) == ["s3"]


def test_split_by_difficulty_write_files(tiny_benchmark_csv: Path):
    split_by_difficulty(tiny_benchmark_csv, write_files=True)
    base = tiny_benchmark_csv.with_suffix("")
    assert Path(f"{base}_easy.csv").exists()
    assert Path(f"{base}_medium.csv").exists()
    assert Path(f"{base}_hard.csv").exists()


def test_resolve_csv_path_bare_name(tmp_path: Path):
    default = tmp_path / "default.csv"
    default.write_text("x\n", encoding="utf-8")
    resolved = _resolve_csv_path(
        "foo", base_dir=tmp_path, default_path=default, require_exists=False
    )
    assert resolved.name == "foo.csv"
    assert resolved.parent == tmp_path.resolve()


def test_process_ggb_keeps_merge_commit_rows(tmp_path: Path, tiny_benchmark_csv: Path):
    # Add a row without merge_commit_hash
    df = pd.read_csv(tiny_benchmark_csv, index_col=0)
    extra = {
        "id": "s4",
        "name": "owner/no-merge",
        "scenario": "{'files_in_merge_conflict': ['z.py'], 'parents': ['a','b']}",
        "difficulty": "easy",
        "project_size": "small",
    }
    df = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)
    inp = tmp_path / "with_and_without.csv"
    out = tmp_path / "filtered.csv"
    df.to_csv(inp, index=True)

    process_ggb(input_csv=str(inp), output_csv=str(out))
    filtered = pd.read_csv(out, index_col=0)
    assert "s4" not in set(filtered["id"].astype(str))
    assert set(filtered["id"].astype(str)) == {"s1", "s2", "s3"}


def test_normalize_ids_case_and_whitespace():
    s = pd.Series(["  AbC  ", "xyz"])
    out = _normalize_ids(
        s, coerce_to_string=True, trim_whitespace=True, case_insensitive=True
    )
    assert list(out) == ["abc", "xyz"]


def test_derive_default_output_path():
    src = Path("/tmp/source.csv")
    ids = Path("/tmp/ids_list.csv")
    out = _derive_default_output_path(src, ids)
    assert out.name == "source_extracted_by_ids_list.csv"


def test_extract_rows_by_ids(tmp_path: Path, tiny_benchmark_csv: Path):
    ids_csv = tmp_path / "ids.csv"
    pd.DataFrame({"id": ["s1", "s3"]}).to_csv(ids_csv, index=False)
    out = extract_rows_by_ids(
        ids_csv=ids_csv,
        source_csv=tiny_benchmark_csv,
        output_csv=tmp_path / "extracted.csv",
    )
    df = pd.read_csv(out, index_col=0)
    assert set(df["id"].astype(str)) == {"s1", "s3"}
