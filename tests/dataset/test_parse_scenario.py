"""Unit tests for scenario parsing helpers."""

from __future__ import annotations

import pytest

from src.dataset.loader import _parse_scenario


def test_parse_scenario_valid_python_dict():
    raw = (
        "{'files_in_merge_conflict': ['a.py'], "
        "'parents': ['aa', 'bb'], "
        "'merge_commit_hash': 'mm'}"
    )
    parsed = _parse_scenario(raw)
    assert parsed["files_in_merge_conflict"] == ["a.py"]
    assert parsed["parents"] == ["aa", "bb"]
    assert parsed["merge_commit_hash"] == "mm"


def test_parse_scenario_malformed_raises():
    with pytest.raises(ValueError, match="Unable to parse"):
        _parse_scenario("{not valid")
