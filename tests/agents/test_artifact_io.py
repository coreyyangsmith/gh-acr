"""Tests for per-agent artifact I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.artifact_io import (
    agent_call_dir,
    file_path_to_slug,
    write_agent_call,
    write_final_artifacts,
)
from src.agents.resilient_invoke import resilient_invoke


def test_file_path_to_slug():
    assert file_path_to_slug("foo/bar.py") == "foo_bar.py"
    assert file_path_to_slug(r"foo\bar.py") == "foo_bar.py"


def test_agent_call_dir_scenario_and_file_level(tmp_path: Path):
    root = tmp_path / "method"
    assert agent_call_dir(root, agent="analyzer") == root / "analyzer"
    assert agent_call_dir(root, agent="summarizer", file_slug="a_py", call_id="a") == (
        root / "a_py" / "summarizer" / "a"
    )
    assert agent_call_dir(None, agent="analyzer") is None


def test_write_agent_call_layout(tmp_path: Path):
    call_dir = tmp_path / "summarizer" / "a"
    dest = write_agent_call(
        call_dir,
        input_text="PROMPT",
        output_text="OUTPUT",
        artifacts={"a.diff": "--- a\n+++ b\n"},
        metadata={"agent": "summarizer", "llm_used": True, "prompt_tokens": 3},
    )
    assert dest == call_dir
    assert (call_dir / "input.txt").read_text(encoding="utf-8") == "PROMPT"
    assert (call_dir / "output.txt").read_text(encoding="utf-8") == "OUTPUT"
    assert (call_dir / "artifacts" / "a.diff").read_text(encoding="utf-8").startswith("---")
    meta = json.loads((call_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["agent"] == "summarizer"
    assert meta["prompt_tokens"] == 3
    assert "timestamp" in meta


def test_write_final_artifacts(tmp_path: Path):
    write_final_artifacts(
        tmp_path,
        file_path="pkg/mod.py",
        resolved_text="merged\n",
        final_diff="diff\n",
    )
    final = tmp_path / "pkg_mod.py" / "final"
    assert (final / "resolved.txt").read_text(encoding="utf-8") == "merged\n"
    assert (final / "final_diff.txt").read_text(encoding="utf-8") == "diff\n"


def test_resilient_invoke_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="MODEL OUT")

    # Simulate RateLimitAndCostHandler having recorded a call
    monkeypatch.setattr(
        "src.agents.observability.get_llm_calls",
        lambda: [
            {
                "node": "summarizer_agent",
                "model_name": "openai/gpt-4o-mini",
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cost_in": 0.001,
                "cost_out": 0.002,
                "total_cost": 0.003,
                "usage_from_api": True,
            }
        ],
    )

    call_dir = tmp_path / "file" / "summarizer" / "a"
    result = resilient_invoke(
        llm,
        "hello prompt",
        context={
            "node": "summarizer_agent",
            "agent": "summarizer",
            "scenario_id": "1",
            "eval_method": "bypass7",
            "file_path": "file.py",
            "call_id": "a",
            "model_name": "openai/gpt-4o-mini",
        },
        artifact_dir=call_dir,
        artifacts={"original.txt": "orig"},
        max_retries=0,
    )
    assert result.content == "MODEL OUT"
    assert (call_dir / "input.txt").read_text(encoding="utf-8") == "hello prompt"
    assert (call_dir / "output.txt").read_text(encoding="utf-8") == "MODEL OUT"
    assert (call_dir / "artifacts" / "original.txt").read_text(encoding="utf-8") == "orig"
    meta = json.loads((call_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "success"
    assert meta["llm_used"] is True
    assert meta["prompt_tokens"] == 10
    assert meta["completion_tokens"] == 4
    assert meta["elapsed_s"] is not None
