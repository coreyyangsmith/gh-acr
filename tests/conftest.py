"""Pytest configuration and shared fixtures for GH-ACR product tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from tests.helpers import FakeEncoder, RecordingLLM


@pytest.fixture(autouse=True)
def _clear_get_backend_cache():
    """Clear lru_cache on get_backend between tests."""
    yield
    try:
        from src.agents.llm_base import get_backend

        get_backend.cache_clear()
    except Exception:
        pass


@pytest.fixture
def clear_api_keys(monkeypatch: pytest.MonkeyPatch):
    """Remove provider API keys from the environment."""
    for key in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_APP_TITLE",
        "HTTP_REFERER",
    ):
        monkeypatch.delenv(key, raising=False)


def _scenario_dict(
    *,
    files: list[str] | None = None,
    parents: list[str] | None = None,
    merge_commit_hash: str = "abc123merge",
) -> str:
    """Return a Python-dict-syntax scenario string (GitGoodBench style)."""
    files = files or ["src/main.py"]
    parents = parents or ["aaaa1111", "bbbb2222"]
    return (
        "{"
        f"'files_in_merge_conflict': {files!r}, "
        f"'parents': {parents!r}, "
        f"'merge_commit_hash': '{merge_commit_hash}'"
        "}"
    )


@pytest.fixture
def tiny_benchmark_csv(tmp_path: Path) -> Path:
    """Write a 3-row GitGoodBench-style CSV and return its path."""
    rows = [
        {
            "id": "s1",
            "name": "owner/repo-a",
            "scenario": _scenario_dict(files=["a.py"], merge_commit_hash="m1"),
            "difficulty": "easy",
            "project_size": "small",
        },
        {
            "id": "s2",
            "name": "owner/repo-b",
            "scenario": _scenario_dict(files=["b.py"], merge_commit_hash="m2"),
            "difficulty": "medium",
            "project_size": "medium",
        },
        {
            "id": "s3",
            "name": "owner/repo-c",
            "scenario": _scenario_dict(files=["c.py"], merge_commit_hash="m3"),
            "difficulty": "hard",
            "project_size": "large",
        },
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "tiny_benchmark.csv"
    df.to_csv(path, index=True)
    return path


@pytest.fixture
def tiny_benchmark_df(tiny_benchmark_csv: Path) -> pd.DataFrame:
    """Load the tiny benchmark via the real loader."""
    from src.dataset.loader import load_benchmark

    return load_benchmark(tiny_benchmark_csv)


@pytest.fixture
def synthetic_agent_state() -> dict[str, Any]:
    """Minimal state dict for agent / multi-agent node tests."""
    path = "src/main.py"
    return {
        "scenario_id": "s1",
        "sample_row": {
            "id": "s1",
            "name": "owner/repo",
            "difficulty": "easy",
            "project_size": "small",
            "scenario_json": {
                "files_in_merge_conflict": [path],
                "parents": ["aaaa", "bbbb"],
                "merge_commit_hash": "mmmm",
            },
        },
        "parent_a_contents": {path: "print('a')\n"},
        "parent_b_contents": {path: "print('b')\n"},
        "ancestor_contents": {path: "print('base')\n"},
        "diffs_a": {path: "--- a\n+++ b\n@@\n-print('base')\n+print('a')\n"},
        "diffs_b": {path: "--- a\n+++ b\n@@\n-print('base')\n+print('b')\n"},
        "model_name": "openai/gpt-4o-mini",
        "status": "prepared",
    }


@pytest.fixture
def recording_llm() -> RecordingLLM:
    """RecordingLLM that returns canned merge-friendly content."""
    llm = RecordingLLM()

    def _invoke(prompt, config=None):
        llm.prompts.append(prompt)
        text = str(prompt).lower()
        if "review" in text or "accept" in text or "reject" in text:
            content = '{"outcome": "ACCEPT", "feedback": "ok"}'
        elif "all_a" in text or "all a" in text or "judge" in text or "bypass" in text:
            content = "MIX"
        elif "plan" in text or "choice" in text:
            content = '{"choice": "A", "rationale": "prefer A"}'
        else:
            content = "print('merged')\n"
        return type("Msg", (), {"content": content})()

    llm.invoke = _invoke  # type: ignore[method-assign]
    llm.ainvoke = lambda prompt, config=None: _invoke(prompt, config)  # type: ignore[method-assign]
    return llm


@pytest.fixture
def mock_get_backend(recording_llm: RecordingLLM):
    """Patch get_backend to return FakeEncoder + RecordingLLM (no network)."""
    encoder = FakeEncoder()
    targets = (
        "src.agents.llm_base.get_backend",
        "src.agents.multi_agent.nodes.get_backend",
        "src.agents.single_agent.merge_agent.get_backend",
    )
    patches = [patch(t, return_value=(encoder, recording_llm)) for t in targets]
    for p in patches:
        p.start()
    try:
        yield patches[0]
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def tiny_local_repo(tmp_path: Path) -> Path:
    """Build a tiny real local git repo (no network) with two parent commits.

    Returns the repo path. Marked for use by pipeline tests that need git.
    """
    pytest.importorskip("git")
    from git import Repo

    repo_dir = tmp_path / "tiny_repo"
    repo_dir.mkdir()
    repo = Repo.init(repo_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test User")
        cw.set_value("user", "email", "test@example.com")

    main_py = repo_dir / "main.py"
    main_py.write_text("print('base')\n", encoding="utf-8")
    repo.index.add(["main.py"])
    base = repo.index.commit("base")

    main_py.write_text("print('a')\n", encoding="utf-8")
    repo.index.add(["main.py"])
    commit_a = repo.index.commit("parent-a")

    repo.git.checkout(base.hexsha, b="side")
    main_py.write_text("print('b')\n", encoding="utf-8")
    repo.index.add(["main.py"])
    repo.index.commit("parent-b")

    repo.git.checkout(commit_a.hexsha)
    return repo_dir
