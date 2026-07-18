"""Tests for LangFuse run-context lifecycle in the CLI runner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.observability import clear_run_context, get_run_context
from src.cli.runner import run_and_save_report


@pytest.fixture(autouse=True)
def _reset_context():
    clear_run_context()
    yield
    clear_run_context()


def _minimal_success_result(*, scenario_id: str = "42", eval_method: str = "agent"):
    """Return a minimal pipeline result that satisfies post-invoke reporting."""
    return {
        "scenario_id": scenario_id,
        "status": "done",
        "eval_method": eval_method,
        "sample_row": {
            "df_index": 0,
            "name": "org/repo",
            "difficulty": "easy",
            "project_size": "small",
            "scenario_json": {
                "files_in_merge_conflict": ["a.py"],
                "parents": ["aaa", "bbb"],
                "merge_commit_hash": "ccc",
            },
        },
        "ancestor_contents": {"a.py": "orig\n"},
        "parent_a_contents": {"a.py": "a\n"},
        "parent_b_contents": {"a.py": "b\n"},
        "truth_contents": {"a.py": "truth\n"},
        "diffs_a": {"a.py": ""},
        "diffs_b": {"a.py": ""},
        "resolved_contents": {"a.py": "merged\n"},
        "final_diffs": {"a.py": ""},
        "token_counts": {},
        "evaluation": {
            "exact_match": {"a.py": False},
            "similarity": {"a.py": 0.5},
            "bleu3": {"a.py": 0.1},
            "rouge_l": {"a.py": 0.2},
            "overall_exact_match": False,
            "overall_bleu3": 0.1,
            "overall_rouge_l": 0.2,
        },
        "bypass_decision": "MIX" if eval_method in ("bypass7", "force_mix") else "",
        "bypass_method": "MIX" if eval_method in ("bypass7", "force_mix") else "NA",
        "summaries": {"a.py": {"summary_a": "sa", "summary_b": "sb"}},
        "reviews": {"a.py": "{}"},
        "review_results": {"a.py": {"outcome": "ACCEPT", "rationale": ""}},
        "resolution_history": {"a.py": ["merged\n"]},
        "conflict_plan": {"a.py": "merge"},
        "commit_messages": {},
        "diffs_truth": {"a.py": ""},
        "commit_messages_a": "",
        "commit_messages_b": "",
    }


@pytest.mark.parametrize("method", ["agent", "bypass7", "force_mix", "base_a"])
def test_runner_sets_context_during_ainvoke(tmp_path: Path, method: str):
    seen: dict[str, str] = {}

    async def _ainvoke(state, config=None):
        ctx = get_run_context()
        seen.update(ctx)
        assert state.get("eval_method") == method
        assert config["run_name"] == f"{method}-scenario-99"
        return _minimal_success_result(scenario_id="99", eval_method=method)

    app = MagicMock()
    app.ainvoke = _ainvoke

    rows = asyncio.run(
        run_and_save_report(
            app,
            "99",
            tmp_path,
            eval_method=method,
            model_name="openai/gpt-4o-mini",
            write_prep=False,
        )
    )

    assert seen["eval_method"] == method
    assert seen["scenario_id"] == "99"
    assert seen["model_name"] == "openai/gpt-4o-mini"
    # Context cleared after the run
    assert get_run_context()["eval_method"] == ""
    assert isinstance(rows, list)
    assert rows
    assert rows[0]["eval_method"] == method


def test_runner_clears_context_and_flushes_on_failure(tmp_path: Path):
    async def _ainvoke(state, config=None):
        assert get_run_context()["eval_method"] == "agent"
        raise RuntimeError("pipeline boom")

    app = MagicMock()
    app.ainvoke = _ainvoke

    with patch("src.agents.observability.flush_langfuse") as flush_mock:
        with pytest.raises(RuntimeError, match="pipeline boom"):
            asyncio.run(
                run_and_save_report(
                    app,
                    "7",
                    tmp_path,
                    eval_method="agent",
                    model_name="openai/x",
                    write_prep=False,
                )
            )
        flush_mock.assert_called()

    assert get_run_context()["eval_method"] == ""


def test_runner_flushes_langfuse_after_success(tmp_path: Path):
    app = MagicMock()
    app.ainvoke = AsyncMock(return_value=_minimal_success_result())

    with patch("src.agents.observability.flush_langfuse") as flush_mock:
        asyncio.run(
            run_and_save_report(
                app,
                "42",
                tmp_path,
                eval_method="agent",
                model_name="openai/x",
                write_prep=False,
            )
        )
        flush_mock.assert_called()
