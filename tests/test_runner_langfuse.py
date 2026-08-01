"""Tests for LangFuse run-context lifecycle in the CLI runner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.observability import clear_run_context, get_run_context
from src.agents.observability import langfuse_tracing as lt
from src.cli.runner import run_and_save_report


@pytest.fixture(autouse=True)
def _reset_context():
    clear_run_context()
    lt.reset_langfuse_circuit_breaker()
    yield
    clear_run_context()
    lt.reset_langfuse_circuit_breaker()


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
        "bypass_decision": "MIX" if eval_method in (
            "bypass7", "better_judge", "force_mix",
            "bj_no_summary", "bj_no_judge", "bj_no_plan", "bj_no_review",
        ) else "",
        "bypass_method": "MIX" if eval_method in (
            "bypass7", "better_judge", "force_mix",
            "bj_no_summary", "bj_no_judge", "bj_no_plan", "bj_no_review",
        ) else "NA",
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


@pytest.mark.parametrize(
    "method",
    [
        "agent",
        "bypass7",
        "better_judge",
        "bj_no_summary",
        "bj_no_judge",
        "bj_no_plan",
        "bj_no_review",
        "force_mix",
        "base_a",
    ],
)
def test_runner_sets_context_during_ainvoke(tmp_path: Path, method: str):
    seen: dict[str, str] = {}

    async def _ainvoke(state, config=None):
        ctx = get_run_context()
        seen.update(ctx)
        assert state.get("eval_method") == method
        assert config["run_name"] == method
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


@pytest.mark.parametrize(
    "method",
    [
        "agent",
        "bypass7",
        "better_judge",
        "bj_no_summary",
        "bj_no_judge",
        "bj_no_plan",
        "bj_no_review",
        "force_mix",
    ],
)
def test_runner_ainvoke_config_includes_langfuse_trace_name(
    tmp_path: Path, method: str, monkeypatch: pytest.MonkeyPatch
):
    """When LangFuse is enabled, graph ainvoke gets method-named langfuse_trace_name."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)

    captured_config: dict = {}
    fake_handler = object()
    fake_cls = MagicMock(return_value=fake_handler)

    async def _ainvoke(state, config=None):
        captured_config.update(config or {})
        return _minimal_success_result(scenario_id="99", eval_method=method)

    app = MagicMock()
    app.ainvoke = _ainvoke

    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        asyncio.run(
            run_and_save_report(
                app,
                "99",
                tmp_path,
                eval_method=method,
                model_name="openai/gpt-4o-mini",
                write_prep=False,
            )
        )

    expected = method
    assert captured_config["run_name"] == expected
    assert captured_config["metadata"]["langfuse_trace_name"] == expected
    assert captured_config["metadata"]["langfuse_session_id"] == "99"
    assert captured_config["metadata"]["eval_method"] == method
    assert method in captured_config["tags"]
    assert "scenario:99" in captured_config["metadata"]["langfuse_tags"]
    assert fake_handler in captured_config["callbacks"]
    # One shared handler per scenario (created once in set_run_context)
    assert fake_cls.call_count == 1


def test_runner_enters_scenario_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from contextlib import contextmanager

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)

    app = MagicMock()
    app.ainvoke = AsyncMock(return_value=_minimal_success_result(scenario_id="42"))
    entered: dict = {"count": 0}

    @contextmanager
    def _fake_obs(name: str):
        entered["count"] += 1
        entered["name"] = name
        yield None

    with patch.object(lt, "_import_callback_handler", return_value=MagicMock(return_value=object())):
        # Local import inside run_and_save_report resolves this package attribute at call time
        with patch("src.agents.observability.scenario_observation", _fake_obs):
            asyncio.run(
                run_and_save_report(
                    app,
                    "42",
                    tmp_path,
                    eval_method="bypass7",
                    model_name="openai/x",
                    write_prep=False,
                )
            )

    assert entered["count"] == 1
    assert entered["name"] == "bypass7"


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
