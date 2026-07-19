"""Tests for CLI runner report schema and helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli.runner import RESULTS_SCHEMA_COLUMNS, run_and_save_report
from tests.test_runner_langfuse import _minimal_success_result


EXPECTED_SCHEMA = [
    "id",
    "repo",
    "file_name",
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
    "eval_method",
    "bypass_method",
    "model_name",
    "tokens_system_prompt",
    "tokens_original",
    "tokens_diff_a",
    "tokens_diff_b",
    "tokens_output",
    "tokens_total",
    "tokens_in",
    "tokens_out",
    "cost_in",
    "cost_out",
    "total_cost",
    "processing_time_s",
    "difficulty",
    "project_size",
]


def test_results_schema_columns_stable():
    assert RESULTS_SCHEMA_COLUMNS == EXPECTED_SCHEMA


def test_price_key_normalization():
    from src.config.model_costs import price_key

    assert price_key("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert price_key("groq:llama") == "groq:llama"
    assert price_key("local:foo") == "local:foo"
    assert price_key("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert (
        price_key("openrouter/meta-llama/llama-3.1-8b-instruct")
        == "openrouter/meta-llama/llama-3.1-8b-instruct"
    )


@pytest.mark.parametrize(
    "method,expected_bypass",
    [
        ("agent", "NA"),
        ("base_a", "NA"),
        ("bypass7", "MIX"),
        ("better_judge", "MIX"),
        ("bj_no_summary", "MIX"),
        ("bj_no_judge", "MIX"),
        ("bj_no_plan", "MIX"),
        ("bj_no_review", "MIX"),
        ("force_mix", "A"),
    ],
)
def test_run_and_save_report_schema_and_artifacts(
    tmp_path: Path, method: str, expected_bypass: str
):
    result = _minimal_success_result(scenario_id="99", eval_method=method)
    if method == "force_mix":
        result["bypass_method"] = "ALL_A"
        result["bypass_decision"] = "ALL_A"

    app = MagicMock()
    app.ainvoke = AsyncMock(return_value=result)

    rows = asyncio.run(
        run_and_save_report(
            app,
            "99",
            tmp_path,
            eval_method=method,
            model_name="openai/gpt-4o-mini",
            write_prep=True,
        )
    )

    assert rows
    row = rows[0]
    for col in RESULTS_SCHEMA_COLUMNS:
        assert col in row
    assert row["bypass_method"] == expected_bypass
    assert row["eval_method"] == method
    assert any((p.parent / "original.txt").exists() for p in tmp_path.rglob("original.txt"))

    # Flat legacy dumps must not be written
    assert not list(tmp_path.rglob("a_summary.txt"))
    assert not list(tmp_path.rglob("resolution1.txt"))

    if method in (
        "bypass7",
        "better_judge",
        "bj_no_summary",
        "bj_no_judge",
        "bj_no_plan",
        "bj_no_review",
        "force_mix",
        "agent",
    ):
        # Runner ensures final/ when nodes did not write during the mocked invoke
        finals = list(tmp_path.rglob("final/resolved.txt"))
        assert finals, f"expected final/resolved.txt for method={method}"
        assert finals[0].read_text(encoding="utf-8") == "merged\n"


def test_runner_passes_artifact_root_in_init_state(tmp_path: Path):
    seen: dict = {}

    async def _ainvoke(state, config=None):
        seen["artifact_root"] = state.get("artifact_root")
        return _minimal_success_result(scenario_id="99", eval_method="bypass7")

    app = MagicMock()
    app.ainvoke = _ainvoke

    asyncio.run(
        run_and_save_report(
            app,
            "99",
            tmp_path,
            eval_method="bypass7",
            model_name="openai/gpt-4o-mini",
            write_prep=False,
        )
    )

    assert seen["artifact_root"] is not None
    assert seen["artifact_root"].endswith("bypass7")
    assert (Path(seen["artifact_root"])).exists()


def test_runner_prepared_state_skips_prep_and_merges_init(tmp_path: Path):
    seen: dict = {}

    async def _ainvoke(state, config=None):
        seen["state"] = dict(state)
        return _minimal_success_result(scenario_id="99", eval_method="agent")

    app = MagicMock()
    app.ainvoke = _ainvoke
    prepared = {
        "sample_row": {
            "id": "99",
            "df_index": 99,
            "name": "o/r",
            "difficulty": "easy",
            "scenario_json": {
                "files_in_merge_conflict": ["f.py"],
                "parents": ["a", "b"],
                "merge_commit_hash": "m",
            },
        },
        "ancestor_contents": {"f.py": "o"},
        "parent_a_contents": {"f.py": "a"},
        "parent_b_contents": {"f.py": "b"},
        "diffs_a": {"f.py": ""},
        "diffs_b": {"f.py": ""},
        "truth_contents": {"f.py": "t"},
        "diffs_truth": {"f.py": ""},
        "commit_messages_a": "",
        "commit_messages_b": "",
        "status": "context_prepared",
    }

    with patch("src.merge_pipeline.pipeline_clone._clone_repo") as clone:
        rows = asyncio.run(
            run_and_save_report(
                app,
                "99",
                tmp_path,
                eval_method="agent",
                model_name="openai/gpt-4o-mini",
                process_mode="clone",
                write_prep=True,
                prepared_state=prepared,
            )
        )
        clone.assert_not_called()

    assert seen["state"]["ancestor_contents"]["f.py"] == "o"
    assert seen["state"]["eval_method"] == "agent"
    assert all(r.get("eval_method") != "prep" for r in rows)
