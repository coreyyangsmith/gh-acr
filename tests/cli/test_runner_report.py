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
    def _price_key(name: str) -> str:
        if name.startswith("openai/"):
            return name
        if name.startswith("groq:"):
            return "groq/" + name.split(":", 1)[1]
        if name.startswith("local:"):
            return name
        return f"openai/{name}"

    assert _price_key("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _price_key("groq:llama") == "groq/llama"
    assert _price_key("local:foo") == "local:foo"
    assert _price_key("gpt-4o-mini") == "openai/gpt-4o-mini"


@pytest.mark.parametrize(
    "method,expected_bypass",
    [
        ("agent", "NA"),
        ("base_a", "NA"),
        ("bypass7", "MIX"),
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
