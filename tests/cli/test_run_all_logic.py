"""Tests for CLI sampling / batching logic in run_all."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.cli import run_all as run_all_mod


def _df_with_difficulty() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "e1", "name": "o/r1", "difficulty": "easy"},
            {"id": "e2", "name": "o/r2", "difficulty": "easy"},
            {"id": "m1", "name": "o/r3", "difficulty": "medium"},
            {"id": "m2", "name": "o/r4", "difficulty": "medium"},
            {"id": "h1", "name": "o/r5", "difficulty": "hard"},
            {"id": "h2", "name": "o/r6", "difficulty": "hard"},
        ]
    )


def test_run_all_difficulty_sampling_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = _df_with_difficulty()
    seen_ids: list[list[str]] = []

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        return [
            {
                "id": scenario_id,
                "repo": "o/r",
                "file_name": "a.py",
                "exact_match": False,
                "similarity": 0.5,
                "bleu3": 0.1,
                "rouge_l": 0.2,
                "eval_method": kwargs.get("eval_method", "base_a"),
                "bypass_method": "NA",
                "model_name": "NA",
                "tokens_system_prompt": 0,
                "tokens_original": 0,
                "tokens_diff_a": 0,
                "tokens_diff_b": 0,
                "tokens_output": 0,
                "tokens_total": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_in": 0.0,
                "cost_out": 0.0,
                "total_cost": 0.0,
                "processing_time_s": 0.0,
                "difficulty": "easy",
                "project_size": "small",
            }
        ]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "BATCH_SIZE", 10
    ):
        # Capture ids by wrapping run_and_save_report
        captured: list[str] = []

        async def _capture(app, scenario_id, output_root, **kwargs):
            captured.append(str(scenario_id))
            return await _fake_report(app, scenario_id, output_root, **kwargs)

        with patch.object(run_all_mod, "run_and_save_report", side_effect=_capture):
            asyncio.run(
                run_all_mod._run_all(
                    max_scenarios=None,
                    mode="clone",
                    methods=["base_a"],
                    model_name=None,
                    results_filename="out.csv",
                    n_easy=1,
                    n_medium=1,
                    n_hard=1,
                    start_index=None,
                    end_index=None,
                )
            )
        seen_ids.append(sorted(captured))

        captured.clear()
        with patch.object(run_all_mod, "run_and_save_report", side_effect=_capture):
            asyncio.run(
                run_all_mod._run_all(
                    max_scenarios=None,
                    mode="clone",
                    methods=["base_a"],
                    model_name=None,
                    results_filename="out2.csv",
                    n_easy=1,
                    n_medium=1,
                    n_hard=1,
                    start_index=None,
                    end_index=None,
                )
            )
        seen_ids.append(sorted(captured))

    assert len(seen_ids[0]) == 3
    assert seen_ids[0] == seen_ids[1]


def test_run_all_max_scenarios_and_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = _df_with_difficulty()
    captured: list[str] = []

    async def _capture(app, scenario_id, output_root, **kwargs):
        captured.append(str(scenario_id))
        return []

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_capture), patch.object(
        run_all_mod, "BATCH_SIZE", 2
    ):
        asyncio.run(
            run_all_mod._run_all(
                max_scenarios=2,
                mode="clone",
                methods=["base_a"],
                model_name=None,
                results_filename="max.csv",
                n_easy=None,
                n_medium=None,
                n_hard=None,
                start_index=None,
                end_index=None,
            )
        )
    assert captured == ["e1", "e2"]

    captured.clear()
    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_capture), patch.object(
        run_all_mod, "BATCH_SIZE", 10
    ):
        asyncio.run(
            run_all_mod._run_all(
                max_scenarios=None,
                mode="clone",
                methods=["base_a"],
                model_name=None,
                results_filename="slice.csv",
                n_easy=None,
                n_medium=None,
                n_hard=None,
                start_index=2,
                end_index=4,
            )
        )
    assert captured == ["m1", "m2"]
