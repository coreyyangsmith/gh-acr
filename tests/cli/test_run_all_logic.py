"""Tests for CLI sampling / batching / concurrency logic in run_all."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.observability import (
    append_llm_call,
    clear_run_context,
    set_run_context,
)
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


def _result_row(scenario_id: str, eval_method: str = "base_a") -> dict:
    return {
        "id": scenario_id,
        "repo": "o/r",
        "file_name": "a.py",
        "exact_match": False,
        "similarity": 0.5,
        "bleu3": 0.1,
        "rouge_l": 0.2,
        "eval_method": eval_method,
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
        "trace_replay_enabled": False,
        "trace_replay_strategy": "",
        "trace_replay_fallback": "",
    }


def _run_kwargs(**overrides):
    base = dict(
        max_scenarios=None,
        mode="clone",
        methods=["base_a"],
        model_name=None,
        results_filename="out.csv",
        n_easy=None,
        n_medium=None,
        n_hard=None,
        start_index=None,
        end_index=None,
        sample_percent=None,
        sample_seed=42,
        concurrency=1,
        method_concurrency=1,
        resume=False,
        trace_replay=False,
    )
    base.update(overrides)
    return base


def _fake_prepared(scenario_id, sample_row=None):
    """Minimal prepared state so run_all tests do not hit real clone/cache."""
    sample = dict(sample_row or {})
    sample.setdefault("id", scenario_id)
    sample.setdefault("name", "o/r")
    sample.setdefault(
        "scenario_json",
        {
            "files_in_merge_conflict": ["a.py"],
            "parents": ["aaa", "bbb"],
            "merge_commit_hash": "mmm",
        },
    )
    sample.setdefault("df_index", scenario_id)
    return {
        "scenario_id": scenario_id,
        "sample_row": sample,
        "ancestor_contents": {"a.py": "base\n"},
        "parent_a_contents": {"a.py": "a\n"},
        "parent_b_contents": {"a.py": "b\n"},
        "diffs_a": {"a.py": ""},
        "diffs_b": {"a.py": ""},
        "truth_contents": {"a.py": "truth\n"},
        "diffs_truth": {"a.py": ""},
        "commit_messages_a": "",
        "commit_messages_b": "",
        "status": "context_prepared",
    }


def test_resolve_concurrency_cli_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_CONCURRENCY", raising=False)
    assert run_all_mod._resolve_concurrency(None) == 8
    assert run_all_mod._resolve_concurrency(8) == 8
    assert run_all_mod._resolve_concurrency(0) == 1
    monkeypatch.setenv("INFERENCE_CONCURRENCY", "6")
    assert run_all_mod._resolve_concurrency(None) == 6
    # CLI wins over env
    assert run_all_mod._resolve_concurrency(2) == 2
    monkeypatch.setenv("INFERENCE_CONCURRENCY", "not-an-int")
    assert run_all_mod._resolve_concurrency(None) == 8


def test_resolve_method_concurrency_defaults():
    assert run_all_mod._resolve_method_concurrency(None, 6) == 6
    assert run_all_mod._resolve_method_concurrency(None, 20) == 8
    assert run_all_mod._resolve_method_concurrency(3, 20) == 3
    assert run_all_mod._resolve_method_concurrency(0, 5) == 1


def test_run_all_difficulty_sampling_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = _df_with_difficulty()
    seen_ids: list[list[str]] = []

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        return [_result_row(str(scenario_id), kwargs.get("eval_method", "base_a"))]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "ensure_prepared", side_effect=_fake_prepared), patch.object(
        run_all_mod, "BATCH_SIZE", 10
    ):
        captured: list[str] = []

        async def _capture(app, scenario_id, output_root, **kwargs):
            captured.append(str(scenario_id))
            return await _fake_report(app, scenario_id, output_root, **kwargs)

        with patch.object(run_all_mod, "run_and_save_report", side_effect=_capture):
            asyncio.run(
                run_all_mod._run_all(
                    **_run_kwargs(
                        results_filename="out.csv",
                        n_easy=1,
                        n_medium=1,
                        n_hard=1,
                    )
                )
            )
        seen_ids.append(sorted(captured))

        captured.clear()
        with patch.object(run_all_mod, "run_and_save_report", side_effect=_capture):
            asyncio.run(
                run_all_mod._run_all(
                    **_run_kwargs(
                        results_filename="out2.csv",
                        n_easy=1,
                        n_medium=1,
                        n_hard=1,
                    )
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
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 2):
        asyncio.run(run_all_mod._run_all(**_run_kwargs(max_scenarios=2, results_filename="max.csv")))
    assert captured == ["e1", "e2"]

    captured.clear()
    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_capture), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    results_filename="slice.csv",
                    start_index=2,
                    end_index=4,
                )
            )
        )
    assert captured == ["m1", "m2"]


def test_run_all_concurrent_writes_all_rows_and_isolates_llm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Concurrent workers write all CSV rows and keep per-scenario token ledgers isolated."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = pd.DataFrame(
        [
            {"id": "s1", "name": "o/r1", "difficulty": "easy"},
            {"id": "s2", "name": "o/r2", "difficulty": "easy"},
            {"id": "s3", "name": "o/r3", "difficulty": "easy"},
        ]
    )
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        sid = str(scenario_id)
        set_run_context(
            eval_method=kwargs.get("eval_method", "agent"),
            scenario_id=sid,
            model_name="openai/gpt-4o-mini",
        )
        # Distinct token counts per scenario so ledger cross-talk would be obvious
        prompt_tokens = {"s1": 11, "s2": 22, "s3": 33}[sid]
        append_llm_call(
            {
                "node": "resolve_conflict_agent",
                "eval_method": kwargs.get("eval_method", "agent"),
                "scenario_id": sid,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            }
        )
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        try:
            # Force overlap between concurrent workers
            await asyncio.sleep(0.15)
        finally:
            with lock:
                active["n"] -= 1
            clear_run_context()
        return [_result_row(sid, kwargs.get("eval_method", "agent"))]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent"],
                    model_name="openai/gpt-4o-mini",
                    results_filename="concurrent.csv",
                    concurrency=3,
                )
            )
        )

    # Workers overlapped (pool actually parallelized)
    assert active["max"] >= 2

    results = pd.read_csv(tmp_path / "data" / "concurrent.csv")
    assert sorted(results["id"].astype(str).tolist()) == ["s1", "s2", "s3"]

    ledger_path = tmp_path / "data" / "concurrent_run_log.jsonl"
    assert ledger_path.exists()
    success_records = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("status") == "success":
            success_records.append(rec)

    assert len(success_records) == 3
    by_id = {str(r["scenario_id"]): r for r in success_records}
    for sid, expected_prompt in (("s1", 11), ("s2", 22), ("s3", 33)):
        calls = by_id[sid].get("llm_calls") or []
        assert len(calls) == 1
        assert calls[0]["scenario_id"] == sid
        assert calls[0]["prompt_tokens"] == expected_prompt


def test_run_all_concurrency_one_matches_row_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Sequential and concurrent modes produce the same scenario id set."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = pd.DataFrame(
        [
            {"id": "a", "name": "o/r1", "difficulty": "easy"},
            {"id": "b", "name": "o/r2", "difficulty": "easy"},
            {"id": "c", "name": "o/r3", "difficulty": "easy"},
        ]
    )

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        # Tiny sleep so concurrency=3 still exercises the pool path
        await asyncio.sleep(0.01)
        return [_result_row(str(scenario_id))]

    def _run(filename: str, concurrency: int) -> set[str]:
        with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
            run_all_mod, "build_graph", return_value=MagicMock()
        ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
            run_all_mod, "ensure_prepared", side_effect=_fake_prepared
        ), patch.object(run_all_mod, "BATCH_SIZE", 10):
            asyncio.run(
                run_all_mod._run_all(
                    **_run_kwargs(
                        results_filename=filename,
                        concurrency=concurrency,
                    )
                )
            )
        out = pd.read_csv(tmp_path / "data" / filename)
        return set(out["id"].astype(str).tolist())

    ids_seq = _run("seq.csv", 1)
    ids_par = _run("par.csv", 3)
    assert ids_seq == ids_par == {"a", "b", "c"}


def test_run_all_progress_total_is_scenarios_times_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Global progress total equals scenarios × methods; mark_done fires per unit."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("GHACR_NO_PROGRESS", "1")
    df = pd.DataFrame(
        [
            {"id": "s1", "name": "o/r1", "difficulty": "easy"},
            {"id": "s2", "name": "o/r2", "difficulty": "easy"},
        ]
    )

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        return [_result_row(str(scenario_id), kwargs.get("eval_method", "agent"))]

    created: list = []

    class _TrackingProgress(run_all_mod.RunProgress):
        def __init__(self, total, **kwargs):
            kwargs.setdefault("disable_tqdm", True)
            super().__init__(total, **kwargs)
            created.append(self)

        def mark_done(self, worker_id, *, ok=True, elapsed_s=None):
            super().mark_done(worker_id, ok=ok, elapsed_s=elapsed_s)

    with patch.object(run_all_mod, "RunProgress", _TrackingProgress), patch.object(
        run_all_mod, "load_benchmark", return_value=df
    ), patch.object(run_all_mod, "build_graph", return_value=MagicMock()), patch.object(
        run_all_mod, "run_and_save_report", side_effect=_fake_report
    ), patch.object(run_all_mod, "ensure_prepared", side_effect=_fake_prepared), patch.object(
        run_all_mod, "BATCH_SIZE", 10
    ):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent", "better_judge"],
                    results_filename="progress.csv",
                    concurrency=2,
                )
            )
        )

    assert len(created) == 1
    prog = created[0]
    assert prog.total == 4  # 2 scenarios × 2 methods
    assert prog.done == 4
    assert prog.ok == 4
    assert prog.failed == 0


def test_run_all_without_resume_wipes_existing_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    results = data / "wipe.csv"
    results.write_text("id,repo\nold,row\n", encoding="utf-8")
    ledger = data / "wipe_run_log.jsonl"
    ledger.write_text(
        json.dumps({"status": "success", "scenario_id": "s1", "eval_method": "agent"})
        + "\n",
        encoding="utf-8",
    )
    failures = data / "wipe_failures.jsonl"
    failures.write_text(
        json.dumps({"status": "failure", "scenario_id": "old", "eval_method": "agent"})
        + "\n",
        encoding="utf-8",
    )

    df = pd.DataFrame([{"id": "s1", "name": "o/r1", "difficulty": "easy"}])

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        return [_result_row(str(scenario_id), kwargs.get("eval_method", "agent"))]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent"],
                    results_filename="wipe.csv",
                    resume=False,
                )
            )
        )

    out = pd.read_csv(results)
    assert "old" not in out["id"].astype(str).tolist()
    assert "s1" in out["id"].astype(str).tolist()
    # Old failures log wiped; no new failures for clean success
    assert not failures.exists() or failures.read_text(encoding="utf-8").strip() == ""


def test_run_all_degraded_writes_flagged_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Soft degradations: flagged CSV rows + ledger degraded + failures JSONL."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("GHACR_NO_PROGRESS", "1")

    df = pd.DataFrame([{"id": "s1", "name": "o/r1", "difficulty": "easy"}])

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        from src.utils.degradation import record_degradation

        record_degradation(
            "prompt_truncation",
            "clipped prompt",
            node="truncating_llm_wrapper",
        )
        return [_result_row(str(scenario_id), kwargs.get("eval_method", "agent"))]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent"],
                    results_filename="degraded.csv",
                    concurrency=1,
                )
            )
        )

    results = tmp_path / "data" / "degraded.csv"
    assert results.exists() and results.stat().st_size > 0
    out = pd.read_csv(results)
    data = out[out["eval_method"] != "prep"]
    assert (data["id"].astype(str) == "s1").any()
    assert bool(data.iloc[0]["soft_degraded"]) is True
    assert data.iloc[0]["degradation_category"] == "prompt_truncation"
    assert int(data.iloc[0]["num_degradations"]) == 1

    ledger_path = tmp_path / "data" / "degraded_run_log.jsonl"
    failures_path = tmp_path / "data" / "degraded_failures.jsonl"
    ledger_recs = [
        json.loads(ln)
        for ln in ledger_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    statuses = [r.get("status") for r in ledger_recs]
    assert "degraded" in statuses
    assert "success" not in statuses or all(
        r.get("status") != "success" or r.get("eval_method") == "prep"
        for r in ledger_recs
    )

    fail_recs = [
        json.loads(ln)
        for ln in failures_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(fail_recs) >= 1
    assert fail_recs[0]["status"] == "degraded"
    assert fail_recs[0]["failure_category"] == "prompt_truncation"
    assert fail_recs[0]["degradation_events"]


def test_run_all_resume_skips_degraded_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Soft-degraded ledger units are skipped on --resume (CSV already written)."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("GHACR_NO_PROGRESS", "1")

    results = data / "resume_deg.csv"
    prior = pd.DataFrame(
        [
            {
                **_result_row("s1", "agent"),
                "soft_degraded": True,
                "degradation_category": "prompt_truncation",
                "num_degradations": 1,
            }
        ]
    )
    prior.to_csv(results, index=False)

    ledger = data / "resume_deg_run_log.jsonl"
    with ledger.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "status": "degraded",
                    "scenario_id": "s1",
                    "eval_method": "agent",
                    "failure_category": "prompt_truncation",
                    "degradation_events": [
                        {"category": "prompt_truncation", "reason": "clipped"},
                    ],
                }
            )
            + "\n"
        )

    failures = data / "resume_deg_failures.jsonl"
    failures.write_text(
        json.dumps(
            {
                "status": "degraded",
                "scenario_id": "s1",
                "eval_method": "agent",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    df = pd.DataFrame([{"id": "s1", "name": "o/r1", "difficulty": "easy"}])
    called: list[tuple[str, str]] = []

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        method = kwargs.get("eval_method", "agent")
        called.append((str(scenario_id), method))
        return [_result_row(str(scenario_id), method)]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent"],
                    results_filename="resume_deg.csv",
                    resume=True,
                    concurrency=1,
                )
            )
        )

    assert called == []
    out = pd.read_csv(results)
    assert (out["id"].astype(str) == "s1").any()
    assert bool(out.iloc[0]["soft_degraded"]) is True


def test_run_all_clean_success_stamps_soft_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Clean successes get soft_degraded=False and empty degradation fields."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("GHACR_NO_PROGRESS", "1")

    df = pd.DataFrame([{"id": "s1", "name": "o/r1", "difficulty": "easy"}])

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        return [_result_row(str(scenario_id), kwargs.get("eval_method", "agent"))]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent"],
                    results_filename="clean.csv",
                    concurrency=1,
                )
            )
        )

    out = pd.read_csv(tmp_path / "data" / "clean.csv")
    data = out[out["eval_method"] != "prep"]
    assert bool(data.iloc[0]["soft_degraded"]) is False
    assert data.iloc[0]["degradation_category"] == "" or pd.isna(
        data.iloc[0]["degradation_category"]
    )
    assert int(data.iloc[0]["num_degradations"]) == 0


def test_run_all_resume_skips_successful_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Resume keeps CSV rows and does not re-invoke done (scenario, method) pairs."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("GHACR_NO_PROGRESS", "1")

    results = data / "resume.csv"
    # Prior CSV row for s1/agent (id column is scenario slug in these tests)
    prior = pd.DataFrame([_result_row("s1", "agent")])
    prior.to_csv(results, index=False)

    ledger = data / "resume_run_log.jsonl"
    with ledger.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "status": "success",
                    "scenario_id": "s1",
                    "eval_method": "agent",
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "status": "failure",
                    "scenario_id": "s2",
                    "eval_method": "agent",
                    "error_type": "RuntimeError",
                    "error_message": "boom",
                }
            )
            + "\n"
        )

    df = pd.DataFrame(
        [
            {"id": "s1", "name": "o/r1", "difficulty": "easy"},
            {"id": "s2", "name": "o/r2", "difficulty": "easy"},
        ]
    )
    called: list[tuple[str, str]] = []

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        method = kwargs.get("eval_method", "agent")
        called.append((str(scenario_id), method))
        return [_result_row(str(scenario_id), method)]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent", "better_judge"],
                    results_filename="resume.csv",
                    resume=True,
                    concurrency=1,
                )
            )
        )

    # s1/agent was successful → skipped; failures and missing units run
    assert ("s1", "agent") not in called
    assert ("s2", "agent") in called
    assert ("s1", "better_judge") in called
    assert ("s2", "better_judge") in called

    # Prior CSV row preserved; new rows appended
    out = pd.read_csv(results)
    assert (out["id"].astype(str) == "s1").any()
    assert (out["eval_method"].astype(str) == "agent").any()
    # New work wrote rows for remaining units
    assert set(called) == {
        ("s2", "agent"),
        ("s1", "better_judge"),
        ("s2", "better_judge"),
    }
    assert len(out) >= 1 + len(called)


def test_run_all_preps_once_per_scenario_runs_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """ensure_prepared once per scenario; run_and_save_report once per method."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("GHACR_NO_PROGRESS", "1")
    df = pd.DataFrame(
        [
            {"id": "s1", "name": "o/r1", "difficulty": "easy"},
            {"id": "s2", "name": "o/r2", "difficulty": "easy"},
        ]
    )
    prep_calls: list[str] = []
    report_calls: list[tuple[str, str]] = []

    def _prep(scenario_id, sample_row=None):
        prep_calls.append(str(scenario_id))
        return _fake_prepared(scenario_id, sample_row)

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        method = kwargs.get("eval_method", "agent")
        report_calls.append((str(scenario_id), method))
        assert kwargs.get("prepared_state") is not None
        assert kwargs.get("write_prep") is False
        return [_result_row(str(scenario_id), method)]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_prep
    ), patch.object(run_all_mod, "BATCH_SIZE", 10):
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(
                    methods=["agent", "better_judge"],
                    results_filename="prep_once.csv",
                    concurrency=1,
                    method_concurrency=2,
                )
            )
        )

    assert sorted(prep_calls) == ["s1", "s2"]
    assert len(prep_calls) == 2
    assert set(report_calls) == {
        ("s1", "agent"),
        ("s1", "better_judge"),
        ("s2", "agent"),
        ("s2", "better_judge"),
    }


def test_run_all_does_not_delete_clones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Batch completion must not call _robust_rmtree on clone dirs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = pd.DataFrame([{"id": "s1", "name": "o/r1", "difficulty": "easy"}])

    async def _fake_report(app, scenario_id, output_root, **kwargs):
        return [_result_row(str(scenario_id))]

    with patch.object(run_all_mod, "load_benchmark", return_value=df), patch.object(
        run_all_mod, "build_graph", return_value=MagicMock()
    ), patch.object(run_all_mod, "run_and_save_report", side_effect=_fake_report), patch.object(
        run_all_mod, "ensure_prepared", side_effect=_fake_prepared
    ), patch.object(run_all_mod, "BATCH_SIZE", 1), patch(
        "src.merge_pipeline.pipeline_clone._robust_rmtree"
    ) as rmtree:
        asyncio.run(
            run_all_mod._run_all(
                **_run_kwargs(results_filename="no_rm.csv", concurrency=1)
            )
        )
        assert rmtree.call_count == 0
