"""Tests for RunLedger resume helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.run_ledger import RunLedger, load_successful_units


def test_load_successful_units_mixed(tmp_path: Path):
    ledger = tmp_path / "run_log.jsonl"
    lines = [
        {"status": "success", "scenario_id": "s1", "eval_method": "agent"},
        {"status": "failure", "scenario_id": "s1", "eval_method": "better_judge"},
        {"status": "degraded", "scenario_id": "s4", "eval_method": "agent"},
        {"status": "success", "scenario_id": "s2", "eval_method": "agent"},
        {"status": "success", "scenario_id": "s3", "eval_method": "prep"},
        {
            "status": "summary",
            "success_count": 2,
            "failure_count": 1,
            "degraded_count": 1,
        },
        "not-json",
        {"status": "success", "scenario_id": "s2", "eval_method": "bj_no_plan"},
    ]
    with ledger.open("w", encoding="utf-8") as fh:
        for item in lines:
            if isinstance(item, str):
                fh.write(item + "\n")
            else:
                fh.write(json.dumps(item) + "\n")

    done = load_successful_units(ledger)
    assert done == {
        ("s1", "agent"),
        ("s2", "agent"),
        ("s2", "bj_no_plan"),
        ("s4", "agent"),
    }
    assert ("s1", "better_judge") not in done


def test_load_successful_units_missing_file(tmp_path: Path):
    assert load_successful_units(tmp_path / "missing.jsonl") == set()


def test_from_existing_reseeds_counts(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    first = RunLedger(path)
    first.record_success(scenario_id="a", eval_method="agent")
    first.record_failure(
        scenario_id="b", eval_method="agent", error=RuntimeError("boom")
    )
    first.record_degraded(
        scenario_id="d",
        eval_method="agent",
        degradation_events=[{"category": "prompt_truncation", "reason": "clipped"}],
    )
    first.record_summary(results_path=str(tmp_path / "out.csv"))

    resumed = RunLedger.from_existing(path)
    assert resumed.success_count == 1
    assert resumed.failure_count == 1
    assert resumed.degraded_count == 1
    resumed.record_success(scenario_id="c", eval_method="agent")
    assert resumed.success_count == 2
    assert resumed.failure_count == 1
    assert resumed.degraded_count == 1


def test_record_failure_classifies_category(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path)
    ledger.record_failure(
        scenario_id="s1",
        eval_method="agent",
        error=RuntimeError("maximum context length exceeded"),
    )
    line = path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["status"] == "failure"
    assert rec["failure_category"] == "token_limit"


def test_failures_path_dual_write(tmp_path: Path):
    ledger_path = tmp_path / "run_log.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    ledger = RunLedger(ledger_path, failures_path=failures_path)

    ledger.record_success(scenario_id="ok", eval_method="agent")
    ledger.record_failure(
        scenario_id="bad",
        eval_method="agent",
        error=RuntimeError("429 rate limit"),
    )
    ledger.record_degraded(
        scenario_id="soft",
        eval_method="agent",
        degradation_events=[
            {"category": "json_parse_fallback", "reason": "bad json"},
        ],
    )
    ledger.record_summary(results_path=str(tmp_path / "out.csv"))

    fail_lines = [
        json.loads(ln)
        for ln in failures_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(fail_lines) == 2
    assert {r["status"] for r in fail_lines} == {"failure", "degraded"}

    all_lines = [
        json.loads(ln)
        for ln in ledger_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(all_lines) == 4
    assert ledger.failure_count == 1
    assert ledger.degraded_count == 1
