"""Tests for durable run heartbeat helpers and watchdog checks."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.run_heartbeat import (
    RunWatchdog,
    WatchdogTimeout,
    clear_cancelled_units,
    format_status_report,
    heartbeat_age_s,
    heartbeat_path_for_results,
    is_unit_cancelled,
    list_active_llm_calls,
    mark_unit_cancelled,
    read_heartbeat,
    register_active_llm_call,
    unregister_active_llm_call,
    write_heartbeat,
)
from src.utils.run_progress import RunProgress


def test_heartbeat_path_for_results():
    p = heartbeat_path_for_results(Path("data/run.csv"))
    assert p.name == "run_heartbeat.json"


def test_write_read_heartbeat(tmp_path: Path):
    path = tmp_path / "hb.json"
    write_heartbeat(path, {"done": 3, "total": 10})
    loaded = read_heartbeat(path)
    assert loaded is not None
    assert loaded["done"] == 3
    assert loaded["total"] == 10
    assert "updated_at" in loaded
    assert "pid" in loaded
    age = heartbeat_age_s(loaded)
    assert age is not None and age < 5.0


def test_active_llm_call_registry():
    register_active_llm_call("c1", node="summarizer", scenario_id="s1", attempt=1)
    calls = list_active_llm_calls()
    assert any(c["call_id"] == "c1" for c in calls)
    unregister_active_llm_call("c1")
    assert all(c["call_id"] != "c1" for c in list_active_llm_calls())


def test_run_progress_writes_heartbeat(tmp_path: Path):
    hb = tmp_path / "progress_heartbeat.json"
    prog = RunProgress(2, disable_tqdm=True, heartbeat_path=hb, heartbeat_min_interval_s=0.01)
    prog.activate()
    wid = prog.acquire_worker()
    prog.mark_started(wid, "repo/scen-1", "agent")
    prog.write_heartbeat(force=True)
    data = json.loads(hb.read_text(encoding="utf-8"))
    assert data["total"] == 2
    assert data["in_flight_count"] == 1
    assert data["in_flight"][0]["scenario"] == "repo/scen-1"
    prog.mark_done(wid, ok=True)
    prog.deactivate()
    data2 = json.loads(hb.read_text(encoding="utf-8"))
    assert data2["done"] == 1
    assert data2["ok"] == 1


def test_abandon_in_flight(tmp_path: Path):
    hb = tmp_path / "hb.json"
    prog = RunProgress(2, disable_tqdm=True, heartbeat_path=hb)
    prog.activate()
    wid = prog.acquire_worker()
    prog.mark_started(wid, "s1", "base_a")
    abandoned = prog.abandon_in_flight(scenario="s1", method="base_a", reason="timeout")
    assert abandoned == wid
    assert prog.failed == 1
    assert prog.done == 1
    prog.deactivate()


def test_format_status_report_with_ledger(tmp_path: Path):
    hb_path = tmp_path / "hb.json"
    write_heartbeat(hb_path, {"done": 1, "total": 5, "ok": 1, "failed": 0, "in_flight_count": 0})
    ledger = tmp_path / "run_log.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "status": "success",
                "eval_method": "agent",
                "scenario_id": "s1",
                "timestamp": "2026-07-19T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = format_status_report(
        heartbeat=read_heartbeat(hb_path),
        ledger_path=ledger,
        failures_path=tmp_path / "failures.jsonl",
    )
    assert "heartbeat:" in report
    assert "latest: status=success" in report


def test_watchdog_skip_mode_soft_skips_overdue_unit(tmp_path: Path):
    hb = tmp_path / "hb.json"
    write_heartbeat(
        hb,
        {
            "done": 0,
            "total": 1,
            "in_flight": [
                {
                    "worker": "W1",
                    "scenario": "repo/s1",
                    "method": "base_a",
                    "stage": "evaluate",
                    "age_s": 9999.0,
                }
            ],
            "active_llm_calls": [],
        },
    )
    skipped: list[dict] = []
    wd = RunWatchdog(
        heartbeat_path=hb,
        stale_s=600.0,
        max_unit_age_s=100.0,
        poll_s=60.0,
        mode="skip",
        on_skip=skipped.append,
    )
    wd._poll_once()
    assert len(skipped) == 1
    assert skipped[0]["scenario"] == "repo/s1"
    assert skipped[0]["method"] == "base_a"
    # Second poll should not duplicate
    wd._poll_once()
    assert len(skipped) == 1


def test_mark_unit_cancelled_registry():
    clear_cancelled_units()
    assert not is_unit_cancelled("s1", "agent")
    assert mark_unit_cancelled("s1", "agent", reason="llm overtime") is True
    assert is_unit_cancelled("s1", "agent")
    assert mark_unit_cancelled("s1", "agent", reason="again") is False
    assert not is_unit_cancelled("s1", "other")
    clear_cancelled_units()
    assert not is_unit_cancelled("s1", "agent")


def test_watchdog_skip_calls_on_skip_for_cancel_hook(tmp_path: Path):
    """Soft-skip on_skip is where run_all marks the cancel registry."""
    clear_cancelled_units()
    hb = tmp_path / "hb.json"
    write_heartbeat(
        hb,
        {
            "done": 0,
            "total": 1,
            "in_flight": [
                {
                    "worker": "W1",
                    "scenario": "repo/s1",
                    "method": "agent",
                    "stage": "evaluate",
                    "age_s": 9999.0,
                }
            ],
            "active_llm_calls": [],
        },
    )

    def on_skip(item: dict) -> None:
        mark_unit_cancelled(
            item["scenario"], item["method"], reason=item.get("reason")
        )

    wd = RunWatchdog(
        heartbeat_path=hb,
        stale_s=600.0,
        max_unit_age_s=100.0,
        poll_s=60.0,
        mode="skip",
        on_skip=on_skip,
    )
    wd._poll_once()
    assert is_unit_cancelled("repo/s1", "agent")
    clear_cancelled_units()


def test_watchdog_timeout_not_retryable_via_status():
    from src.agents.resilient_invoke import is_retryable_error

    err = WatchdogTimeout("soft-skip")
    assert err.status_code == 408
    assert is_retryable_error(err) is False


def test_watchdog_skip_mode_ignores_stale_heartbeat_alone(tmp_path: Path):
    hb = tmp_path / "hb.json"
    write_heartbeat(hb, {"done": 0, "total": 1, "in_flight": [], "active_llm_calls": []})
    payload = json.loads(hb.read_text(encoding="utf-8"))
    payload["updated_at"] = "2020-01-01T00:00:00+00:00"
    hb.write_text(json.dumps(payload), encoding="utf-8")
    skipped: list[dict] = []
    wd = RunWatchdog(
        heartbeat_path=hb,
        stale_s=1.0,
        max_unit_age_s=100.0,
        poll_s=60.0,
        mode="skip",
        on_skip=skipped.append,
    )
    wd._poll_once()
    assert skipped == []


def test_concurrent_heartbeat_writes_remain_valid(tmp_path: Path):
    import threading

    hb = tmp_path / "hb.json"
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            for n in range(20):
                write_heartbeat(hb, {"done": i, "n": n, "total": 100})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    loaded = read_heartbeat(hb)
    assert loaded is not None
    assert "updated_at" in loaded
    assert "done" in loaded


def test_parse_concatenated_heartbeat_json(tmp_path: Path):
    from src.utils.run_heartbeat import _parse_heartbeat_text

    a = json.dumps({"done": 1, "total": 2}, indent=2)
    b = json.dumps({"done": 9, "total": 9}, indent=2)
    parsed = _parse_heartbeat_text(a + "\n" + b)
    assert parsed["done"] == 1
