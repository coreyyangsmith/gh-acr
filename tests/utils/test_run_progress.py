"""Tests for RunProgress tracker (ETA, concurrency, snapshots)."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from src.utils import run_progress as rp
from src.utils.run_progress import (
    RunProgress,
    format_duration,
    get_active_progress,
    get_current_worker_id,
    reset_current_worker_id,
    set_current_worker_id,
    set_stage,
)


@pytest.fixture(autouse=True)
def _clear_active():
    """Ensure no leftover active progress between tests."""
    prev = rp._active
    rp._active = None
    yield
    if rp._active is not None:
        try:
            rp._active.deactivate()
        except Exception:
            pass
    rp._active = prev


def test_format_duration():
    assert format_duration(5) == "5s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3661) == "1h01m"
    assert format_duration(-1) == "--"


def test_eta_before_and_after_completions():
    prog = RunProgress(10, disable_tqdm=True)
    assert prog.format_eta() == "ETA --"
    w = prog.acquire_worker()
    prog.mark_started(w, "s1", "agent")
    time.sleep(0.05)
    prog.mark_done(w, ok=True, elapsed_s=0.05)
    prog.release_worker(w)
    eta = prog.format_eta()
    assert eta.startswith("ETA ")
    assert eta != "ETA --"
    snap = prog.snapshot_line()
    assert "1/10" in snap
    assert "ok=1" in snap
    assert "fail=0" in snap


def test_thread_safe_mark_done():
    prog = RunProgress(20, disable_tqdm=True)
    errors: list[BaseException] = []

    def _worker(i: int):
        try:
            wid = prog.acquire_worker()
            token = set_current_worker_id(wid)
            try:
                prog.mark_started(wid, f"s{i}", "agent")
                set_stage("resolve")
                time.sleep(0.01)
                prog.mark_done(wid, ok=(i % 2 == 0), elapsed_s=0.01)
            finally:
                prog.release_worker(wid)
                reset_current_worker_id(token)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert prog.done == 20
    assert prog.ok + prog.failed == 20
    assert prog.ok == 10
    assert prog.failed == 10


def test_activate_deactivate_and_set_stage_contextvar():
    prog = RunProgress(5, disable_tqdm=True).activate()
    assert get_active_progress() is prog
    wid = prog.acquire_worker()
    token = set_current_worker_id(wid)
    assert get_current_worker_id() == wid
    prog.mark_started(wid, "42", "bj_no_plan")
    set_stage("patch", detail="2 files")
    assert prog._in_flight[wid].stage == "patch"
    prog.mark_done(wid, ok=True)
    prog.release_worker(wid)
    reset_current_worker_id(token)
    prog.deactivate()
    assert get_active_progress() is None


def test_tqdm_disabled_explicitly():
    prog = RunProgress(3, disable_tqdm=True)
    assert prog._pbar is None
    assert prog._tqdm_disabled is True
    w = prog.acquire_worker()
    prog.mark_started(w, "a", "m")
    prog.mark_done(w, ok=True)
    prog.release_worker(w)
    assert "complete" in prog.summary_line() or "3" in prog.summary_line() or "1/3" in prog.snapshot_line()


def test_set_stage_accepts_log_false():
    prog = RunProgress(2, disable_tqdm=True).activate()
    wid = prog.acquire_worker()
    token = set_current_worker_id(wid)
    try:
        prog.mark_started(wid, "s1", "agent")
        # Module-level helper must accept log= (used by clone progress ticks)
        set_stage("clone", detail="10%", log=False)
        assert prog._in_flight[wid].stage == "clone"
        assert prog._in_flight[wid].detail == "10%"
    finally:
        prog.release_worker(wid)
        reset_current_worker_id(token)
        prog.deactivate()


def test_set_stage_skips_duplicate_log(caplog):
    prog = RunProgress(2, disable_tqdm=True).activate()
    wid = prog.acquire_worker()
    token = set_current_worker_id(wid)
    try:
        prog.mark_started(wid, "s1", "agent")
        with caplog.at_level(logging.INFO, logger="src.utils.run_progress"):
            set_stage("clone", detail="repo/x")
            set_stage("clone", detail="repo/x")  # duplicate — no second INFO
        clone_lines = [r for r in caplog.records if "stage=clone" in r.getMessage()]
        assert len(clone_lines) == 1
    finally:
        prog.release_worker(wid)
        reset_current_worker_id(token)
        prog.deactivate()
