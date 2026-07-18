"""Tests for setup_logger."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.logger import setup_logger


def test_setup_logger_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    setup_logger(level="INFO")
    n_handlers = len(logging.getLogger().handlers)
    setup_logger(level="INFO")
    assert len(logging.getLogger().handlers) == n_handlers
    assert (tmp_path / "logs").exists()


def test_setup_logger_named(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = setup_logger(name="test.module", level="DEBUG")
    assert log.name == "test.module"
    assert log.getEffectiveLevel() <= logging.DEBUG
