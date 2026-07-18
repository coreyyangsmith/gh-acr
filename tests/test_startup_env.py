"""Tests for startup .env loading preferences."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_startup_prefers_src_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`src.startup` should load `src/.env` when present."""
    # Isolate from any real process env pollution for this key
    monkeypatch.delenv("GHACR_STARTUP_TEST_MARKER", raising=False)

    src_dir = Path(__file__).resolve().parents[1] / "src"
    env_file = src_dir / ".env"

    # Do not overwrite a real developer .env; skip if one already exists with content
    # Instead, patch Path resolution used by startup.
    fake_env = tmp_path / ".env"
    fake_env.write_text("GHACR_STARTUP_TEST_MARKER=from-src-env\n", encoding="utf-8")

    loaded: list[str] = []

    def _fake_load_dotenv(path=None, *args, **kwargs):
        loaded.append(str(path) if path is not None else "<default>")
        if path is not None and Path(path).is_file():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                monkeypatch.setenv(k.strip(), v.strip())
        return True

    # Reset startup guard and re-run with patched paths
    import src.startup as startup

    monkeypatch.setattr(startup, "_STARTUP_HAS_RUN", False)
    monkeypatch.setattr(startup, "load_dotenv", _fake_load_dotenv)
    monkeypatch.setattr(startup, "find_dotenv", lambda *a, **k: "")

    fake_startup_file = tmp_path / "startup.py"
    fake_startup_file.write_text("# stub\n", encoding="utf-8")

    with patch.object(startup, "__file__", str(tmp_path / "startup.py")):
        # src/.env relative to startup file parent == tmp_path/.env
        startup._run_startup_once()

    assert loaded
    assert loaded[0] == str(fake_env)
    import os

    assert os.getenv("GHACR_STARTUP_TEST_MARKER") == "from-src-env"


def test_startup_falls_back_to_find_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.startup as startup

    fallback = tmp_path / "elsewhere.env"
    fallback.write_text("GHACR_FALLBACK_MARKER=1\n", encoding="utf-8")

    loaded: list[str] = []

    def _fake_load_dotenv(path=None, *args, **kwargs):
        loaded.append(str(path) if path is not None else "<default>")
        return True

    monkeypatch.setattr(startup, "_STARTUP_HAS_RUN", False)
    monkeypatch.setattr(startup, "load_dotenv", _fake_load_dotenv)
    monkeypatch.setattr(startup, "find_dotenv", lambda *a, **k: str(fallback))

    empty_src = tmp_path / "empty_src"
    empty_src.mkdir()
    with patch.object(startup, "__file__", str(empty_src / "startup.py")):
        # empty_src/.env does not exist → find_dotenv path
        startup._run_startup_once()

    assert str(fallback) in loaded
