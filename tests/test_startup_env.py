"""Tests for startup .env loading preferences."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_startup_prefers_root_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`src.startup` should load repo-root `.env` when present."""
    monkeypatch.delenv("GHACR_STARTUP_TEST_MARKER", raising=False)

    # Layout: tmp_path/ is repo root, tmp_path/src/startup.py is the module file
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    root_env = tmp_path / ".env"
    root_env.write_text("GHACR_STARTUP_TEST_MARKER=from-root-env\n", encoding="utf-8")
    (src_dir / ".env").write_text(
        "GHACR_STARTUP_TEST_MARKER=from-src-env\n", encoding="utf-8"
    )

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

    import src.startup as startup

    monkeypatch.setattr(startup, "_STARTUP_HAS_RUN", False)
    monkeypatch.setattr(startup, "load_dotenv", _fake_load_dotenv)
    monkeypatch.setattr(startup, "find_dotenv", lambda *a, **k: "")

    with patch.object(startup, "__file__", str(src_dir / "startup.py")):
        startup._run_startup_once()

    assert loaded
    assert loaded[0] == str(root_env)
    import os

    assert os.getenv("GHACR_STARTUP_TEST_MARKER") == "from-root-env"


def test_startup_falls_back_to_src_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When root `.env` is absent, prefer `src/.env`."""
    monkeypatch.delenv("GHACR_STARTUP_TEST_MARKER", raising=False)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_env = src_dir / ".env"
    src_env.write_text("GHACR_STARTUP_TEST_MARKER=from-src-env\n", encoding="utf-8")

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

    import src.startup as startup

    monkeypatch.setattr(startup, "_STARTUP_HAS_RUN", False)
    monkeypatch.setattr(startup, "load_dotenv", _fake_load_dotenv)
    monkeypatch.setattr(startup, "find_dotenv", lambda *a, **k: "")

    with patch.object(startup, "__file__", str(src_dir / "startup.py")):
        startup._run_startup_once()

    assert loaded
    assert loaded[0] == str(src_env)
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
    # Place startup under a fake src so parents[1] is empty_src's parent without .env
    fake_repo = tmp_path / "repo"
    fake_src = fake_repo / "src"
    fake_src.mkdir(parents=True)
    with patch.object(startup, "__file__", str(fake_src / "startup.py")):
        # Neither repo/.env nor src/.env exists → find_dotenv path
        startup._run_startup_once()

    assert str(fallback) in loaded
