"""Pytest configuration for GH-ACR handler tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `import src...` works from tests/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _clear_get_backend_cache():
    """Clear lru_cache on get_backend between tests."""
    yield
    try:
        from src.agents.llm_base import get_backend

        get_backend.cache_clear()
    except Exception:
        pass


@pytest.fixture
def clear_api_keys(monkeypatch: pytest.MonkeyPatch):
    """Remove provider API keys from the environment."""
    for key in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_APP_TITLE",
        "HTTP_REFERER",
    ):
        monkeypatch.delenv(key, raising=False)
