from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

# Lazy import so consumers without tiktoken can still use word-count fallback
try:  # pragma: no cover
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


@lru_cache(maxsize=None)
def tiktoken_encoder(model_name: str):  # noqa: D401
    """Return a tiktoken encoder for model_name, or None if unavailable."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover
            return None


def count_tokens(encoder: Optional[Any], text: str) -> int:  # noqa: D401
    """Return token count using encoder if available; fallback to words."""
    if encoder is None:
        return len(text.split())
    if hasattr(encoder, "encode"):
        return len(encoder.encode(text))  # type: ignore[attr-defined]
    return len(text.split())


__all__ = ["tiktoken_encoder", "count_tokens"]


