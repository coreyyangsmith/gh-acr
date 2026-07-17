"""Local / cluster transformers LLM handler (``local:<hf_id>``)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .base import BaseLLMHandler


class LocalHandler(BaseLLMHandler):
    """HuggingFace transformers backend for local / cluster runs.

    Heavy loading logic lives in ``backends.local_backend``; this handler
    owns scheme matching and is the registry entry point.
    """

    scheme = "local"
    separator = ":"
    api_key_env = ""  # uses HF tokens via hf_utils, not a single API key

    def require_api_key(self) -> str:
        raise RuntimeError(
            "LocalHandler does not use a single API key; "
            "set HF_API_TOKEN / HUGGINGFACE_HUB_TOKEN if needed for gated models"
        )

    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        # Validate scheme / non-empty id before loading weights
        self.parse_model_id(model_name)
        # Deferred import keeps handlers importable without torch at collection time
        from ..backends.local_backend import create_local_backend

        return create_local_backend(model_name)


__all__ = ["LocalHandler"]
