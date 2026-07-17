"""Abstract base class for LLM inference handlers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class BaseLLMHandler(ABC):
    """Provider-specific LLM backend factory.

    Subclasses implement scheme matching and ``create`` to return an
    ``(encoder, runnable)`` pair consumed by ``get_backend``.
    """

    scheme: str = ""
    separator: str = "/"
    api_key_env: str = ""

    def matches(self, model_name: str) -> bool:
        """Return True if *model_name* belongs to this handler."""
        prefix = f"{self.scheme}{self.separator}"
        return bool(model_name) and model_name.startswith(prefix)

    def parse_model_id(self, model_name: str) -> str:
        """Strip the scheme prefix and return the provider model id."""
        prefix = f"{self.scheme}{self.separator}"
        if not model_name.startswith(prefix):
            raise ValueError(
                f"model_name {model_name!r} does not match scheme "
                f"{self.scheme}{self.separator}"
            )
        model_id = model_name[len(prefix) :].strip()
        if not model_id:
            raise ValueError(
                f"{prefix!r} requires a model id, got {model_name!r}"
            )
        return model_id

    def require_api_key(self) -> str:
        """Load the API key from the environment or raise RuntimeError."""
        if not self.api_key_env:
            raise RuntimeError(
                f"{type(self).__name__} does not define api_key_env"
            )
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} missing – cannot load "
                f"{self.scheme} backend"
            )
        return key

    @abstractmethod
    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        """Return ``(encoder, raw_llm)`` for *model_name*."""


__all__ = ["BaseLLMHandler"]
