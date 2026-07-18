"""Shared test helpers for handler / LLM wrapper tests."""

from __future__ import annotations

from typing import List


class FakeEncoder:
    """Minimal encoder that treats each whitespace-separated token as one id.

    ``encode`` / ``decode`` are invertible for simple whitespace-delimited text,
    which is enough to exercise TruncatingLLMWrapper clipping logic.
    """

    def __init__(self, *, model_max_length: int | None = None):
        self.model_max_length = model_max_length

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        # Preserve empty trailing slots if text ends with space by using split
        parts = text.split()
        return list(range(len(parts)))

    def decode(self, ids: List[int]) -> str:
        # Map ids back to synthetic tokens t0, t1, ...
        return " ".join(f"t{i}" for i in ids)


class RecordingLLM:
    """Runnable stub that records the last prompt passed to ``invoke``."""

    def __init__(self):
        self.prompts: list = []
        self.config = None

    def with_config(self, config=None):
        self.config = config
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return type("Msg", (), {"content": f"echo:{prompt}"})()

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)
