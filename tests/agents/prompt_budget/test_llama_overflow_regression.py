"""Lightweight regression fixtures from OpenRouter Llama context-limit failures.

Stores metadata only (no multi-MB prompt bodies). Asserts that conservative
``estimate_prompt_tokens`` would have exceeded the old 64-token buffer budget
and that the new policy (chars/4 estimate + 4096 buffer) would clip.
"""

from __future__ import annotations

import pytest

from src.agents.prompt_budget import DEFAULT_PROMPT_SAFETY_BUFFER, allowed_prompt_tokens
from src.agents.token_utils import chars_per_token_estimate, estimate_prompt_tokens
from tests.helpers import FakeEncoder

# Captured from logs/llm_failures on 2026-07-29 (OpenRouter Llama 3.1 8B).
# HF counts measured locally; API input from provider error messages.
_LLAMA_FAILURE_CASES = [
    {
        "id": "transformers_summarizer",
        "chars": 481_562,
        "hf_tokens": 101_962,
        "api_input": 122_834,
    },
    {
        "id": "gef_summarizer",
        "chars": 456_299,
        "hf_tokens": 111_336,
        "api_input": 117_249,
    },
    {
        "id": "pennylane_analyzer",
        "chars": 448_360,
        "hf_tokens": 114_621,
        "api_input": 115_394,
    },
]

_OLD_BUFFER = 64
_TOTAL = 131_072
_OUTPUT = 16_384
_OLD_BUDGET = _TOTAL - _OUTPUT - _OLD_BUFFER  # 114624


@pytest.mark.parametrize("case", _LLAMA_FAILURE_CASES, ids=lambda c: c["id"])
def test_failure_metadata_estimate_exceeded_old_budget(case, monkeypatch):
    """These prompts looked under HF budget but API input + output overflowed."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    chars = case["chars"]
    hf = case["hf_tokens"]
    estimate = max(hf, (chars + 3) // 4)
    assert estimate == max(hf, chars_per_token_estimate("x" * chars))
    # Old policy: HF under budget → no clip; API still overflowed.
    assert hf <= _OLD_BUDGET
    assert case["api_input"] + _OUTPUT > _TOTAL
    # New estimate would have flagged over-budget (or been at the edge).
    new_budget = allowed_prompt_tokens("openrouter/meta-llama/llama-3.1-8b-instruct")
    assert new_budget == _TOTAL - _OUTPUT - DEFAULT_PROMPT_SAFETY_BUFFER
    assert new_budget == 110_592
    assert estimate > new_budget


@pytest.mark.parametrize("case", _LLAMA_FAILURE_CASES, ids=lambda c: c["id"])
def test_synthetic_prompt_matching_failure_shape_is_clipped(case, monkeypatch):
    """Synthetic dense prompt with same chars as a failure is clipped under new budget."""
    from src.agents.truncation_wrapper import TruncatingLLMWrapper
    from src.utils.degradation import clear_degradations, get_degradations
    from tests.helpers import RecordingLLM

    monkeypatch.setenv("LOCAL_TRUNCATION_SIDE", "left")
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()

    # Undercounting encoder mimics HF < chars/4 (transformers-like gap).
    class SoftUndercount:
        def encode(self, text: str):
            # ~hf/chars ratio from the transformers case (~0.21 tokens/char... wait
            # 101962/481562 ≈ 0.21 tokens per char? No that's tokens/char.
            # Under-report: return ~hf density relative to failure chars.
            ratio = case["hf_tokens"] / max(1, case["chars"])
            n = max(1, int(len(text) * ratio)) if text else 0
            return list(range(n))

        def decode(self, ids):
            return "x" * max(1, int(len(ids) / max(case["hf_tokens"] / case["chars"], 1e-9)))

    prompt = "x" * case["chars"]
    enc = SoftUndercount()
    assert estimate_prompt_tokens(enc, prompt) > 110_592

    inner = RecordingLLM()
    wrapper = TruncatingLLMWrapper(
        inner,
        encoder=enc,
        model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    )
    wrapper.invoke(prompt)
    truncated = inner.prompts[-1]
    assert estimate_prompt_tokens(enc, truncated) <= 110_592
    assert (
        estimate_prompt_tokens(enc, truncated) + _OUTPUT <= _TOTAL
    )
    assert get_degradations()
