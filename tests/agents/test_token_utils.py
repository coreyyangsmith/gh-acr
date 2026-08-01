"""Tests for token counting helpers and encoder resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents import token_utils
from src.agents.token_utils import (
    chars_per_token_estimate,
    count_tokens,
    estimate_prompt_tokens,
    hf_repo_for_model,
    resolve_encoder,
    tiktoken_encoder,
)
from tests.helpers import FakeEncoder


def test_count_tokens_with_encoder():
    enc = FakeEncoder()
    assert count_tokens(enc, "one two three") == 3
    assert count_tokens(enc, "") == 0


def test_count_tokens_without_encoder_falls_back_to_words():
    assert count_tokens(None, "one two three") == 3
    assert count_tokens(None, "") == 0


def test_count_tokens_encode_exception_falls_back_to_words():
    class Boom:
        def encode(self, text: str):
            raise RuntimeError("boom")

    assert count_tokens(Boom(), "alpha beta gamma") == 3


def test_estimate_prompt_tokens_takes_max_of_hf_and_chars4():
    enc = FakeEncoder()
    # Sparse whitespace: word count vs chars/4 — take the max.
    sparse = "one two three four"
    assert estimate_prompt_tokens(enc, sparse) == max(4, (len(sparse) + 3) // 4)
    assert estimate_prompt_tokens(enc, sparse) == 5  # chars/4 wins (len=18)
    # Dense code-like: chars/4 wins over FakeEncoder's single "word".
    dense = "x" * 1000
    assert count_tokens(enc, dense) == 1
    assert estimate_prompt_tokens(enc, dense) == (1000 + 3) // 4
    assert estimate_prompt_tokens(None, "") == 0


def test_chars_per_token_estimate():
    assert chars_per_token_estimate("") == 0
    assert chars_per_token_estimate("abcd") == 1
    assert chars_per_token_estimate("abcde") == 2


def test_tiktoken_encoder_returns_usable_or_none():
    enc = tiktoken_encoder("gpt-4o-mini")
    if enc is None:
        # Environment without tiktoken encodings is acceptable
        assert True
    else:
        assert hasattr(enc, "encode")
        assert len(enc.encode("hello")) >= 1


@pytest.mark.parametrize(
    "model_name,repo",
    [
        ("openrouter/qwen/qwen3-32b", "Qwen/Qwen3-32B"),
        (
            "openrouter/meta-llama/llama-3.1-8b-instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        ("groq:qwen/qwen3-32b", "Qwen/Qwen3-32B"),
        ("groq:llama-3.1-8b-instant", "meta-llama/Llama-3.1-8B-Instruct"),
    ],
)
def test_hf_repo_for_configured_native_models(model_name, repo):
    assert hf_repo_for_model(model_name) == repo


def test_resolve_encoder_qwen_uses_native_hf_not_cl100k():
    fake_tok = MagicMock(name="QwenTokenizer")
    fake_tok.encode = MagicMock(return_value=[1, 2, 3])

    with (
        patch(
            "src.agents.token_utils._load_hf_tokenizer", return_value=fake_tok
        ) as load_hf,
        patch("src.agents.token_utils.tiktoken_encoder") as tik,
    ):
        enc = resolve_encoder("openrouter/qwen/qwen3-32b")

    assert enc is fake_tok
    load_hf.assert_called_once_with("Qwen/Qwen3-32B")
    tik.assert_not_called()


def test_resolve_encoder_llama_uses_native_hf_not_cl100k():
    fake_tok = MagicMock(name="LlamaTokenizer")

    with (
        patch(
            "src.agents.token_utils._load_hf_tokenizer", return_value=fake_tok
        ) as load_hf,
        patch("src.agents.token_utils.tiktoken_encoder") as tik,
    ):
        enc = resolve_encoder("openrouter/meta-llama/llama-3.1-8b-instruct")

    assert enc is fake_tok
    load_hf.assert_called_once_with("meta-llama/Llama-3.1-8B-Instruct")
    tik.assert_not_called()


def test_resolve_encoder_openai_uses_tiktoken_encoding():
    fake = MagicMock(name="TiktokenEnc")
    with patch("src.agents.token_utils._tiktoken_encoding", return_value=fake) as enc_fn:
        result = resolve_encoder("openrouter/openai/gpt-5-nano")
    assert result is fake
    enc_fn.assert_called()


def test_resolve_encoder_unknown_returns_none_not_cl100k():
    with patch("src.agents.token_utils.tiktoken_encoder") as tik:
        enc = resolve_encoder("openrouter/anthropic/claude-sonnet-4.5")
    assert enc is None
    tik.assert_not_called()


def test_resolve_encoder_propagates_native_tokenizer_failure():
    with patch(
        "src.agents.token_utils._load_hf_tokenizer",
        side_effect=RuntimeError(
            "Failed to load Hugging Face tokenizer 'Qwen/Qwen3-32B' for token counting"
        ),
    ):
        with pytest.raises(RuntimeError, match="token counting"):
            resolve_encoder("openrouter/qwen/qwen3-32b")

def test_load_hf_tokenizer_prefers_tokenizers_path(monkeypatch: pytest.MonkeyPatch):
    token_utils._load_hf_tokenizer.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_abc")

    fake_tok = MagicMock(name="TokenizersAdapter")
    with patch(
        "src.agents.token_utils._load_hf_tokenizer_via_tokenizers",
        return_value=fake_tok,
    ) as via_tok:
        with patch(
            "src.agents.token_utils._load_hf_tokenizer_via_transformers"
        ) as via_tf:
            tok = token_utils._load_hf_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    assert tok is fake_tok
    via_tok.assert_called_once_with(
        "meta-llama/Llama-3.1-8B-Instruct", "hf_test_token_abc"
    )
    via_tf.assert_not_called()
    token_utils._load_hf_tokenizer.cache_clear()


def test_load_hf_tokenizer_falls_back_to_transformers(monkeypatch: pytest.MonkeyPatch):
    token_utils._load_hf_tokenizer.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_abc")

    fake_tok = MagicMock(name="TransformersTokenizer")
    with patch(
        "src.agents.token_utils._load_hf_tokenizer_via_tokenizers",
        side_effect=RuntimeError("no tokenizer.json"),
    ):
        with patch(
            "src.agents.token_utils._load_hf_tokenizer_via_transformers",
            return_value=fake_tok,
        ) as via_tf:
            tok = token_utils._load_hf_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    assert tok is fake_tok
    via_tf.assert_called_once_with(
        "meta-llama/Llama-3.1-8B-Instruct", "hf_test_token_abc"
    )
    token_utils._load_hf_tokenizer.cache_clear()


def test_load_hf_tokenizer_forwards_hf_token(monkeypatch: pytest.MonkeyPatch):
    token_utils._load_hf_tokenizer.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_abc")

    fake_tok = MagicMock(name="CachedTokenizer")
    transformers_mod = MagicMock()
    transformers_mod.AutoTokenizer.from_pretrained = MagicMock(return_value=fake_tok)

    with patch(
        "src.agents.token_utils._load_hf_tokenizer_via_tokenizers",
        side_effect=RuntimeError("force transformers fallback"),
    ):
        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            tok = token_utils._load_hf_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    assert tok is fake_tok
    transformers_mod.AutoTokenizer.from_pretrained.assert_called_once_with(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        token="hf_test_token_abc",
    )
    token_utils._load_hf_tokenizer.cache_clear()


def test_load_hf_tokenizer_includes_underlying_error(monkeypatch: pytest.MonkeyPatch):
    token_utils._load_hf_tokenizer.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    with patch(
        "src.agents.token_utils._load_hf_tokenizer_via_tokenizers",
        side_effect=OSError("tokenizer.json missing"),
    ):
        with patch(
            "src.agents.token_utils._load_hf_tokenizer_via_transformers",
            side_effect=OSError(
                "You are trying to access a gated repo. 401 Client Error."
            ),
        ):
            with pytest.raises(RuntimeError, match="gated repo") as ei:
                token_utils._load_hf_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    assert "tokenizers path failed" in str(ei.value)
    assert ei.value.__cause__ is not None
    token_utils._load_hf_tokenizer.cache_clear()


def test_load_hf_tokenizer_reports_broken_transformers_import(
    monkeypatch: pytest.MonkeyPatch,
):
    token_utils._load_hf_tokenizer.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch(
        "src.agents.token_utils._load_hf_tokenizer_via_tokenizers",
        side_effect=ModuleNotFoundError("No module named 'tokenizers'"),
    ):
        with patch(
            "src.agents.token_utils._load_hf_tokenizer_via_transformers",
            side_effect=ModuleNotFoundError(
                "No module named 'torch._higher_order_ops.triton_kernel_wrap'"
            ),
        ):
            with pytest.raises(RuntimeError, match="triton_kernel_wrap") as ei:
                token_utils._load_hf_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    assert "transformers fallback failed" in str(ei.value)
    token_utils._load_hf_tokenizer.cache_clear()


def test_tokenizers_encoder_adapter_encode():
    inner = MagicMock()
    encoding = MagicMock()
    encoding.ids = [11, 22, 33]
    inner.encode.return_value = encoding
    adapter = token_utils._TokenizersEncoderAdapter(inner)
    assert adapter.encode("hello") == [11, 22, 33]
    inner.encode.assert_called_once_with("hello")


@pytest.mark.slow
def test_llama_hf_tokenizer_golden_count_when_available():
    """Optional: real Llama HF tokenizer count for a short fixed string."""
    try:
        enc = resolve_encoder("openrouter/meta-llama/llama-3.1-8b-instruct")
    except Exception as exc:
        pytest.skip(f"Llama HF tokenizer unavailable: {exc}")
    if enc is None:
        pytest.skip("Llama HF tokenizer unavailable")
    text = "def add(a, b):\n    return a + b\n"
    n = count_tokens(enc, text)
    assert n >= 8
    assert estimate_prompt_tokens(enc, text) >= n