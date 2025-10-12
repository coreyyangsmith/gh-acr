from __future__ import annotations

from typing import Any, Optional, Tuple
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
from langchain_community.llms import HuggingFacePipeline  # type: ignore

from .hf_utils import (
    DEFAULT_HF_CACHE_DIR,
    HF_LOCAL_ONLY,
    HF_TRUST_REMOTE_CODE,
    HF_REVISION,
    get_hf_token,
    should_use_accelerate,
    pick_torch_dtype,
    weights_exist_locally,
    get_device_map_from_env,
    collect_model_devices,
    log_gpu_overview,
)
from ..token_utils import tiktoken_encoder
from langchain_core.messages import AIMessage


logger = logging.getLogger(__name__)


def _prepare_tokenizer_and_model(requested: str):
    os.makedirs(DEFAULT_HF_CACHE_DIR, exist_ok=True)
    hf_token = get_hf_token()
    use_accel = should_use_accelerate()
    dtype = pick_torch_dtype()
    local_ok = weights_exist_locally(requested)

    def _load_tok(local_only: bool):
        return AutoTokenizer.from_pretrained(
            requested,
            use_fast=True,
            cache_dir=DEFAULT_HF_CACHE_DIR,
            token=hf_token,
            revision=HF_REVISION,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
            local_files_only=local_only,
        )

    def _load_model(local_only: bool):
        return AutoModelForCausalLM.from_pretrained(
            requested,
            cache_dir=DEFAULT_HF_CACHE_DIR,
            token=hf_token,
            revision=HF_REVISION,
            trust_remote_code=True,
            local_files_only=local_only,
            device_map=get_device_map_from_env(),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

    try:
        tok = _load_tok(local_only=True)
        model = _load_model(local_only=bool(local_ok))
        logger.info("[local] Loaded tokenizer/model from cache for %s (device_map=auto)", requested)
        logger.info("[local] Model devices (cached): %s", ", ".join(collect_model_devices(model)))
    except Exception:
        if HF_LOCAL_ONLY:
            logger.error("[local] Cache-only load failed for %s and HF_LOCAL_ONLY=1; re-raising", requested)
            raise
        tok = _load_tok(local_only=False)
        model = _load_model(local_only=False)
        logger.info("[local] Downloaded tokenizer/model for %s (device_map=auto)", requested)
        logger.info("[local] Model devices (downloaded): %s", ", ".join(collect_model_devices(model)))

    return tok, model


def create_local_backend(model_name: str) -> Tuple[Optional[Any], Optional[Any]]:  # noqa: D401
    """Initialize a local transformers backend for model_name (local:<id>)."""
    requested = (model_name.split(":", 1)[1] or "").strip()
    if not requested:
        raise ValueError("local: backend requires a model id or path, e.g. local:gpt2")

    tok, model = _prepare_tokenizer_and_model(requested)

    npos = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024)))
    reserve_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
    reserve_new = max(1, min(reserve_new, npos - 32))

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id

    tok.truncation_side = os.getenv("LOCAL_TRUNCATION_SIDE", "left")
    tok.model_max_length = npos - reserve_new

    requested_lower = requested.lower()
    is_qwen3 = "qwen3" in requested_lower or "/qwen3-" in requested_lower
    is_gptoss = ("gpt-oss" in requested_lower) or ("/gpt-oss-" in requested_lower) or ("openai/gpt-oss" in requested_lower)

    if is_qwen3:
        class _FakeGen:
            def __init__(self, text: str):
                self.text = text
        class _FakeResponse:
            def __init__(self, text: str):
                self.generations = [[_FakeGen(text)]]
        class Qwen3ChatWrapper:
            def __init__(self, model_obj, tokenizer_obj):
                self._model = model_obj
                self._tok = tokenizer_obj
                self._callbacks: list[Any] = []
                self._enable_thinking = os.getenv("QWEN3_ENABLE_THINKING", "1").strip().lower() in ("1", "true", "yes", "on")
                if self._enable_thinking:
                    self._temperature = float(os.getenv("QWEN3_TEMPERATURE", "0.6"))
                    self._top_p = float(os.getenv("QWEN3_TOP_P", "0.95"))
                    self._top_k = int(os.getenv("QWEN3_TOP_K", "20"))
                else:
                    self._temperature = float(os.getenv("QWEN3_TEMPERATURE", "0.7"))
                    self._top_p = float(os.getenv("QWEN3_TOP_P", "0.8"))
                    self._top_k = int(os.getenv("QWEN3_TOP_K", "20"))
                default_qwen_max = max(reserve_new, 2048)
                self._max_new = int(os.getenv("QWEN3_MAX_NEW_TOKENS", str(default_qwen_max)))
                try:
                    if os.getenv("QWEN3_ENABLE_YARN", "0").strip().lower() in ("1", "true", "yes", "on"):
                        factor = float(os.getenv("QWEN3_YARN_FACTOR", "4.0"))
                        orig = int(os.getenv("QWEN3_YARN_ORIG_CTX", str(getattr(self._model.config, "max_position_embeddings", 32768))))
                        if hasattr(self._model, "config") and hasattr(self._model.config, "rope_scaling"):
                            self._model.config.rope_scaling = {
                                "rope_type": "yarn",
                                "factor": float(factor),
                                "original_max_position_embeddings": int(orig),
                            }
                except Exception:
                    pass

            def with_config(self, config: dict[str, Any] | None = None):  # type: ignore[override]
                try:
                    if config and "callbacks" in config:
                        self._callbacks = list(config.get("callbacks") or [])
                except Exception:
                    pass
                return self

            def invoke(self, prompt_text: str, config: Any | None = None):  # type: ignore[override]
                try:
                    callbacks = list(self._callbacks)
                    try:
                        if config and isinstance(config, dict):
                            callbacks.extend(list(config.get("callbacks") or []))  # type: ignore[arg-type]
                    except Exception:
                        pass

                    messages = [{"role": "user", "content": str(prompt_text)}]
                    chat_text = self._tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )

                    run_id = id(self) ^ hash(chat_text)
                    for cb in callbacks:
                        try:
                            cb.on_llm_start({"name": "qwen3-local"}, [chat_text], run_id=run_id)
                        except Exception:
                            pass

                    pad_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else self._tok.pad_token_id
                    log_gpu_overview("[local.Qwen3ChatWrapper.invoke]", self._model)
                    generator = pipeline("text-generation", model=self._model, tokenizer=self._tok, batch_size=1, truncation=True)
                    outputs = generator(
                        chat_text,
                        truncation=True,
                        padding=True,
                        do_sample=True,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        top_k=self._top_k,
                        max_new_tokens=self._max_new,
                        eos_token_id=self._tok.eos_token_id,
                        pad_token_id=pad_id,
                        num_return_sequences=1,
                    )
                    full_text = outputs[0].get("generated_text", "")
                    answer_text = full_text[len(chat_text):].lstrip() if full_text.startswith(chat_text) else full_text

                    fake_resp = _FakeResponse(answer_text)
                    for cb in callbacks:
                        try:
                            cb.on_llm_end(fake_resp, run_id=run_id)
                        except Exception:
                            pass

                    return AIMessage(content=answer_text)
                except Exception as e:  # pragma: no cover
                    for cb in self._callbacks:
                        try:
                            cb.on_llm_error(e, run_id=id(self))
                        except Exception:
                            pass
                    raise

        raw_llm = Qwen3ChatWrapper(model, tok)
        enc = tok
        logger.info(
            "[local] Initialized Qwen3 chat wrapper for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, thinking=%s)",
            requested, npos, tok.model_max_length, reserve_new, True,
        )
        return enc, raw_llm

    if is_gptoss:
        class _FakeGen:
            def __init__(self, text: str):
                self.text = text
        class _FakeResponse:
            def __init__(self, text: str):
                self.generations = [[_FakeGen(text)]]
        class GPTOSSChatWrapper:
            def __init__(self, model_obj, tokenizer_obj):
                self._model = model_obj
                self._tok = tokenizer_obj
                self._callbacks: list[Any] = []
                self._reasoning_level = os.getenv("GPT_OSS_REASONING_LEVEL", "medium").strip().lower()
                default_max = max(reserve_new, 2048)
                self._max_new = int(os.getenv("GPT_OSS_MAX_NEW_TOKENS", str(default_max)))
                self._temperature = float(os.getenv("GPT_OSS_TEMPERATURE", "0.7"))
                self._top_p = float(os.getenv("GPT_OSS_TOP_P", "0.9"))

            def with_config(self, config: dict[str, Any] | None = None):  # type: ignore[override]
                try:
                    if config and "callbacks" in config:
                        self._callbacks = list(config.get("callbacks") or [])
                except Exception:
                    pass
                return self

            def invoke(self, prompt_text: str, config: Any | None = None):  # type: ignore[override]
                try:
                    callbacks = list(self._callbacks)
                    try:
                        if config and isinstance(config, dict):
                            callbacks.extend(list(config.get("callbacks") or []))  # type: ignore[arg-type]
                    except Exception:
                        pass

                    sys_msg = {"role": "system", "content": f"Reasoning: {self._reasoning_level}"}
                    messages = [sys_msg, {"role": "user", "content": str(prompt_text)}]
                    chat_text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    run_id = id(self) ^ hash(chat_text)
                    for cb in callbacks:
                        try:
                            cb.on_llm_start({"name": "gpt-oss-local"}, [chat_text], run_id=run_id)
                        except Exception:
                            pass

                    pad_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else self._tok.pad_token_id
                    log_gpu_overview("[local.GPTOSSChatWrapper.invoke]", self._model)
                    generator = pipeline("text-generation", model=self._model, tokenizer=self._tok, batch_size=1, truncation=True)
                    outputs = generator(
                        chat_text,
                        truncation=True,
                        padding=True,
                        do_sample=True,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        max_new_tokens=self._max_new,
                        eos_token_id=self._tok.eos_token_id,
                        pad_token_id=pad_id,
                        num_return_sequences=1,
                    )
                    full_text = outputs[0].get("generated_text", "")
                    answer_text = full_text[len(chat_text):].lstrip() if full_text.startswith(chat_text) else full_text

                    fake_resp = _FakeResponse(answer_text)
                    for cb in callbacks:
                        try:
                            cb.on_llm_end(fake_resp, run_id=run_id)
                        except Exception:
                            pass

                    return AIMessage(content=answer_text)
                except Exception as e:  # pragma: no cover
                    for cb in self._callbacks:
                        try:
                            cb.on_llm_error(e, run_id=id(self))
                        except Exception:
                            pass
                    raise

        raw_llm = GPTOSSChatWrapper(model, tok)
        enc = tok
        logger.info(
            "[local] Initialized GPT-OSS chat wrapper for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, reasoning=%s)",
            requested, npos, tok.model_max_length, reserve_new,
            os.getenv("GPT_OSS_REASONING_LEVEL", "medium").strip().lower(),
        )
        return enc, raw_llm

    log_gpu_overview("[local.HuggingFacePipeline]", model)
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        batch_size=1,
        truncation=True,
        max_new_tokens=reserve_new,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    logger.info(
        "[local] Initialized HuggingFacePipeline for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, truncation=%s)",
        requested, npos, tok.model_max_length, reserve_new, True,
    )
    raw_llm = HuggingFacePipeline(pipeline=hf_pipe)  # type: ignore[call-arg]
    enc = tok
    return enc, raw_llm


__all__ = ["create_local_backend"]


