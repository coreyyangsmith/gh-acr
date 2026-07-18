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
    # Default to 2048 tokens for code generation (merge conflicts need longer outputs)
    # 256 is too small for most merge conflict resolution tasks
    reserve_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "2048"))
    reserve_new = max(1, min(reserve_new, npos - 32))

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id

    tok.truncation_side = os.getenv("LOCAL_TRUNCATION_SIDE", "left")
    # Apply a small buffer to tokenizer model_max_length to avoid edge overflows
    try:
        buffer_tokens = int(os.getenv("LOCAL_TOKENIZER_BUFFER_TOKENS", os.getenv("TOKENIZER_BUFFER_TOKENS", "512")))
    except Exception:
        buffer_tokens = 0
    safe_ctx = max(32, npos - reserve_new - max(0, buffer_tokens))
    tok.model_max_length = safe_ctx

    requested_lower = requested.lower()
    is_qwen3 = "qwen3" in requested_lower or "/qwen3-" in requested_lower
    is_gptoss = ("gpt-oss" in requested_lower) or ("/gpt-oss-" in requested_lower) or ("openai/gpt-oss" in requested_lower)
    is_llama = ("llama" in requested_lower) or ("meta-llama" in requested_lower)

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
                    # Pre-truncate chat_text to safe context window with buffer before pipeline
                    try:
                        buffer_tokens = int(os.getenv("LOCAL_TOKENIZER_BUFFER_TOKENS", os.getenv("TOKENIZER_BUFFER_TOKENS", "512")))
                    except Exception:
                        buffer_tokens = 0
                    try:
                        max_ctx = int(getattr(self._tok, "model_max_length", 0))
                    except Exception:
                        max_ctx = 0
                    try:
                        if buffer_tokens > 0 and max_ctx > 0 and hasattr(self._tok, "encode") and hasattr(self._tok, "decode"):
                            # Reserve for generation as configured for wrapper
                            max_input = max(32, max_ctx - self._max_new - buffer_tokens)
                            ids = self._tok.encode(chat_text)
                            if len(ids) > max_input:
                                chat_text = self._tok.decode(ids[-max_input:])
                    except Exception:
                        pass

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

    if is_llama:
        class _FakeGen:
            def __init__(self, text: str):
                self.text = text
        class _FakeResponse:
            def __init__(self, text: str):
                self.generations = [[_FakeGen(text)]]
        class LlamaChatWrapper:
            def __init__(self, model_obj, tokenizer_obj):
                self._model = model_obj
                self._tok = tokenizer_obj
                self._callbacks: list[Any] = []
                self._temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
                self._top_p = float(os.getenv("LLAMA_TOP_P", "0.9"))
                # Align generation cap with earlier reserve to ensure input+output <= npos
                # Use higher default for code generation tasks
                self._max_new = int(os.getenv("LLAMA_MAX_NEW_TOKENS", os.getenv("LOCAL_MAX_NEW_TOKENS", str(reserve_new))))

            def with_config(self, config: dict[str, Any] | None = None):  # type: ignore[override]
                try:
                    if config and "callbacks" in config:
                        self._callbacks = list(config.get("callbacks") or [])
                except Exception:
                    pass
                return self

            def invoke(self, prompt_text: str, config: Any | None = None):  # type: ignore[override]
                import time as _time
                invoke_start = _time.perf_counter()
                
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
                    )

                    run_id = id(self) ^ hash(chat_text)
                    for cb in callbacks:
                        try:
                            cb.on_llm_start({"name": "llama-local"}, [chat_text], run_id=run_id)
                        except Exception:
                            pass

                    pad_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else self._tok.pad_token_id
                    log_gpu_overview("[local.LlamaChatWrapper.invoke]", self._model)
                    
                    # Log prompt size for debugging on Compute Canada
                    prompt_char_len = len(prompt_text)
                    chat_char_len = len(chat_text)
                    logger.info(
                        "[local.llama] Preparing inference: prompt_chars=%d, chat_chars=%d, max_new_tokens=%d, model_max_length=%d",
                        prompt_char_len, chat_char_len, self._max_new, getattr(self._tok, "model_max_length", 0)
                    )

                    # Tokenization-time truncation (Option A)
                    tokenize_start = _time.perf_counter()
                    try:
                        orig_tokens = len(self._tok.encode(chat_text))
                    except Exception:
                        orig_tokens = 0
                    try:
                        model_max = int(getattr(self._tok, "model_max_length", 0))
                    except Exception:
                        model_max = 0
                    trunc_side = getattr(self._tok, "truncation_side", "left")
                    try:
                        inputs = self._tok(
                            chat_text,
                            truncation=True,
                            max_length=model_max if model_max > 0 else None,
                            return_tensors="pt",
                        )
                        post_tokens = int(getattr(getattr(inputs, "input_ids", []), "shape", [0, 0])[-1]) if hasattr(inputs, "input_ids") else 0
                        tokenize_elapsed = _time.perf_counter() - tokenize_start
                        
                        logger.info(
                            "[local.llama] Tokenization complete: orig_tokens=%d, post_tokens=%d, model_max=%d, time=%.3fs",
                            orig_tokens, post_tokens, model_max, tokenize_elapsed
                        )
                        
                        if orig_tokens and model_max and orig_tokens > model_max:
                            logger.warning(
                                "[local.llama] Input truncated at tokenization: orig=%d, max=%d, after=%d, truncation_side=%s, max_new=%d",
                                int(orig_tokens), int(model_max), int(post_tokens), str(trunc_side), int(self._max_new),
                            )
                    except Exception as e:
                        logger.error("[local.llama] Tokenization failed prior to generate: %s", e)
                        raise

                    generate_start = _time.perf_counter()
                    try:
                        # Ensure all runtime inputs are on the same device as the model
                        try:
                            model_device = next(self._model.parameters()).device
                        except Exception:
                            # Fallback if parameters() access fails
                            model_device = getattr(self._model, "device", None)
                        if model_device is not None:
                            logger.debug("[local.llama] Moving inputs to device: %s", model_device)
                            try:
                                # BatchEncoding supports .to(device)
                                inputs = inputs.to(model_device)
                            except Exception:
                                try:
                                    inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in dict(inputs).items()}
                                except Exception:
                                    pass

                        logger.info(
                            "[local.llama] Starting generate: input_tokens=%d, max_new_tokens=%d, temperature=%.2f, top_p=%.2f",
                            post_tokens, self._max_new, self._temperature, self._top_p
                        )
                        
                        outputs = self._model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=self._temperature,
                            top_p=self._top_p,
                            max_new_tokens=self._max_new,
                            eos_token_id=self._tok.eos_token_id,
                            pad_token_id=pad_id,
                        )
                        generate_elapsed = _time.perf_counter() - generate_start
                        
                    except Exception as e:
                        generate_elapsed = _time.perf_counter() - generate_start
                        logger.error(
                            "[local.llama] generate() failed after %.3fs: input_tokens=%s, model_max_length=%s, max_new=%s, error=%s",
                            generate_elapsed, getattr(getattr(inputs, "input_ids", []), "shape", None), 
                            getattr(self._tok, "model_max_length", None), self._max_new, e,
                        )
                        # Log GPU state on error
                        log_gpu_overview("[local.llama] POST-ERROR GPU state", self._model)
                        raise

                    decode_start = _time.perf_counter()
                    try:
                        input_len = int(getattr(getattr(inputs, "input_ids", []), "shape", [0, 0])[-1]) if hasattr(inputs, "input_ids") else 0
                        seq = outputs[0]
                        output_len = len(seq) - input_len
                        answer_ids = seq[input_len:]
                        answer_text = self._tok.decode(answer_ids, skip_special_tokens=False)
                    except Exception:
                        # Fallback: decode full and strip the prompt text
                        full_text = self._tok.decode(outputs[0], skip_special_tokens=False)
                        answer_text = full_text
                        output_len = len(outputs[0]) if hasattr(outputs[0], "__len__") else 0
                    
                    decode_elapsed = _time.perf_counter() - decode_start
                    total_elapsed = _time.perf_counter() - invoke_start
                    
                    logger.info(
                        "[local.llama] Generation complete: output_tokens=%d, output_chars=%d, "
                        "generate_time=%.3fs, decode_time=%.3fs, total_time=%.3fs, tokens/sec=%.1f",
                        output_len, len(answer_text), generate_elapsed, decode_elapsed, total_elapsed,
                        output_len / generate_elapsed if generate_elapsed > 0 else 0
                    )
                    
                    fake_resp = _FakeResponse(answer_text)

                    for cb in callbacks:
                        try:
                            cb.on_llm_end(fake_resp, run_id=run_id)
                        except Exception:
                            pass

                    return AIMessage(content=answer_text)
                except Exception as e:  # pragma: no cover
                    total_elapsed = _time.perf_counter() - invoke_start
                    logger.error("[local.llama] invoke() failed after %.3fs: %s", total_elapsed, e)
                    for cb in self._callbacks:
                        try:
                            cb.on_llm_error(e, run_id=id(self))
                        except Exception:
                            pass
                    raise

        raw_llm = LlamaChatWrapper(model, tok)
        enc = tok
        logger.info(
            "[local] Initialized Llama chat wrapper for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, truncation_side=%s)",
            requested, npos, tok.model_max_length, reserve_new, getattr(tok, "truncation_side", "left"),
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


