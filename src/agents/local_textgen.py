from __future__ import annotations

from typing import Any, Tuple, Optional
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed  # type: ignore

from .handlers.hf_utils import (
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_LOCAL_MODEL_ID,
    HF_LOCAL_ONLY,
    HF_TRUST_REMOTE_CODE,
    HF_REVISION,
    LOCAL_SEED,
    get_hf_token,
    should_use_accelerate,
    pick_torch_dtype,
    weights_exist_locally,
    get_device_map_from_env,
    collect_model_devices,
    log_gpu_overview,
)


logger = logging.getLogger(__name__)


def build_local_text_generator(model_id: str | None = None, *, local_only: bool | None = None):
    if model_id is None:
        model_id = DEFAULT_LOCAL_MODEL_ID
    if local_only is None:
        local_only = HF_LOCAL_ONLY

    set_seed(LOCAL_SEED)
    os.makedirs(DEFAULT_HF_CACHE_DIR, exist_ok=True)
    logger.info(
        "[local] Initializing HF generator: model_id=%s, local_only=%s, cache_dir=%s",
        model_id,
        bool(local_only),
        DEFAULT_HF_CACHE_DIR,
    )
    hf_token = get_hf_token()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            use_fast=True,
            cache_dir=DEFAULT_HF_CACHE_DIR,
            token=hf_token,
            revision=HF_REVISION,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
        )
        use_accel = should_use_accelerate()
        dtype = pick_torch_dtype()
        local_ok = weights_exist_locally(model_id)

        if use_accel:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                device_map=get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                local_files_only=bool(local_ok),
                device_map=get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        logger.info("[local] Model devices: %s", ", ".join(collect_model_devices(model)))
        logger.info("[local] Loaded model+tokenizer from cache (local_files_only=True): %s", model_id)
    except Exception:
        if bool(local_only):
            logger.error("[local] Cache-only load failed and HF_LOCAL_ONLY is set; re-raising for %s", model_id)
            raise
        logger.info("[local] Cache miss; downloading model to cache: %s -> %s", model_id, DEFAULT_HF_CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=DEFAULT_HF_CACHE_DIR,
            token=hf_token,
            revision=HF_REVISION,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
        )
        use_accel = should_use_accelerate()
        dtype = pick_torch_dtype()
        local_ok = weights_exist_locally(model_id)

        if use_accel:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                device_map=get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
                token=hf_token,
                revision=HF_REVISION,
                trust_remote_code=True,
                local_files_only=bool(local_ok),
                device_map=get_device_map_from_env(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        logger.info("[local] Model devices (download): %s", ", ".join(collect_model_devices(model)))
        logger.info("[local] Downloaded model+tokenizer and cached: %s", model_id)

    npos = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024)))
    reserve_new = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
    reserve_new = max(1, min(reserve_new, npos - 32))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.config.pad_token_id = pad_id

    # Align tokenizer truncation strategy and apply a small safety buffer
    try:
        tokenizer.truncation_side = os.getenv("LOCAL_TRUNCATION_SIDE", "left")
    except Exception:
        pass
    try:
        buffer_tokens = int(os.getenv("LOCAL_TOKENIZER_BUFFER_TOKENS", os.getenv("TOKENIZER_BUFFER_TOKENS", "512")))
    except Exception:
        buffer_tokens = 0
    try:
        safe_ctx = max(32, npos - reserve_new - max(0, buffer_tokens))
        tokenizer.model_max_length = safe_ctx
    except Exception:
        pass

    log_gpu_overview("[local.build_local_text_generator]", model)

    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        truncation=True,
        max_new_tokens=reserve_new,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    logger.info(
        "[local] Created text-generation pipeline for %s (npos=%d, model_max_length=%d, max_new_tokens=%d, truncation=%s)",
        model_id,
        npos,
        tokenizer.model_max_length,
        reserve_new,
        True,
    )
    return generator, tokenizer


def generate_local_text(prompt: str, *, max_new_tokens: int = 64, model_id: str | None = None) -> str:
    generator, tokenizer = build_local_text_generator(model_id)
    # Pre-truncate prompt with a small buffer to reduce risk of overflow
    try:
        buffer_tokens = int(os.getenv("LOCAL_TOKENIZER_BUFFER_TOKENS", os.getenv("TOKENIZER_BUFFER_TOKENS", "512")))
    except Exception:
        buffer_tokens = 0
    try:
        if buffer_tokens > 0 and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            max_len = max(32, int(getattr(tokenizer, "model_max_length", 0)) - max_new_tokens)
            if max_len > 0:
                max_len = max(1, max_len - buffer_tokens)
                ids = tokenizer.encode(prompt)
                if len(ids) > max_len:
                    prompt = tokenizer.decode(ids[-max_len:])
    except Exception:
        pass
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    outputs = generator(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
        num_return_sequences=1,
    )
    generated = outputs[0]["generated_text"]
    return generated[len(prompt):].lstrip() if generated.startswith(prompt) else generated


__all__ = ["build_local_text_generator", "generate_local_text"]


