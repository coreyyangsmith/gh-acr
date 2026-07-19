"""Model pricing and token limits configuration.

This module defines the cost structure and token limits for all supported
LLM models. This information is used for:
1. **Cost estimation**: Calculating the cost of API calls
2. **Token budgeting**: Ensuring prompts fit within model limits
3. **Truncation**: Deciding how much context to preserve when clipping

Model Configuration Keys
------------------------
Each model entry contains:
- **input_limit**: Maximum input tokens accepted
- **output_limit**: Maximum output tokens generated
- **sliding_window**: Whether the model uses sliding window attention
- **total_limit**: Combined input+output limit (for sliding window models)
- **input_cost_per_1k**: Cost per 1000 input tokens (USD)
- **output_cost_per_1k**: Cost per 1000 output tokens (USD)
- **tokenizer**: Name of the tokenizer encoding to use

Model Naming Conventions
------------------------
- "openai/<model>": OpenAI API models (e.g., "openai/gpt-4o-mini")
- "openrouter/<provider>/<model>": OpenRouter models (e.g., "openrouter/anthropic/claude-sonnet-4.5")
- "groq:<model>": Groq API models (e.g., "groq:llama-3.1-8b-instant")
- "local:<path>": Local HuggingFace models (e.g., "local:meta-llama/Llama-3.1-8B")

Example Usage
-------------
>>> from src.config.model_costs import MODEL_COSTS
>>> model_cfg = MODEL_COSTS.get("openai/gpt-4o-mini", {})
>>> max_input = model_cfg.get("input_limit", 4096)
>>> cost_per_1k = model_cfg.get("input_cost_per_1k", 0.0)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_COSTS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # OpenAI Models
    # -------------------------------------------------------------------------
    "openai/gpt-4.1-nano-2025-04-14": {
        "input_limit": 128_000,
        "output_limit": 16_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0.0001,
        "output_cost_per_1k": 0.0004,
        "tokenizer": "o200k_base_encoding",
    },
    "openai/gpt-4o-mini": {
        "input_limit": 128_000,
        "output_limit": 16_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0.00015,
        "output_cost_per_1k": 0.0006,
        "tokenizer": "o200k_base",
    },
    "openai/gpt-5-nano": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00005,
        "output_cost_per_1k": 0.00040,
        "tokenizer": "o200k_base",
    },
    "openai/gpt-5-mini": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.002,
        "tokenizer": "o200k_base",
    },
    "openai/gpt-5": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00125,
        "output_cost_per_1k": 0.010,
        "tokenizer": "o200k_base",
    },

    # -------------------------------------------------------------------------
    # OpenRouter Models (OpenAI-compatible API; costs vary by upstream provider)
    # Add entries for models you use frequently. Lookup also falls back via
    # get_model_config() if only the openrouter/<id> key is present.
    # -------------------------------------------------------------------------
    "openrouter/openai/gpt-5-nano": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00005,   # $0.05 per 1M input tokens
        "output_cost_per_1k": 0.00040,  # $0.40 per 1M output tokens
        "tokenizer": "o200k_base",
    },
    "openrouter/qwen/qwen3-32b": {
        "input_limit": 131_072,
        "output_limit": 16_384,
        "sliding_window": False,
        "total_limit": 131_072,
        "input_cost_per_1k": 0.00029,   # $0.29 per 1M, see provider list
        "output_cost_per_1k": 0.00058,  # $0.58 per 1M, see provider list
        "tokenizer": "qwen",
    },
    # OpenRouter list price $0.02/$0.03 per 1M tokens; 131K context
    "openrouter/meta-llama/llama-3.1-8b-instruct": {
        "input_limit": 131_072,
        "output_limit": 16_384,
        "sliding_window": False,
        "total_limit": 131_072,
        "input_cost_per_1k": 0.00005,
        "output_cost_per_1k": 0.00008,
        "tokenizer": "llama",
    },

    # -------------------------------------------------------------------------
    # Groq Models (API-based, fast inference)
    # -------------------------------------------------------------------------
    "groq:llama-3.1-8b-instant": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": True,
        "total_limit": 128_000,
        "input_cost_per_1k": 0.00005,
        "output_cost_per_1k": 0.00008,
        "tokenizer": "llama",
    },
    "groq:qwen/qwen3-32b": {
        "input_limit": 90_111,
        "output_limit": 40_960,
        "sliding_window": True,
        "total_limit": 131_072,
        "input_cost_per_1k": 0.00029,
        "output_cost_per_1k": 0.00059,
        "tokenizer": "qwen",
    },

    # -------------------------------------------------------------------------
    # Local Models (self-hosted, no API cost)
    # -------------------------------------------------------------------------
    "local:distilbert/distilgpt2": {
        "input_limit": 512,
        "output_limit": 512,
        "sliding_window": True,
        "total_limit": 1024,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "gpt2",
    },
    "local:meta-llama/Llama-3.2-1B": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": True,
        "total_limit": 128_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "llama",
    },
    "local:meta-llama/Llama-3.1-8B": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "llama",
    },
    "local:meta-llama/Llama-3.1-8B-Instruct": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "llama",
    },
    "local:Qwen/Qwen3-8B": {
        "input_limit": 32_000,
        "output_limit": 32_000,
        "sliding_window": False,
        "total_limit": 32_768,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "qwen",
    },
    "local:google/codegemma-7b-it": {
        "input_limit": 8192,
        "output_limit": 8192,
        "sliding_window": False,
        "total_limit": 8192,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "gemma",
    },
    "local:openai/gpt-oss-20b": {
        "input_limit": 128_000,
        "output_limit": 16_000,
        "sliding_window": False,
        "total_limit": 144_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "gpt_oss",
    },
}
"""Mapping of model names to their configuration dictionaries.

Each configuration specifies token limits, costs, and tokenizer details.
Local models have zero cost but may have lower context limits depending
on available GPU memory.
"""


def price_key(model_name: str) -> str:
    """Normalize a model id to the key form used in ``MODEL_COSTS``.

    Handles ``openai/``, ``openrouter/``, ``local:``, and ``groq:`` prefixes.
    Bare OpenAI-style names (e.g. ``gpt-4o-mini``) become ``openai/<name>``.
    """
    try:
        if model_name.startswith(("openai/", "openrouter/", "local:", "groq:")):
            return model_name
        return f"openai/{model_name}"
    except Exception:
        return model_name


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a model, with fallback handling.

    Parameters
    ----------
    model_name
        The model identifier (e.g., "openai/gpt-4o-mini")

    Returns
    -------
    Dict[str, Any]
        Model configuration dict, or empty dict if not found.
    """
    cfg = MODEL_COSTS.get(model_name, {})
    if cfg:
        return dict(cfg)

    key = price_key(model_name)
    if key != model_name:
        cfg = MODEL_COSTS.get(key, {})
        if cfg:
            return dict(cfg)

    # Legacy alias: some callers historically used groq/ instead of groq:
    if model_name.startswith("groq/"):
        alias = "groq:" + model_name.split("/", 1)[1]
        return dict(MODEL_COSTS.get(alias, {}))

    return {}


def estimate_usd_cost(
    model_name: str,
    input_tokens: int | float,
    output_tokens: int | float,
) -> Tuple[float, float, float]:
    """Estimate USD cost from token counts using ``MODEL_COSTS`` rates.

    Returns
    -------
    Tuple[float, float, float]
        ``(cost_in, cost_out, total)`` in USD. Unknown models yield zeros.
    """
    cfg = get_model_config(model_name)
    if not cfg:
        cfg = MODEL_COSTS.get(price_key(model_name), {})
    input_rate = float(cfg.get("input_cost_per_1k", 0.0) or 0.0)
    output_rate = float(cfg.get("output_cost_per_1k", 0.0) or 0.0)
    cost_in = (float(input_tokens) / 1000.0) * input_rate
    cost_out = (float(output_tokens) / 1000.0) * output_rate
    return cost_in, cost_out, cost_in + cost_out


__all__ = [
    "MODEL_COSTS",
    "estimate_usd_cost",
    "get_model_config",
    "price_key",
]
