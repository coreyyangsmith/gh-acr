MODEL_COSTS = {
    # Example cost structure; update as needed when OpenAI publishes new pricing.
    "openai/gpt-4.1-nano-2025-04-14": {
        "input_limit": 128_000,
        "output_limit": 16_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0.0001,
        "output_cost_per_1k": 0.0004,
        "tokenizer": "o200k_base_encoding"
    },
    "openai/gpt-4o-mini": {
        "input_limit": 128_000,
        "output_limit": 16_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0.00015,
        "output_cost_per_1k": 0.0006,
        "tokenizer": "o200k_base"
    },
    "openai/gpt-5-nano": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00005,
        "output_cost_per_1k": 0.00040,
        "tokenizer": "o200k_base"
    },
    "openai/gpt-5-mini": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.002,
        "tokenizer": "o200k_base"
    },
    "openai/gpt-5": {
        "input_limit": 400_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 528_000,
        "input_cost_per_1k": 0.00125,
        "output_cost_per_1k": 0.010,
        "tokenizer": "o200k_base"
    },            
    "local:distilbert/distilgpt2": {
        "input_limit": 512,
        "output_limit": 512,
        "sliding_window": True,
        "total_limit": 1024,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "gpt2"
    },
    "local:meta-llama/Llama-3.2-1B": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": True,
        "total_limit": 128_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "llama"
    },
    "groq:llama-3.1-8b-instant": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": True,
        "total_limit": 128_000,
        "input_cost_per_1k": 0.00005,
        "output_cost_per_1k": 0.00008,
        "tokenizer": "llama"
    },    
    "groq:qwen/qwen3-32b": {
        "input_limit": 90_111,
        "output_limit": 40_960,
        "sliding_window": True,
        "total_limit": 131_072,
        "input_cost_per_1k": 0.00029,
        "output_cost_per_1k": 0.00059,
        "tokenizer": "qwen"
    },        
    "local:meta-llama/Llama-3.1-8B": {
        "input_limit": 128_000,
        "output_limit": 2048,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "llama"
    },
    "local:Qwen/Qwen3-8B": {
        "input_limit": 32_000,
        "output_limit": 32_000,
        "sliding_window": False,
        "total_limit": 32_768,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "qwen"
    },
    "local:google/codegemma-7b-it": {
        "input_limit": 8192,
        "output_limit": 8192,
        "sliding_window": False,
        "total_limit": 8192,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "gemma"
    },
    "local:openai/gpt-oss-20b": {
        "input_limit": 128_000,
        "output_limit": 16_000,
        "sliding_window": False,
        "total_limit": 144_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "gpt_oss"
    },
   "local:meta-llama/Llama-3.1-8B": {
        "input_limit": 128_000,
        "output_limit": 128_000,
        "sliding_window": False,
        "total_limit": 128_000,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "tokenizer": "llama"
    }
}
