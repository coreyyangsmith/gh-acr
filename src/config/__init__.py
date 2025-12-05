"""Configuration package for the merge conflict resolution pipeline.

This package centralizes all configuration settings, constants, and tunable
parameters used throughout the application. It provides a single source of
truth for runtime behavior without requiring code changes.

Modules
-------
- **settings**: Core application settings (paths, batch sizes, intervals)
- **model_costs**: LLM model pricing and token limits
- **rate_limits**: Per-model rate limiting configuration
- **eval_methods**: Evaluation method definitions and ordering

Environment Variables
---------------------
Most configuration values can be overridden via environment variables:

General:
- DATASET_CSV: Path to the benchmark dataset
- BATCH_SIZE: Number of scenarios per processing batch
- LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

Rate Limiting:
- RL_DEFAULT_RPM: Default requests per minute
- RL_DEFAULT_TPM: Default tokens per minute
- RL_MAX_RETRIES: Maximum retry attempts on errors
- RL_BACKOFF_INITIAL: Initial backoff delay in seconds

Model-Specific:
- OPENAI_API_KEY: OpenAI API authentication
- GROQ_API_KEY: Groq API authentication
- HF_TOKEN: HuggingFace Hub token for gated models

Example Usage
-------------
>>> from src.config.settings import DATA_PATH, BATCH_SIZE
>>> from src.config.model_costs import MODEL_COSTS
>>> from src.config.rate_limits import get_limits_for_model
>>> from src.config.eval_methods import ALL_EVAL_METHODS
"""

from .settings import DATA_PATH, BATCH_SIZE, REQUEST_INTERVAL
from .model_costs import MODEL_COSTS
from .rate_limits import get_limits_for_model, BACKOFF_SETTINGS
from .eval_methods import EvalMethod, ALL_EVAL_METHODS, DEFAULT_METHOD_ORDER

__all__ = [
    # Settings
    "DATA_PATH",
    "BATCH_SIZE",
    "REQUEST_INTERVAL",
    # Model costs
    "MODEL_COSTS",
    # Rate limits
    "get_limits_for_model",
    "BACKOFF_SETTINGS",
    # Eval methods
    "EvalMethod",
    "ALL_EVAL_METHODS",
    "DEFAULT_METHOD_ORDER",
]


