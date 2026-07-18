"""Shared utilities package for the merge conflict resolution pipeline.

This package provides cross-cutting utilities used throughout the application:
logging, rate limiting, run ledgers, and other common functionality.

Modules
-------
- **logger**: Centralized logging configuration with file and console output
- **rate_limiter**: Thread-safe rate limiting for LLM API calls
- **run_ledger**: Append-only JSONL success/failure ledger for pipeline runs

Usage
-----
>>> from src.utils.logger import setup_logger, logger
>>> from src.utils.rate_limiter import RateLimiter, LimiterRegistry
>>> from src.utils.run_ledger import RunLedger, capture_logs

Logging
-------
The logging module provides a unified logging setup:

>>> logger = setup_logger(__name__)
>>> logger.info("Processing scenario %d", scenario_id)

Log files are written to `./logs/YYYY-MM-DD.log` with daily rotation.
Configure verbosity via the LOG_LEVEL environment variable.

Rate Limiting
-------------
The rate limiter enforces both RPM and TPM limits for LLM APIs:

>>> limiter = LimiterRegistry.get("model_key", rpm=60, tpm=100000, backoff={})
>>> limiter.acquire(expected_tokens=1000)
>>> # ... make API call ...
>>> limiter.adjust(actual_tokens=800, reserved_tokens=1000)
"""

from .logger import setup_logger, logger
from .rate_limiter import RateLimiter, LimiterRegistry
from .run_ledger import RunLedger, capture_logs

__all__ = [
    "setup_logger",
    "logger",
    "RateLimiter",
    "LimiterRegistry",
    "RunLedger",
    "capture_logs",
]
