"""Application startup hook (runs once per process).

This module performs lightweight, one-time initialization before any CLI entry
points execute. Importing `src.startup` has side-effects by design, but the
module import cache ensures it only runs once per process.

Responsibilities:
- Load environment variables from a `.env` file (if present)
- Configure the root logger
- Optionally set sane defaults for Phoenix tracing
- Optionally trigger early initialization of tracing backends
"""

from __future__ import annotations

import os
from typing import Optional
from phoenix.otel import register

try:  # optional dependency
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False
    def find_dotenv(*args, **kwargs):  # type: ignore
        return ""

from .utils.logger import setup_logger


_STARTUP_HAS_RUN: bool = False


def _run_startup_once() -> None:
    global _STARTUP_HAS_RUN
    if _STARTUP_HAS_RUN:
        return
    _STARTUP_HAS_RUN = True

    # Load env vars from .env (best-effort)
    try:
        env_path = find_dotenv(usecwd=True) or find_dotenv(usecwd=False)
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
    except Exception:
        pass

    # Configure root logger early so subsequent imports can use it
    logger = setup_logger()

    # Provide a permissive default for Phoenix tracing unless explicitly disabled
    if os.getenv("PHOENIX_ENABLED") is None:
        os.environ["PHOENIX_ENABLED"] = "1"

    # configure the Phoenix tracer
    # Respect PHOENIX_COLLECTOR_ENDPOINT if set; otherwise, let register pick defaults
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if endpoint:
        tracer_provider = register(endpoint=endpoint, auto_instrument=True)
    else:
        tracer_provider = register(auto_instrument=True)
    logger.info("Phoenix tracer provider initialised")

    # Test Phoenix connection
    try:
        from opentelemetry import trace as trace_api  # type: ignore
        tracer = trace_api.get_tracer(__name__)
        with tracer.start_as_current_span("phoenix_connection_test") as span:
            span.set_attribute("test", "phoenix_startup")
            span.add_event("Phoenix connection test successful")
        logger.info("Phoenix connection test successful")
    except Exception as e:
        logger.warning(f"Phoenix connection test failed: {e}")

    # Optionally trigger early tracer initialization so first LLM call is traced
    try:
        from .agents import llm_base  # noqa: F401  (import-time side effect: tracer init)
        logger.info("Startup complete: environment loaded, logging configured, tracing ready.")
    except Exception:
        logger.info("Startup complete: environment loaded, logging configured.")


# Execute immediately on import
_run_startup_once()

