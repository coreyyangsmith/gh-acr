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
import socket
from urllib.parse import urlparse

# Phoenix (Arize) is optional; guard imports
try:  # pragma: no cover
    from phoenix.otel import register as phoenix_register  # type: ignore
except Exception:  # pragma: no cover
    phoenix_register = None  # type: ignore

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

    def _is_truthy(val: Optional[str]) -> bool:
        if val is None:
            return False
        return val.strip().lower() in {"1", "true", "yes", "on"}

    def _collector_reachable(url: str) -> bool:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port
            if port is None:
                port = 443 if parsed.scheme == "https" else 80
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except Exception:
            return False

    # Respect explicit disable
    phoenix_enabled_env = os.getenv("PHOENIX_ENABLED")
    phoenix_enabled = _is_truthy(phoenix_enabled_env) if phoenix_enabled_env is not None else True

    tracer_provider = None
    if phoenix_enabled and phoenix_register is not None:
        # Determine endpoint and verify reachability before registering
        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        if _collector_reachable(endpoint):
            try:
                tracer_provider = phoenix_register(endpoint=endpoint, auto_instrument=True)
                logger.info("Phoenix tracer provider initialised")
                # Optional quick span to warm up
                try:
                    from opentelemetry import trace as trace_api  # type: ignore
                    tracer = trace_api.get_tracer(__name__)
                    with tracer.start_as_current_span("phoenix_connection_test") as span:
                        span.set_attribute("test", "phoenix_startup")
                        span.add_event("Phoenix connection test successful")
                    logger.info("Phoenix connection test successful")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Phoenix register failed, disabling tracing: {e}")
                os.environ["PHOENIX_ENABLED"] = "0"
        else:
            logger.info("Phoenix collector not reachable; disabling tracing for this run.")
            os.environ["PHOENIX_ENABLED"] = "0"
    else:
        if phoenix_register is None:
            logger.info("Phoenix not installed; tracing disabled.")
        else:
            logger.info("Phoenix explicitly disabled via PHOENIX_ENABLED.")

    # Optionally trigger early tracer initialization so first LLM call is traced
    try:
        from .agents import llm_base  # noqa: F401  (import-time side effect: tracer init)
        if os.getenv("PHOENIX_ENABLED", "0").strip() in ("1", "true", "TRUE"):
            logger.info("Startup complete: environment loaded, logging configured, tracing ready.")
        else:
            logger.info("Startup complete: environment loaded, logging configured (tracing disabled).")
    except Exception:
        logger.info("Startup complete: environment loaded, logging configured.")


# Execute immediately on import
_run_startup_once()

