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

try:  # optional dependency
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False
    def find_dotenv(*args, **kwargs):  # type: ignore
        return ""

# Langfuse is optional; initialize if configured
try:  # pragma: no cover
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore

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

    # Respect explicit disable (Langfuse)
    lf_enabled_env = os.getenv("LANGFUSE_ENABLED")
    lf_enabled = _is_truthy(lf_enabled_env) if lf_enabled_env is not None else True

    if lf_enabled and Langfuse is not None:
        # Determine endpoint and verify basic reachability
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        # If host is reachable and keys are present, initialize Langfuse
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if public_key and secret_key:
            if _collector_reachable(host):
                try:
                    Langfuse(public_key=public_key, secret_key=secret_key, host=host)  # noqa: F841
                    logger.info("Langfuse initialized (host=%s)", host)
                except Exception as e:
                    logger.warning("Langfuse init failed, disabling tracing: %s", e)
                    os.environ["LANGFUSE_ENABLED"] = "0"
            else:
                logger.info("Langfuse host not reachable; disabling tracing for this run.")
                os.environ["LANGFUSE_ENABLED"] = "0"
        else:
            logger.info("Langfuse keys not set; tracing disabled.")
            os.environ["LANGFUSE_ENABLED"] = "0"
    else:
        if Langfuse is None:
            logger.info("Langfuse not installed; tracing disabled.")
        else:
            logger.info("Langfuse explicitly disabled via LANGFUSE_ENABLED.")

    # Optionally trigger early tracer initialization so first LLM call is traced
    try:
        from .agents import llm_base  # noqa: F401  (import-time side effect for LLM callbacks)
        if os.getenv("LANGFUSE_ENABLED", "0").strip() in ("1", "true", "TRUE"):
            logger.info("Startup complete: environment loaded, logging configured, Langfuse ready.")
        else:
            logger.info("Startup complete: environment loaded, logging configured (Langfuse disabled).")
    except Exception:
        logger.info("Startup complete: environment loaded, logging configured.")


# Execute immediately on import
_run_startup_once()

