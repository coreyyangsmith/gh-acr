"""Application startup hook (runs once per process).

This module performs lightweight, one-time initialization before any CLI entry
points execute. Importing `src.startup` has side effects by design, but the
module import cache ensures it only runs once per process.

Responsibilities
----------------
1. Load environment variables from a `.env` file (if present)
2. Configure the root logger for consistent output
3. Set up LLM callbacks for observability
4. Log environment diagnostics for Compute Canada debugging

Usage
-----
This module is typically imported at the top of entry points::

    import src.startup  # noqa: F401  # Ensure startup runs first

The import triggers `_run_startup_once()` which performs all initialization.
Subsequent imports are no-ops due to the `_STARTUP_HAS_RUN` guard.

Notes
-----
- The module uses best-effort error handling to avoid crashes
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # optional dependency
    from dotenv import find_dotenv, load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False
    def find_dotenv(*args, **kwargs):  # type: ignore
        return ""

from .utils.logger import setup_logger


_STARTUP_HAS_RUN: bool = False


def _log_environment_diagnostics(logger) -> None:
    """Log comprehensive environment diagnostics for debugging on Compute Canada."""
    logger.info("=" * 70)
    logger.info("ENVIRONMENT DIAGNOSTICS")
    logger.info("=" * 70)
    
    # Python info
    logger.info("Python: %s", sys.version)
    logger.info("Python executable: %s", sys.executable)
    logger.info("Platform: %s", sys.platform)
    
    # Key environment variables for truncation/LLM behavior
    truncation_vars = [
        "LOCAL_MAX_NEW_TOKENS",
        "LOCAL_TRUNCATION_SIDE",
        "TRUNCATION_SIDE",
        "LOCAL_TOKENIZER_BUFFER_TOKENS",
        "TOKENIZER_BUFFER_TOKENS",
        "LLAMA_MAX_NEW_TOKENS",
        "LLAMA_TEMPERATURE",
        "LLAMA_TOP_P",
    ]
    logger.info("--- Truncation/LLM Configuration ---")
    for var in truncation_vars:
        val = os.getenv(var, "<not set>")
        logger.info("  %s = %s", var, val)

    # LangFuse (presence only; never log secret values)
    logger.info("--- LangFuse Configuration ---")
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_TRACING_ENABLED"):
        val = os.getenv(var, "<not set>")
        if var.endswith("_KEY") and val != "<not set>":
            val = val[:4] + "****" if len(val) > 4 else "****"
        logger.info("  %s = %s", var, val)
    
    # HuggingFace configuration
    hf_vars = [
        "HF_HOME",
        "HF_CACHE_DIR",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_DEVICE_MAP",
        "HF_TORCH_DTYPE",
        "HF_LOCAL_ONLY",
        "HF_TRUST_REMOTE_CODE",
        "TRANSFORMERS_CACHE",
        "TOKENIZERS_PARALLELISM",
    ]
    logger.info("--- HuggingFace Configuration ---")
    for var in hf_vars:
        val = os.getenv(var, "<not set>")
        # Mask tokens
        if "TOKEN" in var and val != "<not set>":
            val = val[:4] + "****" if len(val) > 4 else "****"
        logger.info("  %s = %s", var, val)
    
    # CUDA/GPU configuration
    cuda_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_HOME",
        "TORCH_CUDA_ARCH_LIST",
    ]
    logger.info("--- CUDA/GPU Configuration ---")
    for var in cuda_vars:
        val = os.getenv(var, "<not set>")
        logger.info("  %s = %s", var, val)
    
    # SLURM environment (for Compute Canada)
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_GPUS_PER_NODE",
        "SLURM_TMPDIR",
        "SLURM_SUBMIT_DIR",
    ]
    logger.info("--- SLURM Environment ---")
    for var in slurm_vars:
        val = os.getenv(var, "<not set>")
        logger.info("  %s = %s", var, val)
    
    # Check PyTorch and CUDA availability
    logger.info("--- PyTorch/CUDA Status ---")
    try:
        import torch
        logger.info("  PyTorch version: %s", torch.__version__)
        logger.info("  CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("  CUDA version: %s", torch.version.cuda)
            logger.info("  GPU count: %d", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info("  GPU %d: %s (%.1f GB)", i, props.name, props.total_memory / 1024**3)
    except ImportError:
        logger.info("  PyTorch not installed")
    except Exception as e:
        logger.warning("  PyTorch check failed: %s", e)
    
    # Memory info
    logger.info("--- System Memory ---")
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info("  Total: %.1f GB", mem.total / 1024**3)
        logger.info("  Available: %.1f GB", mem.available / 1024**3)
        logger.info("  Used: %.1f%% ", mem.percent)
    except ImportError:
        logger.info("  psutil not installed")
    except Exception as e:
        logger.warning("  Memory check failed: %s", e)
    
    # Disk space for SLURM_TMPDIR
    tmpdir = os.getenv("SLURM_TMPDIR", os.getenv("TMPDIR", "/tmp"))
    logger.info("--- Disk Space (TMPDIR: %s) ---", tmpdir)
    try:
        import shutil
        total, used, free = shutil.disk_usage(tmpdir)
        logger.info("  Total: %.1f GB", total / 1024**3)
        logger.info("  Free: %.1f GB", free / 1024**3)
    except Exception as e:
        logger.warning("  Disk check failed: %s", e)
    
    logger.info("=" * 70)


def _run_startup_once() -> None:
    """Execute one-time startup initialization.

    This function is called automatically on module import and is guarded
    to run only once per process.

    Steps:
    1. Load .env file (best-effort)
    2. Configure root logger
    3. Import LLM callbacks to trigger early initialization
    """
    global _STARTUP_HAS_RUN
    if _STARTUP_HAS_RUN:
        return
    _STARTUP_HAS_RUN = True

    # Load environment variables from .env (best-effort).
    # Prefer `src/.env` (next to `src/.env.example`); fall back to find_dotenv.
    try:
        src_env = Path(__file__).resolve().parent / ".env"
        if src_env.is_file():
            load_dotenv(src_env)
        else:
            env_path = find_dotenv(usecwd=True) or find_dotenv(usecwd=False)
            if env_path:
                load_dotenv(env_path)
            else:
                load_dotenv()
    except Exception:
        pass

    # Configure root logger early so subsequent imports can use it
    logger = setup_logger()
    
    # Log environment diagnostics for Compute Canada debugging
    if os.getenv("GHACR_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"):
        _log_environment_diagnostics(logger)

    # Trigger early LLM callback initialization
    try:
        from .agents import llm_base  # noqa: F401
    except Exception:
        pass
    logger.info("Startup complete: environment loaded, logging configured.")


# Execute immediately on import
_run_startup_once()
