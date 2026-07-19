"""Application-wide configuration settings.

This module centralizes runtime-tunable constants so that they can be imported
throughout the project without incurring expensive environment lookups in
multiple places. All settings have sensible defaults but can be overridden
via environment variables.

Settings
--------
- **REQUEST_INTERVAL**: Minimum seconds between GitHub API requests
- **BATCH_SIZE**: Number of scenarios to process per batch
- **DATA_PATH**: Default path to the benchmark dataset CSV

Environment Variables
---------------------
- GITHUB_REQUEST_INTERVAL: Override REQUEST_INTERVAL (default: 1.0)
- BATCH_SIZE: Override BATCH_SIZE (default: 2, must be >= 1)
- DATASET_CSV: Override DATA_PATH (default: data/*.csv)

Example Usage
-------------
>>> from src.config.settings import BATCH_SIZE, DATA_PATH
>>> print(f"Processing {BATCH_SIZE} scenarios from {DATA_PATH}")
"""

from __future__ import annotations

import os
from pathlib import Path


# -----------------------------------------------------------------------------
# GitHub API Rate Limiting
# -----------------------------------------------------------------------------

REQUEST_INTERVAL: float = float(os.getenv("GITHUB_REQUEST_INTERVAL", "1.0"))
"""Minimum seconds to wait between outbound GitHub API requests.

This helps avoid hitting GitHub's rate limits when cloning repositories
or fetching commit information. Can be increased for more conservative
rate limiting or decreased for faster processing when limits allow.
"""


# -----------------------------------------------------------------------------
# Batch Processing Configuration
# -----------------------------------------------------------------------------

_batch_size_env = os.getenv("BATCH_SIZE", "2")
try:
    _batch_size_val = int(_batch_size_env)
except ValueError:
    _batch_size_val = 10
if _batch_size_val < 1:
    _batch_size_val = 1

BATCH_SIZE: int = _batch_size_val
"""Number of scenarios to process per batch.

Larger batch sizes can improve throughput but increase memory usage
and the impact of individual failures. The default of 2 is conservative
and suitable for development; production runs may use 10-50.

Cloned repositories under GHACR_CLONE_DIR / ./repos are kept across
batches (and model runs) so later batches can reuse clones. Prepared
context is also cached under data/context_cache (see GHACR_CONTEXT_CACHE_DIR).
"""


# -----------------------------------------------------------------------------
# Context cache
# -----------------------------------------------------------------------------

_default_context_cache = Path.cwd() / "data" / "context_cache"
_env_context_cache = (os.getenv("GHACR_CONTEXT_CACHE_DIR") or "").strip()
CONTEXT_CACHE_DIR: Path = (
    Path(_env_context_cache) if _env_context_cache else _default_context_cache
)
"""On-disk directory for prepared scenario context (contents, diffs, truth).

Override with GHACR_CONTEXT_CACHE_DIR. Survives across model runs so prep
is not repeated when clones already exist.
"""


# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

# Default dataset path - can be overridden via DATASET_CSV environment variable.
# Prefer setting DATASET_CSV explicitly; the default is the full merge-commits extract.
_default_data_path = (
    Path(__file__).resolve().parents[2] / "data" / "git_good_bench_merge_commits_all.csv"
)
_env_data_path = os.getenv("DATASET_CSV")
DATA_PATH: Path = Path(_env_data_path) if _env_data_path else _default_data_path
"""Default path to the benchmark dataset CSV.

This is the input file containing merge conflict scenarios to process.
The CSV should have columns including 'id', 'name' (repo slug), and
'scenario' (JSON with conflict metadata).

Override with the DATASET_CSV environment variable. Common dataset files:
- git_good_bench_merge_commits_all.csv (full extract; default)
- git_good_bench_merge_commits_all_subset_10_seed42.csv (10% sample)
"""


__all__ = [
    "REQUEST_INTERVAL",
    "BATCH_SIZE",
    "CONTEXT_CACHE_DIR",
    "DATA_PATH",
]
