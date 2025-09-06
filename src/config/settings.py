"""Application-wide configuration settings.

This module centralizes runtime-tunable constants so that they can be imported
throughout the project without incurring expensive environment look-ups in
multiple places.
"""

from __future__ import annotations

import os
from pathlib import Path

# Minimum number of seconds to wait **between** outbound GitHub API requests.
# This can be overridden at runtime by defining the *GITHUB_REQUEST_INTERVAL*
# environment variable. Defaults to *1.0* seconds.

REQUEST_INTERVAL: float = float(os.getenv("GITHUB_REQUEST_INTERVAL", "1.0"))

# Number of scenarios to process per batch when running dataset jobs.
# Can be overridden via environment variable BATCH_SIZE (must be >=1).
_batch_size_env = os.getenv("BATCH_SIZE", "2")
try:
    _batch_size_val = int(_batch_size_env)
except ValueError:
    _batch_size_val = 10
if _batch_size_val < 1:
    _batch_size_val = 1
BATCH_SIZE: int = _batch_size_val

# Default dataset CSV to use for evaluation runs. Can be overridden via
# environment variable DATASET_CSV. This centralizes selection of the input
# benchmark file used across the application.

# DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "single.csv"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "50_easy_instances.csv"