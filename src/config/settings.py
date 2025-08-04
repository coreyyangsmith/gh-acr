"""Application-wide configuration settings.

This module centralizes runtime-tunable constants so that they can be imported
throughout the project without incurring expensive environment look-ups in
multiple places.
"""

from __future__ import annotations

import os

# Minimum number of seconds to wait **between** outbound GitHub API requests.
# This can be overridden at runtime by defining the *GITHUB_REQUEST_INTERVAL*
# environment variable. Defaults to *1.0* seconds.

REQUEST_INTERVAL: float = float(os.getenv("GITHUB_REQUEST_INTERVAL", "1.0"))
