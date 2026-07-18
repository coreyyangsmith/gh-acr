"""Configuration for quantitative change metrics analysis.

Defines metric names, version identifiers, bucket definitions,
color schemes, and the QuantConfig dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Optional


# ── Version identifiers ──────────────────────────────────────────────────

VERSIONS: Final[list[str]] = [
    "previous",       # Merge-base / common ancestor (original.txt)
    "a",              # Parent A version
    "b",              # Parent B version
    "ground_truth",   # Actual merge resolution
    "agent",          # Single-agent LLM output
    "bypass",         # Multi-agent LLM output
]

VERSION_DISPLAY_NAMES: Final[dict[str, str]] = {
    "previous": "Previous (Ancestor)",
    "a": "Parent A",
    "b": "Parent B",
    "ground_truth": "Ground Truth",
    "agent": "Agent",
    "bypass": "Bypass",
}

# Versions that have stored .diff files
DIFF_FILE_VERSIONS: Final[list[str]] = ["a", "b", "ground_truth"]

# Versions whose diffs must be computed programmatically
COMPUTED_DIFF_VERSIONS: Final[list[str]] = ["agent", "bypass"]


# ── Metric names ─────────────────────────────────────────────────────────

# Size metrics (computed per version)
SIZE_METRICS: Final[list[str]] = [
    "loc",
    "sloc",
    "blank_lines",
    "comment_lines",
]

# Change metrics (delta from ancestor, per non-previous version)
CHANGE_METRICS: Final[list[str]] = [
    "loc_delta",
    "sloc_delta",
]

# Diff metrics (from unified diff, per non-previous version)
DIFF_METRICS: Final[list[str]] = [
    "diff_lines_added",
    "diff_lines_removed",
    "diff_net_change",
    "diff_total_change",
    "diff_hunks",
    "diff_total_lines",
]

# Commit metrics (per sample, not per version)
COMMIT_METRICS: Final[list[str]] = [
    "n_commits_a",
    "n_commits_b",
    "n_commits_total",
]

# Scenario metadata (per sample)
SCENARIO_METRICS: Final[list[str]] = [
    "n_conflict_files",
    "n_total_conflicts",
]

# Performance metrics (from results CSV)
PERFORMANCE_METRICS: Final[list[str]] = [
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
]

METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "loc": "Lines of Code",
    "sloc": "Source Lines of Code",
    "blank_lines": "Blank Lines",
    "comment_lines": "Comment Lines",
    "loc_delta": "LOC Change (vs Ancestor)",
    "sloc_delta": "SLOC Change (vs Ancestor)",
    "diff_lines_added": "Lines Added",
    "diff_lines_removed": "Lines Removed",
    "diff_net_change": "Net Change (Added − Removed)",
    "diff_total_change": "Change Magnitude (Added + Removed)",
    "diff_hunks": "Diff Hunks",
    "diff_total_lines": "Diff Total Lines",
    "n_commits_a": "Commits (Branch A)",
    "n_commits_b": "Commits (Branch B)",
    "n_commits_total": "Total Commits",
    "n_conflict_files": "Files in Conflict",
    "n_total_conflicts": "Total Conflicts",
    "exact_match": "Exact Match",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}


# ── Bucket definitions ───────────────────────────────────────────────────

DEFAULT_COMMIT_COUNT_BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("1 commit", 0, 1),
    ("2-3 commits", 2, 3),
    ("4-10 commits", 4, 10),
    ("11+ commits", 11, float("inf")),
]

DEFAULT_CHANGE_MAGNITUDE_BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("Tiny (1-10)", 0, 10),
    ("Small (11-50)", 11, 50),
    ("Medium (51-200)", 51, 200),
    ("Large (201-500)", 201, 500),
    ("Very Large (500+)", 501, float("inf")),
]

DEFAULT_LOC_DELTA_BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("Shrink (< -50)", -float("inf"), -51),
    ("Minor shrink (-50 to -1)", -50, -1),
    ("No change (0)", 0, 0),
    ("Minor grow (1 to 50)", 1, 50),
    ("Grow (> 50)", 51, float("inf")),
]


# ── Color scheme ─────────────────────────────────────────────────────────

VERSION_COLORS: Final[dict[str, str]] = {
    "previous": "#bdbdbd",      # Light gray
    "a": "#7fcdbb",             # Teal
    "b": "#41b6c4",             # Blue-teal
    "ground_truth": "#2c7fb8",  # Blue
    "agent": "#d95f02",         # Orange
    "bypass": "#1b9e77",        # Green
}

# Method identifiers (matching RQ2/RQ3)
SINGLE_AGENT_METHOD: Final[str] = "agent"
MULTI_AGENT_METHOD: Final[str] = "bypass7"


# ── Configuration dataclass ──────────────────────────────────────────────

@dataclass
class QuantConfig:
    """Configuration for quantitative change metrics analysis.

    Attributes
    ----------
    single_agent_method : str
        Eval method identifier for single-agent
    multi_agent_method : str
        Eval method identifier for multi-agent
    metrics : list[str]
        Performance metrics to correlate with
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals
    ci_level : float
        Confidence interval level (0.95 = 95% CI)
    random_state : int
        Random seed for reproducibility
    min_samples : int
        Minimum samples required for statistical tests
    """

    single_agent_method: str = SINGLE_AGENT_METHOD
    multi_agent_method: str = MULTI_AGENT_METHOD
    metrics: list[str] = field(default_factory=lambda: list(PERFORMANCE_METRICS))

    n_bootstrap: int = 2000
    ci_level: float = 0.95
    random_state: int = 42
    min_samples: int = 10

    # Bucket definitions
    commit_count_buckets: list[tuple[str, float, float]] = field(
        default_factory=lambda: list(DEFAULT_COMMIT_COUNT_BUCKETS)
    )
    change_magnitude_buckets: list[tuple[str, float, float]] = field(
        default_factory=lambda: list(DEFAULT_CHANGE_MAGNITUDE_BUCKETS)
    )
    loc_delta_buckets: list[tuple[str, float, float]] = field(
        default_factory=lambda: list(DEFAULT_LOC_DELTA_BUCKETS)
    )

    # Visualization parameters
    figsize_bar: tuple[float, float] = (12, 8)
    figsize_box: tuple[float, float] = (14, 8)
    figsize_heatmap: tuple[float, float] = (12, 10)
    figsize_scatter: tuple[float, float] = (12, 10)
    dpi: int = 150

    # Color scheme
    positive_color: str = "#2ca02c"
    negative_color: str = "#d62728"
    neutral_color: str = "#7f7f7f"
    heatmap_cmap: str = "RdYlGn"

    def get_metric_label(self, metric: str) -> str:
        """Get display label for a metric."""
        return METRIC_DISPLAY_NAMES.get(metric, metric)

    def get_version_label(self, version: str) -> str:
        """Get display label for a version."""
        return VERSION_DISPLAY_NAMES.get(version, version)

    def get_version_color(self, version: str) -> str:
        """Get color for a version."""
        return VERSION_COLORS.get(version, "#333333")


# Default configuration instance
DEFAULT_CONFIG: Final[QuantConfig] = QuantConfig()
