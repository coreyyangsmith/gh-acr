"""Configuration for RQ3 classification analyses.

Defines label mappings, canonical label sets, display names, and visualization parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


# Label mappings: normalize variant labels to canonical forms
LABEL_MAPPINGS: Final[dict[str, str]] = {
    "fewer changes": "favored simplicity",
    "fewer-changes": "favored simplicity",
}

# Canonical labels (in display order)
CANONICAL_LABELS: Final[list[str]] = [
    "favored simplicity",
    "favored complexity",
    "lost information (compression)",
    "misprioritization",
    "feature-oriented",
    "fix-oriented",
    "structural-change-bias",
    "unclear",
    "accurate",
    "vague-commit-message",
    "simple-commit-message",
    "detailed-commit-message",
    "refactor-oriented",
    "modification-bias",
    "test-oriented",
]

# Display names for labels (for plots/tables)
LABEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "favored simplicity": "Favored Simplicity",
    "favored complexity": "Favored Complexity",
    "lost information (compression)": "Lost Information",
    "misprioritization": "Misprioritization",
    "feature-oriented": "Feature-Oriented",
    "fix-oriented": "Fix-Oriented",
    "structural-change-bias": "Structural Change Bias",
    "unclear": "Unclear",
    "accurate": "Accurate",
    "vague-commit-message": "Vague Commit Message",
    "simple-commit-message": "Simple Commit Message",
    "detailed-commit-message": "Detailed Commit Message",
    "refactor-oriented": "Refactor-Oriented",
    "modification-bias": "Modification Bias",
    "test-oriented": "Test-Oriented",
}

# Column name conversion (for CSV headers)
def label_to_column_name(label: str) -> str:
    """Convert a label to a valid column name."""
    return label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()


# Color scheme for labels (grouped by category)
LABEL_COLORS: Final[dict[str, str]] = {
    # Complexity preference
    "favored simplicity": "#2ca02c",      # Green
    "favored complexity": "#d62728",      # Red
    # Information handling
    "lost information (compression)": "#ff7f0e",  # Orange
    "misprioritization": "#9467bd",       # Purple
    # Orientation
    "feature-oriented": "#1f77b4",        # Blue
    "fix-oriented": "#17becf",            # Cyan
    "refactor-oriented": "#bcbd22",       # Yellow-green
    "test-oriented": "#e377c2",           # Pink
    # Biases
    "structural-change-bias": "#8c564b",  # Brown
    "modification-bias": "#7f7f7f",       # Gray
    # Commit message quality
    "vague-commit-message": "#aec7e8",    # Light blue
    "simple-commit-message": "#c5b0d5",   # Light purple
    "detailed-commit-message": "#98df8a", # Light green
    # Other
    "unclear": "#c7c7c7",                 # Light gray
    "accurate": "#2ca02c",                # Green (same as favored simplicity)
}

# Method identifiers (from rq1/rq2)
SINGLE_AGENT_METHOD: Final[str] = "agent"
MULTI_AGENT_METHOD: Final[str] = "bypass7"

# Quality metrics to analyze
QUALITY_METRICS: Final[list[str]] = [
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
]

# Metric display names
METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "exact_match": "Exact Match",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}


def get_label_display_name(label: str) -> str:
    """Get display name for a label."""
    return LABEL_DISPLAY_NAMES.get(label, label.title())


def get_label_color(label: str) -> str:
    """Get color for a label."""
    return LABEL_COLORS.get(label, "#333333")


@dataclass
class RQ3Config:
    """Configuration for RQ3 classification analysis.

    Attributes
    ----------
    label_mappings : dict[str, str]
        Mapping of variant labels to canonical labels
    canonical_labels : list[str]
        List of canonical labels to track
    single_agent_method : str
        Eval method identifier for single-agent
    multi_agent_method : str
        Eval method identifier for multi-agent
    metrics : list[str]
        Quality metrics to analyze
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals
    ci_level : float
        Confidence interval level (0.95 = 95% CI)
    random_state : int
        Random seed for reproducibility
    min_samples : int
        Minimum samples required for statistical tests
    """

    label_mappings: dict[str, str] = field(default_factory=lambda: dict(LABEL_MAPPINGS))
    canonical_labels: list[str] = field(default_factory=lambda: list(CANONICAL_LABELS))
    single_agent_method: str = SINGLE_AGENT_METHOD
    multi_agent_method: str = MULTI_AGENT_METHOD
    metrics: list[str] = field(default_factory=lambda: list(QUALITY_METRICS))
    n_bootstrap: int = 2000
    ci_level: float = 0.95
    random_state: int = 42
    min_samples: int = 10

    # Visualization parameters
    figsize_bar: tuple[float, float] = (12, 8)
    figsize_heatmap: tuple[float, float] = (12, 10)
    figsize_violin: tuple[float, float] = (14, 8)
    dpi: int = 150

    def get_label_display(self, label: str) -> str:
        """Get display name for a label."""
        return get_label_display_name(label)

    def get_label_color(self, label: str) -> str:
        """Get color for a label."""
        return get_label_color(label)

    def apply_mapping(self, label: str) -> str:
        """Apply label mapping to normalize a label."""
        return self.label_mappings.get(label.lower().strip(), label.lower().strip())


# Default configuration instance
DEFAULT_CONFIG: Final[RQ3Config] = RQ3Config()
