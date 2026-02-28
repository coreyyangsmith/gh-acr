"""Configuration for RQ2 analyses.

Defines stratification characteristics, bucket definitions, and visualization parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Optional


# Characteristics available for stratification
STRATIFICATION_CHARACTERISTICS: Final[list[str]] = [
    "difficulty",           # Baseline difficulty (easy/medium/hard)
    "project_size",         # Repository size
    "conflict_size",        # Derived: tokens in conflict
    "file_type",            # Derived: extension from file_name
    "tokens_context",       # Total tokens in prompt/context
]

# Display names for characteristics
CHARACTERISTIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "difficulty": "Baseline Difficulty",
    "project_size": "Project Size",
    "conflict_size": "Conflict Size (tokens)",
    "file_type": "File Type",
    "tokens_context": "Context Size (tokens)",
    "tokens_original": "Original File Size",
    "tokens_diff_a": "Diff A Size",
    "tokens_diff_b": "Diff B Size",
}

# Metrics for improvement calculation
IMPROVEMENT_METRICS: Final[list[str]] = [
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
]

METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "exact_match": "Exact Match",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}

# Default bucket definitions for numeric characteristics
DEFAULT_CONFLICT_SIZE_BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("1-50 tokens", 0, 50),
    ("51-200 tokens", 51, 200),
    ("201-500 tokens", 201, 500),
    ("501-1000 tokens", 501, 1000),
    ("1000+ tokens", 1001, float("inf")),
]

DEFAULT_CONTEXT_SIZE_BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("Small (<1K)", 0, 1000),
    ("Medium (1K-5K)", 1001, 5000),
    ("Large (5K-10K)", 5001, 10000),
    ("Very Large (10K+)", 10001, float("inf")),
]

# File type categories
FILE_TYPE_CATEGORIES: Final[dict[str, list[str]]] = {
    "Python": [".py"],
    "JavaScript/TS": [".js", ".jsx", ".ts", ".tsx"],
    "Java/Kotlin": [".java", ".kt", ".kts"],
    "Go": [".go"],
    "Rust": [".rs"],
    "C/C++": [".c", ".cpp", ".h", ".hpp", ".cc"],
    "Config": [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"],
    "Docs": [".md", ".rst", ".txt"],
    "Other": [],  # Catch-all
}


# Model display names (short versions)
MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "gpt-5-nano": "GPT-5-nano",
}


def get_short_model_name(model_name: str) -> str:
    """Get shortened display name for a model."""
    if model_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_name]
    # Fallback: extract last part after /
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


@dataclass
class RQ2Config:
    """Configuration for RQ2 heterogeneity analysis.

    Attributes
    ----------
    single_agent_method : str
        The eval_method identifier for single-agent (default: "agent")
    multi_agent_method : str
        The eval_method identifier for multi-agent (default: "bypass7")
    primary_metric : str
        Primary metric for improvement calculation
    model_filter : str | None
        If set, filter to only this model_name. If None, use all models.
    per_model_analysis : bool
        If True, generate separate outputs for each model
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals
    ci_level : float
        Confidence interval level (0.95 = 95% CI)
    random_state : int
        Random seed for reproducibility
    min_bucket_size : int
        Minimum samples per bucket to include in analysis
    """

    single_agent_method: str = "agent"
    multi_agent_method: str = "bypass7"
    primary_metric: str = "exact_match"
    metrics: list[str] = field(default_factory=lambda: list(IMPROVEMENT_METRICS))
    
    # Model filtering
    model_filter: Optional[str] = None  # None = all models
    per_model_analysis: bool = True  # Generate per-model outputs
    
    n_bootstrap: int = 2000
    ci_level: float = 0.95
    random_state: int = 42
    min_bucket_size: int = 10

    # Bucket definitions
    conflict_size_buckets: list[tuple[str, float, float]] = field(
        default_factory=lambda: list(DEFAULT_CONFLICT_SIZE_BUCKETS)
    )
    context_size_buckets: list[tuple[str, float, float]] = field(
        default_factory=lambda: list(DEFAULT_CONTEXT_SIZE_BUCKETS)
    )

    # Visualization parameters
    figsize_forest: tuple[float, float] = (10, 8)
    figsize_heatmap: tuple[float, float] = (10, 8)
    figsize_regression: tuple[float, float] = (10, 8)
    figsize_violin: tuple[float, float] = (12, 6)
    dpi: int = 150

    # Color scheme
    positive_color: str = "#2ca02c"  # Green - improvement
    negative_color: str = "#d62728"  # Red - regression
    neutral_color: str = "#7f7f7f"  # Gray - no change
    heatmap_cmap: str = "RdYlGn"  # Red-Yellow-Green diverging colormap

    def get_metric_label(self, metric: str) -> str:
        """Get display label for a metric."""
        return METRIC_DISPLAY_NAMES.get(metric, metric)

    def get_characteristic_label(self, char: str) -> str:
        """Get display label for a characteristic."""
        return CHARACTERISTIC_DISPLAY_NAMES.get(char, char)


# Default configuration instance
DEFAULT_CONFIG: Final[RQ2Config] = RQ2Config()
