"""Configuration for RQ1 analyses.

Defines which methods to compare (single-agent vs multi-agent),
metrics to evaluate, and visualization parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal


# Baseline method identifiers (always select one parent)
BASELINE_METHODS: Final[list[str]] = ["base_a", "base_b"]

# Single-agent method identifiers
SINGLE_AGENT_METHODS: Final[list[str]] = ["agent"]

# Multi-agent method identifiers
MULTI_AGENT_METHODS: Final[list[str]] = ["bypass7"]

# All methods in display order
ALL_METHODS_ORDER: Final[list[str]] = ["base_a", "base_b", "agent", "bypass7"]

# Quality metrics to compare
QUALITY_METRICS: Final[list[str]] = [
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
]

# Display names for metrics
METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "exact_match": "Exact Match %",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}

# Display names for methods
METHOD_DISPLAY_NAMES: Final[dict[str, str]] = {
    "base_a": "Base A",
    "base_b": "Base B",
    "agent": "Single-Agent",
    "bypass7": "Multi-Agent",
}

# Color scheme for methods
METHOD_COLORS: Final[dict[str, str]] = {
    "base_a": "#ff7f0e",      # Orange
    "base_b": "#9467bd",      # Purple
    "agent": "#1f77b4",       # Blue
    "bypass7": "#2ca02c",     # Green
}

# Short display names for models (for publication)
MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "openai/gpt-5-nano": "GPT-5-nano",
    "Baselines": "Baselines",
}


def get_short_model_name(model_name: str) -> str:
    """Get a short display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.split("/")[-1])


@dataclass
class RQ1Config:
    """Configuration for RQ1 single-agent vs multi-agent comparison.

    Attributes
    ----------
    single_agent_method : str
        The eval_method identifier for single-agent (default: "agent")
    multi_agent_method : str
        The eval_method identifier for multi-agent (default: "bypass7")
    baseline_methods : list[str]
        Baseline methods to include for context
    include_baselines : bool
        Whether to include baseline methods in plots
    metrics : list[str]
        Quality metrics to compare
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals
    ci_level : float
        Confidence interval level (0.95 = 95% CI)
    random_state : int
        Random seed for reproducibility
    """

    single_agent_method: str = "agent"
    multi_agent_method: str = "bypass7"
    baseline_methods: list[str] = field(default_factory=lambda: list(BASELINE_METHODS))
    include_baselines: bool = True
    metrics: list[str] = field(default_factory=lambda: list(QUALITY_METRICS))
    n_bootstrap: int = 2000
    ci_level: float = 0.95
    random_state: int = 42

    # Visualization parameters (publication-quality)
    figsize_dumbbell: tuple[float, float] = (12, 8)
    figsize_scatter: tuple[float, float] = (10, 10)
    figsize_win_tie_loss: tuple[float, float] = (12, 6)
    figsize_comparison: tuple[float, float] = (14, 8)
    dpi: int = 300  # High DPI for publication

    # Color scheme (legacy - use METHOD_COLORS for new code)
    single_agent_color: str = "#1f77b4"  # Blue
    multi_agent_color: str = "#2ca02c"  # Green
    regression_color: str = "#d62728"  # Red
    tie_color: str = "#7f7f7f"  # Gray

    def get_method_label(self, method: str) -> str:
        """Get a display label for a method."""
        return METHOD_DISPLAY_NAMES.get(method, method)

    def get_method_color(self, method: str) -> str:
        """Get the color for a method."""
        return METHOD_COLORS.get(method, "#333333")

    def get_all_methods(self) -> list[str]:
        """Get all methods to include in analysis, in display order."""
        methods = []
        if self.include_baselines:
            methods.extend(self.baseline_methods)
        methods.append(self.single_agent_method)
        methods.append(self.multi_agent_method)
        return methods


# Default configuration instance
DEFAULT_CONFIG: Final[RQ1Config] = RQ1Config()
