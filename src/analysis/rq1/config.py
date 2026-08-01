"""Configuration for RQ1 analyses.

Defines which methods to compare (single-agent vs multi-agent),
metrics to evaluate, and visualization parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterable, Optional


# Baseline method identifiers (always select one parent)
BASELINE_METHODS: Final[list[str]] = ["base_a", "base_b"]

# Single-agent method identifiers
SINGLE_AGENT_METHODS: Final[list[str]] = ["agent"]

# Multi-agent method identifiers (preferred resolution order when auto-detecting)
MULTI_AGENT_METHODS: Final[list[str]] = [
    "bypass7",
    "better_judge",
    "bj_no_judge",
    "bj_no_plan",
    "bj_no_review",
    "bj_no_summary",
]

# All methods in display order
ALL_METHODS_ORDER: Final[list[str]] = [
    "base_a",
    "base_b",
    "agent",
    "bypass7",
    "better_judge",
    "bj_no_judge",
    "bj_no_plan",
    "bj_no_review",
    "bj_no_summary",
]

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
    "better_judge": "Better-Judge",
    "bj_no_judge": "BJ (no judge)",
    "bj_no_plan": "BJ (no plan)",
    "bj_no_review": "BJ (no review)",
    "bj_no_summary": "BJ (no summary)",
}

# Color scheme for methods
METHOD_COLORS: Final[dict[str, str]] = {
    "base_a": "#ff7f0e",      # Orange
    "base_b": "#9467bd",      # Purple
    "agent": "#1f77b4",       # Blue
    "bypass7": "#2ca02c",     # Green
    "better_judge": "#2ca02c",  # Green (primary multi-agent)
    "bj_no_judge": "#98df8a",
    "bj_no_plan": "#c5e1a5",
    "bj_no_review": "#a1d99b",
    "bj_no_summary": "#74c476",
}

# Short display names for models (for publication)
MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "openai/gpt-5-nano": "GPT-5-nano",
    "Baselines": "Baselines",
}


def get_short_model_name(model_name: str) -> str:
    """Get a short display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.split("/")[-1])


def infer_multi_agent_method(
    present_methods: Iterable[str],
    *,
    preferred: Optional[str] = None,
) -> Optional[str]:
    """Pick a multi-agent eval_method present in the data.

    Preference order: ``preferred`` (if present), then ``MULTI_AGENT_METHODS``.
    """
    present = {str(m) for m in present_methods if m is not None and str(m).strip()}
    if preferred and preferred in present:
        return preferred
    for method in MULTI_AGENT_METHODS:
        if method in present:
            return method
    return None


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

    def resolve_methods_from_data(
        self,
        present_methods: Iterable[str],
        *,
        auto_multi: bool = True,
    ) -> "RQ1Config":
        """Return a copy with multi-agent method resolved against present eval_methods.

        If ``auto_multi`` and the configured multi-agent method is absent, pick the
        first available method from ``MULTI_AGENT_METHODS`` (e.g. ``better_judge``).
        """
        present = {str(m) for m in present_methods if m is not None and str(m).strip()}
        multi = self.multi_agent_method
        if auto_multi and multi not in present:
            inferred = infer_multi_agent_method(present, preferred=None)
            if inferred is not None:
                multi = inferred

        return RQ1Config(
            single_agent_method=self.single_agent_method,
            multi_agent_method=multi,
            baseline_methods=list(self.baseline_methods),
            include_baselines=self.include_baselines,
            metrics=list(self.metrics),
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
            random_state=self.random_state,
            figsize_dumbbell=self.figsize_dumbbell,
            figsize_scatter=self.figsize_scatter,
            figsize_win_tie_loss=self.figsize_win_tie_loss,
            figsize_comparison=self.figsize_comparison,
            dpi=self.dpi,
            single_agent_color=self.single_agent_color,
            multi_agent_color=self.multi_agent_color,
            regression_color=self.regression_color,
            tie_color=self.tie_color,
        )


# Default configuration instance
DEFAULT_CONFIG: Final[RQ1Config] = RQ1Config()
