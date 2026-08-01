"""Configuration for Better-Judge leave-one-out ablation analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


ANCHOR_METHOD: Final[str] = "better_judge"
AGENT_METHOD: Final[str] = "agent"
BASELINE_METHODS: Final[list[str]] = ["base_a", "base_b"]

DEFAULT_ABLATIONS: Final[list[str]] = [
    "bj_no_summary",
    "bj_no_judge",
    "bj_no_plan",
    "bj_no_review",
]

# Component removed by each ablation (for plot/table labels)
COMPONENT_REMOVED: Final[dict[str, str]] = {
    "bj_no_summary": "Summarizer",
    "bj_no_judge": "Analyzer (routing)",
    "bj_no_plan": "Planner+Reviewer",
    "bj_no_review": "Reviewer",
}

ABLATION_NOTES: Final[dict[str, str]] = {
    "bj_no_plan": "Confounded: removes both planner and reviewer",
}

QUALITY_METRICS: Final[list[str]] = [
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
]

COST_COLUMNS: Final[list[str]] = [
    "total_cost",
    "tokens_total",
    "processing_time_s",
]

METHOD_DISPLAY_NAMES: Final[dict[str, str]] = {
    "base_a": "Base A",
    "base_b": "Base B",
    "agent": "Single-Agent",
    "better_judge": "Better-Judge",
    "bj_no_summary": "BJ (no summary)",
    "bj_no_judge": "BJ (no judge)",
    "bj_no_plan": "BJ (no plan+review)",
    "bj_no_review": "BJ (no review)",
}

METHOD_COLORS: Final[dict[str, str]] = {
    "base_a": "#ff7f0e",
    "base_b": "#9467bd",
    "agent": "#1f77b4",
    "better_judge": "#2ca02c",
    "bj_no_summary": "#74c476",
    "bj_no_judge": "#98df8a",
    "bj_no_plan": "#c5e1a5",
    "bj_no_review": "#a1d99b",
}

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "openai/gpt-5-nano": "GPT-5-nano",
}

METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "exact_match": "Exact Match",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}

# Ladder display order
LADDER_ORDER: Final[list[str]] = [
    "base_a",
    "base_b",
    "agent",
    "bj_no_summary",
    "bj_no_judge",
    "bj_no_plan",
    "bj_no_review",
    "better_judge",
]

STRATA: Final[list[str]] = ["difficulty", "project_size", "conflict_size"]

CONFLICT_SIZE_BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("1-50", 0, 50),
    ("51-200", 51, 200),
    ("201-500", 201, 500),
    ("501-1000", 501, 1000),
    ("1000+", 1001, float("inf")),
]


def get_short_model_name(model_name: str) -> str:
    """Short display name for a model."""
    if model_name is None or (isinstance(model_name, float) and model_name != model_name):
        return "unknown"
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name).split("/")[-1])


def get_method_label(method: str) -> str:
    """Display label for an eval_method."""
    return METHOD_DISPLAY_NAMES.get(method, method)


def get_method_color(method: str) -> str:
    """Color for an eval_method."""
    return METHOD_COLORS.get(method, "#333333")


def get_component_label(ablation: str) -> str:
    """Human-readable component removed by an ablation."""
    return COMPONENT_REMOVED.get(ablation, ablation)


@dataclass
class AblationConfig:
    """Runtime config for ablation analyses."""

    anchor_method: str = ANCHOR_METHOD
    agent_method: str = AGENT_METHOD
    ablations: list[str] = field(default_factory=lambda: list(DEFAULT_ABLATIONS))
    baseline_methods: list[str] = field(default_factory=lambda: list(BASELINE_METHODS))
    metrics: list[str] = field(default_factory=lambda: list(QUALITY_METRICS))
    n_bootstrap: int = 2000
    ci_level: float = 0.95
    random_state: int = 42
    dpi: int = 300
    exclude_soft_degraded: bool = False

    def ladder_methods(self) -> list[str]:
        """Methods in ladder order that are configured for this run."""
        wanted = set(self.baseline_methods) | {self.agent_method, self.anchor_method} | set(self.ablations)
        return [m for m in LADDER_ORDER if m in wanted]

    def all_multi_methods(self) -> list[str]:
        """Anchor + ablations."""
        return [self.anchor_method, *self.ablations]


DEFAULT_CONFIG: Final[AblationConfig] = AblationConfig()
