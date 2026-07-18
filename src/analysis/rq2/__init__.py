"""RQ2: Under what conflict + patch characteristics does multi-agent help/hurt?

This package provides visualizations and analyses to answer Research Question 2
by examining effect heterogeneity across different conflict characteristics.

Visualizations
--------------
1. Stratified Lift Plot (Forest Plot)
   - Shows improvement (Δ = Multi - Single) per stratum
   - Point-range plot with confidence intervals
   - Stratified by: difficulty, conflict size, file type, project size, etc.

2. Heatmap: Difficulty × Conflict Size
   - Shows interaction effects
   - Cell value = Δ improvement
   - Reveals where multi-agent helps/hurts most

3. Logistic Regression Effect Plot
   - Odds ratio / coefficient plot with CIs
   - Quantifies which features predict multi-agent wins

4. Improvement Distribution by Bucket
   - Violin/box plots showing spread of Δ scores
   - Reveals if improvement is consistent or bimodal

Usage
-----
CLI:
    python -m src.analysis.rq2.main --input-csv data/results.csv --output-dir results/rq2

Programmatic:
    from src.analysis.rq2 import generate_all_rq2_figures
    generate_all_rq2_figures(input_csv="data/results.csv", output_dir="results/rq2")
"""

from __future__ import annotations

from .config import RQ2Config, DEFAULT_CONFIG, get_short_model_name, MODEL_DISPLAY_NAMES
from .data import (
    prepare_improvement_data,
    create_buckets,
    compute_stratified_metrics,
    prepare_regression_data,
    aggregate_to_instance_level,
    GranularityType,
)
from .stratified_lift import render_forest_plot, render_stratified_lift_by_characteristic
from .heatmap import render_difficulty_size_heatmap, render_interaction_heatmap
from .regression import render_odds_ratio_plot, fit_logistic_model
from .distribution import render_violin_by_bucket, render_improvement_distributions
from .method_comparison import (
    render_method_comparison_by_difficulty,
    render_method_comparison_by_project_size,
    render_method_comparison_heatmap,
    export_method_comparison_table,
)
from .main import generate_all_rq2_figures

__all__ = [
    # Configuration
    "RQ2Config",
    "DEFAULT_CONFIG",
    "get_short_model_name",
    "MODEL_DISPLAY_NAMES",
    # Data preparation
    "prepare_improvement_data",
    "create_buckets",
    "compute_stratified_metrics",
    "prepare_regression_data",
    "aggregate_to_instance_level",
    "GranularityType",
    # Visualizations
    "render_forest_plot",
    "render_stratified_lift_by_characteristic",
    "render_difficulty_size_heatmap",
    "render_interaction_heatmap",
    "render_odds_ratio_plot",
    "fit_logistic_model",
    "render_violin_by_bucket",
    "render_improvement_distributions",
    # Method comparison
    "render_method_comparison_by_difficulty",
    "render_method_comparison_by_project_size",
    "render_method_comparison_heatmap",
    "export_method_comparison_table",
    # Orchestrator
    "generate_all_rq2_figures",
]
