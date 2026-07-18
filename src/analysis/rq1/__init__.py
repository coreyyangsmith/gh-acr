"""RQ1: Does multi-agent improve resolution quality?

This package provides visualizations and analyses to answer Research Question 1
by comparing single-agent vs multi-agent approaches across different coding models.

Visualizations
--------------
1. Dumbbell Chart (or Grouped Bar)
   - Shows headline lift from multi-agent per coding model
   - Includes bootstrap confidence intervals

2. Scatter Plot: Single vs Multi
   - One point per conflict scenario
   - Points above diagonal = multi-agent improved

3. Win/Tie/Loss Stacked Bars
   - Shows trade-offs and regression risk per model
   - Answers "How often does multi-agent hurt?"

Usage
-----
CLI:
    python -m src.analysis.rq1.main --input-csv data/results.csv --output-dir results/rq1

Programmatic:
    from src.analysis.rq1 import generate_all_rq1_figures
    generate_all_rq1_figures(input_csv="data/results.csv", output_dir="results/rq1")
"""

from __future__ import annotations

from .config import (
    RQ1Config,
    DEFAULT_CONFIG,
    METHOD_COLORS,
    METHOD_DISPLAY_NAMES,
    MODEL_DISPLAY_NAMES,
    get_short_model_name,
)
from .data import (
    prepare_paired_data,
    compute_model_metrics,
    compute_all_methods_metrics,
    compute_paired_delta_statistics,
    common_agent_bypass_ids,
)
from .dumbbell_chart import render_dumbbell_chart, render_grouped_bar_chart, render_all_methods_comparison
from .scatter_plot import render_scatter_plot
from .win_tie_loss import render_win_tie_loss_chart
from .bypass_distribution import (
    compute_bypass_distribution,
    compute_bypass_distribution_per_instance,
    render_bypass_distribution_bars,
    render_bypass_distribution_bars_per_instance,
    render_bypass_pie_charts,
)
from .main import generate_all_rq1_figures

__all__ = [
    # Configuration
    "RQ1Config",
    "DEFAULT_CONFIG",
    "METHOD_COLORS",
    "METHOD_DISPLAY_NAMES",
    # Data preparation
    "prepare_paired_data",
    "compute_model_metrics",
    "compute_all_methods_metrics",
    "compute_paired_delta_statistics",
    "common_agent_bypass_ids",
    # Visualizations
    "render_dumbbell_chart",
    "render_grouped_bar_chart",
    "render_all_methods_comparison",
    "render_scatter_plot",
    "render_win_tie_loss_chart",
    "compute_bypass_distribution",
    "compute_bypass_distribution_per_instance",
    "render_bypass_distribution_bars",
    "render_bypass_distribution_bars_per_instance",
    "render_bypass_pie_charts",
    # Orchestrator
    "generate_all_rq1_figures",
]
