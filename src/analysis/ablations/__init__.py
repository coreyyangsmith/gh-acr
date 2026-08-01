"""Better-Judge leave-one-out ablation analyses.

Compares ``better_judge`` to ``bj_no_summary``, ``bj_no_judge``, ``bj_no_plan``,
and ``bj_no_review`` on a common ID set: component contributions, method ladder,
win/tie/loss, cost–quality Pareto, stratified effects, routing counterfactuals,
cross-model stability, and disagreement case mining.

CLI
---
    python -m src.analysis.ablations.main \\
        --input-csv data/2026_08_01_results.csv \\
        --output-dir results/ablations
"""

from __future__ import annotations

from .config import AblationConfig, DEFAULT_CONFIG, DEFAULT_ABLATIONS
from .main import generate_all_ablation_figures

__all__ = [
    "AblationConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_ABLATIONS",
    "generate_all_ablation_figures",
]
