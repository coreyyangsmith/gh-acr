"""Figure C: Bypass Decision Outcomes and Selection Bias.

Stacked bar of A/B selection rates per model (MIX excluded).

Usage::

    python -m src.analysis.final_paper_figs.figure_c_decision_outcomes
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.final_paper_figs.shared import (
    MODEL_ORDER,
    MODEL_SHORT,
    OUTPUT_DIR,
    apply_style,
    common_ids,
    load_results,
    logger,
    save_fig,
)

def generate_figure_c(results_csv: Path | None = None,
                       output_dir: Path | None = None) -> None:
    """Generate Figure C: Decision outcomes (stacked bar of A/B/MIX selection)."""
    apply_style()
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Figure C: Decision outcomes")

    df = load_results(results_csv)
    common = common_ids(df)

    bypass = df[(df["eval_method"] == "bypass7") & (df["id"].isin(common))].copy()
    bypass["model"] = bypass["model_name"].map(MODEL_SHORT)

    decision_colors = {"A": "#a6cee3", "B": "#b2df8a"}
    decision_order = ["A", "B"]

    # Custom model order: Llama, Qwen, GPT
    fig_model_order = ["LLaMA-3.1-8B", "Qwen3-32B", "GPT-5-nano"]
    # Display names for x-axis labels
    fig_model_labels = {
        "LLaMA-3.1-8B": "Llama-3.1-8B",
        "Qwen3-32B": "Qwen3-32B",
        "GPT-5-nano": "GPT-5-Nano",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.42))

    model_pcts: dict[str, dict[str, float]] = {}
    for model in fig_model_order:
        sub = bypass[bypass["model"] == model]
        # Exclude MIX rows
        sub = sub[sub["bypass_method"].isin(["A", "B"])]
        total = len(sub)
        if total == 0:
            continue
        counts = sub["bypass_method"].value_counts()
        model_pcts[model] = {
            d: counts.get(d, 0) / total * 100 for d in decision_order
        }

    x = np.arange(len(fig_model_order))
    bottoms = np.zeros(len(fig_model_order))
    for decision in decision_order:
        vals = [model_pcts.get(m, {}).get(decision, 0) for m in fig_model_order]
        ax.bar(
            x,
            vals,
            0.55,
            bottom=bottoms,
            label=decision,
            color=decision_colors[decision],
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate percentages (larger text)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(
                    x[i],
                    b + v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([fig_model_labels[m] for m in fig_model_order], fontsize=9)
    ax.set_ylabel("Percentage of Files (%)", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylim(0, 105)

    # Clean legend without extra text
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.90, -0.12),
        ncol=2,
        fontsize=9,
        frameon=True,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    
    plt.tight_layout()
    save_fig(fig, out / "Figure_C_decision_outcomes.pdf")

if __name__ == "__main__":
    generate_figure_c()
