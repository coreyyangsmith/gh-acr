"""Figure 2: Bypass Decision Outcomes (horizontal stacked bar).

A/B/MIX selection shares on the common-id bypass7 subset, rendered as
horizontal bars with a compact vertical footprint for a two-column layout.

Usage::

    python -m src.results.final_paper_figs.figure_2_decision_outcomes
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.results.final_paper_figs.shared import (
        MODEL_SHORT,
        OUTPUT_DIR,
        apply_style,
        common_ids,
        load_results,
        logger,
        save_fig,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.final_paper_figs.shared import (
        MODEL_SHORT,
        OUTPUT_DIR,
        apply_style,
        common_ids,
        load_results,
        logger,
        save_fig,
    )


def generate_figure_2(results_csv: Path | None = None,
                      output_dir: Path | None = None) -> None:
    """Generate Figure 2: horizontal stacked A/B/MIX decision bar chart."""
    apply_style()
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 2: Decision outcomes (horizontal bars)")

    df = load_results(results_csv)
    common = common_ids(df)

    bypass = df[(df["eval_method"] == "bypass7") & (df["id"].isin(common))].copy()
    bypass["model"] = bypass["model_name"].map(MODEL_SHORT)

    decision_colors = {"A": "#a6cee3", "B": "#b2df8a", "MIX": "#fb9a99"}
    decision_order = ["A", "B", "MIX"]

    fig_model_order = ["LLaMA-3.1-8B", "Qwen3-32B", "GPT-5-nano"]
    fig_model_labels = {
        "LLaMA-3.1-8B": "Llama-3.1-8B",
        "Qwen3-32B": "Qwen3-32B",
        "GPT-5-nano": "GPT-5-Nano",
    }

    # Compact horizontal figure: narrow height for three rows
    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

    model_pcts: dict[str, dict[str, float]] = {}
    for model in fig_model_order:
        sub = bypass[bypass["model"] == model]
        sub = sub[sub["bypass_method"].isin(decision_order)]
        total = len(sub)
        if total == 0:
            continue
        counts = sub["bypass_method"].value_counts()
        model_pcts[model] = {
            d: counts.get(d, 0) / total * 100 for d in decision_order
        }

    y = np.arange(len(fig_model_order))
    lefts = np.zeros(len(fig_model_order))
    bar_height = 0.78  # Increased bar thickness from 0.62 to 0.78

    for decision in decision_order:
        vals = [model_pcts.get(m, {}).get(decision, 0) for m in fig_model_order]
        ax.barh(
            y,
            vals,
            bar_height,
            left=lefts,
            label=decision,
            color=decision_colors[decision],
            edgecolor="white",
            linewidth=0.5,
        )
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 3:
                ax.text(
                    l + v / 2,
                    y[i],
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
        lefts += np.array(vals)

    # Models read top-to-bottom in the same order as Figure C (left-to-right)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(
        [fig_model_labels[m] for m in fig_model_order], fontsize=8
    )
    ax.set_xlabel("Percentage of Files (%)", fontweight="bold", fontsize=8)
    ax.set_ylabel("Models", fontweight="bold", fontsize=8)  # <-- Y Axis label inserted here
    ax.set_xlim(0, 105)

    # Legend bottom-right, aligned with x-axis label
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True
    )

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.38)
    save_fig(fig, out / "Figure_2.pdf")


if __name__ == "__main__":
    generate_figure_2()
