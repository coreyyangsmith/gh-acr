"""Figure B: Bypass Advantage Distribution.

Horizontal violins (Sim, BLEU-3, ROUGE-L) with boxplot + mean overlay;
Exact Match as 100% stacked bar (worse / tie / better). Summary table
is exported to Table_Figure_B_bypass_advantage_summary.csv (not drawn on figure).

Usage::

    python -m src.analysis.final_paper_figs.figure_b_bypass_advantage
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from src.analysis.final_paper_figs.shared import (
    MODEL_FIGB_LABELS,
    OUTPUT_DIR,
    PERF,
    apply_style,
    common_ids,
    instance_agg,
    load_results,
    logger,
    save_fig,
)

# Continuous metrics only (violins); EM handled separately
CONTINUOUS_METRICS = [
    ("adv_similarity", "Similarity"),
    ("adv_bleu3", "BLEU-3"),
    ("adv_rouge_l", "ROUGE-L"),
]
VIOLIN_COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4"]  # LLaMA=green, Qwen=orange, GPT=blue (matches Figure D)
# EM stacked bar segment colors: worse, tie, better
# EM_WORSE_COLOR = "#d62728"
# EM_TIE_COLOR = "#7f7f7f"
# EM_BETTER_COLOR = "#2ca02c"

# Shared y-axis limits for the three violin panels (no compression)
YLIM_VIOLINS = (-0.25, 0.65)
FIG_B_MODEL_ORDER = ["LLaMA-3.1-8B", "Qwen3-32B", "GPT-5-nano"]

def _display_label(model: str) -> str:
    """Return display label with normalized GPT casing."""
    raw = MODEL_FIGB_LABELS.get(model, model)
    if raw in {"GPT-5n", "GPT-5-nano"}:
        return "GPT-5-Nano"
    return raw

def _draw_vertical_violin_panel(ax, merged, col, label, positions, short_labels, show_y_labels=True):
    """Draw one vertical violin panel with boxplot overlay and mean dot."""
    data_by_model = [
        merged[merged["model"] == m][col].dropna().values for m in FIG_B_MODEL_ORDER
    ]

    # Violins (vertical: x = position/model, y = advantage)
    parts = ax.violinplot(
        data_by_model,
        positions=positions,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(VIOLIN_COLORS[i])
        body.set_edgecolor("black")
        body.set_linewidth(1.2)
        body.set_alpha(0.75)

    # Boxplot overlay: median + IQR (drawn manually for compatibility)
    for i, data in enumerate(data_by_model):
        if len(data) == 0:
            continue
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        pos = positions[i]
        width = 0.12
        # IQR box (vertical: x from pos-width/2 to pos+width/2, y from q1 to q3)
        ax.add_patch(
            plt.Rectangle(
                (pos - width / 2, q1),
                width,
                q3 - q1,
                fill=False,
                edgecolor="black",
                linewidth=1,
            )
        )
        # Median line (horizontal segment at y=med)
        ax.plot([pos - width / 2, pos + width / 2], [med, med], "k-", linewidth=1.5)

    # Mean as single dot per model
    for i, model in enumerate(FIG_B_MODEL_ORDER):
        vals = merged[merged["model"] == model][col].dropna()
        if len(vals) > 0:
            ax.scatter(
                positions[i],
                vals.mean(),
                color="red",
                s=28,
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)
    ax.set_ylabel("Multi-Agent Win Rate" if show_y_labels else "", fontweight="bold")
    ax.set_title(label, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, fontsize=9, fontweight="bold", rotation=15, ha="right")
    ax.set_ylim(-0.25, 1.05)
    ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax.yaxis.grid(True, linestyle="-", alpha=0.3)
    if not show_y_labels:
        ax.tick_params(labelleft=False)

# def _draw_em_stacked_bar(ax, merged, positions, short_labels, show_y_labels=True):
#     """Draw EM panel: 100% stacked horizontal bar (worse / tie / better) per model."""
#     col = "adv_exact_match"
#     if col not in merged.columns:
#         return

#     segment_pcts = []  # list of (pct_worse, pct_tie, pct_better) per model
#     for model in MODEL_ORDER:
#         vals = merged[merged["model"] == model][col].dropna()
#         n = len(vals)
#         if n == 0:
#             segment_pcts.append((0, 0, 0))
#             continue
#         n_worse = (vals < 0).sum()
#         n_tie = (vals == 0).sum()
#         n_better = (vals > 0).sum()
#         segment_pcts.append((n_worse / n * 100, n_tie / n * 100, n_better / n * 100))

#     bar_height = 0.5
#     left = np.zeros(len(MODEL_ORDER))
#     for seg_idx, (pct_worse, pct_tie, pct_better) in enumerate(segment_pcts):
#         # Order: worse (left), tie (middle), better (right)
#         for width, color in [
#             (pct_worse, EM_WORSE_COLOR),
#             (pct_tie, EM_TIE_COLOR),
#             (pct_better, EM_BETTER_COLOR),
#         ]:
#             if width > 0:
#                 ax.barh(
#                     positions[seg_idx],
#                     width,
#                     left=left[seg_idx],
#                     height=bar_height,
#                     color=color,
#                     edgecolor="white",
#                     linewidth=0.5,
#                 )
#                 # Percent label centered in segment (only if segment wide enough)
#                 if width >= 6:
#                     ax.text(
#                         left[seg_idx] + width / 2,
#                         positions[seg_idx],
#                         f"{width:.0f}%",
#                         ha="center",
#                         va="center",
#                         fontsize=8,
#                         color="white" if width > 15 else "black",
#                         fontweight="bold",
#                     )
#                 left[seg_idx] += width

#     ax.set_xlim(0, 115)
#     ax.set_xticks([0, 25, 50, 75, 100])
#     ax.set_xticklabels(["0", "25", "50", "75", "100"])
#     ax.set_xlabel("Multi-Agent Winrate (Exact Match)", fontweight="bold")
#     ax.set_yticks(positions)
#     ax.set_yticklabels(short_labels if show_y_labels else [""] * len(positions), fontsize=9, fontweight="bold")
#     ax.xaxis.grid(True, linestyle="-", alpha=0.3)

def generate_figure_b(
    results_csv: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate Figure B: Bypass advantage (horizontal violins + EM stacked bar + table)."""
    apply_style()
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Figure B: Bypass advantage distribution")

    df = load_results(results_csv)
    common = common_ids(df)
    inst = instance_agg(df, common)

    agent = inst[inst["method"] == "Agent"].copy()
    bypass = inst[inst["method"] == "Bypass"].copy()
    merged = agent.merge(bypass, on=["id", "model"], suffixes=("_ag", "_by"))

    for m in PERF:
        ac, bc = f"{m}_ag", f"{m}_by"
        if ac in merged.columns and bc in merged.columns:
            merged[f"adv_{m}"] = (
                pd.to_numeric(merged[bc], errors="coerce")
                - pd.to_numeric(merged[ac], errors="coerce")
            )

    # Update labels to use "GPT-5-Nano" instead of "GPT-5n"
    short_labels = []
    for m in FIG_B_MODEL_ORDER:
        short_labels.append(_display_label(m))
    
    n_models = len(FIG_B_MODEL_ORDER)
    positions = np.arange(n_models)

    # Build summary table and export to CSV (not drawn on figure)
    table_rows = []
    for model in FIG_B_MODEL_ORDER:
        row = {"Model": _display_label(model)}
        for col, metric_name in CONTINUOUS_METRICS:
            if col in merged.columns:
                vals = merged[merged["model"] == model][col].dropna()
                pct = (vals > 0).mean() * 100 if len(vals) > 0 else 0
                mu = vals.mean() if len(vals) > 0 else 0
                row[metric_name] = f"{pct:.0f}% / {mu:+.3f}"
            else:
                row[metric_name] = ""
        # col = "adv_exact_match"
        # if col in merged.columns:
        #     vals = merged[merged["model"] == model][col].dropna()
        #     pct = (vals > 0).mean() * 100 if len(vals) > 0 else 0
        #     mu = vals.mean() if len(vals) > 0 else 0
        #     row["Exact Match"] = f"{pct:.0f}% / {mu:+.3f}"
        # else:
        #     row["Exact Match"] = ""
        table_rows.append(row)
    # table_df = pd.DataFrame(table_rows, columns=["Model", "Similarity", "BLEU-3", "ROUGE-L", "Exact Match"])
    table_df = pd.DataFrame(table_rows, columns=["Model", "Similarity", "BLEU-3", "ROUGE-L"])
    table_csv_path = out / "Table_Figure_B_bypass_advantage_summary.csv"
    table_df.to_csv(table_csv_path, index=False)
    logger.info(f"  Saved {table_csv_path.name}")

    # Layout: 3 panels in a row (models on x-axis, win rate on y-axis)
    fig = plt.figure(figsize=(11, 2.5))
    axes = [fig.add_subplot(1, 3, j + 1) for j in range(3)]

    # Share y-axis across the three violin panels
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])

    for idx, (col, label) in enumerate(CONTINUOUS_METRICS):
        if col in merged.columns:
            _draw_vertical_violin_panel(
                axes[idx], merged, col, label, positions, short_labels,
                show_y_labels=(idx == 0),
            )

    # _draw_em_stacked_bar(axes[3], merged, positions, short_labels, show_y_labels=False)

    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.09, right=0.98, wspace=0.18)
    save_fig(fig, out / "Figure_B_bypass_advantage.pdf")

if __name__ == "__main__":
    generate_figure_b()
