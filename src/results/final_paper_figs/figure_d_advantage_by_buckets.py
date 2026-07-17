"""Figure D: Bypass Advantage by Difficulty and Project Size.

Two-row × three-column panel: EM bars + similarity line, faceted by model.
Row 1 = by difficulty, Row 2 = by project size.

Usage::

    python -m src.results.final_paper_figs.figure_d_advantage_by_buckets
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from src.results.final_paper_figs.shared import (
        CONFLICT_BUCKETS,
        DIFF_ORDER,
        OUTPUT_DIR,
        PERF,
        SIZE_ORDER,
        apply_style,
        common_ids,
        instance_agg,
        load_results,
        load_scenario,
        logger,
        save_fig,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.final_paper_figs.shared import (
        CONFLICT_BUCKETS,
        DIFF_ORDER,
        OUTPUT_DIR,
        PERF,
        SIZE_ORDER,
        apply_style,
        common_ids,
        instance_agg,
        load_results,
        load_scenario,
        logger,
        save_fig,
    )


FIG_D_MODEL_ORDER = ["LLaMA-3.1-8B", "Qwen3-32B", "GPT-5-nano"]
FIG_D_FIGSIZE = (13, 5.0)

# Figure D local text sizing. Adjust these values to tune this figure without
# changing the shared style used by other final-paper outputs.
FIG_D_TITLE_FONTSIZE = 12
FIG_D_AXIS_LABEL_FONTSIZE = 12
FIG_D_TICK_LABEL_FONTSIZE = 11
FIG_D_EM_LABEL_FONTSIZE = 9
FIG_D_SIM_LABEL_FONTSIZE = 9
FIG_D_ROW_LABEL_FONTSIZE = 12
FIG_D_LEGEND_FONTSIZE = 10


def _display_model_name(model: str) -> str:
    """Normalize model names for titles and tables."""
    if model in {"GPT-5-nano", "GPT-5-Nano"}:
        return "GPT-5-Nano"
    if model == "LLaMA-3.1-8B":
        return "Llama-3.1-8B"
    return model


def _bucket(n: int) -> str:
    for lo, hi, label in CONFLICT_BUCKETS:
        if lo <= n <= hi:
            return label
    return "11+"


def generate_figure_d(
    results_csv: Path | None = None,
    dataset_csv: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate Figure D: Advantage by difficulty, conflict count, project size."""
    apply_style()
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Figure D: Advantage by buckets")

    df = load_results(results_csv)
    common = common_ids(df)
    scenario = load_scenario(dataset_csv)
    inst = instance_agg(df, common)

    # Compute advantage
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

    # Difficulty from agent side
    if "difficulty_ag" in merged.columns:
        merged["difficulty"] = merged["difficulty_ag"]

    # Enrich with scenario metadata (conflict counts)
    scenario["id"] = scenario["id"].astype(str)
    merged = merged.merge(
        scenario[["id", "n_total_conflicts"]].drop_duplicates(),
        on="id",
        how="left",
    )
    merged["conflict_bucket"] = merged["n_total_conflicts"].apply(_bucket)

    # Project size from agent side
    if "project_size_ag" in merged.columns:
        merged["project_size"] = merged["project_size_ag"]

    # ── Table: same data as figure (stratification × model × category) ─
    table_rows = []
    bucket_labels = [b[2] for b in CONFLICT_BUCKETS]
    available_sizes = [s for s in SIZE_ORDER if s in merged["project_size"].values]

    for model in FIG_D_MODEL_ORDER:
        sub = merged[merged["model"] == model]
        for cat_col, categories, strat_name in [
            ("difficulty", DIFF_ORDER, "Difficulty"),
            ("conflict_bucket", bucket_labels, "Conflict Count"),
            ("project_size", available_sizes, "Project Size"),
        ]:
            for c in categories:
                cell = sub[sub[cat_col] == c]
                n = len(cell)
                em_adv = cell["adv_exact_match"].mean() if n > 0 else np.nan
                sim_adv = cell["adv_similarity"].mean() if n > 0 else np.nan
                table_rows.append({
                    "Stratification": strat_name,
                    "Category": str(c),
                    "Model": _display_model_name(model),
                    "n": n,
                    "EM_Advantage": round(em_adv, 2) if not np.isnan(em_adv) else "",
                    "Similarity_Advantage": round(sim_adv, 2) if not np.isnan(sim_adv) else "",
                })

    table_df = pd.DataFrame(table_rows)
    table_path = out / "Table_Figure_D_advantage_by_buckets.csv"
    table_df.to_csv(table_path, index=False)
    logger.info(f"  Saved {table_path.name}")

    # ── Plot 2×3 grid ────────────────────────────────────────────────
    # Create a mapping that handles case-insensitive model names
    model_colors = {}
    for model in FIG_D_MODEL_ORDER:
        if model == "GPT-5-Nano":
            model_colors[model] = "#1f77b4"
        elif model == "GPT-5-nano":  # Handle lowercase variant
            model_colors[model] = "#1f77b4"
        elif model == "Qwen3-32B":
            model_colors[model] = "#ff7f0e"
        elif model == "LLaMA-3.1-8B":
            model_colors[model] = "#2ca02c"
        else:
            # Fallback color for unknown models
            model_colors[model] = "#7f7f7f"

    fig, axes = plt.subplots(2, 3, figsize=FIG_D_FIGSIZE)

    em_lo, em_hi = -0.03, 0.22
    sim_lo, sim_hi = -0.1, 0.55

    def _draw_row(row_idx, sub_merged, categories, cat_col, xlabel):
        for col_idx, model in enumerate(FIG_D_MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = sub_merged[sub_merged["model"] == model]

            x = np.arange(len(categories))
            em_vals = [
                sub[sub[cat_col] == c]["adv_exact_match"].mean() for c in categories
            ]
            sim_vals = [
                sub[sub[cat_col] == c]["adv_similarity"].mean() for c in categories
            ]
            ns = [len(sub[sub[cat_col] == c]) for c in categories]

            # Replace NaN with 0 for empty buckets
            em_vals = [v if not np.isnan(v) else 0 for v in em_vals]
            sim_vals = [v if not np.isnan(v) else 0 for v in sim_vals]

            # Round values to 2 decimal places for plotting and labels
            em_vals_rounded = [round(v, 2) for v in em_vals]
            sim_vals_rounded = [round(v, 2) for v in sim_vals]

            bars = ax.bar(
                x,
                em_vals_rounded,
                0.55,
                color=model_colors[model],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )
            # EM bar labels: always above bar, centered
            for bar, v in zip(bars, em_vals_rounded):
                ax.annotate(
                    f"{v:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, max(v, 0)),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=FIG_D_EM_LABEL_FONTSIZE,
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )

            ax2 = ax.twinx()
            ax2.plot(
                x,
                sim_vals_rounded,
                "s-",
                color="darkred",
                markersize=5,
                linewidth=1.5,
                zorder=6,
            )
            # Similarity line labels: placed to avoid EM bar labels.
            # Normalize both series to [0,1] in their respective axis ranges
            # so we can detect visual proximity.
            em_range = em_hi - em_lo
            sim_range = sim_hi - sim_lo
            for i, v in enumerate(sim_vals_rounded):
                em_norm = (em_vals_rounded[i] - em_lo) / em_range
                sim_norm = (v - sim_lo) / sim_range
                # If the sim marker is visually close to the EM bar top,
                # push the label below the marker instead of above.
                if abs(sim_norm - em_norm) < 0.12:
                    y_shift = -11
                    va_lbl = "top"
                else:
                    y_shift = 8
                    va_lbl = "bottom"
                ax2.annotate(
                    f"{v:+.2f}",
                    (x[i], v),
                    textcoords="offset points",
                    xytext=(0, y_shift),
                    fontsize=FIG_D_SIM_LABEL_FONTSIZE,
                    color="darkred",
                    ha="center",
                    va=va_lbl,
                    zorder=6,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=FIG_D_TICK_LABEL_FONTSIZE)
            ax.tick_params(axis="y", labelsize=FIG_D_TICK_LABEL_FONTSIZE)
            ax2.tick_params(axis="y", labelsize=FIG_D_TICK_LABEL_FONTSIZE)
            if row_idx == 1:
                ax.set_xlabel(
                    xlabel,
                    fontweight="bold",
                    fontsize=FIG_D_AXIS_LABEL_FONTSIZE,
                )
            ax.set_ylabel(
                "EM Advantage" if col_idx == 0 else "",
                fontweight="bold",
                fontsize=FIG_D_AXIS_LABEL_FONTSIZE,
            )
            if col_idx == len(FIG_D_MODEL_ORDER) - 1:
                ax2.set_ylabel(
                    "Similarity Advantage",
                    color="darkred",
                    fontweight="bold",
                    fontsize=FIG_D_AXIS_LABEL_FONTSIZE,
                )
            else:
                ax2.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(
                    _display_model_name(model),
                    fontweight="bold",
                    fontsize=FIG_D_TITLE_FONTSIZE,
                )

            ax.set_ylim(em_lo, em_hi)
            ax2.set_ylim(sim_lo, sim_hi)

    # Row 1: By difficulty
    _draw_row(0, merged, DIFF_ORDER, "difficulty", "Difficulty")

    # Row 2: By project size
    available_sizes = [s for s in SIZE_ORDER if s in merged["project_size"].values]
    _draw_row(1, merged, available_sizes, "project_size", "Project Size")

    # Row labels - reduced spacing by changing x position from -0.35 to -0.25
    axes[0, 0].annotate(
        "By Difficulty",
        xy=(-0.27, 0.5),
        xycoords="axes fraction",
        fontsize=FIG_D_ROW_LABEL_FONTSIZE,
        fontweight="bold",
        rotation=90,
        va="center",
    )
    # axes[1, 0].annotate(
    #     "By Conflict Count",
    #     xy=(-0.25, 0.5),
    #     xycoords="axes fraction",
    #     fontsize=FIG_D_ROW_LABEL_FONTSIZE,
    #     fontweight="bold",
    #     rotation=90,
    #     va="center",
    # )
    axes[1, 0].annotate(
        "By Project Size",
        xy=(-0.27, 0.5),
        xycoords="axes fraction",
        fontsize=FIG_D_ROW_LABEL_FONTSIZE,
        fontweight="bold",
        rotation=90,
        va="center",
    )

    # Legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.5, label="EM Advantage (bars)"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="darkred",
            linewidth=1.5,
            markersize=5,
            label="Similarity Advantage (line)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=FIG_D_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, out / "Figure_D_advantage_by_buckets.pdf")


if __name__ == "__main__":
    generate_figure_d()
