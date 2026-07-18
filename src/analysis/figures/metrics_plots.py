from __future__ import annotations

"""Metric comparison grid (methods × difficulty) for a results CSV.

Produces a 4×4 grid of boxplots comparing `eval_method` across four metrics
in rows (Exact Match, Similarity, BLEU-3, ROUGE-L) and four difficulty
subsets in columns (All, Easy, Medium, Hard).

Usage (CLI):
    python -m src.analysis.figures.metrics_plots --input-csv data.csv --name run1

This will save a PNG to results/figures/run1_metrics_grid.png
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tyro

from src.config.eval_methods import DEFAULT_METHOD_ORDER


# -----------------------------------------------------------------------------
# Global theme (match the rest of the results figures)
# -----------------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _order_from_present(present: list[str], desired: Optional[list[str]] = None) -> list[str]:
    present = [str(x) for x in present]
    desired = desired or DEFAULT_METHOD_ORDER
    return [m for m in desired if m in present] + [m for m in present if m not in desired]


def _palette_for(labels: list[str], base: str = "deep") -> dict[str, tuple[float, float, float]]:
    pal = sns.color_palette(base, max(3, len(labels)))
    return {lab: pal[i % len(pal)] for i, lab in enumerate(labels)}


def _slugify(s: object) -> str:
    try:
        s2 = str(s).strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")
        s2 = re.sub(r"[^a-z0-9_.-]+", "_", s2)
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2 or "metrics"
    except Exception:
        return "metrics"


def _coerce_metric(series: pd.Series, column: str) -> pd.Series:
    if column == "exact_match":
        # Accept bool-like and common string encodings
        s = series
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int)
        return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)
    # numeric metrics
    return pd.to_numeric(series, errors="coerce")


def _unique_instance_count(df: pd.DataFrame) -> int:
    """Count unique instances across methods using best-available identifier.

    Prefers 'id' if present; otherwise falls back to unique pairs of
    ('repo','file_name') if both present; otherwise returns len(df).
    """
    try:
        if df is None or df.empty:
            return 0
        if "id" in df.columns:
            return int(pd.Series(df["id"]).nunique(dropna=True))
        if {"repo", "file_name"}.issubset(df.columns):
            return int(df[["repo", "file_name"]].dropna().drop_duplicates().shape[0])
        return int(len(df))
    except Exception:
        return int(len(df))


# -----------------------------------------------------------------------------
# Core rendering
# -----------------------------------------------------------------------------

def render_metrics_grid(input_csv: Path | str, name: str, *, output_dir: Path = Path("results/figures"), show: bool = True) -> Path:
    """Render a 4×4 grid (metrics × difficulty) comparing methods and save PNG.

    - Rows (4): exact_match, similarity, bleu3, rouge_l
    - Columns (4): All, Easy, Medium, Hard
    - Each cell: boxplot of the metric by eval_method
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    if "eval_method" not in df.columns:
        raise ValueError("Column 'eval_method' not found in input CSV; required for comparisons.")

    # Normalize columns used for filtering/plotting
    df = df.copy()
    df["eval_method"] = df["eval_method"].astype(str)
    has_difficulty = "difficulty" in df.columns
    if has_difficulty:
        df["difficulty_norm"] = df["difficulty"].astype(str).str.strip().str.lower()

    # Metrics to plot (row order)
    metrics_info: list[tuple[str, str, tuple[Optional[float], Optional[float]]]] = [
        ("exact_match", "Exact Match", (0.0, 1.0)),
        ("similarity", "Similarity", (0.0, 1.0)),
        ("bleu3", "BLEU-3", (0.0, 1.0)),
        ("rouge_l", "ROUGE-L", (0.0, 1.0)),
    ]

    # Difficulty columns (All + three filters)
    diff_keys: list[tuple[str, Optional[str]]] = [
        ("All", None),
        ("Easy", "easy"),
        ("Medium", "medium"),
        ("Hard", "hard"),
    ]

    # Method ordering and palette (based on all rows present)
    present_methods = df["eval_method"].astype(str).unique().tolist()
    order = _order_from_present(present_methods, DEFAULT_METHOD_ORDER)
    pal_map = _palette_for(order, base=("tab20" if len(order) > 10 else "deep"))

    # Figure sizing heuristics: width scales with number of methods
    per_panel_w = max(2.8, 0.42 * max(3, len(order)))
    per_panel_h = 3.2
    fig_w = per_panel_w * len(diff_keys)
    fig_h = per_panel_h * len(metrics_info)
    fig, axes = plt.subplots(nrows=len(metrics_info), ncols=len(diff_keys), figsize=(fig_w, fig_h), sharey="row")

    # Ensure axes is 2D array even if len(...) == 1
    if len(metrics_info) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(diff_keys) == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, (col_name, y_label, ylim) in enumerate(metrics_info):
        # Skip entire row if column missing or all NaN
        if col_name not in df.columns:
            for c in range(len(diff_keys)):
                ax = axes[r][c]
                ax.axis("off")
                ax.text(0.5, 0.5, f"Column '{col_name}' not found", ha="center", va="center")
            continue

        for c, (col_title, diff_key) in enumerate(diff_keys):
            ax = axes[r][c]

            # Subset by difficulty if available/requested
            if diff_key is None or not has_difficulty:
                sub = df
            else:
                sub = df.loc[df["difficulty_norm"] == diff_key]
            n_subset = _unique_instance_count(sub)

            if sub.empty:
                ax.axis("off")
                msg = "No data" if has_difficulty else "No difficulty column"
                ax.text(0.5, 0.5, msg, ha="center", va="center")
                if r == 0:
                    ax.set_title(f"{col_title} (n={n_subset})")
                continue

            # Prepare data for plotting
            if col_name == "exact_match":
                # Compute per-method exact match rate (true count / total valid)
                rows: list[dict[str, object]] = []
                for m in order:
                    mask = sub["eval_method"].astype(str) == m
                    if not mask.any():
                        continue
                    s_raw = sub.loc[mask, "exact_match"]
                    valid = s_raw.notna()
                    den = int(valid.sum())
                    if den == 0:
                        continue
                    num = s_raw[valid].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).sum()
                    rate = float(num) / float(den)
                    rows.append({"eval_method": m, col_name: rate})
                work = pd.DataFrame(rows)
            else:
                work = pd.DataFrame({
                    "eval_method": sub["eval_method"].astype(str),
                    col_name: _coerce_metric(sub[col_name], col_name),
                })
                work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_name])

            if work.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "No values", ha="center", va="center")
                if r == 0:
                    ax.set_title(f"{col_title} (n={n_subset})")
                continue

            # Determine local order based on present methods to avoid empty categories
            present_local = work["eval_method"].astype(str).unique().tolist()
            order_local = [m for m in order if m in present_local]

            sns.boxplot(
                data=work,
                x="eval_method",
                y=col_name,
                hue="eval_method",       # same as x to enable palette mapping without deprecation
                order=order_local,
                hue_order=order_local,
                palette=pal_map,
                dodge=False,
                showfliers=False,
                width=0.65,
                linewidth=1.3,
                legend=False,
                ax=ax,
            )

            # Median callouts per method
            try:
                for i, m in enumerate(order_local):
                    series = (
                        work.loc[work["eval_method"].astype(str) == m, col_name]
                        .astype(float)
                        .dropna()
                    )
                    if series.empty:
                        continue
                    med = float(np.median(series.to_numpy()))
                    ax.annotate(
                        f"median {med:.3f}",
                        (i, med),
                        xytext=(10, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=8,
                        color="#222",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.7),
                        clip_on=False,
                    )
            except Exception:
                # Never fail rendering due to annotation issues
                pass

            # Titles and labels
            if r == 0:
                ax.set_title(f"{col_title} (n={n_subset})")
            if c == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")

            # Tick handling
            if r == len(metrics_info) - 1:
                ax.set_xlabel("method")
                for tick in ax.get_xticklabels():
                    tick.set_rotation(20)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            # Y formatting
            if ylim[0] is not None and ylim[1] is not None:
                ax.set_ylim(*ylim)
            ax.yaxis.grid(True, which="major", linewidth=1.0, alpha=0.25)
            ax.xaxis.grid(False)

    slug = _slugify(name)
    save_path = output_dir / f"{slug}_metrics_grid.png"
    fig.suptitle(f"{name} — Metrics by Method × Difficulty", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@dataclass
class Flags:
    input_csv: Path
    name: str
    output_dir: Path = Path("results/figures")
    show: bool = True


def main(flags: Flags) -> None:
    render_metrics_grid(flags.input_csv, flags.name, output_dir=flags.output_dir, show=flags.show)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


