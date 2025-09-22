from __future__ import annotations

"""Metrics by repository size grid, one figure per difficulty level.

For each difficulty in {All, Easy, Medium, Hard}, renders a grid where
- rows = project sizes: Tiny, Small, Medium, Large, Huge
- cols = metrics: exact_match, similarity, bleu3, rouge_l

Each cell is a boxplot comparing values by `eval_method`.
Exact Match is computed as a per-method rate within the size×difficulty subset.

CLI:
    python -m src.results.figures.metrics_by_size --input-csv data.csv --name run1

Outputs four PNGs, one per difficulty (skip Easy/Medium/Hard if difficulty missing).
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
# Theme
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


SIZE_LABELS: list[str] = ["Small", "Medium", "Large", "Huge"]


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
        return s2 or "metrics_by_size"
    except Exception:
        return "metrics_by_size"


def _normalize_size(series: pd.Series) -> pd.Series:
    mapping = {"tiny": "Small", "small": "Small", "medium": "Medium", "large": "Large", "huge": "Huge"}
    return series.astype(str).str.strip().str.lower().map(lambda s: mapping.get(s, "Unknown"))


def _coerce_metric(series: pd.Series, column: str) -> pd.Series:
    if column == "exact_match":
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        return series.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)
    return pd.to_numeric(series, errors="coerce")


def _unique_instance_count(df: pd.DataFrame) -> int:
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


def _compute_em_rate_per_method(df: pd.DataFrame, *, order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for m in order:
        mask = df["eval_method"].astype(str) == m
        if not mask.any():
            continue
        s_raw = df.loc[mask, "exact_match"]
        valid = s_raw.notna()
        den = int(valid.sum())
        if den == 0:
            continue
        num = s_raw[valid].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).sum()
        rate = float(num) / float(den)
        rows.append({"eval_method": m, "exact_match": rate})
    return pd.DataFrame(rows)


def _plot_grid_for_difficulty(
    df: pd.DataFrame,
    *,
    difficulty_title: str,
    output_path: Path,
    show: bool,
) -> None:
    # Method order + palette based on present methods in this difficulty
    present_methods = df["eval_method"].astype(str).unique().tolist()
    order = _order_from_present(present_methods, DEFAULT_METHOD_ORDER)
    pal_map = _palette_for(order, base=("tab20" if len(order) > 10 else "deep"))

    # Metrics as columns
    metrics_info: list[tuple[str, str, tuple[Optional[float], Optional[float]]]] = [
        ("exact_match", "Exact Match", (0.0, 1.0)),
        ("similarity", "Similarity", (0.0, 1.0)),
        ("bleu3", "BLEU-3", (0.0, 1.0)),
        ("rouge_l", "ROUGE-L", (0.0, 1.0)),
    ]

    # Figure dimensions
    n_rows = len(SIZE_LABELS)
    n_cols = len(metrics_info)
    per_panel_w = max(2.6, 0.42 * max(3, len(order)))
    per_panel_h = 2.6
    fig_w = per_panel_w * n_cols
    fig_h = per_panel_h * n_rows
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w, fig_h), sharex=False, sharey="col")

    # Ensure axes is 2D
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Normalize project size labels
    if "project_size" in df.columns:
        df = df.copy()
        df["project_size_norm"] = _normalize_size(df["project_size"])  # Tiny..Huge or Unknown
    else:
        df = df.copy()
        df["project_size_norm"] = "Unknown"

    for r, size_label in enumerate(SIZE_LABELS):
        df_size = df.loc[df["project_size_norm"] == size_label]
        n_size_unique = _unique_instance_count(df_size)

        for c, (col_name, y_label, ylim) in enumerate(metrics_info):
            ax = axes[r][c]

            if df_size.empty or (col_name not in df_size.columns and col_name != "exact_match"):
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                if r == 0:
                    ax.set_title(y_label)
                if c == 0:
                    ax.set_ylabel(f"{size_label} (n={n_size_unique})")
                continue

            # Prepare per-cell data
            if col_name == "exact_match":
                work = _compute_em_rate_per_method(df_size, order=order)
            else:
                work = pd.DataFrame({
                    "eval_method": df_size["eval_method"].astype(str),
                    col_name: _coerce_metric(df_size[col_name], col_name),
                })
                work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_name])

            if work.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "No values", ha="center", va="center")
                if r == 0:
                    ax.set_title(y_label)
                if c == 0:
                    ax.set_ylabel(f"{size_label} (n={n_size_unique})")
                continue

            # Local order to avoid empty categories
            present_local = work["eval_method"].astype(str).unique().tolist()
            order_local = [m for m in order if m in present_local]

            sns.boxplot(
                data=work,
                x="eval_method",
                y=col_name,
                hue="eval_method",
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

            # Median callouts
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
                pass

            # Labels and formatting
            if r == 0:
                ax.set_title(y_label)
            if c == 0:
                ax.set_ylabel(f"{size_label} (n={n_size_unique})")
            else:
                ax.set_ylabel("")

            if r == n_rows - 1:
                ax.set_xlabel("method")
                for tick in ax.get_xticklabels():
                    tick.set_rotation(20)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if ylim[0] is not None and ylim[1] is not None:
                ax.set_ylim(*ylim)
            ax.yaxis.grid(True, which="major", linewidth=1.0, alpha=0.25)
            ax.xaxis.grid(False)

    fig.suptitle(difficulty_title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def render_metrics_by_size_grids(
    input_csv: Path | str,
    name: str,
    *,
    output_dir: Path = Path("results/figures"),
    show: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    if "eval_method" not in df.columns:
        raise ValueError("Column 'eval_method' not found in input CSV; required for comparisons.")

    # Normalize difficulty
    have_diff = "difficulty" in df.columns
    work = df.copy()
    if have_diff:
        work["difficulty_norm"] = work["difficulty"].astype(str).str.strip().str.lower()

    # Difficulty sheets
    diff_defs: list[tuple[str, Optional[str]]] = [
        ("All", None),
        ("Easy", "easy"),
        ("Medium", "medium"),
        ("Hard", "hard"),
    ]

    outputs: list[Path] = []
    for title, key in diff_defs:
        if key is None or not have_diff:
            subset = work
        else:
            subset = work.loc[work["difficulty_norm"] == key]

        # Skip empty subsets (except All which we still render as empty panels)
        if subset.empty and key is not None:
            continue

        slug = _slugify(name)
        key_slug = "all" if key is None else key
        out_path = output_dir / f"{slug}_{key_slug}_metrics_by_size.png"
        title_full = f"{name} — Metrics by Method × Size — {title}"
        _plot_grid_for_difficulty(subset, difficulty_title=title_full, output_path=out_path, show=show)
        outputs.append(out_path)

    return outputs


@dataclass
class Flags:
    input_csv: Path
    name: str
    output_dir: Path = Path("results/figures")
    show: bool = True


def main(flags: Flags) -> None:
    render_metrics_by_size_grids(flags.input_csv, flags.name, output_dir=flags.output_dir, show=flags.show)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


