from __future__ import annotations

"""Per-model difficulty plots comparing Agent vs Bypass distributions.

For each model in the dataset (or a specific model if provided), this module
renders FOUR separate figures/windows corresponding to difficulty subsets:
All, Easy, Medium, Hard. Each figure shows a 2×2 grid of boxplots for the
four core metrics (Exact Match, Similarity, BLEU-3, ROUGE-L), comparing the
spread of values for two categories: agent vs bypass.

Notes:
- "bypass" aggregates any eval_method that contains the substring "bypass"
  (e.g., bypass, bypass7, bypass_only).
- Exact match is coerced to 0/1 to show its distribution.
- Figures are also saved under results/figures/ by default.

CLI examples:
    # One set for each model present in the CSV
    python -m src.results.figures.metrics_by_model_difficulty \
        --input-csv data.csv --name run1

    # Only for a given model
    python -m src.results.figures.metrics_by_model_difficulty \
        --input-csv data.csv --name run1 --model-name "llama-3.1-8b"
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


# -----------------------------------------------------------------------------
# Theme (match other figures in this project)
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
# Utilities
# -----------------------------------------------------------------------------


def _slugify(s: object) -> str:
    try:
        s2 = (
            str(s)
            .strip()
            .lower()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        s2 = re.sub(r"[^a-z0-9_.-]+", "_", s2)
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2 or "metrics"
    except Exception:
        return "metrics"


def _coerce_metric(series: pd.Series, column: str) -> pd.Series:
    if column == "exact_match":
        s = series
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int)
        return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)
    return pd.to_numeric(series, errors="coerce")


def _unique_instance_count(df: pd.DataFrame) -> int:
    """Best-effort count of unique instances for panel n-labeling."""
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


def _method_to_category(method: object) -> Optional[str]:
    m = str(method).strip().lower()
    if m == "agent":
        return "agent"
    if "bypass" in m:
        return "bypass"
    return None


# -----------------------------------------------------------------------------
# Core rendering
# -----------------------------------------------------------------------------


def _render_one_difficulty(
    df_model: pd.DataFrame,
    *,
    difficulty_key: Optional[str],
    metrics_info: list[tuple[str, str, tuple[Optional[float], Optional[float]]]],
    name: str,
    model_name: str,
    output_dir: Path,
    show: bool,
) -> Optional[Path]:
    has_difficulty = "difficulty" in df_model.columns
    if difficulty_key is None or not has_difficulty:
        sub = df_model
        diff_title = "All"
        diff_slug = "all"
    else:
        work = df_model.copy()
        work["difficulty_norm"] = work["difficulty"].astype(str).str.strip().str.lower()
        sub = work.loc[work["difficulty_norm"] == difficulty_key]
        diff_title = {
            "easy": "Easy",
            "medium": "Medium",
            "hard": "Hard",
        }.get(difficulty_key, difficulty_key.title())
        diff_slug = difficulty_key

    n_subset = _unique_instance_count(sub)

    # Prepare per-metric data frames with a common category column
    # Keep only agent vs bypass
    if sub.empty:
        # Still produce an empty figure for consistency
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8.4, 6.4), sharey=False)
        for ax in axes.ravel():
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.suptitle(f"{name} — {model_name} — {diff_title} (n={n_subset})", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        slug = _slugify(name)
        mslug = _slugify(model_name)
        save_path = output_dir / f"{slug}_{mslug}_by_model_difficulty_{diff_slug}.png"
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    pal_map = {"agent": sns.color_palette("deep")[2], "bypass": sns.color_palette("deep")[3]}
    order_global = ["agent", "bypass"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8.4, 6.4), sharey="row")
    axes = np.array(axes)

    for idx, (col_name, y_label, ylim) in enumerate(metrics_info):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        if col_name not in sub.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Column '{col_name}' not found", ha="center", va="center")
            continue

        work = pd.DataFrame({
            "method_cat": sub["eval_method"].map(_method_to_category),
            col_name: _coerce_metric(sub[col_name], col_name),
        })
        work = work.dropna(subset=["method_cat", col_name])
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_name])

        if work.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No values", ha="center", va="center")
            continue

        order_local = [x for x in order_global if x in work["method_cat"].unique().tolist()]

        sns.boxplot(
            data=work,
            x="method_cat",
            y=col_name,
            order=order_local,
            hue="method_cat",  # for palette mapping without legend
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
            for i, lab in enumerate(order_local):
                series = (
                    work.loc[work["method_cat"].astype(str) == lab, col_name]
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

        # Labels and axes formatting
        ax.set_title(y_label)
        ax.set_xlabel("method")
        ax.set_ylabel(y_label)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        if ylim[0] is not None and ylim[1] is not None:
            ax.set_ylim(*ylim)
        ax.yaxis.grid(True, which="major", linewidth=1.0, alpha=0.25)
        ax.xaxis.grid(False)

    slug = _slugify(name)
    mslug = _slugify(model_name)
    save_path = output_dir / f"{slug}_{mslug}_by_model_difficulty_{diff_slug}.png"
    fig.suptitle(f"{name} — {model_name} — {diff_title} (n={n_subset})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def render_model_difficulty_plots(
    input_csv: Path | str,
    name: str,
    *,
    output_dir: Path = Path("results/figures"),
    show: bool = True,
    model_name: Optional[str] = None,
) -> list[Path]:
    """Render four figures (All, Easy, Medium, Hard) for a given model or all models.

    Returns a list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    if "eval_method" not in df.columns:
        raise ValueError("Column 'eval_method' not found in input CSV; required for comparisons.")
    if "model_name" not in df.columns:
        df["model_name"] = "unknown"

    # Normalize types
    df = df.copy()
    df["eval_method"] = df["eval_method"].astype(str)
    df["model_name"] = df["model_name"].astype(str)

    metrics_info: list[tuple[str, str, tuple[Optional[float], Optional[float]]]] = [
        ("exact_match", "Exact Match", (0.0, 1.0)),
        ("similarity", "Similarity", (0.0, 1.0)),
        ("bleu3", "BLEU-3", (0.0, 1.0)),
        ("rouge_l", "ROUGE-L", (0.0, 1.0)),
    ]

    diff_keys: list[tuple[str, Optional[str]]] = [
        ("All", None),
        ("Easy", "easy"),
        ("Medium", "medium"),
        ("Hard", "hard"),
    ]

    models: list[str]
    if model_name is not None:
        models = [str(model_name).strip()]
    else:
        models = (
            df["model_name"].astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
        )
        models = sorted(models)

    saved: list[Path] = []
    for model in models:
        df_model = df.loc[df["model_name"].astype(str).str.strip() == str(model).strip()].copy()
        if df_model.empty:
            # still create empty figures to indicate no data
            for _, key in diff_keys:
                p = _render_one_difficulty(
                    df_model,
                    difficulty_key=key,
                    metrics_info=metrics_info,
                    name=name,
                    model_name=model,
                    output_dir=output_dir,
                    show=show,
                )
                if p is not None:
                    saved.append(p)
            continue

        for _, key in diff_keys:
            p = _render_one_difficulty(
                df_model,
                difficulty_key=key,
                metrics_info=metrics_info,
                name=name,
                model_name=model,
                output_dir=output_dir,
                show=show,
            )
            if p is not None:
                saved.append(p)

    return saved


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@dataclass
class Flags:
    input_csv: Path
    name: str
    output_dir: Path = Path("results/figures")
    show: bool = True
    split_by_model: bool = True
    model_name: Optional[str] = None


def main(flags: Flags) -> None:
    if flags.split_by_model and flags.model_name is None:
        # All models present in the CSV
        render_model_difficulty_plots(
            flags.input_csv,
            flags.name,
            output_dir=flags.output_dir,
            show=flags.show,
            model_name=None,
        )
    else:
        # Specific model
        render_model_difficulty_plots(
            flags.input_csv,
            flags.name,
            output_dir=flags.output_dir,
            show=flags.show,
            model_name=flags.model_name,
        )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)







