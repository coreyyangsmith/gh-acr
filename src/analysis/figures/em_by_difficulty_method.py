from __future__ import annotations

"""Exact Match bars (agent vs bypass) by difficulty, per model.

For each model in the dataset (or a specific model if provided), this module
renders ONE figure containing FOUR panels corresponding to difficulty subsets:
All, Easy, Medium, Hard. Each panel shows two bars (agent vs bypass) with the
exact match percentage (rate of True over valid rows).

Notes:
- "bypass" aggregates any eval_method that contains the substring "bypass"
  (e.g., bypass, bypass7, bypass_only).
- Figures are saved to results/figures/ by default, one PNG per difficulty per
  model, e.g.: {name}_{model}_em_by_difficulty_all.png

CLI examples:
    # For all models present in the CSV (one PNG per model)
    python -m src.analysis.figures.em_by_difficulty_method \
        --input-csv data.csv --name run1

    # Only for a specific model (one PNG)
    python -m src.analysis.figures.em_by_difficulty_method \
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
        return s2 or "em_by_difficulty"
    except Exception:
        return "em_by_difficulty"


def _method_to_category(method: object) -> Optional[str]:
    m = str(method).strip().lower()
    if m == "agent":
        return "agent"
    if "bypass" in m:
        return "bypass"
    return None


def _coerce_em(series: pd.Series) -> pd.Series:
    """Coerce exact_match-like column to boolean then float (0/1)."""
    s = series
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"]).astype(float)


# -----------------------------------------------------------------------------
# Core rendering (one difficulty per figure)
# -----------------------------------------------------------------------------


def _render_grid_for_model(
    df_model: pd.DataFrame,
    *,
    name: str,
    model_name: str,
    output_dir: Path,
    show: bool,
) -> Optional[Path]:
    """Render a 1×4 grid of EM bars (agent vs bypass) for All/Easy/Medium/Hard."""
    has_difficulty = "difficulty" in df_model.columns

    def _panel_stats(sub: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if sub is None or sub.empty:
            return pd.DataFrame({"method_cat": [], "em_rate": []}), 0
        work = pd.DataFrame({
            "method_cat": sub["eval_method"].map(_method_to_category),
            "em": _coerce_em(sub["exact_match"]) if "exact_match" in sub.columns else np.nan,
        })
        work = work.dropna(subset=["method_cat", "em"])  # require mapping + valid EM
        if work.empty:
            return pd.DataFrame({"method_cat": [], "em_rate": []}), 0
        stats = (
            work.groupby("method_cat")["em"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "em_rate"})
        )
        return stats, int(len(work))

    # Build subsets
    work = df_model.copy()
    if has_difficulty:
        work["difficulty_norm"] = work["difficulty"].astype(str).str.strip().str.lower()

    panels: list[tuple[str, Optional[str]]] = [
        ("All", None),
        ("Easy", "easy"),
        ("Medium", "medium"),
        ("Hard", "hard"),
    ]

    # Color palette consistent with other figures
    pal_map = {"agent": sns.color_palette("deep")[2], "bypass": sns.color_palette("deep")[3]}
    order_global = ["agent", "bypass"]

    # Figure
    fig_w = 3.2 * len(panels)
    fig_h = 4.2
    fig, axes = plt.subplots(nrows=1, ncols=len(panels), figsize=(fig_w, fig_h), sharey=True)
    if len(panels) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (title, key) in enumerate(panels):
        ax = axes[i]
        if key is None or not has_difficulty:
            sub = work if key is None else None
        else:
            sub = work.loc[work["difficulty_norm"] == key]

        stats, n_sub = _panel_stats(sub if sub is not None else pd.DataFrame())

        if sub is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "No difficulty column", ha="center", va="center")
            ax.set_title(f"{title}")
            continue

        if n_sub == 0 or stats.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{title} (n=0)")
            continue

        cats_present = [c for c in order_global if c in stats["method_cat"].astype(str).tolist()]
        if not cats_present:
            cats_present = order_global

        sns.barplot(
            data=stats,
            x="method_cat",
            y="em_rate",
            order=cats_present,
            palette=[pal_map.get(c, "#888888") for c in cats_present],
            ax=ax,
        )

        # Annotations (percentage above bars)
        for j, c in enumerate(cats_present):
            row = stats.loc[stats["method_cat"].astype(str) == c]
            if not row.empty:
                rate = float(row["em_rate"].iloc[0])
                ax.text(j, min(0.98, rate + 0.02), f"{rate*100:.1f}%", ha="center", va="bottom", fontsize=9)

        # Axis formatting
        ax.set_xlabel("method")
        if i == 0:
            ax.set_ylabel("exact match rate")
        else:
            ax.set_ylabel("")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.grid(True, which="major", linewidth=1.0, alpha=0.25)
        ax.xaxis.grid(False)
        ax.set_xticklabels([c for c in cats_present], rotation=15)
        ax.set_title(f"{title} (n={n_sub})")

    slug = _slugify(name)
    mslug = _slugify(model_name)
    save_path = output_dir / f"{slug}_{mslug}_em_by_difficulty.png"
    fig.suptitle(f"{name} — {model_name} — EM rate by difficulty (agent vs bypass)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def render_em_by_difficulty_per_model(
    input_csv: Path | str,
    name: str,
    *,
    output_dir: Path = Path("results/figures"),
    show: bool = True,
    model_name: Optional[str] = None,
) -> list[Path]:
    """Render four EM figures (All/Easy/Medium/Hard) for a given model or all models.

    Returns a list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    if "eval_method" not in df.columns:
        raise ValueError("Column 'eval_method' not found in input CSV; required for comparisons.")
    if "exact_match" not in df.columns:
        raise ValueError("Column 'exact_match' not found in input CSV; required to compute EM rate.")
    if "model_name" not in df.columns:
        df["model_name"] = "unknown"

    df = df.copy()
    df["eval_method"] = df["eval_method"].astype(str)
    df["model_name"] = df["model_name"].astype(str)

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
        p = _render_grid_for_model(
            df_model,
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
        render_em_by_difficulty_per_model(
            flags.input_csv,
            flags.name,
            output_dir=flags.output_dir,
            show=flags.show,
            model_name=None,
        )
    else:
        # Specific model
        render_em_by_difficulty_per_model(
            flags.input_csv,
            flags.name,
            output_dir=flags.output_dir,
            show=flags.show,
            model_name=flags.model_name,
        )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


