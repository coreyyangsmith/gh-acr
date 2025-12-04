from __future__ import annotations

"""Bypass method split for bypass7 (counts of A/B/MIX by difficulty).

Renders a 1×4 grid with columns: All, Easy, Medium, Hard. Each panel shows
bar counts of `bypass_method` categories (A, B, MIX) considering only rows with
`eval_method == 'bypass7'`.

CLI:
    # Single plot (all models combined)
    python -m src.results.figures.bypass_method --input-csv data.csv --name run1

    # Single plot filtered to a specific model
    python -m src.results.figures.bypass_method --input-csv data.csv --name run1 \
        --model-name "llama-3.1-8b"

    # Multiple plots: one per model_name
    python -m src.results.figures.bypass_method --input-csv data.csv --name run1 \
        --split-by-model True

Saves: results/figures/{name}_bypass_method.png
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
# Theme to match other figures
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


def _slugify(s: object) -> str:
    try:
        s2 = str(s).strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")
        s2 = re.sub(r"[^a-z0-9_.-]+", "_", s2)
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2 or "bypass_method"
    except Exception:
        return "bypass_method"


def _counts_for_subset(df: pd.DataFrame) -> tuple[dict[str, int], int]:
    """Return (counts_by_category, n_subset) for A/B/MIX in the provided subset."""
    cats = ["A", "B", "MIX"]
    if df.empty:
        return {c: 0 for c in cats}, 0
    vals = df["bypass_method"].astype(str).str.strip().str.upper()
    counts = {c: int((vals == c).sum()) for c in cats}
    return counts, int(len(df))


def render_bypass_method_split(
    input_csv: Path | str,
    name: str,
    *,
    output_dir: Path = Path("results/figures"),
    show: bool = True,
    model_name: Optional[str] = None,
) -> Path:
    """Render a 1×4 grid of A/B/MIX counts for `eval_method == 'bypass7'`.

    Columns: All, Easy, Medium, Hard
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    if "eval_method" not in df.columns:
        raise ValueError("Column 'eval_method' not found in input CSV.")
    if "bypass_method" not in df.columns:
        raise ValueError("Column 'bypass_method' not found in input CSV.")

    work = df.copy()
    work["eval_method_norm"] = work["eval_method"].astype(str).str.strip().str.lower()
    has_difficulty = "difficulty" in work.columns
    if has_difficulty:
        work["difficulty_norm"] = work["difficulty"].astype(str).str.strip().str.lower()

    # Filter to bypass7 rows
    work = work.loc[work["eval_method_norm"] == "bypass7"].copy()

    # Optional filter by model_name
    model_slug_for_title = None
    model_slug_for_file = None
    if model_name is not None:
        if "model_name" not in work.columns:
            raise ValueError("Column 'model_name' not found in input CSV but 'model_name' filter was provided.")
        work["model_name_norm"] = work["model_name"].astype(str).str.strip().str.lower()
        target_model_norm = str(model_name).strip().lower()
        work = work.loc[work["model_name_norm"] == target_model_norm].copy()
        model_slug_for_title = str(model_name).strip()
        model_slug_for_file = _slugify(model_name)

    diff_keys: list[tuple[str, Optional[str]]] = [
        ("All", None),
        ("Easy", "easy"),
        ("Medium", "medium"),
        ("Hard", "hard"),
    ]

    # Precompute counts for present panels to harmonize y-axis
    panel_data: list[tuple[str, Optional[str], dict[str, int], int, bool]] = []
    # tuple: (title, diff_key, counts, n_subset, is_present)
    max_count = 0
    for title, key in diff_keys:
        if key is None:
            sub = work
            counts, n_sub = _counts_for_subset(sub)
            panel_data.append((title, key, counts, n_sub, True))
            max_count = max(max_count, max(counts.values()) if counts else 0)
        else:
            if not has_difficulty:
                panel_data.append((title, key, {}, 0, False))
            else:
                sub = work.loc[work["difficulty_norm"] == key]
                counts, n_sub = _counts_for_subset(sub)
                panel_data.append((title, key, counts, n_sub, True))
                max_count = max(max_count, max(counts.values()) if counts else 0)

    # Figure
    cat_order = ["A", "B", "MIX"]
    palette = {"A": sns.color_palette("deep")[0], "B": sns.color_palette("deep")[1], "MIX": sns.color_palette("deep")[2]}
    fig_w = 3.2 * len(diff_keys)
    fig_h = 3.8
    fig, axes = plt.subplots(nrows=1, ncols=len(diff_keys), figsize=(fig_w, fig_h), sharey=True)
    if len(diff_keys) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (title, key, counts, n_sub, present) in enumerate(panel_data):
        ax = axes[i]

        if not present:
            ax.axis("off")
            ax.text(0.5, 0.5, "No difficulty column", ha="center", va="center")
            ax.set_title(f"{title}")
            continue

        if n_sub == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{title} (n=0)")
            continue

        panel_df = pd.DataFrame({
            "bypass_method_cat": cat_order,
            "count": [counts.get(c, 0) for c in cat_order],
        })

        sns.barplot(
            data=panel_df,
            x="bypass_method_cat",
            y="count",
            order=cat_order,
            palette=[palette[c] for c in cat_order],
            ax=ax,
        )

        # Annotations (counts above bars)
        for j, c in enumerate(cat_order):
            v = int(counts.get(c, 0))
            ax.text(j, v + max(1, int(0.02 * max(1, max_count))), str(v), ha="center", va="bottom", fontsize=9)

        # Axis formatting
        ax.set_xlabel("bypass_method")
        if i == 0:
            ax.set_ylabel("count")
        else:
            ax.set_ylabel("")
        ax.set_title(f"{title} (n={n_sub})")
        ax.yaxis.grid(True, which="major", linewidth=1.0, alpha=0.25)
        ax.xaxis.grid(False)
        ax.set_ylim(0, max_count * 1.15 + 0.5)

    slug = _slugify(name)
    suffix = f"_{model_slug_for_file}" if model_slug_for_file else ""
    save_path = output_dir / f"{slug}{suffix}_bypass_method.png"
    title_suffix = f" — model={model_slug_for_title}" if model_slug_for_title else ""
    fig.suptitle(f"{name}{title_suffix} — bypass_method split for eval_method=bypass7", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


@dataclass
class Flags:
    input_csv: Path
    name: str
    output_dir: Path = Path("results/figures")
    show: bool = True
    split_by_model: bool = True
    model_name: Optional[str] = None


def main(flags: Flags) -> None:
    if flags.split_by_model:
        df = pd.read_csv(flags.input_csv)
        if "model_name" not in df.columns:
            raise ValueError("Column 'model_name' not found in input CSV but '--split-by-model' was provided.")

        work = df.copy()
        work["eval_method_norm"] = work["eval_method"].astype(str).str.strip().str.lower()
        work = work.loc[work["eval_method_norm"] == "bypass7"].copy()

        # Use stripped originals for user-friendly titles; iterate in sorted order for determinism
        if work.empty:
            # No bypass7 rows; still run once to produce an empty plot labeled "All"
            render_bypass_method_split(
                flags.input_csv,
                flags.name,
                output_dir=flags.output_dir,
                show=flags.show,
            )
            return

        model_values = (
            work["model_name"].astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
        )
        for model in sorted(model_values):
            render_bypass_method_split(
                flags.input_csv,
                flags.name,
                output_dir=flags.output_dir,
                show=flags.show,
                model_name=model,
            )
    else:
        render_bypass_method_split(
            flags.input_csv,
            flags.name,
            output_dir=flags.output_dir,
            show=flags.show,
            model_name=flags.model_name,
        )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


