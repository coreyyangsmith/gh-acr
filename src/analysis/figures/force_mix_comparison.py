"""Compare force_mix results (LLaMA-8B, Qwen-8B) with final-paper results.

Produces FOUR figures — one per difficulty level (All, Easy, Medium, Hard).
Each figure contains FOUR horizontal bar panels side-by-side, one per metric:
  Exact Match | Similarity | BLEU-3 | ROUGE-L

Y-axis rows = model × method combos, with short human-readable labels:
  e.g. "LLaMA [mix]", "Qwen-8B [mix]", "LLaMA agent", "Qwen-32B bypass" …

Each bar shows the mean value (exact-match rate or mean continuous score),
and each Y-tick label is annotated with the number of unique instances used.

Data sources:
  - data/2026_04_26_llama_force_mix.csv   → LLaMA-3.1-8B · force_mix
  - data/2026_04_26_qwen_force_mix.csv    → Qwen3-8B · force_mix
  - data/2026_01_results_final.csv        → Qwen3-32B / LLaMA-3.1-8B /
                                            GPT-5-nano · agent & bypass7

Usage::

    python -m src.analysis.figures.force_mix_comparison

Outputs four PNGs under results/figures/:
    force_mix_vs_final_all.png
    force_mix_vs_final_easy.png
    force_mix_vs_final_medium.png
    force_mix_vs_final_hard.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tyro

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_CSV = Path("data/2026_04_26_llama_force_mix.csv")
DEFAULT_QWEN_CSV  = Path("data/2026_04_26_qwen_force_mix.csv")
DEFAULT_FINAL_CSV = Path("data/2026_01_results_final.csv")
DEFAULT_OUTPUT_DIR = Path("results/figures")

FINAL_METHODS = {"agent", "bypass7"}

METRICS: list[tuple[str, str]] = [
    ("exact_match", "Exact Match"),
    ("similarity",  "Similarity"),
    ("bleu3",       "BLEU-3"),
    ("rouge_l",     "ROUGE-L"),
]

DIFFICULTIES = [
    (None,     "All"),
    ("easy",   "Easy"),
    ("medium", "Medium"),
    ("hard",   "Hard"),
]

# Short display labels for model × method combos
_MODEL_SHORT: dict[str, str] = {
    "local:meta-llama/Llama-3.1-8B-Instruct": "LLaMA",
    "local:Qwen/Qwen3-8B":                    "Qwen-8B",
    "groq:qwen/qwen3-32b":                    "Qwen-32B",
    "openai/gpt-5-nano":                      "GPT",
}

_METHOD_SHORT: dict[str, str] = {
    "force_mix": "[mix]",
    "agent":     "agent",
    "bypass7":   "bypass",
}

# Color per method so methods are visually consistent across models
_METHOD_COLOR: dict[str, str] = {
    "force_mix": "#e07b39",
    "agent":     "#4c72b0",
    "bypass7":   "#55a868",
}

# Desired row order (bottom → top when plotted with default barh orientation)
_ROW_ORDER: list[tuple[str, str]] = [
    ("openai/gpt-5-nano",                      "agent"),
    ("openai/gpt-5-nano",                      "bypass7"),
    ("groq:qwen/qwen3-32b",                    "agent"),
    ("groq:qwen/qwen3-32b",                    "bypass7"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", "agent"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", "bypass7"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", "force_mix"),
    ("local:Qwen/Qwen3-8B",                    "force_mix"),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _short_label(model: str, method: str) -> str:
    m = _MODEL_SHORT.get(model, model.split("/")[-1])
    t = _METHOD_SHORT.get(method, method)
    return f"{m} {t}"


def _coerce_em(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).astype(float)


def load_combined(
    llama_csv: Path,
    qwen_csv: Path,
    final_csv: Path,
) -> pd.DataFrame:
    df_llama = pd.read_csv(llama_csv)
    df_qwen  = pd.read_csv(qwen_csv)
    df_final = pd.read_csv(final_csv)

    for df in (df_llama, df_qwen, df_final):
        df.drop(df[df["eval_method"] == "prep"].index, inplace=True)

    df_final = df_final[df_final["eval_method"].isin(FINAL_METHODS)].copy()

    combined = pd.concat([df_llama, df_qwen, df_final], ignore_index=True)

    if "exact_match" in combined.columns:
        combined["exact_match"] = _coerce_em(combined["exact_match"])

    combined["_label"] = [
        _short_label(str(m), str(e))
        for m, e in zip(combined["model_name"], combined["eval_method"])
    ]
    combined["_method"] = combined["eval_method"].astype(str)
    return combined


def _subset(df: pd.DataFrame, diff_key: Optional[str]) -> pd.DataFrame:
    if diff_key is None:
        return df
    return df[df["difficulty"].astype(str).str.lower() == diff_key]


def _agg_row(sub: pd.DataFrame, metric: str) -> tuple[float, int]:
    """Return (mean_value, n_unique_instances) for one model×method slice."""
    n = int(sub["id"].nunique()) if "id" in sub.columns else len(sub)
    if metric == "exact_match":
        vals = sub["exact_match"].dropna()
        mean = float(vals.mean()) if len(vals) > 0 else float("nan")
    else:
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        mean = float(vals.mean()) if len(vals) > 0 else float("nan")
    return mean, n


def build_table(
    df: pd.DataFrame,
    diff_key: Optional[str],
) -> pd.DataFrame:
    """Build a summary DataFrame with one row per model×method, columns = metrics + n."""
    sub_diff = _subset(df, diff_key)

    rows = []
    present = set(zip(df["model_name"].astype(str), df["_method"].astype(str)))

    for model, method in _ROW_ORDER:
        if (model, method) not in present:
            continue
        mask = (
            (sub_diff["model_name"].astype(str) == model)
            & (sub_diff["_method"].astype(str) == method)
        )
        slice_ = sub_diff[mask]
        if slice_.empty:
            continue
        label = _short_label(model, method)
        row: dict[str, object] = {"label": label, "method": method, "model": model}
        for col, _ in METRICS:
            val, n = _agg_row(slice_, col)
            row[col] = val
            row["n"] = n  # same n for all metrics (same slice)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_one_figure(
    table: pd.DataFrame,
    diff_title: str,
    output_path: Path,
) -> None:
    """Render one figure (4 horizontal bar panels) for one difficulty level."""
    if table.empty:
        return

    labels   = table["label"].tolist()          # Y-axis tick labels
    methods  = table["method"].tolist()
    ns       = table["n"].tolist()
    colors   = [_METHOD_COLOR.get(m, "#888888") for m in methods]

    n_rows  = len(labels)
    n_cols  = len(METRICS)
    fig_h   = max(2.5, 0.45 * n_rows + 1.2)
    fig_w   = 3.5 * n_cols

    fig, axes = plt.subplots(
        nrows=1, ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey=True,
    )

    y_pos = np.arange(n_rows)

    for col_idx, (metric_col, metric_label) in enumerate(METRICS):
        ax = axes[col_idx]
        vals = table[metric_col].tolist()

        bars = ax.barh(
            y_pos,
            vals,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            height=0.65,
        )

        # Value label at end of each bar
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                v + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#222",
            )

        ax.set_xlim(0, 1.08)
        ax.set_xlabel(metric_label, fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Y-axis ticks only on the leftmost panel
        if col_idx == 0:
            # Annotate tick labels with N count
            y_tick_labels = [
                f"{lbl}  (N={n:,})" for lbl, n in zip(labels, ns)
            ]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_tick_labels, fontsize=9)
        else:
            ax.tick_params(axis="y", left=False)

    # Legend for methods
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_METHOD_COLOR[m], label=_METHOD_SHORT.get(m, m))
        for m in ["agent", "bypass7", "force_mix"]
        if m in set(methods)
    ]
    fig.legend(
        handles=legend_handles,
        title="method",
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f"Force-Mix vs Final Results — {diff_title}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_force_mix_comparison(
    llama_csv: Path = DEFAULT_LLAMA_CSV,
    qwen_csv: Path  = DEFAULT_QWEN_CSV,
    final_csv: Path = DEFAULT_FINAL_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> list[Path]:
    """Generate four per-difficulty figures; return list of saved PNG paths."""
    df = load_combined(llama_csv, qwen_csv, final_csv)

    saved: list[Path] = []
    for diff_key, diff_title in DIFFICULTIES:
        table = build_table(df, diff_key)
        slug  = diff_title.lower()
        out   = output_dir / f"force_mix_vs_final_{slug}.png"
        _plot_one_figure(table, diff_title, out)
        saved.append(out)
        print(f"  Saved: {out}")

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class Flags:
    llama_csv: Path  = DEFAULT_LLAMA_CSV
    qwen_csv: Path   = DEFAULT_QWEN_CSV
    final_csv: Path  = DEFAULT_FINAL_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR
    show: bool       = False


def main(flags: Flags) -> None:
    paths = generate_force_mix_comparison(
        llama_csv=flags.llama_csv,
        qwen_csv=flags.qwen_csv,
        final_csv=flags.final_csv,
        output_dir=flags.output_dir,
        show=flags.show,
    )
    print(f"\nDone. {len(paths)} figure(s) saved.")


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)
