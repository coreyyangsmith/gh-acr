"""End-to-end rendering of comparison tables and plots for evaluation results.

This module provides a CLI and helpers to produce:
- per-method summary tables and method×model tables
- boxplots and grouped bar charts for metrics (similarity, BLEU-3, ROUGE-L,
  processing time, tokens, and costs)
- convenience exports of CSV/Markdown tables used in the paper/README

The public entrypoint is `main(Flags)` and the script can be invoked via
`python -m src.results.compare_methods --help`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import tyro
import re

from .data_loader import load_results
from .tables import method_summary
from src.config.eval_methods import DEFAULT_METHOD_ORDER


# =============================================================================
# Global plotting theme (clean, readable, consistent)
# =============================================================================

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
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    },
)


# =============================================================================
# CLI flags
# =============================================================================

@dataclass
class Flags:
    """CLI flags controlling which artifacts to render and where to save them."""
    results_csv: Optional[Path] = None
    output_dir: Path = Path("results")
    show: bool = True
    # Optional: filter results to rows whose 'file_name' contains this substring (case-insensitive)
    file: Optional[str] = None
    # Optional: filter results by difficulty label(s). Accepts comma-separated values (e.g., "easy,medium").
    difficulty: Optional[str] = None


# =============================================================================
# Small utilities
# =============================================================================

def _order_from_present(present: Iterable[str], desired: Optional[list[str]] = None) -> list[str]:
    """Return labels in the desired order followed by any unknown ones.

    Ensures consistent method ordering across all visualizations.
    """
    present_list = [str(x) for x in present]
    desired = desired or DEFAULT_METHOD_ORDER
    return [m for m in desired if m in present_list] + [m for m in present_list if m not in desired]


def _palette_for(labels: list[str], base: str = "deep") -> dict[str, tuple[float, float, float]]:
    """Stable mapping from label → color for consistent plots across figures."""
    pal = sns.color_palette(base, max(3, len(labels)))
    return {lab: pal[i % len(pal)] for i, lab in enumerate(labels)}


def _slugify_model_name(name: object) -> str:
    """Filesystem-safe key derived from a model name."""
    s = str(name)
    try:
        s = s.strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")
        s = re.sub(r"[^a-z0-9_.-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unknown_model"
    except Exception:
        return "unknown_model"


def _save_markdown_table(df: pd.DataFrame, *, save_md_path: Path) -> None:
    """Write a GitHub-flavored Markdown table for the provided DataFrame."""
    if df is None or df.empty:
        return
    cols = [str(c) for c in df.columns]

    def _fmt_cell(v: object) -> str:
        if pd.isna(v):
            return ""
        return str(v)

    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = [_fmt_cell(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")

    save_md_path.write_text("\n".join(lines), encoding="utf-8")


def _save_box_summary_table(
    df: pd.DataFrame,
    *,
    column: str,
    save_csv_path: Path,
    method_order: Optional[list[str]] = None,
) -> None:
    """Write a per-method box summary to CSV.

    Columns: count, min, q1, median, q3, max, mean, std.
    Only methods present in the input are included, ordered by `method_order`.
    """
    if "eval_method" not in df.columns or column not in df.columns:
        return
    work = df[["eval_method", column]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])
    if work.empty:
        return

    order = _order_from_present(work["eval_method"].astype(str).unique().tolist(), method_order)

    rows: list[dict[str, object]] = []
    for method in order:
        s = work.loc[work["eval_method"].astype(str) == method, column].astype(float).dropna()
        if s.empty:
            continue
        q1, med, q3 = np.percentile(s.to_numpy(), [25, 50, 75])
        rows.append(
            {
                "method": method,
                "count": int(s.size),
                "min": float(s.min()),
                "q1": float(q1),
                "median": float(med),
                "q3": float(q3),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1) if s.size > 1 else 0.0),
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(save_csv_path, index=False)


# =============================================================================
# Method x Model summary table (EM + medians)
# =============================================================================

def _method_model_summary(
    df: pd.DataFrame, *, method_order: Optional[list[str]] = None

) -> pd.DataFrame:
    """Return summary grouped by (eval_method, model_name).

    Columns produced if source columns exist:
      - EM: mean of exact_match (converted to {0,1})
      - similarity_median: median of similarity
      - BLEU_median: median of bleu3
      - ROUGE_median: median of rouge_l
    """
    if "eval_method" not in df.columns:
        return pd.DataFrame()

    work = pd.DataFrame()
    work["eval_method"] = df["eval_method"].astype(str)
    # Ensure model_name exists
    work["model_name"] = (
        df.get("model_name", pd.Series(["unknown"] * len(df)))
        .astype(str)
        .fillna("unknown")
    )

    # EM (exact_match → {0,1})
    if "exact_match" in df.columns:
        em_series = df["exact_match"]
        # Handle different encodings of truthy values
        if pd.api.types.is_bool_dtype(em_series):
            em_numeric = em_series.astype(int)
        else:
            em_numeric = (
                em_series.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)
            )
        work["EM"] = em_numeric

    # Numeric conversions for medians (ignore non-numeric via NaN)
    if "similarity" in df.columns:
        work["similarity_median_src"] = pd.to_numeric(df["similarity"], errors="coerce")
    if "bleu3" in df.columns:
        work["BLEU_median_src"] = pd.to_numeric(df["bleu3"], errors="coerce")
    if "rouge_l" in df.columns:
        work["ROUGE_median_src"] = pd.to_numeric(df["rouge_l"], errors="coerce")

    if work.empty:
        return pd.DataFrame()

    agg_map: dict[str, str] = {}
    if "EM" in work.columns:
        agg_map["EM"] = "mean"
    if "similarity_median_src" in work.columns:
        agg_map["similarity_median_src"] = "median"
    if "BLEU_median_src" in work.columns:
        agg_map["BLEU_median_src"] = "median"
    if "ROUGE_median_src" in work.columns:
        agg_map["ROUGE_median_src"] = "median"

    if not agg_map:
        return pd.DataFrame()

    grouped = work.groupby(["eval_method", "model_name"], dropna=False).agg(agg_map).reset_index()
    # Rename aggregated columns to desired output names
    rename_map = {
        "similarity_median_src": "similarity_median",
        "BLEU_median_src": "BLEU_median",
        "ROUGE_median_src": "ROUGE_median",
    }
    grouped = grouped.rename(columns=rename_map)

    # Order rows: methods by DEFAULT_METHOD_ORDER, then model_name alphabetically
    present_methods = grouped["eval_method"].astype(str).unique().tolist()
    ordered_methods = _order_from_present(present_methods, method_order)
    grouped["_method_order_key"] = grouped["eval_method"].astype(str).map({m: i for i, m in enumerate(ordered_methods)})
    grouped = grouped.sort_values(by=["_method_order_key", "model_name"]).drop(columns=["_method_order_key"])\
                     .reset_index(drop=True)

    # Round numeric columns to 3 decimals for readability
    numeric_cols = [c for c in ["EM", "similarity_median", "BLEU_median", "ROUGE_median"] if c in grouped.columns]
    if numeric_cols:
        grouped[numeric_cols] = grouped[numeric_cols].astype(float).round(3)

    return grouped

# =============================================================================
# Box plots (improved styling, simpler code)
# =============================================================================

def _boxplot_metric(
    df: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    save_path: Path,
    show: bool,
    method_order: Optional[list[str]] = None,
    ylim: Optional[tuple[float, float]] = None,
    label_fmt: Optional[Callable[[float], str]] = None,
    show_mean_marker: bool = True,
    show_counts: bool = True,
) -> None:
    """
    Clean, publication-ready boxplot with:
      - consistent palettes per method
      - hidden fliers (robust look)
      - mean shown as a white diamond (optional)
      - optional per-method sample size under the x labels
      - thicker median line; subtle grid

    Uses hue='eval_method' with dodge=False to avoid seaborn's palette-without-hue deprecation.
    """
    if "eval_method" not in df.columns or column not in df.columns:
        return

    work = df[["eval_method", column]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])
    if work.empty:
        return

    # Order + palette
    present = work["eval_method"].astype(str).unique().tolist()
    order = _order_from_present(present, method_order)
    pal_map = _palette_for(order, base=("tab20" if len(order) > 10 else "deep"))

    # Figure
    fig_w = max(10, len(order) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    sns.boxplot(
        data=work,
        x="eval_method",
        y=column,
        hue="eval_method",         # same as x to enable palette without deprecation
        order=order,
        hue_order=order,
        palette=pal_map,           # dict mapping level -> color
        dodge=False,               # draw single box per category
        legend=False,              # no redundant legend
        showfliers=False,
        width=0.65,
        linewidth=1.5,
        boxprops=dict(edgecolor="black", zorder=2),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.25),
        capprops=dict(color="black", linewidth=1.25),
        ax=ax,
    )

    # Y formatting
    if ylim is not None:
        ax.set_ylim(*ylim)
        span = (ylim[1] - ylim[0]) if (ylim[1] - ylim[0]) > 0 else 1.0
        ax.yaxis.set_major_locator(MultipleLocator(span / 5.0))
    ax.yaxis.grid(True, which="major", linewidth=1.2, alpha=0.25)
    ax.xaxis.grid(False)

    # Labels
    ax.set_xlabel("method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=20)

    # Mean diamonds + labels
    if show_mean_marker:
        fmt = (lambda v: f"{v:.3f}") if label_fmt is None else label_fmt
        for i, m in enumerate(order):
            vals = (
                work.loc[work["eval_method"].astype(str) == m, column]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            if vals.size == 0:
                continue
            mu = float(np.mean(vals))
            ax.scatter(i, mu, marker="D", s=40, zorder=3, edgecolor="black", facecolor="white")
            if len(order) <= 8:
                # use annotate (not text) to support xy offsets without error
                ax.annotate(
                    fmt(mu),
                    (i, mu),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    clip_on=False,
                )

    # Sample counts under tick labels
    if show_counts:
        counts = work.groupby("eval_method")[column].apply(lambda s: s.dropna().shape[0]).reindex(order)
        xticks = ax.get_xticks()
        for x, m in zip(xticks, order):
            n = int(counts.get(m, 0))
            # place just below the tick labels
            ax.text(
                x,
                -0.08,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=8,
                color="#555",
                transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
            )

    # ------------------------------------------------------------------
    # Statistical callouts: median, Q1/Q3, and 95% CI of the mean
    # ------------------------------------------------------------------
    try:
        # Light clutter control: only annotate if not too many methods
        if len(order) <= 12:
            fmt = (lambda v: f"{v:.3f}") if label_fmt is None else label_fmt
            for i, m in enumerate(order):
                series = (
                    work.loc[work["eval_method"].astype(str) == m, column]
                    .astype(float)
                    .dropna()
                )
                if series.empty:
                    continue
                values = series.to_numpy()
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                n = values.size
                # 95% CI of the mean using normal approx
                if n > 1:
                    sem = std / np.sqrt(n)
                    ci_low = mean - 1.96 * sem
                    ci_high = mean + 1.96 * sem
                else:
                    ci_low = mean
                    ci_high = mean

                # Place small text labels slightly to the right of each box position
                x_text = i + 0.18
                # Median
                ax.annotate(
                    f"median {fmt(float(med))}",
                    (i, med),
                    xytext=(12, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.7),
                )
                # Q1/Q3 stacked labels near respective quartiles
                ax.annotate(
                    f"q1 {fmt(float(q1))}",
                    (i, q1),
                    xytext=(12, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#444",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.7),
                )
                ax.annotate(
                    f"q3 {fmt(float(q3))}",
                    (i, q3),
                    xytext=(12, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#444",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.7),
                )

                # 95% CI whisker line at x offset and a label in the middle
                x_ci = i + 0.26
                ax.plot([x_ci, x_ci], [ci_low, ci_high], color="#333", linewidth=1.2, zorder=3)
                ax.plot([x_ci - 0.03, x_ci + 0.03], [ci_low, ci_low], color="#333", linewidth=1.2, zorder=3)
                ax.plot([x_ci - 0.03, x_ci + 0.03], [ci_high, ci_high], color="#333", linewidth=1.2, zorder=3)
                ax.annotate(
                    f"95% CI\n{fmt(float(ci_low))}–{fmt(float(ci_high))}",
                    (x_ci, (ci_low + ci_high) / 2.0),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.7),
                )
    except Exception:
        # Fail-safe: never break rendering if annotation fails
        pass

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Bars by model (single simple function used everywhere)
# =============================================================================

def _bars_by_model(
    df: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    save_path: Path,
    show: bool,
    method_order: Optional[list[str]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """
    Grouped bars by model with 95% CI. Works for metrics or exact_match (bool-like).
    Expects columns: eval_method, model_name (optional), <column>.
    """
    if "eval_method" not in df.columns or column not in df.columns:
        return

    work = df.copy()
    # Ensure model_name exists
    work["model_name"] = work.get("model_name", pd.Series(["unknown"] * len(work))).astype(str).fillna("unknown")
    # Coerce metric
    if column == "exact_match":
        work[column] = work[column].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])
    if work.empty:
        return

    # Order + palette
    methods_present = work["eval_method"].astype(str).unique().tolist()
    order = _order_from_present(methods_present, method_order)
    models = work["model_name"].astype(str).unique().tolist()
    hue_pal = _palette_for(models, base=("tab20" if len(models) > 10 else "deep"))

    fig_w = max(10, len(order) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    sns.barplot(
        data=work,
        x="eval_method",
        y=column,
        hue="model_name",
        order=order,
        hue_order=models,
        estimator=np.mean,
        errorbar=("ci", 95),
        capsize=0.05,
        palette=[hue_pal[m] for m in models],
        ax=ax,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.xticks(rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Render all outputs
# =============================================================================

def _render_all_outputs(df: pd.DataFrame, *, output_dir: Path, show: bool) -> None:
    """Render all tables and figures into `output_dir`.

    This orchestrates the common set of outputs used throughout analyses and the
    README, guarding each artifact on the presence of its required columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base summary table (per method) and a simple markdown export for quick viewing
    summary = method_summary(df)
    _save_markdown_table(summary, save_md_path=output_dir / "table_all.md")

    # Method x Model summary (EM mean + medians for similarity/BLEU/ROUGE)
    mm_summary = _method_model_summary(df, method_order=DEFAULT_METHOD_ORDER)
    if mm_summary is not None and not mm_summary.empty:
        mm_csv = output_dir / "table_method_model.csv"
        mm_md = output_dir / "table_method_model.md"
        mm_summary.to_csv(mm_csv, index=False)
        _save_markdown_table(mm_summary, save_md_path=mm_md)

    # -------------------------
    # Exact Match — Bars
    # -------------------------
    _bars_by_model(
        df,
        column="exact_match",
        ylabel="exact match rate",
        title="Exact Match Rate by Method (bars by model)",
        save_path=output_dir / "compare_exact_match_rate_bars.png",
        show=show,
        method_order=DEFAULT_METHOD_ORDER,
        ylim=(0.0, 1.0),
    )

    # -------------------------
    # Similarity — Box + Bars
    # -------------------------
    if "similarity" in df.columns and df["similarity"].notna().any():
        _boxplot_metric(
            df,
            column="similarity",
            ylabel="similarity",
            title="Similarity by Method",
            save_path=output_dir / "compare_similarity_box.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
            ylim=(0.0, 1.0),
            show_mean_marker=True,
        )
        _bars_by_model(
            df,
            column="similarity",
            ylabel="similarity",
            title="Similarity by Method (bars by model)",
            save_path=output_dir / "compare_similarity_bars.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
        )
        _save_box_summary_table(
            df, column="similarity", save_csv_path=output_dir / "table_similarity.csv", method_order=DEFAULT_METHOD_ORDER
        )

    # -------------------------
    # BLEU-3 — Box + Bars
    # -------------------------
    if "bleu3" in df.columns and df["bleu3"].notna().any():
        _boxplot_metric(
            df,
            column="bleu3",
            ylabel="BLEU-3",
            title="BLEU-3 by Method",
            save_path=output_dir / "compare_bleu3_box.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
            show_mean_marker=True,
        )
        _bars_by_model(
            df,
            column="bleu3",
            ylabel="BLEU-3",
            title="BLEU-3 by Method (bars by model)",
            save_path=output_dir / "compare_bleu3_bars.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
        )
        _save_box_summary_table(
            df, column="bleu3", save_csv_path=output_dir / "table_bleu3.csv", method_order=DEFAULT_METHOD_ORDER
        )

    # -------------------------
    # ROUGE-L — Box + Bars
    # -------------------------
    if "rouge_l" in df.columns and df["rouge_l"].notna().any():
        _boxplot_metric(
            df,
            column="rouge_l",
            ylabel="ROUGE-L",
            title="ROUGE-L by Method",
            save_path=output_dir / "compare_rouge_l_box.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
            show_mean_marker=True,
        )
        _bars_by_model(
            df,
            column="rouge_l",
            ylabel="ROUGE-L",
            title="ROUGE-L by Method (bars by model)",
            save_path=output_dir / "compare_rouge_l_bars.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
        )
        _save_box_summary_table(
            df, column="rouge_l", save_csv_path=output_dir / "table_rouge_l.csv", method_order=DEFAULT_METHOD_ORDER
        )

    # -------------------------
    # Processing time — Box + Bars
    # -------------------------
    if "processing_time_s" in df.columns and df["processing_time_s"].notna().any():
        _boxplot_metric(
            df,
            column="processing_time_s",
            ylabel="seconds",
            title="Processing Time by Method",
            save_path=output_dir / "compare_processing_time_box.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
            show_mean_marker=True,
        )
        _bars_by_model(
            df,
            column="processing_time_s",
            ylabel="seconds",
            title="Processing Time by Method (bars by model)",
            save_path=output_dir / "compare_processing_time_bars.png",
            show=show,
            method_order=DEFAULT_METHOD_ORDER,
        )
        _save_box_summary_table(
            df, column="processing_time_s", save_csv_path=output_dir / "table_processing_time_s.csv", method_order=DEFAULT_METHOD_ORDER
        )

    # -------------------------
    # Tokens (in / out / total) — Box + Bars
    # -------------------------
    have_tokens_in = "tokens_in" in df.columns and df["tokens_in"].notna().any()
    have_tokens_out = "tokens_out" in df.columns and df["tokens_out"].notna().any()
    if have_tokens_in or have_tokens_out:
        tokens_df = df.copy()
        if have_tokens_in and have_tokens_out:
            tokens_df["tokens_total"] = tokens_df["tokens_in"].astype(float) + tokens_df["tokens_out"].astype(float)
        elif have_tokens_in:
            tokens_df["tokens_total"] = tokens_df["tokens_in"].astype(float)
        elif have_tokens_out:
            tokens_df["tokens_total"] = tokens_df["tokens_out"].astype(float)

        if have_tokens_in:
            _boxplot_metric(
                tokens_df,
                column="tokens_in",
                ylabel="tokens",
                title="Input Tokens by Method",
                save_path=output_dir / "compare_tokens_in_box.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _bars_by_model(
                tokens_df,
                column="tokens_in",
                ylabel="tokens",
                title="Input Tokens by Method (bars by model)",
                save_path=output_dir / "compare_tokens_in_bars.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _save_box_summary_table(
                tokens_df, column="tokens_in", save_csv_path=output_dir / "table_tokens_in.csv", method_order=DEFAULT_METHOD_ORDER
            )

        if have_tokens_out:
            _boxplot_metric(
                tokens_df,
                column="tokens_out",
                ylabel="tokens",
                title="Output Tokens by Method",
                save_path=output_dir / "compare_tokens_out_box.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _bars_by_model(
                tokens_df,
                column="tokens_out",
                ylabel="tokens",
                title="Output Tokens by Method (bars by model)",
                save_path=output_dir / "compare_tokens_out_bars.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _save_box_summary_table(
                tokens_df, column="tokens_out", save_csv_path=output_dir / "table_tokens_out.csv", method_order=DEFAULT_METHOD_ORDER
            )

        if "tokens_total" in tokens_df.columns and tokens_df["tokens_total"].notna().any():
            _boxplot_metric(
                tokens_df,
                column="tokens_total",
                ylabel="tokens",
                title="Total Tokens by Method",
                save_path=output_dir / "compare_tokens_total_box.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _bars_by_model(
                tokens_df,
                column="tokens_total",
                ylabel="tokens",
                title="Total Tokens by Method (bars by model)",
                save_path=output_dir / "compare_tokens_total_bars.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _save_box_summary_table(
                tokens_df, column="tokens_total", save_csv_path=output_dir / "table_tokens_total.csv", method_order=DEFAULT_METHOD_ORDER
            )

    # -------------------------
    # Cost (in / out / total) — Box + Bars
    # -------------------------
    have_cost_in = "cost_in" in df.columns and df["cost_in"].notna().any()
    have_cost_out = "cost_out" in df.columns and df["cost_out"].notna().any()
    have_total_cost = "total_cost" in df.columns and df["total_cost"].notna().any()
    if have_cost_in or have_cost_out or have_total_cost:
        cost_df = df.copy()
        if not have_total_cost:
            if have_cost_in and have_cost_out:
                cost_df["total_cost"] = cost_df["cost_in"].astype(float) + cost_df["cost_out"].astype(float)
            elif have_cost_in:
                cost_df["total_cost"] = cost_df["cost_in"].astype(float)
            elif have_cost_out:
                cost_df["total_cost"] = cost_df["cost_out"].astype(float)

        if have_cost_in:
            _boxplot_metric(
                cost_df,
                column="cost_in",
                ylabel="$",
                title="Input Cost by Method",
                save_path=output_dir / "compare_cost_in_box.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _bars_by_model(
                cost_df,
                column="cost_in",
                ylabel="$",
                title="Input Cost by Method (bars by model)",
                save_path=output_dir / "compare_cost_in_bars.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _save_box_summary_table(
                cost_df, column="cost_in", save_csv_path=output_dir / "table_cost_in.csv", method_order=DEFAULT_METHOD_ORDER
            )

        if have_cost_out:
            _boxplot_metric(
                cost_df,
                column="cost_out",
                ylabel="$",
                title="Output Cost by Method",
                save_path=output_dir / "compare_cost_out_box.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _bars_by_model(
                cost_df,
                column="cost_out",
                ylabel="$",
                title="Output Cost by Method (bars by model)",
                save_path=output_dir / "compare_cost_out_bars.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _save_box_summary_table(
                cost_df, column="cost_out", save_csv_path=output_dir / "table_cost_out.csv", method_order=DEFAULT_METHOD_ORDER
            )

        if "total_cost" in cost_df.columns and cost_df["total_cost"].notna().any():
            _boxplot_metric(
                cost_df,
                column="total_cost",
                ylabel="$",
                title="Total Cost by Method",
                save_path=output_dir / "compare_total_cost_box.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _bars_by_model(
                cost_df,
                column="total_cost",
                ylabel="$",
                title="Total Cost by Method (bars by model)",
                save_path=output_dir / "compare_total_cost_bars.png",
                show=show,
                method_order=DEFAULT_METHOD_ORDER,
            )
            _save_box_summary_table(
                cost_df, column="total_cost", save_csv_path=output_dir / "table_total_cost.csv", method_order=DEFAULT_METHOD_ORDER
            )


# =============================================================================
# Entrypoint
# =============================================================================

def main(flags: Flags) -> None:
    """Entry point for CLI: load results, optionally filter, and render outputs."""
    flags.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(flags.results_csv)
    df = data.dataframe
    # Optional filter by file name substring
    if flags.file:
        if "file_name" not in df.columns:
            raise ValueError("File filter provided but 'file_name' column not found in results.")
        needle = str(flags.file).strip().lower()
        if needle:
            df = df[df["file_name"].astype(str).str.lower().str.contains(needle, na=False)]
    # Optional filter by difficulty (comma-separated allowed)
    if flags.difficulty:
        if "difficulty" not in df.columns:
            raise ValueError("Difficulty filter provided but 'difficulty' column not found in results.")
        desired = [s.strip().lower() for s in str(flags.difficulty).split(",") if s.strip()]
        if desired:
            df = df[df["difficulty"].astype(str).str.lower().isin(desired)]
    _render_all_outputs(df, output_dir=flags.output_dir, show=flags.show)


if __name__ == "__main__":
    parsed_flags = tyro.cli(Flags)
    main(parsed_flags)
