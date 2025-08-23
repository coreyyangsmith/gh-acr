from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tyro

from .data_loader import load_results
from .tables import method_summary
from src.config.eval_methods import DEFAULT_METHOD_ORDER


# Fixed method order and consistent coloring across all charts


def _build_palette_map(method_order: list[str] = DEFAULT_METHOD_ORDER) -> dict[str, tuple[float, float, float]]:
    palette = sns.color_palette("deep", len(method_order))
    return {m: palette[i] for i, m in enumerate(method_order)}


@dataclass
class Flags:
    results_csv: Optional[Path] = None
    output_dir: Path = Path("results")
    show: bool = True


def _plot_metric_bars(
    table: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    show: bool,
    ci_low_col: Optional[str] = None,
    ci_high_col: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    method_order: Optional[list[str]] = None,
    palette_map: Optional[dict[str, tuple[float, float, float]]] = None,
    label_fmt: Optional[Callable[[float], str]] = None,
) -> None:
    work_cols = ["method", value_col]
    if ci_low_col and ci_high_col:
        work_cols += [ci_low_col, ci_high_col]
    # Include percentile/median columns if present to enable candlestick overlay
    p25_col = value_col.replace("mean", "p25") if value_col.endswith("_mean") else None
    p75_col = value_col.replace("mean", "p75") if value_col.endswith("_mean") else None
    median_col = value_col.replace("mean", "median") if value_col.endswith("_mean") else None
    for c in [p25_col, p75_col, median_col]:
        if c and c in table.columns:
            work_cols.append(c)
    work = table[work_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
    if work.empty:
        return

    # Respect fixed method order; append any unknown methods after
    present_methods = work["method"].astype(str).tolist()
    desired_order = method_order or DEFAULT_METHOD_ORDER
    ordered_methods = [m for m in desired_order if m in present_methods] + [m for m in present_methods if m not in desired_order]

    # Build arrays in that order
    value_by_method = dict(zip(work["method"].astype(str), work[value_col].astype(float)))
    methods = ordered_methods
    values = np.array([value_by_method.get(m, np.nan) for m in methods], dtype=float)

    yerr = None
    if ci_low_col and ci_high_col and ci_low_col in work and ci_high_col in work:
        lows = work[ci_low_col].to_numpy(dtype=float)
        highs = work[ci_high_col].to_numpy(dtype=float)
        lower = np.where(np.isfinite(values) & np.isfinite(lows), values - lows, np.nan)
        upper = np.where(np.isfinite(values) & np.isfinite(highs), highs - values, np.nan)
        yerr = np.vstack([lower, upper]) if np.isfinite(lower).any() or np.isfinite(upper).any() else None

    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    if palette_map is None:
        palette_map = _build_palette_map()
    default_color = (0.7, 0.7, 0.7)
    colors = [palette_map.get(m, default_color) for m in methods]
    bars = plt.bar(x, values, yerr=yerr, capsize=4 if yerr is not None else 0, color=colors)
    plt.xticks(x, methods, rotation=20)
    plt.ylabel(ylabel)
    plt.xlabel("method")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)

    # Draw median line and interquartile candle (p25-p75) if present
    if p25_col and p75_col and median_col and all(c in work.columns for c in [p25_col, p75_col, median_col]):
        p25_by_method = dict(zip(work["method"].astype(str), work[p25_col].astype(float)))
        p75_by_method = dict(zip(work["method"].astype(str), work[p75_col].astype(float)))
        med_by_method = dict(zip(work["method"].astype(str), work[median_col].astype(float)))
        for i, m in enumerate(methods):
            p25 = p25_by_method.get(m, np.nan)
            p75 = p75_by_method.get(m, np.nan)
            med = med_by_method.get(m, np.nan)
            if not (np.isfinite(p25) and np.isfinite(p75)):
                continue
            # Candle: vertical line from p25 to p75 at bar center
            cx = bars[i].get_x() + bars[i].get_width() / 2.0
            plt.vlines(cx, p25, p75, colors="k", linewidth=3)
            # Median tick mark
            if np.isfinite(med):
                plt.hlines(med, cx - bars[i].get_width() * 0.2, cx + bars[i].get_width() * 0.2, colors="k", linewidth=2)

    # Annotate values on top of bars (consider error bar upper bound if present)
    upper_err = None
    if yerr is not None and isinstance(yerr, np.ndarray) and yerr.shape[0] == 2:
        upper_err = yerr[1]
    # Determine a small vertical offset for labels
    if ylim is not None:
        offset = 0.02 * (ylim[1] - ylim[0])
    else:
        vmax = float(np.nanmax(values)) if len(values) > 0 else 1.0
        offset = 0.02 * vmax if vmax > 0 else 0.02
    for i, bar in enumerate(bars):
        v = float(values[i]) if np.isfinite(values[i]) else np.nan
        if not np.isfinite(v):
            continue
        top = v + (float(upper_err[i]) if upper_err is not None and np.isfinite(upper_err[i]) else 0.0)
        label_text = label_fmt(v) if label_fmt is not None else f"{v:.3f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            top + offset,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _plot_grouped_bars_two_series(
    table: pd.DataFrame,
    *,
    col_in: str,
    col_out: str,
    label_in: str,
    label_out: str,
    ylabel: str,
    title: str,
    save_path: Path,
    show: bool,
    method_order: Optional[list[str]] = None,
    ylim: Optional[tuple[float, float]] = None,
    label_fmt: Optional[Callable[[float], str]] = None,
) -> None:
    # Expect table columns: method, col_in, col_out
    work = table[["method", col_in, col_out]].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    present_methods = work["method"].astype(str).tolist()
    desired_order = method_order or DEFAULT_METHOD_ORDER
    methods = [m for m in desired_order if m in present_methods] + [m for m in present_methods if m not in desired_order]
    work = work.set_index("method").reindex(methods)

    vals_in = work[col_in].astype(float).to_numpy()
    vals_out = work[col_out].astype(float).to_numpy()

    if np.all(~np.isfinite(vals_in)) and np.all(~np.isfinite(vals_out)):
        return

    n = len(methods)
    x = np.arange(n)
    width = 0.38
    plt.figure(figsize=(max(10, n * 1.2), 6))
    colors = sns.color_palette("deep", 2)
    bars_in = plt.bar(x - width / 2, vals_in, width, label=label_in, color=colors[0])
    bars_out = plt.bar(x + width / 2, vals_out, width, label=label_out, color=colors[1])

    plt.xticks(x, methods, rotation=20)
    plt.ylabel(ylabel)
    plt.xlabel("method")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.legend()

    # Label formatter and offset
    if ylim is not None:
        offset = 0.02 * (ylim[1] - ylim[0])
    else:
        vmax = float(np.nanmax([vals_in.max(initial=np.nan), vals_out.max(initial=np.nan)]))
        offset = 0.02 * vmax if np.isfinite(vmax) and vmax > 0 else 0.02

    def _format(v: float) -> str:
        if label_fmt is not None:
            return label_fmt(v)
        return f"{v:.3f}"

    for i, b in enumerate(bars_in):
        v = float(vals_in[i]) if np.isfinite(vals_in[i]) else np.nan
        if np.isfinite(v):
            plt.text(b.get_x() + b.get_width() / 2.0, v + offset, _format(v), ha="center", va="bottom", fontsize=9, clip_on=False)
    for i, b in enumerate(bars_out):
        v = float(vals_out[i]) if np.isfinite(vals_out[i]) else np.nan
        if np.isfinite(v):
            plt.text(b.get_x() + b.get_width() / 2.0, v + offset, _format(v), ha="center", va="bottom", fontsize=9, clip_on=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _boxplot_metric(
    df: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    save_path: Path,
    show: bool,
    method_order: Optional[list[str]] = None,
    palette_map: Optional[dict[str, tuple[float, float, float]]] = None,
    ylim: Optional[tuple[float, float]] = None,
    label_fmt: Optional[Callable[[float], str]] = None,
    show_mean_marker: bool = True,
) -> None:
    # Expect df columns: eval_method, column
    if "eval_method" not in df.columns or column not in df.columns:
        return
    work = df[["eval_method", column]].copy().replace([np.inf, -np.inf], np.nan)
    # Ensure numeric for robust plotting (e.g., exact_match booleans)
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    if work.empty:
        return
    desired_order = method_order or DEFAULT_METHOD_ORDER
    present = work["eval_method"].astype(str).unique().tolist()
    order = [m for m in desired_order if m in present] + [m for m in present if m not in desired_order]
    work["eval_method"] = pd.Categorical(work["eval_method"].astype(str), categories=order, ordered=True)

    plt.figure(figsize=(max(10, len(order) * 1.2), 6))
    if palette_map is None:
        palette_map = _build_palette_map(order)
    ax = plt.gca()
    sns.boxplot(
        data=work,
        x="eval_method",
        y=column,
        order=order,
        palette=palette_map,
        showfliers=False,
        width=0.6,
    )
    plt.xticks(rotation=20)
    plt.ylabel(ylabel)
    plt.xlabel("method")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    _annotate_boxplot_values(ax, work, order, column, label_fmt=label_fmt, show_mean=show_mean_marker)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _annotate_boxplot_values(
    ax: plt.Axes,
    work: pd.DataFrame,
    order: list[str],
    column: str,
    *,
    label_fmt: Optional[Callable[[float], str]] = None,
    show_mean: bool = True,
) -> None:
    """Annotate whiskers (lo/hi), quartiles (Q1/Q3), median, and optionally mean for each box."""
    fmt = (lambda v: f"{v:.3f}") if label_fmt is None else label_fmt
    x_dx = 0.32
    text_kw = dict(
        ha="left", va="center", fontsize=8, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

    try:
        for i, method in enumerate(order):
            series = (
                work.loc[work["eval_method"].astype(str) == method, column]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            if series.size == 0:
                continue

            q1, med, q3 = np.percentile(series, [25, 50, 75])
            iqr = q3 - q1
            low_bound = q1 - 1.5 * iqr
            high_bound = q3 + 1.5 * iqr
            lo_candidates = series[series >= low_bound]
            hi_candidates = series[series <= high_bound]
            wlo = float(np.min(lo_candidates)) if lo_candidates.size else np.nan
            whi = float(np.max(hi_candidates)) if hi_candidates.size else np.nan

            x_right = i + x_dx

            if np.isfinite(wlo):
                ax.annotate(f"lo {fmt(wlo)}", (x_right, wlo), textcoords="offset points", xytext=(2, 0), **text_kw)
            ax.annotate(f"Q1 {fmt(q1)}", (x_right, q1), textcoords="offset points", xytext=(2, 0), **text_kw)
            ax.annotate(f"med {fmt(med)}", (x_right, med), textcoords="offset points", xytext=(2, 0), fontweight="bold", **text_kw)
            ax.annotate(f"Q3 {fmt(q3)}", (x_right, q3), textcoords="offset points", xytext=(2, 0), **text_kw)
            if np.isfinite(whi):
                ax.annotate(f"hi {fmt(whi)}", (x_right, whi), textcoords="offset points", xytext=(2, 0), **text_kw)

            if show_mean:
                mu = float(np.mean(series))
                ax.scatter(i, mu, marker="D", s=30, zorder=3, edgecolor="black", facecolor="white")
                ax.annotate(
                    f"μ {fmt(mu)}",
                    (i, mu),
                    textcoords="offset points",
                    xytext=(6, 0),
                    ha="left",
                    va="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )
    except Exception:
        pass


def _save_box_summary_table(
    df: pd.DataFrame,
    *,
    column: str,
    save_csv_path: Path,
    method_order: Optional[list[str]] = None,
) -> None:
    """Write per-method box summary (min, Q1, median, Q3, max, mean, count) to CSV."""
    if "eval_method" not in df.columns or column not in df.columns:
        return
    work = df[["eval_method", column]].copy().replace([np.inf, -np.inf], np.nan)
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    if work.empty:
        return
    desired_order = method_order or DEFAULT_METHOD_ORDER
    present = work["eval_method"].astype(str).unique().tolist()
    order = [m for m in desired_order if m in present] + [m for m in present if m not in desired_order]

    rows: list[dict[str, object]] = []
    for method in order:
        series = work.loc[work["eval_method"].astype(str) == method, column].astype(float).dropna().to_numpy()
        if series.size == 0:
            continue
        q1, med, q3 = np.percentile(series, [25, 50, 75])
        rows.append(
            {
                "method": method,
                "count": int(series.size),
                "min": float(np.min(series)),
                "q1": float(q1),
                "median": float(med),
                "q3": float(q3),
                "max": float(np.max(series)),
                "mean": float(np.mean(series)),
            }
        )
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.to_csv(save_csv_path, index=False)


def main(flags: Flags) -> None:
    flags.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(flags.results_csv)
    df = data.dataframe

    sns.set_theme(style="whitegrid")

    summary = method_summary(df)
    palette_map = _build_palette_map(DEFAULT_METHOD_ORDER)

    # 1) Exact Match Rate (bar, not boxplot)
    if "exact_match_rate" in summary.columns:
        _plot_metric_bars(
            summary,
            value_col="exact_match_rate",
            ylabel="exact match rate",
            title="Exact Match Rate by Method",
            save_path=flags.output_dir / "compare_exact_match_rate_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )

    # 2) Similarity (boxplot)
    if "similarity" in df.columns and df["similarity"].notna().any():
        _boxplot_metric(
            df,
            column="similarity",
            ylabel="similarity",
            title="Similarity by Method",
            save_path=flags.output_dir / "compare_similarity_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )
        _save_box_summary_table(
            df,
            column="similarity",
            save_csv_path=flags.output_dir / "table_similarity.csv",
            method_order=DEFAULT_METHOD_ORDER,
        )

    # 3) BLEU-3 (boxplot)
    if "bleu3" in df.columns and df["bleu3"].notna().any():
        _boxplot_metric(
            df,
            column="bleu3",
            ylabel="BLEU-3",
            title="BLEU-3 by Method",
            save_path=flags.output_dir / "compare_bleu3_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )
        _save_box_summary_table(
            df,
            column="bleu3",
            save_csv_path=flags.output_dir / "table_bleu3.csv",
            method_order=DEFAULT_METHOD_ORDER,
        )

    # 4) ROUGE-L (boxplot)
    if "rouge_l" in df.columns and df["rouge_l"].notna().any():
        _boxplot_metric(
            df,
            column="rouge_l",
            ylabel="ROUGE-L",
            title="ROUGE-L by Method",
            save_path=flags.output_dir / "compare_rouge_l_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )
        _save_box_summary_table(
            df,
            column="rouge_l",
            save_csv_path=flags.output_dir / "table_rouge_l.csv",
            method_order=DEFAULT_METHOD_ORDER,
        )

    # Additional aggregated plots
    # Processing time (boxplot)
    if "processing_time_s" in df.columns and df["processing_time_s"].notna().any():
        _boxplot_metric(
            df,
            column="processing_time_s",
            ylabel="seconds",
            title="Processing Time by Method",
            save_path=flags.output_dir / "compare_processing_time_total_bars.png",
            show=flags.show,
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )
        # Duplicate to keep expected file present
        _boxplot_metric(
            df,
            column="processing_time_s",
            ylabel="seconds",
            title="Processing Time by Method",
            save_path=flags.output_dir / "compare_processing_time_avg_bars.png",
            show=flags.show,
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )
        _save_box_summary_table(
            df,
            column="processing_time_s",
            save_csv_path=flags.output_dir / "table_processing_time_s.csv",
            method_order=DEFAULT_METHOD_ORDER,
        )

    # Tokens (input, output, total) — totals and averages
    have_tokens_in = "tokens_in" in df.columns
    have_tokens_out = "tokens_out" in df.columns
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
                save_path=flags.output_dir / "compare_tokens_total_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _save_box_summary_table(
                tokens_df,
                column="tokens_in",
                save_csv_path=flags.output_dir / "table_tokens_in.csv",
                method_order=DEFAULT_METHOD_ORDER,
            )
        if have_tokens_out:
            _boxplot_metric(
                tokens_df,
                column="tokens_out",
                ylabel="tokens",
                title="Output Tokens by Method",
                save_path=flags.output_dir / "compare_tokens_avg_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _save_box_summary_table(
                tokens_df,
                column="tokens_out",
                save_csv_path=flags.output_dir / "table_tokens_out.csv",
                method_order=DEFAULT_METHOD_ORDER,
            )
        if "tokens_total" in tokens_df.columns:
            _boxplot_metric(
                tokens_df,
                column="tokens_total",
                ylabel="tokens",
                title="Total Tokens by Method",
                save_path=flags.output_dir / "compare_tokens_total_total_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _boxplot_metric(
                tokens_df,
                column="tokens_total",
                ylabel="tokens",
                title="Total Tokens by Method",
                save_path=flags.output_dir / "compare_tokens_total_avg_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _save_box_summary_table(
                tokens_df,
                column="tokens_total",
                save_csv_path=flags.output_dir / "table_tokens_total.csv",
                method_order=DEFAULT_METHOD_ORDER,
            )

    # Cost (input, output, total) — totals and averages
    have_cost_in = "cost_in" in df.columns
    have_cost_out = "cost_out" in df.columns
    have_total_cost = "total_cost" in df.columns
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
                save_path=flags.output_dir / "compare_cost_total_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _save_box_summary_table(
                cost_df,
                column="cost_in",
                save_csv_path=flags.output_dir / "table_cost_in.csv",
                method_order=DEFAULT_METHOD_ORDER,
            )
        if have_cost_out:
            _boxplot_metric(
                cost_df,
                column="cost_out",
                ylabel="$",
                title="Output Cost by Method",
                save_path=flags.output_dir / "compare_cost_avg_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _save_box_summary_table(
                cost_df,
                column="cost_out",
                save_csv_path=flags.output_dir / "table_cost_out.csv",
                method_order=DEFAULT_METHOD_ORDER,
            )
        if "total_cost" in cost_df.columns:
            _boxplot_metric(
                cost_df,
                column="total_cost",
                ylabel="$",
                title="Total Cost by Method",
                save_path=flags.output_dir / "compare_total_cost_total_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _boxplot_metric(
                cost_df,
                column="total_cost",
                ylabel="$",
                title="Total Cost by Method",
                save_path=flags.output_dir / "compare_total_cost_avg_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
            )
            _save_box_summary_table(
                cost_df,
                column="total_cost",
                save_csv_path=flags.output_dir / "table_total_cost.csv",
                method_order=DEFAULT_METHOD_ORDER,
            )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


