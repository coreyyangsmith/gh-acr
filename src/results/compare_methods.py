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


# Fixed method order and consistent coloring across all charts
DEFAULT_METHOD_ORDER: list[str] = ["base_a", "base_b", "agent", "multi", "bypass_multi"]


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


def main(flags: Flags) -> None:
    flags.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(flags.results_csv)
    df = data.dataframe

    sns.set_theme(style="whitegrid")

    summary = method_summary(df)
    palette_map = _build_palette_map(DEFAULT_METHOD_ORDER)

    # 1) Exact Match Rate (with 95% CI)
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

    # 2) Similarity (mean)
    if "similarity_mean" in summary.columns and summary["similarity_mean"].notna().any():
        _plot_metric_bars(
            summary,
            value_col="similarity_mean",
            ylabel="mean similarity",
            title="Similarity by Method",
            save_path=flags.output_dir / "compare_similarity_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )

    # 3) BLEU-3 (mean)
    if "bleu3_mean" in summary.columns and summary["bleu3_mean"].notna().any():
        _plot_metric_bars(
            summary,
            value_col="bleu3_mean",
            ylabel="mean BLEU-3",
            title="BLEU-3 by Method",
            save_path=flags.output_dir / "compare_bleu3_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )

    # 4) ROUGE-L (mean)
    if "rouge_l_mean" in summary.columns and summary["rouge_l_mean"].notna().any():
        _plot_metric_bars(
            summary,
            value_col="rouge_l_mean",
            ylabel="mean ROUGE-L",
            title="ROUGE-L by Method",
            save_path=flags.output_dir / "compare_rouge_l_bars.png",
            show=flags.show,
            ylim=(0.0, 1.0),
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )

    # Additional aggregated plots
    # Processing time (sum and mean)
    if "processing_time_s" in df.columns:
        agg_time = df.groupby("eval_method")["processing_time_s"].agg(total="sum", avg="mean").reset_index().rename(columns={"eval_method": "method"})
        # Total processing time
        _plot_metric_bars(
            agg_time.rename(columns={"total": "processing_time_total_s"}),
            value_col="processing_time_total_s",
            ylabel="seconds",
            title="Total Processing Time by Method",
            save_path=flags.output_dir / "compare_processing_time_total_bars.png",
            show=flags.show,
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
        )
        # Average per item
        _plot_metric_bars(
            agg_time.rename(columns={"avg": "processing_time_avg_s"}),
            value_col="processing_time_avg_s",
            ylabel="seconds per item",
            title="Average Processing Time per Item by Method",
            save_path=flags.output_dir / "compare_processing_time_avg_bars.png",
            show=flags.show,
            method_order=DEFAULT_METHOD_ORDER,
            palette_map=palette_map,
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

        # Grouped plots: input vs output on same chart (totals and averages)
        if have_tokens_in and have_tokens_out:
            agg_tok_total = tokens_df.groupby("eval_method")[
                ["tokens_in", "tokens_out"]
            ].sum(min_count=1).reset_index().rename(columns={"eval_method": "method"})
            _plot_grouped_bars_two_series(
                agg_tok_total,
                col_in="tokens_in",
                col_out="tokens_out",
                label_in="Input Tokens",
                label_out="Output Tokens",
                ylabel="tokens",
                title="Tokens (Total) by Method",
                save_path=flags.output_dir / "compare_tokens_total_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
            )

            agg_tok_avg = tokens_df.groupby("eval_method")[
                ["tokens_in", "tokens_out"]
            ].mean().reset_index().rename(columns={"eval_method": "method"})
            _plot_grouped_bars_two_series(
                agg_tok_avg,
                col_in="tokens_in",
                col_out="tokens_out",
                label_in="Input Tokens (avg)",
                label_out="Output Tokens (avg)",
                ylabel="tokens per item",
                title="Tokens (Average per Item) by Method",
                save_path=flags.output_dir / "compare_tokens_avg_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
            )

        # Total tokens single series plots remain
        if "tokens_total" in tokens_df.columns:
            agg_tok_total_single = tokens_df.groupby("eval_method")["tokens_total"].agg(total="sum", avg="mean").reset_index().rename(columns={"eval_method": "method"})
            _plot_metric_bars(
                agg_tok_total_single.rename(columns={"total": "tokens_total_total"}),
                value_col="tokens_total_total",
                ylabel="tokens",
                title="Total Tokens (Total) by Method",
                save_path=flags.output_dir / "compare_tokens_total_total_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
                label_fmt=lambda v: f"{int(round(v)):,}",
            )
            _plot_metric_bars(
                agg_tok_total_single.rename(columns={"avg": "tokens_total_avg"}),
                value_col="tokens_total_avg",
                ylabel="tokens per item",
                title="Total Tokens (Average per Item) by Method",
                save_path=flags.output_dir / "compare_tokens_total_avg_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
                label_fmt=lambda v: f"{int(round(v)):,}",
            )

    # Cost (input, output, total) — totals and averages
    have_cost_in = "cost_in" in df.columns
    have_cost_out = "cost_out" in df.columns
    have_total_cost = "total_cost" in df.columns
    if have_cost_in or have_cost_out or have_total_cost:
        cost_df = df.copy()
        # Ensure a total cost column
        if not have_total_cost:
            if have_cost_in and have_cost_out:
                cost_df["total_cost"] = cost_df["cost_in"].astype(float) + cost_df["cost_out"].astype(float)
            elif have_cost_in:
                cost_df["total_cost"] = cost_df["cost_in"].astype(float)
            elif have_cost_out:
                cost_df["total_cost"] = cost_df["cost_out"].astype(float)

        # Grouped cost plots: input vs output
        if have_cost_in and have_cost_out:
            agg_cost_total = cost_df.groupby("eval_method")[
                ["cost_in", "cost_out"]
            ].sum(min_count=1).reset_index().rename(columns={"eval_method": "method"})
            _plot_grouped_bars_two_series(
                agg_cost_total,
                col_in="cost_in",
                col_out="cost_out",
                label_in="Input Cost",
                label_out="Output Cost",
                ylabel="$",
                title="Cost (Total) by Method",
                save_path=flags.output_dir / "compare_cost_total_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                label_fmt=lambda v: f"${v:,.4f}",
            )

            agg_cost_avg = cost_df.groupby("eval_method")[
                ["cost_in", "cost_out"]
            ].mean().reset_index().rename(columns={"eval_method": "method"})
            _plot_grouped_bars_two_series(
                agg_cost_avg,
                col_in="cost_in",
                col_out="cost_out",
                label_in="Input Cost (avg)",
                label_out="Output Cost (avg)",
                ylabel="$ per item",
                title="Cost (Average per Item) by Method",
                save_path=flags.output_dir / "compare_cost_avg_grouped_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                label_fmt=lambda v: f"${v:,.4f}",
            )

        # Total cost single series plots remain
        if "total_cost" in cost_df.columns:
            agg_cost_single = cost_df.groupby("eval_method")["total_cost"].agg(total="sum", avg="mean").reset_index().rename(columns={"eval_method": "method"})
            _plot_metric_bars(
                agg_cost_single.rename(columns={"total": "total_cost_total"}),
                value_col="total_cost_total",
                ylabel="$",
                title="Total Cost (Total) by Method",
                save_path=flags.output_dir / "compare_total_cost_total_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
                label_fmt=lambda v: f"${v:,.4f}",
            )
            _plot_metric_bars(
                agg_cost_single.rename(columns={"avg": "total_cost_avg"}),
                value_col="total_cost_avg",
                ylabel="$ per item",
                title="Total Cost (Average per Item) by Method",
                save_path=flags.output_dir / "compare_total_cost_avg_bars.png",
                show=flags.show,
                method_order=DEFAULT_METHOD_ORDER,
                palette_map=palette_map,
                label_fmt=lambda v: f"${v:,.4f}",
            )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


