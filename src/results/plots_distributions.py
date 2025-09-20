from __future__ import annotations

"""Distribution visualizations: ECDFs/violins, token composition, and cost breakdowns."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ecdf_or_violin(df: pd.DataFrame, *, metrics: list[str], kind: str = "violin", save_prefix: Optional[Path] = None, show: bool = True) -> None:
    """Plot ECDF or violin per metric, faceted by `eval_method`.

    Saves one PNG per metric to `<save_prefix>_{metric}_{kind}.png` if `save_prefix` is provided.
    """
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        if kind == "ecdf":
            sns.ecdfplot(data=df, x=metric, hue="eval_method")
            plt.title(f"ECDF of {metric} by method")
        else:
            sns.violinplot(data=df, x="eval_method", y=metric, inner="box", cut=0)
            plt.title(f"Distribution of {metric} by method")
        plt.tight_layout()
        if save_prefix is not None:
            plt.savefig(Path(f"{save_prefix}_{metric}_{kind}.png"), dpi=150)
        if show:
            plt.show()
        plt.close()


def token_composition_stacks(df: pd.DataFrame, *, save_path: Optional[Path] = None, show: bool = True) -> None:
    """Stacked bars of average token component shares per method."""
    cols = [
        "tokens_system_prompt",
        "tokens_original",
        "tokens_diff_a",
        "tokens_diff_b",
        "tokens_output",
    ]
    if not set(cols).issubset(df.columns):
        return

    share = df.groupby("eval_method")[cols].mean()
    totals = share.sum(axis=1)
    share = share.div(totals, axis=0)

    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(share))
    methods = share.index.tolist()
    for c in cols:
        plt.bar(methods, share[c].values, bottom=bottom, label=c)
        bottom += share[c].values
    plt.legend()
    plt.title("Token Composition Share by Method")
    plt.ylabel("share of tokens_total")
    plt.xlabel("method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def cost_breakdown(df: pd.DataFrame, *, save_path: Optional[Path] = None, show: bool = True) -> None:
    """Stacked cost shares (input/output) per method with $/1k token annotations."""
    if not {"cost_in", "cost_out"}.issubset(df.columns):
        return
    agg = df.groupby("eval_method")[
        ["cost_in", "cost_out", "tokens_in", "tokens_out"]
    ].sum(min_count=1)
    # Compute rates ($ per 1k tokens)
    agg["rate_in_per_1k"] = (agg["cost_in"] / agg["tokens_in"]).replace([np.inf, -np.inf], np.nan) * 1000.0
    agg["rate_out_per_1k"] = (agg["cost_out"] / agg["tokens_out"]).replace([np.inf, -np.inf], np.nan) * 1000.0

    bars = agg[["cost_in", "cost_out"]].div(agg[["cost_in", "cost_out"]].sum(axis=1), axis=0).fillna(0)
    bars = bars.rename(columns={"cost_in": "Input Cost", "cost_out": "Output Cost"})

    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(bars))
    methods = bars.index.tolist()
    for c in ["Input Cost", "Output Cost"]:
        plt.bar(methods, bars[c].values, bottom=bottom, label=c)
        bottom += bars[c].values

    # Annotate rates on top
    for i, m in enumerate(methods):
        rin = agg.loc[m, "rate_in_per_1k"]
        rout = agg.loc[m, "rate_out_per_1k"]
        label = f"in: ${rin:.3f}/1k, out: ${rout:.3f}/1k" if np.isfinite(rin) and np.isfinite(rout) else ""
        if label:
            plt.text(i, 1.02, label, ha="center", va="bottom", fontsize=9, rotation=0)

    plt.legend(loc="upper right")
    plt.title("Cost Breakdown by Method (share) with $/1k token rates")
    plt.ylabel("share of total cost")
    plt.xlabel("method")
    plt.ylim(0, 1.15)
    plt.xticks(rotation=20)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

