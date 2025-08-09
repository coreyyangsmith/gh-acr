from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, exact_match_rate, cost_per_success, time_per_success


PrimaryMetric = Literal["similarity", "bleu3", "rouge_l"]


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per eval_method with core metrics and 95% CIs for EM rate."""
    if "eval_method" not in df.columns:
        raise ValueError("missing eval_method in results")

    grouped = df.groupby("eval_method", dropna=False)
    rows: list[dict] = []
    for method, g in grouped:
        em = exact_match_rate(g["exact_match"]) if "exact_match" in g else np.nan
        em_lo, em_hi = bootstrap_ci(g["exact_match"].astype(int), statistic=np.mean) if "exact_match" in g else (np.nan, np.nan)
        # Aggregated quality per dollar: mean(similarity)/mean(total_cost)
        qpd = np.nan
        if {"similarity", "total_cost"}.issubset(g.columns):
            denom = float(g["total_cost"].mean())
            qpd = float(g["similarity"].mean()) / denom if denom and not np.isnan(denom) and denom != 0 else np.nan

        rows.append(
            {
                "method": method,
                "n": int(len(g)),
                "exact_match_rate": round(em, 4),
                "em_ci_low": round(em_lo, 4),
                "em_ci_high": round(em_hi, 4),
                "similarity_mean": round(float(g["similarity"].mean()), 4) if "similarity" in g else np.nan,
                "similarity_median": round(float(g["similarity"].median()), 4) if "similarity" in g else np.nan,
                "bleu3_mean": round(float(g.get("bleu3", pd.Series(dtype=float)).mean()), 4) if "bleu3" in g else np.nan,
                "rouge_l_mean": round(float(g.get("rouge_l", pd.Series(dtype=float)).mean()), 4) if "rouge_l" in g else np.nan,
                "avg_total_cost": round(float(g["total_cost"].mean()), 6) if "total_cost" in g else np.nan,
                "avg_processing_time_s": round(float(g["processing_time_s"].mean()), 3) if "processing_time_s" in g else np.nan,
                "cost_per_success": round(cost_per_success(g["total_cost"], g["exact_match"]), 6) if {"total_cost", "exact_match"}.issubset(g.columns) else np.nan,
                "time_per_success_s": round(time_per_success(g["processing_time_s"], g["exact_match"]), 3) if {"processing_time_s", "exact_match"}.issubset(g.columns) else np.nan,
                "quality_per_dollar": round(qpd, 6),
            }
        )

    table = pd.DataFrame(rows).sort_values(["exact_match_rate", "similarity_mean"], ascending=[False, False]).reset_index(drop=True)
    return table


def by_difficulty_leaderboard(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "difficulty" not in df.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for difficulty, g in df.groupby("difficulty"):
        out[str(difficulty)] = method_summary(g)
    return out


def pairwise_win_matrix(df: pd.DataFrame, metric: PrimaryMetric = "similarity") -> pd.DataFrame:
    """Compute pairwise win rates: rows beat columns on metric, ties split.

    Expects each row corresponds to a file result for a given method and file identifier combination.
    It will pivot by a unique key (repo + file_name) to align methods.
    """
    required = {"eval_method", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"missing required columns for pairwise matrix: {required}")

    # Build a key to align samples. Prefer an explicit id if present.
    if "id" in df.columns:
        key = df["id"].astype(str)
    else:
        key = df[[c for c in ["repo", "file_name"] if c in df.columns]].astype(str).agg("__".join, axis=1)
    tmp = df.assign(_key=key)

    methods = sorted(tmp["eval_method"].unique().tolist())
    win_counts = pd.DataFrame(0.0, index=methods, columns=methods)
    total_counts = pd.DataFrame(0.0, index=methods, columns=methods)

    for k, g in tmp.groupby("_key"):
        pivot = g.pivot(index="_key", columns="eval_method", values=metric)
        if pivot.shape[0] != 1:
            continue
        values = pivot.iloc[0]
        for r in methods:
            for c in methods:
                if r == c:
                    continue
                vr = values.get(r, np.nan)
                vc = values.get(c, np.nan)
                if pd.isna(vr) or pd.isna(vc):
                    continue
                if vr > vc:
                    win_counts.loc[r, c] += 1
                elif vr < vc:
                    # no increment for r beating c
                    pass
                else:
                    # tie: split
                    win_counts.loc[r, c] += 0.5
                total_counts.loc[r, c] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (win_counts / total_counts).fillna(0.0) * 100.0
    return pct.round(1)


def pairwise_cost_win_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Lower total_cost is better."""
    return pairwise_win_matrix(df, metric="total_cost").applymap(lambda x: 100 - x)

