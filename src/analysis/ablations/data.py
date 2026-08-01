"""Data preparation for Better-Judge ablation analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..stats import (
    paired_bootstrap_mean_ci,
    p_value_binomial_sign_test_two_sided,
    p_value_wilcoxon_signed_rank,
)
from .config import (
    AblationConfig,
    DEFAULT_CONFIG,
    CONFLICT_SIZE_BUCKETS,
    COST_COLUMNS,
)


def _coerce_bool_metric(series: pd.Series) -> pd.Series:
    """Coerce exact_match-like series to float 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "1.0", "yes", "y", "t"])
        .astype(float)
    )


def _scenario_key(df: pd.DataFrame) -> pd.Series:
    """Unique key per file-level row (id + file_name when available)."""
    if "id" in df.columns and "file_name" in df.columns:
        return df["id"].astype(str) + "::" + df["file_name"].astype(str)
    if "id" in df.columns:
        return df["id"].astype(str)
    cols = [c for c in ["repo", "file_name"] if c in df.columns]
    if cols:
        return df[cols].astype(str).agg("__".join, axis=1)
    return pd.Series(np.arange(len(df)), index=df.index).astype(str)


def prepare_results(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Clean results CSV for ablation analysis."""
    work = df.copy()
    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])
    if config.exclude_soft_degraded and "soft_degraded" in work.columns:
        degraded = work["soft_degraded"].astype(str).str.lower().isin(["true", "1", "1.0"])
        work = work[~degraded].copy()
    work["_key"] = _scenario_key(work)
    return work


def common_keys_for_methods(
    df: pd.DataFrame,
    methods: list[str],
    *,
    model_name: Optional[str] = None,
    require_all_models: bool = False,
) -> set[str]:
    """Keys present for every method (optionally within one model).

    For LLM methods, keys are intersected within each model then optionally
    across models when ``require_all_models`` is True. Baselines (empty model)
    are matched by key only when included.
    """
    work = df[df["eval_method"].isin(methods)].copy()
    if work.empty:
        return set()

    llm_methods = [m for m in methods if m not in ("base_a", "base_b")]
    baseline_methods = [m for m in methods if m in ("base_a", "base_b")]

    if model_name is not None:
        llm = work[work["eval_method"].isin(llm_methods)]
        llm = llm[llm["model_name"] == model_name]
        sets: list[set[str]] = []
        for m in llm_methods:
            sets.append(set(llm[llm["eval_method"] == m]["_key"].astype(str).unique()))
        for m in baseline_methods:
            sets.append(set(work[work["eval_method"] == m]["_key"].astype(str).unique()))
        if not sets:
            return set()
        return set.intersection(*sets)

    # Per-model intersection of LLM methods, then intersect across models
    llm = work[work["eval_method"].isin(llm_methods)]
    models = llm["model_name"].dropna().unique().tolist() if "model_name" in llm.columns else []
    if not models:
        sets = [set(work[work["eval_method"] == m]["_key"].astype(str).unique()) for m in methods]
        return set.intersection(*sets) if sets else set()

    per_model: list[set[str]] = []
    for model in models:
        model_df = llm[llm["model_name"] == model]
        msets = [set(model_df[model_df["eval_method"] == m]["_key"].astype(str).unique()) for m in llm_methods]
        if not msets or any(len(s) == 0 for s in msets):
            continue
        keys = set.intersection(*msets)
        for bm in baseline_methods:
            keys &= set(work[work["eval_method"] == bm]["_key"].astype(str).unique())
        per_model.append(keys)

    if not per_model:
        return set()
    if require_all_models:
        return set.intersection(*per_model)
    # Union of per-model common sets (each model compared on its own coverage)
    return set.union(*per_model)


def filter_to_common(
    df: pd.DataFrame,
    keys: set[str],
    methods: list[str],
    *,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Restrict to given keys and methods."""
    work = df[df["eval_method"].isin(methods) & df["_key"].astype(str).isin(keys)].copy()
    if model_name is not None and "model_name" in work.columns:
        # Keep baselines (NaN model) plus the requested model
        mask = work["model_name"].isna() | (work["model_name"] == model_name)
        # Also keep baseline methods regardless
        mask = mask | work["eval_method"].isin(["base_a", "base_b"])
        work = work[mask]
    return work


def list_models(df: pd.DataFrame) -> list[str]:
    """Non-null model names present for LLM methods."""
    if "model_name" not in df.columns:
        return ["unknown"]
    models = df["model_name"].dropna().unique().tolist()
    return sorted(str(m) for m in models)


@dataclass
class PairedDeltaResult:
    """Paired delta statistics for one comparison."""

    model_name: str
    method_a: str
    method_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_delta: float  # a - b
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    wins: int  # a > b
    ties: int
    losses: int  # a < b


def _paired_wide(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
) -> pd.DataFrame:
    """Pivot two methods to columns ``a`` and ``b`` for a metric."""
    sub = df[df["eval_method"].isin([method_a, method_b])].copy()
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    # Prefer model-aware keys: group by key + model for LLM rows
    if "model_name" in sub.columns:
        sub["_pivot_key"] = sub["_key"].astype(str) + "||" + sub["model_name"].fillna("__bl__").astype(str)
    else:
        sub["_pivot_key"] = sub["_key"].astype(str)

    pivot = sub.pivot_table(
        index="_pivot_key",
        columns="eval_method",
        values=metric,
        aggfunc="first",
    )
    if method_a not in pivot.columns or method_b not in pivot.columns:
        return pd.DataFrame()
    out = pivot[[method_a, method_b]].dropna().copy()
    out.columns = ["a", "b"]
    return out


def compute_paired_delta(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
    *,
    model_name: Optional[str] = None,
    config: AblationConfig = DEFAULT_CONFIG,
) -> Optional[PairedDeltaResult]:
    """Mean(method_a - method_b) with bootstrap CI and significance test."""
    work = df
    if model_name is not None and "model_name" in df.columns:
        work = df[(df["model_name"] == model_name) | df["eval_method"].isin(["base_a", "base_b"])]

    wide = _paired_wide(work, method_a, method_b, metric)
    if wide.empty:
        return None

    deltas = (wide["a"] - wide["b"]).to_numpy(dtype=float)
    mean_delta = float(np.mean(deltas))
    ci_low, ci_high = paired_bootstrap_mean_ci(
        deltas,
        n_boot=config.n_bootstrap,
        ci=config.ci_level,
        random_state=config.random_state,
    )

    tol = 1e-9 if metric == "exact_match" else 1e-6
    wins = int(np.sum(deltas > tol))
    losses = int(np.sum(deltas < -tol))
    ties = int(len(deltas) - wins - losses)

    if metric == "exact_match":
        p_value = p_value_binomial_sign_test_two_sided(wins, losses)
        test = "binomial_sign"
    else:
        p_value = p_value_wilcoxon_signed_rank(deltas)
        test = "wilcoxon"

    return PairedDeltaResult(
        model_name=model_name or "all",
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        n_pairs=len(wide),
        mean_a=float(wide["a"].mean()),
        mean_b=float(wide["b"].mean()),
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test=test,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def compute_component_contributions(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Paired deltas: anchor - ablation (quality drop when removing component)."""
    rows: list[dict] = []
    models = list_models(df[df["eval_method"].isin(config.all_multi_methods())])

    for model in models:
        methods = [config.anchor_method, *config.ablations]
        keys = common_keys_for_methods(df, methods, model_name=model)
        if not keys:
            continue
        sub = filter_to_common(df, keys, methods, model_name=model)
        for ablation in config.ablations:
            for metric in config.metrics:
                if metric not in sub.columns:
                    continue
                result = compute_paired_delta(
                    sub,
                    config.anchor_method,
                    ablation,
                    metric,
                    model_name=model,
                    config=config,
                )
                if result is None:
                    continue
                rows.append(
                    {
                        "model_name": result.model_name,
                        "ablation": ablation,
                        "component": ablation,
                        "metric": metric,
                        "n_pairs": result.n_pairs,
                        "mean_anchor": result.mean_a,
                        "mean_ablation": result.mean_b,
                        "mean_delta": result.mean_delta,
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "p_value": result.p_value,
                        "test": result.test,
                        "wins": result.wins,
                        "ties": result.ties,
                        "losses": result.losses,
                    }
                )
    return pd.DataFrame(rows)


def compute_method_ladder_means(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Mean quality (± bootstrap CI) per method on the common multi-method set."""
    rows: list[dict] = []
    ladder = config.ladder_methods()
    # Common set among agent + anchor + ablations (baselines joined by key)
    core = [config.agent_method, config.anchor_method, *config.ablations]
    models = list_models(df[df["eval_method"].isin(core)])

    for model in models:
        keys = common_keys_for_methods(df, core, model_name=model)
        if not keys:
            continue
        # Include baselines on those keys
        methods = list(dict.fromkeys([*config.baseline_methods, *core]))
        sub = filter_to_common(df, keys, methods, model_name=model)

        for method in ladder:
            method_df = sub[sub["eval_method"] == method]
            if method in config.baseline_methods:
                method_df = sub[sub["eval_method"] == method]
                method_df = method_df[method_df["_key"].astype(str).isin(keys)]
            elif "model_name" in method_df.columns:
                method_df = method_df[method_df["model_name"] == model]

            if method_df.empty:
                continue

            for metric in config.metrics:
                if metric not in method_df.columns:
                    continue
                vals = pd.to_numeric(method_df[metric], errors="coerce").dropna().to_numpy()
                if vals.size == 0:
                    continue
                mean = float(np.mean(vals))
                ci_low, ci_high = paired_bootstrap_mean_ci(
                    vals,
                    n_boot=config.n_bootstrap,
                    ci=config.ci_level,
                    random_state=config.random_state,
                )
                rows.append(
                    {
                        "model_name": model,
                        "method": method,
                        "metric": metric,
                        "n": int(vals.size),
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def compute_wtl_matrix(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "exact_match",
) -> pd.DataFrame:
    """Win/tie/loss for BJ vs each ablation and each method vs agent."""
    rows: list[dict] = []
    models = list_models(df[df["eval_method"].isin(config.all_multi_methods())])
    comparisons: list[tuple[str, str, str]] = []
    for abl in config.ablations:
        comparisons.append((config.anchor_method, abl, "anchor_vs_ablation"))
    for method in [config.anchor_method, *config.ablations]:
        comparisons.append((method, config.agent_method, "method_vs_agent"))

    for model in models:
        core = [config.agent_method, config.anchor_method, *config.ablations]
        keys = common_keys_for_methods(df, core, model_name=model)
        if not keys:
            continue
        sub = filter_to_common(df, keys, core, model_name=model)
        for method_a, method_b, kind in comparisons:
            result = compute_paired_delta(
                sub, method_a, method_b, metric, model_name=model, config=config
            )
            if result is None:
                continue
            total = result.wins + result.ties + result.losses
            rows.append(
                {
                    "model_name": model,
                    "comparison": kind,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_pairs": result.n_pairs,
                    "wins": result.wins,
                    "ties": result.ties,
                    "losses": result.losses,
                    "win_pct": result.wins / total if total else np.nan,
                    "tie_pct": result.ties / total if total else np.nan,
                    "loss_pct": result.losses / total if total else np.nan,
                    "mean_delta": result.mean_delta,
                    "p_value": result.p_value,
                }
            )
    return pd.DataFrame(rows)


def _conflict_size(df: pd.DataFrame) -> pd.Series:
    """Token conflict size proxy from diff token columns."""
    a = pd.to_numeric(df.get("tokens_diff_a", pd.Series(np.nan, index=df.index)), errors="coerce")
    b = pd.to_numeric(df.get("tokens_diff_b", pd.Series(np.nan, index=df.index)), errors="coerce")
    return a.fillna(0) + b.fillna(0)


def _bucket_conflict_size(series: pd.Series) -> pd.Series:
    labels = []
    for val in series:
        if pd.isna(val):
            labels.append("unknown")
            continue
        assigned = "unknown"
        for name, lo, hi in CONFLICT_SIZE_BUCKETS:
            if lo <= float(val) <= hi:
                assigned = name
                break
        labels.append(assigned)
    return pd.Series(labels, index=series.index)


def compute_stratified_component_deltas(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "exact_match",
) -> pd.DataFrame:
    """Component Δ (anchor - ablation) stratified by difficulty / size / conflict size."""
    rows: list[dict] = []
    models = list_models(df[df["eval_method"].isin(config.all_multi_methods())])

    work = df.copy()
    work["_conflict_size"] = _conflict_size(work)
    work["_conflict_bucket"] = _bucket_conflict_size(work["_conflict_size"])

    strata_cols = {
        "difficulty": "difficulty",
        "project_size": "project_size",
        "conflict_size": "_conflict_bucket",
    }

    for model in models:
        methods = [config.anchor_method, *config.ablations]
        keys = common_keys_for_methods(work, methods, model_name=model)
        if not keys:
            continue
        sub = filter_to_common(work, keys, methods, model_name=model)
        sub = sub[sub["model_name"] == model]

        for stratum_name, col in strata_cols.items():
            if col not in sub.columns:
                continue
            for bucket in sorted(sub[col].dropna().astype(str).unique()):
                bucket_keys = set(
                    sub[sub[col].astype(str) == bucket]["_key"].astype(str).unique()
                )
                if not bucket_keys:
                    continue
                bucket_df = sub[sub["_key"].astype(str).isin(bucket_keys)]
                for ablation in config.ablations:
                    result = compute_paired_delta(
                        bucket_df,
                        config.anchor_method,
                        ablation,
                        metric,
                        model_name=model,
                        config=config,
                    )
                    if result is None or result.n_pairs < 5:
                        continue
                    rows.append(
                        {
                            "model_name": model,
                            "stratum": stratum_name,
                            "bucket": bucket,
                            "ablation": ablation,
                            "metric": metric,
                            "n_pairs": result.n_pairs,
                            "mean_delta": result.mean_delta,
                            "ci_low": result.ci_low,
                            "ci_high": result.ci_high,
                            "p_value": result.p_value,
                            "wins": result.wins,
                            "ties": result.ties,
                            "losses": result.losses,
                        }
                    )
    return pd.DataFrame(rows)


def compute_cost_quality(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Mean quality and cost/tokens/time per method × model on common set."""
    rows: list[dict] = []
    core = [config.agent_method, config.anchor_method, *config.ablations]
    models = list_models(df[df["eval_method"].isin(core)])

    for model in models:
        keys = common_keys_for_methods(df, core, model_name=model)
        if not keys:
            continue
        sub = filter_to_common(df, keys, core, model_name=model)
        sub = sub[sub["model_name"] == model]
        for method in core:
            method_df = sub[sub["eval_method"] == method]
            if method_df.empty:
                continue
            row: dict = {
                "model_name": model,
                "method": method,
                "n": len(method_df),
            }
            for metric in config.metrics:
                if metric in method_df.columns:
                    row[f"mean_{metric}"] = float(
                        pd.to_numeric(method_df[metric], errors="coerce").mean()
                    )
            for col in COST_COLUMNS:
                if col in method_df.columns:
                    row[f"mean_{col}"] = float(
                        pd.to_numeric(method_df[col], errors="coerce").mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def compute_routing_counterfactuals(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "similarity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare better_judge routing vs bj_no_judge (forced MIX).

    Returns
    -------
    agreement_df
        Distribution of BJ bypass_method and paired quality vs no-judge.
    conditional_df
        Quality deltas conditioned on BJ's A/B/MIX decision.
    """
    agreement_rows: list[dict] = []
    conditional_rows: list[dict] = []
    models = list_models(df[df["eval_method"].isin([config.anchor_method, "bj_no_judge"])])

    for model in models:
        methods = [config.anchor_method, "bj_no_judge"]
        if "bj_no_judge" not in config.ablations and "bj_no_judge" not in df["eval_method"].unique():
            continue
        keys = common_keys_for_methods(df, methods, model_name=model)
        if not keys:
            continue
        sub = filter_to_common(df, keys, methods, model_name=model)
        sub = sub[sub["model_name"] == model]

        bj = sub[sub["eval_method"] == config.anchor_method][
            ["_key", "bypass_method", metric, "exact_match"]
        ].rename(
            columns={
                "bypass_method": "bj_decision",
                metric: "bj_metric",
                "exact_match": "bj_em",
            }
        )
        nj = sub[sub["eval_method"] == "bj_no_judge"][
            ["_key", "bypass_method", metric, "exact_match"]
        ].rename(
            columns={
                "bypass_method": "nj_decision",
                metric: "nj_metric",
                "exact_match": "nj_em",
            }
        )
        paired = bj.merge(nj, on="_key", how="inner")
        if paired.empty:
            continue

        paired["bj_decision"] = paired["bj_decision"].fillna("UNK").astype(str).str.upper()
        paired["delta"] = paired["bj_metric"] - paired["nj_metric"]
        paired["bj_better"] = paired["delta"] > 1e-6
        paired["nj_better"] = paired["delta"] < -1e-6

        for decision, grp in paired.groupby("bj_decision"):
            agreement_rows.append(
                {
                    "model_name": model,
                    "bj_decision": decision,
                    "n": len(grp),
                    "pct": len(grp) / len(paired),
                    f"mean_{metric}_bj": float(grp["bj_metric"].mean()),
                    f"mean_{metric}_no_judge": float(grp["nj_metric"].mean()),
                    "mean_delta": float(grp["delta"].mean()),
                    "bj_win_pct": float(grp["bj_better"].mean()),
                    "no_judge_win_pct": float(grp["nj_better"].mean()),
                    "mean_em_bj": float(grp["bj_em"].mean()) if "bj_em" in grp else np.nan,
                    "mean_em_no_judge": float(grp["nj_em"].mean()) if "nj_em" in grp else np.nan,
                }
            )
            conditional_rows.append(
                {
                    "model_name": model,
                    "bj_decision": decision,
                    "n": len(grp),
                    "mean_delta": float(grp["delta"].mean()),
                    "ci_low": paired_bootstrap_mean_ci(
                        grp["delta"].to_numpy(),
                        n_boot=config.n_bootstrap,
                        ci=config.ci_level,
                        random_state=config.random_state,
                    )[0],
                    "ci_high": paired_bootstrap_mean_ci(
                        grp["delta"].to_numpy(),
                        n_boot=config.n_bootstrap,
                        ci=config.ci_level,
                        random_state=config.random_state,
                    )[1],
                    "would_bypass": decision in {"A", "B"},
                }
            )

        # Overall agreement row
        agreement_rows.append(
            {
                "model_name": model,
                "bj_decision": "ALL",
                "n": len(paired),
                "pct": 1.0,
                f"mean_{metric}_bj": float(paired["bj_metric"].mean()),
                f"mean_{metric}_no_judge": float(paired["nj_metric"].mean()),
                "mean_delta": float(paired["delta"].mean()),
                "bj_win_pct": float(paired["bj_better"].mean()),
                "no_judge_win_pct": float(paired["nj_better"].mean()),
                "mean_em_bj": float(paired["bj_em"].mean()),
                "mean_em_no_judge": float(paired["nj_em"].mean()),
            }
        )

    return pd.DataFrame(agreement_rows), pd.DataFrame(conditional_rows)


def compute_disagreement_cases(
    df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "similarity",
    top_k: int = 50,
) -> pd.DataFrame:
    """Mine high-leverage disagreement instances across ablations."""
    rows: list[dict] = []
    models = list_models(df[df["eval_method"].isin(config.all_multi_methods())])
    core = [config.agent_method, config.anchor_method, *config.ablations]

    for model in models:
        keys = common_keys_for_methods(df, core, model_name=model)
        if not keys:
            continue
        sub = filter_to_common(df, keys, core, model_name=model)
        sub = sub[sub["model_name"] == model]

        wide_frames = []
        for method in core:
            mdf = sub[sub["eval_method"] == method][["_key", "id", "repo", "file_name", metric, "exact_match"]].copy()
            mdf = mdf.rename(
                columns={
                    metric: f"{method}_metric",
                    "exact_match": f"{method}_em",
                }
            )
            # Keep meta from first
            wide_frames.append(mdf)

        # Successive merge on _key
        paired = wide_frames[0]
        for frame in wide_frames[1:]:
            meta = ["id", "repo", "file_name"]
            drop_meta = [c for c in meta if c in frame.columns and c in paired.columns]
            frame = frame.drop(columns=drop_meta, errors="ignore")
            paired = paired.merge(frame, on="_key", how="inner")

        if paired.empty:
            continue

        anchor_m = f"{config.anchor_method}_metric"
        agent_m = f"{config.agent_method}_metric"
        anchor_em = f"{config.anchor_method}_em"
        agent_em = f"{config.agent_method}_em"

        for ablation in config.ablations:
            abl_m = f"{ablation}_metric"
            if abl_m not in paired.columns:
                continue
            # Ablation beats full BJ
            better = paired[paired[abl_m] > paired[anchor_m] + 1e-6].copy()
            better["case_type"] = "ablation_beats_bj"
            better["ablation"] = ablation
            better["delta"] = better[abl_m] - better[anchor_m]

            # Component removal collapses quality (large drop)
            worse = paired[paired[anchor_m] > paired[abl_m] + 0.05].copy()
            worse["case_type"] = "component_hurts"
            worse["ablation"] = ablation
            worse["delta"] = worse[anchor_m] - worse[abl_m]

            for chunk in (better, worse):
                if chunk.empty:
                    continue
                chunk = chunk.nlargest(top_k, "delta")
                for _, r in chunk.iterrows():
                    rows.append(
                        {
                            "model_name": model,
                            "case_type": r["case_type"],
                            "ablation": ablation,
                            "id": r.get("id"),
                            "repo": r.get("repo"),
                            "file_name": r.get("file_name"),
                            "key": r["_key"],
                            "metric": metric,
                            "bj_metric": r[anchor_m],
                            "ablation_metric": r[abl_m],
                            "agent_metric": r[agent_m] if agent_m in paired.columns else np.nan,
                            "delta": r["delta"],
                            "bj_em": r.get(anchor_em),
                            "agent_em": r.get(agent_em),
                        }
                    )

        # Agent wins EM but BJ also wins — which ablation matches agent
        if agent_em in paired.columns and anchor_em in paired.columns:
            both = paired[(paired[agent_em] >= 0.5) & (paired[anchor_em] >= 0.5)]
            for _, r in both.head(top_k).iterrows():
                matching = []
                for ablation in config.ablations:
                    em_col = f"{ablation}_em"
                    if em_col in paired.columns and r.get(em_col, 0) >= 0.5:
                        matching.append(ablation)
                rows.append(
                    {
                        "model_name": model,
                        "case_type": "agent_and_bj_em",
                        "ablation": ",".join(matching) if matching else "none",
                        "id": r.get("id"),
                        "repo": r.get("repo"),
                        "file_name": r.get("file_name"),
                        "key": r["_key"],
                        "metric": metric,
                        "bj_metric": r[anchor_m],
                        "ablation_metric": np.nan,
                        "agent_metric": r[agent_m],
                        "delta": r[anchor_m] - r[agent_m],
                        "bj_em": r.get(anchor_em),
                        "agent_em": r.get(agent_em),
                    }
                )

        # Only one ablation collapses EM while BJ succeeds
        if anchor_em in paired.columns:
            bj_ok = paired[paired[anchor_em] >= 0.5]
            for _, r in bj_ok.iterrows():
                collapsed = []
                for ablation in config.ablations:
                    em_col = f"{ablation}_em"
                    if em_col in paired.columns and r.get(em_col, 1) < 0.5:
                        collapsed.append(ablation)
                if len(collapsed) == 1:
                    rows.append(
                        {
                            "model_name": model,
                            "case_type": "single_component_em_collapse",
                            "ablation": collapsed[0],
                            "id": r.get("id"),
                            "repo": r.get("repo"),
                            "file_name": r.get("file_name"),
                            "key": r["_key"],
                            "metric": "exact_match",
                            "bj_metric": r[anchor_m],
                            "ablation_metric": r.get(f"{collapsed[0]}_metric", np.nan),
                            "agent_metric": r[agent_m] if agent_m in paired.columns else np.nan,
                            "delta": np.nan,
                            "bj_em": r.get(anchor_em),
                            "agent_em": r.get(agent_em),
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Deduplicate and keep top per case_type
    out = out.drop_duplicates(subset=["model_name", "case_type", "ablation", "key"])
    return out


def compute_cross_model_stability(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare component Δ across models for the same ablation×metric."""
    if contributions.empty or "model_name" not in contributions.columns:
        return pd.DataFrame()

    models = sorted(contributions["model_name"].unique().tolist())
    if len(models) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for (ablation, metric), grp in contributions.groupby(["ablation", "metric"]):
        pivot = grp.set_index("model_name")["mean_delta"]
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                if m1 not in pivot.index or m2 not in pivot.index:
                    continue
                d1 = float(pivot[m1])
                d2 = float(pivot[m2])
                rows.append(
                    {
                        "ablation": ablation,
                        "metric": metric,
                        "model_a": m1,
                        "model_b": m2,
                        "delta_a": d1,
                        "delta_b": d2,
                        "delta_diff": d1 - d2,
                        "same_sign": (d1 >= 0 and d2 >= 0) or (d1 < 0 and d2 < 0),
                    }
                )
    return pd.DataFrame(rows)
