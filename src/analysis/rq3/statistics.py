"""Statistical analysis utilities for RQ3.

Computes correlations between labels and performance metrics,
performs stratified analysis, and runs statistical tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import RQ3Config, DEFAULT_CONFIG, label_to_column_name


logger = logging.getLogger(__name__)


@dataclass
class LabelPerformanceStats:
    """Performance statistics for samples with a specific label.

    Attributes
    ----------
    label : str
        The label name
    n_samples : int
        Number of samples with this label
    agent_mean : dict[str, float]
        Mean agent performance per metric
    bypass_mean : dict[str, float]
        Mean bypass performance per metric
    delta_mean : dict[str, float]
        Mean delta (bypass - agent) per metric
    bypass_win_rate : dict[str, float]
        Percentage of samples where bypass wins
    """

    label: str
    n_samples: int
    agent_mean: dict[str, float]
    bypass_mean: dict[str, float]
    delta_mean: dict[str, float]
    bypass_win_rate: dict[str, float]


@dataclass
class StratifiedStats:
    """Stratified performance statistics.

    Attributes
    ----------
    stratify_by : str
        The stratification variable (e.g., "difficulty")
    label : str
        The label being analyzed
    strata : dict[str, LabelPerformanceStats]
        Statistics per stratum
    """

    stratify_by: str
    label: str
    strata: dict[str, LabelPerformanceStats]


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(random_state)
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return (np.nan, np.nan)

    boot_means = np.empty(n_boot)
    n = len(clean)
    for i in range(n_boot):
        sample = clean[rng.integers(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci_level) / 2
    return (float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha)))


def compute_performance_by_label(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute performance statistics for each label.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with agent/bypass metrics and label columns
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Performance statistics per label
    """
    logger.info("Computing performance by label...")
    
    rows = []
    
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name not in paired_df.columns:
            continue
        
        # Filter to samples with this label
        with_label = paired_df[paired_df[col_name] == 1]
        without_label = paired_df[paired_df[col_name] == 0]
        
        n_with = len(with_label)
        n_without = len(without_label)
        
        if n_with < config.min_samples:
            continue
        
        row = {
            "label": label,
            "display_name": config.get_label_display(label),
            "n_with_label": n_with,
            "n_without_label": n_without,
        }
        
        for metric in config.metrics:
            agent_col = f"agent_{metric}"
            bypass_col = f"bypass_{metric}"
            delta_col = f"delta_{metric}"
            wins_col = f"bypass_wins_{metric}"
            
            if delta_col not in paired_df.columns:
                continue
            
            # Stats for samples WITH this label
            if agent_col in with_label.columns:
                agent_vals = pd.to_numeric(with_label[agent_col], errors="coerce").dropna()
                row[f"agent_{metric}_with"] = agent_vals.mean() if len(agent_vals) > 0 else np.nan
            
            if bypass_col in with_label.columns:
                bypass_vals = pd.to_numeric(with_label[bypass_col], errors="coerce").dropna()
                row[f"bypass_{metric}_with"] = bypass_vals.mean() if len(bypass_vals) > 0 else np.nan
            
            delta_vals = pd.to_numeric(with_label[delta_col], errors="coerce").dropna()
            if len(delta_vals) > 0:
                row[f"delta_{metric}_with"] = delta_vals.mean()
                ci = _bootstrap_ci(delta_vals.values, config.n_bootstrap, config.ci_level, config.random_state)
                row[f"delta_{metric}_ci_low"] = ci[0]
                row[f"delta_{metric}_ci_high"] = ci[1]
            
            if wins_col in with_label.columns:
                wins = with_label[wins_col].sum()
                row[f"bypass_win_rate_{metric}_with"] = 100 * wins / n_with if n_with > 0 else np.nan
            
            # Stats for samples WITHOUT this label (for comparison)
            if n_without >= config.min_samples:
                delta_vals_without = pd.to_numeric(without_label[delta_col], errors="coerce").dropna()
                if len(delta_vals_without) > 0:
                    row[f"delta_{metric}_without"] = delta_vals_without.mean()
                
                if wins_col in without_label.columns:
                    wins_without = without_label[wins_col].sum()
                    row[f"bypass_win_rate_{metric}_without"] = 100 * wins_without / n_without if n_without > 0 else np.nan
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    logger.info(f"  Computed stats for {len(result)} labels")
    return result


def compute_stratified_analysis(
    paired_df: pd.DataFrame,
    stratify_by: str,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute performance stratified by a characteristic.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with metrics and label columns
    stratify_by : str
        Column to stratify by (e.g., "difficulty", "project_size")
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Stratified performance statistics
    """
    logger.info(f"Computing stratified analysis by {stratify_by}...")
    
    if stratify_by not in paired_df.columns:
        logger.warning(f"  Column {stratify_by} not found")
        return pd.DataFrame()
    
    rows = []
    
    for stratum, stratum_df in paired_df.groupby(stratify_by, dropna=False):
        if len(stratum_df) < config.min_samples:
            continue
        
        for label in config.canonical_labels:
            col_name = label_to_column_name(label)
            if col_name not in stratum_df.columns:
                continue
            
            with_label = stratum_df[stratum_df[col_name] == 1]
            n_with = len(with_label)
            
            if n_with < config.min_samples:
                continue
            
            row = {
                "stratify_by": stratify_by,
                "stratum": str(stratum),
                "label": label,
                "display_name": config.get_label_display(label),
                "n_samples": n_with,
            }
            
            for metric in config.metrics:
                delta_col = f"delta_{metric}"
                wins_col = f"bypass_wins_{metric}"
                
                if delta_col in with_label.columns:
                    delta_vals = pd.to_numeric(with_label[delta_col], errors="coerce").dropna()
                    if len(delta_vals) > 0:
                        row[f"delta_{metric}"] = delta_vals.mean()
                
                if wins_col in with_label.columns:
                    wins = with_label[wins_col].sum()
                    row[f"bypass_win_rate_{metric}"] = 100 * wins / n_with if n_with > 0 else np.nan
            
            rows.append(row)
    
    result = pd.DataFrame(rows)
    logger.info(f"  Computed {len(result)} stratified statistics")
    return result


def compute_statistical_tests(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute statistical tests comparing performance with/without labels.

    Performs:
    - Chi-square test for bypass win rate (categorical)
    - T-test for delta (continuous)
    - Mann-Whitney U test (non-parametric)

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with metrics and label columns
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Statistical test results
    """
    logger.info("Computing statistical tests...")
    
    rows = []
    
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name not in paired_df.columns:
            continue
        
        with_label = paired_df[paired_df[col_name] == 1]
        without_label = paired_df[paired_df[col_name] == 0]
        
        n_with = len(with_label)
        n_without = len(without_label)
        
        if n_with < config.min_samples or n_without < config.min_samples:
            continue
        
        row = {
            "label": label,
            "display_name": config.get_label_display(label),
            "n_with_label": n_with,
            "n_without_label": n_without,
        }
        
        for metric in config.metrics:
            delta_col = f"delta_{metric}"
            wins_col = f"bypass_wins_{metric}"
            
            if delta_col not in paired_df.columns:
                continue
            
            # Get delta values
            delta_with = pd.to_numeric(with_label[delta_col], errors="coerce").dropna().values
            delta_without = pd.to_numeric(without_label[delta_col], errors="coerce").dropna().values
            
            if len(delta_with) < config.min_samples or len(delta_without) < config.min_samples:
                continue
            
            # T-test
            try:
                t_stat, t_pval = stats.ttest_ind(delta_with, delta_without, equal_var=False)
                row[f"ttest_stat_{metric}"] = t_stat
                row[f"ttest_pval_{metric}"] = t_pval
            except Exception:
                pass
            
            # Mann-Whitney U test
            try:
                u_stat, u_pval = stats.mannwhitneyu(delta_with, delta_without, alternative="two-sided")
                row[f"mannwhitney_stat_{metric}"] = u_stat
                row[f"mannwhitney_pval_{metric}"] = u_pval
            except Exception:
                pass
            
            # Chi-square for win rate
            if wins_col in with_label.columns and wins_col in without_label.columns:
                try:
                    wins_with = with_label[wins_col].sum()
                    losses_with = n_with - wins_with
                    wins_without = without_label[wins_col].sum()
                    losses_without = n_without - wins_without
                    
                    contingency = [[wins_with, losses_with], [wins_without, losses_without]]
                    chi2, chi_pval, dof, expected = stats.chi2_contingency(contingency)
                    row[f"chi2_stat_{metric}"] = chi2
                    row[f"chi2_pval_{metric}"] = chi_pval
                except Exception:
                    pass
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    logger.info(f"  Computed tests for {len(result)} labels")
    return result


def compute_mcnemar_test(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> dict:
    """Exact McNemar (binomial sign test on discordant pairs) for overall method difference.

    Builds the paired 2x2 table (Agent vs Bypass EM) and tests whether Bypass is
    better than Agent using discordant pairs: under H0, c ~ Binomial(b+c, 0.5).

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with agent_exact_match and bypass_exact_match columns
    config : RQ3Config
        Configuration

    Returns
    -------
    dict
        Keys: n_total, n_both_correct, n_both_wrong, b (agent_only), c (bypass_only),
        n_discordant, p_value, test
    """
    agent_col = "agent_exact_match"
    bypass_col = "bypass_exact_match"
    if agent_col not in paired_df.columns or bypass_col not in paired_df.columns:
        logger.warning("McNemar: missing agent_exact_match or bypass_exact_match")
        return {}
    agent_em = pd.to_numeric(paired_df[agent_col], errors="coerce").fillna(0).astype(int)
    bypass_em = pd.to_numeric(paired_df[bypass_col], errors="coerce").fillna(0).astype(int)
    n_total = len(paired_df)
    n_both_correct = ((agent_em == 1) & (bypass_em == 1)).sum()
    n_both_wrong = ((agent_em == 0) & (bypass_em == 0)).sum()
    b = ((agent_em == 1) & (bypass_em == 0)).sum()  # Agent correct, Bypass wrong
    c = ((agent_em == 0) & (bypass_em == 1)).sum()  # Agent wrong, Bypass correct
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        result_binom = stats.binomtest(int(c), n_discordant, p=0.5, alternative="greater")
        p_value = float(result_binom.pvalue)
    return {
        "n_total": n_total,
        "n_both_correct": int(n_both_correct),
        "n_both_wrong": int(n_both_wrong),
        "b_agent_only": int(b),
        "c_bypass_only": int(c),
        "n_discordant": int(n_discordant),
        "p_value": p_value,
        "test": "McNemar_exact_binomial",
    }


def compute_selector_mcnemar_test(
    results_df: pd.DataFrame,
    metric: str = "exact_match",
) -> dict:
    """McNemar / exact sign test evaluating the Bypass selector directly.

    Treats the selector as a paired decision problem: for each sample where
    Bypass had two candidate diffs (A and B), compare the chosen diff's
    performance against the rejected diff's performance.

    - chosen  = the diff actually selected by the Bypass selector (bypass7 row)
    - rejected = the other candidate diff (base_a if chosen==B, base_b if chosen==A)

    Discordant pairs:
      b = chosen correct, rejected wrong  (selector wins)
      c = chosen wrong,   rejected correct (selector loses)

    Under H0 (selector is random), b and c are symmetric.
    One-sided exact binomial: H1 = selector wins more often than chance.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results CSV with eval_method in {agent, bypass7, base_a, base_b}
        and bypass_method in {A, B, MIX}.
    metric : str
        Performance metric column to use (default: exact_match).

    Returns
    -------
    dict
        Keys: n_total, n_both_correct, n_both_wrong, b_selector_wins,
        c_selector_loses, n_discordant, p_value, test, metric
    """
    logger.info(f"Computing selector McNemar test on metric={metric}...")

    required_cols = {"id", "eval_method", "bypass_method", metric}
    if not required_cols.issubset(results_df.columns):
        missing = required_cols - set(results_df.columns)
        logger.warning(f"Selector McNemar: missing columns {missing}")
        return {}

    def to_binary(series: pd.Series) -> pd.Series:
        return series.apply(lambda x: 1 if str(x).lower() in ["true", "1", "1.0"] else 0)

    # base_a and base_b: one row per id (deduplicated)
    base_a = (
        results_df[results_df["eval_method"] == "base_a"]
        .drop_duplicates("id")[["id", metric]]
        .rename(columns={metric: f"{metric}_a"})
    )
    base_b = (
        results_df[results_df["eval_method"] == "base_b"]
        .drop_duplicates("id")[["id", metric]]
        .rename(columns={metric: f"{metric}_b"})
    )

    # bypass7: one row per id — the chosen diff
    bypass7 = (
        results_df[results_df["eval_method"] == "bypass7"]
        .drop_duplicates("id")[["id", "bypass_method", metric]]
        .rename(columns={metric: f"{metric}_chosen", "bypass_method": "chosen_diff"})
    )

    # Merge all three
    merged = bypass7.merge(base_a, on="id", how="inner").merge(base_b, on="id", how="inner")

    # Keep only rows where selector chose A or B (drop MIX)
    merged = merged[merged["chosen_diff"].isin(["A", "B"])].copy()

    # Build rejected metric
    merged[f"{metric}_rejected"] = merged.apply(
        lambda r: r[f"{metric}_b"] if r["chosen_diff"] == "A" else r[f"{metric}_a"],
        axis=1,
    )

    # Convert to binary
    em_chosen = to_binary(merged[f"{metric}_chosen"])
    em_rejected = to_binary(merged[f"{metric}_rejected"])

    n_total = len(merged)
    n_both_correct = ((em_chosen == 1) & (em_rejected == 1)).sum()
    n_both_wrong = ((em_chosen == 0) & (em_rejected == 0)).sum()
    b = ((em_chosen == 1) & (em_rejected == 0)).sum()  # selector wins
    c = ((em_chosen == 0) & (em_rejected == 1)).sum()  # selector loses
    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
    else:
        result_binom = stats.binomtest(int(b), int(n_discordant), p=0.5, alternative="greater")
        p_value = float(result_binom.pvalue)

    logger.info(
        f"  n={n_total}, both_correct={n_both_correct}, both_wrong={n_both_wrong}, "
        f"b(selector_wins)={b}, c(selector_loses)={c}, p={p_value:.4e}"
    )

    return {
        "n_total": int(n_total),
        "n_both_correct": int(n_both_correct),
        "n_both_wrong": int(n_both_wrong),
        "b_selector_wins": int(b),
        "c_selector_loses": int(c),
        "n_discordant": int(n_discordant),
        "p_value": p_value,
        "test": "selector_McNemar_exact_binomial",
        "metric": metric,
    }


def compute_label_improvement_tests(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
    metric: str = "exact_match",
) -> pd.DataFrame:
    """Per-label Fisher's exact test on binary improve = (Bypass > Agent).

    improve = 1 if Bypass_EM > Agent_EM, else 0 (tie or Agent wins).
    For each label, tests whether the label changes P(improve) via Fisher's exact
    on the 2x2 table (label_present x improve). Reports risk difference, relative risk,
    and Haldane-Anscombe corrected odds ratio.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with agent/bypass metric columns and label columns
    config : RQ3Config
        Configuration
    metric : str
        Metric for defining improve (default: exact_match)

    Returns
    -------
    pd.DataFrame
        One row per label with fisher_pval, risk_diff, relative_risk, odds_ratio_ha, etc.
    """
    logger.info(f"Computing label improvement tests (improve = Bypass > Agent on {metric})...")
    agent_col = f"agent_{metric}"
    bypass_col = f"bypass_{metric}"
    if agent_col not in paired_df.columns or bypass_col not in paired_df.columns:
        return pd.DataFrame()
    agent_em = pd.to_numeric(paired_df[agent_col], errors="coerce").fillna(0)
    bypass_em = pd.to_numeric(paired_df[bypass_col], errors="coerce").fillna(0)
    improve = (bypass_em > agent_em).astype(int)
    rows = []
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name not in paired_df.columns:
            continue
        label_present = (pd.to_numeric(paired_df[col_name], errors="coerce").fillna(0) != 0).values
        a = int((label_present & (improve == 1)).sum())
        b = int((label_present & (improve == 0)).sum())
        c = int((~label_present & (improve == 1)).sum())
        d = int((~label_present & (improve == 0)).sum())
        n_with = a + b
        n_without = c + d
        if n_with < config.min_samples or n_without < config.min_samples:
            continue
        try:
            _, fisher_pval = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
        except Exception:
            fisher_pval = np.nan
        p1 = a / n_with if n_with > 0 else np.nan
        p0 = c / n_without if n_without > 0 else np.nan
        risk_diff = (p1 - p0) if (pd.notna(p1) and pd.notna(p0)) else np.nan
        relative_risk = (p1 / p0) if (pd.notna(p0) and p0 > 0) else np.nan
        odds_ratio_ha = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)) if (b + 0.5) * (c + 0.5) != 0 else np.nan
        se_rd = np.sqrt(p1 * (1 - p1) / n_with + p0 * (1 - p0) / n_without) if n_with > 0 and n_without > 0 and pd.notna(p1) and pd.notna(p0) else np.nan
        z = 1.96
        risk_diff_ci_low = (risk_diff - z * se_rd) if pd.notna(se_rd) and pd.notna(risk_diff) else np.nan
        risk_diff_ci_high = (risk_diff + z * se_rd) if pd.notna(se_rd) and pd.notna(risk_diff) else np.nan
        rows.append({
            "label": label,
            "display_name": config.get_label_display(label),
            "n_with_label": n_with,
            "n_without_label": n_without,
            "n_improve_with": a,
            "n_improve_without": c,
            "p_improve_with": p1,
            "p_improve_without": p0,
            "risk_diff": risk_diff,
            "risk_diff_ci_low": risk_diff_ci_low,
            "risk_diff_ci_high": risk_diff_ci_high,
            "relative_risk": relative_risk,
            "odds_ratio_ha": odds_ratio_ha,
            "fisher_pval": fisher_pval,
            "significant": pd.notna(fisher_pval) and fisher_pval < 0.05,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("risk_diff", ascending=False)
    logger.info(f"  Computed improvement tests for {len(result)} labels")
    return result


def compute_label_difficulty_interaction(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute interaction between labels and difficulty.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Interaction matrix (label x difficulty)
    """
    return compute_stratified_analysis(paired_df, "difficulty", config)


def compute_label_project_size_interaction(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute interaction between labels and project size.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Interaction matrix (label x project_size)
    """
    return compute_stratified_analysis(paired_df, "project_size", config)


def compute_label_winner_correlation(
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
    metric: str = "exact_match",
) -> pd.DataFrame:
    """Compute correlation between labels and which method wins.
    
    Analyzes whether certain labels are more common in bypass-winning 
    samples vs agent-winning samples.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with delta columns
    config : RQ3Config
        Configuration
    metric : str
        Metric to use for determining winner

    Returns
    -------
    pd.DataFrame
        Label distribution by winner type with chi-square test results
    """
    logger.info(f"Computing label-winner correlations for {metric}...")
    
    delta_col = f"delta_{metric}"
    if delta_col not in paired_df.columns:
        return pd.DataFrame()
    
    # Categorize samples by winner
    paired_df = paired_df.copy()
    paired_df["winner"] = "tie"
    paired_df.loc[paired_df[delta_col] > 0, "winner"] = "bypass"
    paired_df.loc[paired_df[delta_col] < 0, "winner"] = "agent"
    
    bypass_wins = paired_df[paired_df["winner"] == "bypass"]
    agent_wins = paired_df[paired_df["winner"] == "agent"]
    ties = paired_df[paired_df["winner"] == "tie"]
    
    n_bypass = len(bypass_wins)
    n_agent = len(agent_wins)
    n_ties = len(ties)
    
    rows = []
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name not in paired_df.columns:
            continue
        
        # Count samples with this label in each category
        bypass_with_label = bypass_wins[col_name].sum() if n_bypass > 0 else 0
        agent_with_label = agent_wins[col_name].sum() if n_agent > 0 else 0
        tie_with_label = ties[col_name].sum() if n_ties > 0 else 0
        
        # Calculate percentages
        pct_bypass = (bypass_with_label / n_bypass * 100) if n_bypass > 0 else 0
        pct_agent = (agent_with_label / n_agent * 100) if n_agent > 0 else 0
        pct_tie = (tie_with_label / n_ties * 100) if n_ties > 0 else 0
        
        # Difference: positive means bypass-winning samples have more of this label
        pct_diff_vs_agent = pct_bypass - pct_agent
        pct_diff_vs_tie = pct_bypass - pct_tie  # Also compare bypass vs tie
        
        # Chi-square test for independence (bypass wins vs agent wins)
        chi2_pval_vs_agent = np.nan
        odds_ratio_vs_agent = np.nan
        if n_bypass >= 5 and n_agent >= 5:
            try:
                # 2x2 contingency table:
                # [[with_label_bypass, without_label_bypass],
                #  [with_label_agent, without_label_agent]]
                a = bypass_with_label  # bypass wins WITH label
                b = n_bypass - bypass_with_label  # bypass wins WITHOUT label
                c = agent_with_label  # agent wins WITH label
                d = n_agent - agent_with_label  # agent wins WITHOUT label
                
                contingency = [[a, b], [c, d]]
                chi2, chi2_pval_vs_agent, dof, expected = stats.chi2_contingency(contingency)
                
                # Odds ratio: how much more likely is the label in bypass-winning samples?
                if b > 0 and c > 0:
                    odds_ratio_vs_agent = (a * d) / (b * c)
            except Exception:
                pass
        
        # Chi-square test for bypass wins vs ties (useful when agent rarely wins)
        chi2_pval_vs_tie = np.nan
        odds_ratio_vs_tie = np.nan
        if n_bypass >= 5 and n_ties >= 5:
            try:
                a = bypass_with_label  # bypass wins WITH label
                b = n_bypass - bypass_with_label  # bypass wins WITHOUT label
                c = tie_with_label  # tie WITH label
                d = n_ties - tie_with_label  # tie WITHOUT label
                
                contingency = [[a, b], [c, d]]
                chi2, chi2_pval_vs_tie, dof, expected = stats.chi2_contingency(contingency)
                
                # Odds ratio: how much more likely is label when bypass wins vs tie?
                if b > 0 and c > 0:
                    odds_ratio_vs_tie = (a * d) / (b * c)
            except Exception:
                pass
        
        row = {
            "label": label,
            "display_name": config.get_label_display(label),
            "n_bypass_wins": n_bypass,
            "n_agent_wins": n_agent,
            "n_ties": n_ties,
            "count_in_bypass_wins": int(bypass_with_label),
            "count_in_agent_wins": int(agent_with_label),
            "count_in_ties": int(tie_with_label),
            "pct_in_bypass_wins": pct_bypass,
            "pct_in_agent_wins": pct_agent,
            "pct_in_ties": pct_tie,
            "pct_diff_vs_agent": pct_diff_vs_agent,  # positive = more common in bypass wins than agent wins
            "pct_diff_vs_tie": pct_diff_vs_tie,  # positive = more common in bypass wins than ties
            "odds_ratio_vs_agent": odds_ratio_vs_agent,  # >1 = more common in bypass wins
            "odds_ratio_vs_tie": odds_ratio_vs_tie,  # >1 = more common when bypass wins than tie
            "chi2_pval_vs_agent": chi2_pval_vs_agent,
            "chi2_pval_vs_tie": chi2_pval_vs_tie,
        }
        rows.append(row)
    
    result = pd.DataFrame(rows)
    
    # Sort by absolute percentage difference (vs tie, since agent rarely wins)
    if not result.empty:
        result = result.sort_values("pct_diff_vs_tie", key=abs, ascending=False)
    
    logger.info(f"  Computed correlations for {len(result)} labels")
    return result


def generate_summary_report(
    label_summary: pd.DataFrame,
    performance_by_label: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    stratified_difficulty: pd.DataFrame,
    stratified_project_size: pd.DataFrame,
    label_winner_correlation: Optional[pd.DataFrame] = None,
    mcnemar_result: Optional[dict] = None,
    label_improvement_tests: Optional[pd.DataFrame] = None,
    selector_mcnemar_result: Optional[dict] = None,
    config: RQ3Config = DEFAULT_CONFIG,
) -> str:
    """Generate a text summary report of findings.

    Parameters
    ----------
    label_summary : pd.DataFrame
        Label distribution summary
    performance_by_label : pd.DataFrame
        Performance statistics per label
    statistical_tests : pd.DataFrame
        Statistical test results
    stratified_difficulty : pd.DataFrame
        Stratified analysis by difficulty
    stratified_project_size : pd.DataFrame
        Stratified analysis by project size
    label_winner_correlation : pd.DataFrame, optional
        Correlation between labels and winner type
    mcnemar_result : dict, optional
        Result of compute_mcnemar_test (global paired test vs Agent)
    label_improvement_tests : pd.DataFrame, optional
        Result of compute_label_improvement_tests (Fisher's exact per label)
    selector_mcnemar_result : dict, optional
        Result of compute_selector_mcnemar_test (chosen vs rejected diff)
    config : RQ3Config
        Configuration

    Returns
    -------
    str
        Markdown-formatted summary report
    """
    lines = [
        "# RQ3 Classification Analysis Summary",
        "",
        "## How to Read This Report",
        "",
        "This report compares the performance of two code conflict resolution methods:",
        "- **Agent**: Single-agent approach",
        "- **Bypass (Multi-Agent)**: Multi-agent approach with bypass mechanism",
        "",
        "**Key Metrics:**",
        "- **Delta EM (Exact Match)**: Bypass score minus Agent score. *Positive = Bypass is better*",
        "- **Delta Similarity**: Same interpretation. Range: -1 to +1",
        "- **Win Rate**: Percentage of samples where Bypass outperformed Agent",
        "- **P-value**: Statistical significance. *p < 0.05 = statistically significant difference*",
        "",
        "---",
        "",
        "## Label Distribution",
        "",
    ]
    
    if not label_summary.empty:
        total_samples = label_summary["count"].sum() if "count" in label_summary.columns else 0
        lines.append(f"Total labeled samples analyzed: **{int(label_summary.iloc[0]['count'] / (label_summary.iloc[0]['percentage']/100)) if not label_summary.empty and label_summary.iloc[0]['percentage'] > 0 else 'N/A'}**")
        lines.append("")
        lines.append("| Label | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for _, row in label_summary.head(10).iterrows():
            lines.append(f"| {row['display_name']} | {row['count']} | {row['percentage']:.1f}% |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Performance Comparison: Bypass vs Agent")
    lines.append("")
    
    if not performance_by_label.empty:
        # Calculate overall statistics
        all_deltas = []
        bypass_wins_total = 0
        agent_wins_total = 0
        total_samples = 0
        
        for _, row in performance_by_label.iterrows():
            delta_em = row.get("delta_exact_match_with", np.nan)
            n = row.get("n_with_label", 0)
            win_rate = row.get("bypass_win_rate_exact_match_with", 0)
            if pd.notna(delta_em) and n > 0:
                all_deltas.append(delta_em)
                bypass_wins_total += (win_rate / 100) * n
                agent_wins_total += ((100 - win_rate) / 100) * n
                total_samples += n
        
        if all_deltas:
            avg_delta = np.mean(all_deltas)
            
            # Overall interpretation
            lines.append("### Overall Finding")
            lines.append("")
            if avg_delta > 0.01:
                lines.append(f"**Bypass outperforms Agent** on average across all labels.")
                lines.append(f"- Average Delta EM: **+{avg_delta:.3f}** (Bypass advantage)")
            elif avg_delta < -0.01:
                lines.append(f"**Agent outperforms Bypass** on average across all labels.")
                lines.append(f"- Average Delta EM: **{avg_delta:.3f}** (Agent advantage)")
            else:
                lines.append(f"**Performance is similar** between Agent and Bypass on average.")
                lines.append(f"- Average Delta EM: **{avg_delta:.3f}**")
            lines.append("")
        
        # Find labels where bypass wins significantly
        bypass_better = []
        agent_better = []
        for _, row in performance_by_label.iterrows():
            delta_em = row.get("delta_exact_match_with", np.nan)
            if pd.notna(delta_em):
                item = {
                    "label": row["display_name"],
                    "delta": delta_em,
                    "n": row["n_with_label"],
                    "win_rate": row.get("bypass_win_rate_exact_match_with", 0),
                }
                if delta_em > 0.05:
                    bypass_better.append(item)
                elif delta_em < -0.05:
                    agent_better.append(item)
        
        if bypass_better:
            lines.append("### Labels Where Bypass Wins (Delta EM > +0.05)")
            lines.append("")
            lines.append("For samples with these labels, the **Bypass method significantly outperforms Agent**:")
            lines.append("")
            lines.append("| Label | Delta EM | Win Rate | N | Interpretation |")
            lines.append("|-------|----------|----------|---|----------------|")
            for item in sorted(bypass_better, key=lambda x: x["delta"], reverse=True)[:10]:
                strength = "Strong" if item["delta"] > 0.3 else ("Moderate" if item["delta"] > 0.15 else "Slight")
                lines.append(f"| {item['label']} | +{item['delta']:.3f} | {item['win_rate']:.1f}% | {item['n']} | {strength} Bypass advantage |")
            lines.append("")
        
        if agent_better:
            lines.append("### Labels Where Agent Wins (Delta EM < -0.05)")
            lines.append("")
            lines.append("For samples with these labels, the **Agent method outperforms Bypass**:")
            lines.append("")
            lines.append("| Label | Delta EM | Win Rate | N | Interpretation |")
            lines.append("|-------|----------|----------|---|----------------|")
            for item in sorted(agent_better, key=lambda x: x["delta"])[:10]:
                strength = "Strong" if item["delta"] < -0.3 else ("Moderate" if item["delta"] < -0.15 else "Slight")
                lines.append(f"| {item['label']} | {item['delta']:.3f} | {item['win_rate']:.1f}% | {item['n']} | {strength} Agent advantage |")
            lines.append("")
        
        if not bypass_better and not agent_better:
            lines.append("### No Strong Method Preference")
            lines.append("")
            lines.append("No labels showed a strong preference (|Delta EM| > 0.05) for either method.")
            lines.append("")
    
    # Global Method Comparison (McNemar)
    if mcnemar_result:
        lines.append("---")
        lines.append("")
        lines.append("## Global Method Comparison (McNemar Test)")
        lines.append("")
        lines.append("Paired test on exact match: we only use *discordant* pairs (one method correct, the other wrong).")
        lines.append("Under the null that methods are equivalent, the number of pairs where Bypass is correct and Agent is wrong")
        lines.append("should be symmetric with the reverse. One-sided exact binomial test: **Bypass is better than Agent on EM**.")
        lines.append("")
        n_tot = mcnemar_result.get("n_total", 0)
        b_val = mcnemar_result.get("b_agent_only", 0)
        c_val = mcnemar_result.get("c_bypass_only", 0)
        n_disc = mcnemar_result.get("n_discordant", 0)
        p_val = mcnemar_result.get("p_value", np.nan)
        lines.append("| Quantity | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| N (total pairs) | {n_tot} |")
        lines.append(f"| Agent correct, Bypass wrong (b) | {b_val} |")
        lines.append(f"| Agent wrong, Bypass correct (c) | {c_val} |")
        lines.append(f"| Discordant pairs (b + c) | {n_disc} |")
        pval_str = f"{p_val:.2e}" if pd.notna(p_val) and p_val < 0.001 else (f"{p_val:.4f}" if pd.notna(p_val) else "N/A")
        lines.append(f"| P-value (exact binomial, one-sided) | {pval_str} |")
        lines.append("")
        if pd.notna(p_val) and p_val < 0.05:
            lines.append("**Conclusion:** Bypass is significantly better than Agent on exact match (p < 0.05).")
        else:
            lines.append("**Conclusion:** The global paired test does not show a significant advantage for Bypass at α = 0.05.")
        lines.append("")

    # Selector McNemar section
    if selector_mcnemar_result:
        lines.append("---")
        lines.append("")
        lines.append("## Selector Quality: Chosen vs Rejected Diff (McNemar Test)")
        lines.append("")
        lines.append("This test evaluates the Bypass *selector* directly, independent of the single-agent baseline.")
        lines.append("For each sample, Bypass produces two candidate diffs (A and B) and a selector picks one.")
        lines.append("We compare the **chosen** diff's exact match against the **rejected** diff's exact match.")
        lines.append("")
        lines.append("Discordant pairs:")
        lines.append("- **b** = chosen correct, rejected wrong (selector picks the better diff)")
        lines.append("- **c** = chosen wrong, rejected correct (selector picks the worse diff)")
        lines.append("")
        lines.append("Under H0 (selector is random), b and c are symmetric.")
        lines.append("One-sided exact binomial: H1 = selector wins more often than chance.")
        lines.append("")
        n_tot = selector_mcnemar_result.get("n_total", 0)
        b_val = selector_mcnemar_result.get("b_selector_wins", 0)
        c_val = selector_mcnemar_result.get("c_selector_loses", 0)
        n_disc = selector_mcnemar_result.get("n_discordant", 0)
        p_val = selector_mcnemar_result.get("p_value", np.nan)
        met = selector_mcnemar_result.get("metric", "exact_match")
        lines.append(f"**Metric:** {met}")
        lines.append("")
        lines.append("| Quantity | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| N (samples with both candidates) | {n_tot} |")
        lines.append(f"| Both correct | {selector_mcnemar_result.get('n_both_correct', 0)} |")
        lines.append(f"| Both wrong | {selector_mcnemar_result.get('n_both_wrong', 0)} |")
        lines.append(f"| b: chosen correct, rejected wrong (selector wins) | {b_val} |")
        lines.append(f"| c: chosen wrong, rejected correct (selector loses) | {c_val} |")
        lines.append(f"| Discordant pairs (b + c) | {n_disc} |")
        pval_str = f"{p_val:.2e}" if pd.notna(p_val) and p_val < 0.001 else (f"{p_val:.4f}" if pd.notna(p_val) else "N/A")
        lines.append(f"| P-value (exact binomial, one-sided H1: b > c) | {pval_str} |")
        lines.append("")
        if pd.notna(p_val) and p_val < 0.05:
            lines.append(f"**Conclusion:** The selector picks the better diff significantly more often than chance (p < 0.05). b={b_val} vs c={c_val}.")
        elif pd.notna(p_val) and p_val < 0.1:
            lines.append(f"**Conclusion:** Marginal evidence that the selector picks the better diff more often than chance (p < 0.10). b={b_val} vs c={c_val}.")
        else:
            lines.append(f"**Conclusion:** No significant evidence that the selector picks the better diff more often than chance (p={pval_str}). b={b_val} vs c={c_val}.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Statistical Significance")
    lines.append("")
    lines.append("Statistical tests compare the performance *difference* between samples with vs without each label.")
    lines.append("A significant result (p < 0.05) means the label is associated with different method effectiveness.")
    lines.append("")
    
    if not statistical_tests.empty:
        # Find significant p-values with interpretation
        sig_tests = []
        for _, row in statistical_tests.iterrows():
            for metric in config.metrics:
                pval_col = f"ttest_pval_{metric}"
                if pval_col in row and pd.notna(row[pval_col]) and row[pval_col] < 0.05:
                    sig_tests.append({
                        "label": row["display_name"],
                        "metric": metric,
                        "pval": row[pval_col],
                    })
        
        if sig_tests:
            lines.append("### Statistically Significant Results (p < 0.05)")
            lines.append("")
            lines.append("| Label | Metric | P-value | Significance Level |")
            lines.append("|-------|--------|---------|-------------------|")
            for item in sorted(sig_tests, key=lambda x: x["pval"])[:15]:
                if item["pval"] < 0.001:
                    sig_level = "Highly significant (p < 0.001)"
                elif item["pval"] < 0.01:
                    sig_level = "Very significant (p < 0.01)"
                else:
                    sig_level = "Significant (p < 0.05)"
                lines.append(f"| {item['label']} | {item['metric']} | {item['pval']:.4f} | {sig_level} |")
            lines.append("")
            
            lines.append("**Interpretation:** These labels show statistically significant differences in how")
            lines.append("the two methods (Agent vs Bypass) perform. This suggests these labels are")
            lines.append("predictive of which method will work better for a given sample.")
            lines.append("")
        else:
            lines.append("### No Statistically Significant Results")
            lines.append("")
            lines.append("No labels showed statistically significant differences (p < 0.05) between methods.")
            lines.append("This could mean:")
            lines.append("- The sample size is too small to detect differences")
            lines.append("- The labels are not predictive of method performance")
            lines.append("- Both methods perform similarly regardless of label")
            lines.append("")
    
    # Mann-Whitney U Test section
    if not statistical_tests.empty:
        lines.append("### Mann-Whitney U Test")
        lines.append("")
        lines.append("The Mann-Whitney U test is a non-parametric alternative to the T-test. It compares")
        lines.append("the *distribution* of performance deltas (Bypass − Agent) between samples with vs")
        lines.append("without each label, without assuming normality. We report it alongside the T-test")
        lines.append("because performance metrics may not be normally distributed.")
        lines.append("")
        sig_mw = []
        for _, row in statistical_tests.iterrows():
            for metric in config.metrics:
                pval_col = f"mannwhitney_pval_{metric}"
                if pval_col in row and pd.notna(row[pval_col]) and row[pval_col] < 0.05:
                    sig_mw.append({
                        "label": row["display_name"],
                        "metric": metric,
                        "pval": row[pval_col],
                    })
        if sig_mw:
            lines.append("**Statistically significant Mann-Whitney results (p < 0.05):**")
            lines.append("")
            lines.append("| Label | Metric | P-value | Significance Level |")
            lines.append("|-------|--------|---------|-------------------|")
            for item in sorted(sig_mw, key=lambda x: x["pval"])[:15]:
                if item["pval"] < 0.001:
                    sig_level = "Highly significant (p < 0.001)"
                elif item["pval"] < 0.01:
                    sig_level = "Very significant (p < 0.01)"
                else:
                    sig_level = "Significant (p < 0.05)"
                pval_str = f"{item['pval']:.2e}" if item["pval"] < 0.001 else f"{item['pval']:.4f}"
                lines.append(f"| {item['label']} | {item['metric']} | {pval_str} | {sig_level} |")
            lines.append("")
        else:
            lines.append("No Mann-Whitney results reached statistical significance (p < 0.05).")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Stratified Analysis")
    lines.append("")
    lines.append("Performance breakdown by task characteristics:")
    lines.append("")
    
    if not stratified_difficulty.empty:
        lines.append("### By Difficulty Level")
        lines.append("")
        lines.append("How does method performance vary across difficulty levels for each label?")
        lines.append("")
        lines.append("See `stratified_difficulty.csv` for the full breakdown. Key columns:")
        lines.append("- `delta_exact_match`: Bypass - Agent score (positive = Bypass better)")
        lines.append("- `bypass_win_rate_exact_match`: % of samples where Bypass won")
        lines.append("")
    
    if not stratified_project_size.empty:
        lines.append("### By Project Size")
        lines.append("")
        lines.append("How does method performance vary across project sizes for each label?")
        lines.append("")
        lines.append("See `stratified_project_size.csv` for the full breakdown.")
        lines.append("")
    
    # Label-Winner Correlation Section
    lines.append("---")
    lines.append("")
    lines.append("## Label Distribution by Winner")
    lines.append("")
    lines.append("Which labels are more common when Bypass wins vs when methods tie (or Agent wins)?")
    lines.append("This reveals what characteristics predict Bypass success.")
    lines.append("")
    lines.append("**Metric:** Winner is determined by **exact match** (Bypass wins if Bypass EM > Agent EM; tie if equal).")
    lines.append("The Chi-squared tests assess whether each label is more/less prevalent in Bypass-winning samples vs Tie samples.")
    lines.append("")
    
    if label_winner_correlation is not None and not label_winner_correlation.empty:
        # Get winner counts
        first_row = label_winner_correlation.iloc[0]
        n_bypass = int(first_row.get("n_bypass_wins", 0))
        n_agent = int(first_row.get("n_agent_wins", 0))
        n_ties = int(first_row.get("n_ties", 0))
        
        lines.append(f"**Sample breakdown:** Bypass wins: {n_bypass} | Agent wins: {n_agent} | Ties: {n_ties}")
        lines.append("")
        
        # Determine which comparison to use
        # If agent rarely wins, compare bypass wins vs ties
        use_tie_comparison = n_agent < 5
        
        if use_tie_comparison:
            lines.append("*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*")
            lines.append("")
            
            # Find labels more/less common in bypass-winning samples vs ties
            bypass_favored = []
            tie_favored = []
            for _, row in label_winner_correlation.iterrows():
                pct_diff = row.get("pct_diff_vs_tie", 0)
                chi2_pval = row.get("chi2_pval_vs_tie", np.nan)
                if abs(pct_diff) >= 5:  # At least 5 percentage point difference
                    item = {
                        "label": row["display_name"],
                        "pct_bypass": row.get("pct_in_bypass_wins", 0),
                        "pct_tie": row.get("pct_in_ties", 0),
                        "pct_diff": pct_diff,
                        "odds_ratio": row.get("odds_ratio_vs_tie", np.nan),
                        "pval": chi2_pval,
                        "significant": pd.notna(chi2_pval) and chi2_pval < 0.05,
                    }
                    if pct_diff > 0:
                        bypass_favored.append(item)
                    else:
                        tie_favored.append(item)
            
            if bypass_favored:
                lines.append("### Labels More Common When Bypass Wins (vs Tie)")
                lines.append("")
                lines.append("These labels appear more frequently when Bypass outperforms Agent than when they tie:")
                lines.append("")
                lines.append("| Label | % in Bypass Wins | % in Ties | Difference | P-value | Significant? |")
                lines.append("|-------|------------------|-----------|------------|--------|--------------|")
                for item in sorted(bypass_favored, key=lambda x: x["pct_diff"], reverse=True):
                    sig_marker = "Yes (p<0.05)" if item["significant"] else "No"
                    pval = item["pval"]
                    pval_str = "N/A" if pd.isna(pval) else (f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}")
                    lines.append(f"| {item['label']} | {item['pct_bypass']:.1f}% | {item['pct_tie']:.1f}% | +{item['pct_diff']:.1f}pp | {pval_str} | {sig_marker} |")
                lines.append("")
                
                # Interpretation
                sig_items = [x for x in bypass_favored if x["significant"]]
                if sig_items:
                    labels_str = ", ".join([x["label"] for x in sig_items[:3]])
                    lines.append(f"**Interpretation:** Labels like *{labels_str}* are significantly more common")
                    lines.append("when Bypass wins, suggesting these characteristics predict Bypass success.")
                    lines.append("")
            
            if tie_favored:
                lines.append("### Labels More Common When Methods Tie (vs Bypass Win)")
                lines.append("")
                lines.append("These labels appear more frequently when methods tie than when Bypass wins:")
                lines.append("")
                lines.append("| Label | % in Bypass Wins | % in Ties | Difference | P-value | Significant? |")
                lines.append("|-------|------------------|-----------|------------|--------|--------------|")
                for item in sorted(tie_favored, key=lambda x: x["pct_diff"]):
                    sig_marker = "Yes (p<0.05)" if item["significant"] else "No"
                    pval = item["pval"]
                    pval_str = "N/A" if pd.isna(pval) else (f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}")
                    lines.append(f"| {item['label']} | {item['pct_bypass']:.1f}% | {item['pct_tie']:.1f}% | {item['pct_diff']:.1f}pp | {pval_str} | {sig_marker} |")
                lines.append("")
                
                # Interpretation
                sig_items = [x for x in tie_favored if x["significant"]]
                if sig_items:
                    labels_str = ", ".join([x["label"] for x in sig_items[:3]])
                    lines.append(f"**Interpretation:** Labels like *{labels_str}* are significantly more common")
                    lines.append("when methods tie, suggesting Bypass provides no advantage for these cases.")
                    lines.append("")
            
            if not bypass_favored and not tie_favored:
                lines.append("No labels showed a meaningful difference (>=5pp) between Bypass wins and ties.")
                lines.append("")
        
        else:
            # Original comparison: bypass wins vs agent wins
            bypass_favored = []
            agent_favored = []
            for _, row in label_winner_correlation.iterrows():
                pct_diff = row.get("pct_diff_vs_agent", 0)
                chi2_pval = row.get("chi2_pval_vs_agent", np.nan)
                if abs(pct_diff) >= 5:  # At least 5 percentage point difference
                    item = {
                        "label": row["display_name"],
                        "pct_bypass": row.get("pct_in_bypass_wins", 0),
                        "pct_agent": row.get("pct_in_agent_wins", 0),
                        "pct_diff": pct_diff,
                        "odds_ratio": row.get("odds_ratio_vs_agent", np.nan),
                        "pval": chi2_pval,
                        "significant": pd.notna(chi2_pval) and chi2_pval < 0.05,
                    }
                    if pct_diff > 0:
                        bypass_favored.append(item)
                    else:
                        agent_favored.append(item)
            
            if bypass_favored:
                lines.append("### Labels More Common When Bypass Wins")
                lines.append("")
                lines.append("These labels appear more frequently when Bypass outperforms Agent:")
                lines.append("")
                lines.append("| Label | % in Bypass Wins | % in Agent Wins | Difference | P-value | Significant? |")
                lines.append("|-------|------------------|-----------------|------------|--------|--------------|")
                for item in sorted(bypass_favored, key=lambda x: x["pct_diff"], reverse=True):
                    sig_marker = "Yes (p<0.05)" if item["significant"] else "No"
                    pval = item["pval"]
                    pval_str = "N/A" if pd.isna(pval) else (f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}")
                    lines.append(f"| {item['label']} | {item['pct_bypass']:.1f}% | {item['pct_agent']:.1f}% | +{item['pct_diff']:.1f}pp | {pval_str} | {sig_marker} |")
                lines.append("")
                
                sig_items = [x for x in bypass_favored if x["significant"]]
                if sig_items:
                    labels_str = ", ".join([x["label"] for x in sig_items[:3]])
                    lines.append(f"**Interpretation:** Labels like *{labels_str}* are significantly more common")
                    lines.append("in Bypass-winning samples, suggesting these favor the multi-agent approach.")
                    lines.append("")
            
            if agent_favored:
                lines.append("### Labels More Common When Agent Wins")
                lines.append("")
                lines.append("These labels appear more frequently when Agent outperforms Bypass:")
                lines.append("")
                lines.append("| Label | % in Bypass Wins | % in Agent Wins | Difference | P-value | Significant? |")
                lines.append("|-------|------------------|-----------------|------------|--------|--------------|")
                for item in sorted(agent_favored, key=lambda x: x["pct_diff"]):
                    sig_marker = "Yes (p<0.05)" if item["significant"] else "No"
                    pval = item["pval"]
                    pval_str = "N/A" if pd.isna(pval) else (f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}")
                    lines.append(f"| {item['label']} | {item['pct_bypass']:.1f}% | {item['pct_agent']:.1f}% | {item['pct_diff']:.1f}pp | {pval_str} | {sig_marker} |")
                lines.append("")
                
                sig_items = [x for x in agent_favored if x["significant"]]
                if sig_items:
                    labels_str = ", ".join([x["label"] for x in sig_items[:3]])
                    lines.append(f"**Interpretation:** Labels like *{labels_str}* are significantly more common")
                    lines.append("in Agent-winning samples, suggesting these favor the single-agent approach.")
                    lines.append("")
            
            if not bypass_favored and not agent_favored:
                lines.append("No labels showed a meaningful difference (>=5pp) between winner groups.")
                lines.append("")
        
        lines.append("See `label_winner_correlation.csv` for the full breakdown including odds ratios and p-values.")
        lines.append("")
    else:
        lines.append("Label-winner correlation analysis not available.")
        lines.append("")
    
    # Per-Label Improvement Analysis (Fisher's exact on improve = Bypass > Agent)
    if label_improvement_tests is not None and not label_improvement_tests.empty:
        lines.append("---")
        lines.append("")
        lines.append("## Per-Label Improvement Analysis (Fisher's Exact)")
        lines.append("")
        lines.append("For each label we test whether the label changes the *probability of improvement* (Bypass EM > Agent EM).")
        lines.append("**improve** = 1 if Bypass wins, 0 otherwise (tie or Agent wins). Fisher's exact test on the 2×2 table")
        lines.append("(label present × improve). Effect sizes: risk difference, relative risk, and Haldane-Anscombe corrected odds ratio.")
        lines.append("")
        lines.append("| Label | P(improve given label) | P(improve given no label) | Risk Diff | Rel. Risk | OR (HA) | Fisher p | Sig? |")
        lines.append("|-------|--------------------|-------------------------|------------|-----------|---------|---------|------|")
        for _, row in label_improvement_tests.iterrows():
            p1 = row.get("p_improve_with", np.nan)
            p0 = row.get("p_improve_without", np.nan)
            rd = row.get("risk_diff", np.nan)
            rr = row.get("relative_risk", np.nan)
            or_ha = row.get("odds_ratio_ha", np.nan)
            fp = row.get("fisher_pval", np.nan)
            sig = "Yes" if (pd.notna(fp) and fp < 0.05) else "No"
            p1_s = f"{p1:.3f}" if pd.notna(p1) else "—"
            p0_s = f"{p0:.3f}" if pd.notna(p0) else "—"
            rd_s = f"{rd:+.3f}" if pd.notna(rd) else "—"
            rr_s = f"{rr:.2f}" if pd.notna(rr) else "—"
            or_s = f"{or_ha:.2f}" if pd.notna(or_ha) else "—"
            fp_s = f"{fp:.2e}" if pd.notna(fp) and fp < 0.001 else (f"{fp:.4f}" if pd.notna(fp) else "—")
            lines.append(f"| {row.get('display_name', row.get('label', ''))} | {p1_s} | {p0_s} | {rd_s} | {rr_s} | {or_s} | {fp_s} | {sig} |")
        lines.append("")
        lines.append("See `label_improvement_tests.csv` for full counts and `rq3_label_improvement_forest.png` for a forest plot of risk differences.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Glossary")
    lines.append("")
    lines.append("| Term | Definition |")
    lines.append("|------|------------|")
    lines.append("| Delta | Difference between Bypass and Agent scores (Bypass - Agent) |")
    lines.append("| Exact Match (EM) | Binary metric: 1 if output exactly matches ground truth, 0 otherwise |")
    lines.append("| Similarity | Continuous metric (0-1) measuring how similar output is to ground truth |")
    lines.append("| Win Rate | Percentage of samples where one method outperformed the other |")
    lines.append("| P-value | Probability of observing this difference by chance (lower = more significant) |")
    lines.append("| T-test | Parametric test comparing means of two groups (assumes normality) |")
    lines.append("| Mann-Whitney U | Non-parametric test comparing distributions of two groups (no normality assumption) |")
    lines.append("| Chi-squared | Test for association between categorical variables (e.g., label prevalence vs winner type) |")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Complexity Correlation Functions
# =============================================================================

def compute_complexity_performance_correlation(
    complexity_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute correlation between complexity metrics and performance.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Complexity metrics DataFrame with sample_id, method columns
    results_df : pd.DataFrame
        Results DataFrame with performance metrics
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Correlation coefficients and p-values
    """
    logger.info("Computing complexity-performance correlations...")
    
    if complexity_df.empty or results_df.empty:
        return pd.DataFrame()
    
    # Complexity metrics to correlate
    complexity_cols = [
        "sloc", "lloc", "cc_total", "cc_avg", "cc_max", "cc_count",
        "h_difficulty", "h_effort", "h_bugs", "mi_score",
    ]
    complexity_cols = [c for c in complexity_cols if c in complexity_df.columns]
    
    # Performance metrics
    perf_cols = [m for m in config.metrics if m in results_df.columns]
    
    rows = []
    
    for method in ["agent", "bypass"]:
        # Get method-specific data
        method_complexity = complexity_df[complexity_df["method"] == method].copy()
        
        if method_complexity.empty:
            continue
        
        # Merge with results
        # Filter results to this method
        if "eval_method" in results_df.columns:
            if method == "bypass":
                method_results = results_df[results_df["eval_method"].str.contains("bypass", case=False, na=False)].copy()
            else:
                method_results = results_df[results_df["eval_method"] == "agent"].copy()
        else:
            method_results = results_df.copy()
        
        # Convert IDs to string for consistent merging
        method_complexity["sample_id"] = method_complexity["sample_id"].astype(str)
        method_results["id"] = method_results["id"].astype(str)
        
        # Merge on sample_id / id
        merged = method_complexity.merge(
            method_results,
            left_on="sample_id",
            right_on="id",
            how="inner",
        )
        
        if len(merged) < 10:
            continue
        
        for comp_col in complexity_cols:
            for perf_col in perf_cols:
                try:
                    # Get valid values
                    valid = merged[[comp_col, perf_col]].dropna()
                    if len(valid) < 10:
                        continue
                    
                    x = valid[comp_col].values
                    y = valid[perf_col].values
                    
                    # Pearson correlation
                    pearson_r, pearson_p = stats.pearsonr(x, y)
                    
                    # Spearman correlation (more robust)
                    spearman_r, spearman_p = stats.spearmanr(x, y)
                    
                    rows.append({
                        "method": method,
                        "complexity_metric": comp_col,
                        "performance_metric": perf_col,
                        "n_samples": len(valid),
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                    })
                except Exception as e:
                    logger.debug(f"Error computing correlation {comp_col} vs {perf_col}: {e}")
    
    result = pd.DataFrame(rows)
    
    if not result.empty:
        # Sort by absolute correlation
        result["abs_spearman"] = result["spearman_r"].abs()
        result = result.sort_values("abs_spearman", ascending=False)
        result = result.drop(columns=["abs_spearman"])
    
    logger.info(f"  Computed {len(result)} correlation pairs")
    return result


def compute_complexity_by_label(
    complexity_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute average complexity metrics per label.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Complexity metrics DataFrame
    aggregate_df : pd.DataFrame
        Aggregate DataFrame with label columns
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Mean complexity per label
    """
    logger.info("Computing complexity by label...")
    
    if complexity_df.empty or aggregate_df.empty:
        return pd.DataFrame()
    
    # Get ground truth complexity only
    gt_complexity = complexity_df[complexity_df["method"] == "ground_truth"].copy()
    
    if gt_complexity.empty:
        return pd.DataFrame()
    
    # Merge with aggregate to get labels
    merged = gt_complexity.merge(
        aggregate_df,
        on="sample_id",
        how="inner",
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Complexity metrics to analyze
    complexity_cols = [
        "sloc", "lloc", "cc_total", "cc_avg", "mi_score",
    ]
    complexity_cols = [c for c in complexity_cols if c in merged.columns]
    
    rows = []
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name not in merged.columns:
            continue
        
        with_label = merged[merged[col_name] == 1]
        without_label = merged[merged[col_name] == 0]
        
        if len(with_label) < 5 or len(without_label) < 5:
            continue
        
        row = {
            "label": label,
            "display_name": config.get_label_display(label),
            "n_with_label": len(with_label),
            "n_without_label": len(without_label),
        }
        
        for col in complexity_cols:
            with_mean = with_label[col].mean()
            without_mean = without_label[col].mean()
            diff = with_mean - without_mean
            
            row[f"{col}_with"] = with_mean
            row[f"{col}_without"] = without_mean
            row[f"{col}_diff"] = diff
            
            # T-test
            try:
                t_stat, p_val = stats.ttest_ind(
                    with_label[col].dropna(),
                    without_label[col].dropna(),
                    equal_var=False,
                )
                row[f"{col}_pval"] = p_val
            except Exception:
                row[f"{col}_pval"] = np.nan
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    logger.info(f"  Computed complexity for {len(result)} labels")
    return result


def compute_gt_complexity_vs_performance(
    complexity_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Analyze if ground truth complexity predicts method performance.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Complexity metrics DataFrame
    paired_df : pd.DataFrame
        Paired performance data (agent vs bypass)
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Correlation between GT complexity and method performance delta
    """
    logger.info("Computing GT complexity vs performance delta...")
    
    if complexity_df.empty or paired_df.empty:
        return pd.DataFrame()
    
    # Get ground truth complexity
    gt_complexity = complexity_df[complexity_df["method"] == "ground_truth"].copy()
    
    if gt_complexity.empty:
        return pd.DataFrame()
    
    # Merge with paired data
    merged = gt_complexity.merge(
        paired_df,
        on="sample_id",
        how="inner",
    )
    
    if len(merged) < 10:
        return pd.DataFrame()
    
    complexity_cols = ["sloc", "lloc", "cc_total", "cc_avg", "mi_score"]
    complexity_cols = [c for c in complexity_cols if c in merged.columns]
    
    rows = []
    for comp_col in complexity_cols:
        for metric in config.metrics:
            delta_col = f"delta_{metric}"
            if delta_col not in merged.columns:
                continue
            
            try:
                valid = merged[[comp_col, delta_col]].dropna()
                if len(valid) < 10:
                    continue
                
                x = valid[comp_col].values
                y = valid[delta_col].values
                
                spearman_r, spearman_p = stats.spearmanr(x, y)
                
                rows.append({
                    "gt_complexity_metric": comp_col,
                    "performance_delta": delta_col,
                    "n_samples": len(valid),
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "interpretation": (
                        "Higher GT complexity → Bypass better" if spearman_r > 0.1
                        else "Higher GT complexity → Agent better" if spearman_r < -0.1
                        else "No clear relationship"
                    ),
                })
            except Exception as e:
                logger.debug(f"Error: {e}")
    
    result = pd.DataFrame(rows)
    logger.info(f"  Computed {len(result)} GT complexity correlations")
    return result
