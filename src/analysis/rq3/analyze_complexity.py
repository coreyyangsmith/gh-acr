"""Comprehensive complexity analysis for RQ3.

Aggregates complexity data across all sources, generates plots and tables,
and creates interpretive findings for the summary report.

Usage:
    python -m src.analysis.rq3.analyze_complexity --output-dir results/rq3
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class ComplexityAnalysisConfig:
    """Configuration for complexity analysis."""
    
    output_dir: Path = Path("results/rq3")
    figsize: tuple[int, int] = (12, 8)
    dpi: int = 150
    
    # Key metrics to analyze
    complexity_metrics: list[str] = None
    
    def __post_init__(self):
        if self.complexity_metrics is None:
            self.complexity_metrics = [
                "sloc", "lloc", "cc_total", "cc_avg", "cc_max", 
                "mi_score", "h_difficulty", "h_bugs"
            ]


def load_all_complexity_data(output_dir: Path) -> pd.DataFrame:
    """Load and combine all complexity metrics from subfolders."""
    all_dfs = []
    
    for subfolder in output_dir.iterdir():
        if subfolder.is_dir():
            metrics_file = subfolder / "complexity_metrics.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                df["model_source"] = subfolder.name
                all_dfs.append(df)
                print(f"  Loaded {len(df)} rows from {subfolder.name}")
    
    if not all_dfs:
        print("  No complexity data found!")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows from {len(all_dfs)} sources")
    return combined


def compute_method_statistics(df: pd.DataFrame, config: ComplexityAnalysisConfig) -> pd.DataFrame:
    """Compute aggregate statistics by method."""
    if df.empty:
        return pd.DataFrame()
    
    # Filter valid data
    valid_df = df[~df["parse_error"]].copy()
    
    methods = ["a_only", "b_only", "ground_truth", "agent", "bypass"]
    methods = [m for m in methods if m in valid_df["method"].unique()]
    
    rows = []
    for method in methods:
        method_data = valid_df[valid_df["method"] == method]
        row = {"method": method, "n_samples": len(method_data)}
        
        for metric in config.complexity_metrics:
            if metric in method_data.columns:
                values = method_data[metric].dropna()
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
                row[f"{metric}_median"] = values.median()
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_method_comparison(df: pd.DataFrame, config: ComplexityAnalysisConfig) -> pd.DataFrame:
    """Compare agent and bypass complexity to ground truth."""
    if df.empty:
        return pd.DataFrame()
    
    valid_df = df[~df["parse_error"]].copy()
    
    rows = []
    for sample_id in valid_df["sample_id"].unique():
        sample_data = valid_df[valid_df["sample_id"] == sample_id]
        
        gt_data = sample_data[sample_data["method"] == "ground_truth"]
        agent_data = sample_data[sample_data["method"] == "agent"]
        bypass_data = sample_data[sample_data["method"] == "bypass"]
        
        if gt_data.empty or (agent_data.empty and bypass_data.empty):
            continue
        
        row = {"sample_id": sample_id}
        
        for metric in config.complexity_metrics:
            if metric not in gt_data.columns:
                continue
            
            gt_val = gt_data[metric].iloc[0] if not gt_data.empty else np.nan
            agent_val = agent_data[metric].iloc[0] if not agent_data.empty else np.nan
            bypass_val = bypass_data[metric].iloc[0] if not bypass_data.empty else np.nan
            
            row[f"gt_{metric}"] = gt_val
            row[f"agent_{metric}"] = agent_val
            row[f"bypass_{metric}"] = bypass_val
            row[f"agent_diff_{metric}"] = agent_val - gt_val if pd.notna(agent_val) and pd.notna(gt_val) else np.nan
            row[f"bypass_diff_{metric}"] = bypass_val - gt_val if pd.notna(bypass_val) and pd.notna(gt_val) else np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_complexity_comparison(df: pd.DataFrame, config: ComplexityAnalysisConfig) -> dict[str, Path]:
    """Generate complexity comparison plots."""
    outputs = {}
    valid_df = df[~df["parse_error"]].copy()
    
    if valid_df.empty:
        return outputs
    
    methods = ["a_only", "b_only", "ground_truth", "agent", "bypass"]
    methods = [m for m in methods if m in valid_df["method"].unique()]
    
    palette = {
        "a_only": "#7fcdbb",
        "b_only": "#41b6c4", 
        "ground_truth": "#2c7fb8",
        "agent": "#d95f02",
        "bypass": "#1b9e77",
    }
    
    # 1. Multi-metric boxplot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    key_metrics = ["sloc", "cc_avg", "cc_max", "mi_score", "h_difficulty", "h_bugs"]
    metric_labels = {
        "sloc": "Source Lines of Code",
        "cc_avg": "Avg Cyclomatic Complexity",
        "cc_max": "Max Cyclomatic Complexity",
        "mi_score": "Maintainability Index",
        "h_difficulty": "Halstead Difficulty",
        "h_bugs": "Estimated Bugs",
    }
    
    for idx, metric in enumerate(key_metrics):
        if metric not in valid_df.columns:
            continue
        ax = axes[idx]
        sns.boxplot(
            data=valid_df,
            x="method",
            y=metric,
            hue="method",
            order=methods,
            hue_order=methods,
            palette=palette,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Code Complexity Metrics by Method", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_all_metrics_comparison.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["all_metrics_comparison"] = plot_path
    print(f"  Saved: {plot_path}")
    
    # 2. Agent vs Bypass vs Ground Truth comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Filter to just agent, bypass, ground_truth
    compare_df = valid_df[valid_df["method"].isin(["agent", "bypass", "ground_truth"])]
    compare_methods = ["ground_truth", "agent", "bypass"]
    
    for idx, metric in enumerate(["cc_avg", "mi_score", "sloc"]):
        ax = axes[idx]
        if metric in compare_df.columns:
            sns.violinplot(
                data=compare_df,
                x="method",
                y=metric,
                hue="method",
                order=compare_methods,
                hue_order=compare_methods,
                palette={m: palette[m] for m in compare_methods},
                ax=ax,
                legend=False,
            )
        ax.set_xlabel("")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(f"{metric_labels.get(metric, metric)}")
    
    plt.suptitle("Complexity: Agent vs Bypass vs Ground Truth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_agent_bypass_gt.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["agent_bypass_gt"] = plot_path
    print(f"  Saved: {plot_path}")
    
    # 3. Maintainability Index distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        method_data = valid_df[valid_df["method"] == method]["mi_score"]
        if len(method_data) > 0:
            sns.kdeplot(
                data=method_data,
                ax=ax,
                label=method,
                color=palette.get(method, "#333333"),
                linewidth=2,
            )
    
    ax.axvline(x=20, color="green", linestyle="--", alpha=0.7, label="MI Grade A (≥20)")
    ax.axvline(x=10, color="orange", linestyle="--", alpha=0.7, label="MI Grade B (≥10)")
    
    ax.set_xlabel("Maintainability Index", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Maintainability Index Distribution by Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_mi_distribution.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["mi_distribution"] = plot_path
    print(f"  Saved: {plot_path}")
    
    return outputs


def compute_complexity_performance_correlation(
    complexity_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    config: ComplexityAnalysisConfig,
) -> pd.DataFrame:
    """Correlate ground truth complexity with method performance."""
    if complexity_df.empty or paired_df.empty:
        return pd.DataFrame()
    
    # Get ground truth complexity
    gt_complexity = complexity_df[complexity_df["method"] == "ground_truth"].copy()
    
    if gt_complexity.empty:
        return pd.DataFrame()
    
    # Convert sample_id to string for merge
    gt_complexity["sample_id"] = gt_complexity["sample_id"].astype(str)
    
    # Handle different column names for sample ID
    paired_df = paired_df.copy()
    if "id" in paired_df.columns and "sample_id" not in paired_df.columns:
        paired_df["sample_id"] = paired_df["id"].astype(str)
    elif "sample_id" in paired_df.columns:
        paired_df["sample_id"] = paired_df["sample_id"].astype(str)
    else:
        return pd.DataFrame()
    
    merged = gt_complexity.merge(paired_df, on="sample_id", how="inner")
    
    if len(merged) < 10:
        return pd.DataFrame()
    
    rows = []
    perf_metrics = ["delta_exact_match", "delta_similarity", "delta_bleu3"]
    
    for comp_metric in config.complexity_metrics:
        if comp_metric not in merged.columns:
            continue
        
        for perf_metric in perf_metrics:
            if perf_metric not in merged.columns:
                continue
            
            valid = merged[[comp_metric, perf_metric]].dropna()
            if len(valid) < 10:
                continue
            
            try:
                spearman_r, spearman_p = stats.spearmanr(valid[comp_metric], valid[perf_metric])
                pearson_r, pearson_p = stats.pearsonr(valid[comp_metric], valid[perf_metric])
                
                rows.append({
                    "complexity_metric": comp_metric,
                    "performance_metric": perf_metric,
                    "n_samples": len(valid),
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                })
            except Exception:
                pass
    
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_spearman"] = result["spearman_r"].abs()
        result = result.sort_values("abs_spearman", ascending=False)
        result = result.drop(columns=["abs_spearman"])
    
    return result


def plot_complexity_performance_scatter(
    complexity_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    config: ComplexityAnalysisConfig,
) -> dict[str, Path]:
    """Create scatter plots of complexity vs performance."""
    outputs = {}
    
    gt_complexity = complexity_df[complexity_df["method"] == "ground_truth"].copy()
    if gt_complexity.empty or paired_df.empty:
        return outputs
    
    gt_complexity["sample_id"] = gt_complexity["sample_id"].astype(str)
    
    # Handle different column names for sample ID
    paired_df = paired_df.copy()
    if "id" in paired_df.columns and "sample_id" not in paired_df.columns:
        paired_df["sample_id"] = paired_df["id"].astype(str)
    elif "sample_id" in paired_df.columns:
        paired_df["sample_id"] = paired_df["sample_id"].astype(str)
    else:
        return outputs
    
    merged = gt_complexity.merge(paired_df, on="sample_id", how="inner")
    
    if len(merged) < 10:
        return outputs
    
    # Create 2x2 scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    plots = [
        ("cc_avg", "delta_exact_match", "Avg Cyclomatic Complexity", "Delta Exact Match"),
        ("mi_score", "delta_exact_match", "Maintainability Index", "Delta Exact Match"),
        ("sloc", "delta_similarity", "Source Lines of Code", "Delta Similarity"),
        ("cc_max", "delta_similarity", "Max Cyclomatic Complexity", "Delta Similarity"),
    ]
    
    for idx, (x_col, y_col, x_label, y_label) in enumerate(plots):
        ax = axes.flatten()[idx]
        
        if x_col not in merged.columns or y_col not in merged.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        
        valid = merged[[x_col, y_col]].dropna()
        if len(valid) < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            continue
        
        ax.scatter(valid[x_col], valid[y_col], alpha=0.4, s=30, edgecolor="none")
        
        # Trend line
        z = np.polyfit(valid[x_col], valid[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Correlation
        r, pval = stats.spearmanr(valid[x_col], valid[y_col])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        ax.annotate(
            f"r = {r:.3f}{sig}\nn = {len(valid)}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    
    plt.suptitle("Ground Truth Complexity vs Bypass Advantage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_vs_performance.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["complexity_vs_performance"] = plot_path
    print(f"  Saved: {plot_path}")
    
    return outputs


def generate_complexity_summary_text(
    method_stats: pd.DataFrame,
    comparison_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> str:
    """Generate interpretive text for complexity findings."""
    lines = []
    
    lines.append("## Code Complexity Analysis")
    lines.append("")
    lines.append("This section analyzes code complexity metrics across different methods ")
    lines.append("and their relationship to merge resolution performance.")
    lines.append("")
    
    # Method comparison
    if not method_stats.empty:
        lines.append("### Complexity by Method")
        lines.append("")
        lines.append("Average complexity metrics across all analyzed samples:")
        lines.append("")
        lines.append("| Method | SLOC | CC Avg | MI Score | Samples |")
        lines.append("|--------|------|--------|----------|---------|")
        
        for _, row in method_stats.iterrows():
            method = row["method"]
            sloc = row.get("sloc_mean", 0)
            cc = row.get("cc_avg_mean", 0)
            mi = row.get("mi_score_mean", 0)
            n = row.get("n_samples", 0)
            lines.append(f"| {method} | {sloc:.0f} | {cc:.2f} | {mi:.1f} | {n} |")
        
        lines.append("")
        
        # Interpretation
        gt_row = method_stats[method_stats["method"] == "ground_truth"]
        agent_row = method_stats[method_stats["method"] == "agent"]
        bypass_row = method_stats[method_stats["method"] == "bypass"]
        
        if not gt_row.empty and not agent_row.empty and not bypass_row.empty:
            gt_cc = gt_row["cc_avg_mean"].iloc[0]
            agent_cc = agent_row["cc_avg_mean"].iloc[0]
            bypass_cc = bypass_row["cc_avg_mean"].iloc[0]
            
            gt_mi = gt_row["mi_score_mean"].iloc[0]
            agent_mi = agent_row["mi_score_mean"].iloc[0]
            bypass_mi = bypass_row["mi_score_mean"].iloc[0]
            
            lines.append("**Key Findings:**")
            lines.append("")
            
            # CC comparison
            if agent_cc < bypass_cc:
                lines.append(f"- **Agent produces simpler code** (CC: {agent_cc:.2f}) than Bypass ({bypass_cc:.2f})")
            else:
                lines.append(f"- **Bypass produces simpler code** (CC: {bypass_cc:.2f}) than Agent ({agent_cc:.2f})")
            
            # MI comparison
            if agent_mi > bypass_mi:
                lines.append(f"- **Agent has higher maintainability** (MI: {agent_mi:.1f}) than Bypass ({bypass_mi:.1f})")
            else:
                lines.append(f"- **Bypass has higher maintainability** (MI: {bypass_mi:.1f}) than Agent ({agent_mi:.1f})")
            
            # Comparison to ground truth
            agent_cc_diff = agent_cc - gt_cc
            bypass_cc_diff = bypass_cc - gt_cc
            
            closer = "Agent" if abs(agent_cc_diff) < abs(bypass_cc_diff) else "Bypass"
            lines.append(f"- **{closer} complexity is closer to ground truth**")
            lines.append("")
    
    # Correlation findings
    if not correlation_df.empty:
        lines.append("### Complexity vs Performance Correlation")
        lines.append("")
        lines.append("How does ground truth complexity predict which method performs better?")
        lines.append("")
        
        # Get significant correlations
        sig_corrs = correlation_df[correlation_df["spearman_p"] < 0.05].copy()
        
        if not sig_corrs.empty:
            lines.append("**Statistically Significant Correlations (p < 0.05):**")
            lines.append("")
            lines.append("| Complexity Metric | Performance Metric | Spearman r | Interpretation |")
            lines.append("|-------------------|-------------------|------------|----------------|")
            
            for _, row in sig_corrs.head(10).iterrows():
                comp = row["complexity_metric"]
                perf = row["performance_metric"]
                r = row["spearman_r"]
                
                if r > 0.1:
                    interp = "Higher complexity -> Bypass advantage"
                elif r < -0.1:
                    interp = "Higher complexity -> Agent advantage"
                else:
                    interp = "Weak relationship"
                
                lines.append(f"| {comp} | {perf} | {r:.3f} | {interp} |")
            
            lines.append("")
            
            # Overall interpretation - use similarity correlations which have stronger signal
            sim_corrs = sig_corrs[sig_corrs["performance_metric"].str.contains("similarity|bleu")]
            if not sim_corrs.empty:
                # For SLOC, lloc, cc_total: positive r means higher complexity -> bypass advantage
                # For MI: negative r means lower maintainability -> bypass advantage (i.e., complex code)
                sloc_corr = sim_corrs[sim_corrs["complexity_metric"] == "sloc"]
                if not sloc_corr.empty:
                    avg_r = sloc_corr["spearman_r"].mean()
                    if avg_r > 0.1:
                        lines.append("**Overall Finding:** More complex code (higher SLOC, CC) strongly correlates with ")
                        lines.append("Bypass advantage, suggesting multi-agent approaches excel at handling complex merges.")
                    elif avg_r < -0.1:
                        lines.append("**Overall Finding:** More complex code correlates with Agent advantage, ")
                        lines.append("suggesting single-agent approaches are more robust to complexity.")
                    else:
                        lines.append("**Overall Finding:** Code complexity has minimal impact on which method performs better.")
                lines.append("")
        else:
            lines.append("No statistically significant correlations found between complexity and performance.")
            lines.append("This suggests code complexity alone does not strongly predict method effectiveness.")
            lines.append("")
    
    # Comparison analysis
    if not comparison_df.empty:
        lines.append("### Agent vs Bypass Complexity Differences")
        lines.append("")
        
        # Calculate how often each method produces code closer to ground truth
        agent_closer_cc = 0
        bypass_closer_cc = 0
        agent_closer_mi = 0
        bypass_closer_mi = 0
        
        for _, row in comparison_df.iterrows():
            agent_cc_diff = abs(row.get("agent_diff_cc_avg", np.nan))
            bypass_cc_diff = abs(row.get("bypass_diff_cc_avg", np.nan))
            
            if pd.notna(agent_cc_diff) and pd.notna(bypass_cc_diff):
                if agent_cc_diff < bypass_cc_diff:
                    agent_closer_cc += 1
                else:
                    bypass_closer_cc += 1
            
            agent_mi_diff = abs(row.get("agent_diff_mi_score", np.nan))
            bypass_mi_diff = abs(row.get("bypass_diff_mi_score", np.nan))
            
            if pd.notna(agent_mi_diff) and pd.notna(bypass_mi_diff):
                if agent_mi_diff < bypass_mi_diff:
                    agent_closer_mi += 1
                else:
                    bypass_closer_mi += 1
        
        total_cc = agent_closer_cc + bypass_closer_cc
        total_mi = agent_closer_mi + bypass_closer_mi
        
        if total_cc > 0:
            lines.append(f"**Cyclomatic Complexity alignment with ground truth:**")
            lines.append(f"- Agent closer: {agent_closer_cc}/{total_cc} ({100*agent_closer_cc/total_cc:.1f}%)")
            lines.append(f"- Bypass closer: {bypass_closer_cc}/{total_cc} ({100*bypass_closer_cc/total_cc:.1f}%)")
            lines.append("")
        
        if total_mi > 0:
            lines.append(f"**Maintainability Index alignment with ground truth:**")
            lines.append(f"- Agent closer: {agent_closer_mi}/{total_mi} ({100*agent_closer_mi/total_mi:.1f}%)")
            lines.append(f"- Bypass closer: {bypass_closer_mi}/{total_mi} ({100*bypass_closer_mi/total_mi:.1f}%)")
            lines.append("")
    
    return "\n".join(lines)


def parse_model_source(source_name: str) -> tuple[str, str]:
    """Parse model source name into (model_name, outcome)."""
    # Examples: "2025-11-09-gpt5nano-failure", "2025-11-11-gpt5nano-pass"
    parts = source_name.split("-")
    if len(parts) >= 5:
        model = parts[3]  # e.g., "gpt5nano", "qwen3", "llama"
        outcome = parts[4]  # e.g., "pass", "failure"
        return model, outcome
    return source_name, "unknown"


def compute_stats_by_source(df: pd.DataFrame, config: ComplexityAnalysisConfig) -> pd.DataFrame:
    """Compute complexity statistics grouped by model source."""
    if df.empty or "model_source" not in df.columns:
        return pd.DataFrame()
    
    valid_df = df[~df["parse_error"]].copy()
    
    # Parse model and outcome
    valid_df["model"], valid_df["outcome"] = zip(*valid_df["model_source"].apply(parse_model_source))
    
    rows = []
    for source in valid_df["model_source"].unique():
        source_data = valid_df[valid_df["model_source"] == source]
        model, outcome = parse_model_source(source)
        
        for method in ["agent", "bypass", "ground_truth"]:
            method_data = source_data[source_data["method"] == method]
            if method_data.empty:
                continue
            
            row = {
                "source": source,
                "model": model,
                "outcome": outcome,
                "method": method,
                "n_samples": len(method_data),
            }
            
            for metric in config.complexity_metrics:
                if metric in method_data.columns:
                    values = method_data[metric].dropna()
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std()
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_complexity_by_model(df: pd.DataFrame, config: ComplexityAnalysisConfig) -> dict[str, Path]:
    """Create plots comparing complexity across different models."""
    outputs = {}
    
    if df.empty or "model_source" not in df.columns:
        return outputs
    
    valid_df = df[~df["parse_error"]].copy()
    valid_df["model"], valid_df["outcome"] = zip(*valid_df["model_source"].apply(parse_model_source))
    valid_df["model_outcome"] = valid_df["model"] + "-" + valid_df["outcome"]
    
    # Get unique model-outcomes for ordering
    model_outcomes = sorted(valid_df["model_outcome"].unique())
    
    # Color palette for models
    model_colors = {
        "gpt5nano": "#1f77b4",
        "qwen3": "#ff7f0e", 
        "llama": "#2ca02c",
        "gemma": "#d62728",
    }
    
    # 1. Complexity by Model-Outcome for Agent outputs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    agent_df = valid_df[valid_df["method"] == "agent"]
    metrics = ["sloc", "cc_avg", "mi_score", "h_difficulty"]
    metric_labels = {
        "sloc": "Source Lines of Code",
        "cc_avg": "Avg Cyclomatic Complexity",
        "mi_score": "Maintainability Index",
        "h_difficulty": "Halstead Difficulty",
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        if metric in agent_df.columns:
            sns.boxplot(
                data=agent_df,
                x="model_outcome",
                y=metric,
                hue="model_outcome",
                order=model_outcomes,
                ax=ax,
                legend=False,
            )
        ax.set_xlabel("")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(f"Agent {metric_labels.get(metric, metric)}")
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Agent Output Complexity by Model & Outcome", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_by_model_agent.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["by_model_agent"] = plot_path
    print(f"  Saved: {plot_path}")
    
    # 2. Pass vs Failure comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, method in enumerate(["agent", "bypass", "ground_truth"]):
        ax = axes[idx]
        method_df = valid_df[valid_df["method"] == method]
        
        if "cc_avg" in method_df.columns:
            sns.boxplot(
                data=method_df,
                x="model",
                y="cc_avg",
                hue="outcome",
                ax=ax,
            )
        ax.set_xlabel("Model")
        ax.set_ylabel("Avg Cyclomatic Complexity")
        ax.set_title(f"{method.replace('_', ' ').title()}")
        ax.legend(title="Outcome", loc="upper right")
    
    plt.suptitle("Cyclomatic Complexity: Pass vs Failure by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_pass_vs_failure.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["pass_vs_failure"] = plot_path
    print(f"  Saved: {plot_path}")
    
    # 3. Maintainability Index by Model & Outcome
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, method in enumerate(["agent", "bypass", "ground_truth"]):
        ax = axes[idx]
        method_df = valid_df[valid_df["method"] == method]
        
        if "mi_score" in method_df.columns:
            sns.violinplot(
                data=method_df,
                x="model",
                y="mi_score",
                hue="outcome",
                split=True,
                ax=ax,
            )
        ax.set_xlabel("Model")
        ax.set_ylabel("Maintainability Index")
        ax.set_title(f"{method.replace('_', ' ').title()}")
        ax.legend(title="Outcome", loc="upper right")
        ax.axhline(y=20, color="green", linestyle="--", alpha=0.5, linewidth=1)
    
    plt.suptitle("Maintainability Index: Pass vs Failure by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_mi_pass_vs_failure.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["mi_pass_vs_failure"] = plot_path
    print(f"  Saved: {plot_path}")
    
    # 4. Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Aggregate by model-outcome for agent
    agent_agg = agent_df.groupby("model_outcome").agg({
        "cc_avg": ["mean", "std"],
        "mi_score": ["mean", "std"],
    }).reset_index()
    agent_agg.columns = ["model_outcome", "cc_avg_mean", "cc_avg_std", "mi_score_mean", "mi_score_std"]
    
    # CC Avg bar chart
    ax = axes[0]
    x_pos = range(len(agent_agg))
    ax.bar(x_pos, agent_agg["cc_avg_mean"], yerr=agent_agg["cc_avg_std"], capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agent_agg["model_outcome"], rotation=45, ha="right")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    ax.set_title("Agent CC by Model-Outcome")
    
    # MI bar chart
    ax = axes[1]
    ax.bar(x_pos, agent_agg["mi_score_mean"], yerr=agent_agg["mi_score_std"], capsize=5, alpha=0.7, color="green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agent_agg["model_outcome"], rotation=45, ha="right")
    ax.set_ylabel("Maintainability Index")
    ax.set_title("Agent MI by Model-Outcome")
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.7, label="MI Grade B threshold")
    ax.legend()
    
    plt.suptitle("Agent Output Complexity Summary by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_model_summary.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["model_summary"] = plot_path
    print(f"  Saved: {plot_path}")
    
    # 5. Heatmap of mean complexity by model-outcome and method
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pivot_data = valid_df.groupby(["model_outcome", "method"])["cc_avg"].mean().unstack()
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title("Average Cyclomatic Complexity: Model-Outcome vs Method", fontsize=14, fontweight="bold")
        ax.set_xlabel("Method")
        ax.set_ylabel("Model-Outcome")
    
    plt.tight_layout()
    
    plot_path = config.output_dir / "complexity_heatmap_by_model.png"
    fig.savefig(plot_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["heatmap_by_model"] = plot_path
    print(f"  Saved: {plot_path}")
    
    return outputs


def generate_model_comparison_text(stats_by_source: pd.DataFrame) -> str:
    """Generate interpretive text comparing complexity across models."""
    if stats_by_source.empty:
        return ""
    
    lines = []
    lines.append("### Complexity by Model and Outcome")
    lines.append("")
    lines.append("Comparison of code complexity across different LLM models and pass/failure outcomes:")
    lines.append("")
    
    # Filter to agent outputs
    agent_stats = stats_by_source[stats_by_source["method"] == "agent"].copy()
    
    if not agent_stats.empty:
        lines.append("**Agent Output Complexity:**")
        lines.append("")
        lines.append("| Model | Outcome | SLOC | CC Avg | MI Score | Samples |")
        lines.append("|-------|---------|------|--------|----------|---------|")
        
        for _, row in agent_stats.sort_values(["model", "outcome"]).iterrows():
            model = row["model"]
            outcome = row["outcome"]
            sloc = row.get("sloc_mean", 0)
            cc = row.get("cc_avg_mean", 0)
            mi = row.get("mi_score_mean", 0)
            n = row.get("n_samples", 0)
            lines.append(f"| {model} | {outcome} | {sloc:.0f} | {cc:.2f} | {mi:.1f} | {n} |")
        
        lines.append("")
        
        # Compare pass vs failure for each model
        models = agent_stats["model"].unique()
        lines.append("**Pass vs Failure Findings:**")
        lines.append("")
        
        for model in models:
            model_data = agent_stats[agent_stats["model"] == model]
            pass_data = model_data[model_data["outcome"] == "pass"]
            fail_data = model_data[model_data["outcome"].isin(["failure", "fail"])]
            
            if not pass_data.empty and not fail_data.empty:
                pass_cc = pass_data["cc_avg_mean"].iloc[0]
                fail_cc = fail_data["cc_avg_mean"].iloc[0]
                pass_mi = pass_data["mi_score_mean"].iloc[0]
                fail_mi = fail_data["mi_score_mean"].iloc[0]
                
                cc_diff = pass_cc - fail_cc
                mi_diff = pass_mi - fail_mi
                
                if abs(cc_diff) > 0.1 or abs(mi_diff) > 1:
                    if cc_diff > 0:
                        lines.append(f"- **{model}**: Pass cases have higher complexity (CC +{cc_diff:.2f}) than failure cases")
                    else:
                        lines.append(f"- **{model}**: Failure cases have higher complexity (CC +{-cc_diff:.2f}) than pass cases")
                    
                    if mi_diff > 0:
                        lines.append(f"  - Pass cases have better maintainability (MI +{mi_diff:.1f})")
                    else:
                        lines.append(f"  - Failure cases have better maintainability (MI +{-mi_diff:.1f})")
        
        lines.append("")
    
    # Bypass stats
    bypass_stats = stats_by_source[stats_by_source["method"] == "bypass"].copy()
    
    if not bypass_stats.empty:
        lines.append("**Bypass Output Complexity:**")
        lines.append("")
        lines.append("| Model | Outcome | SLOC | CC Avg | MI Score | Samples |")
        lines.append("|-------|---------|------|--------|----------|---------|")
        
        for _, row in bypass_stats.sort_values(["model", "outcome"]).iterrows():
            model = row["model"]
            outcome = row["outcome"]
            sloc = row.get("sloc_mean", 0)
            cc = row.get("cc_avg_mean", 0)
            mi = row.get("mi_score_mean", 0)
            n = row.get("n_samples", 0)
            lines.append(f"| {model} | {outcome} | {sloc:.0f} | {cc:.2f} | {mi:.1f} | {n} |")
        
        lines.append("")
    
    return "\n".join(lines)


def run_complexity_analysis(output_dir: Path) -> None:
    """Run full complexity analysis and generate outputs."""
    config = ComplexityAnalysisConfig(output_dir=output_dir)
    
    print("=" * 60)
    print("Complexity Analysis for RQ3")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading complexity data...")
    complexity_df = load_all_complexity_data(output_dir)
    
    if complexity_df.empty:
        print("No complexity data found. Exiting.")
        return
    
    # Load paired data for correlation analysis
    paired_path = output_dir / "paired_data.csv"
    paired_df = pd.DataFrame()
    if paired_path.exists():
        paired_df = pd.read_csv(paired_path)
        print(f"\n  Loaded {len(paired_df)} paired samples")
    
    # Compute statistics
    print("\n2. Computing method statistics...")
    method_stats = compute_method_statistics(complexity_df, config)
    if not method_stats.empty:
        stats_path = output_dir / "complexity_method_stats.csv"
        method_stats.to_csv(stats_path, index=False)
        print(f"  Saved: {stats_path}")
    
    # Compute comparison
    print("\n3. Computing method comparison...")
    comparison_df = compute_method_comparison(complexity_df, config)
    if not comparison_df.empty:
        comp_path = output_dir / "complexity_comparison.csv"
        comparison_df.to_csv(comp_path, index=False)
        print(f"  Saved: {comp_path}")
    
    # Compute correlations
    print("\n4. Computing complexity-performance correlations...")
    correlation_df = compute_complexity_performance_correlation(complexity_df, paired_df, config)
    if not correlation_df.empty:
        corr_path = output_dir / "complexity_correlations.csv"
        correlation_df.to_csv(corr_path, index=False)
        print(f"  Saved: {corr_path}")
    
    # Generate plots
    print("\n5. Generating plots...")
    plot_complexity_comparison(complexity_df, config)
    plot_complexity_performance_scatter(complexity_df, paired_df, config)
    
    # Compute stats by model source
    print("\n6. Computing stats by model source...")
    stats_by_source = compute_stats_by_source(complexity_df, config)
    if not stats_by_source.empty:
        source_stats_path = output_dir / "complexity_by_model_source.csv"
        stats_by_source.to_csv(source_stats_path, index=False)
        print(f"  Saved: {source_stats_path}")
    
    # Generate model comparison plots
    print("\n7. Generating model comparison plots...")
    plot_complexity_by_model(complexity_df, config)
    
    # Generate summary text
    print("\n8. Generating summary text...")
    summary_text = generate_complexity_summary_text(method_stats, comparison_df, correlation_df)
    
    # Add model comparison text
    model_comparison_text = generate_model_comparison_text(stats_by_source)
    if model_comparison_text:
        summary_text += "\n\n" + model_comparison_text
    
    # Save complexity summary
    summary_path = output_dir / "complexity_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Saved: {summary_path}")
    
    # Append to main summary if exists
    main_summary_path = output_dir / "rq3_summary.md"
    if main_summary_path.exists():
        with open(main_summary_path, "r", encoding="utf-8") as f:
            existing = f.read()
        
        if "## Code Complexity Analysis" not in existing:
            with open(main_summary_path, "a", encoding="utf-8") as f:
                f.write("\n\n---\n\n")
                f.write(summary_text)
            print(f"  Appended complexity section to: {main_summary_path}")
    
    print("\n" + "=" * 60)
    print("Complexity analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    import tyro
    
    @dataclass
    class Args:
        output_dir: Path = Path("results/rq3")
    
    args = tyro.cli(Args)
    run_complexity_analysis(args.output_dir)
