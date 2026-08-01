"""CLI orchestrator for Better-Judge leave-one-out ablation analyses.

Usage
-----
    python -m src.analysis.ablations.main \\
        --input-csv data/2026_08_01_results.csv \\
        --output-dir results/ablations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import tyro

from .config import (
    AblationConfig,
    DEFAULT_ABLATIONS,
    ANCHOR_METHOD,
    get_component_label,
)
from .data import (
    prepare_results,
    compute_component_contributions,
    compute_method_ladder_means,
    compute_wtl_matrix,
    compute_cost_quality,
    compute_stratified_component_deltas,
    compute_routing_counterfactuals,
    compute_disagreement_cases,
    compute_cross_model_stability,
    list_models,
)
from .plots import (
    render_component_forest,
    render_ablation_ladder,
    render_wtl_bars,
    render_cost_quality_pareto,
    render_stratified_forest,
    render_difficulty_component_heatmap,
    render_routing_conditional,
    render_cross_model_stability,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AblationFlags:
    """CLI flags for ablation analyses."""

    input_csv: Path
    output_dir: Path = Path("results/ablations")
    show: bool = False
    anchor_method: str = ANCHOR_METHOD
    ablations: list[str] = field(default_factory=lambda: list(DEFAULT_ABLATIONS))
    exclude_soft_degraded: bool = False
    n_bootstrap: int = 2000

    # Toggles
    component_forest: bool = True
    ladder: bool = True
    wtl: bool = True
    cost_pareto: bool = True
    stratified: bool = True
    routing: bool = True
    cross_model: bool = True
    disagreement_cases: bool = True


def _add_component_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ablation" not in df.columns:
        return df
    out = df.copy()
    out["component_label"] = out["ablation"].map(get_component_label)
    return out


def generate_all_ablation_figures(
    input_csv: str | Path,
    output_dir: str | Path = "results/ablations",
    *,
    show: bool = False,
    config: Optional[AblationConfig] = None,
    component_forest: bool = True,
    ladder: bool = True,
    wtl: bool = True,
    cost_pareto: bool = True,
    stratified: bool = True,
    routing: bool = True,
    cross_model: bool = True,
    disagreement_cases: bool = True,
) -> dict[str, Path]:
    """Run the full ablation analysis suite."""
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = AblationConfig()

    logger.info("Loading data from %s", input_path)
    raw = pd.read_csv(input_path)
    logger.info("Loaded %d rows", len(raw))

    if "eval_method" not in raw.columns:
        raise ValueError("Input CSV must have 'eval_method' column")

    present = set(raw["eval_method"].dropna().astype(str).unique())
    missing_abl = [a for a in config.ablations if a not in present]
    if config.anchor_method not in present:
        raise ValueError(
            f"Anchor method {config.anchor_method!r} not in CSV. Present: {sorted(present)}"
        )
    if missing_abl:
        logger.warning("Ablations missing from CSV (skipped): %s", missing_abl)
        config = AblationConfig(
            anchor_method=config.anchor_method,
            agent_method=config.agent_method,
            ablations=[a for a in config.ablations if a in present],
            baseline_methods=list(config.baseline_methods),
            metrics=list(config.metrics),
            n_bootstrap=config.n_bootstrap,
            ci_level=config.ci_level,
            random_state=config.random_state,
            dpi=config.dpi,
            exclude_soft_degraded=config.exclude_soft_degraded,
        )

    df = prepare_results(raw, config)
    models = list_models(df[df["eval_method"].isin(config.all_multi_methods())])
    logger.info("Models: %s", [str(m) for m in models])
    logger.info("Anchor=%s ablations=%s", config.anchor_method, config.ablations)

    outputs: dict[str, Path] = {}

    # --- 1. Component contributions ---
    logger.info("Computing component contributions (LOO deltas)...")
    contributions = compute_component_contributions(df, config)
    contributions = _add_component_labels(contributions)
    path = output_path / "ablation_component_contributions.csv"
    contributions.to_csv(path, index=False)
    outputs["component_contributions_csv"] = path
    logger.info("  Saved: %s (%d rows)", path, len(contributions))

    if component_forest and not contributions.empty:
        for metric in ("exact_match", "similarity"):
            if metric not in contributions["metric"].unique():
                continue
            path = output_path / f"ablation_component_forest_{metric}.png"
            render_component_forest(
                contributions, config, metric=metric, output_path=path, show=show
            )
            outputs[f"component_forest_{metric}"] = path
            logger.info("  Saved: %s", path)

    # --- 2. Ladder ---
    logger.info("Computing ablation ladder means...")
    ladder_df = compute_method_ladder_means(df, config)
    path = output_path / "ablation_ladder_means.csv"
    ladder_df.to_csv(path, index=False)
    outputs["ladder_csv"] = path
    logger.info("  Saved: %s (%d rows)", path, len(ladder_df))

    if ladder and not ladder_df.empty:
        for metric in ("exact_match", "similarity"):
            path = output_path / f"ablation_ladder_{metric}.png"
            render_ablation_ladder(ladder_df, config, metric=metric, output_path=path, show=show)
            outputs[f"ladder_{metric}"] = path
            logger.info("  Saved: %s", path)

    # --- 3. WTL ---
    logger.info("Computing win/tie/loss matrices...")
    wtl_frames = []
    for metric in ("exact_match", "similarity"):
        wtl_df = compute_wtl_matrix(df, config, metric=metric)
        if not wtl_df.empty:
            wtl_frames.append(wtl_df)
    wtl_all = pd.concat(wtl_frames, ignore_index=True) if wtl_frames else pd.DataFrame()
    path = output_path / "ablation_wtl.csv"
    wtl_all.to_csv(path, index=False)
    outputs["wtl_csv"] = path
    logger.info("  Saved: %s (%d rows)", path, len(wtl_all))

    if wtl and not wtl_all.empty:
        for comparison in ("anchor_vs_ablation", "method_vs_agent"):
            for metric in ("exact_match", "similarity"):
                sub = wtl_all[
                    (wtl_all["comparison"] == comparison) & (wtl_all["metric"] == metric)
                ]
                if sub.empty:
                    continue
                path = output_path / f"ablation_wtl_{comparison}_{metric}.png"
                render_wtl_bars(sub, config, comparison=comparison, output_path=path, show=show)
                outputs[f"wtl_{comparison}_{metric}"] = path
                logger.info("  Saved: %s", path)

    # --- 4. Cost–quality ---
    logger.info("Computing cost-quality Pareto data...")
    cost_df = compute_cost_quality(df, config)
    path = output_path / "ablation_cost_quality.csv"
    cost_df.to_csv(path, index=False)
    outputs["cost_quality_csv"] = path
    logger.info("  Saved: %s (%d rows)", path, len(cost_df))

    if cost_pareto and not cost_df.empty:
        for quality_col, cost_col, tag in (
            ("mean_similarity", "mean_total_cost", "similarity_cost"),
            ("mean_exact_match", "mean_total_cost", "em_cost"),
            ("mean_similarity", "mean_tokens_total", "similarity_tokens"),
            ("mean_similarity", "mean_processing_time_s", "similarity_time"),
        ):
            if quality_col not in cost_df.columns or cost_col not in cost_df.columns:
                continue
            path = output_path / f"ablation_pareto_{tag}.png"
            render_cost_quality_pareto(
                cost_df,
                config,
                quality_col=quality_col,
                cost_col=cost_col,
                output_path=path,
                show=show,
            )
            outputs[f"pareto_{tag}"] = path
            logger.info("  Saved: %s", path)

    # --- 5. Stratified ---
    logger.info("Computing stratified component effects...")
    strat_frames = []
    for metric in ("exact_match", "similarity"):
        s = compute_stratified_component_deltas(df, config, metric=metric)
        if not s.empty:
            strat_frames.append(s)
    strat_df = pd.concat(strat_frames, ignore_index=True) if strat_frames else pd.DataFrame()
    strat_df = _add_component_labels(strat_df)
    path = output_path / "ablation_stratified_deltas.csv"
    strat_df.to_csv(path, index=False)
    outputs["stratified_csv"] = path
    logger.info("  Saved: %s (%d rows)", path, len(strat_df))

    if stratified and not strat_df.empty:
        for stratum in ("difficulty", "project_size", "conflict_size"):
            for metric in ("exact_match", "similarity"):
                sub = strat_df[
                    (strat_df["stratum"] == stratum) & (strat_df["metric"] == metric)
                ]
                if sub.empty:
                    continue
                path = output_path / f"ablation_stratified_{stratum}_{metric}.png"
                render_stratified_forest(
                    sub, config, stratum=stratum, output_path=path, show=show
                )
                outputs[f"stratified_{stratum}_{metric}"] = path
                logger.info("  Saved: %s", path)

        # Difficulty × component heatmaps
        for metric in ("exact_match", "similarity"):
            sub = strat_df[
                (strat_df["stratum"] == "difficulty") & (strat_df["metric"] == metric)
            ]
            if sub.empty:
                continue
            path = output_path / f"ablation_heatmap_difficulty_{metric}.png"
            render_difficulty_component_heatmap(sub, config, output_path=path, show=show)
            outputs[f"heatmap_difficulty_{metric}"] = path
            logger.info("  Saved: %s", path)
            for model in models:
                msub = sub[sub["model_name"] == model]
                if msub.empty:
                    continue
                short = str(model).split("/")[-1].replace(":", "_")
                path = output_path / f"ablation_heatmap_difficulty_{metric}_{short}.png"
                render_difficulty_component_heatmap(
                    msub, config, model_name=model, output_path=path, show=show
                )
                outputs[f"heatmap_difficulty_{metric}_{short}"] = path

    # --- 6. Routing counterfactuals ---
    if routing:
        logger.info("Computing judge routing counterfactuals...")
        agreement_df, conditional_df = compute_routing_counterfactuals(df, config)
        path = output_path / "ablation_routing_agreement.csv"
        agreement_df.to_csv(path, index=False)
        outputs["routing_agreement_csv"] = path
        path = output_path / "ablation_routing_conditional.csv"
        conditional_df.to_csv(path, index=False)
        outputs["routing_conditional_csv"] = path
        logger.info(
            "  Saved routing CSVs (%d agreement, %d conditional rows)",
            len(agreement_df),
            len(conditional_df),
        )
        if not conditional_df.empty:
            path = output_path / "ablation_routing_conditional.png"
            render_routing_conditional(conditional_df, config, output_path=path, show=show)
            outputs["routing_conditional"] = path
            logger.info("  Saved: %s", path)

    # --- 7. Cross-model stability ---
    if cross_model:
        logger.info("Computing cross-model component stability...")
        stability = compute_cross_model_stability(contributions)
        path = output_path / "ablation_cross_model_stability.csv"
        stability.to_csv(path, index=False)
        outputs["cross_model_csv"] = path
        if not stability.empty:
            for metric in ("exact_match", "similarity"):
                path = output_path / f"ablation_cross_model_{metric}.png"
                render_cross_model_stability(
                    stability, config, metric=metric, output_path=path, show=show
                )
                outputs[f"cross_model_{metric}"] = path
                logger.info("  Saved: %s", path)

    # --- 8. Disagreement cases ---
    if disagreement_cases:
        logger.info("Mining disagreement / complementarity cases...")
        cases = compute_disagreement_cases(df, config)
        path = output_path / "ablation_disagreement_cases.csv"
        cases.to_csv(path, index=False)
        outputs["disagreement_cases_csv"] = path
        logger.info("  Saved: %s (%d rows)", path, len(cases))

    # --- Summary markdown ---
    summary_path = output_path / "ABLATION_SUMMARY.md"
    _write_summary(summary_path, contributions, ladder_df, cost_df, config)
    outputs["summary_md"] = summary_path
    logger.info("  Saved: %s", summary_path)

    logger.info("Ablation analysis complete. Generated %d outputs in %s", len(outputs), output_path)
    return outputs


def _write_summary(
    path: Path,
    contributions: pd.DataFrame,
    ladder: pd.DataFrame,
    cost_df: pd.DataFrame,
    config: AblationConfig,
) -> None:
    lines = [
        "# Better-Judge Ablation Analysis Summary",
        "",
        f"Anchor method: `{config.anchor_method}`",
        f"Ablations: {', '.join(f'`{a}`' for a in config.ablations)}",
        "",
        "## Component contributions (mean Δ = BJ − ablation)",
        "",
        "| Model | Component | Metric | N | Mean Δ | 95% CI | p |",
        "|-------|-----------|--------|---|--------|--------|---|",
    ]
    if not contributions.empty:
        for _, r in contributions.iterrows():
            if r["metric"] not in ("exact_match", "similarity"):
                continue
            label = get_component_label(r["ablation"])
            model = str(r["model_name"]).split("/")[-1]
            p = r["p_value"]
            p_s = f"{p:.3g}" if pd.notna(p) else "NA"
            lines.append(
                f"| {model} | {label} | {r['metric']} | {r['n_pairs']} | "
                f"{r['mean_delta']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {p_s} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — |")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "- `bj_no_plan` removes **both** planner and reviewer (confounded); "
        "prefer `bj_no_review` to isolate the review loop."
    )
    lines.append(
        "- Positive Δ means Better-Judge outperforms the ablation "
        "(removing that component hurts)."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(flags: AblationFlags) -> None:
    """CLI entry point."""
    config = AblationConfig(
        anchor_method=flags.anchor_method,
        ablations=list(flags.ablations),
        n_bootstrap=flags.n_bootstrap,
        exclude_soft_degraded=flags.exclude_soft_degraded,
    )
    generate_all_ablation_figures(
        input_csv=flags.input_csv,
        output_dir=flags.output_dir,
        show=flags.show,
        config=config,
        component_forest=flags.component_forest,
        ladder=flags.ladder,
        wtl=flags.wtl,
        cost_pareto=flags.cost_pareto,
        stratified=flags.stratified,
        routing=flags.routing,
        cross_model=flags.cross_model,
        disagreement_cases=flags.disagreement_cases,
    )


if __name__ == "__main__":
    parsed = tyro.cli(AblationFlags)
    main(parsed)
