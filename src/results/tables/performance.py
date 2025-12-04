from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from src.config.eval_methods import DEFAULT_METHOD_ORDER


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"]).astype(bool)


def _coerce_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _order_methods(present: list[str]) -> list[str]:
    present = [str(x) for x in present]
    return [m for m in DEFAULT_METHOD_ORDER if m in present] + [m for m in present if m not in DEFAULT_METHOD_ORDER]


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def _summarize_group(df: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "em_pct": None,
        "best_judgement_pct": None,
        "bleu3_median": None,
        "rouge_l_median": None,
        "similarity_median": None,
    }
    # EM %
    if "exact_match" in df.columns:
        em = _coerce_bool(df["exact_match"])  # True/False
        den = int(em.notna().sum())
        if den > 0:
            out["em_pct"] = 100.0 * float(em.sum()) / float(den)
    # Best Judgement %
    if "best_judgement" in df.columns:
        bj = _coerce_bool(df["best_judgement"])  # True/False
        den_bj = int(bj.notna().sum())
        if den_bj > 0:
            out["best_judgement_pct"] = 100.0 * float(bj.sum()) / float(den_bj)
    # medians
    if "bleu3" in df.columns:
        out["bleu3_median"] = _coerce_num(df["bleu3"]).median()
    if "rouge_l" in df.columns:
        out["rouge_l_median"] = _coerce_num(df["rouge_l"]).median()
    if "similarity" in df.columns:
        out["similarity_median"] = _coerce_num(df["similarity"]).median()
    return out


def _markdown_table_for_model(df_model: pd.DataFrame, model_name: str) -> str:
    # Ensure eval_method exists; otherwise create a single pseudo-method "all"
    if "eval_method" not in df_model.columns:
        df_model = df_model.copy()
        df_model["eval_method"] = "all"

    methods_present = df_model["eval_method"].astype(str).unique().tolist()
    methods_ordered = _order_methods(methods_present)

    lines: list[str] = []
    lines.append(f"### {model_name}")
    lines.append("")
    lines.append("| Eval Method | EM % | Best Judgement % | BLEU-3 (median) | ROUGE-L (median) | Similarity (median) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    # Per-method rows
    for m in methods_ordered:
        sub = df_model.loc[df_model["eval_method"].astype(str) == m]
        if sub.empty:
            continue
        stats = _summarize_group(sub)
        lines.append(
            "| "
            + f"{m} | "
            + (f"{stats['em_pct']:.1f}" if stats["em_pct"] is not None else "-")
            + " | "
            + (f"{stats['best_judgement_pct']:.1f}" if stats["best_judgement_pct"] is not None else "-")
            + " | "
            + _fmt(stats["bleu3_median"]) + " | "
            + _fmt(stats["rouge_l_median"]) + " | "
            + _fmt(stats["similarity_median"]) + " |"
        )

    # Total row across the model (all methods)
    total_stats = _summarize_group(df_model)
    lines.append(
        "| "
        + "All | "
        + (f"{total_stats['em_pct']:.1f}" if total_stats["em_pct"] is not None else "-")
        + " | "
        + (f"{total_stats['best_judgement_pct']:.1f}" if total_stats["best_judgement_pct"] is not None else "-")
        + " | "
        + _fmt(total_stats["bleu3_median"]) + " | "
        + _fmt(total_stats["rouge_l_median"]) + " | "
        + _fmt(total_stats["similarity_median"]) + " |"
    )

    lines.append("")
    return "\n".join(lines)


def build_markdown_report(df: pd.DataFrame) -> str:
    work = df.copy()
    if "model_name" not in work.columns:
        work["model_name"] = "unknown"
    work["model_name"] = work["model_name"].astype(str).fillna("unknown").replace({"": "unknown"})

    sections: list[str] = ["## Model Performance"]
    for model in sorted(work["model_name"].unique().tolist()):
        df_model = work.loc[work["model_name"] == model]
        sections.append(_markdown_table_for_model(df_model, model))
    sections.append("")
    return "\n".join(sections)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@dataclass
class Flags:
    input_csv: Path
    output_md: Path = Path("results/tables/performance.md")


def main(flags: Flags) -> None:
    flags.output_md.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(flags.input_csv)
    md = build_markdown_report(df)
    flags.output_md.write_text(md, encoding="utf-8")
    print(flags.output_md)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)
