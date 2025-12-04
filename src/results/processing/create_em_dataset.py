from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class Flags:
    input_csv: Path
    output_dir: Path


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce common truthy/falsey encodings to boolean."""
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    result = pd.Series([None] * len(s), index=s.index, dtype="object")
    result[s.isin(truthy)] = True
    result[s.isin(falsy)] = False
    # Anything else -> NaN (ignored in all-true check)
    return result.astype("boolean")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sanitize_path_component(text: str) -> str:
    # Replace path-unfriendly characters and trim
    safe = (
        str(text)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
        .replace("*", "_")
        .replace("?", "_")
        .replace("\"", "'")
        .replace("<", "(")
        .replace(">", ")")
        .replace("|", "+")
    )
    # Collapse whitespace
    safe = " ".join(safe.split())
    return safe or "unknown"


def _aggregate_group(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows that share the same id within a (model_name, eval_method) group.

    Rules:
    - tokens_*: sum
    - bleu3, rouge_l, similarity: mean
    - exact_match: True only if all files True, else False (ignoring NaNs)
    - best_judgement: True only if all files True, else False (ignoring NaNs)
    - keep representative repo/file counts for reference
    """
    # Identify token columns by common names
    token_cols: list[str] = [
        c for c in df.columns if c.startswith("tokens_") or c in {"tokens_total", "tokens_in", "tokens_out"}
    ]

    numeric_mean_cols: list[str] = [c for c in ["bleu3", "rouge_l", "similarity"] if c in df.columns]

    # Prepare exact_match
    has_em = "exact_match" in df.columns
    if has_em:
        em_bool = _coerce_bool(df["exact_match"])
        # all True if there are no False values among non-null entries
        em_all_true = bool((~(em_bool.fillna(True) == False)).all())  # noqa: E712
    else:
        em_all_true = False

    # Prepare best_judgement
    has_bj = "best_judgement" in df.columns
    if has_bj:
        bj_bool = _coerce_bool(df["best_judgement"]) 
        # all True if there are no False values among non-null entries
        bj_all_true = bool((~(bj_bool.fillna(True) == False)).all())  # noqa: E712
    else:
        bj_all_true = False

    # Build aggregated row
    out: dict[str, object] = {}

    # id, model_name, eval_method
    for key in ["id", "model_name", "eval_method"]:
        if key in df.columns:
            # take the first (they are identical within the group)
            out[key] = df[key].iloc[0]

    # repo could differ when multiple files match one id; keep first and counts
    if "repo" in df.columns:
        out["repo"] = df["repo"].iloc[0]
    if "file_name" in df.columns:
        out["n_files"] = int(df["file_name"].nunique())

    # Preserve bypass_method if present (use first non-null value)
    if "bypass_method" in df.columns:
        vals = df["bypass_method"].dropna()
        out["bypass_method"] = vals.iloc[0] if not vals.empty else pd.NA

    # Preserve difficulty and project_size if present (use first non-null value)
    if "difficulty" in df.columns:
        dvals = df["difficulty"].dropna()
        out["difficulty"] = dvals.iloc[0] if not dvals.empty else pd.NA
    if "project_size" in df.columns:
        pvals = df["project_size"].dropna()
        out["project_size"] = pvals.iloc[0] if not pvals.empty else pd.NA

    # Sum tokens
    for c in token_cols:
        out[c] = _coerce_numeric(df[c]).sum(min_count=1)

    # Means for metrics
    for c in numeric_mean_cols:
        out[c] = _coerce_numeric(df[c]).mean()

    if has_em:
        out["exact_match"] = em_all_true
    if has_bj:
        out["best_judgement"] = bj_all_true

    # Optional helpers
    if "total_cost" in df.columns:
        out["total_cost_sum"] = _coerce_numeric(df["total_cost"]).sum(min_count=1)
    if "processing_time_s" in df.columns:
        out["processing_time_s_sum"] = _coerce_numeric(df["processing_time_s"]).sum(min_count=1)

    return pd.DataFrame([out])


def _aggregate_per_id(group: pd.DataFrame) -> pd.DataFrame:
    # group contains the subset for a particular (model_name, eval_method)
    if "id" not in group.columns:
        # If no id, treat each row as its own aggregated instance
        return group.copy()
    aggregated: list[pd.DataFrame] = []
    for _, df_id in group.groupby("id", dropna=False):
        aggregated.append(_aggregate_group(df_id))
    return pd.concat(aggregated, ignore_index=True)


def _write_partitioned(
    df: pd.DataFrame, *, output_dir: Path, model_col: str = "model_name", method_col: str = "eval_method"
) -> list[Path]:
    written: list[Path] = []
    # Collect aggregated frames to also write combined CSVs per model and globally
    per_model_agg: dict[str, list[pd.DataFrame]] = {}
    global_agg: list[pd.DataFrame] = []
    if model_col not in df.columns or method_col not in df.columns:
        raise ValueError("Input data must include 'model_name' and 'eval_method' columns")

    for (model, method), sub in df.groupby([model_col, method_col], dropna=False):
        model_s = _sanitize_path_component("unknown" if pd.isna(model) else str(model))
        method_s = _sanitize_path_component("unknown" if pd.isna(method) else str(method))
        out_dir = output_dir / model_s / method_s
        out_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate within this partition per id
        agg = _aggregate_per_id(sub)
        # Sort for readability
        sort_cols: list[str] = [c for c in ["id", "repo"] if c in agg.columns]
        if sort_cols:
            agg = agg.sort_values(sort_cols)

        save_path = out_dir / "dataset.csv"
        agg.to_csv(save_path, index=False)
        written.append(save_path)

        # Stash for per-model combined and global combined
        per_model_agg.setdefault(model_s, []).append(agg)
        global_agg.append(agg)

    # Write combined.csv per model
    for model_s, frames in per_model_agg.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        model_dir = output_dir / model_s
        model_dir.mkdir(parents=True, exist_ok=True)
        combined_path = model_dir / "combined.csv"
        combined.to_csv(combined_path, index=False)
        written.append(combined_path)

    # Write models_combined.csv at root level
    if global_agg:
        all_combined = pd.concat(global_agg, ignore_index=True)
        root_combined_path = output_dir / "models_combined.csv"
        all_combined.to_csv(root_combined_path, index=False)
        written.append(root_combined_path)

    return written


def run(flags: Flags) -> list[Path]:
    df = pd.read_csv(flags.input_csv)

    # Normalize booleans and ensure presence of key columns
    if "exact_match" in df.columns:
        df["exact_match"] = _coerce_bool(df["exact_match"]).astype("boolean")
    if "best_judgement" in df.columns:
        df["best_judgement"] = _coerce_bool(df["best_judgement"]).astype("boolean")

    # Light normalization for names
    if "model_name" not in df.columns:
        df["model_name"] = "unknown"
    if "eval_method" not in df.columns:
        raise ValueError("CSV must include 'eval_method' column")

    flags.output_dir.mkdir(parents=True, exist_ok=True)

    return _write_partitioned(df, output_dir=flags.output_dir)


def parse_args(argv: Optional[Iterable[str]] = None) -> Flags:
    p = argparse.ArgumentParser(description="Create per-model/eval EM dataset with per-id aggregation.")
    p.add_argument("--input", "-i", required=True, help="Path to input CSV")
    p.add_argument("--output", "-o", required=True, help="Root output directory")
    args = p.parse_args(list(argv) if argv is not None else None)
    return Flags(input_csv=Path(args.input), output_dir=Path(args.output))


def main(argv: Optional[Iterable[str]] = None) -> None:
    flags = parse_args(argv)
    written = run(flags)
    # Print written paths for quick inspection
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
