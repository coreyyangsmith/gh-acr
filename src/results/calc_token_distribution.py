"""Compute token distribution summaries from a results CSV.

This script reads a CSV (flag: --csv) and prints a JSON summary with:
- overall sums of tokens_in, tokens_out, tokens_total, and instance counts
- sums grouped by project_size (with instance counts)
- sums grouped by method (prefers column `eval_method`, falls back to `method`) with counts

Optionally write the JSON to a file with --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


TokenColumns = dict[str, str]


def _detect_method_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the method column to use, or None if not found."""
    for candidate in ("eval_method", "method"):
        if candidate in df.columns:
            return candidate
    return None


def _detect_token_columns(df: pd.DataFrame) -> TokenColumns:
    """Detect token-related column names with sensible fallbacks.

    Returns mapping with keys: tokens_in, tokens_out, tokens_total. If
    tokens_total is absent, it will be computed as tokens_in + tokens_out.
    """
    cols: TokenColumns = {}

    # Required inputs (try canonical names first)
    in_candidates = ["tokens_in", "prompt_tokens", "input_tokens"]
    out_candidates = ["tokens_out", "completion_tokens", "output_tokens", "tokens_output"]
    total_candidates = ["tokens_total", "total_tokens"]

    def _pick(cands: list[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    in_col = _pick(in_candidates)
    out_col = _pick(out_candidates)
    total_col = _pick(total_candidates)

    if in_col is None and out_col is None and total_col is None:
        raise ValueError(
            "No token columns found. Expected one of: tokens_in/tokens_out/tokens_total or common aliases."
        )

    if in_col is None:
        # Sometimes only total is available; allow computing in/out=0
        in_col = "__zero__"
    if out_col is None:
        out_col = "__zero__"

    cols["tokens_in"] = in_col
    cols["tokens_out"] = out_col
    cols["tokens_total"] = total_col or "__compute__"
    return cols


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, replacing NaNs with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _materialize_tokens(df: pd.DataFrame, cols: TokenColumns) -> pd.DataFrame:
    """Return a DataFrame with materialized numeric token columns."""
    work = df.copy()

    if cols["tokens_in"] == "__zero__":
        work["__tokens_in_num"] = 0
    else:
        work["__tokens_in_num"] = _coerce_numeric(work[cols["tokens_in"]])

    if cols["tokens_out"] == "__zero__":
        work["__tokens_out_num"] = 0
    else:
        work["__tokens_out_num"] = _coerce_numeric(work[cols["tokens_out"]])

    if cols["tokens_total"] == "__compute__":
        work["__tokens_total_num"] = work["__tokens_in_num"] + work["__tokens_out_num"]
    else:
        work["__tokens_total_num"] = _coerce_numeric(work[cols["tokens_total"]])

    return work


def _sum_triplet(df: pd.DataFrame) -> Dict[str, int]:
    """Sum numeric token triplet columns and include instance count."""
    return {
        "tokens_in": int(df["__tokens_in_num"].sum()),
        "tokens_out": int(df["__tokens_out_num"].sum()),
        "tokens_total": int(df["__tokens_total_num"].sum()),
        "instances": int(df.shape[0]),
    }


def _group_summaries(df: pd.DataFrame, by: str) -> Dict[str, Dict[str, int]]:
    """Group by a column and compute token sums per group.

    Returns a mapping group_value -> {tokens_in, tokens_out, tokens_total}.
    """
    results: Dict[str, Dict[str, int]] = {}
    if by not in df.columns:
        return results
    for key, sub in df.groupby(by, dropna=False):
        label = "unknown" if pd.isna(key) else str(key)
        results[label] = _sum_triplet(sub)
    # Deterministic order
    return {k: results[k] for k in sorted(results)}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute token distribution summaries from a results CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to input results CSV")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON summary")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)

    method_col = _detect_method_column(df)
    token_cols = _detect_token_columns(df)
    work = _materialize_tokens(df, token_cols)

    payload: Dict[str, Any] = {
        "csv": str(args.csv),
        "columns": {
            "method": method_col or "",
            "tokens_in": token_cols.get("tokens_in", ""),
            "tokens_out": token_cols.get("tokens_out", ""),
            "tokens_total": token_cols.get("tokens_total", ""),
        },
        "overall": _sum_triplet(work),
        "by_project_size": _group_summaries(work, by="project_size"),
        "by_method": _group_summaries(work, by=(method_col or "__absent__")),
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()


