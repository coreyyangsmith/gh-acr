from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tyro

from .data_loader import load_results


def _coerce_similarity(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def _normalize_eval_method(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _normalize_bypass_method(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    # Reduce to {A, B, MIX, NA}
    s = s.replace({"ALL_A": "A", "ALL_B": "B", "": pd.NA})
    return s


def _closest_base_lookup(df: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "file_name", "eval_method", "similarity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Results missing required columns for base comparison: {sorted(missing)}")

    work = df.copy()
    work["eval_method_norm"] = _normalize_eval_method(work["eval_method"])  # type: ignore[index]

    has_file_name = (~work["file_name"].isna()) & (work["file_name"].astype(str).str.len() > 0)
    base_rows = work[has_file_name & work["eval_method_norm"].isin(["base_a", "base_b"])].copy()
    if base_rows.empty:
        # No bases found → return empty frame with proper index/columns
        return pd.DataFrame(columns=["id", "file_name", "closest_base"]).set_index(["id", "file_name"]).copy()

    base_rows["similarity_num"] = _coerce_similarity(base_rows["similarity"])  # type: ignore[index]

    # Max similarity per (id, file_name, eval_method)
    grp = (
        base_rows.groupby(["id", "file_name", "eval_method_norm"])  # type: ignore[index]
        ["similarity_num"].max()
        .reset_index()
    )

    # Pivot to get base_a/base_b side-by-side
    pivot = (
        grp.pivot(index=["id", "file_name"], columns="eval_method_norm", values="similarity_num")
        .rename(columns={"base_a": "sim_base_a", "base_b": "sim_base_b"})
    )

    # Determine closest base; ties or missing → NA
    a = pivot.get("sim_base_a")
    b = pivot.get("sim_base_b")
    if a is None or b is None:
        pivot["closest_base"] = pd.NA
    else:
        is_a = (a > b)
        is_b = (b > a)
        closest: pd.Series = pd.Series(pd.NA, index=pivot.index, dtype="object")
        closest[is_a] = "A"
        closest[is_b] = "B"
        pivot["closest_base"] = closest

    return pivot[["closest_base"]]


def enrich_with_best_judgement(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Prepare helpers
    work["eval_method_norm"] = _normalize_eval_method(work["eval_method"])  # type: ignore[index]
    if "bypass_method" in work.columns:
        work["bypass_method_norm"] = _normalize_bypass_method(work["bypass_method"])  # type: ignore[index]
    else:
        work["bypass_method_norm"] = pd.NA

    # Compute closest base per (id, file_name)
    base_lookup = _closest_base_lookup(work)

    # Determine which rows are bypass-like and have concrete file_name
    is_bypass_like = work["eval_method_norm"].str.startswith("bypass")  # type: ignore[index]
    has_file_name = (~work["file_name"].isna()) & (work["file_name"].astype(str).str.len() > 0)
    target_rows = work[is_bypass_like & has_file_name].copy()

    # Join closest base info
    if not target_rows.empty and not base_lookup.empty:
        joined = target_rows.merge(
            base_lookup,
            left_on=["id", "file_name"],  # type: ignore[list-item]
            right_index=True,
            how="left",
        )
    else:
        joined = target_rows.copy()
        joined["closest_base"] = pd.NA

    # Compute best_judgement as boolean with NA support
    # True iff bypass_method in {A,B} and equals closest_base
    bm = joined["bypass_method_norm"]
    cb = joined["closest_base"]
    cond_known_choice = bm.isin(["A", "B"]) & cb.isin(["A", "B"])  # type: ignore[operator]
    best_series: pd.Series = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    best_series[cond_known_choice] = (bm[cond_known_choice] == cb[cond_known_choice])  # type: ignore[index]

    # Start with all NA, then fill for bypass rows
    out_best = pd.Series(pd.NA, index=work.index, dtype="boolean")
    out_best.loc[joined.index] = best_series

    # For eval_method == 'agent' explicitly ensure NA (already NA by default)
    # Left as-is for other non-bypass methods
    work["best_judgement"] = out_best

    # Clean helper columns
    work.drop(columns=[c for c in ["eval_method_norm", "bypass_method_norm"] if c in work.columns], inplace=True)

    return work


@dataclass
class Flags:
    input_csv: Optional[Path] = None
    output_csv: Optional[Path] = None


def main(flags: Flags) -> None:
    # Load
    if flags.input_csv is None:
        data = load_results(None)
        src_path = data.path
        df = data.dataframe
    else:
        src_path = Path(flags.input_csv)
        data = load_results(src_path)
        df = data.dataframe

    # Validate minimal columns
    required_cols = {"id", "file_name", "eval_method", "similarity"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    if "bypass_method" not in df.columns:
        # Create placeholder if missing so the logic still runs and leaves NA
        df = df.copy()
        df["bypass_method"] = pd.NA

    enriched = enrich_with_best_judgement(df)

    # Output path
    out_path = flags.output_csv or (src_path.parent / f"{src_path.stem}_with_best_judgement.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print({"input": str(src_path), "output": str(out_path), "rows": len(enriched)})


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


