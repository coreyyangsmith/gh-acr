from __future__ import annotations

"""
Filter dataset for cases where the bypass method chose "MIX" and copy their folders.

Usage (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.analysis.processing.filter_mix_cases data\\2025_10_18_Final_Results.csv repos_parent

- Direct script:
  python src\\results\\processing\\filter_mix_cases.py data\\input.csv repos_parent

Behavior:
- Loads the CSV and finds unique IDs where `bypass_method` normalizes to "MIX"
- For each selected ID, copies the child folder named exactly as the ID from the
  given parent directory into an output directory
- Standardizes the instance layout and splits into per-file instances (ID-1, ID-2, ...)
- Writes a manifest and a CSV of filtered rows
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import tyro

try:
    # Prefer project logger when available (works when executed as a module)
    from utils.logger import logger as app_logger  # type: ignore

    logger = app_logger
except Exception:  # pragma: no cover - fallback when run directly as a script
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

# Reuse robust helpers from the sampler to stay consistent with instance layout
# and parallel copy behavior.
from .create_samples import (  # noqa: E402
    _detect_id_column,
    _copy_folders_parallel,
)


def _resolve_output_dir(input_csv: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir
    # Default: <input_parent>/<input_stem>_mix_only
    return input_csv.parent / f"{input_csv.stem}_mix_only"


def _normalize_bypass_method(series: pd.Series) -> pd.Series:
    """
    Normalize bypass method to {A, B, MIX, NA}.
    Accepts variants like 'ALL_A'/'ALL_B', 'Mix', etc.
    """
    s = series.astype(str).str.strip().str.upper()
    s = s.replace({"ALL_A": "A", "ALL_B": "B", "": pd.NA, "NONE": pd.NA, "NA": pd.NA})
    # Anything not A/B becomes MIX only if explicitly equals MIX; otherwise leave as-is
    # to avoid over-eager mapping.
    return s


def _detect_bypass_columns(header_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (bypass_method_col, bypass_decision_col) if present.
    We prefer 'bypass_method' when available; 'bypass_decision' can be a fallback
    that contains 'ALL_A' | 'ALL_B' | 'MIX'.
    """
    cols = set(header_df.columns.astype(str))
    bm = "bypass_method" if "bypass_method" in cols else None
    bd = "bypass_decision" if "bypass_decision" in cols else None
    return bm, bd


@dataclass
class Flags:
    """Parameters controlling inputs, filtering, and copying behavior."""

    input_csv: Path
    parent_dir: Path

    output_dir: Optional[Path] = None
    id_column: str = "id"
    encoding: str = "utf-8"
    delimiter: Optional[str] = None  # Auto-detect when None
    strict: bool = False  # When True, error if any ID folder is missing
    workers: int = 8  # Threaded copies; I/O bound


def _collect_mix_ids(
    *,
    input_csv: Path,
    encoding: str,
    delimiter: Optional[str],
    requested_id_col: str,
) -> Tuple[List[str], str, str]:
    """
    Scan the CSV in chunks and return the unique IDs where bypass_method == 'MIX'.
    Returns (selected_ids_in_order, detected_id_col, chosen_bypass_col)
    """
    # 1) Read header to detect the actual ID column and bypass columns
    header_df = pd.read_csv(
        input_csv,
        sep=delimiter if delimiter is not None else None,
        engine="python" if delimiter is None else "c",
        encoding=encoding,
        nrows=0,
    )
    id_col = _detect_id_column(header_df, requested_id_col)
    bypass_method_col, bypass_decision_col = _detect_bypass_columns(header_df)
    if not bypass_method_col and not bypass_decision_col:
        raise ValueError(
            "Input CSV does not contain 'bypass_method' or 'bypass_decision' columns required to filter MIX cases."
        )

    # 2) Stream rows to find MIX cases
    engine = "python" if delimiter is None else "c"
    usecols = [id_col]
    chosen_bypass_col = bypass_method_col or bypass_decision_col  # prefer method, else decision
    if chosen_bypass_col not in usecols:
        usecols.append(chosen_bypass_col)  # type: ignore[arg-type]

    # Keep insertion order of first appearance
    seen: Set[str] = set()
    selected_ids: List[str] = []

    for chunk in pd.read_csv(
        input_csv,
        sep=delimiter if delimiter is not None else None,
        engine=engine,
        encoding=encoding,
        usecols=usecols,
        chunksize=200_000,
        dtype={id_col: str},
    ):
        # Normalize ID and bypass method
        ids = chunk[id_col].astype(str).str.strip()

        bm_series = chunk[chosen_bypass_col]  # type: ignore[index]
        bm_norm = _normalize_bypass_method(bm_series)

        # If we are reading decision instead of method, map 'ALL_A'/'ALL_B' to A/B,
        # 'MIX' remains MIX.
        # Our normalization leaves strings as-is except for capitalization and ALL_* mapping.
        mix_mask = bm_norm == "MIX"
        if not mix_mask.any():
            continue

        for id_value in ids[mix_mask]:
            if id_value and id_value not in seen:
                seen.add(id_value)
                selected_ids.append(id_value)

    if not selected_ids:
        logger.warning("No MIX cases found in input CSV.")

    return selected_ids, id_col, chosen_bypass_col  # type: ignore[return-value]


def _write_manifest(
    manifest_path: Path,
    *,
    input_csv: Path,
    parent_dir: Path,
    output_dir: Path,
    id_column: str,
    filter_column: str,
    selected_ids: List[str],
    copied_count: int,
    missing_ids: List[str],
    failed: List[Tuple[str, str]],
) -> None:
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "parent_dir": str(parent_dir),
        "output_dir": str(output_dir),
        "id_column": id_column,
        "filter": {filter_column: "MIX"},
        "selected_ids": selected_ids,
        "copied_count": copied_count,
        "missing_ids": missing_ids,
        "failed": [{"id": i, "error": e} for i, e in failed],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Also emit a simple CSV of the selected IDs for convenience
    ids_csv = output_dir / "mix_ids.csv"
    pd.DataFrame({id_column: selected_ids}).to_csv(ids_csv, index=False)


def main(flags: Flags) -> None:
    input_csv = flags.input_csv
    parent_dir = flags.parent_dir

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise NotADirectoryError(f"Parent directory not found or not a directory: {parent_dir}")

    output_dir = _resolve_output_dir(input_csv, flags.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Filtering MIX cases: input=%s, parent_dir=%s, output_dir=%s",
        str(input_csv),
        str(parent_dir),
        str(output_dir),
    )

    selected_ids, detected_id_col, filter_col = _collect_mix_ids(
        input_csv=input_csv,
        encoding=flags.encoding,
        delimiter=flags.delimiter,
        requested_id_col=flags.id_column,
    )
    if not selected_ids:
        # Still write empty artifacts for determinism
        (output_dir / "mix_ids.csv").write_text(f"{detected_id_col}\n", encoding="utf-8")
        (output_dir / "mix_rows.csv").write_text("", encoding="utf-8")
        _write_manifest(
            output_dir / "manifest.json",
            input_csv=input_csv,
            parent_dir=parent_dir,
            output_dir=output_dir,
            id_column=detected_id_col,
            filter_column=filter_col,
            selected_ids=[],
            copied_count=0,
            missing_ids=[],
            failed=[],
        )
        print(
            {
                "input": str(input_csv),
                "parent_dir": str(parent_dir),
                "output_dir": str(output_dir),
                "id_column": detected_id_col,
                "selected_ids": 0,
                "selected_rows_csv": str(output_dir / "mix_rows.csv"),
                "selected_rows": 0,
                "copied": 0,
                "missing": 0,
                "failed": 0,
            }
        )
        return

    logger.info("Found %d unique IDs with MIX bypass", len(selected_ids))

    copied_count, missing_ids, failed = _copy_folders_parallel(selected_ids, parent_dir, output_dir, flags.workers)

    if (missing_ids or failed) and flags.strict:
        missing_preview = ", ".join(missing_ids[:5]) + ("..." if len(missing_ids) > 5 else "")
        failed_preview = ", ".join([f"{i}:{e}" for i, e in failed[:3]]) + ("..." if len(failed) > 3 else "")
        raise FileNotFoundError(
            f"Strict mode: some IDs missing or failed to copy. Missing: [{missing_preview}] | Failed: [{failed_preview}]"
        )

    # Export full rows for selected IDs to a new CSV
    selected_rows_csv = output_dir / "mix_rows.csv"
    engine = "python" if flags.delimiter is None else "c"
    read_kwargs = {
        "sep": flags.delimiter if flags.delimiter is not None else None,
        "engine": engine,
        "encoding": flags.encoding,
        "chunksize": 200_000,
        "dtype": None,
    }
    selected_set = set(selected_ids)
    header_written = False
    selected_rows = 0
    for chunk in pd.read_csv(input_csv, **read_kwargs):
        if detected_id_col not in chunk.columns:
            continue
        mask = chunk[detected_id_col].astype(str).isin(selected_set)
        sub = chunk[mask]
        if sub.empty:
            continue
        sub.to_csv(
            selected_rows_csv,
            index=False,
            mode="a",
            header=not header_written,
            encoding=flags.encoding,
        )
        header_written = True
        selected_rows += int(len(sub))

    # Emit manifest and CSV of selected IDs
    _write_manifest(
        output_dir / "manifest.json",
        input_csv=input_csv,
        parent_dir=parent_dir,
        output_dir=output_dir,
        id_column=detected_id_col,
        filter_column=filter_col,
        selected_ids=selected_ids,
        copied_count=copied_count,
        missing_ids=missing_ids,
        failed=failed,
    )

    # Print concise summary suitable for quick inspection
    summary = {
        "input": str(input_csv),
        "parent_dir": str(parent_dir),
        "output_dir": str(output_dir),
        "id_column": detected_id_col,
        "selected_ids": len(selected_ids),
        "selected_rows_csv": str(selected_rows_csv),
        "selected_rows": selected_rows,
        "copied": copied_count,
        "missing": len(missing_ids),
        "failed": len(failed),
    }
    print(summary)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


