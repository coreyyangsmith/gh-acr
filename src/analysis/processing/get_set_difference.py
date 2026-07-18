from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

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


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_")


def _resolve_column_case_insensitive(
    available: Sequence[str], requested: str
) -> Optional[str]:
    """
    Return the exact column name from available that matches requested
    in a case-insensitive way. Returns None when not found.
    """
    if requested in available:
        return requested
    lowered = {c.lower(): c for c in available}
    return lowered.get(requested.lower())


def _normalize_key_series(
    s: pd.Series, *, strip: bool, case_insensitive: bool
) -> pd.Series:
    out = s.astype(str)
    if strip:
        out = out.str.strip()
    if case_insensitive:
        out = out.str.lower()
    return out.fillna("")


def _detect_master_id_column(header: Sequence[str], preferred: str) -> str:
    """
    Resolve the master ID column with forgiving fallbacks:
      1) exact preferred
      2) 'id' (case-insensitive)
      3) common variants containing 'id' or equal to 'sample_id'
    """
    if preferred in header:
        return preferred
    lowered = {c.lower(): c for c in header}
    if "id" in lowered:
        return lowered["id"]
    candidates = [c for c in header if "id" in c.lower() or c.lower() == "sample_id"]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Could not detect an ID column in master CSV. "
        f"Tried '{preferred}', 'id', or '*id*' variants. Columns: {list(header)}"
    )


def _detect_diff_id_column(header: Sequence[str]) -> str:
    """
    Resolve the diff ID column using:
      1) 'id' (case-insensitive) when present
      2) first 'Unnamed:*' column (case-insensitive) when present
      3) if single column exists, use it
      4) otherwise, fallback to the first column
    """
    lowered = {c.lower(): c for c in header}
    if "id" in lowered:
        return lowered["id"]
    unnamed_candidates = [c for c in header if _normalize(c).startswith("unnamed")]
    if unnamed_candidates:
        return unnamed_candidates[0]
    if len(header) == 1:
        return header[0]
    if not header:
        raise ValueError("Diff CSV appears to have no columns.")
    return header[0]


@dataclass
class Flags:
    """
    Write rows that are present in the master CSV but NOT present in the diff CSV,
    matching on ID only. The diff CSV needs only an 'id' column; if absent, an
    'Unnamed:*' or the first column is used as IDs.

    Examples:
      python -m src.analysis.processing.get_set_difference data\\master.csv data\\diff.csv
      python -m src.analysis.processing.get_set_difference data\\master.csv data\\diff.csv --master-id-column id
    """

    # Inputs
    master_csv: Path
    diff_csv: Path

    # Output (defaults beside master as '<master_stem>_minus_<diff_stem>.csv')
    output_csv: Optional[Path] = None

    # Column controls
    master_id_column: str = "id"

    # Parsing controls
    encoding: str = "utf-8"
    delimiter: Optional[str] = None  # Auto-detect when None
    chunksize: int = 200_000

    # Normalization of key values
    strip_whitespace: bool = True
    case_insensitive: bool = True


def main(flags: Flags) -> None:
    if not flags.master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {flags.master_csv}")
    if not flags.diff_csv.exists():
        raise FileNotFoundError(f"Diff CSV not found: {flags.diff_csv}")

    out_path = (
        flags.output_csv
        if flags.output_csv is not None
        else flags.master_csv.parent / f"{flags.master_csv.stem}_minus_{flags.diff_csv.stem}.csv"
    )

    # Probe headers to pick/resolve ID columns
    engine = "python" if flags.delimiter is None else "c"
    master_header = pd.read_csv(
        flags.master_csv, sep=flags.delimiter if flags.delimiter is not None else None, engine=engine, encoding=flags.encoding, nrows=0
    )
    diff_header = pd.read_csv(
        flags.diff_csv, sep=flags.delimiter if flags.delimiter is not None else None, engine=engine, encoding=flags.encoding, nrows=0
    )
    master_id_col = _detect_master_id_column(list(master_header.columns), flags.master_id_column)
    diff_id_col = _detect_diff_id_column(list(diff_header.columns))
    logger.info("Using ID columns: master=%s, diff=%s", master_id_col, diff_id_col)

    # Build the set of keys present in the diff CSV
    read_kwargs = {
        "sep": flags.delimiter if flags.delimiter is not None else None,
        "engine": engine,
        "encoding": flags.encoding,
        "chunksize": flags.chunksize,
        "dtype": None,
        "usecols": [diff_id_col],  # only read IDs for the diff side
    }
    diff_keys: Set[str] = set()
    for chunk in pd.read_csv(flags.diff_csv, **read_kwargs):
        if diff_id_col not in chunk.columns:
            continue
        ids = _normalize_key_series(
            chunk[diff_id_col], strip=flags.strip_whitespace, case_insensitive=flags.case_insensitive
        )
        diff_keys.update(ids.tolist())
    logger.info("Collected %d unique keys from diff CSV", len(diff_keys))

    # Prepare to stream master CSV and write rows whose key is NOT in diff_keys
    # Reset output file if it exists
    try:
        if out_path.exists():
            out_path.unlink()
    except Exception:
        pass

    rows_in: int = 0
    rows_out: int = 0

    master_read_kwargs = {
        "sep": flags.delimiter if flags.delimiter is not None else None,
        "engine": engine,
        "encoding": flags.encoding,
        "chunksize": flags.chunksize,
        "dtype": None,
    }
    header_written = False
    for chunk in pd.read_csv(flags.master_csv, **master_read_kwargs):
        rows_in += int(len(chunk))
        if master_id_col not in chunk.columns:
            logger.warning("Skipping chunk missing master ID column: %s", master_id_col)
            continue
        ids = _normalize_key_series(
            chunk[master_id_col], strip=flags.strip_whitespace, case_insensitive=flags.case_insensitive
        )
        mask_keep = ~ids.isin(diff_keys)
        kept = chunk[mask_keep]
        if kept.empty:
            continue
        kept.to_csv(
            out_path,
            mode="a",
            index=False,
            header=not header_written,
            encoding=flags.encoding,
        )
        header_written = True
        rows_out += int(len(kept))

    print(
        {
            "master_csv": str(flags.master_csv),
            "diff_csv": str(flags.diff_csv),
            "output_csv": str(out_path),
            "master_id_column": master_id_col,
            "diff_id_column": diff_id_col,
            "rows_in_master": rows_in,
            "rows_written": rows_out,
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


