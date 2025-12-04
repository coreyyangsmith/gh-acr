from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import json
import shutil

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


def _detect_id_column(df: pd.DataFrame, requested: str) -> str:
    """Return the column to use as the ID, with forgiving fallbacks."""
    if requested in df.columns:
        return requested
    lowered = {c.lower(): c for c in df.columns}
    if "id" in lowered:
        return lowered["id"]
    candidates = [c for c in df.columns if c.lower() in {"sample_id"} or "id" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"ID column '{requested}' not found and no '*id*' column detected in input CSV."
    )


def _detect_column(df: pd.DataFrame, requested: str, *, fallbacks: Iterable[str] = ()) -> str:
    """Return an existing column name, trying requested and then any fallbacks (case-insensitive)."""
    if requested in df.columns:
        return requested
    lowered = {c.lower(): c for c in df.columns}
    if requested.lower() in lowered:
        return lowered[requested.lower()]
    for fb in fallbacks:
        if fb in df.columns:
            return fb
        if fb.lower() in lowered:
            return lowered[fb.lower()]
    raise ValueError(
        f"Required column '{requested}' not found (tried fallbacks: {list(fallbacks)})"
    )


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a variety of truthy/falsey types into pandas nullable boolean dtype."""
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64").replace({1: True, 0: False}).astype("boolean")
    s = s.astype(str).str.strip().str.lower()
    true_set = {"true", "t", "1", "yes", "y"}
    false_set = {"false", "f", "0", "no", "n", ""}
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[s.isin(true_set)] = True
    out[s.isin(false_set)] = False
    return out


@dataclass
class Flags:
    """
    Split a sampled dataset directory (output of create_samples.py) into pass/ and fail/
    subfolders based on whether the original results CSV has a bypass row with exact_match=True
    for each per-file instance.

    Typical usage (from repo root):
      python -m src.results.processing.filter_samples path\to\sample_dir data\results_all.csv
    """

    # Directory produced by create_samples.py (contains ID-1, ID-2, ..., plus manifest)
    sample_dir: Path
    # Original results CSV that contains per-file rows with columns like:
    # id, file_name, eval_method (bypass-like), exact_match
    original_csv: Path

    # Column controls
    id_column: str = "id"
    file_name_column: str = "file_name"
    exact_match_column: str = "exact_match"
    eval_method_column: str = "eval_method"
    eval_method_contains: str = "bypass"  # Case-insensitive substring filter

    # CSV parsing
    encoding: str = "utf-8"
    delimiter: Optional[str] = None  # Auto-detect when None
    chunksize: int = 200_000

    # Behavior
    dry_run: bool = False  # If True, compute and report classification without moving folders
    fail_on_unmatched: bool = False  # If True, raise if any instance cannot be matched to CSV rows


@dataclass(frozen=True)
class InstanceEntry:
    """Represents a per-file instance directory like <sample_dir>/<id>-<index>."""

    path: Path
    id_value: str
    file_name: Optional[str]  # Derived from default/<unit_name>/ if present


def _list_instance_dirs(sample_dir: Path) -> List[Path]:
    """Return top-level directories in sample_dir that look like instance folders."""
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample directory not found or not a directory: {sample_dir}")
    # Exclude common non-instance entries
    excluded = {"pass", "fail", "agent", "bypass", "default", "__pycache__"}
    out: List[Path] = []
    for child in sample_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in excluded:
            continue
        # Require that an instance contains at least one of expected subdirs or default structure
        # This is heuristic; create_samples.py produces ID-N folders at top level.
        out.append(child)
    return out


def _parse_instance(entry: Path) -> InstanceEntry:
    """Parse ID and file unit name from an instance folder."""
    name = entry.name
    # Split at the last hyphen to extract trailing index
    id_value = name
    if "-" in name:
        parts = name.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            id_value = parts[0]
    unit_name: Optional[str] = None
    default_dir = entry / "default"
    if default_dir.exists() and default_dir.is_dir():
        subdirs = [p for p in default_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            unit_name = subdirs[0].name
    return InstanceEntry(path=entry, id_value=id_value, file_name=unit_name)


def _read_schema_probing(original_csv: Path, *, encoding: str, delimiter: Optional[str]) -> Tuple[str, str, str, str]:
    """Read only header to resolve column names (id, file_name, exact_match, eval_method)."""
    header_df = pd.read_csv(
        original_csv,
        sep=delimiter if delimiter is not None else None,
        engine="python" if delimiter is None else "c",
        encoding=encoding,
        nrows=0,
    )
    id_col = _detect_id_column(header_df, "id")
    file_col = _detect_column(header_df, "file_name", fallbacks=("filename", "file", "path", "filePath"))
    em_col = _detect_column(header_df, "exact_match", fallbacks=("em",))
    eval_col = _detect_column(header_df, "eval_method", fallbacks=("method", "eval", "mode"))
    return id_col, file_col, em_col, eval_col


def _classify_from_csv(
    original_csv: Path,
    *,
    id_col: str,
    file_col: str,
    em_col: str,
    eval_col: str,
    target_ids: Set[str],
    target_pairs: Set[Tuple[str, str]],
    eval_method_contains: str,
    encoding: str,
    delimiter: Optional[str],
    chunksize: int,
) -> Tuple[Dict[Tuple[str, str], bool], Dict[str, bool]]:
    """
    Stream the original CSV and build:
      - per_key_pass[(id, file_name)] = True if any bypass-like row has exact_match True
        (present if at least one row was seen for that key), else False if rows seen but none True.
      - id_any_pass[id] = True if any bypass-like row (across files) has exact_match True.
    """
    per_key_seen_true: Dict[Tuple[str, str], bool] = {}
    per_key_seen_any: Set[Tuple[str, str]] = set()
    id_seen_true: Dict[str, bool] = {}

    engine = "python" if delimiter is None else "c"
    read_kwargs = {
        "sep": delimiter if delimiter is not None else None,
        "engine": engine,
        "encoding": encoding,
        "chunksize": chunksize,
        "dtype": None,
    }
    contains = eval_method_contains.lower()

    for chunk in pd.read_csv(original_csv, **read_kwargs):
        # Coerce types
        if id_col not in chunk.columns or file_col not in chunk.columns or em_col not in chunk.columns or eval_col not in chunk.columns:
            # Skip chunks that somehow miss required columns
            continue

        # Restrict to relevant ids first for efficiency
        chunk[id_col] = chunk[id_col].astype(str)
        chunk = chunk[chunk[id_col].isin(target_ids)]
        if chunk.empty:
            continue

        # Filter to bypass-like eval methods
        eval_like = chunk[eval_col].astype(str).str.contains(contains, case=False, na=False)
        chunk = chunk[eval_like]
        if chunk.empty:
            continue

        # Prepare keys and EM
        f_names = chunk[file_col].astype(str)
        keys = pd.Series(list(zip(chunk[id_col].astype(str), f_names.astype(str))), index=chunk.index)
        em = _coerce_bool(chunk[em_col]).fillna(False)

        # Update per-key maps but only for target pairs if file_name known
        if target_pairs:
            mask_pairs = keys.isin(target_pairs)
            if mask_pairs.any():
                keys_sub = keys[mask_pairs]
                em_sub = em[mask_pairs]
                for k, v in zip(keys_sub, em_sub):
                    per_key_seen_any.add(k)
                    if v:
                        per_key_seen_true[k] = True
                    else:
                        per_key_seen_true.setdefault(k, False)

        # Update id-level map
        ids_sub = chunk[id_col].astype(str)
        for i, v in zip(ids_sub, em):
            if v:
                id_seen_true[i] = True
            else:
                id_seen_true.setdefault(i, False)

    # Collapse per-key into definitive True/False only for keys observed
    per_key_pass: Dict[Tuple[str, str], bool] = {}
    for k in per_key_seen_any:
        per_key_pass[k] = bool(per_key_seen_true.get(k, False))

    id_any_pass: Dict[str, bool] = {i: bool(val) for i, val in id_seen_true.items()}
    return per_key_pass, id_any_pass


def _move_instance(entry: InstanceEntry, *, dest_root: Path, dry_run: bool) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / entry.path.name
    if dry_run:
        logger.info("DRY-RUN: would move %s -> %s", str(entry.path), str(target))
        return
    try:
        entry.path.rename(target)
    except Exception:
        shutil.move(str(entry.path), str(target))


def main(flags: Flags) -> None:
    sample_dir = flags.sample_dir
    original_csv = flags.original_csv

    if not sample_dir.exists() or not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample directory not found or not a directory: {sample_dir}")
    if not original_csv.exists():
        raise FileNotFoundError(f"Original CSV not found: {original_csv}")

    logger.info("Filtering samples in %s using %s", str(sample_dir), str(original_csv))

    # Resolve schema from header
    id_col, file_col, em_col, eval_col = _read_schema_probing(
        original_csv, encoding=flags.encoding, delimiter=flags.delimiter
    )
    logger.info(
        "Detected columns: id=%s, file_name=%s, exact_match=%s, eval_method=%s",
        id_col,
        file_col,
        em_col,
        eval_col,
    )

    # Discover instances
    instance_dirs = _list_instance_dirs(sample_dir)
    instances = [_parse_instance(p) for p in instance_dirs]
    if not instances:
        logger.warning("No instance folders found under %s", str(sample_dir))

    # Build target sets
    target_ids: Set[str] = {e.id_value for e in instances}
    target_pairs: Set[Tuple[str, str]] = {(e.id_value, e.file_name) for e in instances if e.file_name is not None}

    # Read CSV and classify
    per_key_pass, id_any_pass = _classify_from_csv(
        original_csv=original_csv,
        id_col=id_col,
        file_col=file_col,
        em_col=em_col,
        eval_col=eval_col,
        target_ids=target_ids,
        target_pairs=target_pairs,
        eval_method_contains=flags.eval_method_contains,
        encoding=flags.encoding,
        delimiter=flags.delimiter,
        chunksize=flags.chunksize,
    )

    # Decide pass/fail per instance
    pass_entries: List[InstanceEntry] = []
    fail_entries: List[InstanceEntry] = []
    unmatched: List[InstanceEntry] = []

    for e in instances:
        key = (e.id_value, e.file_name) if e.file_name is not None else None
        decision: Optional[bool] = None
        if key is not None and key in per_key_pass:
            decision = per_key_pass[key]
        else:
            # Fallback to id-level if we couldn't find a per-file mapping
            if e.id_value in id_any_pass:
                decision = id_any_pass[e.id_value]

        if decision is None:
            unmatched.append(e)
            # Default classification: treat as fail if not found
            fail_entries.append(e)
        elif decision:
            pass_entries.append(e)
        else:
            fail_entries.append(e)

    logger.info(
        "Classification complete: pass=%d, fail=%d, unmatched=%d",
        len(pass_entries),
        len(fail_entries),
        len(unmatched),
    )
    if flags.fail_on_unmatched and unmatched:
        raise RuntimeError(f"{len(unmatched)} instances could not be matched to CSV rows.")

    # Move folders
    pass_dir = sample_dir / "pass"
    fail_dir = sample_dir / "fail"
    for e in pass_entries:
        _move_instance(e, dest_root=pass_dir, dry_run=flags.dry_run)
    for e in fail_entries:
        _move_instance(e, dest_root=fail_dir, dry_run=flags.dry_run)

    # Write manifest and summaries
    manifest = {
        "sample_dir": str(sample_dir),
        "original_csv": str(original_csv),
        "eval_method_contains": flags.eval_method_contains,
        "columns": {
            "id": id_col,
            "file_name": file_col,
            "exact_match": em_col,
            "eval_method": eval_col,
        },
        "counts": {
            "total_instances": len(instances),
            "pass": len(pass_entries),
            "fail": len(fail_entries),
            "unmatched": len(unmatched),
        },
        "dry_run": flags.dry_run,
    }
    manifest_path = sample_dir / "filtered_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Convenience CSVs
    def _entries_to_df(entries: List[InstanceEntry]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "instance_dir": [e.path.name for e in entries],
                "id": [e.id_value for e in entries],
                "file_name": [e.file_name if e.file_name is not None else "" for e in entries],
            }
        )

    _entries_to_df(pass_entries).to_csv(sample_dir / "pass_instances.csv", index=False)
    _entries_to_df(fail_entries).to_csv(sample_dir / "fail_instances.csv", index=False)
    if unmatched:
        _entries_to_df(unmatched).to_csv(sample_dir / "unmatched_instances.csv", index=False)

    # Print concise summary
    print(
        {
            "sample_dir": str(sample_dir),
            "pass": len(pass_entries),
            "fail": len(fail_entries),
            "unmatched": len(unmatched),
            "manifest": str(manifest_path),
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


