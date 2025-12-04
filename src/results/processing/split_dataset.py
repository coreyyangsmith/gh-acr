from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    Create pass and fail copies from a parent directory based on bypass exact_match in a CSV.

    Originals are left in place; we copy matching folders into new SIBLING directories named
    '<parent_name>_pass' and '<parent_name>_fail' beside the given parent directory.

    Typical usage (from repo root):
      python -m src.results.processing.split_dataset data\\results_all.csv path\\to\\parent_dir
    """

    # CSV with per-file result rows that include id, file_name, eval_method, exact_match
    source_csv: Path
    # Directory containing either ID folders or per-file instance folders (ID-1, ID-2, ...)
    parent_dir: Path

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

    # Copy behavior
    workers: int = 8
    dry_run: bool = False
    fail_on_unmatched: bool = False  # If True, error if any folder cannot be matched


@dataclass(frozen=True)
class Entry:
    """Represents a top-level folder under parent_dir."""

    path: Path
    id_value: str
    file_name: Optional[str]  # Derived from default/<unit_name>/ if present (for ID-N)
    is_instance: bool  # True if looks like ID-<n>, else assumed ID folder


def _list_top_level_dirs(parent_dir: Path) -> List[Path]:
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise NotADirectoryError(f"Parent directory not found or not a directory: {parent_dir}")
    excluded = {"pass", "fail", "__pycache__"}
    out: List[Path] = []
    for child in parent_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in excluded:
            continue
        out.append(child)
    return out


def _parse_entry(path: Path) -> Entry:
    """Parse id and optional unit from folder name and structure."""
    name = path.name
    is_instance = False
    id_value = name
    if "-" in name:
        parts = name.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            id_value = parts[0]
            is_instance = True
    unit_name: Optional[str] = None
    default_dir = path / "default"
    if default_dir.exists() and default_dir.is_dir():
        subdirs = [p for p in default_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            unit_name = subdirs[0].name
    return Entry(path=path, id_value=id_value, file_name=unit_name, is_instance=is_instance)


def _read_schema(source_csv: Path, *, encoding: str, delimiter: Optional[str]) -> Tuple[str, str, str, str]:
    header_df = pd.read_csv(
        source_csv,
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


def _classify(
    source_csv: Path,
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
    Build:
      - per_key_pass[(id, file_name)] = True/False for observed pairs under bypass-like methods
      - id_any_pass[id] = True/False based on any bypass-like row across files
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

    for chunk in pd.read_csv(source_csv, **read_kwargs):
        if id_col not in chunk.columns or file_col not in chunk.columns or em_col not in chunk.columns or eval_col not in chunk.columns:
            continue
        chunk[id_col] = chunk[id_col].astype(str)
        chunk = chunk[chunk[id_col].isin(target_ids)]
        if chunk.empty:
            continue
        mask_bypass = chunk[eval_col].astype(str).str.contains(contains, case=False, na=False)
        chunk = chunk[mask_bypass]
        if chunk.empty:
            continue
        keys = pd.Series(list(zip(chunk[id_col].astype(str), chunk[file_col].astype(str))), index=chunk.index)
        em = _coerce_bool(chunk[em_col]).fillna(False)
        if target_pairs:
            mask_pairs = keys.isin(target_pairs)
            if mask_pairs.any():
                for k, v in zip(keys[mask_pairs], em[mask_pairs]):
                    per_key_seen_any.add(k)
                    if v:
                        per_key_seen_true[k] = True
                    else:
                        per_key_seen_true.setdefault(k, False)
        for i, v in zip(chunk[id_col].astype(str), em):
            if v:
                id_seen_true[i] = True
            else:
                id_seen_true.setdefault(i, False)

    per_key_pass: Dict[Tuple[str, str], bool] = {k: bool(per_key_seen_true.get(k, False)) for k in per_key_seen_any}
    id_any_pass: Dict[str, bool] = {i: bool(val) for i, val in id_seen_true.items()}
    return per_key_pass, id_any_pass


def _copy_folder(src: Path, dst_root: Path, *, dry_run: bool) -> Tuple[str, bool, Optional[str]]:
    """Copy src folder into dst_root/src.name. Returns (name, ok, error)."""
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dry_run:
        logger.info("DRY-RUN: would copy %s -> %s", str(src), str(dst))
        return (src.name, True, None)
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return (src.name, True, None)
    except Exception as exc:
        return (src.name, False, str(exc))


def _copy_many(entries: List[Entry], *, dest_root: Path, workers: int, dry_run: bool) -> Tuple[int, List[Tuple[str, str]]]:
    """Copy many folders in parallel. Returns (copied_count, failed[(name, error)])."""
    copied = 0
    failed: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_copy_folder, e.path, dest_root, dry_run=dry_run): e for e in entries}
        for fut in as_completed(futures):
            name, ok, err = fut.result()
            if ok:
                copied += 1
            else:
                failed.append((name, err or "unknown_error"))
    return copied, failed


def _write_pass_fail_csv_rows(
    source_csv: Path,
    *,
    pass_csv_path: Path,
    fail_csv_path: Path,
    id_col: str,
    file_col: str,
    pass_keys: Set[Tuple[str, str]],
    fail_keys: Set[Tuple[str, str]],
    id_pass_ids: Set[str],
    id_fail_ids: Set[str],
    target_ids: Set[str],
    encoding: str,
    delimiter: Optional[str],
    chunksize: int,
) -> Tuple[int, int]:
    """
    Stream the source CSV and write out two CSVs that contain the original rows for ALL eval methods:
      - pass_csv: rows whose (id, file_name) is in pass_keys (when file_name present), else whose id is in id_pass_ids
      - fail_csv: rows whose (id, file_name) is in fail_keys (when file_name present), else whose id is in id_fail_ids
    Rows must have id in target_ids. Rows that do not match any rule fall back to FAIL.
    Returns (num_pass_rows, num_fail_rows).
    """
    # Reset output files if they already exist
    try:
        if pass_csv_path.exists():
            pass_csv_path.unlink()
    except Exception:
        pass
    try:
        if fail_csv_path.exists():
            fail_csv_path.unlink()
    except Exception:
        pass

    engine = "python" if delimiter is None else "c"
    read_kwargs = {
        "sep": delimiter if delimiter is not None else None,
        "engine": engine,
        "encoding": encoding,
        "chunksize": chunksize,
        "dtype": None,
    }
    header_pass_written = False
    header_fail_written = False
    n_pass = 0
    n_fail = 0

    for chunk in pd.read_csv(source_csv, **read_kwargs):
        if id_col not in chunk.columns:
            continue
        # Restrict to target ids
        chunk[id_col] = chunk[id_col].astype(str)
        chunk = chunk[chunk[id_col].isin(target_ids)]
        if chunk.empty:
            continue

        # Build key membership masks
        if file_col in chunk.columns:
            file_series = chunk[file_col].astype(str)
            has_file = (~chunk[file_col].isna()) & (file_series.str.len() > 0)
            keys_series = pd.Series(list(zip(chunk[id_col].astype(str), file_series)), index=chunk.index)
            key_pass_mask = has_file & keys_series.isin(pass_keys)  # type: ignore[arg-type]
            key_fail_mask = has_file & keys_series.isin(fail_keys)  # type: ignore[arg-type]
            key_known_mask = has_file & keys_series.isin(pass_keys.union(fail_keys))  # type: ignore[arg-type]
        else:
            key_pass_mask = pd.Series(False, index=chunk.index)
            key_fail_mask = pd.Series(False, index=chunk.index)
            key_known_mask = pd.Series(False, index=chunk.index)

        id_pass_mask = chunk[id_col].isin(id_pass_ids)
        id_fail_mask = chunk[id_col].isin(id_fail_ids)

        # Apply precedence: per-file key decision if known; otherwise fall back to id-level.
        pass_mask = key_pass_mask | (~key_known_mask & id_pass_mask)
        fail_mask = key_fail_mask | (~key_known_mask & id_fail_mask) | (~key_known_mask & ~id_pass_mask & ~id_fail_mask)

        pass_rows = chunk[pass_mask]
        fail_rows = chunk[fail_mask]

        if not pass_rows.empty:
            pass_rows.to_csv(
                pass_csv_path,
                mode="a",
                index=False,
                header=not header_pass_written,
                encoding=encoding,
            )
            header_pass_written = True
            n_pass += int(len(pass_rows))

        if not fail_rows.empty:
            fail_rows.to_csv(
                fail_csv_path,
                mode="a",
                index=False,
                header=not header_fail_written,
                encoding=encoding,
            )
            header_fail_written = True
            n_fail += int(len(fail_rows))

    return n_pass, n_fail


def main(flags: Flags) -> None:
    source_csv = flags.source_csv
    parent_dir = flags.parent_dir
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise NotADirectoryError(f"Parent directory not found or not a directory: {parent_dir}")

    logger.info("Splitting dataset from %s based on %s", str(parent_dir), str(source_csv))

    id_col, file_col, em_col, eval_col = _read_schema(
        source_csv, encoding=flags.encoding, delimiter=flags.delimiter
    )
    logger.info(
        "Detected columns: id=%s, file_name=%s, exact_match=%s, eval_method=%s",
        id_col,
        file_col,
        em_col,
        eval_col,
    )

    # Discover top-level entries
    top_dirs = _list_top_level_dirs(parent_dir)
    entries = [_parse_entry(p) for p in top_dirs]
    if not entries:
        logger.warning("No top-level folders found under %s", str(parent_dir))

    # Build target sets for classification
    target_ids: Set[str] = {e.id_value for e in entries}
    target_pairs: Set[Tuple[str, str]] = {(e.id_value, e.file_name) for e in entries if e.file_name is not None}

    per_key_pass, id_any_pass = _classify(
        source_csv=source_csv,
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

    # Classify entries
    pass_entries: List[Entry] = []
    fail_entries: List[Entry] = []
    unmatched: List[Entry] = []
    for e in entries:
        key = (e.id_value, e.file_name) if e.file_name is not None else None
        decision: Optional[bool] = None
        if key is not None and key in per_key_pass:
            decision = per_key_pass[key]
        else:
            if e.id_value in id_any_pass:
                decision = id_any_pass[e.id_value]
        if decision is None:
            unmatched.append(e)
            fail_entries.append(e)  # default to fail when no evidence
        elif decision:
            pass_entries.append(e)
        else:
            fail_entries.append(e)

    logger.info(
        "Classification: pass=%d, fail=%d, unmatched=%d",
        len(pass_entries),
        len(fail_entries),
        len(unmatched),
    )
    if flags.fail_on_unmatched and unmatched:
        raise RuntimeError(f"{len(unmatched)} folders could not be matched to CSV rows.")

    # Copy into new sibling '<parent>_pass' and '<parent>_fail' folders beside parent_dir
    parent_sibling_root = parent_dir.parent
    pass_dir = parent_sibling_root / f"{parent_dir.name}_pass"
    fail_dir = parent_sibling_root / f"{parent_dir.name}_fail"
    copied_pass, failed_pass = _copy_many(pass_entries, dest_root=pass_dir, workers=flags.workers, dry_run=flags.dry_run)
    copied_fail, failed_fail = _copy_many(fail_entries, dest_root=fail_dir, workers=flags.workers, dry_run=flags.dry_run)

    # Manifest + summaries
    pass_csv_path = parent_sibling_root / f"{source_csv.stem}_pass.csv"
    fail_csv_path = parent_sibling_root / f"{source_csv.stem}_fail.csv"
    # Derive pass/fail sets for CSV writing (all methods)
    pass_keys: Set[Tuple[str, str]] = {k for k, v in per_key_pass.items() if v}
    fail_keys: Set[Tuple[str, str]] = {k for k, v in per_key_pass.items() if not v}
    id_pass_ids: Set[str] = {i for i, v in id_any_pass.items() if v}
    id_fail_ids: Set[str] = {i for i, v in id_any_pass.items() if not v}
    manifest = {
        "parent_dir": str(parent_dir),
        "source_csv": str(source_csv),
        "eval_method_contains": flags.eval_method_contains,
        "columns": {
            "id": id_col,
            "file_name": file_col,
            "exact_match": em_col,
            "eval_method": eval_col,
        },
        "output_dirs": {
            "pass_dir": str(pass_dir),
            "fail_dir": str(fail_dir),
        },
        "output_csvs": {
            "pass_csv": str(pass_csv_path),
            "fail_csv": str(fail_csv_path),
        },
        "counts": {
            "total_top_level_folders": len(entries),
            "pass": len(pass_entries),
            "fail": len(fail_entries),
            "unmatched": len(unmatched),
            "copied_pass": copied_pass,
            "copied_fail": copied_fail,
            "failed_pass": len(failed_pass),
            "failed_fail": len(failed_fail),
        },
        "dry_run": flags.dry_run,
    }
    manifest_path = parent_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _to_df(items: List[Entry]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "folder": [e.path.name for e in items],
                "id": [e.id_value for e in items],
                "file_name": [e.file_name if e.file_name is not None else "" for e in items],
                "is_instance": [e.is_instance for e in items],
            }
        )

    # Write requested CSV outputs beside the parent directory with original filename stem,
    # copying the original rows for ALL eval methods, partitioned by pass/fail instance membership
    n_pass_rows, n_fail_rows = _write_pass_fail_csv_rows(
        source_csv=source_csv,
        pass_csv_path=pass_csv_path,
        fail_csv_path=fail_csv_path,
        id_col=id_col,
        file_col=file_col,
        pass_keys=pass_keys,
        fail_keys=fail_keys,
        id_pass_ids=id_pass_ids,
        id_fail_ids=id_fail_ids,
        target_ids=target_ids,
        encoding=flags.encoding,
        delimiter=flags.delimiter,
        chunksize=flags.chunksize,
    )

    if failed_pass or failed_fail:
        logger.warning("Some copies failed: pass=%d, fail=%d", len(failed_pass), len(failed_fail))

    print(
        {
            "parent_dir": str(parent_dir),
            "pass_dir": str(pass_dir),
            "fail_dir": str(fail_dir),
            "manifest": str(manifest_path),
            "pass_csv": str(pass_csv_path),
            "fail_csv": str(fail_csv_path),
            "pass_rows": n_pass_rows,
            "fail_rows": n_fail_rows,
            "pass": len(pass_entries),
            "fail": len(fail_entries),
            "unmatched": len(unmatched),
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


