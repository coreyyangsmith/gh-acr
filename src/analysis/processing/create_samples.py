"""Create a random sample of row IDs from a CSV and copy matching folders.

Usage (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.analysis.processing.create_samples data\2025_10_18_Final_Results.csv repos_parent --sample_size 100

- Direct script:
  python src\results\processing\create_samples.py data\input.csv repos_parent --sample_size 100

Behavior:
- Loads the ID column (default: "id") from the CSV
- Randomly selects the requested number of unique IDs (without replacement)
- For each selected ID, copies the child folder named exactly as the ID from the
  given parent directory into an output directory
- Writes a small manifest with details and the list of selected IDs

Notes:
- If fewer unique IDs exist than the requested sample size, all unique IDs will
  be selected.
- Missing folders are reported; in non-strict mode they are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import json
import random
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


def _detect_id_column(df: pd.DataFrame, requested: str) -> str:
    """Return the column to use as the ID, with forgiving fallbacks.

    Preference order:
    1) exact match of requested
    2) case-insensitive match of "id"
    3) common variants (contains "id", or equals "sample_id")
    """
    if requested in df.columns:
        return requested
    lowered = {c.lower(): c for c in df.columns}
    if "id" in lowered:
        return lowered["id"]
    candidates = [
        c for c in df.columns if c.lower() in {"sample_id"} or "id" in c.lower()
    ]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"ID column '{requested}' not found and no '*id*' column detected in input CSV."
    )


def _resolve_output_dir(input_csv: Path, output_dir: Optional[Path], sample_size: int) -> Path:
    if output_dir is not None:
        return output_dir
    # Default: <input_parent>/<input_stem>_samples_<N>
    return input_csv.parent / f"{input_csv.stem}_samples_{sample_size}"


@dataclass
class Flags:
    """Parameters controlling inputs, sampling, and copying behavior."""

    input_csv: Path
    parent_dir: Path
    sample_size: int

    output_dir: Optional[Path] = None
    id_column: str = "id"
    seed: int = 233541234
    encoding: str = "utf-8"
    delimiter: Optional[str] = None  # Auto-detect when None
    strict: bool = False  # When True, error if any ID folder is missing
    workers: int = 8  # Threaded copies; I/O bound


def _read_id_series(input_csv: Path, encoding: str, delimiter: Optional[str], requested_id_col: str) -> Tuple[pd.Series, str]:
    """Read and return the ID column as strings, along with the detected column name."""
    # Read only the header first to detect the actual column name
    header_df = pd.read_csv(
        input_csv,
        sep=delimiter if delimiter is not None else None,
        engine="python" if delimiter is None else "c",
        encoding=encoding,
        nrows=0,
    )
    id_col = _detect_id_column(header_df, requested_id_col)

    series = pd.read_csv(
        input_csv,
        sep=delimiter if delimiter is not None else None,
        engine="python" if delimiter is None else "c",
        encoding=encoding,
        usecols=[id_col],
        dtype={id_col: str},
    )[id_col].astype(str)

    # Normalize common oddities
    series = series.fillna("").str.strip()
    non_empty = series != ""
    series = series[non_empty]
    return series, id_col


def _choose_sample_ids(all_ids: Iterable[str], sample_size: int, seed: int) -> List[str]:
    unique_ids = list(dict.fromkeys(all_ids))  # preserve order while deduping
    if not unique_ids:
        return []
    if sample_size >= len(unique_ids):
        return unique_ids
    rng = random.Random(seed)
    return rng.sample(unique_ids, k=sample_size)


def _copy_one_folder(src_parent: Path, dst_parent: Path, id_value: str) -> Tuple[str, bool, Optional[str]]:
    """Copy one folder named exactly as `id_value`. Returns (id, success, error)."""
    src = src_parent / id_value
    if not src.exists() or not src.is_dir():
        return id_value, False, "source_missing"
    dst = dst_parent / id_value
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        # After copying, normalize the instance layout per project conventions
        _normalize_instance_layout(dst)
        # Then split into per-file instances named ID-1, ID-2, ... when applicable
        _split_instance_per_file(dst_parent, dst, id_value)
        return id_value, True, None
    except Exception as exc:  # pragma: no cover - unlikely platform-specific errors
        return id_value, False, str(exc)


def _copy_folders_parallel(ids: List[str], parent_dir: Path, output_dir: Path, workers: int) -> Tuple[int, List[str], List[Tuple[str, str]]]:
    """Copy ID-named folders in parallel.

    Returns (copied_count, missing_ids, failed_with_errors)
    where failed_with_errors is a list of (id, error_message).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_ids: List[str] = []
    failed: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_copy_one_folder, parent_dir, output_dir, id_value): id_value
            for id_value in ids
        }
        for future in as_completed(futures):
            id_value = futures[future]
            try:
                _id, ok, error = future.result()
                if ok:
                    copied_count += 1
                else:
                    if error == "source_missing":
                        missing_ids.append(id_value)
                    else:
                        failed.append((id_value, error or "unknown_error"))
            except Exception as exc:  # pragma: no cover - defensive
                failed.append((id_value, str(exc)))

    return copied_count, missing_ids, failed


def _merge_directories(src: Path, dst: Path) -> None:
    """Recursively merge the contents of `src` into `dst`. Creates `dst` if needed."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _normalize_instance_layout(instance_root: Path) -> None:
    """Standardize per-instance folder layout.

    - Rename/merge `bypass7` into `bypass`.
    - Create `default/` and move top-level directories that look like file names
      (contain a dot, e.g. `swift_version.py`) into it.
    Result: each instance ideally has only `agent/`, `bypass/`, and `default/` at the top level.
    """
    if not instance_root.exists():
        return

    # 1) Rename or merge bypass7 -> bypass
    bypass7_dir = instance_root / "bypass7"
    bypass_dir = instance_root / "bypass"
    if bypass7_dir.exists() and bypass7_dir.is_dir():
        if bypass_dir.exists() and bypass_dir.is_dir():
            _merge_directories(bypass7_dir, bypass_dir)
            shutil.rmtree(bypass7_dir, ignore_errors=True)
        else:
            try:
                bypass7_dir.rename(bypass_dir)
            except Exception:
                shutil.move(str(bypass7_dir), str(bypass_dir))

    # 2) Create default/ and move file-name-like folders into it
    default_dir = instance_root / "default"
    default_dir.mkdir(exist_ok=True)

    # Snapshot children to avoid iterator invalidation during moves
    for child in list(instance_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name in {"agent", "bypass", "default"}:
            continue
        # Heuristic: default-run folders are named like file names and contain a dot
        if "." in name:
            destination = default_dir / name
            if destination.exists() and destination.is_dir():
                _merge_directories(child, destination)
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.rename(destination)
                except Exception:
                    shutil.move(str(child), str(destination))


def _split_instance_per_file(dst_parent: Path, instance_root: Path, id_value: str) -> None:
    """Split a copied instance into per-file sub-instances named ID-1, ID-2, ...

    The function assumes the instance has been normalized so that:
    - `bypass7` has already been merged/renamed to `bypass`
    - `default/` contains one subdirectory per file unit (e.g., `swift_version.py`)

    Behavior:
    - If 0 or 1 file units are present, rename the instance to `ID-1`.
    - If multiple file units are present, create sibling instances `ID-1`, `ID-2`, ...
      Each new instance contains copies of `agent/` and `bypass/` (if present) and a
      `default/` folder containing exactly one file unit. The original instance is removed.
    """
    try:
        default_dir = instance_root / "default"
        agent_dir = instance_root / "agent"
        bypass_dir = instance_root / "bypass"

        file_units: List[Path] = []
        if default_dir.exists() and default_dir.is_dir():
            file_units = [p for p in default_dir.iterdir() if p.is_dir()]
            file_units.sort(key=lambda p: p.name.lower())

        # Helper: copy only the matching agent file and bypass subfolder
        def _copy_agent_and_bypass_for_unit(unit_name: str, src_root: Path, dst_instance: Path) -> None:
            src_agent_dir = src_root / "agent"
            src_bypass_dir = src_root / "bypass"

            # Agent: copy agent/<unit_name> or agent/<unit_name>.txt if present
            if src_agent_dir.exists() and src_agent_dir.is_dir():
                direct = src_agent_dir / unit_name
                txt_variant = src_agent_dir / f"{unit_name}.txt"
                src_agent_item = direct if direct.exists() else (txt_variant if txt_variant.exists() else None)
                if src_agent_item is not None:
                    dest_agent_dir = dst_instance / "agent"
                    dest_agent_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_agent_dir / src_agent_item.name
                    if src_agent_item.is_dir():
                        shutil.copytree(src_agent_item, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_agent_item, dest_path)

            # Bypass: copy bypass/<unit_name> if present (prefer dir, else file)
            if src_bypass_dir.exists() and src_bypass_dir.is_dir():
                src_bypass_item = src_bypass_dir / unit_name
                if src_bypass_item.exists():
                    dest_bypass_dir = dst_instance / "bypass"
                    dest_bypass_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_bypass_dir / src_bypass_item.name
                    if src_bypass_item.is_dir():
                        shutil.copytree(src_bypass_item, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_bypass_item, dest_path)

        # 0 or 1 file unit: rename to ID-1 and prune agent/bypass to only matching unit (if present)
        if len(file_units) <= 1:
            unit_name = file_units[0].name if len(file_units) == 1 else None
            target = dst_parent / f"{id_value}-1"
            if instance_root != target:
                try:
                    instance_root.rename(target)
                    instance_root = target
                except Exception:
                    shutil.move(str(instance_root), str(target))
                    instance_root = target

            if unit_name is not None:
                # Prune agent to only the matching item(s): <unit_name> or <unit_name>.txt
                current_agent_dir = instance_root / "agent"
                if current_agent_dir.exists() and current_agent_dir.is_dir():
                    keep_names = {unit_name, f"{unit_name}.txt"}
                    for child in list(current_agent_dir.iterdir()):
                        if child.name not in keep_names:
                            if child.is_dir():
                                shutil.rmtree(child, ignore_errors=True)
                            else:
                                try:
                                    child.unlink()
                                except Exception:
                                    pass

                # Prune bypass to only the matching subfolder
                current_bypass_dir = instance_root / "bypass"
                if current_bypass_dir.exists() and current_bypass_dir.is_dir():
                    for child in list(current_bypass_dir.iterdir()):
                        if child.name != unit_name:
                            if child.is_dir():
                                shutil.rmtree(child, ignore_errors=True)
                            else:
                                try:
                                    child.unlink()
                                except Exception:
                                    pass
            return

        # Multiple file units: create per-file instances and remove the original
        for index, unit in enumerate(file_units, start=1):
            new_instance = dst_parent / f"{id_value}-{index}"
            new_instance.mkdir(parents=True, exist_ok=True)

            # Copy only the matching agent file and bypass subfolder into the new instance
            _copy_agent_and_bypass_for_unit(unit.name, instance_root, new_instance)

            # Move this file unit into new_instance/default/<unit.name>
            new_default = new_instance / "default"
            new_default.mkdir(exist_ok=True)
            destination = new_default / unit.name
            try:
                unit.rename(destination)
            except Exception:
                shutil.move(str(unit), str(destination))

        # Remove the now-empty original instance folder (best-effort)
        shutil.rmtree(instance_root, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - defensive; do not stop the whole pipeline
        logger.warning("Per-file split failed for %s: %s", str(instance_root), str(exc))


def _write_manifest(
    manifest_path: Path,
    *,
    input_csv: Path,
    parent_dir: Path,
    output_dir: Path,
    id_column: str,
    sample_size: int,
    seed: int,
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
        "sample_size_requested": sample_size,
        "seed": seed,
        "selected_ids": selected_ids,
        "copied_count": copied_count,
        "missing_ids": missing_ids,
        "failed": [{"id": i, "error": e} for i, e in failed],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Also emit a simple CSV of the selected IDs for convenience
    ids_csv = output_dir / "sample_ids.csv"
    pd.DataFrame({id_column: selected_ids}).to_csv(ids_csv, index=False)


def main(flags: Flags) -> None:
    input_csv = flags.input_csv
    parent_dir = flags.parent_dir

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise NotADirectoryError(f"Parent directory not found or not a directory: {parent_dir}")

    output_dir = _resolve_output_dir(input_csv, flags.output_dir, flags.sample_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Creating sample: input=%s, parent_dir=%s, output_dir=%s, size=%d, seed=%d",
        str(input_csv),
        str(parent_dir),
        str(output_dir),
        flags.sample_size,
        flags.seed,
    )

    id_series, detected_id_col = _read_id_series(
        input_csv=input_csv,
        encoding=flags.encoding,
        delimiter=flags.delimiter,
        requested_id_col=flags.id_column,
    )

    selected_ids = _choose_sample_ids(id_series.tolist(), flags.sample_size, flags.seed)
    if not selected_ids:
        raise ValueError("No IDs found in input CSV after filtering. Nothing to sample.")

    logger.info("Selected %d IDs (requested %d)", len(selected_ids), flags.sample_size)

    copied_count, missing_ids, failed = _copy_folders_parallel(
        selected_ids, parent_dir, output_dir, flags.workers
    )

    if (missing_ids or failed) and flags.strict:
        missing_preview = ", ".join(missing_ids[:5]) + ("..." if len(missing_ids) > 5 else "")
        failed_preview = ", ".join([f"{i}:{e}" for i, e in failed[:3]]) + ("..." if len(failed) > 3 else "")
        raise FileNotFoundError(
            f"Strict mode: some IDs missing or failed to copy. Missing: [{missing_preview}] | Failed: [{failed_preview}]"
        )

    # Export full rows for selected IDs to a new CSV
    selected_rows_csv = output_dir / "sample_rows.csv"
    # Stream through the CSV to avoid loading it fully into memory
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
            # Handle edge case where some chunks somehow miss column detection
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
        sample_size=flags.sample_size,
        seed=flags.seed,
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


