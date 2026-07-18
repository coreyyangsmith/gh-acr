from __future__ import annotations

"""
Verify that each unique instance `id` from a CSV has a corresponding subfolder under a parent directory.

Usage (from repo root):
  python -m src.analysis.processing.verify_repo_counts data\\results.csv D:\\path\\to\\instances
  python src\\results\\processing\\verify_repo_counts.py data\\results.csv D:\\path\\to\\instances

Behavior:
- Reads the CSV, detects an ID column (default 'id', with simple auto-detection fallbacks).
- Gathers unique IDs from that column.
- Lists immediate subdirectories of the provided parent directory.
- Prints counts and lists for:
  - Total unique IDs
  - Total folders under parent
  - Matching folders (IDs that exist as subfolders)
  - Missing folders (IDs without a subfolder)
  - Extra folders (subfolders not present in the CSV)
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def _normalize_header(name: str) -> str:
    """Lowercase and strip non-alphanumerics commonly used in id column names."""
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_")


def _detect_id_column(header: List[str], preferred: str | None) -> str:
    """
    Detect the ID column from the header.
    Priority:
      1) Explicit preferred name if present
      2) Common variants: 'id', 'instance_id', 'instanceid'
    Raises ValueError if none are present.
    """
    header_set = set(header)
    if preferred and preferred in header_set:
        return preferred

    normalized_map = {col: _normalize_header(col) for col in header}
    # try exact 'id' first
    for col, norm in normalized_map.items():
        if norm == "id":
            return col
    # common alternates
    for candidate in ("instance_id", "instanceid"):
        for col, norm in normalized_map.items():
            if norm == candidate:
                return col

    raise ValueError(
        "Could not detect an ID column. "
        f"Available columns: {sorted(header)}. "
        "Pass --id-column to specify explicitly."
    )


def _read_unique_ids(csv_path: Path, id_column: str | None) -> Tuple[Set[str], str]:
    """Read CSV and return unique ID strings plus the resolved id column name."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        header = [h.strip() for h in reader.fieldnames]
        resolved_col = _detect_id_column(header, id_column)

        unique_ids: Set[str] = set()
        for row in reader:
            raw = row.get(resolved_col, "")
            if raw is None:
                continue
            value = str(raw).strip()
            if value != "":
                unique_ids.add(value)

    return unique_ids, resolved_col


def _list_immediate_subdirs(parent: Path) -> Set[str]:
    """Return the set of immediate subdirectory names under parent."""
    if not parent.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Parent path is not a directory: {parent}")
    return {p.name for p in parent.iterdir() if p.is_dir()}


def _print_list(label: str, items: Iterable[str], limit: int = 0) -> None:
    items_list = sorted(items, key=lambda s: s.lower())
    count = len(items_list)
    print(f"{label}: {count}")
    if count == 0:
        return
    if limit and count > limit:
        preview = items_list[:limit]
        print(f"  first {limit}:")
        for name in preview:
            print(f"    {name}")
        print("  ...")
    else:
        for name in items_list:
            print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that each unique instance ID in a CSV has a corresponding subfolder."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "parent_dir", type=Path, help="Parent directory containing per-instance subfolders."
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Explicit ID column name in the CSV. Defaults to auto-detection (prefers 'id').",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="Max items to print for each list (0 = no limit). Default: 50",
    )
    args = parser.parse_args()

    unique_ids, resolved_col = _read_unique_ids(args.csv_path, args.id_column)
    subfolders = _list_immediate_subdirs(args.parent_dir)

    ids_present_as_folders = unique_ids.intersection(subfolders)
    ids_missing_folders = unique_ids.difference(subfolders)
    extra_folders = subfolders.difference(unique_ids)

    print(
        {
            "csv": str(args.csv_path),
            "parent_dir": str(args.parent_dir),
            "id_column": resolved_col,
            "unique_ids_total": len(unique_ids),
            "folders_total": len(subfolders),
            "matching_folders": len(ids_present_as_folders),
            "missing_folders": len(ids_missing_folders),
            "extra_folders": len(extra_folders),
        }
    )

    # Detailed lists (limited)
    _print_list("Missing folder(s) for ID(s)", ids_missing_folders, limit=args.list_limit)
    _print_list("Matching folder(s) (ID exists as folder)", ids_present_as_folders, limit=args.list_limit)
    _print_list("Extra folder(s) (no matching ID in CSV)", extra_folders, limit=args.list_limit)


if __name__ == "__main__":
    main()


