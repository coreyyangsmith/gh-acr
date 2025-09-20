from __future__ import annotations

"""Split counts by difficulty and project size, exporting CSV and Markdown.

Project size can be provided directly or derived from lines-of-code columns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import tyro


# =========================
# Helpers
# =========================

SIZE_LABELS = ["Tiny", "Small", "Medium", "Large", "Huge"]


def _size_from_loc(loc: float) -> str:
    """Map a (non-negative) LOC count to a size bucket."""
    try:
        v = float(loc)
    except Exception:
        return "Unknown"
    if not np.isfinite(v) or v < 0:
        return "Unknown"
    if v < 1_000:
        return "Tiny"
    if v < 10_000:
        return "Small"
    if v < 100_000:
        return "Medium"
    if v < 1_000_000:
        return "Large"
    return "Huge"


def _normalize_project_size_column(df: pd.DataFrame, *, project_size_col: Optional[str], allow_derive_from_loc: bool) -> pd.Series:
    """Normalize a project size column or derive it from LOC if permitted."""
    # Use project_size column from CSV (do not compute from LOC unless explicitly allowed)
    col = project_size_col if project_size_col else ("project_size" if "project_size" in df.columns else None)
    if col is not None and col in df.columns:
        series = df[col].astype(str).str.strip()
        mapping = {"tiny": "Tiny", "small": "Small", "medium": "Medium", "large": "Large", "huge": "Huge"}
        normalized = series.str.lower().map(lambda s: mapping.get(s, "Unknown"))
        return normalized

    if not allow_derive_from_loc:
        raise ValueError("project_size column not found. Provide --project-size-col or enable --allow-derive-from-loc.")

    # Optional fallback: derive from LOC if allowed
    if "code_lines" in df.columns:
        loc = pd.to_numeric(df["code_lines"], errors="coerce").clip(lower=0)
        for extra in ["blank_lines", "comment_lines"]:
            if extra in df.columns:
                loc_extra = pd.to_numeric(df[extra], errors="coerce").clip(lower=0)
                loc = loc.add(loc_extra.fillna(0), fill_value=0)
        return loc.map(_size_from_loc)

    any_line_cols = [c for c in df.columns if c.endswith("_lines")]
    if any_line_cols:
        totals = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").clip(lower=0) for c in any_line_cols}).sum(axis=1)
        return totals.map(_size_from_loc)

    return pd.Series(["Unknown"] * len(df))


def _save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a simple GitHub-flavored markdown table to `path`."""
    if df is None or df.empty:
        return
    cols = [str(c) for c in df.columns]

    def _fmt(v: object) -> str:
        if pd.isna(v):
            return ""
        return str(int(v)) if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()) else str(v)

    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = [_fmt(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# CLI
# =========================


@dataclass
class Flags:
    """Arguments controlling inputs and normalization behavior."""
    input_csv: Path
    output_dir: Path = Path("results")
    difficulty_col: str = "difficulty"
    # If the CSV uses a different column name for project size categories
    project_size_col: Optional[str] = None
    # By default we require project_size to be present; set True to derive from LOC if missing
    allow_derive_from_loc: bool = False


def main(flags: Flags) -> None:
    """Compute a difficulty×project size count table and save CSV/Markdown."""
    flags.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(flags.input_csv)

    # Determine difficulty column
    if flags.difficulty_col not in df.columns:
        # try lowercase variant
        alt = None
        for c in df.columns:
            if str(c).strip().lower() == "difficulty":
                alt = c
                break
        if alt is None:
            raise ValueError(f"Difficulty column '{flags.difficulty_col}' not found in {flags.input_csv}")
        flags.difficulty_col = alt

    # Determine project size categories from CSV
    size_series = _normalize_project_size_column(
        df,
        project_size_col=flags.project_size_col,
        allow_derive_from_loc=flags.allow_derive_from_loc,
    )

    # Build a normalized working frame
    work = pd.DataFrame({
        "difficulty": df[flags.difficulty_col].astype(str).str.strip(),
        "project_size": size_series,
    })

    # Enforce categorical order for size; difficulty will be alphabetical by default
    work["project_size"] = pd.Categorical(work["project_size"], categories=SIZE_LABELS + ["Unknown"], ordered=True)

    # Compute counts via groupby to avoid pivot_table values=None incompatibilities
    counts = (
        work.groupby(["difficulty", "project_size"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    pivot = counts.reset_index()

    # Difficulty ordering: prefer easy, medium, hard if present
    preferred_order = ["easy", "medium", "hard"]
    present_diffs = [str(d) for d in work["difficulty"].unique().tolist()]
    ordered = [d for d in preferred_order if d in present_diffs] + [d for d in sorted(present_diffs) if d not in preferred_order]
    pivot = pivot.set_index("difficulty").reindex(ordered).reset_index()

    # Move Unknown to the end if present
    cols = list(pivot.columns)
    if "Unknown" in cols:
        fixed_cols = [c for c in cols if c != "Unknown"] + ["Unknown"]
        pivot = pivot[fixed_cols]

    # Save CSV and Markdown
    out_csv = flags.output_dir / "difficulty_by_project_size.csv"
    out_md = flags.output_dir / "difficulty_by_project_size.md"
    pivot.to_csv(out_csv, index=False)
    _save_markdown_table(pivot, out_md)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


