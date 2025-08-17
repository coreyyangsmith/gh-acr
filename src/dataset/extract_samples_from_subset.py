from __future__ import annotations

"""Extract rows from a source CSV whose IDs appear in another CSV.

Usage examples (PowerShell):
    # All params are flags (no positionals)
    python -m src.dataset.extract_samples_from_subset --ids-csv data/ids.csv --source-csv data/source.csv --output-csv data/source_filtered.csv --ids-column id --source-id-column id

    # Case-insensitive match on stringified IDs; preserve leading index column
    python -m src.dataset.extract_samples_from_subset --ids-csv data/ids.csv --source-csv data/source.csv --case-insensitive true --preserve-index-col true
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import tyro


def _normalize_ids(
    series: pd.Series,
    *,
    coerce_to_string: bool,
    trim_whitespace: bool,
    case_insensitive: bool,
) -> pd.Series:
    """Normalize an ID column for robust matching across CSVs."""
    result = series
    if coerce_to_string:
        result = result.astype(str)
    if trim_whitespace and pd.api.types.is_string_dtype(result):
        result = result.str.strip()
    if case_insensitive and pd.api.types.is_string_dtype(result):
        result = result.str.lower()
    return result


def _derive_default_output_path(source_csv: Path, ids_csv: Path) -> Path:
    src = source_csv.expanduser().resolve()
    ids = ids_csv.expanduser().resolve()
    return src.with_name(f"{src.stem}_extracted_by_{ids.stem}.csv")


def extract_rows_by_ids(
    *,
    ids_csv: str | Path,
    source_csv: str | Path,
    output_csv: Optional[str | Path] = None,
    ids_column: str = "id",
    source_id_column: str = "id",
    case_insensitive: bool = False,
    trim_whitespace: bool = True,
    coerce_to_string: bool = True,
    preserve_index_col: bool = True,
    drop_duplicates_in_ids: bool = True,
) -> Path:
    """Filter ``source_csv`` rows whose ``source_id_column`` is in ``ids_csv``'s ``ids_column``.

    Returns the path to the written CSV.
    """

    ids_path = Path(ids_csv).expanduser().resolve()
    src_path = Path(source_csv).expanduser().resolve()
    if output_csv is None:
        out_path = _derive_default_output_path(src_path, ids_path)
    else:
        out_path = Path(output_csv).expanduser().resolve()

    if not ids_path.exists():
        raise FileNotFoundError(f"IDs CSV not found: {ids_path}")
    if not src_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {src_path}")

    # Load IDs ---------------------------------------------------------------
    ids_df = pd.read_csv(ids_path)
    if ids_column not in ids_df.columns:
        raise KeyError(
            f"Column '{ids_column}' not found in IDs CSV. Available: {list(ids_df.columns)}"
        )
    ids_series = _normalize_ids(
        ids_df[ids_column],
        coerce_to_string=coerce_to_string,
        trim_whitespace=trim_whitespace,
        case_insensitive=case_insensitive,
    )
    if drop_duplicates_in_ids:
        ids_series = ids_series.drop_duplicates()
    id_set = set(ids_series.tolist())

    # Load source and filter -------------------------------------------------
    # If preserving the leading index column convention, read the first column as index
    read_kwargs = {"index_col": 0} if preserve_index_col else {}
    src_df = pd.read_csv(src_path, **read_kwargs)
    if source_id_column not in src_df.columns:
        raise KeyError(
            f"Column '{source_id_column}' not found in source CSV. Available: {list(src_df.columns)}"
        )

    work = src_df.copy()
    work[source_id_column] = _normalize_ids(
        work[source_id_column],
        coerce_to_string=coerce_to_string,
        trim_whitespace=trim_whitespace,
        case_insensitive=case_insensitive,
    )
    filtered = work[work[source_id_column].isin(id_set)]

    # Write output -----------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_path, index=preserve_index_col)

    # Friendly printout
    try:
        rel_out = out_path.relative_to(Path.cwd())
    except ValueError:
        rel_out = out_path
    print(
        f"Extracted {len(filtered)}/{len(src_df)} rows using {len(id_set)} IDs → {rel_out}"
    )

    return out_path


def cli(
    *,
    ids_csv: str | Path,
    source_csv: str | Path,
    output_csv: Optional[str | Path] = None,
    ids_column: str = "id",
    source_id_column: str = "id",
    case_insensitive: bool = False,
    trim_whitespace: bool = True,
    coerce_to_string: bool = True,
    preserve_index_col: bool = True,
    drop_duplicates_in_ids: bool = True,
) -> None:
    """CLI wrapper – all parameters are flags.

    Required flags: --ids-csv, --source-csv
    Optional flags: --output-csv, --ids-column, --source-id-column, --case-insensitive,
                    --trim-whitespace, --coerce-to-string, --preserve-index-col,
                    --drop-duplicates-in-ids
    """

    extract_rows_by_ids(
        ids_csv=ids_csv,
        source_csv=source_csv,
        output_csv=output_csv,
        ids_column=ids_column,
        source_id_column=source_id_column,
        case_insensitive=case_insensitive,
        trim_whitespace=trim_whitespace,
        coerce_to_string=coerce_to_string,
        preserve_index_col=preserve_index_col,
        drop_duplicates_in_ids=drop_duplicates_in_ids,
    )


if __name__ == "__main__":
    tyro.cli(cli)


