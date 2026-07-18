"""Aggregation and CSV export utilities for RQ3 analyses.

Creates aggregate CSV files with sample IDs, label counts, and binary label columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import RQ3Config, DEFAULT_CONFIG, label_to_column_name
from .data import (
    ClassificationEntry,
    GroupedClassification,
    group_by_base_id,
    grouped_to_dataframe,
)


logger = logging.getLogger(__name__)


def create_aggregate_dataframe(
    entries: list[ClassificationEntry],
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Create an aggregate DataFrame from classification entries.

    Parameters
    ----------
    entries : list[ClassificationEntry]
        List of classification entries
    config : RQ3Config
        Configuration with canonical labels

    Returns
    -------
    pd.DataFrame
        Aggregate DataFrame with:
        - sample_id: Base sample ID
        - file_count: Number of files in sample
        - total_labels: Total label occurrences
        - unique_label_count: Number of unique labels
        - source_file: Source JSON filename
        - Binary columns for each canonical label
        - Count columns for each canonical label
    """
    grouped = group_by_base_id(entries)
    return grouped_to_dataframe(grouped, config)


def export_aggregate_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Export aggregate DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregate DataFrame
    output_path : Path
        Output file path

    Returns
    -------
    Path
        Path to the exported file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"  Exported aggregate CSV: {output_path} ({len(df)} samples)")
    return output_path


def create_combined_aggregate(
    source_dfs: dict[str, pd.DataFrame],
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Combine multiple aggregate DataFrames.

    Parameters
    ----------
    source_dfs : dict[str, pd.DataFrame]
        Dictionary mapping source name to DataFrame
    config : RQ3Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all samples
    """
    if not source_dfs:
        return pd.DataFrame()

    combined = pd.concat(source_dfs.values(), ignore_index=True)
    
    # If same sample appears in multiple sources, keep all (they may have different labels)
    logger.info(f"  Combined {len(combined)} samples from {len(source_dfs)} sources")
    
    return combined


def compute_label_summary(
    df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute summary statistics for each label.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregate DataFrame with binary label columns
    config : RQ3Config
        Configuration with canonical labels

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with:
        - label: Label name
        - count: Number of samples with this label
        - percentage: Percentage of samples with this label
        - total_occurrences: Total count across all files
    """
    rows = []
    n_samples = len(df)
    
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        count_col = f"{col_name}_count"
        
        if col_name in df.columns:
            count = df[col_name].sum()
            percentage = 100 * count / n_samples if n_samples > 0 else 0
            
            total_occurrences = 0
            if count_col in df.columns:
                total_occurrences = df[count_col].sum()
            
            rows.append({
                "label": label,
                "display_name": config.get_label_display(label),
                "count": int(count),
                "percentage": percentage,
                "total_occurrences": int(total_occurrences),
            })
    
    summary = pd.DataFrame(rows)
    summary = summary.sort_values("count", ascending=False)
    
    return summary


def compute_co_occurrence_matrix(
    df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute label co-occurrence matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregate DataFrame with binary label columns
    config : RQ3Config
        Configuration with canonical labels

    Returns
    -------
    pd.DataFrame
        Square matrix with co-occurrence counts
    """
    # Get label columns that exist in the data
    label_cols = []
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name in df.columns:
            label_cols.append((label, col_name))
    
    # Build co-occurrence matrix
    n_labels = len(label_cols)
    matrix = pd.DataFrame(
        index=[l[0] for l in label_cols],
        columns=[l[0] for l in label_cols],
        dtype=int,
    )
    
    for i, (label_i, col_i) in enumerate(label_cols):
        for j, (label_j, col_j) in enumerate(label_cols):
            # Count samples where both labels are present
            co_count = ((df[col_i] == 1) & (df[col_j] == 1)).sum()
            matrix.loc[label_i, label_j] = co_count
    
    return matrix


def export_label_summary(
    df: pd.DataFrame,
    output_path: Path,
    config: RQ3Config = DEFAULT_CONFIG,
) -> Path:
    """Export label summary to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregate DataFrame
    output_path : Path
        Output file path
    config : RQ3Config
        Configuration

    Returns
    -------
    Path
        Path to the exported file
    """
    summary = compute_label_summary(df, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info(f"  Exported label summary: {output_path}")
    return output_path


def export_co_occurrence_matrix(
    df: pd.DataFrame,
    output_path: Path,
    config: RQ3Config = DEFAULT_CONFIG,
) -> Path:
    """Export co-occurrence matrix to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregate DataFrame
    output_path : Path
        Output file path
    config : RQ3Config
        Configuration

    Returns
    -------
    Path
        Path to the exported file
    """
    matrix = compute_co_occurrence_matrix(df, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)
    logger.info(f"  Exported co-occurrence matrix: {output_path}")
    return output_path
