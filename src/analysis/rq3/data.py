"""Data loading and parsing utilities for RQ3 analyses.

Handles JSON classification file parsing, ID extraction, label grouping,
and merging with performance results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RQ3Config, DEFAULT_CONFIG, label_to_column_name


logger = logging.getLogger(__name__)


@dataclass
class ClassificationEntry:
    """A single classification entry for one file.

    Attributes
    ----------
    full_id : str
        The full ID including file suffix (e.g., "106151546310-1")
    base_id : str
        The base sample ID (e.g., "106151546310")
    file_index : int
        The file index within the sample (e.g., 1)
    labels : list[str]
        List of labels assigned to this file
    source_file : str
        Name of the source JSON file
    """

    full_id: str
    base_id: str
    file_index: int
    labels: list[str]
    source_file: str


@dataclass
class GroupedClassification:
    """Aggregated classification data for a sample (base ID).

    Attributes
    ----------
    base_id : str
        The base sample ID
    file_count : int
        Number of files in this sample
    label_counts : dict[str, int]
        Count of each label across all files
    unique_labels : set[str]
        Unique set of labels across all files
    source_file : str
        Name of the source JSON file
    entries : list[ClassificationEntry]
        Individual file entries
    """

    base_id: str
    file_count: int
    label_counts: dict[str, int]
    unique_labels: set[str]
    source_file: str
    entries: list[ClassificationEntry] = field(default_factory=list)


def extract_base_id(full_id: str) -> tuple[str, int]:
    """Extract base ID and file index from a full ID.

    Parameters
    ----------
    full_id : str
        The full ID including file suffix (e.g., "106151546310-1")

    Returns
    -------
    tuple[str, int]
        Tuple of (base_id, file_index)

    Examples
    --------
    >>> extract_base_id("106151546310-1")
    ('106151546310', 1)
    >>> extract_base_id("106151546310-2")
    ('106151546310', 2)
    """
    parts = full_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    # Fallback: treat entire ID as base ID with index 1
    return full_id, 1


def apply_label_mappings(labels: list[str], config: RQ3Config = DEFAULT_CONFIG) -> list[str]:
    """Apply label mappings to normalize labels.

    Parameters
    ----------
    labels : list[str]
        List of raw labels
    config : RQ3Config
        Configuration with label mappings

    Returns
    -------
    list[str]
        List of normalized labels
    """
    return [config.apply_mapping(label) for label in labels]


def parse_classification_json(
    path: Path,
    config: RQ3Config = DEFAULT_CONFIG,
) -> list[ClassificationEntry]:
    """Parse a classification JSON file.

    Parameters
    ----------
    path : Path
        Path to the JSON file
    config : RQ3Config
        Configuration with label mappings

    Returns
    -------
    list[ClassificationEntry]
        List of classification entries
    """
    logger.info(f"Parsing classification file: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classifications = data.get("classifications", {})
    source_file = path.stem

    entries = []
    for full_id, labels in classifications.items():
        base_id, file_index = extract_base_id(full_id)
        normalized_labels = apply_label_mappings(labels, config)
        
        entry = ClassificationEntry(
            full_id=full_id,
            base_id=base_id,
            file_index=file_index,
            labels=normalized_labels,
            source_file=source_file,
        )
        entries.append(entry)

    logger.info(f"  Parsed {len(entries)} classification entries from {source_file}")
    return entries


def group_by_base_id(
    entries: list[ClassificationEntry],
) -> dict[str, GroupedClassification]:
    """Group classification entries by base ID.

    Parameters
    ----------
    entries : list[ClassificationEntry]
        List of classification entries

    Returns
    -------
    dict[str, GroupedClassification]
        Dictionary mapping base_id to grouped classification
    """
    groups: dict[str, list[ClassificationEntry]] = {}
    
    for entry in entries:
        if entry.base_id not in groups:
            groups[entry.base_id] = []
        groups[entry.base_id].append(entry)

    result: dict[str, GroupedClassification] = {}
    
    for base_id, group_entries in groups.items():
        # Count labels across all files
        label_counts: dict[str, int] = {}
        unique_labels: set[str] = set()
        
        for entry in group_entries:
            for label in entry.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                unique_labels.add(label)

        # Get source file (should be the same for all entries)
        source_file = group_entries[0].source_file

        result[base_id] = GroupedClassification(
            base_id=base_id,
            file_count=len(group_entries),
            label_counts=label_counts,
            unique_labels=unique_labels,
            source_file=source_file,
            entries=group_entries,
        )

    return result


def parse_multiple_json_files(
    paths: list[Path],
    config: RQ3Config = DEFAULT_CONFIG,
) -> dict[str, list[ClassificationEntry]]:
    """Parse multiple classification JSON files.

    Parameters
    ----------
    paths : list[Path]
        List of paths to JSON files
    config : RQ3Config
        Configuration with label mappings

    Returns
    -------
    dict[str, list[ClassificationEntry]]
        Dictionary mapping source filename to list of entries
    """
    result: dict[str, list[ClassificationEntry]] = {}
    
    for path in paths:
        entries = parse_classification_json(path, config)
        source_file = path.stem
        result[source_file] = entries

    return result


def load_results_csv(path: Path) -> pd.DataFrame:
    """Load a results CSV file.

    Parameters
    ----------
    path : Path
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    logger.info(f"Loading results CSV: {path}")
    df = pd.read_csv(path)
    logger.info(f"  Loaded {len(df)} rows with columns: {list(df.columns)[:10]}...")
    return df


def grouped_to_dataframe(
    grouped: dict[str, GroupedClassification],
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Convert grouped classifications to a DataFrame.

    Parameters
    ----------
    grouped : dict[str, GroupedClassification]
        Dictionary of grouped classifications
    config : RQ3Config
        Configuration with canonical labels

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample, binary columns for labels
    """
    rows = []
    
    for base_id, group in grouped.items():
        row: dict[str, Any] = {
            "sample_id": base_id,
            "file_count": group.file_count,
            "total_labels": sum(group.label_counts.values()),
            "unique_label_count": len(group.unique_labels),
            "source_file": group.source_file,
        }
        
        # Add binary columns for each canonical label
        for label in config.canonical_labels:
            col_name = label_to_column_name(label)
            # Binary: 1 if label appears at all, 0 otherwise
            row[col_name] = 1 if label in group.unique_labels else 0
            # Count: how many times this label appears
            row[f"{col_name}_count"] = group.label_counts.get(label, 0)
        
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def merge_with_results(
    label_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Merge label data with results data.

    Merges on sample_id (from labels) and id (from results).
    Keeps only rows from results that match agent or bypass7 methods.

    Parameters
    ----------
    label_df : pd.DataFrame
        DataFrame with label data (from grouped_to_dataframe)
    results_df : pd.DataFrame
        DataFrame with results data (from load_results_csv)
    config : RQ3Config
        Configuration with method identifiers

    Returns
    -------
    pd.DataFrame
        Merged DataFrame
    """
    logger.info("Merging label data with results data...")
    
    # Ensure id column is string type for merging
    label_df = label_df.copy()
    label_df["sample_id"] = label_df["sample_id"].astype(str)
    
    results_df = results_df.copy()
    results_df["id"] = results_df["id"].astype(str)
    
    # Filter results to relevant methods
    methods = {config.single_agent_method, config.multi_agent_method}
    results_filtered = results_df[results_df["eval_method"].isin(methods)].copy()
    
    logger.info(f"  Filtered results to {len(results_filtered)} rows (methods: {methods})")
    
    # Merge on id
    merged = results_filtered.merge(
        label_df,
        left_on="id",
        right_on="sample_id",
        how="inner",
    )
    
    logger.info(f"  Merged dataset has {len(merged)} rows")
    
    return merged


def compute_method_pairs(
    merged_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Create paired data comparing agent vs bypass7 for each sample.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame with labels and results
    config : RQ3Config
        Configuration with method identifiers

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample, columns for both methods' metrics
    """
    logger.info("Computing method pairs...")
    
    # Get label columns
    label_cols = [label_to_column_name(l) for l in config.canonical_labels]
    label_cols = [c for c in label_cols if c in merged_df.columns]
    
    # Get metadata columns
    meta_cols = ["id", "difficulty", "project_size", "source_file"] + label_cols
    meta_cols = [c for c in meta_cols if c in merged_df.columns]
    
    # Pivot for each method
    agent_df = merged_df[merged_df["eval_method"] == config.single_agent_method].copy()
    bypass_df = merged_df[merged_df["eval_method"] == config.multi_agent_method].copy()
    
    # Rename metric columns
    for metric in config.metrics:
        if metric in agent_df.columns:
            agent_df = agent_df.rename(columns={metric: f"agent_{metric}"})
        if metric in bypass_df.columns:
            bypass_df = bypass_df.rename(columns={metric: f"bypass_{metric}"})
    
    # Select columns to keep
    agent_metric_cols = [f"agent_{m}" for m in config.metrics if f"agent_{m}" in agent_df.columns]
    bypass_metric_cols = [f"bypass_{m}" for m in config.metrics if f"bypass_{m}" in bypass_df.columns]
    
    agent_df = agent_df[["id"] + agent_metric_cols].drop_duplicates(subset=["id"])
    bypass_df = bypass_df[["id"] + bypass_metric_cols].drop_duplicates(subset=["id"])
    
    # Get metadata from either source
    meta_df = merged_df[meta_cols].drop_duplicates(subset=["id"])
    
    # Merge
    paired = meta_df.merge(agent_df, on="id", how="inner")
    paired = paired.merge(bypass_df, on="id", how="inner")
    
    # Compute deltas
    for metric in config.metrics:
        agent_col = f"agent_{metric}"
        bypass_col = f"bypass_{metric}"
        if agent_col in paired.columns and bypass_col in paired.columns:
            # Handle boolean exact_match
            if metric == "exact_match":
                paired[agent_col] = paired[agent_col].apply(
                    lambda x: 1 if str(x).lower() in ["true", "1", "1.0"] else 0
                )
                paired[bypass_col] = paired[bypass_col].apply(
                    lambda x: 1 if str(x).lower() in ["true", "1", "1.0"] else 0
                )
            paired[f"delta_{metric}"] = paired[bypass_col] - paired[agent_col]
            paired[f"bypass_wins_{metric}"] = paired[f"delta_{metric}"] > 0
            paired[f"agent_wins_{metric}"] = paired[f"delta_{metric}"] < 0
    
    logger.info(f"  Created {len(paired)} paired samples")
    
    return paired
