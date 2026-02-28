"""Load code samples from folder structure and compute complexity metrics.

Handles the folder structure:
- sample_id/
  - agent/<filename>.py.txt
  - bypass/<filename>/bypass_<filename>.py.txt
  - default/<filename>/a.txt, b.txt, ground_truth.txt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .complexity import CodeMetrics, calculate_metrics


logger = logging.getLogger(__name__)


# Methods to analyze
METHODS = ["agent", "bypass", "a_only", "b_only", "ground_truth"]


@dataclass
class SampleCodes:
    """Container for all code versions of a sample.

    Attributes
    ----------
    sample_id : str
        Sample identifier
    agent : str
        Agent output code
    bypass : str
        Bypass output code
    a_only : str
        Version A source
    b_only : str
        Version B source
    ground_truth : str
        Ground truth merge
    file_name : str
        Name of the file being merged
    """

    sample_id: str
    agent: str = ""
    bypass: str = ""
    a_only: str = ""
    b_only: str = ""
    ground_truth: str = ""
    file_name: str = ""

    def get_code(self, method: str) -> str:
        """Get code for a specific method."""
        return getattr(self, method, "")


def _read_file_safe(path: Path) -> str:
    """Read file contents safely.

    Parameters
    ----------
    path : Path
        Path to file

    Returns
    -------
    str
        File contents or empty string if error
    """
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Error reading {path}: {e}")
    return ""


def _find_agent_file(sample_folder: Path) -> tuple[str, str]:
    """Find the agent output file in the sample folder.

    Parameters
    ----------
    sample_folder : Path
        Path to sample folder

    Returns
    -------
    tuple[str, str]
        (file_name, code) or ("", "") if not found
    """
    agent_folder = sample_folder / "agent"
    if not agent_folder.exists():
        return "", ""

    # Find .txt files in agent folder
    txt_files = list(agent_folder.glob("*.txt"))
    if not txt_files:
        return "", ""

    # Use the first .txt file found
    agent_file = txt_files[0]
    file_name = agent_file.stem  # e.g., "letta_agent.py"

    code = _read_file_safe(agent_file)
    return file_name, code


def _find_bypass_file(sample_folder: Path, file_name: str) -> str:
    """Find the bypass output file in the sample folder.

    Parameters
    ----------
    sample_folder : Path
        Path to sample folder
    file_name : str
        File name (e.g., "letta_agent.py")

    Returns
    -------
    str
        Bypass code or empty string if not found
    """
    bypass_folder = sample_folder / "bypass"
    if not bypass_folder.exists():
        return ""

    # Try the expected path: bypass/<file_name>/bypass_<file_name>.txt
    # file_name might be like "letta_agent.py" or "src_crewai_crew.py"
    subfolder = bypass_folder / file_name
    if subfolder.exists():
        # Look for bypass_<file_name>.txt
        bypass_file = subfolder / f"bypass_{file_name}.txt"
        if bypass_file.exists():
            return _read_file_safe(bypass_file)

    # Fallback: look in any subfolder for bypass_*.txt
    for subfolder in bypass_folder.iterdir():
        if subfolder.is_dir():
            bypass_files = list(subfolder.glob("bypass_*.txt"))
            if bypass_files:
                return _read_file_safe(bypass_files[0])

    return ""


def _find_default_files(sample_folder: Path, file_name: str) -> tuple[str, str, str]:
    """Find a.txt, b.txt, ground_truth.txt in default folder.

    Parameters
    ----------
    sample_folder : Path
        Path to sample folder
    file_name : str
        File name (e.g., "letta_agent.py")

    Returns
    -------
    tuple[str, str, str]
        (a_code, b_code, ground_truth_code)
    """
    default_folder = sample_folder / "default"
    if not default_folder.exists():
        return "", "", ""

    # Try expected path: default/<file_name>/
    subfolder = default_folder / file_name
    if subfolder.exists():
        a_code = _read_file_safe(subfolder / "a.txt")
        b_code = _read_file_safe(subfolder / "b.txt")
        gt_code = _read_file_safe(subfolder / "ground_truth.txt")
        return a_code, b_code, gt_code

    # Fallback: look in any subfolder
    for subfolder in default_folder.iterdir():
        if subfolder.is_dir():
            a_file = subfolder / "a.txt"
            b_file = subfolder / "b.txt"
            gt_file = subfolder / "ground_truth.txt"
            if a_file.exists() or b_file.exists() or gt_file.exists():
                return (
                    _read_file_safe(a_file),
                    _read_file_safe(b_file),
                    _read_file_safe(gt_file),
                )

    return "", "", ""


def load_sample_codes(
    sample_folder: Path,
    model_name: str = "",
) -> SampleCodes:
    """Load code for all 5 methods from a sample folder.

    Parameters
    ----------
    sample_folder : Path
        Path to sample folder (e.g., .../4886326609-1/)
    model_name : str
        Model name (used for determining if think tags need stripping)

    Returns
    -------
    SampleCodes
        Container with code for all methods
    """
    sample_id = sample_folder.name

    # Find agent file and get file name
    file_name, agent_code = _find_agent_file(sample_folder)

    if not file_name:
        # Try to find file name from bypass or default folders
        bypass_folder = sample_folder / "bypass"
        if bypass_folder.exists():
            subfolders = [f for f in bypass_folder.iterdir() if f.is_dir()]
            if subfolders:
                file_name = subfolders[0].name

    if not file_name:
        default_folder = sample_folder / "default"
        if default_folder.exists():
            subfolders = [f for f in default_folder.iterdir() if f.is_dir()]
            if subfolders:
                file_name = subfolders[0].name

    # Find bypass code
    bypass_code = _find_bypass_file(sample_folder, file_name)

    # Find default files (a, b, ground_truth)
    a_code, b_code, gt_code = _find_default_files(sample_folder, file_name)

    return SampleCodes(
        sample_id=sample_id,
        agent=agent_code,
        bypass=bypass_code,
        a_only=a_code,
        b_only=b_code,
        ground_truth=gt_code,
        file_name=file_name,
    )


def compute_sample_metrics(
    sample_folder: Path,
    model_name: str = "",
) -> dict[str, CodeMetrics]:
    """Compute complexity metrics for all 5 methods of a sample.

    Parameters
    ----------
    sample_folder : Path
        Path to sample folder
    model_name : str
        Model name (for determining think tag stripping)

    Returns
    -------
    dict[str, CodeMetrics]
        Metrics keyed by method name
    """
    codes = load_sample_codes(sample_folder, model_name)

    # Determine if we need to strip think tags (Qwen models)
    strip_think = "qwen" in model_name.lower()

    metrics = {}
    for method in METHODS:
        code = codes.get_code(method)
        if code:
            # Only strip think tags for agent method with qwen
            should_strip = strip_think and method == "agent"
            metrics[method] = calculate_metrics(code, strip_think=should_strip)
        else:
            metrics[method] = CodeMetrics.empty(f"No code found for {method}")

    return metrics


def _find_sample_folders(case_folder: Path, sample_id: str) -> list[Path]:
    """Find all sample folders matching a base sample ID.
    
    Folders may be named as:
    - sample_id (exact match)
    - sample_id-1, sample_id-2, etc. (with suffix)
    
    Parameters
    ----------
    case_folder : Path
        Path to the case folder
    sample_id : str
        Base sample ID to match
    
    Returns
    -------
    list[Path]
        List of matching folder paths
    """
    sample_id_str = str(sample_id)
    
    # Try exact match first
    exact = case_folder / sample_id_str
    if exact.exists():
        return [exact]
    
    # Try with suffix pattern (sample_id-N)
    matching = []
    for folder in case_folder.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            # Check if folder starts with sample_id and has suffix
            if folder_name.startswith(sample_id_str + "-"):
                matching.append(folder)
            elif folder_name == sample_id_str:
                matching.append(folder)
    
    return sorted(matching)


def process_all_samples(
    case_folder: Path,
    sample_ids: list[str],
    model_name: str = "",
) -> pd.DataFrame:
    """Process all samples and return metrics DataFrame.

    Parameters
    ----------
    case_folder : Path
        Path to the case folder containing sample subfolders
    sample_ids : list[str]
        List of sample IDs to process (may be base IDs)
    model_name : str
        Model name for determining processing options

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for all samples and methods
    """
    logger.info(f"Processing {len(sample_ids)} samples from {case_folder}")

    rows = []
    processed = 0
    errors = 0

    for sample_id in sample_ids:
        # Find all matching folders (handles base ID + suffix folders)
        sample_folders = _find_sample_folders(case_folder, str(sample_id))
        
        if not sample_folders:
            logger.debug(f"No sample folder found for: {sample_id}")
            errors += 1
            continue

        # Process the first matching folder
        # (usually sample_id-1 is representative)
        sample_folder = sample_folders[0]
        
        try:
            metrics_dict = compute_sample_metrics(sample_folder, model_name)

            for method, metrics in metrics_dict.items():
                row = {
                    "sample_id": str(sample_id),
                    "full_sample_id": sample_folder.name,
                    "method": method,
                    **metrics.to_dict(),
                }
                rows.append(row)

            processed += 1
        except Exception as e:
            logger.warning(f"Error processing {sample_id}: {e}")
            errors += 1

    logger.info(f"  Processed {processed} samples, {errors} errors")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def aggregate_metrics_by_method(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics by method (mean, std, etc.).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame from process_all_samples

    Returns
    -------
    pd.DataFrame
        Aggregated statistics per method
    """
    if metrics_df.empty:
        return pd.DataFrame()

    # Numeric columns to aggregate
    numeric_cols = [
        "sloc", "lloc", "comments", "blank",
        "cc_total", "cc_avg", "cc_max", "cc_count",
        "h_vocabulary", "h_length", "h_difficulty", "h_effort", "h_bugs",
        "mi_score",
    ]

    # Filter to existing columns
    numeric_cols = [c for c in numeric_cols if c in metrics_df.columns]

    # Filter out parse errors
    valid_df = metrics_df[~metrics_df["parse_error"]].copy()

    if valid_df.empty:
        return pd.DataFrame()

    # Aggregate by method
    agg_dict = {col: ["mean", "std", "min", "max"] for col in numeric_cols}
    agg_dict["sample_id"] = "count"

    agg_df = valid_df.groupby("method").agg(agg_dict)

    # Flatten column names
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.rename(columns={"sample_id_count": "n_samples"})
    agg_df = agg_df.reset_index()

    return agg_df


def compute_complexity_deltas(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute complexity deltas between methods.

    Computes differences between agent/bypass and ground_truth,
    and between agent and bypass.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame from process_all_samples

    Returns
    -------
    pd.DataFrame
        DataFrame with deltas for each sample
    """
    if metrics_df.empty:
        return pd.DataFrame()

    # Pivot to get methods as columns
    metrics_cols = [
        "sloc", "lloc", "cc_total", "cc_avg", "cc_max", "mi_score",
        "h_difficulty", "h_bugs",
    ]
    metrics_cols = [c for c in metrics_cols if c in metrics_df.columns]

    rows = []
    for sample_id in metrics_df["sample_id"].unique():
        sample_data = metrics_df[metrics_df["sample_id"] == sample_id]

        row = {"sample_id": sample_id}

        # Get metrics for each method
        method_metrics = {}
        for method in METHODS:
            method_data = sample_data[sample_data["method"] == method]
            if len(method_data) == 1:
                method_metrics[method] = method_data.iloc[0]

        # Compute deltas
        for metric in metrics_cols:
            # Agent vs ground_truth
            if "agent" in method_metrics and "ground_truth" in method_metrics:
                row[f"agent_vs_gt_{metric}"] = (
                    method_metrics["agent"][metric] -
                    method_metrics["ground_truth"][metric]
                )

            # Bypass vs ground_truth
            if "bypass" in method_metrics and "ground_truth" in method_metrics:
                row[f"bypass_vs_gt_{metric}"] = (
                    method_metrics["bypass"][metric] -
                    method_metrics["ground_truth"][metric]
                )

            # Agent vs bypass
            if "agent" in method_metrics and "bypass" in method_metrics:
                row[f"agent_vs_bypass_{metric}"] = (
                    method_metrics["agent"][metric] -
                    method_metrics["bypass"][metric]
                )

        # Add ground truth complexity as reference
        if "ground_truth" in method_metrics:
            for metric in metrics_cols:
                row[f"gt_{metric}"] = method_metrics["ground_truth"][metric]

        rows.append(row)

    return pd.DataFrame(rows)
