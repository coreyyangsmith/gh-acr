"""Load code samples from case folder structure and compute quantitative metrics.

Extends the pattern from ``rq3.complexity_loader`` to also read:
- ``original.txt``  (ancestor / "Previous")
- ``a.diff``, ``b.diff``, ``ground_truth.diff``
- ``a_commit_message.txt``, ``b_commit_message.txt``

and computes diff metrics for agent/bypass outputs programmatically.

Folder structure handled::

    sample_id-1/
    ├── agent/<filename>.txt
    ├── bypass/<filename>/bypass_<filename>.txt
    └── default/<filename>/
        ├── original.txt          # ancestor ("Previous")
        ├── a.txt                 # parent A
        ├── b.txt                 # parent B
        ├── ground_truth.txt      # actual merge
        ├── a.diff
        ├── b.diff
        ├── ground_truth.diff
        ├── a_commit_message.txt
        └── b_commit_message.txt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import VERSIONS, QuantConfig, DEFAULT_CONFIG
from .metrics import (
    compute_version_metrics,
    count_commits,
    VersionMetrics,
)

logger = logging.getLogger(__name__)


# ── File reading helpers ─────────────────────────────────────────────────


def _read_file_safe(path: Path) -> str:
    """Read file contents safely, returning empty string on error."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Error reading {path}: {e}")
    return ""


# ── Sample data container ────────────────────────────────────────────────


@dataclass
class SampleData:
    """Container for all raw data from a single sample folder.

    Attributes
    ----------
    sample_id : str
        Sample identifier (folder name)
    file_name : str
        Name of the conflicted file
    previous_code : str
        Ancestor / merge-base code (original.txt)
    a_code, b_code, gt_code, agent_code, bypass_code : str
        Code for each version
    a_diff, b_diff, gt_diff : str
        Stored unified diffs (from .diff files)
    a_commit_msg, b_commit_msg : str
        Raw commit message text for each branch
    """

    sample_id: str
    file_name: str = ""

    # Code versions
    previous_code: str = ""
    a_code: str = ""
    b_code: str = ""
    gt_code: str = ""
    agent_code: str = ""
    bypass_code: str = ""

    # Stored diffs
    a_diff: str = ""
    b_diff: str = ""
    gt_diff: str = ""

    # Commit messages
    a_commit_msg: str = ""
    b_commit_msg: str = ""


# ── Folder navigation ────────────────────────────────────────────────────


def _find_agent_file(sample_folder: Path) -> tuple[str, str]:
    """Find the agent output file and return (file_name, code)."""
    agent_folder = sample_folder / "agent"
    if not agent_folder.exists():
        return "", ""
    txt_files = list(agent_folder.glob("*.txt"))
    if not txt_files:
        return "", ""
    agent_file = txt_files[0]
    return agent_file.stem, _read_file_safe(agent_file)


def _find_bypass_file(sample_folder: Path, file_name: str) -> str:
    """Find the bypass output file and return code."""
    bypass_folder = sample_folder / "bypass"
    if not bypass_folder.exists():
        return ""
    subfolder = bypass_folder / file_name
    if subfolder.exists():
        bypass_file = subfolder / f"bypass_{file_name}.txt"
        if bypass_file.exists():
            return _read_file_safe(bypass_file)
    # Fallback: search in any subfolder
    for sub in bypass_folder.iterdir():
        if sub.is_dir():
            for bf in sub.glob("bypass_*.txt"):
                return _read_file_safe(bf)
    return ""


def _find_default_folder(sample_folder: Path, file_name: str) -> Path | None:
    """Find the default/<file_name>/ subfolder."""
    default_folder = sample_folder / "default"
    if not default_folder.exists():
        return None
    # Try exact match
    candidate = default_folder / file_name
    if candidate.exists() and candidate.is_dir():
        return candidate
    # Fallback: first subfolder
    for sub in default_folder.iterdir():
        if sub.is_dir():
            return sub
    return None


def _resolve_file_name(sample_folder: Path) -> str:
    """Try to determine the file name from available folders."""
    # Try agent folder first
    agent_folder = sample_folder / "agent"
    if agent_folder.exists():
        txt_files = list(agent_folder.glob("*.txt"))
        if txt_files:
            return txt_files[0].stem

    # Try bypass folder
    bypass_folder = sample_folder / "bypass"
    if bypass_folder.exists():
        subfolders = [f for f in bypass_folder.iterdir() if f.is_dir()]
        if subfolders:
            return subfolders[0].name

    # Try default folder
    default_folder = sample_folder / "default"
    if default_folder.exists():
        subfolders = [f for f in default_folder.iterdir() if f.is_dir()]
        if subfolders:
            return subfolders[0].name

    return ""


# ── Main loading function ────────────────────────────────────────────────


def load_sample_data(sample_folder: Path) -> SampleData:
    """Load all raw data for a single sample from its folder.

    Parameters
    ----------
    sample_folder : Path
        Path to the sample folder (e.g. ``.../4886326609-1/``)

    Returns
    -------
    SampleData
        Container with all code, diffs, and commit messages
    """
    sample_id = sample_folder.name

    # Determine file name
    file_name, agent_code = _find_agent_file(sample_folder)
    if not file_name:
        file_name = _resolve_file_name(sample_folder)

    # Load bypass code
    bypass_code = _find_bypass_file(sample_folder, file_name)

    # Load default folder contents
    default_sub = _find_default_folder(sample_folder, file_name)
    if default_sub is not None:
        previous_code = _read_file_safe(default_sub / "original.txt")
        a_code = _read_file_safe(default_sub / "a.txt")
        b_code = _read_file_safe(default_sub / "b.txt")
        gt_code = _read_file_safe(default_sub / "ground_truth.txt")
        a_diff = _read_file_safe(default_sub / "a.diff")
        b_diff = _read_file_safe(default_sub / "b.diff")
        gt_diff = _read_file_safe(default_sub / "ground_truth.diff")
        a_commit_msg = _read_file_safe(default_sub / "a_commit_message.txt")
        b_commit_msg = _read_file_safe(default_sub / "b_commit_message.txt")
    else:
        previous_code = a_code = b_code = gt_code = ""
        a_diff = b_diff = gt_diff = ""
        a_commit_msg = b_commit_msg = ""

    return SampleData(
        sample_id=sample_id,
        file_name=file_name,
        previous_code=previous_code,
        a_code=a_code,
        b_code=b_code,
        gt_code=gt_code,
        agent_code=agent_code,
        bypass_code=bypass_code,
        a_diff=a_diff,
        b_diff=b_diff,
        gt_diff=gt_diff,
        a_commit_msg=a_commit_msg,
        b_commit_msg=b_commit_msg,
    )


# ── Compute metrics for a sample ─────────────────────────────────────────


def compute_sample_quantitative_metrics(
    sample_folder: Path,
) -> list[dict]:
    """Compute quantitative metrics for all versions of a sample.

    Returns one dict per version, each containing:
    - ``sample_id``, ``version``, ``file_name``
    - All :class:`VersionMetrics` fields
    - ``n_commits_a``, ``n_commits_b``, ``n_commits_total`` (same for all versions)

    Parameters
    ----------
    sample_folder : Path
        Path to the sample folder

    Returns
    -------
    list[dict]
        One dict per version (up to 6: previous, a, b, ground_truth, agent, bypass)
    """
    data = load_sample_data(sample_folder)
    ancestor = data.previous_code

    # Map version → (code, diff_text_or_None)
    version_map: dict[str, tuple[str, str | None]] = {
        "previous": (data.previous_code, None),  # diff from itself = empty
        "a": (data.a_code, data.a_diff or None),
        "b": (data.b_code, data.b_diff or None),
        "ground_truth": (data.gt_code, data.gt_diff or None),
        "agent": (data.agent_code, None),          # compute programmatically
        "bypass": (data.bypass_code, None),         # compute programmatically
    }

    # Commit counts (same for all versions)
    n_commits_a = count_commits(data.a_commit_msg)
    n_commits_b = count_commits(data.b_commit_msg)
    n_commits_total = n_commits_a + n_commits_b

    rows: list[dict] = []
    for version, (code, diff_text) in version_map.items():
        # Skip versions without code (but always include previous)
        if not code and version != "previous":
            continue

        # For "previous", the diff from itself is trivially empty
        if version == "previous":
            base = compute_version_metrics(code, code)
            vm = VersionMetrics(
                loc=base.loc,
                sloc=base.sloc,
                blank_lines=base.blank_lines,
                comment_lines=base.comment_lines,
            )
        else:
            vm = compute_version_metrics(code, ancestor, diff_text)

        row = {
            "sample_id": data.sample_id,
            "version": version,
            "file_name": data.file_name,
            **vm.to_dict(),
            "n_commits_a": n_commits_a,
            "n_commits_b": n_commits_b,
            "n_commits_total": n_commits_total,
        }
        rows.append(row)

    return rows


# ── Folder-matching helpers (from complexity_loader) ─────────────────────


def _find_sample_folders(case_folder: Path, sample_id: str) -> list[Path]:
    """Find all sample folders matching a base sample ID.

    Folders may be named as:
    - sample_id (exact match)
    - sample_id-1, sample_id-2, etc. (with suffix)
    """
    sample_id_str = str(sample_id)

    exact = case_folder / sample_id_str
    if exact.exists():
        return [exact]

    matching = []
    for folder in case_folder.iterdir():
        if folder.is_dir():
            name = folder.name
            if name.startswith(sample_id_str + "-") or name == sample_id_str:
                matching.append(folder)

    return sorted(matching)


# ── Batch processing ─────────────────────────────────────────────────────


def process_all_samples(
    case_folder: Path,
    sample_ids: list[str],
    model_name: str = "",
) -> pd.DataFrame:
    """Process all samples and return a metrics DataFrame.

    Parameters
    ----------
    case_folder : Path
        Path to the case folder containing sample subfolders
    sample_ids : list[str]
        List of sample IDs to process
    model_name : str
        Model name (for logging)

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per (sample_id, version) and all metric columns
    """
    logger.info(
        f"Processing {len(sample_ids)} samples from {case_folder}"
        + (f" (model={model_name})" if model_name else "")
    )

    all_rows: list[dict] = []
    processed = 0
    errors = 0

    for sample_id in sample_ids:
        folders = _find_sample_folders(case_folder, str(sample_id))
        if not folders:
            logger.debug(f"No sample folder found for: {sample_id}")
            errors += 1
            continue

        # Use first matching folder
        folder = folders[0]
        try:
            rows = compute_sample_quantitative_metrics(folder)
            all_rows.extend(rows)
            processed += 1
        except Exception as e:
            logger.warning(f"Error processing {sample_id}: {e}")
            errors += 1

    logger.info(f"  Processed {processed} samples, {errors} errors")

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def aggregate_metrics_by_version(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics by version (mean, std, median).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame from :func:`process_all_samples`

    Returns
    -------
    pd.DataFrame
        Aggregated statistics per version
    """
    if metrics_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "loc", "sloc", "blank_lines", "comment_lines",
        "loc_delta", "sloc_delta",
        "diff_lines_added", "diff_lines_removed",
        "diff_net_change", "diff_total_change",
        "diff_hunks", "diff_total_lines",
        "n_commits_a", "n_commits_b", "n_commits_total",
    ]
    numeric_cols = [c for c in numeric_cols if c in metrics_df.columns]

    agg_dict = {col: ["mean", "std", "median", "min", "max"] for col in numeric_cols}
    agg_dict["sample_id"] = "count"

    agg_df = metrics_df.groupby("version").agg(agg_dict)

    # Flatten column names
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.rename(columns={"sample_id_count": "n_samples"})
    agg_df = agg_df.reset_index()

    return agg_df


def compute_quantitative_deltas(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise deltas between versions (agent vs GT, bypass vs GT, etc.).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame from :func:`process_all_samples`

    Returns
    -------
    pd.DataFrame
        DataFrame with delta columns for each sample
    """
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "loc", "sloc", "loc_delta", "sloc_delta",
        "diff_lines_added", "diff_lines_removed",
        "diff_total_change", "diff_hunks",
    ]
    metric_cols = [c for c in metric_cols if c in metrics_df.columns]

    rows = []
    for sample_id in metrics_df["sample_id"].unique():
        sample_data = metrics_df[metrics_df["sample_id"] == sample_id]
        row: dict = {"sample_id": sample_id}

        # Extract per-version metrics
        version_vals: dict[str, dict] = {}
        for version in ["ground_truth", "agent", "bypass", "a", "b", "previous"]:
            vd = sample_data[sample_data["version"] == version]
            if len(vd) == 1:
                version_vals[version] = vd.iloc[0]

        # Pairwise deltas
        pairs = [
            ("agent", "ground_truth", "agent_vs_gt"),
            ("bypass", "ground_truth", "bypass_vs_gt"),
            ("agent", "bypass", "agent_vs_bypass"),
        ]
        for v1, v2, prefix in pairs:
            if v1 in version_vals and v2 in version_vals:
                for metric in metric_cols:
                    val1 = version_vals[v1].get(metric, 0)
                    val2 = version_vals[v2].get(metric, 0)
                    try:
                        row[f"{prefix}_{metric}"] = float(val1) - float(val2)
                    except (TypeError, ValueError):
                        pass

        # Include ground truth absolute values as reference
        if "ground_truth" in version_vals:
            for metric in metric_cols:
                val = version_vals["ground_truth"].get(metric, 0)
                try:
                    row[f"gt_{metric}"] = float(val)
                except (TypeError, ValueError):
                    pass

        # Include commit counts
        if "previous" in version_vals:
            for col in ["n_commits_a", "n_commits_b", "n_commits_total"]:
                val = version_vals["previous"].get(col, 0)
                try:
                    row[col] = int(val)
                except (TypeError, ValueError):
                    pass
        elif version_vals:
            # commits are the same on every version row, use any
            first = next(iter(version_vals.values()))
            for col in ["n_commits_a", "n_commits_b", "n_commits_total"]:
                val = first.get(col, 0)
                try:
                    row[col] = int(val)
                except (TypeError, ValueError):
                    pass

        rows.append(row)

    return pd.DataFrame(rows)
