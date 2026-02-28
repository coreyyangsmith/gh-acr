"""Core metric computation for quantitative change analysis.

Provides functions to:
- Parse unified diffs and extract added/removed/hunk counts
- Count commits from commit message files
- Compute LOC/SLOC from code text (reusing radon)
- Generate unified diffs between two code strings
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ── Diff Metrics ─────────────────────────────────────────────────────────


@dataclass
class DiffMetrics:
    """Metrics extracted from a unified diff.

    Attributes
    ----------
    diff_lines_added : int
        Number of ``+`` lines (excluding ``+++`` header)
    diff_lines_removed : int
        Number of ``-`` lines (excluding ``---`` header)
    diff_net_change : int
        added - removed
    diff_total_change : int
        added + removed  (change magnitude / "PR length")
    diff_hunks : int
        Number of ``@@`` hunk headers
    diff_total_lines : int
        Total lines in the diff text
    """

    diff_lines_added: int = 0
    diff_lines_removed: int = 0
    diff_net_change: int = 0
    diff_total_change: int = 0
    diff_hunks: int = 0
    diff_total_lines: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def parse_unified_diff(diff_text: str) -> DiffMetrics:
    """Parse a unified diff string and extract metrics.

    Parameters
    ----------
    diff_text : str
        Unified diff text (e.g. contents of a ``.diff`` file)

    Returns
    -------
    DiffMetrics
        Extracted diff metrics
    """
    if not diff_text or not diff_text.strip():
        return DiffMetrics()

    lines = diff_text.splitlines()
    added = 0
    removed = 0
    hunks = 0

    for line in lines:
        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    return DiffMetrics(
        diff_lines_added=added,
        diff_lines_removed=removed,
        diff_net_change=added - removed,
        diff_total_change=added + removed,
        diff_hunks=hunks,
        diff_total_lines=len(lines),
    )


def generate_unified_diff(
    old_code: str,
    new_code: str,
    old_label: str = "ancestor",
    new_label: str = "version",
) -> str:
    """Generate a unified diff between two code strings.

    Used to compute diff metrics for agent/bypass outputs
    (whose .diff files are not stored on disk).

    Parameters
    ----------
    old_code : str
        The base/ancestor code
    new_code : str
        The modified code
    old_label : str
        Label for the old file in diff headers
    new_label : str
        Label for the new file in diff headers

    Returns
    -------
    str
        Unified diff text
    """
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_label,
            tofile=new_label,
        )
    )


# ── LOC / SLOC Metrics ──────────────────────────────────────────────────


@dataclass
class SizeMetrics:
    """Size metrics for a code file.

    Attributes
    ----------
    loc : int
        Total line count (including blanks and comments)
    sloc : int
        Source lines of code (non-blank, non-comment)
    blank_lines : int
        Number of blank lines
    comment_lines : int
        Number of comment lines
    """

    loc: int = 0
    sloc: int = 0
    blank_lines: int = 0
    comment_lines: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_size_metrics(code: str) -> SizeMetrics:
    """Compute LOC/SLOC metrics for a code string.

    Uses ``radon.raw.analyze`` when available; falls back to a
    simple line-count heuristic.

    Parameters
    ----------
    code : str
        Source code text

    Returns
    -------
    SizeMetrics
        Computed size metrics
    """
    if not code or not code.strip():
        return SizeMetrics()

    lines = code.splitlines()
    total_loc = len(lines)

    try:
        from radon.raw import analyze

        raw = analyze(code)
        return SizeMetrics(
            loc=total_loc,
            sloc=raw.sloc,
            blank_lines=raw.blank,
            comment_lines=raw.comments + raw.multi,
        )
    except Exception:
        # Fallback: simple heuristic
        blank = sum(1 for l in lines if not l.strip())
        comment = sum(1 for l in lines if l.strip().startswith("#"))
        return SizeMetrics(
            loc=total_loc,
            sloc=total_loc - blank - comment,
            blank_lines=blank,
            comment_lines=comment,
        )


# ── Commit Metrics ───────────────────────────────────────────────────────


def count_commits(commit_message_text: str) -> int:
    """Count the number of commits in a commit message file.

    Commits in the file are separated by the ``-----`` delimiter
    (see ``pipeline_clone.py`` lines 441-464).

    Parameters
    ----------
    commit_message_text : str
        Raw contents of ``a_commit_message.txt`` or ``b_commit_message.txt``

    Returns
    -------
    int
        Number of commits found (0 if empty)
    """
    if not commit_message_text or not commit_message_text.strip():
        return 0

    # Each commit is separated by "\n\n-----\n\n"
    # Also handle variations in whitespace
    sections = re.split(r"\n\n-{3,}\n\n", commit_message_text.strip())
    # Filter out empty sections
    sections = [s for s in sections if s.strip()]
    return len(sections)


# ── Combined per-version computation ─────────────────────────────────────


@dataclass
class VersionMetrics:
    """All quantitative metrics for a single version of a sample.

    Combines size metrics, change metrics (delta from ancestor),
    and diff metrics into a flat structure suitable for DataFrame rows.
    """

    # Size
    loc: int = 0
    sloc: int = 0
    blank_lines: int = 0
    comment_lines: int = 0

    # Change deltas (vs ancestor)
    loc_delta: int = 0
    sloc_delta: int = 0

    # Diff
    diff_lines_added: int = 0
    diff_lines_removed: int = 0
    diff_net_change: int = 0
    diff_total_change: int = 0
    diff_hunks: int = 0
    diff_total_lines: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_version_metrics(
    code: str,
    ancestor_code: str,
    diff_text: Optional[str] = None,
) -> VersionMetrics:
    """Compute all quantitative metrics for one version of a sample.

    Parameters
    ----------
    code : str
        The code for this version
    ancestor_code : str
        The ancestor / previous code (for computing deltas)
    diff_text : str, optional
        Pre-computed unified diff text. If None, the diff is
        generated programmatically from ``ancestor_code`` → ``code``.

    Returns
    -------
    VersionMetrics
        Combined metrics
    """
    # Size metrics for this version
    size = compute_size_metrics(code)

    # Size metrics for ancestor (for computing deltas)
    ancestor_size = compute_size_metrics(ancestor_code)

    # Diff metrics
    if diff_text is None:
        diff_text = generate_unified_diff(ancestor_code, code)
    diff = parse_unified_diff(diff_text)

    return VersionMetrics(
        loc=size.loc,
        sloc=size.sloc,
        blank_lines=size.blank_lines,
        comment_lines=size.comment_lines,
        loc_delta=size.loc - ancestor_size.loc,
        sloc_delta=size.sloc - ancestor_size.sloc,
        diff_lines_added=diff.diff_lines_added,
        diff_lines_removed=diff.diff_lines_removed,
        diff_net_change=diff.diff_net_change,
        diff_total_change=diff.diff_total_change,
        diff_hunks=diff.diff_hunks,
        diff_total_lines=diff.diff_total_lines,
    )
