from __future__ import annotations

"""Script/CLI to evaluate *GitGoodBench* model outputs against ground-truth.

The *data/output* directory produced by the merge-assistant pipeline follows the
nested structure::

    data/output/{id}/{encoded_repo_path}/
        ├── a.txt               # Assistant (single-agent) output
        ├── b.txt               # Assistant (multi-agent) output
        └── ground_truth.txt    # Reference implementation

For every triple we check **exact** file content equality (byte-for-byte) and
report aggregate statistics:

* How many cases exactly match ``a.txt``?
* How many cases exactly match ``b.txt``?
* How many cases match neither (failures)?

Usage (PowerShell example):
    python -m src.dataset.process_ground_truths --base-dir data/output

You may point to a custom directory containing the same folder layout::

    python -m src.dataset.process_ground_truths --base-dir path/to/outputs
"""

from pathlib import Path


import tyro

__all__ = [
    "evaluate_ground_truths",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def evaluate_ground_truths(*, base_dir: str | Path = "data/output") -> None:  # noqa: D401 – imperative mood is fine
    """Compare **ground_truth.txt** against *a.txt* and *b.txt* for all outputs.

    Parameters
    ----------
    base_dir
        Root directory that contains the per-sample output folders.  The
        function traverses **recursively** to locate every ``ground_truth.txt``
        file and performs comparisons in its parent directory.
    """

    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    gt_files = list(base_path.rglob("ground_truth.txt"))
    if not gt_files:
        raise FileNotFoundError(
            f"No ground_truth.txt files found under: {base_path}")

    # Counters ----------------------------------------------------------------
    total = 0
    match_a = 0
    match_b = 0
    match_any = 0

    # Iterate -----------------------------------------------------------------
    for gt_path in gt_files:
        parent = gt_path.parent
        a_path = parent / "a.txt"
        b_path = parent / "b.txt"

        # Skip if the associated prediction files are missing -----------------
        if not a_path.exists() or not b_path.exists():
            # Silently ignore incomplete folders – they might belong to failed runs
            continue

        total += 1

        # Read files exactly as binary to avoid any newline translation issues
        gt_bytes = gt_path.read_bytes()
        a_bytes = a_path.read_bytes()
        b_bytes = b_path.read_bytes()

        # Compare -------------------------------------------------------------
        any_match_here = False
        if gt_bytes == a_bytes:
            match_a += 1
            any_match_here = True
        if gt_bytes == b_bytes:
            match_b += 1
            any_match_here = True
        if any_match_here:
            match_any += 1

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------
    print("================ Evaluation Summary ================")
    print(f"Base directory : {base_path.relative_to(Path.cwd())}")
    print(f"Total samples  : {total}")
    ratio_a = match_a / total if total else 0
    print(f"Match a.txt    : {match_a} ({ratio_a:.2%})")
    ratio_b = match_b / total if total else 0
    print(f"Match b.txt    : {match_b} ({ratio_b:.2%})")
    print(f"No match       : {total - match_any}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tyro.cli(evaluate_ground_truths)
