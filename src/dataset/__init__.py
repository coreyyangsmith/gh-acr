"""Dataset utilities for GitHub Auto Conflict Resolver (GH-ACR).

This package contains small, focused utilities for preparing and slicing the
GitGoodBench (GGB) dataset used throughout the project.

Submodules provide:
- add_difficulty: CLI to add or update a difficulty column on a CSV
- loader: Typed loader that normalizes the GGB CSV into a pandas DataFrame
- get_subset: Sample a percentage subset of the dataset deterministically
- extract_samples_from_subset: Filter rows whose IDs appear in another CSV
- process_ggb: Extract scenarios that include a merge commit hash
- split_utils: Convenience helpers to split the dataset by difficulty
- process_ground_truths: Compare assistant outputs against ground truth files
"""

 