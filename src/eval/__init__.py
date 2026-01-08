"""Evaluation metrics package for merge conflict resolution.

This package provides metrics for comparing predicted merge results against
ground truth. All metrics are designed to work with file-level content maps
(Dict[str, str]) where keys are file paths and values are file contents.

Available Metrics
-----------------
- **Exact Match**: Binary match after normalization
- **BLEU-3**: N-gram precision with brevity penalty (up to trigrams)
- **ROUGE-L**: Longest Common Subsequence F1 score

Common Interface
----------------
All metric modules provide three functions:
1. `<metric>_score(pred, truth)`: Single-pair scoring
2. `per_file(pred_map, truth_map)`: Dict of per-file scores
3. `overall(pred_map, truth_map)`: Aggregate score (bool or float)

Example Usage
-------------
>>> from src.eval import exact_match, bleu, rouge_l
>>> 
>>> pred = {"file.py": "def foo(): return 1"}
>>> truth = {"file.py": "def foo(): return 1"}
>>> 
>>> # Exact match (returns True for perfect match)
>>> exact_match.overall(pred, truth)
True
>>> 
>>> # BLEU-3 (returns 0-1 score)
>>> bleu.overall(pred, truth)
1.0
>>> 
>>> # ROUGE-L (returns 0-1 score)  
>>> rouge_l.overall(pred, truth)
1.0

Text Normalization
------------------
All metrics apply normalization before comparison:
- CRLF → LF conversion
- Trailing whitespace removal
- Case handling varies by metric

Performance Notes
-----------------
- BLEU and ROUGE have optional dependencies (sacrebleu, rouge-score)
- Fallback implementations are provided when libraries unavailable
- Per-file operations are O(n) in number of files
"""

from . import exact_match
from . import bleu
from . import rouge_l
from .reporter import Reporter

__all__ = [
    "exact_match",
    "bleu",
    "rouge_l",
    "Reporter",
]



