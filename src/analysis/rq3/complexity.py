"""Code complexity metrics calculation using radon.

Provides functions to calculate cyclomatic complexity, Halstead metrics,
maintainability index, and raw metrics for Python code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# Radon imports
from radon.complexity import cc_visit, SCORE
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze


logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Container for all code complexity metrics.

    Attributes
    ----------
    sloc : int
        Source lines of code (non-blank, non-comment)
    lloc : int
        Logical lines of code
    comments : int
        Number of comment lines
    multi_comments : int
        Number of multi-line comment lines
    blank : int
        Number of blank lines
    cc_total : float
        Total cyclomatic complexity across all functions
    cc_avg : float
        Average cyclomatic complexity per function
    cc_max : float
        Maximum cyclomatic complexity of any function
    cc_count : int
        Number of functions/methods analyzed
    h_vocabulary : float
        Halstead vocabulary (h1 + h2)
    h_length : float
        Halstead program length (N1 + N2)
    h_difficulty : float
        Halstead difficulty
    h_effort : float
        Halstead effort
    h_bugs : float
        Estimated bugs (Halstead)
    h_time : float
        Estimated time to program (seconds)
    h_volume : float
        Halstead volume
    mi_score : float
        Maintainability Index (0-100, higher is better)
    parse_error : bool
        Whether the code failed to parse
    error_message : str
        Error message if parsing failed
    """

    # Raw metrics
    sloc: int = 0
    lloc: int = 0
    comments: int = 0
    multi_comments: int = 0
    blank: int = 0

    # Cyclomatic complexity
    cc_total: float = 0.0
    cc_avg: float = 0.0
    cc_max: float = 0.0
    cc_count: int = 0

    # Halstead metrics
    h_vocabulary: float = 0.0
    h_length: float = 0.0
    h_difficulty: float = 0.0
    h_effort: float = 0.0
    h_bugs: float = 0.0
    h_time: float = 0.0
    h_volume: float = 0.0

    # Maintainability
    mi_score: float = 0.0

    # Error tracking
    parse_error: bool = False
    error_message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def empty(cls, error_message: str = "") -> "CodeMetrics":
        """Create empty metrics with error flag."""
        return cls(parse_error=True, error_message=error_message)


def strip_think_tags(code: str) -> str:
    """Remove <think>...</think> blocks from code (Qwen models).

    Parameters
    ----------
    code : str
        Raw code that may contain think tags

    Returns
    -------
    str
        Code with think tags removed
    """
    # Pattern to match <think>...</think> including newlines
    pattern = r"<think>.*?</think>\s*"
    cleaned = re.sub(pattern, "", code, flags=re.DOTALL)
    return cleaned.strip()


def clean_code_for_analysis(code: str, strip_think: bool = True) -> str:
    """Clean code for complexity analysis.

    Parameters
    ----------
    code : str
        Raw code string
    strip_think : bool
        Whether to strip <think> tags

    Returns
    -------
    str
        Cleaned code ready for analysis
    """
    if strip_think:
        code = strip_think_tags(code)

    # Remove any leading/trailing whitespace
    code = code.strip()

    return code


def calculate_raw_metrics(code: str) -> dict:
    """Calculate raw metrics using radon.

    Parameters
    ----------
    code : str
        Python source code

    Returns
    -------
    dict
        Raw metrics (sloc, lloc, comments, multi, blank)
    """
    try:
        raw = analyze(code)
        return {
            "sloc": raw.sloc,
            "lloc": raw.lloc,
            "comments": raw.comments,
            "multi_comments": raw.multi,
            "blank": raw.blank,
        }
    except Exception as e:
        logger.debug(f"Error calculating raw metrics: {e}")
        return {
            "sloc": 0,
            "lloc": 0,
            "comments": 0,
            "multi_comments": 0,
            "blank": 0,
        }


def calculate_cyclomatic_complexity(code: str) -> dict:
    """Calculate cyclomatic complexity metrics using radon.

    Parameters
    ----------
    code : str
        Python source code

    Returns
    -------
    dict
        CC metrics (cc_total, cc_avg, cc_max, cc_count)
    """
    try:
        blocks = cc_visit(code)

        if not blocks:
            return {
                "cc_total": 0.0,
                "cc_avg": 0.0,
                "cc_max": 0.0,
                "cc_count": 0,
            }

        complexities = [block.complexity for block in blocks]
        return {
            "cc_total": sum(complexities),
            "cc_avg": np.mean(complexities) if complexities else 0.0,
            "cc_max": max(complexities) if complexities else 0.0,
            "cc_count": len(blocks),
        }
    except Exception as e:
        logger.debug(f"Error calculating CC: {e}")
        return {
            "cc_total": 0.0,
            "cc_avg": 0.0,
            "cc_max": 0.0,
            "cc_count": 0,
        }


def calculate_halstead_metrics(code: str) -> dict:
    """Calculate Halstead metrics using radon.

    Parameters
    ----------
    code : str
        Python source code

    Returns
    -------
    dict
        Halstead metrics
    """
    try:
        halstead_result = h_visit(code)

        # h_visit returns a list of HalsteadReport for each function
        # We aggregate across all functions
        if not halstead_result:
            return {
                "h_vocabulary": 0.0,
                "h_length": 0.0,
                "h_difficulty": 0.0,
                "h_effort": 0.0,
                "h_bugs": 0.0,
                "h_time": 0.0,
                "h_volume": 0.0,
            }

        # Sum/average across all function-level reports
        total = halstead_result.total if hasattr(halstead_result, 'total') else halstead_result

        # Handle both module-level and function-level results
        if hasattr(total, 'vocabulary'):
            return {
                "h_vocabulary": total.vocabulary or 0.0,
                "h_length": total.length or 0.0,
                "h_difficulty": total.difficulty or 0.0,
                "h_effort": total.effort or 0.0,
                "h_bugs": total.bugs or 0.0,
                "h_time": total.time or 0.0,
                "h_volume": total.volume or 0.0,
            }
        else:
            # Fallback for different radon versions
            return {
                "h_vocabulary": 0.0,
                "h_length": 0.0,
                "h_difficulty": 0.0,
                "h_effort": 0.0,
                "h_bugs": 0.0,
                "h_time": 0.0,
                "h_volume": 0.0,
            }
    except Exception as e:
        logger.debug(f"Error calculating Halstead: {e}")
        return {
            "h_vocabulary": 0.0,
            "h_length": 0.0,
            "h_difficulty": 0.0,
            "h_effort": 0.0,
            "h_bugs": 0.0,
            "h_time": 0.0,
            "h_volume": 0.0,
        }


def calculate_maintainability_index(code: str) -> float:
    """Calculate Maintainability Index using radon.

    Parameters
    ----------
    code : str
        Python source code

    Returns
    -------
    float
        Maintainability Index (0-100, higher is better)
    """
    try:
        mi = mi_visit(code, True)  # True = use multi-line strings
        return mi if mi is not None else 0.0
    except Exception as e:
        logger.debug(f"Error calculating MI: {e}")
        return 0.0


def calculate_metrics(
    code: str,
    strip_think: bool = True,
) -> CodeMetrics:
    """Calculate all code complexity metrics.

    Parameters
    ----------
    code : str
        Python source code
    strip_think : bool
        Whether to strip <think> tags (for Qwen models)

    Returns
    -------
    CodeMetrics
        All calculated metrics
    """
    if not code or not code.strip():
        return CodeMetrics.empty("Empty code")

    # Clean the code
    cleaned_code = clean_code_for_analysis(code, strip_think=strip_think)

    if not cleaned_code:
        return CodeMetrics.empty("Code empty after cleaning")

    try:
        # Calculate all metrics
        raw = calculate_raw_metrics(cleaned_code)
        cc = calculate_cyclomatic_complexity(cleaned_code)
        halstead = calculate_halstead_metrics(cleaned_code)
        mi = calculate_maintainability_index(cleaned_code)

        return CodeMetrics(
            # Raw metrics
            sloc=raw["sloc"],
            lloc=raw["lloc"],
            comments=raw["comments"],
            multi_comments=raw["multi_comments"],
            blank=raw["blank"],
            # Cyclomatic complexity
            cc_total=cc["cc_total"],
            cc_avg=cc["cc_avg"],
            cc_max=cc["cc_max"],
            cc_count=cc["cc_count"],
            # Halstead
            h_vocabulary=halstead["h_vocabulary"],
            h_length=halstead["h_length"],
            h_difficulty=halstead["h_difficulty"],
            h_effort=halstead["h_effort"],
            h_bugs=halstead["h_bugs"],
            h_time=halstead["h_time"],
            h_volume=halstead["h_volume"],
            # Maintainability
            mi_score=mi,
            # No error
            parse_error=False,
            error_message="",
        )
    except SyntaxError as e:
        return CodeMetrics.empty(f"Syntax error: {e}")
    except Exception as e:
        return CodeMetrics.empty(f"Error: {e}")


def get_complexity_grade(cc: float) -> str:
    """Get complexity grade from cyclomatic complexity score.

    Parameters
    ----------
    cc : float
        Cyclomatic complexity score

    Returns
    -------
    str
        Grade (A-F)
    """
    if cc <= 5:
        return "A"
    elif cc <= 10:
        return "B"
    elif cc <= 20:
        return "C"
    elif cc <= 30:
        return "D"
    elif cc <= 40:
        return "E"
    else:
        return "F"


def get_mi_grade(mi: float) -> str:
    """Get maintainability grade from MI score.

    Parameters
    ----------
    mi : float
        Maintainability Index score

    Returns
    -------
    str
        Grade (A-C)
    """
    if mi >= 20:
        return "A"
    elif mi >= 10:
        return "B"
    else:
        return "C"
