"""RQ3 Classification Analysis Module.

Analyzes failure/success classifications and their correlation
with model performance characteristics (bypass vs agent, difficulty, project size).
"""

from .main import generate_all_rq3_figures, RQ3Flags

__all__ = ["generate_all_rq3_figures", "RQ3Flags"]
