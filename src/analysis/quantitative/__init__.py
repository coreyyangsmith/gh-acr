"""Quantitative Change Metrics Analysis Module.

Computes commit counts, PR length, LOC/SLOC changes, and diff-based
metrics for each merge scenario version (Previous, A, B, Ground Truth,
Agent, Bypass), then correlates these metrics against RQ2 performance
data and RQ3 classification labels.
"""

from .main import generate_all_quantitative, QuantFlags

__all__ = ["generate_all_quantitative", "QuantFlags"]
