"""RQ3 Classification Analysis Module.

Analyzes failure/success classifications and their correlation
with model performance characteristics (bypass vs agent, difficulty, project size).
"""

__all__ = ["generate_all_rq3_figures", "RQ3Flags"]


def __getattr__(name: str):
    """Lazily import the full RQ3 pipeline and its optional dependencies."""
    if name in __all__:
        from .main import RQ3Flags, generate_all_rq3_figures

        values = {
            "generate_all_rq3_figures": generate_all_rq3_figures,
            "RQ3Flags": RQ3Flags,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
