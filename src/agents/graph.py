"""Public facade re-exporting the *merge pipeline* graph builder.

This separation keeps higher-level *agent* abstractions decoupled from the
low-level merge-resolution logic located in :pymod:`src.merge_pipeline`.
"""

from .graph_router import build_graph
