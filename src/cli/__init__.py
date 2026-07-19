"""Command-line interface package for the merge conflict resolution pipeline.

This package provides CLI entry points for running the merge resolution
pipeline on benchmark datasets. It handles scenario processing, result
collection, and output file generation.

Modules
-------
- **runner**: Core logic for processing individual scenarios
- **run_all**: Main entry point for batch processing

Entry Points
------------
The primary entry point is `run_all.main()`, which can be invoked via:

```bash
uv run python -m src.cli.run_all --max-scenarios 10 --methods bypass7
```

Or with tyro CLI parsing:

```python
from src.cli.run_all import main
main(max_scenarios=10, methods=["bypass7"])
```

Configuration
-------------
Key parameters:
- **max_scenarios**: Limit number of scenarios to process
- **mode**: Processing mode (currently only "clone" supported)
- **methods**: List of evaluation methods to run
- **model_name**: LLM model to use for agent-based methods
- **n_easy/n_medium/n_hard**: Sample by difficulty

Output Files
------------
Results are written to:
- `data/<model_name>/<scenario_id>/`: Shared conflict inputs per file
- `data/<model_name>/<scenario_id>/<method>/`: Per-agent call artifacts
  (`summarizer/`, `analyzer/`, `planner/`, `resolver/`, `reviewer/`, `final/`)
- `data/YYYY_MM_DD_results_all.csv`: Consolidated metrics

Example Usage
-------------
>>> # Run all evaluation methods on 10 scenarios
>>> uv run python -m src.cli.run_all --max-scenarios 10

>>> # Run only bypass7 with a specific model
>>> uv run python -m src.cli.run_all --methods bypass7 --model-name openai/gpt-4o-mini

>>> # Sample by difficulty
>>> uv run python -m src.cli.run_all --n-easy 5 --n-medium 5 --n-hard 5
"""

from .runner import run_and_save_report, RESULTS_SCHEMA_COLUMNS
from .run_all import main

__all__ = [
    "run_and_save_report",
    "RESULTS_SCHEMA_COLUMNS",
    "main",
]
