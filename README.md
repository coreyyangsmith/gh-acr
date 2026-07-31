# GH-ACR: GitHub Auto Conflict Resolver

Merge conflict resolution pipeline for evaluating LLM-based and baseline methods on the [GitGoodBench](https://github.com/JetBrains-Research/git-good-bench) dataset.

---

## Quick Start

1. **Preprocess** the GitGoodBench data (optional if you already have a filtered dataset)
2. **Configure** the dataset path via `DATASET_CSV` (see [Dataset Configuration](#dataset-configuration))
3. **Run** the pipeline with the desired methods
4. **Analyze** results with built-in tools

```bash
# Run all methods (base_a, base_b, agent, bypass7, force_mix) on up to 10 scenarios
uv run python -m src.cli.run_all --mode clone --max-scenarios 10

# Analyze results (uses most recent data/*_results_all.csv by default)
uv run python -m src.analysis.compare_methods
```

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python and dependencies)
- Python 3.10+ (installed automatically by uv if needed)
- `OPENAI_API_KEY` (or other LLM API key) for agent-based methods
- `GITHUB_TOKEN` for cloning repositories

Install dependencies from the repository root:

```bash
uv sync                 # core deps
uv sync --extra analysis
uv sync --extra local-llm
uv sync --extra dev     # analysis + local-llm + pytest-cov
```

Copy `.env.example` to `.env` at the repository root and fill in your API keys.

---

## Data Preprocessing

After obtaining the [GitGoodBench](https://github.com/JetBrains-Research/git-good-bench) dataset, filter scenarios that include merge commit hashes. Follow these steps after cloning their repo to obtain the initial dataset:

```bash
# Default: uses DATA_PATH from config, outputs to data/git_good_bench_merge_commits.csv
uv run python -m src.dataset.processing.extract_merge_scenario_from_ggb

# Or specify paths explicitly
uv run python -m src.dataset.processing.extract_merge_scenario_from_ggb --input-csv data/git_good_bench.csv --output-csv data/git_good_bench_merge_commits.csv
```

**Remove duplicates** (by `merge_commit_hash`):

```bash
uv run python -m src.analysis.utils.remove_duplicates data/git_good_bench_merge_commits.csv
```

Output defaults to `data/uniques.csv` unless `--output-csv` is specified.

---

## Subset Creation

**Split by difficulty** (Easy, Medium, Hard):

```bash
uv run python -m src.dataset.processing.split_dataset_by_difficulty data/git_good_bench_merge_commits.csv
```

This writes `*_easy.csv`, `*_medium.csv`, and `*_hard.csv` next to the input file.

**Sample a random subset**:

```bash
uv run python -m src.dataset.processing.get_subset data/git_good_bench_merge_commits.csv --percent 10 --seed 42
uv run python -m src.dataset.processing.get_subset data/git_good_bench_merge_commits_easy.csv --percent 10 --seed 42
```

Output is written next to the input with a suffix like `_subset_10_seed42.csv`.

---

## Dataset Configuration

The pipeline loads scenarios from a CSV whose path is set by:

- **Environment variable** `DATASET_CSV` (overrides default), or
- **Default path** in `src/config/settings.py` (e.g. `data/git_good_bench_merge_commits_all.csv`)

```bash
# PowerShell
$env:DATASET_CSV="data/git_good_bench_merge_commits.csv"

# Bash
export DATASET_CSV=data/git_good_bench_merge_commits.csv
```

---

## Running the Pipeline

Now we can run the pipeline as we have our dataset. Copy the csv into this repository and configure the DATASET_CSV (above).

Use the consolidated entrypoint:

```bash
uv run python -m src.cli.run_all --mode clone [options]
```

### Supported methods

- `base_a` – Baseline: always select Parent A
- `base_b` – Baseline: always select Parent B
- `agent` – Single-turn LLM resolver
- `bypass7` – Multi-agent resolver with bypass/merge decisions
- `force_mix` – Multi-agent resolver that always takes the mix path

### Main parameters

| Parameter | Description |
|-----------|-------------|
| `--mode` | `clone` (default) – clone repos locally |
| `--methods` | Space-separated list of methods (default: all five) |
| `--max-scenarios N` | Limit total scenarios |
| `--n-easy N --n-medium M --n-hard K` | Sample by difficulty instead |
| `--model-name` | LLM override (see below) |
| `--results-filename FILE` | Custom output CSV name (under `data/`) |
| `--resume` | Keep existing CSV/ledger/failures log; skip units already recorded as success; re-run failures, degradations, and missing units |
| `--start-index N --end-index M` | Batch processing by row index |

### Model name formats

- `openai/<model>` – OpenAI API
- `local:<hf_repo_or_path>` – Local model via Transformers (e.g. `local:meta-llama/Llama-3.1-8B`)
- `groq:<model>` – Groq API (e.g. `groq:llama-3.1-8b-instant`)

### Examples

```bash
# Run all methods, limit to 50 scenarios
uv run python -m src.cli.run_all --mode clone --max-scenarios 50

# Run only baselines and agent
uv run python -m src.cli.run_all --mode clone --methods base_a base_b agent --max-scenarios 50

# Sample by difficulty with a specific model
uv run python -m src.cli.run_all --n-easy 20 --n-medium 20 --n-hard 10 --mode clone --model-name openai/gpt-4o-mini

# Run bypass7 with a custom results file
uv run python -m src.cli.run_all --mode clone --methods base_a base_b agent bypass7 --model-name groq:llama-3.1-8b-instant --results-filename 2025_09_30_llama.csv
```

---

## Obtaining Results

### Output locations

- **Consolidated CSV**: `data/YYYY_MM_DD_results_all.csv` (or the name passed to `--results-filename`) — successful units only
- **Run ledger**: `data/<results_stem>_run_log.jsonl` — per-unit `success` / `failure` / `degraded` / `summary` records
- **Failures log**: `data/<results_stem>_failures.jsonl` — failure and degraded units only (for diagnosis and `--resume` re-runs)
- **Per-scenario artifacts**: `data/<model_name>/<scenario_id>/` (diffs, ground truth, agent outputs)
- **LLM failure traces**: `logs/llm_failures/` when an LLM call exhausts retries

Soft degradations (prompt truncation, JSON/heuristic fallbacks, unclear judge verdicts) complete without raising but are recorded as `degraded` (not written to the results CSV) so `--resume` can re-run them.

### Analyzing results

**Method comparison** (tables, boxplots, metrics):

```bash
# Uses most recent data/*_results_all.csv
uv run python -m src.analysis.compare_methods

# Specify a results file
uv run python -m src.analysis.compare_methods --results-csv data/2025_09_30_llama.csv

# Optional: filter by difficulty or file name
uv run python -m src.analysis.compare_methods --results-csv data/results.csv --difficulty easy,medium
```

**Tables and plots** (Pareto, cost vs quality, leaderboards):

```bash
uv run python -m src.analysis.main --results-csv data/YYYY_MM_DD_results_all.csv
```

Outputs go to `results/` by default.

---

## Groq Usage

Direct Groq model id: `groq:llama-3.1-8b-instant` (API: `llama-3.1-8b-instant`).

Published developer-tier limits (see [Groq rate limits](https://console.groq.com/docs/rate-limits)):
**30 RPM / 6,000 TPM / 14,400 RPD / 500,000 TPD**. Soft client defaults are ~90% of those ceilings (`27` RPM / `5400` TPM). Enable local pacing for concurrent runs:

```bash
$env:GROQ_API_KEY="<your_key>"
$env:RL_ENABLE_WAITING="1"
# Optional overrides / timeout / watchdog:
# $env:GROQ_REQUEST_TIMEOUT="120"
# $env:OPENROUTER_REQUEST_TIMEOUT="600"
# $env:OPENAI_REQUEST_TIMEOUT="600"
# $env:GHACR_LLM_REQUEST_TIMEOUT="600"
# $env:GHACR_WATCHDOG="1"
uv run python -m src.cli.run_all --mode clone --methods agent bypass7 `
  --model-name groq:llama-3.1-8b-instant `
  --concurrency 2 --method-concurrency 2 `
  --results-filename 2025_09_30_groq.csv `
  --watchdog
```

Check live health (heartbeat + latest ledger event) without starting work:

```bash
uv run python -m src.cli.run_all --status --results-filename 2025_09_30_groq.csv
```

If a run stalls or the watchdog soft-skips / aborts, diagnostics are written next to the results CSV (`*_watchdog_stacks.txt`). Soft-skip (`--watchdog-mode skip`, default) marks the unit as a ledger timeout and cancels further LLM retries; OpenAI/OpenRouter HTTP timeouts (default 600s) free blocked workers so concurrency can advance. Restart safely with `--resume` — completed `success` / `degraded` units in `*_run_log.jsonl` are skipped (timeout failures are retried):

```bash
uv run python -m src.cli.run_all --mode clone --methods agent bypass7 `
  --model-name groq:llama-3.1-8b-instant `
  --results-filename 2025_09_30_groq.csv `
  --resume --watchdog
```

Outputs are nested under `data/groq_<model>/...` for Windows-safe paths.
