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
python -m src.cli.run_all --mode clone --max-scenarios 10

# Analyze results (uses most recent data/*_results_all.csv by default)
python -m src.analysis.compare_methods
```

---

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` (or other LLM API key) for agent-based methods
- `GITHUB_TOKEN` for cloning repositories

Copy `.env.example` to `.env` at the repository root and fill in your API keys.

---

## Data Preprocessing

After obtaining the [GitGoodBench](https://github.com/JetBrains-Research/git-good-bench) dataset, filter scenarios that include merge commit hashes. Follow these steps after cloning their repo to obtain the initial dataset:

```bash
# Default: uses DATA_PATH from config, outputs to data/git_good_bench_merge_commits.csv
python -m src.dataset.processing.extract_merge_scenario_from_ggb

# Or specify paths explicitly
python -m src.dataset.processing.extract_merge_scenario_from_ggb --input-csv data/git_good_bench.csv --output-csv data/git_good_bench_merge_commits.csv
```

**Remove duplicates** (by `merge_commit_hash`):

```bash
python -m src.analysis.utils.remove_duplicates data/git_good_bench_merge_commits.csv
```

Output defaults to `data/uniques.csv` unless `--output-csv` is specified.

---

## Subset Creation

**Split by difficulty** (Easy, Medium, Hard):

```bash
python -m src.dataset.processing.split_dataset_by_difficulty data/git_good_bench_merge_commits.csv
```

This writes `*_easy.csv`, `*_medium.csv`, and `*_hard.csv` next to the input file.

**Sample a random subset**:

```bash
python -m src.dataset.processing.get_subset data/git_good_bench_merge_commits.csv --percent 10 --seed 42
python -m src.dataset.processing.get_subset data/git_good_bench_merge_commits_easy.csv --percent 10 --seed 42
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
python -m src.cli.run_all --mode clone [options]
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
| `--start-index N --end-index M` | Batch processing by row index |

### Model name formats

- `openai/<model>` – OpenAI API
- `local:<hf_repo_or_path>` – Local model via Transformers (e.g. `local:meta-llama/Llama-3.1-8B`)
- `groq:<model>` – Groq API (e.g. `groq:llama-3.1-8b-instant`)

### Examples

```bash
# Run all methods, limit to 50 scenarios
python -m src.cli.run_all --mode clone --max-scenarios 50

# Run only baselines and agent
python -m src.cli.run_all --mode clone --methods base_a base_b agent --max-scenarios 50

# Sample by difficulty with a specific model
python -m src.cli.run_all --n-easy 20 --n-medium 20 --n-hard 10 --mode clone --model-name openai/gpt-4o-mini

# Run bypass7 with a custom results file
python -m src.cli.run_all --mode clone --methods base_a base_b agent bypass7 --model-name groq:llama-3.1-8b-instant --results-filename 2025_09_30_llama.csv
```

---

## Obtaining Results

### Output locations

- **Consolidated CSV**: `data/YYYY_MM_DD_results_all.csv` (or the name passed to `--results-filename`)
- **Per-scenario artifacts**: `data/<model_name>/<scenario_id>/` (diffs, ground truth, agent outputs)

### Analyzing results

**Method comparison** (tables, boxplots, metrics):

```bash
# Uses most recent data/*_results_all.csv
python -m src.analysis.compare_methods

# Specify a results file
python -m src.analysis.compare_methods --results-csv data/2025_09_30_llama.csv

# Optional: filter by difficulty or file name
python -m src.analysis.compare_methods --results-csv data/results.csv --difficulty easy,medium
```

**Tables and plots** (Pareto, cost vs quality, leaderboards):

```bash
python -m src.analysis.main --results-csv data/YYYY_MM_DD_results_all.csv
```

Outputs go to `results/` by default.

---

## Groq Usage

```bash
$env:GROQ_API_KEY="<your_key>"
python -m src.cli.run_all --mode clone --methods agent bypass7 --model-name groq:llama-3.1-8b-instant --results-filename 2025_09_30_groq.csv
```

Outputs are nested under `data/groq_<model>/...` for Windows-safe paths.