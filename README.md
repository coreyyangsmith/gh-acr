# Observability
We use open-source LangFuse for LLM observability and debugging. Navigate to the [LangFuse GitHub](https://github.com/langfuse/langfuse) for detailed installation instructions.

After cloning the repository, navigate to the installed folder and run:

```bash
docker compose up
```

After setting up your project, populate the following environment variables:
```bash
LANGFUSE_ENABLED=1
LANGFUSE_HOST="http://localhost:3000"
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

# Data Preprocessing
After running the GitGoodBench data preprocessing, we must do additional filtering for our purposes.
```
python -m src.dataset.process_ggb --input-csv 2025-08-11-ggb --output-csv git_good_bench_merge_commits
```

Takes in data/git_good_bench.csv and outputs `git_good_bench_merge_commits.csv`



# Subset
If you're processing a subset of the entire dataset, we can split by difficulty or a random seed % of the overall dataset:

Split to Easy, Medium, Hard
```
python -m src.dataset.split_utils data/git_good_bench_merge_commits.csv
```

```
python -m src.dataset.get_subset data/git_good_bench_merge_commits_easy.csv --percent 10 --seed 42

python -m src.dataset.get_subset data/git_good_bench_merge_commits.csv --percent 5 --seed 42
```

# Inference
Go to `src/dataset/loader.py` and set the appropriate file path
## Single scenario
- Parameters
  - `scenario_id` (positional): dataset index (e.g., `1505`) or slug from CSV `id`.
  - `--mode` (`api`|`clone`): how to read repo data. Default: `api`.
  - `--eval-method` (`base_a`|`base_b`|`agent`|`multi`): resolver. Default: `agent`.
  - `--model-name` (optional): LLM backend name (e.g., `openai/gpt-4o-mini`).

Examples
```bash
python -m src.cli.run_single 1505 --mode clone --eval-method base_a
python -m src.cli.run_single some_repo__some_pr --mode api --eval-method agent --model-name openai/gpt-4o-mini
```

## Batch (multiple scenarios)
- Parameters
  - `--max-scenarios N`: process first N scenarios (mutually exclusive with difficulty sampling).
  - `--n-easy N --n-medium M --n-hard K`: sample per difficulty buckets.
  - `--mode` (`api`|`clone`): default `api`.
  - `--eval-method` (`base_a`|`base_b`|`agent`|`multi`).
  - `--model-name` (optional): LLM backend.

Examples
```bash
python -m src.cli.run_batch --max-scenarios 10 --mode clone --eval-method base_a
python -m src.cli.run_batch --max-scenarios 10 --mode clone --eval-method base_b
python -m src.cli.run_batch --max-scenarios 10 --mode clone --eval-method agent
python -m src.cli.run_batch --max-scenarios 10 --mode clone --eval-method multi

python -m src.cli.run_batch --n-easy 10 --n-medium 10 --n-hard 10 --mode clone --eval-method agent
```

## Run all methods over the dataset
- Command runs each of: `base_a`, `base_b`, `agent`, `multi` for the same subset.
- Parameters
  - `--mode` (`api`|`clone`): default `clone`.
  - `--methods ...`: optional subset list of methods to run (space-separated).
  - `--max-scenarios N`: optional cap on total scenarios.
  - `--n-easy N --n-medium M --n-hard K`: instead of `max-scenarios`, sample by difficulty.
  - `--model-name`: optional LLM backend override.

Examples
```bash
python -m src.cli.run_all --mode clone
python -m src.cli.run_all --methods base_a base_b agent --max-scenarios 50 --mode clone
python -m src.cli.run_all --methods base_a base_b agent multi bypass --max-scenarios 50 --mode clone --model_name "local:distilbert/distilgpt2"
python -m src.cli.run_all --methods base_a base_b agent multi bypass --max-scenarios 50 --mode clone

python -m src.cli.run_all --methods multi --max-scenarios 50 --mode clone
python -m src.cli.run_all --methods bypass --max-scenarios 50 --mode clone
python -m src.cli.run_all --methods dynamic --max-scenarios 50 --mode clone

python -m src.cli.run_all --n-easy 20 --n-medium 20 --n-hard 10 --mode clone --model-name openai/gpt-4o-mini

```

# Results
```
python -m src.results.main
```


# Utils
Extract Subset
```
python -m src.dataset.extract_samples_from_subset --ids-csv data/2025_08_11_results_all.csv --source-csv data/git_good_bench_merge_commits_easy_subset_10_seed42.csv --output-csv data/source_filtered.csv --ids-column id --source-id-column ,
```