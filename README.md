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

Remove duplicates
python -m src.results.remove_duplicates --input-csv data/git_good_bench_merge_commits.csv

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

python -m src.cli.run_all --mode clone --methods agent bypass bypass2 bypass3 bypass_only

python -m src.cli.run_all --mode clone --methods agent bypass bypass_only bypass2 bypass3
python -m src.cli.run_all --mode clone --methods agent bypass_only2 bypass5 bypass6 bypass7 bypass8

python -m src.cli.run_all --mode clone --methods base_a base_b agent bypass7 --model-name openai/gpt-5-nano
python -m src.cli.run_all --mode clone --methods base_a base_b agent bypass7 --model-name openai/gpt-5-mini
python -m src.cli.run_all --mode clone --methods base_a base_b agent bypass7 --model-name openai/gpt-5
python -m src.cli.run_all --mode clone --methods base_a base_b agent bypass7 --model-name openai/gpt-5



python -m src.cli.run_all --mode clone --methods agent multi --model-name local:meta-llama/Llama-3.2-1B
python -m src.cli.run_all --mode clone --methods agent multi --model-name local:meta-llama/Llama-3.1-8B --results_filename 2025-09-10-llama-hard
python -m src.cli.run_all --mode clone --methods agent multi --model-name local:Qwen/Qwen3-8B --results_filename 2025-09-10-qwen-hard
python -m src.cli.run_all --mode clone --methods agent multi --model-name local:google/codegemma-7b-it --results_filename 2025-09-10-gemma-hard

# gpt-oss-20b (local via Transformers)
# Pre-download (optional, offline):
#   huggingface-cli download openai/gpt-oss-20b --local-dir data/models/openai__gpt-oss-20b --include "*"
# Recommended env (PowerShell one-liners):
#   $env:HF_LOCAL_ONLY="0"; $env:HF_DEVICE_MAP="auto"; $env:HF_TORCH_DTYPE="auto"; $env:HF_TRUST_REMOTE_CODE="1"; $env:GPT_OSS_REASONING_LEVEL="medium"
# Run with gpt-oss-20b:
python -m src.cli.run_all --mode clone --methods agent multi --model-name local:openai/gpt-oss-20b --results_filename 2025-09-11-gpt-oss-20b


```

# Results
```
python -m src.results.main
python -m src.results.compare_methods
```
python src/dataset/add_difficulty.py data/results/qwen_easy.csv --difficulty easy --output data/results/qwen_easy_processed.csv
python src/dataset/add_difficulty.py data/results/gemma_easy.csv --difficulty easy --output data/results/gemma_easy_processed.csv
python src/dataset/add_difficulty.py data/results/llama_easy.csv --difficulty easy --output data/results/llama_easy_processed.csv
python src/dataset/add_difficulty.py data/results/qwen_medium.csv --difficulty medium --output data/results/qwen_medium_processed.csv
python src/dataset/add_difficulty.py data/results/gemma_medium.csv --difficulty medium --output data/results/gemma_medium_processed.csv
python src/dataset/add_difficulty.py data/results/llama_medium.csv --difficulty medium --output data/results/llama_medium_processed.csv
python src/dataset/add_difficulty.py data/results/qwen_hard.csv --difficulty hard --output data/results/qwen_hard_processed.csv
python src/dataset/add_difficulty.py data/results/gemma_hard.csv --difficulty hard --output data/results/gemma_hard_processed.csv
python src/dataset/add_difficulty.py data/results/llama_hard.csv --difficulty hard --output data/results/llama_hard_processed.csv

Processing Outputs (find missing)
python -m src.results.find_missing_results --results-csv data/2025_08_29_results_all.csv --results-per-instance 10 --remove-prep


# Utils
Extract Subset
```
python -m src.dataset.extract_samples_from_subset --ids-csv data/2025_08_11_results_all.csv --source-csv data/git_good_bench_merge_commits_easy_subset_10_seed42.csv --output-csv data/source_filtered.csv --ids-column id --source-id-column ,
```