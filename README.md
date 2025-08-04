# Observability
We use open-source Arize Phoenix for LLM observability and debugging.

```bash
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest
```

clone | api
agent | base

# Data Preprocessing
After running the GitGoodBench data preprocessing, we must do additional filtering for our purposes.
```
python -m src.dataset.process_ggb
```

Takes in data/git_good_bench.csv and outputs `git_good_bench_merge_commits.csv`


# Inference
```bash
python -m src.cli.run_single 1505 --mode clone --eval-method base
```

```bash
python -m src.cli.run_batch --max-scenarios 10 --mode clone --eval-method agent
python -m src.cli.run_batch --max-scenarios 10 --mode clone --eval-method base

python -m src.cli.run_batch --n-easy 10 --n-medium 10 --n-hard 10 --mode clone --eval-method agent
```