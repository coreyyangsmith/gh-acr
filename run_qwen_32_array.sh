#!/bin/bash
# =============================================================================
# Qwen3-32B (local) — batched Slurm ARRAY job for Compute Canada (Fir).
#
# Splits the full GitGoodBench dataset (git_good_bench_merge_commits_all.csv,
# 1909 scenarios) into 20 index-range batches. Each array task is an
# INDEPENDENT job with its own 48h wall clock and its own results CSV, so the
# 20 jobs can run in parallel without racing on output.
#
# Submit all 20 jobs at once:
#     sbatch run_qwen_32_array.sh
# Submit/re-run a single batch (e.g. batch 7):
#     sbatch --array=7 run_qwen_32_array.sh
#
# RESUME: re-submitting the same array reuses the per-batch results file; the
# RunLedger in src.cli.run_all skips already-completed (scenario, method) units,
# so hitting the 48h wall is safe — just resubmit the affected indices.
# =============================================================================
#SBATCH --job-name=qwen32-batch
#SBATCH --output=logs/qwen32-batch-%A_%a.out
#SBATCH --error=logs/qwen32-batch-%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
# Qwen3-32B is ~66 GB in bf16; two full H100s (80 GB each) hold the weights
# (sharded via device_map="auto") plus KV cache for a full ~40k-token window.
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOURNAME@EMAIL.ca
# Fir requires an allocation account — uncomment and set yours:
# #SBATCH --account=def-YOURPI

set -euo pipefail
mkdir -p logs

############################
# BATCH CONFIGURATION      #
############################
# Total scenarios in the default dataset (git_good_bench_merge_commits_all.csv).
# Override the dataset with DATASET_CSV and update TOTAL_SCENARIOS to match.
TOTAL_SCENARIOS="${TOTAL_SCENARIOS:-1909}"
NUM_BATCHES="${NUM_BATCHES:-20}"
# Stable tag so resubmissions resume into the same per-batch CSV (no date).
RUN_TAG="${RUN_TAG:-qwen32_full}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
# ceil(TOTAL / NUM_BATCHES) so the union of all batches covers the whole set.
BATCH_SIZE=$(( (TOTAL_SCENARIOS + NUM_BATCHES - 1) / NUM_BATCHES ))
START_INDEX=$(( TASK_ID * BATCH_SIZE ))
END_INDEX=$(( START_INDEX + BATCH_SIZE ))   # end is EXCLUSIVE; pandas clamps
RESULTS_FILE="${RUN_TAG}_batch${TASK_ID}.csv"

echo "=================================================================="
echo "JOB STARTED: $(date)"
echo "=================================================================="
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-n/a}  TASK_ID: ${TASK_ID}"
echo "BATCH: ${TASK_ID}/${NUM_BATCHES}  rows [${START_INDEX}, ${END_INDEX})  (batch_size=${BATCH_SIZE})"
echo "RESULTS_FILE: ${RESULTS_FILE}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-n/a}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-not set}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-n/a}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "HOSTNAME: $(hostname)"
echo "=================================================================="

############################
# 0) Modules / toolchains  #
############################
# StdEnv/2023 provides GCCcore 12.3 + Python 3.11 and the matching wheelhouse
# with prebuilt torch/transformers/accelerate wheels.
module reset
module load StdEnv/2023 gcc/12.3 python/3.11
unset PYTHONPATH   # avoid a login-profile pyarrow/NumPy leak

# Rust toolchain (only needed if a pure-source dep slips through; harmless).
if module avail rust 2>&1 | grep -qi rust; then module load rust || true; fi
if [ -x "$HOME/.cargo/bin/cargo" ]; then
  export RUSTUP_HOME="$HOME/.rustup"
  export CARGO_HOME="$HOME/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
  export RUSTUP_HOME="$SLURM_TMPDIR/.rustup"
  export CARGO_HOME="$SLURM_TMPDIR/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
  if curl -fsSLI https://sh.rustup.rs >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable
    . "$CARGO_HOME/env"
  else
    echo "WARN: no outbound network from compute node; skipping Rust install." >&2
  fi
fi
rustc --version || true
cargo --version || true

##################################
# 1) Clean venv on node-local FS #
##################################
export PYTHONNOUSERSITE=1
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
python -V

# Persistent HF model cache in the project tree so the ~66 GB of Qwen3-32B
# shards are downloaded once and reused by every batch/job.
# NOTE: to avoid 20 tasks racing on a first-time download, warm the cache once
# before submitting the array (e.g. an interactive `hf download Qwen/Qwen3-32B`
# or a single `sbatch --array=0`). huggingface_hub file-locks downloads, but a
# pre-warmed cache is faster and gentler on the shared filesystem.
export HF_HOME="${SLURM_SUBMIT_DIR:-$(pwd)}/data/models"
export HF_HUB_CACHE="$HF_HOME"
export HF_CACHE_DIR="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TMPDIR="$SLURM_TMPDIR"

# Alliance wheelhouse (prebuilt, hardware-optimized wheels).
export PIP_FIND_LINKS="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
$SLURM_TMPDIR/wheels"
mkdir -p "$SLURM_TMPDIR/wheels"
export PIP_ONLY_BINARY=":all:"
export PIP_NO_BUILD_ISOLATION=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_NO_CACHE_DIR=1

python -m pip install --upgrade pip wheel setuptools

#########################################################
# 2) Verify the cluster provides the wheels we now need #
#########################################################
echo "=== Alliance wheelhouse availability (python 3.11) ==="
if command -v avail_wheels >/dev/null 2>&1; then
  avail_wheels torch --version 2.7.1 --python 3.11 || true
  avail_wheels transformers accelerate tokenizers safetensors huggingface_hub --python 3.11 || true
else
  echo "avail_wheels not on PATH (non-Alliance host?); relying on preflight download."
fi

#############################################################
# 3) Install project + local-LLM deps from the wheelhouse   #
#############################################################
# pyproject.toml [local-llm] extra: transformers==4.56.1, tokenizers==0.22.0,
# huggingface_hub==0.34.4, safetensors==0.5.3, accelerate==1.10.1, torch==2.7.1.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python -m pip install \
  "packaging<25" \
  "langchain-core>=0.3,<0.4" \
  "langgraph>=0.5.4" \
  "pandas==2.2.3"

cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
pandas==2.2.3
TXT

echo "=== Preflight: downloading all wheels for gh-acr[local-llm] ==="
if ! python -m pip download --only-binary=:all: -c "$SLURM_TMPDIR/constraints.txt" \
      ".[local-llm]" -d "$SLURM_TMPDIR/wheels"; then
  echo "At least one dependency has no prebuilt wheel for this Python/arch."
  echo "→ Check with: avail_wheels <pkg> --python 3.11"
  exit 1
fi

python -m pip install \
  -c "$SLURM_TMPDIR/constraints.txt" \
  -e ".[local-llm]" \
  --upgrade --upgrade-strategy eager

#############################################
# 4) Sanity check the critical versions     #
#############################################
python - <<'PY'
import importlib
for m in ["torch", "transformers", "accelerate", "tokenizers",
          "safetensors", "langgraph", "langchain_core", "packaging"]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", getattr(mod, "version", None))
        print(f"{m}: version={ver}")
    except Exception as e:
        print(f"{m}: IMPORT FAILED -> {type(e).__name__}: {e}")
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available(),
          "device_count:", torch.cuda.device_count(),
          "cuda:", getattr(torch.version, "cuda", None))
except Exception as e:
    print("torch cuda probe failed:", e)
PY

################################
# 5) Runtime / LLM configuration
################################
export GHACR_DEBUG=1
export BATCH_SIZE=1                 # process each scenario end-to-end before cleanup

# CRITICAL for local 32B at full context: keep scenario concurrency at 1 so a
# single 2xH100 job holds one active stream (~82 GB). Raising this risks OOM.
export INFERENCE_CONCURRENCY=1

# Shard across the two H100s in bf16.
export HF_DEVICE_MAP=auto
export HF_TORCH_DTYPE=bf16

# Truncation/context handling (native Qwen3 window is 32,768 tokens).
# Shared prompt_budget + TruncatingLLMWrapper use MODEL_COSTS for
# local:Qwen/Qwen3-32B (input≈30720, output=2048, total=32768) minus this buffer.
export LOCAL_MAX_NEW_TOKENS=2048
export LOCAL_TRUNCATION_SIDE=left
export LOCAL_TOKENIZER_BUFFER_TOKENS=512
export TOKENIZER_BUFFER_TOKENS=512
export PROMPT_TRUNCATION_BUFFER=4096

# Qwen3 generation knobs (see local_backend.py Qwen3ChatWrapper).
export QWEN3_ENABLE_THINKING=0
export QWEN3_MAX_NEW_TOKENS=2048
export QWEN3_TEMPERATURE=0.7
export QWEN3_TOP_P=0.8
export QWEN3_TOP_K=20
# To match the API's 131K window instead of the 40K native window:
# export QWEN3_ENABLE_YARN=1
# export QWEN3_YARN_FACTOR=4.0

export TOKENIZERS_PARALLELISM=false
export LOG_LEVEL=INFO

echo "LLM configuration: INFERENCE_CONCURRENCY=$INFERENCE_CONCURRENCY HF_DEVICE_MAP=$HF_DEVICE_MAP HF_TORCH_DTYPE=$HF_TORCH_DTYPE"
echo "  LOCAL_MAX_NEW_TOKENS=$LOCAL_MAX_NEW_TOKENS PROMPT_TRUNCATION_BUFFER=$PROMPT_TRUNCATION_BUFFER LOCAL_TRUNCATION_SIDE=$LOCAL_TRUNCATION_SIDE"

################################
# 6) GPU diagnostics before run
################################
echo ""
echo "=== GPU Status Before Run ==="
nvidia-smi || echo "nvidia-smi not available"
echo ""

################################
# 7) Run this batch            #
################################
echo "=================================================================="
echo "Starting batch ${TASK_ID}: rows [${START_INDEX}, ${END_INDEX}) — $(date)"
echo "Model: local:Qwen/Qwen3-32B | Methods: agent bypass7 | Mode: clone"
echo "=================================================================="

START_TIME=$(date +%s)

srun --export=ALL python -m src.cli.run_all \
  --methods agent bypass7 \
  --mode clone \
  --model-name local:Qwen/Qwen3-32B \
  --results-filename "$RESULTS_FILE" \
  --start-index "$START_INDEX" \
  --end-index "$END_INDEX"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=================================================================="
echo "JOB COMPLETED (batch ${TASK_ID}): $(date)"
echo "Rows processed: [${START_INDEX}, ${END_INDEX})"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: ${ELAPSED}s ($(($ELAPSED / 60))m $(($ELAPSED % 60))s)"
echo "=================================================================="

echo ""
echo "=== GPU Status After Run ==="
nvidia-smi || echo "nvidia-smi not available"

exit $EXIT_CODE
