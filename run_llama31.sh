#!/bin/bash
#SBATCH --job-name=llama318-batch1
#SBATCH --output=logs/llama318-batch1-%j.out
#SBATCH --error=logs/llama318-batch1-%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOURNAME@EMAIL.ca

set -euo pipefail
mkdir -p logs

############################
# BATCH CONFIGURATION
############################
BATCH_NUM=1
START_INDEX=0
END_INDEX=75

############################
# DEBUG: Log job start info
############################
echo "=================================================================="
echo "JOB STARTED: $(date)"
echo "=================================================================="
echo "BATCH: $BATCH_NUM (rows $START_INDEX to $END_INDEX)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-not set}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-not set}"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "=================================================================="

############################
# 0) Modules / toolchains  #
############################
# Use a clean, predictable module stack on Compute Canada.
module reset
module load StdEnv/2023 gcc/12.3 python/3.11
# DO NOT load 'arrow' here (it brings a pyarrow built for NumPy 1.x).
# If a login profile exported PYTHONPATH, nuke it:
unset PYTHONPATH

# Rust toolchain (prefer module or $HOME install; else guarded rustup to node-local scratch)
if module avail rust 2>&1 | grep -qi rust; then module load rust || true; fi
if [ -x "$HOME/.cargo/bin/cargo" ]; then
  export RUSTUP_HOME="$HOME/.rustup"
  export CARGO_HOME="$HOME/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "No cargo found; installing temporary Rust toolchain to \$SLURM_TMPDIR..."
  export RUSTUP_HOME="$SLURM_TMPDIR/.rustup"
  export CARGO_HOME="$SLURM_TMPDIR/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
  if curl -fsSLI https://sh.rustup.rs >/dev/null; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
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

# Hugging Face / tokenizers knobs (safe defaults on CC)
export HF_HOME="$SLURM_SUBMIT_DIR/data/models"        # persistent cache in your project tree
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
# If tokenizer "Already borrowed" persists and you cannot change code to allocate
# separate tokenizers per thread, as a last resort you can globally disable fast tokenizers:
# export TRANSFORMERS_NO_FAST_TOKENIZER=1

# (Optional) PyTorch allocator tuning for huge prompts
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Faster temp on node
export TMPDIR="$SLURM_TMPDIR"

# Compute Canada wheels cache
export PIP_FIND_LINKS="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
$SLURM_TMPDIR/wheels"
mkdir -p "$SLURM_TMPDIR/wheels"

export PIP_ONLY_BINARY=":all:"          # never build from source
export PIP_NO_BUILD_ISOLATION=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_NO_CACHE_DIR=1

python -m pip install --upgrade pip wheel setuptools

##############################################
# 2) Tidy up your requirements file (in-tree)
##############################################
REQ_FILE="$SLURM_SUBMIT_DIR/src/requirements.txt"

# If repo pins packaging==25.0, relax to <25 (common resolver fix)
if grep -Eq '(^|\s)packaging==25\.0(\s|$)' "$REQ_FILE"; then
  echo "Patching packaging pin (25.0 -> <25) in $REQ_FILE"
  sed -E -i 's/packaging==25\.0/packaging<25/g' "$REQ_FILE"
fi

# (Optional) Modernize these if pinned too low (safe no-ops otherwise)
sed -E -i 's/^langgraph[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langgraph>=0.5.4/' "$REQ_FILE" || true
sed -E -i 's/^langchain-core[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langchain-core>=0.3,<0.4/' "$REQ_FILE" || true

##############################################
# 3) Choose ONE NumPy/pyarrow compatibility lane
##############################################
# Lane A (recommended): Hide site pyarrow (we already did by not loading arrow and unsetting PYTHONPATH)
# and DO NOT require pyarrow in your venv. If your requirements.txt lists pyarrow, drop it:
# (Uncomment the next line if pyarrow is listed and you don't need it.)
# sed -i '/^pyarrow[[:space:]=<>].*/d' "$REQ_FILE"

# Lane B (if you *do* need pyarrow features): install pyarrow wheel matching your venv NumPy.
# Keep modern NumPy and pull a compatible pyarrow into the venv:
# python -m pip install --only-binary=:all: --no-build-isolation "pyarrow==16.1.*"

############################################
# 4) Guard versions before solving (helps resolver)
############################################
python -m pip install \
  "packaging<25" \
  "langchain-core>=0.3,<0.4" \
  "langgraph>=0.5.4" \
  "pandas==2.2.3"
# ^ pandas 2.2.x works with NumPy 1.23+ and won't force pyarrow unless it's importable

####################################
# 5) Preflight: download all wheels (fail-fast if any sdist)
####################################
echo "=== Preflight: checking for missing wheels ==="
cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
pandas==2.2.3
TXT

if ! python -m pip download --only-binary=:all: -c "$SLURM_TMPDIR/constraints.txt" \
      -r "$REQ_FILE" -d "$SLURM_TMPDIR/wheels"; then
  echo "At least one dependency has no prebuilt wheel."
  echo "→ Pin a wheelable version or vendor a wheel into \$SLURM_TMPDIR/wheels."
  exit 1
fi

#############################################
# 6) Install project requirements (eager)
#############################################
python -m pip install \
  -c "$SLURM_TMPDIR/constraints.txt" \
  -r "$REQ_FILE" \
  --upgrade --upgrade-strategy eager

#############################################
# 7) Sanity check imports (confirms no site pyarrow leak)
#############################################
python - <<'PY'
import sys, numpy as np
print("numpy:", np.__version__)
try:
    import pyarrow as pa
    print("pyarrow:", pa.__version__, "→", pa.__file__)
except Exception as e:
    print("pyarrow not importable (ok if unused):", type(e).__name__, e)
import pandas as pd
print("pandas:", pd.__version__)
PY

################################
# 8) Configure LLM/Truncation Settings
################################
echo "Configuring LLM and truncation settings..."

# Enable debug diagnostics
export GHACR_DEBUG=1

# Batch size: 1 = fully process each scenario (all methods) before cleanup
# This ensures repo is cloned once, used for all methods, then cleaned up
export BATCH_SIZE=1

# Truncation configuration - CRITICAL for proper context window handling
# Use 2048 output tokens (sufficient for merge conflict resolution)
export LOCAL_MAX_NEW_TOKENS=2048
export LLAMA_MAX_NEW_TOKENS=2048

# Truncation side: "left" keeps the end of the prompt (usually the code/diff)
export LOCAL_TRUNCATION_SIDE=left

# Buffer tokens to prevent edge-case overflows
export LOCAL_TOKENIZER_BUFFER_TOKENS=512
export TOKENIZER_BUFFER_TOKENS=512

# LLM generation parameters
export LLAMA_TEMPERATURE=0.7
export LLAMA_TOP_P=0.9

# Disable tokenizers parallelism to prevent "Already borrowed" errors
export TOKENIZERS_PARALLELISM=false

# Log level for detailed debugging
export LOG_LEVEL=INFO

echo "LLM Configuration:"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  LOCAL_MAX_NEW_TOKENS=$LOCAL_MAX_NEW_TOKENS"
echo "  LLAMA_MAX_NEW_TOKENS=$LLAMA_MAX_NEW_TOKENS"
echo "  LOCAL_TRUNCATION_SIDE=$LOCAL_TRUNCATION_SIDE"
echo "  LOCAL_TOKENIZER_BUFFER_TOKENS=$LOCAL_TOKENIZER_BUFFER_TOKENS"
echo "  LLAMA_TEMPERATURE=$LLAMA_TEMPERATURE"
echo "  LLAMA_TOP_P=$LLAMA_TOP_P"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "  GHACR_DEBUG=$GHACR_DEBUG"

################################
# 9) GPU diagnostics before run
################################
echo ""
echo "=== GPU Status Before Run ==="
nvidia-smi || echo "nvidia-smi not available"
echo ""

################################
# 10) Run your actual workload
################################
echo "=================================================================="
echo "Starting pipeline run: $(date)"
echo "BATCH $BATCH_NUM: Processing rows $START_INDEX to $END_INDEX"
echo "=================================================================="
cd "$SLURM_SUBMIT_DIR"

# Run with explicit error handling and timing
START_TIME=$(date +%s)

srun --export=ALL python -m src.cli.run_all \
  --methods agent bypass7 \
  --mode clone \
  --model-name local:meta-llama/Llama-3.1-8B-Instruct \
  --results-filename 2026_01_08_results_llama31_8_batch${BATCH_NUM}.csv \
  --start-index $START_INDEX \
  --end-index $END_INDEX

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=================================================================="
echo "JOB COMPLETED: $(date)"
echo "BATCH $BATCH_NUM: Processed rows $START_INDEX to $END_INDEX"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: ${ELAPSED}s ($(($ELAPSED / 60))m $(($ELAPSED % 60))s)"
echo "=================================================================="

# Final GPU status
echo ""
echo "=== GPU Status After Run ==="
nvidia-smi || echo "nvidia-smi not available"

exit $EXIT_CODE

