#!/bin/bash
#SBATCH --job-name=qwen32-gh-acr
#SBATCH --output=logs/qwen32-gh-acr-%j.out
#SBATCH --error=logs/qwen32-gh-acr-%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
# Qwen3-32B is ~64 GB in bf16 (weights only). Two full H100s (80 GB each)
# leave ample headroom for the KV cache/activations of long merge-conflict
# prompts. device_map="auto" (see src/agents/handlers/local_backend.py) shards
# the model across both GPUs automatically.
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
# Larger host RAM than the 8B job: safetensors are memory-mapped during the
# sharded load, but 128 GB avoids OOM while materializing/offloading shards.
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=corey.yangsmith@ucalgary.ca

set -euo pipefail
mkdir -p logs

echo "=================================================================="
echo "JOB STARTED: $(date)"
echo "=================================================================="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-n/a}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-n/a}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-not set}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-n/a}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "HOSTNAME: $(hostname)"
echo "=================================================================="

############################
# 0) Modules / toolchains  #
############################
# Clean, predictable stack on the Alliance (Compute Canada) clusters.
# StdEnv/2023 ships GCCcore 12.3 + Python 3.11/3.12/3.13 and the matching
# wheelhouse that provides torch/transformers/accelerate as prebuilt wheels.
module reset
module load StdEnv/2023 gcc/12.3 python/3.11
# DO NOT load 'arrow' here (it brings a pyarrow built for NumPy 1.x).
# Nuke any PYTHONPATH a login profile may have exported.
unset PYTHONPATH

# Rust toolchain (module if present, else guarded rustup to node-local scratch).
# Only needed if a pure-source dependency slips through; harmless otherwise.
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

# HuggingFace / tokenizers knobs (persistent model cache in the project tree so
# the ~65 GB of Qwen3-32B shards are downloaded once and reused across jobs).
export HF_HOME="${SLURM_SUBMIT_DIR:-$(pwd)}/data/models"
export HF_HUB_CACHE="$HF_HOME"
export HF_CACHE_DIR="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

# PyTorch allocator tuning helps with the large, variable-length prompts.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TMPDIR="$SLURM_TMPDIR"

# Alliance wheelhouse (prebuilt, hardware-optimized wheels). Kept as FIND_LINKS
# plus a node-local wheels dir so the preflight download can vendor anything
# missing while on the login/build node.
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

#########################################################
# 2) Verify the cluster provides the wheels we now need #
#########################################################
# The pipeline's local backend (transformers 4.56.1) is what actually supports
# Qwen3; torch 2.7.1 + accelerate 1.10.1 are pinned in pyproject.toml under the
# [local-llm] extra. Confirm the Alliance wheelhouse can satisfy them for the
# loaded Python before we spend a GPU allocation. `avail_wheels` is the Alliance
# helper documented at https://docs.alliancecan.ca/wiki/Available_wheels
echo "=== Alliance wheelhouse availability (python 3.11) ==="
if command -v avail_wheels >/dev/null 2>&1; then
  avail_wheels torch --version 2.7.1 --python 3.11 || true
  avail_wheels transformers accelerate tokenizers safetensors huggingface_hub --python 3.11 || true
else
  echo "avail_wheels not on PATH (non-Alliance host?); relying on preflight download below."
fi

#############################################################
# 3) Install project + local-LLM deps from the wheelhouse   #
#############################################################
# NOTE: the repo migrated from src/requirements.txt to pyproject.toml. The
# [local-llm] extra pulls transformers==4.56.1, tokenizers==0.22.0,
# huggingface_hub==0.34.4, safetensors==0.5.3, accelerate==1.10.1, torch==2.7.1.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Guard versions first so the resolver sticks to a known-good lane.
python -m pip install \
  "packaging<25" \
  "langchain-core>=0.3,<0.4" \
  "langgraph>=0.5.4" \
  "pandas==2.2.3"

cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
pandas==2.2.3
TXT

####################################
# 4) Preflight download (fail-fast)
####################################
echo "=== Preflight: downloading all wheels for gh-acr[local-llm] ==="
if ! python -m pip download --only-binary=:all: -c "$SLURM_TMPDIR/constraints.txt" \
      ".[local-llm]" -d "$SLURM_TMPDIR/wheels"; then
  echo "At least one dependency has no prebuilt wheel for this Python/arch."
  echo "→ Pin a wheelable version or vendor a wheel into \$SLURM_TMPDIR/wheels."
  echo "→ Check with: avail_wheels <pkg> --python 3.11"
  exit 1
fi

#############################################
# 5) Editable install of the project        #
#############################################
python -m pip install \
  -c "$SLURM_TMPDIR/constraints.txt" \
  -e ".[local-llm]" \
  --upgrade --upgrade-strategy eager

#############################################
# 6) Sanity check the critical versions     #
#############################################
python - <<'PY'
import importlib
for m in ["torch", "transformers", "accelerate", "tokenizers",
          "safetensors", "langgraph", "langchain_core", "packaging"]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", getattr(mod, "version", None))
        print(f"{m}: version={ver} file={getattr(mod, '__file__', None)}")
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
# 7) Runtime / LLM configuration
################################
export GHACR_DEBUG=1
# Process each scenario end-to-end (all methods) before cleanup.
export BATCH_SIZE=1

# device_map="auto" shards Qwen3-32B across the two H100s.
export HF_DEVICE_MAP=auto
export HF_TORCH_DTYPE=bf16

# Truncation/context handling (native Qwen3 window is 32,768 tokens).
# Shared prompt_budget + TruncatingLLMWrapper use MODEL_COSTS for
# local:Qwen/Qwen3-32B (input≈30720, output=2048, total=32768) minus this buffer.
export LOCAL_MAX_NEW_TOKENS=2048
export LOCAL_TRUNCATION_SIDE=left
export LOCAL_TOKENIZER_BUFFER_TOKENS=512
export TOKENIZER_BUFFER_TOKENS=512
export PROMPT_TRUNCATION_BUFFER=64

# Qwen3-specific generation knobs (see local_backend.py Qwen3ChatWrapper).
# Match the non-thinking sampling profile recommended by Qwen for Qwen3.
export QWEN3_ENABLE_THINKING=0
export QWEN3_MAX_NEW_TOKENS=2048
export QWEN3_TEMPERATURE=0.7
export QWEN3_TOP_P=0.8
export QWEN3_TOP_K=20
# To match the API's 131K window, uncomment to enable YARN rope scaling:
# export QWEN3_ENABLE_YARN=1
# export QWEN3_YARN_FACTOR=4.0

export TOKENIZERS_PARALLELISM=false
export LOG_LEVEL=INFO

echo "LLM configuration:"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  HF_HOME=$HF_HOME"
echo "  HF_DEVICE_MAP=$HF_DEVICE_MAP  HF_TORCH_DTYPE=$HF_TORCH_DTYPE"
echo "  LOCAL_MAX_NEW_TOKENS=$LOCAL_MAX_NEW_TOKENS  QWEN3_MAX_NEW_TOKENS=$QWEN3_MAX_NEW_TOKENS"
echo "  LOCAL_TRUNCATION_SIDE=$LOCAL_TRUNCATION_SIDE  PROMPT_TRUNCATION_BUFFER=$PROMPT_TRUNCATION_BUFFER"
echo "  LOCAL_TOKENIZER_BUFFER_TOKENS=$LOCAL_TOKENIZER_BUFFER_TOKENS"
echo "  QWEN3_ENABLE_THINKING=$QWEN3_ENABLE_THINKING  QWEN3_TEMPERATURE=$QWEN3_TEMPERATURE"

################################
# 8) GPU diagnostics before run
################################
echo ""
echo "=== GPU Status Before Run ==="
nvidia-smi || echo "nvidia-smi not available"
echo ""

################################
# 9) Run the actual workload   #
################################
echo "=================================================================="
echo "Starting pipeline run: $(date)"
echo "Model: local:Qwen/Qwen3-32B | Methods: agent bypass7 | Mode: clone"
echo "=================================================================="

START_TIME=$(date +%s)

srun --export=ALL python -m src.cli.run_all \
  --methods agent bypass7 \
  --mode clone \
  --model-name local:Qwen/Qwen3-32B \
  --results-filename "$(date +%Y_%m_%d)_results_qwen32_all.csv"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=================================================================="
echo "JOB COMPLETED: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: ${ELAPSED}s ($(($ELAPSED / 60))m $(($ELAPSED % 60))s)"
echo "=================================================================="

echo ""
echo "=== GPU Status After Run ==="
nvidia-smi || echo "nvidia-smi not available"

exit $EXIT_CODE
