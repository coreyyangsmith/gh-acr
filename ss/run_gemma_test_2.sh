#!/bin/bash
#SBATCH --job-name=gh-acr
#SBATCH --output=logs/gemma2-gh-acr-%j.out
#SBATCH --error=logs/gemma2-gh-acr-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=corey.yangsmith@ucalgary.ca

set -euo pipefail
mkdir -p logs

############################
# 0) Modules / toolchains  #
############################
module purge
module load gcc python/3.11 || true

# Prefer CUDA 12.4 (best for H100 + official cu124 wheels), else fall back to 12.1.
if module avail 2>/dev/null | grep -qE 'cuda/12\.4'; then
  module load cuda/12.4
  export TORCH_CUDA_FLAVOR="cu124"
elif module avail 2>/dev/null | grep -qE 'cuda/12\.1'; then
  module load cuda/12.1
  export TORCH_CUDA_FLAVOR="cu121"
else
  echo "WARNING: No cuda/12.4 or cuda/12.1 module found; proceeding with CPU-only torch."
  export TORCH_CUDA_FLAVOR="cpu"
fi

# Avoid CVMFS Arrow/pyarrow (compiled vs NumPy 1.x → ABI crash with NumPy 2.x)
module unload arrow || true

# Rust toolchain (for any build-time utilities some deps might use)
if module avail 2>/dev/null | grep -qi '^rust'; then module load rust || true; fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "No cargo found; installing temporary Rust toolchain to \$SLURM_TMPDIR..."
  export RUSTUP_HOME="$SLURM_TMPDIR/.rustup"
  export CARGO_HOME="$SLURM_TMPDIR/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
  . "$CARGO_HOME/env"
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

# Local caches (node-local = faster, avoids $HOME quotas)
export HF_HOME="$SLURM_TMPDIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Pip knobs suitable for CC
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

##############################################
# 2) Quick fixups to your requirements file  #
##############################################
REQ_FILE="$SLURM_SUBMIT_DIR/src/requirements.txt"

# Fix common pins that cause resolver/ABI grief
if grep -Eq '(^|\s)packaging==25\.0(\s|$)' "$REQ_FILE"; then
  echo "Patching packaging pin (25.0 -> <25) in $REQ_FILE"
  sed -E -i 's/packaging==25\.0/packaging<25/g' "$REQ_FILE"
fi

# Ensure transformer pin uses '==' not '='
sed -E -i 's/^transformers=([0-9]+\.[0-9]+\.[0-9]+)/transformers==\1/' "$REQ_FILE" || true

# Keep your intended ranges for langchain/langgraph
sed -E -i 's/^langgraph[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langgraph>=0.5.4/' "$REQ_FILE" || true
sed -E -i 's/^langchain-core[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langchain-core>=0.3,<0.4/' "$REQ_FILE" || true

#####################################
# 3) CUDA-aware Torch preinstall    #
#####################################
# Install PyTorch first with the correct CUDA flavor (so HF picks GPU).
if [[ "${TORCH_CUDA_FLAVOR}" == "cu124" ]]; then
  TORCH_IDX="https://download.pytorch.org/whl/cu124"
  python -m pip install --upgrade --index-url "$TORCH_IDX" "torch==2.5.1"
elif [[ "${TORCH_CUDA_FLAVOR}" == "cu121" ]]; then
  TORCH_IDX="https://download.pytorch.org/whl/cu121"
  python -m pip install --upgrade --index-url "$TORCH_IDX" "torch==2.5.1"
else
  # CPU fallback
  TORCH_IDX="https://download.pytorch.org/whl/cpu"
  python -m pip install --upgrade --index-url "$TORCH_IDX" "torch==2.5.1"
fi

# Core HF runtime bits (for CodeGemma + accelerate)
python -m pip install --upgrade "transformers==4.56.1" "accelerate>=0.33.0" "safetensors>=0.4.5" "sentencepiece>=0.2.0"

############################################
# 4) Extra constraints to avoid ABI issues #
############################################
cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
# Use a NumPy 2 stack and wheel-built pyarrow to avoid CVMFS Arrow ABI clashes
numpy==2.2.*
pandas==2.2.3
pyarrow==16.1.*
# Optional: keep scipy modern but available as wheel
scipy==1.14.*
TXT

####################################
# 5) Preflight download (all wheels)
####################################
echo "=== Preflight: checking for missing wheels ==="
if ! python -m pip download --only-binary=:all: -c "$SLURM_TMPDIR/constraints.txt" \
      -r "$REQ_FILE" -d "$SLURM_TMPDIR/wheels"; then
  echo "At least one dependency has no prebuilt wheel."
  echo "→ Pin a wheelable version or vendor a wheel into \$SLURM_TMPDIR/wheels."
  exit 1
fi

#############################################
# 6) Install project requirements (eager)   #
#############################################
python -m pip install \
  -c "$SLURM_TMPDIR/constraints.txt" \
  -r "$REQ_FILE" \
  --upgrade --upgrade-strategy eager

#############################################
# 7) Sanity check the critical versions     #
#############################################
echo "=== CUDA & Torch sanity ==="
nvidia-smi || true
python - <<'PY'
import torch, transformers, numpy, pandas
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
try:
    import pyarrow as pa
    print("pyarrow", pa.__version__)
except Exception as e:
    print("pyarrow import failed:", e)
PY

# Helpful PyTorch allocator tweak for large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

################################
# 8) Run your actual workload  #
################################
echo "Step 1: Running Job..." | tee "logs/job_${SLURM_JOB_ID}.out"
srun --export=ALL python -m src.cli.run_all \
  --methods agent agent2 agent3 new_bypass2 new_bypass3 bypass7 \
  --mode clone \
  --model-name local:google/codegemma-7b-it \
  --results-filename 2025_09_10_results_gemma_hard_sample.csv