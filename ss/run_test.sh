#!/bin/bash
#SBATCH --job-name=gh-acr
#SBATCH --output=logs/gh-acr-%j.out
#SBATCH --error=logs/gh-acr-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=corey.yangsmith@ucalgary.ca

set -euo pipefail
mkdir -p logs

# Base modules
module load gcc arrow/16.1.0 python/3.11 || true

# Rust toolchain (module if present, else rustup to node-local scratch)
if module avail 2>/dev/null | grep -qi '^rust'; then module load rust || true; fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "No cargo found; installing temporary Rust toolchain to \$SLURM_TMPDIR..."
  export RUSTUP_HOME="$SLURM_TMPDIR/.rustup"
  export CARGO_HOME="$SLURM_TMPDIR/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
  . "$CARGO_HOME/env"
  rustc --version
  cargo --version
else
  echo "Using system/module Rust toolchain:"
  rustc --version || true
  cargo --version || true
fi

# Venv in node-local scratch
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

echo "Step 0: Install dependencies"

# Wheelhouse links (Compute Canada)
export WHEEL_X86="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3"
export WHEEL_GEN="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic"
export WHEEL_GEN2="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
export LOCAL_WHEELS="$SLURM_TMPDIR/wheels"
mkdir -p "$LOCAL_WHEELS"

# Prefer these first; allow PyPI fallback
export PIP_FIND_LINKS="$WHEEL_X86 $WHEEL_GEN $WHEEL_GEN2 $LOCAL_WHEELS"

# CRITICAL: never build from source → avoids broken pyproject.toml builds
export PIP_ONLY_BINARY=":all:"
export PIP_NO_BUILD_ISOLATION=1
export PIP_DEFAULT_TIMEOUT=120

# Upgrade pip (PyPI allowed)
pip install --upgrade pip wheel setuptools

# Constraints you want to pin
cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
pandas==2.2.3
TXT

# Ensure a binary wheel for rapidfuzz (if not in CC wheelhouse)
if ! ls $WHEEL_X86/rapidfuzz-*.whl $WHEEL_GEN/rapidfuzz-*.whl $WHEEL_GEN2/rapidfuzz-*.whl >/dev/null 2>&1; then
  echo "No rapidfuzz wheel in CC wheelhouse; downloading a binary wheel..."
  pip download --only-binary=:all: --no-deps -d "$LOCAL_WHEELS" rapidfuzz==3.13.0 || true
fi

# (Optional but very useful) Pre-flight: name any deps without wheels up front
echo "=== Preflight: checking for missing wheels ==="
if ! pip download --only-binary=:all: -c "$SLURM_TMPDIR/constraints.txt" \
      -r "$SLURM_SUBMIT_DIR/src/requirements.txt" -d "$LOCAL_WHEELS"; then
  echo "At least one dependency has no wheel for this platform/Python."
  echo "Check the last lines above for the culprit, then either:"
  echo "  - Pin a version that has wheels, or"
  echo "  - Prebuild/vendor a wheel into \$LOCAL_WHEELS."
  exit 1
fi

# Install (wheelhouse first; PyPI wheels as fallback; no source builds)
pip install -c "$SLURM_TMPDIR/constraints.txt" -r "$SLURM_SUBMIT_DIR/src/requirements.txt"

echo "Step 1: Running Job..." | tee "logs/job_${SLURM_JOB_ID}.out"

# Run the job (ensure env is exported)
srun --export=ALL python -m src.cli.run_all \
  --methods base_a base_b agent multi \
  --max-scenarios 50 \
  --mode clone
