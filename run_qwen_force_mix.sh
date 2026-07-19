#!/bin/bash
#SBATCH --job-name=qwen-force-mix
#SBATCH --output=logs/qwen-force-mix-%j.out
#SBATCH --error=logs/qwen-force-mix-%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOURNAME@EMAIL.ca

set -euo pipefail
mkdir -p logs

############################
# BATCH CONFIGURATION
############################
RESULTS_FILE="$(date +%Y_%m_%d)_qwen_force_mix.csv"

# 10% pre-generated subset (191 rows, seed=42)
# Generated from data/git_good_bench_merge_commits_all.csv via:
#   python -m src.dataset.processing.get_subset data/git_good_bench_merge_commits_all.csv --percent 10 --seed 42
export DATASET_CSV="$SLURM_SUBMIT_DIR/data/git_good_bench_merge_commits_all_subset_10_seed42.csv"

############################
# DEBUG: Log job start info
############################
echo "=================================================================="
echo "JOB STARTED: $(date)"
echo "=================================================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-not set}"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "DATASET_CSV: $DATASET_CSV"
echo "=================================================================="

############################
# 0) Modules / toolchains  #
############################
module load gcc arrow/16.1.0 python/3.11 || true

# Rust toolchain
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

# Pip settings
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
# 2) Tidy up requirements file               #
##############################################
REQ_FILE="$SLURM_SUBMIT_DIR/src/requirements.txt"
if grep -Eq '(^|\s)packaging==25\.0(\s|$)' "$REQ_FILE"; then
  sed -E -i 's/packaging==25\.0/packaging<25/g' "$REQ_FILE"
fi
sed -E -i 's/^langgraph[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langgraph>=0.5.4/' "$REQ_FILE" || true
sed -E -i 's/^langchain-core[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langchain-core>=0.3,<0.4/' "$REQ_FILE" || true

############################################
# 3) Guard versions                        #
############################################
python -m pip install \
  "packaging<25" \
  "langchain-core>=0.3,<0.4" \
  "langgraph>=0.5.4"

cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
pandas==2.2.3
TXT

####################################
# 4) Preflight wheel download      #
####################################
echo "=== Preflight: checking for missing wheels ==="
if ! python -m pip download --only-binary=:all: -c "$SLURM_TMPDIR/constraints.txt" \
      -r "$REQ_FILE" -d "$SLURM_TMPDIR/wheels"; then
  echo "At least one dependency has no prebuilt wheel."
  exit 1
fi

#############################################
# 5) Install requirements                   #
#############################################
python -m pip install \
  -c "$SLURM_TMPDIR/constraints.txt" \
  -r "$REQ_FILE" \
  --upgrade --upgrade-strategy eager

#############################################
# 6) Sanity check                           #
#############################################
python - <<'PY'
import importlib
mods = ["langgraph", "langchain_core", "packaging"]
for m in mods:
    mod = importlib.import_module(m)
    ver = getattr(mod, "__version__", getattr(mod, "version", None))
    print(f"{m}: version={ver}")
PY

################################
# 7) Runtime configuration     #
################################
export GHACR_DEBUG=1
export BATCH_SIZE=1
export LOCAL_MAX_NEW_TOKENS=2048
export LOCAL_TRUNCATION_SIDE=left
export LOCAL_TOKENIZER_BUFFER_TOKENS=512
export TOKENIZER_BUFFER_TOKENS=512
export TOKENIZERS_PARALLELISM=false
export LOG_LEVEL=INFO

echo "DATASET_CSV: $DATASET_CSV"
echo "RESULTS_FILE: $RESULTS_FILE"

################################
# 8) Run pipeline              #
################################
echo "=================================================================="
echo "Starting pipeline run: $(date)"
echo "Method: force_mix | Model: local:Qwen/Qwen3-8B"
echo "Dataset: $DATASET_CSV"
echo "=================================================================="
cd "$SLURM_SUBMIT_DIR"

START_TIME=$(date +%s)

srun --export=ALL python -m src.cli.run_all \
  --methods force_mix \
  --mode clone \
  --model-name local:Qwen/Qwen3-8B \
  --results-filename "$RESULTS_FILE"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=================================================================="
echo "JOB COMPLETED: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: ${ELAPSED}s ($(($ELAPSED / 60))m $(($ELAPSED % 60))s)"
echo "=================================================================="
nvidia-smi || echo "nvidia-smi not available"

exit $EXIT_CODE
