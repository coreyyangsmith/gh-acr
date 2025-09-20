#!/bin/bash
#SBATCH --job-name=gh-acr
#SBATCH --output=logs/llama8-gh-acr-%j.out
#SBATCH --error=logs/gh-acr-%j.err
#SBATCH --time=36:00:00
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
fi
rustc --version || true
cargo --version || true

##################################
# 1) Clean venv on node-local FS #
##################################
# (Avoid user-site pollution)
export PYTHONNOUSERSITE=1

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
python -V

# Pip knobs suitable for CC
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
# 2) Quick fixups to your requirements file  #
##############################################
REQ_FILE="$SLURM_SUBMIT_DIR/src/requirements.txt"

# If repo pins packaging==25.0, relax to <25 (resolves resolver conflicts)
if grep -Eq '(^|\s)packaging==25\.0(\s|$)' "$REQ_FILE"; then
  echo "Patching packaging pin (25.0 -> <25) in $REQ_FILE"
  sed -E -i 's/packaging==25\.0/packaging<25/g' "$REQ_FILE"
fi

# (Optional) If your repo explicitly pins older langgraph/langchain-core, bump them.
# These sed lines are safe no-ops if not present.
sed -E -i 's/^langgraph[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langgraph>=0.5.4/' "$REQ_FILE" || true
sed -E -i 's/^langchain-core[[:space:]]*==[0-9]+\.[0-9]+\.[0-9]+/langchain-core>=0.3,<0.4/' "$REQ_FILE" || true

#####################################
# 3) Compatibility guard preinstalls #
#####################################
# Install guard versions first so the resolver sticks to them.
python -m pip install \
  "packaging<25" \
  "langchain-core>=0.3,<0.4" \
  "langgraph>=0.5.4"

############################################
# 4) Optional extra constraints you want   #
############################################
cat > "$SLURM_TMPDIR/constraints.txt" <<'TXT'
# Example pin kept from your script; add more if you like.
pandas==2.2.3
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
python - <<'PY'
import importlib
mods = ["langgraph", "langchain_core", "packaging"]
for m in mods:
    mod = importlib.import_module(m)
    spec = mod.__spec__
    ver = getattr(mod, "__version__", getattr(mod, "version", None))
    locs = list(getattr(spec, "submodule_search_locations", []) or [])
    print(f"{m}: version={ver} loader={type(spec.loader).__name__} "
          f"file={getattr(mod, '__file__', None)} paths={locs}")

# Soft warning (do not raise) if langgraph is a namespace package
lg = importlib.import_module("langgraph")
if lg.__spec__ and lg.__spec__.loader.__class__.__name__ == "NamespaceLoader":
    print("WARNING: 'langgraph' is importing as a namespace package "
          "(likely a local 'langgraph/' dir on sys.path). "
          "This breaks introspection. Rename the local folder.")
PY


################################
# 8) Run your actual workload  #
################################
echo "Step 1: Running Job..." | tee "logs/job_${SLURM_JOB_ID}.out"
srun --export=ALL python -m src.cli.run_all \
  --methods agent agent2 agent3 new_bypass2 new_bypass3 bypass7 \
  --mode clone \
  --model-name local:meta-llama/Llama-3.1-8B \
  --results-filename 2025_09_10_results_llama_8_sample.csv
