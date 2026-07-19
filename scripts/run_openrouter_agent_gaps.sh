#!/usr/bin/env bash
# Re-run agent coverage gaps via OpenRouter for base_a, base_b, and agent.
#
# Usage:
#   python scripts/build_agent_gap_subsets.py
#   ./scripts/run_openrouter_agent_gaps.sh gpt-5-nano
#   ./scripts/run_openrouter_agent_gaps.sh llama-3.1-8b --resume
#   ./scripts/run_openrouter_agent_gaps.sh qwen3-32b --concurrency 2
#
# Env:
#   OPENROUTER_API_KEY (required; or via .env)
#   GITHUB_TOKEN (recommended)
#   CONCURRENCY / METHOD_CONCURRENCY optional overrides

set -euo pipefail

MODEL_LABEL="${1:-}"
shift || true

if [[ -z "$MODEL_LABEL" ]]; then
  echo "Usage: $0 {gpt-5-nano|llama-3.1-8b|qwen3-32b} [--resume] [--concurrency N]" >&2
  exit 2
fi

RESUME=0
CONCURRENCY="${CONCURRENCY:-4}"
METHOD_CONCURRENCY="${METHOD_CONCURRENCY:-3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --method-concurrency) METHOD_CONCURRENCY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODEL_LABEL" in
  gpt-5-nano)  MODEL_NAME="openrouter/openai/gpt-5-nano" ;;
  llama-3.1-8b) MODEL_NAME="openrouter/meta-llama/llama-3.1-8b-instruct" ;;
  qwen3-32b)   MODEL_NAME="openrouter/qwen/qwen3-32b" ;;
  *) echo "Unknown model label: $MODEL_LABEL" >&2; exit 2 ;;
esac

python scripts/build_agent_gap_subsets.py

SUBSET="data/agent_coverage_gaps/subsets/${MODEL_LABEL}_agent_gaps_needs_reprocess.csv"
if [[ ! -f "$SUBSET" ]]; then
  echo "Missing subset CSV: $SUBSET" >&2
  exit 1
fi

DATE_STAMP="$(date +%Y_%m_%d)"
RESULTS="${DATE_STAMP}_openrouter_${MODEL_LABEL}_agent_gaps_base_agent.csv"
N_SCENARIOS=$(($(wc -l < "$SUBSET") - 1))

export DATASET_CSV="$ROOT/$SUBSET"

echo "============================================================"
echo " GH-ACR OpenRouter agent-gap fill"
echo " Model label: $MODEL_LABEL"
echo " Model name:  $MODEL_NAME"
echo " Methods:     base_a base_b agent"
echo " Scenarios:   $N_SCENARIOS"
echo " Dataset:     $DATASET_CSV"
echo " Concurrency: $CONCURRENCY  method_concurrency=$METHOD_CONCURRENCY"
echo " Resume:      $RESUME"
echo " Results:     data/$RESULTS"
echo " Started:     $(date -Is)"
echo "============================================================"

ARGS=(
  -m src.cli.run_all
  --methods base_a base_b agent
  --mode clone
  --model-name "$MODEL_NAME"
  --concurrency "$CONCURRENCY"
  --method-concurrency "$METHOD_CONCURRENCY"
  --results-filename "$RESULTS"
)
if [[ "$RESUME" -eq 1 ]]; then
  ARGS+=(--resume)
fi

python "${ARGS[@]}"
