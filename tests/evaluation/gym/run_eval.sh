#!/usr/bin/env bash
# Build the Gym dataset from tests/evaluation datasets, then run the Challenger
# (agent harness) + Judge (verifier) under NeMo Gym.
#
#   GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh [--limit N]
#
# Env knobs: DATASET (default text_shopping), JUDGE_ENABLED (default false),
#   CONCURRENCY (default 3 — parallel scenarios; mind hosted-model rate limits),
#   RETAIL_ASSISTANT_URL (default http://localhost:8009).
#
# Prereqs: assistant running (staging/VLM build) — start it with
#   EXPOSE_AGENT_DIAGNOSTICS=true so per-turn tool/skill diagnostics are recorded;
#   NeMo Gym installed + setup_gym.sh done; .env with CHALLENGER_MODEL_* (and
#   JUDGE_MODEL_* if judging).
set -euo pipefail

GYM="${GYM_DIR:-$HOME/Gym}"
ADAPTER="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ADAPTER/../../.." && pwd)"

set -a; . "${ENV_FILE:-$REPO/.env}" 2>/dev/null || true; set +a

# The challenger/judge model API keys are stored under NVIDIA_API_KEY in .env
# (CHALLENGER_MODEL_API_KEY / JUDGE_MODEL_API_KEY are empty). The framework reads
# them directly, so fall back to NVIDIA_API_KEY when unset.
: "${CHALLENGER_MODEL_API_KEY:=${NVIDIA_API_KEY:-}}"; export CHALLENGER_MODEL_API_KEY
: "${JUDGE_MODEL_API_KEY:=${NVIDIA_API_KEY:-}}"; export JUDGE_MODEL_API_KEY

# Which dataset to run (text_shopping | visual_uploads | style_guide | image_shopping)
DATASET="${DATASET:-text_shopping}"

# 1. scenarios.yaml -> Gym JSONL, then copy to the fixed path the config points at
#    (config jsonl_fpath is static -> avoids oc.env resolution pitfalls; the copy is
#    what actually selects the dataset each run).
"$GYM/.venv/bin/python" "$ADAPTER/build_dataset.py" --dataset "$DATASET"
cp "$ADAPTER/build/$DATASET.jsonl" "$ADAPTER/build/active.jsonl"

# 2. clear Gym caches so a changed dataset doesn't conflict
rm -f "$ADAPTER"/build/*_metrics*.json "$ADAPTER"/build/*_prepare.jsonl 2>/dev/null || true
rm -rf "$ADAPTER/results/preprocessed_datasets" 2>/dev/null || true

# 3. collect rollouts (Gym starts challenger_agent + challenger_judge)
cd "$GYM"
exec .venv/bin/gym eval run \
  --config environments/retail_challenger/config.yaml \
  --agent challenger_agent \
  --input "$ADAPTER/build/active.jsonl" \
  --output "$ADAPTER/results/${DATASET}_rollouts.jsonl" \
  --split validation \
  --concurrency "${CONCURRENCY:-3}" \
  +skip_venv_if_present=true \
  "$@"
