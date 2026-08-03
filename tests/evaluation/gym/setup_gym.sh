#!/usr/bin/env bash
# Register this adapter's Gym servers with a NeMo Gym checkout (symlinks only —
# the code stays in tests/evaluation/). Run once after installing NeMo Gym.
#
#   GYM_DIR=/path/to/Gym bash tests/evaluation/gym/setup_gym.sh
set -euo pipefail

GYM="${GYM_DIR:-$HOME/Gym}"
ADAPTER="$(cd "$(dirname "$0")" && pwd)"

[ -x "$GYM/.venv/bin/gym" ] || { echo "NeMo Gym venv missing — run 'uv venv && uv sync --extra dev' in $GYM"; exit 1; }

# Agent harness (responses_api_agent) and verifier (resources server).
ln -sfn "$ADAPTER/responses_api_agents/challenger_agent" "$GYM/responses_api_agents/challenger_agent"
ln -sfn "$ADAPTER/resources_servers/challenger_judge"    "$GYM/resources_servers/challenger_judge"

# Environment config. Generated (not symlinked) so the committed config.yaml stays
# portable: the __ADAPTER_DIR__ sentinel is resolved to this checkout's absolute path.
mkdir -p "$GYM/environments/retail_challenger"
# rm first: the target may be a stale symlink into the adapter — a bare '>' would
# write THROUGH it and clobber the source config.
rm -f "$GYM/environments/retail_challenger/config.yaml"
sed "s#__ADAPTER_DIR__#$ADAPTER#g" "$ADAPTER/config.yaml" \
  > "$GYM/environments/retail_challenger/config.yaml"

# Reuse Gym's venv for the repo-owned servers (skip_venv_if_present).
ln -sfn "$GYM/.venv" "$ADAPTER/responses_api_agents/challenger_agent/.venv"
ln -sfn "$GYM/.venv" "$ADAPTER/resources_servers/challenger_judge/.venv"

echo "Registered challenger_agent + challenger_judge with NeMo Gym at $GYM"
