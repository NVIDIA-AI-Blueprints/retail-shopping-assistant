#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prove the prefix cache reuses work, on the attention KV blocks AND the
# Mamba state, before Notebook 5 records anything.
#
# On this hybrid model the hit-rate counter climbs even when the Mamba state
# is recomputed on every hit. Only time to first token tells the truth: the
# same long prompt twice, and the second must be much faster.
#
#   notebook/gpu/verify_prefix_cache.sh
#
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-nemotron-3.5-super}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron35}"
KEYFILE="${KEYFILE:-$HOME/.nemotron_api_key}"
API_KEY="${NEMOTRON_API_KEY:-$(cat "$KEYFILE" 2>/dev/null || echo '')}"
PREFIX_WORDS="${PREFIX_WORDS:-8000}"   # ~10k tokens, like a trace's system prompt and tools

for tool in jq bc curl; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool required (apt install $tool)"; exit 1; }
done
AUTH=(-H "Authorization: Bearer ${API_KEY}")

echo "=== 1. Flags in effect ==="
docker logs "$CONTAINER_NAME" 2>&1 \
  | grep -iE 'prefix.cach|shared.prefix|mamba.cache|block.size|prefix.match' \
  | tail -10 | sed 's/^/  /' || echo "  (no log lines; is CONTAINER_NAME=${CONTAINER_NAME} right?)"
echo "  Want: enable_prefix_caching=True, a shared-prefix-checkpoint line, mamba_cache_mode=align."

metrics() {
  curl -s "${AUTH[@]}" "${BASE_URL}/metrics" | grep -E '^vllm:prefix_cache_(queries|hits)_total' | sed 's/^/  /' || true
}
echo
echo "=== 2. Counters before ==="
metrics

words() { python3 -c "print('$1 ' + ' '.join(['analysis'] * ${PREFIX_WORDS}))"; }
payload() {
  jq -n --arg m "$MODEL" --arg p "$1" \
    '{model: $m, messages: [{role: "user", content: ($p + "\n\nSummarize in one word.")}], max_tokens: 1, temperature: 0}'
}
timed() {
  local start end
  start=$(date +%s.%N)
  curl -s "${AUTH[@]}" -H "Content-Type: application/json" -d "$(payload "$1")" \
    "${BASE_URL}/v1/chat/completions" >/dev/null
  end=$(date +%s.%N)
  echo "$end - $start" | bc
}

echo
echo "=== 3. Time for a repeated ${PREFIX_WORDS}-word prompt ==="
# The tested prompt starts with a fresh word, so neither the warmup nor an
# earlier run of this script has cached it.
timed "$(words "warmup-$RANDOM")" >/dev/null || true
PROMPT="$(words "run-$(date +%s%N)")"
T1=$(timed "$PROMPT")
T2=$(timed "$PROMPT")
SPEEDUP=$(echo "scale=1; $T1 / $T2" | bc)
echo "  cold ${T1}s   warm ${T2}s   speedup ${SPEEDUP}x"

echo
echo "=== 4. Counters after ==="
metrics

cat <<EOF

=== Reading it ===
  Speedup above ~2x, hits climbing   reuse works end to end; run Notebook 5.
  Hits climbing, speedup ~1x         attention cached, Mamba recomputed. Check
                                     step 1 shows the Mamba flags, or redeploy.
  No hits                            prefix caching is off (expected after
                                     PREFIX_CACHE=0), or the prompt is shorter
                                     than one block.
EOF
