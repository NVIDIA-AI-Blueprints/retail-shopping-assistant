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
# ~24k tokens, near the trace's longest call. At ~8k, prefill on 4x H100 costs
# less than the fixed per-request overhead and the speedup never reaches 2x.
# Must stay under deploy.sh's MAX_MODEL_LEN.
PREFIX_WORDS="${PREFIX_WORDS:-24000}"

for tool in python3 bc curl; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool required (apt install $tool)"; exit 1; }
done
AUTH=(-H "Authorization: Bearer ${API_KEY}")

echo "=== 1. Flags in effect ==="
docker logs "$CONTAINER_NAME" 2>&1 \
  | grep -iE 'prefix.cach|mamba.cache|block.size|prefix.match' \
  | tail -10 | sed 's/^/  /' || echo "  (no log lines; is CONTAINER_NAME=${CONTAINER_NAME} right?)"
echo "  Want: enable_prefix_caching=True, mamba_cache_mode=align."

metrics() {
  curl -s "${AUTH[@]}" "${BASE_URL}/metrics" | grep -E '^vllm:prefix_cache_(queries|hits)_total' | sed 's/^/  /' || true
}
echo
echo "=== 2. Counters before ==="
metrics

# Request bodies go through files: a long prompt as one argument exceeds the
# kernel's 128 KB per-argument limit.
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
payload() {
  python3 - "$MODEL" "$1" "$PREFIX_WORDS" > "$2" <<'PY'
import json, sys
model, first, words = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = first + " " + " ".join(["analysis"] * words) + "\n\nSummarize in one word."
print(json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1, "temperature": 0}))
PY
}
timed() {
  local out="$WORKDIR/response" code
  code=$(curl -s "${AUTH[@]}" -H "Content-Type: application/json" -d @"$1" \
    -o "$out" -w '%{http_code} %{time_total}' "${BASE_URL}/v1/chat/completions") || code="000 0"
  if [[ "${code%% *}" != "200" ]]; then
    echo "ERROR: request failed (HTTP ${code%% *}): $(head -c 300 "$out" 2>/dev/null)" >&2
    return 1
  fi
  echo "${code#* }"
}

echo
echo "=== 3. Time for a repeated ${PREFIX_WORDS}-word prompt ==="
# The tested prompt starts with a fresh word, so neither the warmup nor an
# earlier run of this script has cached it.
payload "warmup-$RANDOM" "$WORKDIR/warmup.json"
payload "run-$(date +%s%N)" "$WORKDIR/prompt.json"
timed "$WORKDIR/warmup.json" >/dev/null || exit 1
T1=$(timed "$WORKDIR/prompt.json") || exit 1
# Best of three: one warm request alone is sometimes slowed by unrelated work.
T2=""
for _ in 1 2 3; do
  T=$(timed "$WORKDIR/prompt.json") || exit 1
  if [[ -z "$T2" ]] || (( $(echo "$T < $T2" | bc) )); then T2=$T; fi
done
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
