#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Serve Nemotron 3.5 Super with vLLM for Notebook 5, on the GPU machine.
#
#   export HF_TOKEN=hf_...
#   notebook/gpu/deploy.sh                  # prefix caching on
#   PREFIX_CACHE=0 notebook/gpu/deploy.sh   # off, for Notebook 5's Step 5
#
# Defaults are 4x H100 80 GB, BF16, TP=4. Every setting is an environment
# variable; README.md in this folder says which to change for another machine.
#
# Binds 127.0.0.1 only and requires an API key, generated once into KEYFILE.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Super-VL-09212026}"
# Pinned: stock vLLM does not load this architecture.
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e}"

PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron35}"
SERVED_NAME="${SERVED_NAME:-nemotron-3.5-super}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
KEYFILE="${KEYFILE:-$HOME/.nemotron_api_key}"

GPUS="${GPUS:-all}"                    # all | 0,1,2,3
TP="${TP:-4}"                          # tensor parallel = GPUs used
MIN_TOTAL_GB="${MIN_TOTAL_GB:-280}"    # BF16 weights are ~240 GB, plus cache
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Notebook 5 goes up to 128 conversations at once; fewer seqs queues them.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
# The trace's longest call is ~27k prompt + ~1.6k reply tokens. A shorter
# window leaves more memory for the prefix cache.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

PREFIX_CACHE="${PREFIX_CACHE:-1}"      # 0 turns prefix caching off
MAMBA_PREFIX="${MAMBA_PREFIX:-1}"      # 0 caches attention only
BLOCK_SIZE="${BLOCK_SIZE:-64}"
EXPOSE="${EXPOSE:-loopback}"           # loopback | lan

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
: "${HF_TOKEN:?export HF_TOKEN=hf_... (needs access to the nvidia org)}"
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }

if [[ "$GPUS" == "all" ]]; then
  QUERY=(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  DOCKER_GPUS="all"
else
  QUERY=(nvidia-smi --id="$GPUS" --query-gpu=memory.total --format=csv,noheader,nounits)
  DOCKER_GPUS="\"device=${GPUS}\""
fi
GPU_COUNT=$("${QUERY[@]}" | wc -l)
TOTAL_GB=$(( $("${QUERY[@]}" | awk '{s+=$1} END {print s}') / 1024 ))
echo "==> ${GPU_COUNT} GPU(s) (${GPUS}), ${TOTAL_GB} GB, TP=${TP}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

(( GPU_COUNT >= TP )) || { echo "ERROR: TP=${TP} needs ${TP} GPUs, found ${GPU_COUNT}."; exit 1; }
if (( TOTAL_GB < MIN_TOTAL_GB )); then
  echo "ERROR: ${TOTAL_GB} GB is below MIN_TOTAL_GB=${MIN_TOTAL_GB}. This checkpoint is"
  echo "       BF16 (~240 GB); stopping before the download. See README.md."
  exit 1
fi

case "$EXPOSE" in
  loopback) BIND_HOST="127.0.0.1" ;;
  lan)      BIND_HOST="0.0.0.0" ;;
  *) echo "ERROR: EXPOSE must be loopback | lan"; exit 1 ;;
esac

if [[ ! -f "$KEYFILE" ]]; then
  (umask 077 && openssl rand -hex 32 > "$KEYFILE")
  echo "==> generated API key at ${KEYFILE}"
fi
chmod 600 "$KEYFILE"
API_KEY="$(cat "$KEYFILE")"
mkdir -p "$HF_CACHE"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ARGS=(
  --host "$BIND_HOST"
  --port "$PORT"
  --api-key "$API_KEY"

  --tensor-parallel-size "$TP"
  --trust-remote-code
  --async-scheduling
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"

  # Doc guidance: float32 SSM state for BF16 and FP8 checkpoints.
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float32

  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3
  --chat-template-content-format string
)

if [[ "$PREFIX_CACHE" == "1" ]]; then
  ARGS+=(
    --enable-prefix-caching
    --block-size "$BLOCK_SIZE"
    --prefix-match-unit "$BLOCK_SIZE"
  )
  # Without these the Mamba state is recomputed on every prefix hit, and the
  # hit-rate counter still climbs. verify_prefix_cache.sh measures TTFT.
  if [[ "$MAMBA_PREFIX" == "1" ]]; then
    ARGS+=( --enable-mamba-shared-prefix-checkpoint --mamba-cache-mode align )
  fi
else
  ARGS+=( --no-enable-prefix-caching )
fi

if (( TP >= 8 )); then
  ARGS+=( --enable-expert-parallel )
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "==> pulling ${VLLM_IMAGE}"
docker pull "$VLLM_IMAGE"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> starting ${CONTAINER_NAME} on ${BIND_HOST}:${PORT}  prefix cache=${PREFIX_CACHE} mamba=${MAMBA_PREFIX} block=${BLOCK_SIZE}"
docker run -d --name "$CONTAINER_NAME" --init --restart unless-stopped \
  --gpus "$DOCKER_GPUS" \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e VLLM_USE_FASTOKENS=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  "${ARGS[@]}" >/dev/null

echo "==> waiting for the server. The first run downloads ~240 GB quietly; later"
echo "    starts load from ${HF_CACHE} in minutes. Logs: docker logs -f ${CONTAINER_NAME}"
for _ in $(seq 1 240); do
  if curl -sf -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "==> ready: http://127.0.0.1:${PORT}  model ${SERVED_NAME}"
    echo "    next: notebook/gpu/verify_prefix_cache.sh, then Notebook 5"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "==> container exited. Last 60 lines:"
    docker logs --tail 60 "$CONTAINER_NAME" 2>&1 || true
    echo
    echo "    Rejected --enable-mamba-shared-prefix-checkpoint? Retry with MAMBA_PREFIX=0"
    echo "    and record that Mamba reuse was off. Out of memory? See README.md."
    exit 1
  fi
  sleep 15
done
echo "==> not ready after 60 minutes: docker logs ${CONTAINER_NAME}"
exit 1
