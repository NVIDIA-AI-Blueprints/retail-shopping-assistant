#!/usr/bin/env bash
#
# Load generation and measurement against the locally deployed LLM NIM.
#
# Only meaningful when the app LLM is a local NIM (docker-compose-nim-local.yaml).
# A hosted endpoint publishes no metrics and rate-limits sustained load, so there
# is nothing to measure and no way to measure it.
#
# This script covers the subcommands that need environment plumbing -- resolving
# the served model name, locating the vllm CLI, finding a tokenizer on disk. The
# whole-application harnesses in this directory take their own arguments and are
# run directly with python3; see ./bench.sh with no arguments for the list.
#
# For the monitoring stack that collects what these produce, see
# ../monitoring/dashboard.sh.
#
# Usage: ./bench.sh {load|sweep|top}

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The NIM as seen from this host. The load tools run here rather than in a
# container, so they reach it over the published port.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
# Only needs the standard library. The sweep additionally needs vLLM's benchmark
# CLI; see stage_sweep.
PY="${PYTHON:-python3}"

RED=$'\033[31m'; GRN=$'\033[32m'; BLU=$'\033[34m'; RST=$'\033[0m'
log() { printf '%s[%s]%s %s\n' "$BLU" "$(date +%H:%M:%S)" "$RST" "$*"; }
ok()  { printf '%s  ok%s %s\n' "$GRN" "$RST" "$*"; }
die() { printf '%sfail%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

require_nim() {
  curl -sf --max-time 5 "http://$VLLM_HOST:$VLLM_PORT/health" >/dev/null \
    || die "the NIM is not responding on $VLLM_HOST:$VLLM_PORT - start it first"
}

# Terminal view straight off /metrics. No Prometheus, no Grafana, no browser --
# useful over a bare SSH session or when docker is unavailable.
stage_top() {
  VLLM_BASE="http://$VLLM_HOST:$VLLM_PORT" "$PY" "$HERE/metrics_top.py" "$@"
}

# Drives concurrent traffic so the panels have something to show. An idle
# dashboard looks identical to a broken one.
stage_load() {
  local conc="${1:-8}" reqs="${2:-32}"
  require_nim
  log "Generating load: $reqs requests at concurrency $conc"
  VLLM_BASE="http://$VLLM_HOST:$VLLM_PORT" \
    "$PY" "$HERE/loadgen.py" "$conc" "$reqs"
}

# Concurrency sweep to find where the server stops keeping up. Load comes from
# vLLM's own benchmark; this adds the metric peaks the benchmark cannot see.
stage_sweep() {
  require_nim

  # The sweep drives load with vLLM's own benchmark rather than a homegrown
  # client, so the throughput and latency figures are the ones vLLM's
  # maintainers publish. That means the vllm CLI has to be importable here --
  # it is not needed for anything else in this directory.
  local vllm_bin="${VLLM_BIN:-}"
  if [[ -z "$vllm_bin" ]]; then
    if command -v vllm >/dev/null 2>&1; then
      vllm_bin="$(dirname "$(command -v vllm)")"
    else
      die "the 'vllm' CLI is not on PATH, and the sweep needs it for 'vllm bench serve'.

  Install it into a virtualenv (it is a large download):
    python3 -m venv ~/vllm-bench && ~/vllm-bench/bin/pip install vllm==0.17.1
    VLLM_BIN=~/vllm-bench/bin $0 sweep ${1:-}

  For a quick check without it, './bench.sh load' needs nothing extra."
    fi
  fi
  log "vllm CLI: $vllm_bin"

  local tok="${TOKENIZER:-}"
  if [[ -z "$tok" ]]; then
    # bench serve counts tokens locally, so it needs tokenizer files on disk.
    # The served name is tried first, but a NIM reports a catalog name rather
    # than a path, and resolving that name against HuggingFace fails without
    # credentials -- so fall back to any local checkpoint that has a tokenizer.
    tok=$(curl -sf "http://$VLLM_HOST:$VLLM_PORT/v1/models" \
          | "$PY" -c 'import json,sys
d=json.load(sys.stdin)["data"]
print(d[0].get("root") or d[0]["id"] if d else "")' 2>/dev/null || true)
    if [[ ! -f "$tok/tokenizer_config.json" ]]; then
      # Search the NIM cache and any local checkpoint directory. The NIM unpacks
      # tokenizer files somewhere under its cache, so this looks a few levels in.
      #
      # Only existing directories are passed to find, and the result is guarded
      # with `|| true`. Without both, one absent path makes find exit non-zero,
      # and `set -o pipefail` then kills the script here -- silently, after it
      # has already found a usable tokenizer, which is a memorably annoying way
      # to spend an afternoon.
      local search=()
      for d in "${LOCAL_NIM_CACHE:-$HOME/.cache/nim}" /data/models "$HOME/nemotron-tokenizer"; do
        [[ -d "$d" ]] && search+=("$d")
      done
      if (( ${#search[@]} )); then
        # Snapshot directories before tmp ones: the NIM leaves tokenizer files in
        # both, but its tmp copies are not guaranteed to outlive a restart.
        tok=$(find "${search[@]}" -maxdepth 5 -name tokenizer_config.json \
                -printf '%h\n' 2>/dev/null | grep -v '/tmp/' | head -1 || true)
        [[ -z "$tok" ]] && tok=$(find "${search[@]}" -maxdepth 5 \
                -name tokenizer_config.json -printf '%h\n' 2>/dev/null | head -1 || true)
      fi
    fi
  fi
  if [[ -z "$tok" ]]; then
    die "no tokenizer found, and 'bench serve' counts tokens locally so it needs one.

  Fetch just the tokenizer files (a few MB, not the weights):
    pip install huggingface_hub
    hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \\
      --include 'tokenizer*' 'config.json' --local-dir ~/nemotron-tokenizer

  Then re-run with:
    TOKENIZER=~/nemotron-tokenizer $0 sweep ${1:-}"
  fi
  log "tokenizer: $tok"

  # Sweeping at the default shape measures a workload this application does not
  # run. SWEEP_INPUT_LEN / SWEEP_OUTPUT_LEN / SWEEP_PREFIX_LEN come from
  # turn_profile.py; without them the answer can be off by a multiple.
  if [[ -z "${SWEEP_INPUT_LEN:-}" ]]; then
    printf '%s warn%s no SWEEP_INPUT_LEN set, sweeping at the generic benchmark shape.\n' \
      $'\033[33m' "$RST"
    printf '       This application reads far more than it writes, so that answers a\n'
    printf '       different question. Take the real shape from turn_profile.py first.\n'
  fi

  VLLM_BASE="http://$VLLM_HOST:$VLLM_PORT" VENV_BIN="$vllm_bin" TOKENIZER="$tok" \
    "$PY" "$HERE/saturate.py" "${1:-}"
}

case "${1:-}" in
  load)   shift || true; stage_load "$@" ;;
  sweep)  shift || true; stage_sweep "$@" ;;
  top)    shift || true; stage_top "$@" ;;
  up|down|status|urls|logs)
    die "'$1' belongs to the monitoring stack

  This script generates load and takes measurements. The Prometheus + Grafana
  stack lives next door:
    ../monitoring/dashboard.sh $1" ;;
  *)      die "usage: $0 {load|sweep|top}

  load [c] [n]   generate concurrent traffic straight at the NIM,
                 so the dashboard panels have something to show
  sweep [levels] concurrency sweep to find the saturation point,
                 e.g. 'sweep 1,8,32,128'   (default 1,4,8,16,32,64,128)
                 set SWEEP_INPUT_LEN / SWEEP_OUTPUT_LEN / SWEEP_PREFIX_LEN
                 from turn_profile.py, or it measures the wrong workload
  top [secs]     live terminal metrics view, no browser needed

  The whole-application harnesses take their own arguments -- run them directly:

    python3 benchmarks/turn_profile.py --out /tmp/profile.json
        where one turn's time goes, and the token shape the sweep needs

    python3 benchmarks/shopper_study.py --levels 1,2,4,8,16,32 --think 0
        how many concurrent shoppers, at a stated latency target

    python3 benchmarks/journey_load.py
        journeys at rising concurrency for a fixed duration per level

    python3 benchmarks/concurrent_shoppers.py --shoppers N
        do concurrent shoppers queue behind each other end to end

  Step-by-step: docs/PERFORMANCE.md" ;;
esac
