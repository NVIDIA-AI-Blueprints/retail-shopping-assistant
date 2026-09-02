#!/usr/bin/env bash
#
# The observability stack for the locally deployed LLM NIM.
#
# Only meaningful when the app LLM is a local NIM (docker-compose-nim-local.yaml).
# A hosted endpoint publishes no metrics, so there would be nothing to scrape.
#
# Stack is Prometheus + Grafana running vLLM's official dashboard, adapted from
#   https://github.com/vllm-project/vllm/blob/v0.17.1/examples/online_serving/prometheus_grafana/
#
# This script manages the stack. To generate load or take measurements, see
# ../benchmarks/bench.sh -- kept separate because standing up a scraper and
# driving traffic through the system are different jobs, and conflating them
# made it easy to benchmark a deployment nobody had verified.
#
# Usage: ./dashboard.sh {up|down|status|urls|logs}

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The NIM as seen from this host. Prometheus itself reaches it as nemotron:8000
# over the compose network; these are only for this script's health checks.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROM_PORT="${PROM_PORT:-9090}"
# 3000 is deliberately avoided: the app's nginx front end claims it, and 6006 is
# Phoenix.
GRAFANA_PORT="${GRAFANA_PORT:-6005}"
PY="${PYTHON:-python3}"

# docker-compose.yaml reads GRAFANA_PORT too, so both sides stay in agreement
# whether the port is overridden or left at the default.
export GRAFANA_PORT

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'
BOLD=$'\033[1m'; RST=$'\033[0m'
log()  { printf '%s[%s]%s %s\n' "$BLU" "$(date +%H:%M:%S)" "$RST" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '%s warn%s %s\n' "$YLW" "$RST" "$*"; }
die()  { printf '%sfail%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

compose() { docker compose -f "$HERE/docker-compose.yaml" "$@"; }

stage_up() {
  curl -sf --max-time 5 "http://$VLLM_HOST:$VLLM_PORT/health" >/dev/null \
    || die "vLLM is not responding on $VLLM_HOST:$VLLM_PORT - start it first"
  ok "vLLM endpoint healthy"

  # Confirm the server actually exposes /metrics before standing up a scraper.
  curl -sf --max-time 5 "http://$VLLM_HOST:$VLLM_PORT/metrics" \
    | grep -q '^vllm:' || die "/metrics is not exposing vllm: series"
  ok "/metrics exposes vllm: series"

  log "Starting Prometheus + Grafana"
  compose up -d

  log "Waiting for Prometheus to report the vllm target as up"
  local waited=0 state=""
  while (( waited < 120 )); do
    state=$(curl -sf --max-time 3 \
      "http://127.0.0.1:$PROM_PORT/api/v1/targets?state=active" 2>/dev/null \
      | "$PY" -c '
import json, sys
try:
    t = json.load(sys.stdin)["data"]["activeTargets"]
except Exception:
    sys.exit()
for x in t:
    if x["labels"].get("job") == "vllm":
        print(x["health"]); break
' 2>/dev/null || true)
    [[ "$state" == "up" ]] && break
    sleep 3; waited=$((waited + 3))
  done
  [[ "$state" == "up" ]] \
    && ok "Prometheus is scraping vllm (target healthy after ${waited}s)" \
    || warn "target state is '${state:-unknown}' after ${waited}s - check '$0 logs'"

  log "Waiting for Grafana to provision the dashboard"
  waited=0
  local uid=""
  while (( waited < 120 )); do
    uid=$(curl -sf --max-time 3 \
      "http://127.0.0.1:$GRAFANA_PORT/api/search?query=vLLM&type=dash-db" 2>/dev/null \
      | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit()
if d: print(d[0]["uid"])
' 2>/dev/null || true)
    [[ -n "$uid" ]] && break
    sleep 3; waited=$((waited + 3))
  done
  [[ -n "$uid" ]] \
    && ok "dashboard provisioned (uid $uid)" \
    || warn "dashboard not visible yet - check '$0 logs'"

  printf '\n'
  stage_urls

  # An idle dashboard looks exactly like a broken one, so say so here rather
  # than letting someone conclude the stack is broken when it is merely bored.
  printf '\n%sNo traffic yet, so every panel reads zero.%s Drive some:\n' "$BOLD" "$RST"
  printf '  ../benchmarks/bench.sh load 8 32\n'
}

stage_down() {
  log "Stopping monitoring stack"
  compose down
  ok "stopped (metric history is kept in the docker volumes)"
}

stage_urls() {
  local dash
  dash=$(curl -sf --max-time 3 \
    "http://127.0.0.1:$GRAFANA_PORT/api/search?query=vLLM&type=dash-db" 2>/dev/null \
    | "$PY" -c 'import json,sys
d=json.load(sys.stdin)
print(d[0]["url"] if d else "")' 2>/dev/null || true)

  printf '%sDashboard%s\n' "$BOLD" "$RST"
  printf '  Grafana     http://localhost:%s%s\n' "$GRAFANA_PORT" "${dash:-}"
  printf '  Prometheus  http://localhost:%s\n' "$PROM_PORT"
  printf '  raw metrics http://%s:%s/metrics\n' "$VLLM_HOST" "$VLLM_PORT"
  printf '\n%sBoth are bound to loopback. To reach them from your laptop:%s\n' "$BOLD" "$RST"
  printf '  ssh -N -L %s:localhost:%s -L %s:localhost:%s %s@<this-host>\n' \
    "$GRAFANA_PORT" "$GRAFANA_PORT" "$PROM_PORT" "$PROM_PORT" "$(whoami)"
  printf '  then open http://localhost:%s\n' "$GRAFANA_PORT"
}

stage_status() {
  compose ps
  printf '\n'
  if curl -sf --max-time 3 "http://127.0.0.1:$PROM_PORT/-/healthy" >/dev/null 2>&1; then
    ok "Prometheus healthy"
    # All three targets are listed, not just vllm. A missing exporter does not
    # error later -- it shows up as a panel reading zero, which is
    # indistinguishable from a real zero, so it has to be caught here.
    curl -sf "http://127.0.0.1:$PROM_PORT/api/v1/targets?state=active" \
      | "$PY" -c '
import json, sys
for t in json.load(sys.stdin)["data"]["activeTargets"]:
    print(f"     target {t[\"labels\"].get(\"job\")}: {t[\"health\"]}"
          f"  last scrape {t.get(\"lastScrape\",\"?\")[:19]}")
    if t.get("lastError"): print("       error:", t["lastError"])
' 2>/dev/null || true
  else
    warn "Prometheus not responding on $PROM_PORT"
  fi
  curl -sf --max-time 3 "http://127.0.0.1:$GRAFANA_PORT/api/health" >/dev/null 2>&1 \
    && ok "Grafana healthy" || warn "Grafana not responding on $GRAFANA_PORT"
}

stage_logs() { compose logs --tail "${1:-40}"; }

case "${1:-up}" in
  up)     stage_up ;;
  down)   stage_down ;;
  status) stage_status ;;
  urls)   stage_urls ;;
  logs)   shift || true; stage_logs "$@" ;;
  top|load|sweep)
    die "'$1' moved to ../benchmarks/bench.sh

  This script manages the monitoring stack. Load generation and measurement
  live next door:
    ../benchmarks/bench.sh $1 ${2:-}" ;;
  *)      die "usage: $0 {up|down|status|urls|logs}

  up             start Prometheus + Grafana + exporters, wait until scraping
  down           stop the stack (history preserved in volumes)
  status         container, target and scrape health for all three jobs
  urls           dashboard URLs and the SSH tunnel command
  logs [n]       container logs

  To generate load or take measurements, see ../benchmarks/bench.sh" ;;
esac
