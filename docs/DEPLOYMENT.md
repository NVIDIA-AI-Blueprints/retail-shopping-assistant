# 🚀 Deployment Guide

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Fresh Deployment](#-fresh-deployment)
- [Deployment Options](#%EF%B8%8F-deployment-options)
- [Local Deployment](#-local-deployment)
- [Cloud Deployment](#%EF%B8%8F-cloud-deployment)
- [Production Deployment](#-production-deployment)
- [Configuration](#%EF%B8%8F-configuration)
- [Monitoring](#-monitoring)
- [Troubleshooting](#%EF%B8%8F-troubleshooting)

## 🎯 Overview

This guide covers deploying the Retail Shopping Assistant. Model routing lives
in one file, `shared/configs/models.yaml`. Each model role can independently
use an external endpoint, a local NIM container, or be disabled.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 20.04+ or equivalent Linux distribution
- **CPU**: 8+ cores
- **RAM**: 32GB system memory
- **Storage**: 50GB available disk space
- **Network**: Stable internet connection

#### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 16+ cores
- **RAM**: 128GB+ system memory
- **Storage**: 100GB+ available disk space
- **GPUs**: 4x H100 (for local NIM deployment)
- **Network**: High-speed internet connection

### Software Dependencies

#### Required Software
- **Docker**: Version 20.10+ with Docker Compose plugin
- **NVIDIA Container Toolkit**: For GPU acceleration
- **NVIDIA Drivers**: Latest compatible drivers
- **Git**: For repository cloning
- **Python deploy helper dependencies**: From the cloned repo, install on the
  host with the same Python interpreter used to run `scripts/model_config.py`:
  ```bash
  python -m pip install --user -r requirements-deploy.txt
  ```

#### Optional Software
- **Kubernetes**: For production orchestration
- **Helm**: For Kubernetes deployments
- **Prometheus**: For monitoring
- **Grafana**: For visualization

### NVIDIA Account Setup

1. **Create NVIDIA Account**:
   - Visit [NVIDIA NGC](https://ngc.nvidia.com/)
   - Sign up for a free account

2. **Generate API Key**:
   - Navigate to **API Keys** in your account settings
   - Generate a new API key
   - Copy the key (starts with `nvapi-`)

3. **Accept Terms**:
   - Accept the terms of service for required NIM containers
   - Ensure you have access to the NVIDIA Container Registry

## 🚀 Fresh Deployment

This is the shortest path for a new environment with hosted NVIDIA endpoints.

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

docker login nvcr.io
# Username: $oauthtoken
# Password: your NVIDIA API key

python -m pip install --user -r requirements-deploy.txt

cp .env.example .env
$EDITOR .env
source .env

python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

Open `http://localhost:3000`.

The deploy helper resolves models from `shared/configs/models.yaml`, starts
only local NIM containers referenced by roles with `source: local_nim`, and then
starts the app stack from `docker-compose.yaml`.

The env file is a sourceable shell profile. Source the profile you want before
validation or deployment; `COMPOSE_DISABLE_ENV_FILE=1` keeps Docker Compose
from auto-parsing repo-root `.env` as dotenv and mixing environments.

For an existing deployment, restart the catalog service when changing the
text/image embedding model or catalog data source. Its fingerprint reuses only
matching complete collections and rebuilds mismatches. After the catalog is
healthy, restart the chain server so its process-lifetime cached capability
contract matches the active catalog.

## 🎛️ Deployment Options

Model routing is per role:

| `source` | Meaning | Local NIMs started |
|----------|---------|--------------------|
| `endpoint` | Use the role's `base_url`/`model` or env overrides | none |
| `local_nim` | Start and use the referenced local NIM service | that service only |
| `disabled` | Capability is intentionally unavailable | none |

Use `shared/configs/models.yaml` to choose the source for each role. Copy
`.env.example` to a private env profile such as `.env`, `.env.hosted`, or
`.env.local-nim`, edit it, then `source` the profile before running validation,
deployment, or raw Docker Compose commands.

Set `CATALOG_IMAGE_EMBEDDING_ENABLED=false` in the sourced profile when a
deployment should skip image embedding clients and image collection population.
Text retrieval remains enabled.

## 🏠 Local Deployment

Use this only when this machine will run local NIM containers.

### Step 1: Environment Setup

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

cp .env.example .env.local-nim
$EDITOR .env.local-nim
source .env.local-nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod a+w "$LOCAL_NIM_CACHE"
```

Then edit `shared/configs/models.yaml` and set each local role to
`source: local_nim` with the matching `local_service`.

### Step 2: Verify GPU Setup

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Step 3: Authenticate with NVIDIA Registry

```bash
# Login to NVIDIA Container Registry
docker login nvcr.io

# Username: oauthtoken
# Password: your_nvapi_key_here
```

### Step 4: Validate and Deploy

```bash
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
docker compose -f docker-compose.yaml logs -f
```

The helper starts only the NIM services referenced by roles with
`source: local_nim`, then starts the application services.

### Step 5: Verify Deployment

```bash
docker compose -f docker-compose.yaml ps
docker compose -f docker-compose-nim-local.yaml ps
curl http://localhost:8009/health
curl http://localhost:3000
```

## ☁️ Cloud Deployment

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

# Authenticate with NVIDIA Container Registry
docker login nvcr.io
# Use oauthtoken as the username and your NGC API key as the password

# Create and source an environment profile for hosted endpoints
cp .env.example .env.hosted
$EDITOR .env.hosted
source .env.hosted
```

### Step 2: Validate Model Routing

```bash
python scripts/model_config.py show --validate
```

### Step 3: Deploy Application

```bash
# Start application services only
python scripts/model_config.py deploy --build

# Monitor startup
docker compose -f docker-compose.yaml logs -f
```

### Step 4: Verify Deployment

```bash
# Check service status
docker compose -f docker-compose.yaml ps
```

## 🏭 Production Deployment

### Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.24+)
- Helm (3.0+)
- NVIDIA GPU Operator installed
- Ingress controller configured

#### Step 1: Create Namespace

```bash
kubectl create namespace retail-assistant
kubectl config set-context --current --namespace=retail-assistant
```

#### Step 2: Create ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: retail-assistant-config
data:
  config.yaml: |
    llm_port: "https://api.nvcf.nvidia.com/v1/chat/completions"
    llm_name: "meta/llama-3.1-70b-instruct"
    retriever_port: "https://api.nvcf.nvidia.com/v1/embeddings"
    memory_port: "http://memory-retriever:8011"
    rails_port: "https://api.nvcf.nvidia.com/v1/chat/completions"
    memory_length: 16384
    top_k_retrieve: 4
    deepagents_recursion_limit: 24
    max_catalog_searches_per_turn: 3
    max_product_detail_reads_per_turn: 2
    multimodal: true
```

#### Step 3: Create Secret

```bash
kubectl create secret generic nvidia-api-keys \
  --from-literal=ngc-api-key=your_nvapi_key_here \
  --from-literal=llm-api-key=your_nvapi_key_here \
  --from-literal=embed-api-key=your_nvapi_key_here \
  --from-literal=rail-api-key=your_nvapi_key_here
```

#### Step 4: Deploy with Helm

```bash
# Add Helm repository (if using a chart)
helm repo add retail-assistant https://charts.example.com
helm repo update

# Deploy the application
helm install retail-assistant retail-assistant/retail-assistant \
  --namespace retail-assistant \
  --set nvidiaApiKey=your_nvapi_key_here
```

### Docker Swarm Deployment

#### Step 1: Initialize Swarm

```bash
docker swarm init
```

#### Step 2: Create Secrets

```bash
echo "your_nvapi_key_here" | docker secret create ngc-api-key -
echo "your_nvapi_key_here" | docker secret create llm-api-key -
echo "your_nvapi_key_here" | docker secret create embed-api-key -
echo "your_nvapi_key_here" | docker secret create rail-api-key -
```

#### Step 3: Deploy Stack

```bash
docker stack deploy -c docker-compose.prod.yaml retail-assistant
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NGC_API_KEY` | NVIDIA NGC API key | Yes | - |
| `LLM_API_KEY` | Language model API key | Yes | - |
| `VLM_API_KEY` | Optional VLM media perception API key; Compose falls back to `NVIDIA_API_KEY` when unset | When `vlm` uses an authenticated endpoint and `NVIDIA_API_KEY` is unset | `NVIDIA_API_KEY` |
| `EMBED_API_KEY` | Embedding model API key | Yes | - |
| `RAIL_API_KEY` | Guardrails API key | Yes | - |
| `GUARDRAILS_ENABLED` | Default chain-server guardrails setting for requests that omit `guardrails`; accepts true/false, yes/no, on/off, or 1/0 | No | `true` |
| `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` | Shared deadline for the Deep Agents graph and grounding editor before the durable turn fails cleanly | No | `45` |
| `EXPOSE_AGENT_DIAGNOSTICS` | Expose detailed agent/tool traces in query responses; enable only behind a trusted operator or evaluation surface | No | `false` |
| `CATALOG_SEARCH_TIMEOUT_SECONDS` | Optional chain-server timeout for catalog search requests | No | no timeout |
| `MAX_CATALOG_SEARCHES_PER_TURN` | Caps distinct catalog taxonomy-plus-hard-constraint scope executions in one assistant turn; a repeated scope is stopped even when semantic wording changes | No | `3` |
| `MAX_PRODUCT_DETAIL_READS_PER_TURN` | Caps Deep Agents product-detail reads in one assistant turn | No | `2` |
| `CHECKPOINT_STORE` | Deep Agents conversation checkpoint store; currently supports only `memory` | No | `memory` |
| `MEMORY_DATABASE_URL` | SQLite URL for durable raw turns and cart state; Compose supplies the named-volume path | No | Compose: `sqlite:////data/context.db` |
| `MEMORY_SQLITE_BUSY_TIMEOUT_MS` | SQLite lock wait for the single memory-service writer | No | `5000` |
| `MEMORY_TURN_ABANDON_SECONDS` | Age at which startup or the next turn start marks an unfinished `started` turn abandoned | No | `300` |
| `MEMORY_RECENT_TURNS` | Maximum prior context-eligible raw turns returned at the next durable turn start | No | `8` |
| `WEATHER_ENABLED` | Enables provider calls for the registered event-weather tool; keep false until the selected provider plan's rights are confirmed | No | `false` |
| `WEATHER_API_KEY` | Visual Crossing server-side credential, read indirectly from the variable named by chain-server weather config | When weather is enabled | empty |
| `LOCAL_NIM_CACHE` | NIM cache directory | Local only | `~/.cache/nim` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `NODE_ENV` | Node environment | No | `production` |

### Event Weather Tool

The chain server registers the provider-neutral
`get_weather_forecast_tool`, with Visual Crossing as the first adapter. It is a
read-only capability granted only by `event-context`; that modifier can be
selected only beside `outfit-styling`. No provider request runs during startup
or health checks. With the default `WEATHER_ENABLED=false`, qualified shopper
turns fail closed without a provider call or key.

The complete non-secret configuration is in
`shared/configs/chain_server/config.yaml`:

```yaml
weather:
  enabled: false
  provider: visual_crossing
  base_url: https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline
  api_key_env: WEATHER_API_KEY
  timeout_seconds: 3.0
  max_provider_attempts: 2
  max_forecast_horizon_days: 15
  max_range_days: 15
  receipt_ttl_seconds: 3600
```

`receipt_ttl_seconds` controls the short-lived reusable
`weather_forecast.v1` evidence boundary. It must be 1–21,600 seconds. The
memory-owned projection retains at most four valid scopes regardless of this
TTL.

To enable qualified shopper turns or run an explicit direct-client test, set
`WEATHER_ENABLED=true` and provide `WEATHER_API_KEY` through an ignored `.env`,
the process environment, or the deployment secret manager. Compose passes those
two variables only to `chain-server`; it does not bake a value into an image or
expose it to catalog, memory, guardrail, UI, or local-NIM services. The config
stores only the variable name, never the secret value. Enabling the client
without the named key fails closed, and no MCP server is required. The local
process runner enforces the same boundary by removing both weather variables
from memory, guardrail, catalog, and React process environments while retaining
them for the chain server.

The request-bound shopper wrapper permits one zero-argument model-visible call
attempt on an eligible turn. Without a complete effective location/date scope,
the runtime hides and execution-blocks the tool for that turn. Activation
compiles current-turn authority first; the execution model cannot supply or
repair provider arguments. A scope resolution that produces a complete scope
requires this call before accepting a prose response. For an unchanged
complete scope, only an explicit `weather_refresh=true` activation does so;
comparisons and other turns block weather. Within a
scope-valid call,
`max_provider_attempts: 2` allows one client-internal retry only
after a timeout or HTTP 5xx. HTTP 400 maps to generic
`weather_request_invalid`; other 4xx, connection, and response-validation
failures are not retried. The mandatory activation step owns only
`event_context_next_question` for event context. It is required exactly with
`event-context` and omitted otherwise. Before nested validation, an activation
without `event-context` has every event-context question, scope, refresh, and
receipt field discarded. Those ungranted controls therefore cannot mutate
weather state, expose weather, or consume an activation correction. The
activation model chooses the accepted question from
semantic conversation plus the typed current scope: `event_location` only
when destination is missing and material, `event_venue` only after destination
is established when venue or setting is missing and material, `event_date`
only after destination and any material venue are established when live weather
is enabled and material and the effective scope has no bounded date, and
`none` otherwise. For an existing scope, response contract 5 supplies a
separate, derived lane of the exact completed shopper/assistant turns named by
its component source identities. A request-local tools-disabled semantic
resolver compares that lane with the current query through one forced typed
control call.
It is neither a business tool nor a subagent, and any missing, invalid,
timed-out, or unclear result fails closed. The resolver returns only the
semantic relation and does not duplicate scope extraction. Activation alone
may then submit one atomic scope selection: it copies the revision and chooses
`retain`, `set`, or `clear` independently for location and date. Only
current-turn authority can supply `set`. A missing location/date question is durably bound to the
scope and stamped at finalization with its originating turn ID and sequence.
It persists even when the location/date authority values do not otherwise
change. Intervening product work neither answers nor repeats it. A
resolver-approved `same_subject` relation may authorize ordinary same-subject
retains, while `new_subject` clears them. Completing a pending component while
retaining its stored counterpart requires `answers_pending`, the exact opaque
pending source-turn handle at the resolver boundary, and activation setting the
component it names. That relation is pending-answer-only; a reply that also
changes or withdraws the opposite component uses `same_subject`. Runtime sends
the handle to memory only through the server-authored completion control.
Memory rechecks the exact live binding and canonical shape atomically, retaining
an existing counterpart, keeping a current-turn replacement, or rotating an
absent counterpart into a newly source-bound pending question. Preserving an otherwise unanswered pending binding during
a same-subject update requires the same exact server-owned source handle;
otherwise memory stamps the current finalized turn. A pure authority compiler
receives the validated proposal and applies the resolver only to prior-state
operations. An unavailable or unclear resolver clears every proposed retain
without erasing current-turn `set` values, rejects receipt/refresh reuse, and
blocks prior-dependent weather. Self-contained incomplete proposals use normal
missing-component handling; a complete `set`/`set` replacement may require
weather. Prior raw turns and summary prose never
become adapter authority. An explicitly shopper-stated
outdoor patio, beach, garden, rooftop, or open-air setting makes enabled live
weather material; with destination and that setting but no bounded date,
activation selects `event_date`. Skill selection, location, venue,
materiality, and intent remain model-owned semantic guidance. The dynamic enum
is typed argument consistency, not an intent router or keyword routing layer.
Only the accepted activation result authorizes that event-context follow-up;
the server does not infer one from weather configuration or missing context.
If activation repeats the exact live pending question while the resolver says
`unchanged`, the server accepts `none`, creates no scope resolution, and leaves
the original pending source binding untouched.
The same activation may bind
one currently valid `weather_receipt_id` only with
`event_context_next_question=none`, no scope update, no refresh request, and
exact equality to the effective location/date scope. Binding a receipt hides
and execution-blocks another weather call for that turn.
`weather_refresh` defaults to false and is accepted only with event context,
question `none`, no scope update, no receipt, and an unchanged complete scope.
Accepted `event_location` or `event_venue` hides and execution-blocks weather.
Never infer beach, outdoor/indoor setting, or terrain from a destination.
Missing location or date authority may also deny weather. Those weather
decisions never revoke catalog, detail, historical-product resolution,
availability, promotions, cart, or policy tools granted by the other selected
skills, and they do not close the primary skill's normal tool loop.
`event-context` contributes the weather tool additively to the selected grant
union; every business tool retains its ordinary independent validation,
budget, and synthesis behavior.

`confirmed_saved_zip` omits both location text fields during activation and
reaches the weather client only when a
deterministic gate accepts a current location-neutral statement explicitly
naming `my`/`the` usual/home area, a bare affirmative immediately after the
assistant's usual/home-area question, or a strict date-only follow-up
immediately after an accepted confirmation.
`shopper_provided_location` instead copies one bounded exact named-place,
address, or postal-code phrase from the current shopper turn into the scope.
For an abbreviation or geographically ambiguous name, `location_query` is
required: it must preserve that exact phrase as its first component and append
only one or two comma-separated region/country qualifiers. Keep
`location="NYC"` and use `location_query="NYC, NY"`; `Springfield, TX` is a
valid explicit regional assumption. It never adds an unstated ZIP or numeric
component and is omitted only when `location` is already sufficiently
qualified. Semantic equivalence remains model-owned
rather than deterministic proof and is correctable through
provider-resolution disclosure.
The adapter sends this bounded named place directly to Visual Crossing
Timeline, using `location_query` only in that qualifier-preserving form. It
does not synthesize a ZIP or use a separate geocoder. Visual Crossing's
`resolvedAddress` becomes the reversible `resolved_location` assumption shown
for shopper-provided mode.
Any current explicit
place, question, negation, uncertainty, or location-override cue rejects saved
mode; explicit destination takes precedence and disables saved-ZIP fallback.
Modal lowercase `may be` is treated as uncertainty, while calendar `May 5`
remains a valid date.

The request uses one exact ISO event date, one complete inclusive range, or
`relative_date=next_week` only when the shopper used the exact phrase `next
week`. Exact `<weekday> next week` requires a matching lowercase `weekday` and
is resolved from the turn's captured UTC date to that exact day inside the next
Monday-through-Sunday window. Bare `next week` omits `weekday` and resolves to
the full range. Missing, mismatched, mixed, negated, or superseded weekday
authority fails closed. Missing date authority never permits a placeholder
weather call. An unambiguous single-day phrase
such as `tomorrow` is resolved by the model against that same anchor into an
exact ISO date. Server UTC rather than caller/shopper local time is an explicit
current limitation. For other ambiguous or unresolved relative dates, only an
accepted `event_date` decision may authorize a clarification; the server does
not infer it from enabled weather or a missing date.

For `shopper_provided_location`, model-visible evidence includes the
provider-resolved place and deterministic rendering discloses it as the
forecast-location assumption, making any `location_query` qualification
reversible. That field is omitted in `confirmed_saved_zip` mode. Current
successful evidence has precedence. Otherwise, only one explicitly bound,
unexpired exact-scope `weather_forecast.v1` receipt can support reuse; unbound
receipts are non-evidence. Prior
durable assistant forecast summaries are replaced with a redaction marker in
both graph and grounding-editor recent discussion, while remaining stored and
exactly replayable; prior weather tool messages are excluded from prior
evidence. A receipt is promoted only from a same-ID successful call/result pair
and only during completed atomic finalization. Memory prunes expiry,
supersedes older same-scope success, and caps active receipts at four. It stores
no saved ZIP, raw provider request/response data, prepared provider endpoint
URL, key, exception, or failure. The pinned public attribution URL remains in
validated evidence.
Current successful deterministic final rendering appends one exact canonical block with every
validated daily date, condition, available low/high temperature, precipitation
probability/types, Visual Crossing attribution, and forecast-change warning.
The protected event decision renderer is selected structurally when
`event-context` is active, no current non-weather business-tool activity
occurred, and a current typed weather outcome (success or failure) or explicitly
bound valid receipt exists.
Missing-location/venue or an empty draft skips its decision editor. A separate
prior-candidate fallback uses deterministic event assembly only when the draft
is empty. A comparison with current product-resolution/detail activity remains
on ordinary grounding and uses a bound receipt silently without repeating
exact forecast facts or the prior canonical block. Other protected
weather-evidence turns with a
nonempty draft give that narrow editor only bounded shopper-authored event text
and the server-owned deterministic weather styling direction. Any attempted
current non-weather business tool keeps the response on the normal
business-evidence grounding path.
For ordinary weather-plus-business paths, grounding-editor sentences containing
weather-domain fact language or fact-shaped dates/values are removed while
ordinary grounded styling language remains. The context-only decision accepts
only exact two-key JSON containing a shopper-grounded venue quote plus one or
two distinct allowlisted adjustment codes. Malformed/extra-key output, a
missing or non-shopper quote, and unknown/duplicate/wrong-cardinality codes fall
back. The server maps valid codes to fixed phrases and deterministically
assembles exact prior names, its weather direction, only the accepted question,
and a current typed weather failure or current canonical success block. It never asks for
state, region, country, or finer location solely because the lookup failed.
Weather may inform general styling judgment but does not prove any
product-performance property or create an unstated catalog constraint.

Raw weather tool arguments and output are redacted from both operator
diagnostics and failed-turn partial graph capture. The tool-call record retains
only categorical `request_shape`, `location_source`, `provider_input`, and
`outcome`; it never includes a location, ZIP, date, resolved place, URL, body,
or exception. Receipt handling adds only a categorical lifecycle value such as
promotion prepared or bound; it never exposes the receipt ID, scope, or
evidence. Saved profile ZIP is recursively
scrubbed from diagnostic string keys and values. The final assistant summary
remains ordinary durable conversation text: memory can store and exactly
replay it.
Before any operator enables shopper traffic, confirm that the selected Visual
Crossing plan permits the intended attribution, display, storage, and sharing.
The review must cover that durable summary and forecast processing by the
downstream app model and output guardrails. The repository does not select a
plan or assert that these rights have been obtained. Review the current
[pricing and edition terms](https://www.visualcrossing.com/weather-data-pricing/)
and [service terms](https://www.visualcrossing.com/weather-service-terms/).

An optional direct provider smoke makes one invocation and at most the
configured two provider attempts; only timeout or HTTP 5xx triggers the second:

```bash
python scripts/weather_smoke.py
```

Before running it, set `WEATHER_ENABLED=true`, `WEATHER_API_KEY`, and
`WEATHER_SMOKE_LOCATION` in the private process environment. Optionally set either
`WEATHER_SMOKE_DATE` or the complete
`WEATHER_SMOKE_START_DATE`/`WEATHER_SMOKE_END_DATE` pair. The command prints
only provider/config label, request mode, window length, outcome category,
schema validity, and latency. It never prints the exact location, dates,
forecast, key, URL, provider body, or raw exception. It is not run by startup,
CI, health checks, or shopper traffic.

The direct smoke's output remains metadata-only. It does not exercise the
agent's narrow saved-ZIP confirmation gate, shopper-authored location
grounding, bare or weekday-qualified `next week` normalization, invalid-schema
attempt consumption, current-turn evidence, canonical forecast block, or
durable summary/receipt projection paths.

### Durable Conversation Turns

Compose runs one memory-service SQLite replica at
`sqlite:////data/context.db` and mounts the `memory-data` named volume at
`/data`. Its host port is bound to `127.0.0.1:8011`; sibling containers use the
private Compose network. The service has no authentication, so non-Compose
deployments must preserve an equivalent internal-only boundary. The chain
server starts a durable row before guardrail/model/tool work,
receives the rolling summary, bounded model-context-eligible
shopper/assistant turns, a separate product ledger, one current weather-planning
scope, its at-most-three exact source turns in an isolated lane used only for
subject resolution and protected styling provenance, at most four valid typed
weather receipts, and the authoritative cart,
then finalizes the row as
`completed`, `blocked`, or `failed`. An exact retry of
a finalized request replays the stored response and output without another
model turn. Blocked turns remain stored for exact replay and audit but are
excluded from both the next-turn service projection and the chain prompt
formatter.

Turn start negotiates additive memory response fields with
`response_contract=5`, interpreted as the caller's maximum supported version.
Memory returns the highest version it supports up to that maximum. New memory
defaults an unversioned caller to the exact
legacy response shape and computes that caller's bounded raw tail from sequence
zero rather than from the invisible summary watermark. New chain accepts a
missing contract marker as version 1. That v1 compatibility includes accepting
staging-era abandoned raw turns with no assistant text and filtering them
before model context; v2+ raw turns remain strictly memory-owned eligible
rows. Contract 3 remains supported with its
legacy transition and without the v4-only pending question or its source
fields; contract 4 advertises atomic scope-finalize write capability. Contract
5 adds only `current_weather_scope_source_turns`, required even when empty and
absent from contracts 1–4. Its sorted, deduplicated completed rows exactly
match the adjacent scope's unique `(turn_id, sequence)` pointers and are read
without the summary watermark or raw-tail limit. Deploy memory first and chain
second.
A new chain negotiating only v3 fails closed for an atomic scope update, while
an older chain remains usable after rollback. Fresh projection DDL also uses
database defaults for every additive non-null summary/receipt/scope column,
matching the defaults added to upgraded databases by migrations 8 through 11.
Migration 11 keeps the core scope lane strict pre-v4-readable as described
below.

`DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` is one model-stage deadline shared by the
active graph and grounding editor. The editor receives only the remaining time.
A graph timeout records `agent_timeout`, captures bounded partial graph messages,
clears unsent products and images, and finalizes the durable turn as failed. A
grounding timeout records `grounding_timeout` and also finalizes as failed;
search-only evidence uses deterministic catalog rendering, structurally
selected context-only event turns use deterministic event assembly, and other
non-search turns receive a fixed retry/cart-check response instead of the
unverified draft. The
same response rule applies to ordinary editor errors and empty or
whitespace-only output, recorded as `grounding_error`. Invalid structured
context-only event output instead falls back deterministically. The request
checkpoint is deleted only after finalization succeeds. This live deadline is separate from
`MEMORY_TURN_ABANDON_SECONDS`, which handles unfinished turns left by a crash or
process loss.

`MEMORY_DATABASE_URL` accepts SQLite URLs only. The busy timeout must be
non-negative, the abandoned-turn threshold must be positive, and
`MEMORY_RECENT_TURNS` is bounded by the service to 1–50 records.

SQLite uses WAL mode, foreign-key enforcement, and the configured busy timeout.
This is a single-writer, single-memory-service deployment boundary, not a shared
multi-replica conversation store. At memory-service startup and before each
turn start, unfinished rows older than `MEMORY_TURN_ABANDON_SECONDS` become
`abandoned`; there is no continuous expiration worker. An exact abandoned
request retry reopens the same durable turn only when it is still the latest
conversation sequence, preserving the request ID used by cart idempotency while
rotating its service-issued `attempt_id`. Older abandoned turns are superseded.
A finalize must echo the current attempt token, so a late worker cannot overwrite
a reopened attempt; the chain server replaces that stale result with a safe
superseded-attempt response. Other finalize outages preserve the grounded
response and add `memory_finalize_error`. Operators must define transcript
retention and backup policy. Deleting the `memory-data` volume deletes the
durable transcript, cart, and mutation replay records.

Stored turns contain shopper/assistant text plus bounded replay output and
ordered event envelopes. Raw uploaded media, model reasoning, and the full graph
message/tool transcript are not stored there. On finalization, ordered product
cards are also stored as a `candidate_set_presented` event and folded into a
compact product-reference projection. The deterministic resolver can recover a
unique typed reference from those same-conversation events after a chain-server
restart or on another worker; zero or multiple matches require clarification.
The projection keeps the newest complete candidate sets within a 16,384-
character serialized cap, and the serving runtime permits at most one batched
resolver call per turn.
Preferences, sentiment, active anchors, fuzzy/embedding lookup, and
cross-conversation resolution are not implemented. Catalog revisions are
recorded when supplied but are not yet used to invalidate stored evidence.

Migration 9 adds the receipt lane to the versioned conversation projection,
not to raw turns or the rolling summary. On completed finalization, a paired
successful weather call/result can be promoted atomically with response replay,
summary advancement, events, and product projection. Memory filters invalid or
expired receipts at turn start, supersedes the older receipt for the same exact
location/date scope, and enforces the four-item cap. Failure and raw provider
material are never promoted.

Migration 10 adds one current weather-planning scope to that projection. Its
location and normalized date-window components retain separate memory-stamped
source turns. Contract 3 keeps the legacy `continue`/`replace` transition.
Contract 4 resolves both components atomically with explicit
`retain`/`set`/`clear` actions, an expected scope revision, and one optional
pending `event_location` or `event_date` binding. A changed scope clears old
receipts, and a same-finalize promotion must exactly match the resulting
complete scope. This storage layer is not an event/anchor registry and contains
no venue, occasion, product, styling, or forecast facts. Serving compilation
admits only current-turn `set` authority; the provider tool has no
location/date arguments and reads the effective scope directly.

Migration 11 moves the v4 pending question and its source fields into defaulted
`current_weather_pending_json`. It extracts a complete binding written by the
pre-split v4 work, drops incomplete unsourceable pending fields, and removes
those keys from
`current_weather_scope_json`, keeping that core lane parseable by strict
pre-v4 models during rollback. The pending payload records its scope revision
and is merged only when that revision matches the core scope. A rollback-era
core mutation therefore leaves stale pending state inert.

Response contract 5 is derived from existing durable turns and scope pointers;
it adds no SQLite column or migration. A max-v4 chain receives the exact v4
shape from v5 memory. A v5 chain negotiating v4 retains the bounded raw-tail
fallback and fails closed if a referenced source is no longer present there.
The v5 lane may overlap summarized or recent turns, but it feeds only
weather-subject resolution and protected styling provenance—not general
conversation context, compaction, current-turn `set` authority, or forecast
evidence.

Memory receipt-promotion conflicts retain their exact error codes through the
chain HTTP client and trigger one finalize retry without the optional
promotion. Scope revision, resolution, and status conflicts are authority
conflicts: the runtime discards the draft/product output and terminalizes the
turn as failed without applying the disputed scope update.

Expiry filtering occurs atomically once at durable turn start. The accepted
active set is the validity snapshot for the request; no second wall-clock
expiry check runs while that request is in flight. Although the internal memory
projection retains normalized evidence, the pre-activation model sees only
receipt ID/type, shopper location/date scope, and `valid_until`. Full evidence
stays server-side until explicit activation binding makes it eligible for
grounding.

### Graph Checkpointing

`CHECKPOINT_STORE=memory` preserves the in-process development and test
behavior. Each graph thread is keyed by a collision-safe pair of conversation
ID and request ID and holds only one request's working Deep Agents state. The runtime deletes it after the
durable turn finalizes successfully. A finalize failure preserves that request's
checkpoint for diagnosis or retry instead of discarding the incomplete attempt.
Checkpoints still disappear when the chain-server process restarts and are not
shared across replicas, but they are no longer the source of shopper memory or
product-reference authorization.

`memory` is currently the only accepted value; an empty or different value
fails during chain-server initialization instead of silently falling back to
process-local state. The durable memory-service turn record, not the graph
checkpoint, supplies cross-turn continuity. A shared graph backend is therefore
not required for this request-scoped design; production durability and scale
remain bounded by the single-replica SQLite memory service.

### Store Policy Content

Operator-managed policy content lives in
`shared/configs/chain_server/store_policies.yaml`, under `SHARED_CONFIG_ROOT`
at runtime. The bundled template is disabled. Replace every
`[Operator placeholder]` title or body, then set `configured: true`; enabled
content that retains the marker fails closed with `policy_load_failed`. Restart
the chain server after changing the file because policy content is cached on
first use.

### Configuration File

Chain-server service behavior is configured in
`shared/configs/chain_server/config.yaml`. Model roles and endpoints are
configured separately in `shared/configs/models.yaml`:

```yaml
retriever_port: "http://localhost:8010"
memory_port: "http://localhost:8011"
rails_port: "http://localhost:8012"
memory_length: 16384
deepagents_recursion_limit: 24
max_catalog_searches_per_turn: 3
max_product_detail_reads_per_turn: 2
guardrails_enabled: true
```

The legacy routing and chatter prompt keys remain in that file for compatibility
paths; they do not configure the serving Deep Agents runtime.

### Updating Catalog Filter Metadata

The active runtime gets available product filters from the catalog retriever
after the catalog data is loaded. Do not maintain product categories in
`shared/configs/chain_server/config.yaml`.

The authoritative guide for this workflow is
[Catalog Schema and Filters](CATALOG_FILTERS.md). The short version is: the
JSONL sidecar declares field types and uses, while all values, ranges, coverage,
and taxonomy scopes are discovered from the ingested rows.

#### How to Update Filters

1. **Update Product Data**: Add products or values to the JSONL configured by
   `shared/configs/catalog_retriever/config.yaml`.
2. **Declare New Field Meaning**: Only for an entirely new field, add its type
   and `filter`, `semantic`, and/or `detail` uses to the adjacent schema
   sidecar. Never add enum values or category applicability rules.
3. **Restart/Reindex Catalog**: Restart the catalog retriever so it loads the
   new data and synchronizes its indexes.
4. **Verify Live Capabilities**: Wait for catalog health, then check
   `http://localhost:8010/capabilities`.
5. **Restart Chain Server**: Restart it so it drops the prior
   process-lifetime cached contract.
6. **Verify Cached Capabilities**: Check the chain-server aggregate at
   `http://localhost:8009/capabilities` before serving traffic.

#### Catalog Retriever Configuration

```yaml
# shared/configs/catalog_retriever/config.yaml
data_source: "/app/shared/data/enriched_products.jsonl"
schema_source: "/app/shared/data/enriched_products.schema.yaml"
```

The service fingerprint automatically rebuilds indexes when data, sidecar,
embedding models, image-search state, referenced local image bytes, or the
semantic template changes.

### Model Routing

Model endpoints are selected from one file: `shared/configs/models.yaml`.
Service behavior stays in each service's normal config file, while model base
URLs, model names, API-key environment variables, and local NIM service metadata
live in `models.yaml`.

Each role has a `source`:

| Source | Use case |
|--------|----------|
| `endpoint` | Hosted NVIDIA endpoint, remote NIM endpoint, or any OpenAI-compatible HTTP endpoint |
| `local_nim` | A NIM service started from `docker-compose-nim-local.yaml` by the deploy helper |
| `disabled` | Optional capability intentionally turned off for a deployment |

If `api_key_env` is set, `show --validate` requires that environment variable
to be present and the runtime sends it to the model endpoint. For local NIM
roles that do not need request-time auth, use `api_key_env: null`. Local NIM
container startup credentials are separate and are listed once under
`local_nims.required_env`.

The `vlm` role controls image/video media perception for user uploads. It uses
a hosted endpoint by default and can be set to `disabled` when media perception
should be off. Image embedding search remains controlled separately by the
`image_embedding` role and `CATALOG_IMAGE_EMBEDDING_ENABLED`.

#### Standard Deployment Flow

```bash
python -m pip install --user -r requirements-deploy.txt
cp .env.example .env
$EDITOR .env
source .env

python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

`show --validate` prints the resolved model routing without printing key values.
It fails if a required API-key variable or endpoint variable is missing.

For fully local NIMs:

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

Before running that command, edit each desired role in
`shared/configs/models.yaml` to use `source: local_nim`.

For a single remote NIM host in local app-code mode:

```bash
python skills/retail-local-runner/scripts/local_runner.py configure --nim-host http://HOST
python skills/retail-local-runner/scripts/local_runner.py start
```

The local runner writes ignored `.local-run/model-endpoints.env` with the
derived per-role base URLs. Its React development server keeps browser API
requests under the scoped `/local-api` prefix on port `3000` and proxies them
to the chain server without Create React App's package-proxy Host restriction,
so a remote browser does not also need port `8009` forwarded. Development
responses use `Cache-Control: no-store` so a browser or forwarding layer cannot
retain a bundle with an obsolete API base URL.

#### Adding or Changing Models

Edit `shared/configs/models.yaml` and update one role entry:

```yaml
models:
  app_llm:
    source: endpoint
    provider: openai_compatible
    base_url_env: LLM_BASE_URL
    model_env: LLM_MODEL
    api_key_env: LLM_API_KEY
```

For a Compose-managed local NIM, use `source: local_nim` and reference a local
service:

```yaml
models:
  image_embedding:
    source: local_nim
    provider: openai_compatible
    local_service: nvclip
    api_key_env: null
```

For VLM media perception through a hosted endpoint:

```yaml
models:
  vlm:
    source: endpoint
    provider: openai_compatible
    base_url_env: VLM_BASE_URL
    model_env: VLM_MODEL
    api_key_env: VLM_API_KEY
```

The sourceable `.env.example` sets `VLM_API_KEY` from `NVIDIA_API_KEY` by
default. Docker Compose also passes `NVIDIA_API_KEY` as the fallback for
`VLM_API_KEY` so hosted Omni media perception works with the single-key
developer setup.

For VLM media perception through the Compose-managed local Omni NIM:

```yaml
models:
  vlm:
    source: local_nim
    provider: openai_compatible
    local_service: nemotron_omni
    api_key_env: null
```

Then deploy with:

```bash
export LLM_BASE_URL=https://your-endpoint/v1
export LLM_MODEL=your-model-name
export LLM_API_KEY=...
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

For locally deployed roles, reference a `local_service` in `models.yaml`. The
deploy helper starts only those local NIM services.

### Performance Tuning

#### GPU Memory Optimization

```yaml
# In docker-compose-nim-local.yaml
environment:
  - NIM_KVCACHE_PERCENT=.5  # Adjust based on GPU memory
  - NIM_MAX_BATCH_SIZE=1    # Reduce for memory constraints
```

#### System Resource Limits

```yaml
# In docker-compose.yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

## 📊 Monitoring

### Health Checks

```bash
# Check individual services
curl http://localhost:8009/health  # Chain server
curl http://localhost:8010/health  # Catalog retriever
curl http://localhost:8011/health  # Memory retriever
curl http://localhost:8012/health  # Guardrails
curl http://localhost:3000         # UI
```

### Logging

```bash
# View application logs
docker compose -f docker-compose.yaml logs -f

# View NIM logs
docker compose -f docker-compose-nim-local.yaml logs -f

# View specific service logs
docker compose -f docker-compose.yaml logs -f chain-server
```

### Metrics Collection

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'retail-assistant'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8010', 'localhost:8011']
```

#### Grafana Dashboard

Create a Grafana dashboard with the following metrics:
- Request rate and latency
- GPU utilization
- Memory usage
- Error rates
- Response times by agent

### Alerting

Set up alerts for:
- Service health status
- High error rates
- GPU memory usage
- Response time degradation
- API key expiration

## 🛠️ Troubleshooting

### Common Issues

#### 1. NIM Container Pull Failures

**Symptoms**: Docker pull errors for nvcr.io containers

**Solutions**:
```bash
# Verify NGC API key
echo $NGC_API_KEY

# Re-authenticate
docker login nvcr.io

# Clear Docker cache
docker system prune -a

# Check network connectivity
curl -I https://nvcr.io
```

#### 2. GPU Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch sizes in config
# Edit docker-compose-nim-local.yaml
environment:
  - NIM_KVCACHE_PERCENT=.3
  - NIM_MAX_BATCH_SIZE=1

# Restart NIMs
docker compose -f docker-compose-nim-local.yaml restart
```

#### 3. Service Startup Failures

**Symptoms**: Services fail to start or crash

**Solutions**:
```bash
# Check service logs
docker compose -f docker-compose.yaml logs

# Check resource usage
docker stats

# Verify dependencies
docker compose -f docker-compose.yaml ps

# Check port conflicts
sudo netstat -tulpn | grep :8000
```

#### 4. Performance Issues

**Symptoms**: Slow response times

**Solutions**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor system resources
htop

# Check network latency (for cloud deployment)
ping api.nvcf.nvidia.com

# Optimize configuration
# Edit chain_server/app/config.yaml
top_k_retrieve: 2  # Reduce for faster responses
deepagents_recursion_limit: 24  # Raise modestly for multi-item outfit planning
max_catalog_searches_per_turn: 3  # Bound distinct taxonomy-plus-hard-constraint scopes
max_product_detail_reads_per_turn: 2  # Bound product-detail reads per turn
```

#### 5. Authentication Issues

**Symptoms**: API key errors

**Solutions**:
```bash
# Verify API key format
echo $NGC_API_KEY | head -c 10

# Check key permissions
# Ensure key has access to required NIMs

# Test API key
curl -H "Authorization: Bearer $NGC_API_KEY" \
  https://api.nvcf.nvidia.com/v1/models
```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Restart services
docker compose -f docker-compose.yaml restart

# View debug logs
docker compose -f docker-compose.yaml logs -f
```

### Recovery Procedures

#### Service Recovery

```bash
# Restart specific service
docker compose -f docker-compose.yaml restart chain-server

# Restart all services
docker compose -f docker-compose.yaml restart

# Rebuild and restart
docker compose -f docker-compose.yaml up -d --build
```

#### Data Recovery

```bash
# Backup volumes
docker run --rm -v retail-shopping-assistant_milvus_data:/data \
  -v $(pwd):/backup alpine tar czf /backup/milvus_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v retail-shopping-assistant_milvus_data:/data \
  -v $(pwd):/backup alpine tar xzf /backup/milvus_backup.tar.gz -C /data

# Back up memory-service SQLite while its writer is stopped
docker compose stop memory-retriever
docker run --rm -v retail-shopping-assistant_memory-data:/data \
  -v $(pwd):/backup alpine tar czf /backup/memory-data_backup.tar.gz -C /data .
docker compose start memory-retriever

# Restore memory-service SQLite while its writer is stopped
docker compose stop memory-retriever
docker run --rm -v retail-shopping-assistant_memory-data:/data \
  -v $(pwd):/backup alpine tar xzf /backup/memory-data_backup.tar.gz -C /data
docker compose start memory-retriever
```

`docker compose down -v` removes `memory-data`; back it up first when durable
turns or carts must survive teardown.

## 🔒 Security Considerations

### Network Security

- Use HTTPS in production
- Implement API authentication
- Configure firewall rules
- Use VPN for remote access

### Data Security

- Encrypt sensitive data at rest
- Use secure API keys
- Implement access controls
- Treat durable shopper/assistant turns and replay diagnostics as customer data;
  restrict database and backup access and define retention/deletion policy
- Regular security updates

### Container Security

- Scan images for vulnerabilities
- Use non-root users
- Implement resource limits
- Regular image updates

## 📈 Scaling

### Horizontal Scaling

The request-scoped MemorySaver does not carry shopper memory between turns, so
it does not itself block additional chain-server workers. Each in-flight request
still completes on one worker. The remaining durable-state limit is the memory
service's single local SQLite writer. Replace it with a validated shared/
multi-writer store before increasing memory-service replicas.

The following scaling example is future-only. Do not apply it as a complete
production topology until shared durable memory, server-owned identity, and
traffic testing are in place.

```yaml
# In docker-compose.yaml
deploy:
  replicas: 3
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

### Load Balancing

The bundled UI sends uploaded media as base64 JSON. Keep any reverse proxy
request-body limit aligned with `media_input.max_video_bytes` after base64
expansion. With the default 50 MiB raw video cap, `nginx.conf` uses
`client_max_body_size 80m`. Keep API proxy read/send timeouts high enough for
media analysis and retrieval; the bundled `nginx.conf` uses 300 seconds.

```yaml
# nginx.conf
upstream retail_assistant {
    server chain-server:8000;
    server chain-server:8001;
    server chain-server:8002;
}
```

### Auto-scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: retail-assistant-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: retail-assistant
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

For more information, see the [main README](../README.md) or [API documentation](API.md). 
