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
  - [Model Sampling and Output Limits](#model-sampling-and-output-limits): where each model call's temperature and max tokens are set
- [Monitoring](#-monitoring)
- [Troubleshooting](#%EF%B8%8F-troubleshooting)

## 🎯 Overview

This guide covers deploying the Retail Shopping Assistant. Model routing lives
in one file, `shared/configs/models.yaml`. Each model role can independently
use an external endpoint, a locally deployed model, or be disabled. Locally
deployed models are vLLM serving Hugging Face checkpoints from
`docker-compose-model-local.yaml`.

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
- **GPUs**: 8x H100 (for locally deployed models; 5 are used by default)
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

### NVIDIA Account Setup

1. **Create NVIDIA Account**:
   - Visit [NVIDIA NGC](https://ngc.nvidia.com/)
   - Sign up for a free account

2. **Generate API Key**:
   - Navigate to **API Keys** in your account settings
   - Generate a new API key
   - Copy the key (starts with `nvapi-`)

3. **Accept Terms**:
   - Ensure you have access to the NVIDIA Container Registry
   - For local NIMs, request access to the Nemotron 3.5 Super checkpoint on
     Hugging Face and create an `HF_TOKEN`

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
only locally deployed models referenced by roles with `source: local_model`, and then
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

| `source` | Meaning | Local models started |
|----------|---------|--------------------|
| `endpoint` | Use the role's `base_url`/`model` or env overrides | none |
| `local_model` | Start and use the referenced locally deployed model | that service only |
| `disabled` | Capability is intentionally unavailable | none |

Use `shared/configs/models.yaml` to choose the source for each role. Copy
`.env.example` to a private env profile such as `.env`, `.env.hosted`, or
`.env.local-nim`, edit it, then `source` the profile before running validation,
deployment, or raw Docker Compose commands.

Image embedding is off by default: catalog indexing populates the text
collection only, and needs just the text embedding endpoint. Set
`CATALOG_IMAGE_EMBEDDING_ENABLED=true` in the sourced profile to also build
image embedding clients and populate the image collection, which additionally
requires a reachable `image_embedding` endpoint.

That default is declared in one place, `shared/configs/catalog_retriever/config.yaml`,
as `image_embedding_enabled`. Compose and the env profiles pass the environment
variable through without a default of their own, so an empty value means "not
set" and the config file decides. Guardrails works the same way, with its
default in `shared/configs/chain_server/config.yaml`.

## 🏠 Local Deployment

Use this only when this machine will serve the models itself. The local setup
mirrors the hosted default: Nemotron 3.5 Super answers the shopper and reads
photo and video uploads, and Nemotron 3 Embed 1B embeds the catalog. Both are
locally deployed models: Hugging Face checkpoints served by vLLM from
`docker-compose-model-local.yaml`:

| Service | Checkpoint | GPUs |
|---------|------------|------|
| `local-llm` | `nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026`, BF16, tensor parallel 4 | `LOCAL_LLM_GPUS`, default `0,1,2,3` |
| `local-embedding` | `nvidia/Nemotron-3-Embed-1B-BF16` | `LOCAL_EMBED_GPU`, default `4` |

The defaults fit an 8x H100 80 GB machine. `.env.local-models.example` points
the app LLM, media and text embedding at them through the environment, so
`models.yaml` does not change. Image embedding and guardrails stay hosted.

### Step 1: Environment Setup

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

cp .env.local-models.example .env.local-models
$EDITOR .env.local-models   # HF_TOKEN, with access to the Nemotron 3.5 Super checkpoint
source .env.local-models
mkdir -p "$HF_CACHE"
```

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

The vLLM images come from Docker Hub; the UI image builds from an `nvcr.io`
base image, which needs an NGC login:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: your NGC API key
```

### Step 4: Start the Models, Then the App

```bash
docker compose -f docker-compose-model-local.yaml up -d --wait local-llm local-embedding
python scripts/model_config.py show --validate
docker compose -f docker-compose.yaml up -d --build
```

`--wait` returns once both report healthy. Start the app after that: the
catalog indexer embeds the catalog once, at startup. The first start downloads
the chat model's ~240 GB into `HF_CACHE` and can take an hour; follow it with
`docker compose -f docker-compose-model-local.yaml logs -f local-llm`.

### Step 5: Index the catalog

Serving containers do not index themselves. Rebuilding an index begins by
dropping the collection, so it must happen exactly once, and a container cannot
know whether it is the only one doing it.

Under Compose this is already handled: the `catalog-indexer` service runs
`python -m app.index_catalog` once and exits, and `catalog-retriever` depends on
its successful completion, so it cannot start against a missing index. `up`
re-runs it, which covers any change to the catalog data, schema, or embedding
model.

To index out of band -- forcing a rebuild without cycling Compose, or after
editing catalog files in place -- run it directly:

```bash
docker compose exec catalog-retriever python -m app.index_catalog
```

Safe to repeat and safe to leave unconditionally in a pipeline: it checks the
catalog fingerprint first and does nothing when the index is already current.

### Step 6: Verify Deployment

Check `/ready`, not only `/health`. They answer different questions and are
meant to disagree: `/health` says the process is alive and should not be
restarted, `/ready` says it should be sent traffic. A catalog container with an
unbuilt index is alive and unable to answer.

```bash
docker compose -f docker-compose.yaml ps
docker compose -f docker-compose-model-local.yaml ps

curl http://localhost:8009/ready    # chain server
curl http://localhost:8010/ready    # catalog: 503 until the index is built
curl http://localhost:8011/ready    # memory: 503 until migrations finish
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

### Kubernetes

No Kubernetes manifests or Helm chart ship with this repository. These notes
are for whoever writes them.

#### What is already decided, so it need not be re-decided here

- **No session affinity.** A graph checkpoint is keyed on
  `(conversation_id, request_id)`, so two turns of one conversation never share
  one and nothing carries across turns inside a pod. Any chain-server replica
  can serve any turn. Verified by running a four-turn conversation, including a
  referring expression and a cart write, with every turn on a different pod.
- **Probes are distinct.** `/health` is liveness and stays on the event loop, so
  a loaded pod answers it instantly and is not killed for being busy. `/ready`
  is readiness and is allowed to say no. Neither checks a dependency: a
  readiness probe that fails on a downstream service removes every pod at once
  and turns a partial outage into a total one.
- **Draining works.** uvicorn is PID 1 in every container, so SIGTERM starts a
  drain rather than stopping at a shell. `terminationGracePeriodSeconds` must
  exceed `SHUTDOWN_GRACE_SECONDS` (160 for the chain server, which must clear a
  turn that may run to `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS`) or the kubelet
  sends SIGKILL mid-drain.
- **Streaming.** The Ingress must not buffer and must allow a whole turn:
  `proxy_buffering off` and a read timeout above the turn budget. Turns are
  server-sent events over a single long request.

#### What must be decided here

- **Catalog indexing is a Job**, `parallelism: 1`, using the catalog image with
  `command: ["python", "-m", "app.index_catalog"]`. Run it before the pods that
  read the index; they will sit unready until it succeeds, which is correct and
  needs no coordination.
- **More than one memory replica requires PostgreSQL.** SQLite is a
  single-writer file on a single-mount volume. Set `MEMORY_DATABASE_URL` and
  move existing data with `scripts/copy_memory_to_postgres.py` first.
- **`shared/` is still a bind mount.** All four services read configuration,
  data, images, and Python modules from it, and a hostPath does not exist on
  another node. It is read-only at runtime, so the fix is
  `COPY ./shared /app/shared` in the four Dockerfiles and dropping the mounts.
  Open, and deliberately deferred to this work.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NGC_API_KEY` | NVIDIA NGC API key | Yes | - |
| `LLM_API_KEY` | Language model API key | Yes | - |
| `APP_LLM_TEMPERATURE` | Temperature for the shopping agent and grounding editor; overrides `config.yaml`; see [Model Sampling and Output Limits](#model-sampling-and-output-limits) | No | `0.7` |
| `APP_LLM_FREQUENCY_PENALTY` | Optional frequency penalty for the same calls | No | off |
| `LLM_MAX_OUTPUT_TOKENS` | Shopping agent output ceiling per call; overrides `config.yaml` | No | `1024` |
| `GROUNDING_EDITOR_MAX_OUTPUT_TOKENS` | Grounding editor output ceiling per call; overrides `config.yaml` | No | `1024` |
| `VLM_BASE_URL`, `VLM_MODEL` | Media perception endpoint and model; set separately from `LLM_*` | No | Same as `app_llm` in `models.yaml` |
| `VLM_API_KEY` | Optional VLM media perception API key; Compose falls back to `NVIDIA_API_KEY` when unset | When `vlm` uses an authenticated endpoint and `NVIDIA_API_KEY` is unset | `NVIDIA_API_KEY` |
| `EMBED_API_KEY` | Embedding model API key | Yes | - |
| `RAIL_API_KEY` | Guardrails API key | Yes | - |
| `GUARDRAILS_ENABLED` | Default chain-server guardrails setting for requests that omit `guardrails`; accepts true/false, yes/no, on/off, or 1/0. Guardrails is opt-in: set this to enable it | No | `false` |
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
| `WEATHER_ENABLED` | Registers the forecast tool with the shopper agent (needs `WEATHER_API_KEY`) | No | `false` |
| `WEATHER_API_KEY` | Visual Crossing server-side credential, read indirectly from the variable named by chain-server weather config | Only when directly constructing an enabled weather client | empty |
| `HF_TOKEN` | Hugging Face token with access to the local LLM checkpoint | Local only | - |
| `HF_CACHE` | Hugging Face cache the locally deployed models download into | Local only | `~/.cache/huggingface` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `NODE_ENV` | Node environment | No | `production` |

### Weather Tool

The chain server includes a provider-neutral daily weather client and
`get_weather_forecast_tool`, with Visual Crossing as the first adapter. It is
off by default. Enabled, the tool is registered with the shopping agent and
granted only by the `destination-weather` skill; it forecasts a place the
shopper named, for dates within the 15-day horizon, at most twice per turn.
Disabled, it is not registered at all, and startup, health checks, shopper
turns, and offline tests perform no provider request and require no weather key.
It has no FastAPI route and no UI of its own.

The complete non-secret configuration is in
`shared/configs/chain_server/config.yaml`:

```yaml
weather:
  enabled: false
  provider: visual_crossing
  base_url: https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline
  api_key_env: WEATHER_API_KEY
  timeout_seconds: 3.0
  max_forecast_horizon_days: 15
  max_range_days: 15
```

To enable it, set `WEATHER_ENABLED=true` and provide
`WEATHER_API_KEY` through an ignored `.env`, the process environment, or the
deployment secret manager. Compose passes those two variables only to
`chain-server`; it does not bake a value into an image or expose it to catalog,
memory, guardrail, UI, or locally deployed model services. The config stores only the
variable name, never the secret value. Enabling the client without the named
key fails closed, and no MCP server is required. The local process runner
enforces the same boundary by removing both weather variables from memory,
guardrail, catalog, and React process environments while retaining them for the
chain server.

An optional direct provider smoke makes at most one request per invocation:

```bash
python scripts/weather_smoke.py
```

Before running it, set `WEATHER_ENABLED=true`, `WEATHER_API_KEY`, and
`WEATHER_SMOKE_ZIP` in the private process environment. Optionally set either
`WEATHER_SMOKE_DATE` or the complete
`WEATHER_SMOKE_START_DATE`/`WEATHER_SMOKE_END_DATE` pair. The command prints
only provider/config label, request mode, window length, outcome category,
schema validity, and latency. It never prints the ZIP, dates, location,
forecast, key, URL, provider body, or raw exception. It is not run by startup,
CI, health checks, or shopper traffic.

The adapter emits normalized daily forecast evidence and attribution metadata
but persists nothing. Before a later slice displays or stores this evidence,
operators must confirm the selected Visual Crossing plan's attribution,
storage, sharing, and uncertainty requirements in the
[pricing terms](https://www.visualcrossing.com/weather-data-editions/) and
[service terms](https://www.visualcrossing.com/weather-service-terms/).

### Durable Conversation Turns

Compose runs one memory-service SQLite replica at
`sqlite:////data/context.db` and mounts the `memory-data` named volume at
`/data`. Its host port is bound to `127.0.0.1:8011`; sibling containers use the
private Compose network. The service has no authentication, so non-Compose
deployments must preserve an equivalent internal-only boundary. The chain
server starts a durable row before guardrail/model/tool work,
receives bounded model-context-eligible shopper/assistant turns plus the
authoritative cart, and finalizes the row as `completed`, `blocked`, or
`failed`. An exact retry of
a finalized request replays the stored response and output without another
model turn. Blocked turns remain stored for exact replay and audit but are
excluded from both the next-turn service projection and the chain prompt
formatter.

`DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` is one model-stage deadline shared by the
active graph and grounding editor. The editor receives only the remaining time.
A graph timeout records `agent_timeout`, captures bounded partial graph messages,
clears unsent products and images, and finalizes the durable turn as failed. A
grounding timeout records `grounding_timeout` and also finalizes as failed;
search-only evidence uses deterministic catalog rendering, while other turns
receive a fixed retry/cart-check response instead of the unverified draft. The
same response rule applies to editor errors and empty or whitespace-only output,
recorded as `grounding_error`. The request checkpoint is deleted only after finalization
succeeds. This live deadline is separate from
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
guardrails_enabled: false
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
URLs, model names, API-key environment variables, and locally deployed model metadata
live in `models.yaml`.

Each role has a `source`:

| Source | Use case |
|--------|----------|
| `endpoint` | Hosted NVIDIA endpoint, remote model host, or any OpenAI-compatible HTTP endpoint |
| `local_model` | A locally deployed model started from `docker-compose-model-local.yaml` by the deploy helper |
| `disabled` | Optional capability intentionally turned off for a deployment |

If `api_key_env` is set, `show --validate` requires that environment variable
to be present and the runtime sends it to the model endpoint. For locally
deployed roles that do not need request-time auth, use `api_key_env: null`.
Local model container startup credentials are separate and are listed once under
`local_models.required_env`.

The `vlm` role controls image/video media perception for user uploads. By
default it uses the same model as `app_llm`, Nemotron 3.5 Super VL, which reads
photos and video. It is configured separately through `VLM_*`, so replacing the
app LLM leaves media perception where it is; set `VLM_*` as well to move it.
It can be set to `disabled` when media perception should be off. Image
embedding search remains controlled separately by the `image_embedding` role
and `CATALOG_IMAGE_EMBEDDING_ENABLED`.

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

For locally deployed models, source the local profile, which sets the roles' `*_BASE_URL`
and `*_MODEL` to the local services, and start them before the app, as in
[Local Deployment](#-local-deployment):

```bash
source .env.local-models
docker compose -f docker-compose-model-local.yaml up -d --wait local-llm local-embedding
python scripts/model_config.py show --validate
docker compose -f docker-compose.yaml up -d --build
```

To have `deploy` start them instead, set those roles in
`shared/configs/models.yaml` to `source: local_model` with `local_service:
local-llm` or `local-embedding`. Keep the profile sourced: a role's own
`base_url` and `model` take precedence over the service's.

For a single remote model host in local app-code mode:

```bash
python skills/retail-local-runner/scripts/local_runner.py configure --nim-host http://HOST
python skills/retail-local-runner/scripts/local_runner.py start
```

The local runner writes ignored `.local-run/model-endpoints.env` with the
derived per-role base URLs.

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

For a Compose-managed locally deployed model, use `source: local_model` and reference a
service under `local_models.services`:

```yaml
models:
  text_embedding:
    source: local_model
    provider: openai_compatible
    local_service: local-embedding
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

Then deploy with:

```bash
export LLM_BASE_URL=https://your-endpoint/v1
export LLM_MODEL=your-model-name
export LLM_API_KEY=...
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

For locally deployed roles, reference a `local_service` in `models.yaml`. The
deploy helper starts only those services.

#### Model Sampling and Output Limits

The chain server calls the app LLM in four places. Each call's temperature and
output token ceiling are set in one place, listed here:

| Call | Temperature | Max output tokens |
|------|-------------|-------------------|
| Shopping agent: picks tools, writes the reply | `llm_temperature` in `shared/configs/chain_server/config.yaml`, default `0.7`; `APP_LLM_TEMPERATURE` overrides | `llm_max_output_tokens` in the same file, default `1024`; `LLM_MAX_OUTPUT_TOKENS` overrides |
| Grounding editor: rewrites the draft to match the tools' product data | Same client as the agent: `llm_temperature` | `grounding_editor_max_output_tokens` in the same file, default `1024`; `GROUNDING_EDITOR_MAX_OUTPUT_TOKENS` overrides |
| Media perception: reads uploaded photos and videos | `0.6`, `_TEMPERATURE` in `chain_server/src/media_perception.py` | `4096`, `_MAX_TOKENS` in the same file; change `_MAX_ANALYSIS_CHARS` (`16000`) with it, as it caps the analysis passed to the agent |
| Vocabulary judge: matches shopper words to catalog values | `0`, in `chain_server/src/vocabulary_judge.py` | `1200`, `_MAX_TOKENS` in the same file |

`APP_LLM_FREQUENCY_PENALTY` optionally adds a frequency penalty to the agent
and grounding editor. It is off when unset.

To change a default so it holds for every deployment and every way of running
the chain server:

- **Agent and grounding editor:** edit `llm_temperature`,
  `llm_max_output_tokens` and `grounding_editor_max_output_tokens` in
  `shared/configs/chain_server/config.yaml`. It is the only place these
  defaults live. It is mounted into the container, so a restart of
  `chain-server` picks it up. `.env.example` and `docker-compose.yaml` only
  pass the overrides through, empty by default.

- **Media perception and the vocabulary judge:** these have no environment
  variable or config key. Edit the values in the files named above, then
  rebuild the chain server with `docker compose up -d --build chain-server`.
  Compose mounts only `shared/` into the container, so a code edit takes
  effect only after a rebuild.

To change the agent's settings for one deployment only, set the environment
variables in the env profile you source; they win over these defaults. Compose
passes all four to `chain-server`.

A reply that reaches its token ceiling is cut off mid-sentence, with no error.
Long answers, such as a detailed comparison of several products, can need more
than 1024 tokens.

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

# View locally deployed model logs
docker compose -f docker-compose-model-local.yaml logs -f

# View specific service logs
docker compose -f docker-compose.yaml logs -f chain-server
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. nvcr.io Pull Failures

**Symptoms**: Docker pull errors for nvcr.io containers

**Solutions**:
```bash
# Verify the NGC API key is set (without printing it)
test -n "$NGC_API_KEY" && echo set

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

# Pick free GPUs for the local models (LOCAL_LLM_GPUS must hold LOCAL_LLM_TP
# IDs), or lower --gpu-memory-utilization in docker-compose-model-local.yaml
LOCAL_LLM_GPUS=4,5,6,7 LOCAL_EMBED_GPU=3 \
  docker compose -f docker-compose-model-local.yaml up -d --wait local-llm local-embedding

# Or return to hosted endpoints: source .env instead of .env.local-models
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
sudo netstat -tulpn | grep -E ':(3000|8009|8010|8011|8012)'
```

#### 4. Performance Issues

**Symptoms**: Slow response times

**Solutions**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor system resources
htop

# Bound per-turn work in shared/configs/chain_server/config.yaml:
#   deepagents_recursion_limit         graph steps per turn
#   max_catalog_searches_per_turn      distinct catalog search scopes per turn
#   max_product_detail_reads_per_turn  product-detail reads per turn
```

#### 5. Authentication Issues

**Symptoms**: API key errors

**Solutions**:
```bash
# Show which key variables each role needs and whether they are set
python scripts/model_config.py show --validate
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

### Load Balancing

The bundled UI sends uploaded media as base64 JSON. Keep any reverse proxy
request-body limit aligned with `media_input.max_video_bytes` after base64
expansion. With the default 50 MiB raw video cap, `nginx.conf` uses
`client_max_body_size 80m`. Keep API proxy read/send timeouts high enough for
media analysis and retrieval; the bundled `nginx.conf` uses 300 seconds.

---

For more information, see the [main README](../README.md) or [API documentation](API.md). 
