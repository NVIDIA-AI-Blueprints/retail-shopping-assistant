# 🚀 Deployment Guide

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Hosted Endpoints](#-hosted-endpoints)
- [Deployment Options](#%EF%B8%8F-deployment-options)
- [Locally Hosted Models](#-locally-hosted-models)
- [Production Deployment](#-production-deployment)
- [Configuration](#%EF%B8%8F-configuration)
  - [Model Sampling and Output Limits](#model-sampling-and-output-limits): where each model call's temperature and max tokens are set
- [Monitoring](#-monitoring)
- [Troubleshooting](#%EF%B8%8F-troubleshooting)

## 🎯 Overview

This guide covers deploying the Retail Shopping Assistant. Every model URL and
model name is set in one file, `.env.example`, which you copy and source.
`shared/configs/models.yaml` lists the model roles and the variables each one
reads. Each role can independently use an external endpoint, a locally
deployed model, or be disabled. Locally
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
- **GPUs**: 6x H100 80 GB for locally deployed models (see [GPU Sizing](#gpu-sizing))
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
  The only dependency is PyYAML. Where the system Python has no `pip` and
  refuses `--user` installs, as on Ubuntu 24.04, install it from the
  distribution instead and run the helpers with `python3`:
  ```bash
  sudo apt-get install -y python3-yaml
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
   - For locally hosted models, request access on Hugging Face to the Nemotron 3.5 Super
     checkpoint and to `meta-llama/Llama-3.1-8B-Instruct`, then create an `HF_TOKEN`

## 🚀 Hosted Endpoints

Every model role runs on an NVIDIA-hosted endpoint, so this needs no GPU. It is
the shortest path for a new environment, and it is the same whether the machine
is a laptop or a cloud VM.

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

Open `http://localhost:3000`. To watch it come up, or to check what is running:

```bash
docker compose -f docker-compose.yaml ps
docker compose -f docker-compose.yaml logs -f
```

The deploy helper resolves models from `shared/configs/models.yaml`, starts
only locally deployed models referenced by roles with `source: local_model`, and then
starts the app stack from `docker-compose.yaml`.

The env file is a sourceable shell profile. Source the profile you want before
validation or deployment; `COMPOSE_DISABLE_ENV_FILE=1` keeps Docker Compose
from auto-parsing repo-root `.env` as dotenv and mixing environments.

Copy the template rather than editing it: every `.env.*` is gitignored except
the templates, so your filled-in copy stays out of git along with its keys.

For an existing deployment, restart the catalog service when changing the
text/image embedding model or catalog data source. Its fingerprint reuses only
matching complete collections and rebuilds mismatches. After the catalog is
healthy, restart the chain server so its process-lifetime cached capability
contract matches the active catalog.

## 🎛️ Deployment Options

Model routing is per role:

| `source` | Meaning | Local models started |
|----------|---------|--------------------|
| `endpoint` | Use the URL and model in the role's environment variables | none |
| `local_model` | Start and use the referenced locally deployed model | that service only |
| `disabled` | Capability is intentionally unavailable | none |

Use `shared/configs/models.yaml` to choose the source for each role. Copy
`.env.example` to a private env profile such as `.env` or `.env.hosted`, or
`.env.local-models.example` to `.env.local-models` for the self-hosted path,
edit it, then `source` the profile before running validation, deployment, or
raw Docker Compose commands.

There are two ways to point a role at a model on this machine, and they differ
in who starts the container:

| Approach | How | Who starts vLLM |
|----------|-----|-----------------|
| Environment overrides | Keep `source: endpoint` and set `LLM_BASE_URL` and friends to the local container, as `.env.local-models.example` does | **You do**, with `docker compose -f docker-compose-model-local.yaml up` |
| Declared in `models.yaml` | Set `source: local_model` and `local_service: local-llm` | `python scripts/model_config.py deploy` |

The [Locally Hosted Models](#-locally-hosted-models) walkthrough below uses the first
approach, because it self-hosts without editing a tracked file. The consequence
is that `scripts/model_config.py deploy` prints "No locally deployed models
required by models.yaml" and starts only the application: nothing in
`models.yaml` asked for a local service, so there is nothing for it to start.
That is correct behavior, not a failure. Start the models yourself first, as
[Step 4](#step-4-start-the-models-then-the-app) does.

Every model role can move to your own GPUs: the language, media,
text-embedding and guardrail roles each have a local service in
`docker-compose-model-local.yaml`, and `.env.local-models.example` points all
of them there. Image embedding is the exception; it has no role or local
service and stays off.

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

## 🏠 Locally Hosted Models

Use this only when this machine will serve the models itself. The local setup
mirrors the hosted default: Nemotron 3.5 Super answers the shopper and reads
photo and video uploads, Nemotron 3 Embed 1B embeds the catalog, and three
guardrail judges check each turn. All are locally deployed models: Hugging
Face checkpoints served by vLLM from `docker-compose-model-local.yaml`:

| Service | Role | Checkpoint | Weights | GPU memory in use | Default GPU |
|---------|------|------------|---------|-------------------|-------------|
| `local-llm` | `app_llm`, `vlm` | [`nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026`](https://docs.nvidia.com/nemo/automodel/model-coverage/omni/nvidia/nemotron-3-5-super-vl), BF16, tensor parallel 4 | ~227 GB | ~70 GB on each of 4 GPUs | 0-3, its own |
| `local-embedding` | `text_embedding` | [`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | ~2 GB | ~4 GB | 4, shared |
| `local-content-safety` | `content_safety` | [`nvidia/Nemotron-3.5-Content-Safety`](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety) | ~9 GB | ~22 GB | 4, shared |
| `local-topic-control` | `topic_control` | [`nvidia/llama-3.1-nemoguard-8b-topic-control`](https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control) LoRA on `meta-llama/Llama-3.1-8B-Instruct` | ~15 GB | ~28 GB | 4, shared |
| `local-video-safety` | `multimodal_safety` | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8) | ~35 GB | ~72 GB | 5, its own |

Memory in use was measured on H100 80 GB GPUs. It is more than the weights
because vLLM also reserves a KV cache, plus a few GB of working memory.

The default layout needs six 80 GB GPUs. `.env.local-models.example` points
every role at these services through the environment, so `models.yaml` does not change,
and adds video to the guarded modalities. Guardrails stay off by default, as on
the hosted path; turn them on per session with the UI's Guardrails toggle, or
for the deployment with `GUARDRAILS_ENABLED=true`. To deploy without them, set
`GUARDRAILS_AVAILABLE=false` in `.env.local-models`: the three guardrail
models are not started, which frees GPU 5 and 50 GB of GPU 4, and the UI has
no Guardrails toggle. The same setting works on the hosted path
([Guardrails](GUARDRAILS.md#deploying-without-guardrails)).

The guardrail service picks its request shape from the model name: the
dedicated on-topic/off-topic check runs only for a name containing
`topic-control`, and the Omni video request only for one containing
`nemotron-3-nano-omni`. The served names in the compose file are chosen to
match; keep them if you change a checkpoint.

### GPU Sizing

Two models fill their GPUs: the chat model reserves most of four, and Omni
most of one. The other three are small, so they share one GPU, GPU 4, using
about 54 of its 80 GB. Each reserves its weights plus a fixed 8 GiB KV cache
(`--kv-cache-memory-bytes`) rather than a share of the GPU, which vLLM would
misjudge while its neighbours load at the same time.

| Layout | GPUs (80 GB) | Placement | Set |
|--------|--------------|-----------|-----|
| Every model local (default) | 6 | 0-3 chat; 4 embedding, content safety and topic control; 5 video safety | nothing |
| Every model local, one GPU each | 8 | 0-3 chat; 4 embedding; 5 content safety; 6 topic control; 7 video safety | `LOCAL_CONTENT_SAFETY_GPU=5 LOCAL_TOPIC_CONTROL_GPU=6 LOCAL_VIDEO_SAFETY_GPU=7` |
| No guardrails | 5 | 0-3 chat; 4 embedding | `GUARDRAILS_AVAILABLE=false` |
| Guardrails hosted | 5 | 0-3 chat; 4 embedding | comment out the guardrails block in `.env.local-models` ([Step 4](#step-4-start-the-models-then-the-app)) |
| No guardrails, embedding hosted | 4 | 0-3 chat | `GUARDRAILS_AVAILABLE=false`, and comment out the embedding lines ([Step 4](#step-4-start-the-models-then-the-app)) |

`LOCAL_LLM_GPUS` (with `LOCAL_LLM_TP` set to as many GPUs), `LOCAL_EMBED_GPU`,
`LOCAL_CONTENT_SAFETY_GPU`, `LOCAL_TOPIC_CONTROL_GPU` and
`LOCAL_VIDEO_SAFETY_GPU` move any model to other GPU ids.

Omni runs at FP8, which leaves about 32 GB of KV cache on one 80 GB card. Its
BF16 checkpoint (~66 GB) leaves little room on that card; on a larger GPU, set
`LOCAL_VIDEO_SAFETY_HF_MODEL` to
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`.

**You need**

- Docker 20.10+ with the Compose plugin, and the NVIDIA Container Toolkit
- Your user in the `docker` group (`sudo usermod -aG docker $USER`, then log
  out and back in), or `sudo` in front of each `docker` command
- Python 3 with PyYAML on the host, for the deploy helpers (see
  [Software Dependencies](#software-dependencies))
- **Six 80 GB GPUs** for the default layout, such as H100 80 GB; see
  [GPU Sizing](#gpu-sizing) for the other layouts
- About 300 GB of disk for the checkpoints, most of it the chat model's BF16 weights
- A Hugging Face token with access to the two gated checkpoints: the
  Nemotron 3.5 Super chat model, and `meta-llama/Llama-3.1-8B-Instruct`, the
  topic-control adapter's base, which Meta approves by hand. The other three
  are public.
- Network access from the `local-video-safety` container at start, which
  installs vLLM's audio packages before serving

With every role local, no NGC account or NVIDIA API key is needed: the vLLM
images come from Docker Hub, the checkpoints from Hugging Face, and the UI's
`nvcr.io` base image pulls without a login.

### Step 1: Environment Setup

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant
python3 -m pip install --user -r requirements-deploy.txt   # or: sudo apt-get install -y python3-yaml

cp .env.local-models.example .env.local-models
${EDITOR:-nano} .env.local-models   # set HF_TOKEN
source .env.local-models
mkdir -p "$HF_CACHE"
```

### Step 2: Verify GPU Setup

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Step 3: Check Hugging Face Access

The two gated checkpoints fail only once their download starts, which can be
well into the first start. Check the token first; each line should print `200`
(`403` means the access request is still pending or was declined):

```bash
for repo in nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026 meta-llama/Llama-3.1-8B-Instruct; do
  curl -s -o /dev/null -w "%{http_code} $repo\n" \
    -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/$repo/resolve/main/config.json"
done
```

### Step 4: Start the Models, Then the App

```bash
echo "$LOCAL_MODEL_SERVICES"
docker compose -f docker-compose-model-local.yaml up -d --wait $LOCAL_MODEL_SERVICES
python3 scripts/model_config.py show --validate
docker compose -f docker-compose.yaml up -d --build
```

`LOCAL_MODEL_SERVICES` is set by `.env.local-models` to the model services its
roles point at: all five by default, and `local-llm local-embedding` with
`GUARDRAILS_AVAILABLE=false`. `--wait` returns once every model reports healthy. Start the app after that:
the catalog indexer embeds the catalog once, at startup. The first start
downloads ~300 GB into `HF_CACHE`, most of it the chat model, and can take an
hour; follow it with
`docker compose -f docker-compose-model-local.yaml logs -f local-llm`.

Each new shell needs `source .env.local-models` before these commands.

Both compose files share one Compose project, so `up` on either one warns that
the other's containers are "orphan containers" and suggests
`--remove-orphans`. Ignore the warning: that flag would stop the models.

To keep text embedding on the hosted endpoint, comment out the
`TEXT_EMBED_*` and `EMBED_API_KEY` lines in `.env.local-models`.
To keep guardrails hosted, comment out the `RAILS_*`, `RAIL_API_KEY` and
`MULTIMODAL_SAFETY_*` lines there, and set `RAIL_API_KEY` and
`MULTIMODAL_SAFETY_API_KEY` to build.nvidia.com keys. To leave guardrails out,
set `GUARDRAILS_AVAILABLE=false` there. Each drops the matching services from
`LOCAL_MODEL_SERVICES` after the next `source .env.local-models`.
[GPU Sizing](#gpu-sizing) lists the GPUs each of these layouts needs and the
variables that move models.

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

# Each model server lists the model it serves
for port in 8000 8001 8003 8004 8005; do curl -s http://localhost:$port/v1/models; echo; done

curl http://localhost:8009/ready    # chain server
curl http://localhost:8010/ready    # catalog: 503 until the index is built
curl http://localhost:8011/ready    # memory: 503 until migrations finish
curl http://localhost:3000
```

To check the guardrails, open the UI, turn on the **Guardrails** toggle (it
starts off) and ask something unrelated to shopping; the topic control judge
should refuse it. With the toggle off the same question gets an answer.

To check video safety, keep the toggle on and attach
`tests/evaluation/datasets/val/assets/casual_lady_fall.mp4` with "Find me
shoes like the ones in this video"; the turn should proceed and report
"Video safety and retail relevance" as used. Two limits apply to video, which
[Guardrails](GUARDRAILS.md#media) describes as not yet vetted:

- For about a minute after `local-video-safety` starts, the first videos take
  longer than `GUARDRAILS_TIMEOUT_SECONDS` (15 s) to judge, so those turns stop
  with the guardrails-unavailable message. Once one has been judged, later
  videos take about 2 s.
- The judge includes the video's audio, and a video with no audio track
  cannot be judged: it is an `error`, which stops the turn under the default
  `closed` failure mode.

### Model Metrics

Running the models yourself gets you what a hosted endpoint cannot: no rate
limit, and the model's own metrics. vLLM serves Prometheus metrics with nothing
to enable:

```bash
curl -s http://localhost:8000/metrics | grep -E '^vllm:'
```

[PERFORMANCE.md](PERFORMANCE.md) covers reading them.

### Stopping

```bash
docker compose -f docker-compose.yaml down
docker compose -f docker-compose-model-local.yaml down   # if you started them
```

No GPUs of your own? [NVIDIA Brev](https://developer.nvidia.com/brev) offers
pay-as-you-go GPU instances, and [BREV.md](BREV.md) has a walkthrough.

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

### Model Endpoints and API Keys

Each model role has its own endpoint, model name, and key variable, so any one
of them can be repointed without touching the others. Every URL and model name
is set in `.env.example`, and nowhere else: `shared/configs/models.yaml` only
names the variables a role reads, and a role whose variables are empty stops
the service at startup with the names of the missing variables.

| Role | What it does | Host in `.env.example` | URL and model variables | Key variable |
|------|--------------|------------------------|-------------------------|--------------|
| `app_llm` | Answers the shopper and drives tool use | `inference-api.nvidia.com` | `LLM_BASE_URL`, `LLM_MODEL` | `LLM_API_KEY` |
| `vlm` | Reads photo and video uploads | `inference-api.nvidia.com` | `VLM_BASE_URL`, `VLM_MODEL` | `VLM_API_KEY` |
| `text_embedding` | Embeds the catalog and text queries | `inference-api.nvidia.com` | `TEXT_EMBED_BASE_URL`, `TEXT_EMBED_MODEL` | `EMBED_API_KEY` |
| `content_safety` | Checks text and images for unsafe content (guardrails, off by default) | `integrate.api.nvidia.com` | `RAILS_CONTENT_BASE_URL`, `RAILS_CONTENT_MODEL` | `RAIL_API_KEY` |
| `topic_control` | Checks whether a request is on topic (guardrails) | `integrate.api.nvidia.com` | `RAILS_TOPIC_BASE_URL`, `RAILS_TOPIC_MODEL` | `RAIL_API_KEY` |
| `multimodal_safety` | Checks video, including embedded audio (guardrails) | `integrate.api.nvidia.com` | `MULTIMODAL_SAFETY_BASE_URL`, `MULTIMODAL_SAFETY_MODEL` | `MULTIMODAL_SAFETY_API_KEY` |

Image embedding is not shipped as a role; to turn it on, add an
`image_embedding` role to `models.yaml` with the same shape and set its
variables in your profile.

**The two hosts issue different keys, and they are not interchangeable.** A key
that works against `inference-api.nvidia.com` will be rejected by
`integrate.api.nvidia.com` and vice versa. The roles that are on by default all
use `inference-api.nvidia.com`, so one key in `NVIDIA_API_KEY` runs them.
Turning guardrails on adds the `integrate.api.nvidia.com` roles, which need a
build.nvidia.com key in `RAIL_API_KEY` and `MULTIMODAL_SAFETY_API_KEY`. A
mismatch is the most common first-deploy failure: the language model answers
normally while catalog search or guardrails return authentication errors, which
looks like a broken service rather than a key problem.
`python scripts/model_config.py show --validate` reports which URL, model and
key variables are missing, though it cannot tell whether a present key is the
right one for its host.

Compose chains sensible fallbacks so a single-key deployment works: `VLM_API_KEY`
falls back to `NVIDIA_API_KEY`, and `MULTIMODAL_SAFETY_API_KEY` falls back to
`VLM_API_KEY` and then `NVIDIA_API_KEY`. Set the specific variables when the
roles live on different hosts.

To repoint a role, set its `*_BASE_URL` and `*_MODEL` variables in your
profile, together: hosts name the same model differently
(`inference-api.nvidia.com` adds an `nvidia/` prefix). `NGC_API_KEY` is
unrelated to inference: it authenticates `nvcr.io` image pulls.

### Environment Variables

A default written as a `config.yaml` key lives only in
`shared/configs/chain_server/config.yaml`; look it up there. The variable
overrides it, and left empty it leaves the key in place. `docker-compose.yaml`
and `.env.example` pass these variables through empty, so change a default in
`config.yaml`, not in either of them. The exception is the model URL and model
name variables, whose values live only in `.env.example`.

The memory service has no config file; a default written as a constant lives
in its code, named in the table. `MEMORY_MAX_CONCURRENT_REQUESTS` has one more
copy, in `memory_retriever/Dockerfile`, because uvicorn's admission limit is
set there; a unit test fails if the two differ, so change both together.

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NGC_API_KEY` | NVIDIA NGC API key, for `nvcr.io` image pulls only | Only to pull images | - |
| `NVIDIA_API_KEY` | Fallback inference key for roles whose own key variable is unset | No | - |
| `LLM_API_KEY` | Language model API key | Yes | - |
| `LLM_BASE_URL` / `LLM_MODEL` | Shopping agent endpoint and model | Yes | `.env.example` |
| `TEXT_EMBED_BASE_URL` / `TEXT_EMBED_MODEL` | Catalog text embedding endpoint and model | Yes | `.env.example` |
| `APP_LLM_TEMPERATURE` | Temperature for the shopping agent and grounding editor; see [Model Sampling and Output Limits](#model-sampling-and-output-limits) | No | `config.yaml`: `llm_temperature` |
| `APP_LLM_FREQUENCY_PENALTY` | Optional frequency penalty for the same calls | No | off |
| `LLM_MAX_OUTPUT_TOKENS` | Shopping agent output ceiling per call | No | `config.yaml`: `llm_max_output_tokens` |
| `GROUNDING_EDITOR_MAX_OUTPUT_TOKENS` | Grounding editor output ceiling per call | No | `config.yaml`: `grounding_editor_max_output_tokens` |
| `VLM_BASE_URL`, `VLM_MODEL` | Media perception endpoint and model; set separately from `LLM_*` | Yes | `.env.example`, the same model as `app_llm` |
| `VLM_API_KEY` | Optional VLM media perception API key; Compose falls back to `NVIDIA_API_KEY` when unset | When `vlm` uses an authenticated endpoint and `NVIDIA_API_KEY` is unset | `NVIDIA_API_KEY` |
| `EMBED_API_KEY` | Embedding model API key | Yes | - |
| `RAIL_API_KEY` | Guardrails API key | Yes | - |
| `GUARDRAILS_URL` | Chain-server URL for the guardrail service | No | `http://rails:8012` |
| `RAILS_CONTENT_BASE_URL` / `RAILS_CONTENT_MODEL` | Endpoint and model for current text/image content safety | With guardrails | `.env.example` |
| `RAILS_TOPIC_BASE_URL` / `RAILS_TOPIC_MODEL` | Topic-control endpoint and model. The dedicated Topic Control model is preferred; selecting Content Safety applies the configured retail policy through `custom_policy` | With guardrails | `.env.example` |
| `MULTIMODAL_SAFETY_API_KEY` | Key for the independently routed video safety judge; Compose falls back to the VLM/NVIDIA key | When the video safety endpoint requires authentication | `VLM_API_KEY` |
| `MULTIMODAL_SAFETY_BASE_URL` | OpenAI-compatible endpoint for the video safety judge | With guardrails | `.env.example` |
| `MULTIMODAL_SAFETY_MODEL` | Video safety model, independent of perception | With guardrails | `.env.example` |
| `MULTIMODAL_SAFETY_VIDEO_FPS` | Temporal sampling rate sent to Nemotron Omni. The complete video object and embedded audio are submitted, but the model evaluates sampled frames | No | `rails.py` (2.0) |
| `GUARDRAILS_INPUT_EXECUTION_MODE` | Run the content and topic input rails in `parallel` or `sequential` mode | No | `rails.py` (parallel) |
| `GUARDRAILS_AVAILABLE` | `false` deploys without guardrails: no guardrail model is called or needs a key, `.env.local-models` starts no guardrail model, the UI hides the Guardrails toggle, and a request with `guardrails: true` gets a 400. Cannot be combined with `GUARDRAILS_ENABLED=true`. Read from the environment only, as both services and `.env.local-models` need it | No | true |
| `GUARDRAILS_ENABLED` | Default chain-server guardrails setting for requests that omit `guardrails`; accepts true/false, yes/no, on/off, or 1/0. Guardrails is opt-in: set this to enable it | No | `config.yaml`: `guardrails_enabled` (off) |
| `GUARDRAILS_CLIENT_CAN_DISABLE` | With guardrails enabled, whether a request's `guardrails: false` is honoured. Off by default, so callers cannot switch off the deployment's guardrails; set it for evaluation runs that compare with and without them | No | false |
| `GUARDRAILS_FAILURE_MODE` | Required-check error/timeout behavior: `open` bypasses and `closed` stops the turn. Explicit unsafe decisions always block in either mode | No | `config.yaml`: `guardrails_failure_mode` (closed) |
| `GUARDRAILS_TIMEOUT_SECONDS` | Timeout for each isolated guardrail service decision. Both services read it: the chain server bounds its call, the guardrail service bounds the judges behind it | No | `config.yaml`: `guardrails_timeout_seconds`, and `rails.py` (15.0) |
| `GUARDRAILS_SPECULATIVE_MAIN_MODEL_ENABLED` | For guarded text-only turns, overlap the first app-model step with input guardrails while holding every tool behind the allow decision. Reduces latency but blocked turns can still incur one app-model request. Media remains sequential | No | `config.yaml`: `guardrails_speculative_main_model_enabled` (off) |
| `GUARDRAILS_SUPPORTED_MODALITIES` | Modalities enforced by guardrails and advertised to clients | No | `config.yaml`: `guardrails_supported_modalities` |
| `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` | Shared deadline for the Deep Agents graph and grounding editor before the durable turn fails cleanly | No | `config.yaml`: `deepagents_execution_timeout_seconds` |
| `DEEPAGENTS_RECURSION_LIMIT` | Maximum model-and-tool rounds in one turn before the graph stops | No | `config.yaml`: `deepagents_recursion_limit` |
| `GROUNDING_EDITOR_RESERVE_SECONDS` | Slice of the turn deadline held back so the grounding editor can still run after the agent loop | No | `config.yaml`: `grounding_editor_reserve_seconds` |
| `GROUNDING_REWRITE_ENABLED` | Let the grounding editor rewrite a reply that drifted from tool evidence | No | `config.yaml`: `grounding_rewrite_enabled` |
| `GROUNDING_REWRITE_MAX_EVIDENCE_CHARS` | Ceiling on the evidence passed to that editor | No | `config.yaml`: `grounding_rewrite_max_evidence_chars` |
| `EXPOSE_AGENT_DIAGNOSTICS` | Expose detailed agent/tool traces in query responses; enable only behind a trusted operator or evaluation surface | No | `config.yaml`: `expose_agent_diagnostics` (off) |
| `CATALOG_SEARCH_TIMEOUT_SECONDS` | Optional chain-server timeout for catalog search requests | No | no timeout |
| `CATALOG_RETRIEVER_URL` / `MEMORY_RETRIEVER_URL` | Where the chain server reaches the retrieval services | No | `config.yaml`: `retriever_port`, `memory_port` |
| `CATALOG_IMAGE_EMBEDDING_ENABLED` | Build image embeddings and the image collection at index time, enabling visual search | No | `catalog_retriever/config.yaml`: `image_embedding_enabled` (off) |
| `CATALOG_VECTOR_INDEX_TYPE` | Vector index for both catalog collections: `AUTOINDEX` (CPU) or `GPU_CAGRA` (needs the Milvus `-gpu` image); see [Vector Search](VECTOR_SEARCH.md) | No | `catalog_retriever/config.yaml`: `vector_index.type` (`AUTOINDEX`) |
| `CATALOG_GPU_SEARCH` | With `GPU_CAGRA`, search on the GPU too rather than building on the GPU and searching on the CPU | No | `catalog_retriever/config.yaml`: `vector_index.gpu_search` (off) |
| `CATALOG_DATA_SOURCE` / `CATALOG_SCHEMA_SOURCE` | Catalog JSONL and schema sidecar paths; see [Catalog Architecture](CATALOG_ARCHITECTURE.md#replace-the-catalog) | No | `catalog_retriever/config.yaml`: `data_source`, `schema_source` |
| `CATALOG_DB_PORT` | Milvus URI the catalog service connects to | No | `catalog_retriever/config.yaml`: `db_port` |
| `MAX_CATALOG_SEARCHES_PER_TURN` | Caps distinct catalog taxonomy-plus-hard-constraint scope executions in one assistant turn; a repeated scope is stopped even when semantic wording changes | No | `config.yaml`: `max_catalog_searches_per_turn` |
| `MAX_PRODUCT_DETAIL_READS_PER_TURN` | Caps Deep Agents product-detail reads in one assistant turn | No | `config.yaml`: `max_product_detail_reads_per_turn` |
| `CHECKPOINT_STORE` | Deep Agents conversation checkpoint store; currently supports only `memory` | No | `memory` |
| `MEMORY_DATABASE_URL` | SQLite URL for durable raw turns and cart state; Compose supplies the named-volume path | No | Compose: `sqlite:////data/context.db` |
| `MEMORY_SQLITE_BUSY_TIMEOUT_MS` | SQLite lock wait for the single memory-service writer | No | `DEFAULT_BUSY_TIMEOUT_MS` in `memory_retriever/src/database.py` |
| `MEMORY_MAX_CONCURRENT_REQUESTS` | Requests the memory service works on at once; sizes its connection pool, its threadpool and uvicorn's admission limit together | No | `DEFAULT_MAX_CONCURRENT_REQUESTS` in `memory_retriever/src/database.py`, and `memory_retriever/Dockerfile` |
| `MEMORY_TURN_ABANDON_SECONDS` | Age at which startup or the next turn start marks an unfinished `started` turn abandoned | No | `DEFAULT_ABANDONED_SECONDS` in `memory_retriever/src/conversations.py` |
| `MEMORY_RECENT_TURNS` | Maximum prior context-eligible raw turns returned at the next durable turn start | No | `DEFAULT_RECENT_TURNS_LIMIT` in `memory_retriever/src/conversations.py` |
| `WEATHER_ENABLED` | Registers the forecast tool with the shopper agent (needs `WEATHER_API_KEY`) | No | `config.yaml`: `weather.enabled` (off) |
| `WEATHER_API_KEY` | Visual Crossing server-side credential, read indirectly from the variable named by chain-server weather config | Only when directly constructing an enabled weather client | empty |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Where traces are exported; unset disables export. See [Observability](OBSERVABILITY.md) | No | unset |
| `OTEL_SERVICE_NAME` | Service name traces are attributed to | No | `chain-server` |
| `RELAY_ENABLED` | Route model calls through NeMo Relay for richer LLM spans | No | `config.yaml`: `relay_enabled` (off) |
| `RELAY_OTLP_ENDPOINT` | Where the in-container Relay sends its spans | No | `http://127.0.0.1:4318` |
| `INSTALL_RELAY` | Build argument that installs the Relay dependency into the chain-server image; required before `RELAY_ENABLED` can work | No | `false` |
| `SHUTDOWN_GRACE_SECONDS` | Time the chain server and memory service drain in-flight turns before exiting | No | image default |
| `CHAIN_SERVER_RELOAD` | Reload the chain server on source changes; development only | No | off |
| `MEMORY_DB_USER` / `MEMORY_DB_PASSWORD` / `MEMORY_DB_NAME` | PostgreSQL credentials under the `postgres` Compose profile. **Change these before any shared deployment**; the defaults are `memory` for all three | No | `memory` |
| `HF_TOKEN` | Hugging Face token with access to the gated local checkpoints: the LLM and Llama 3.1 8B Instruct | Local only | - |
| `HF_CACHE` | Hugging Face cache the locally deployed models download into | Local only | `~/.cache/huggingface` |
| `LOCAL_LLM_HF_MODEL` | Checkpoint the local vLLM server loads | Local only | `nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026` |
| `LOCAL_LLM_GPUS` | Comma-separated GPU ids for the local LLM; supply as many as `LOCAL_LLM_TP` | Local only | `0,1,2,3` |
| `LOCAL_LLM_TP` | Tensor parallel size for the local LLM | Local only | `4` |
| `LOCAL_LLM_VLLM_IMAGE` | vLLM image serving the local LLM | Local only | `vllm/vllm-openai:v0.30.0` |
| `LOCAL_EMBED_GPU` | GPU id for the local embedding model | Local only | `4` |
| `LOCAL_EMBED_VLLM_IMAGE` | vLLM image serving local embeddings | Local only | `vllm/vllm-openai:v0.24.0` |
| `LOCAL_CONTENT_SAFETY_GPU` | GPU id for the local content safety model | Local only | `4` |
| `LOCAL_TOPIC_CONTROL_GPU` | GPU id for the local topic control model | Local only | `4` |
| `LOCAL_VIDEO_SAFETY_GPU` | GPU id for the local video safety model | Local only | `5` |
| `LOCAL_VIDEO_SAFETY_HF_MODEL` | Checkpoint the local video safety server loads; the `-BF16` variant needs about twice the memory | Local only | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` |
| `LOCAL_CONTENT_SAFETY_VLLM_IMAGE` / `LOCAL_TOPIC_CONTROL_VLLM_IMAGE` / `LOCAL_VIDEO_SAFETY_VLLM_IMAGE` | vLLM images serving the local guardrail models | Local only | `vllm/vllm-openai:v0.20.0` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `NODE_ENV` | Node environment | No | `production` |
| `SHARED_CONFIG_ROOT` | Directory the services read `shared/configs/` from | No | `/app/shared/configs` |

### Weather Tool

The chain server ships a provider-neutral daily weather client and
`get_weather_forecast_tool`, with Visual Crossing as the first adapter. It is
**off by default**: disabled, it is not registered at all, and startup, health
checks, shopper turns, and offline tests make no provider request and need no
weather key.

Enabled, the tool is granted only by the `destination-weather` skill, forecasts
a place the shopper named within a 15-day horizon, and runs at most twice per
turn. It has no FastAPI route and no UI of its own.

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

To turn it on, set `WEATHER_ENABLED=true` and supply `WEATHER_API_KEY` through
an ignored `.env`, the process environment, or a secret manager. The config
stores only the variable *name*, never the value, and enabling the client
without that key fails closed. Compose passes both variables to `chain-server`
alone — never to the catalog, memory, guardrail, UI, or local model services —
and the local process runner enforces the same boundary.

`python scripts/weather_smoke.py` makes at most one provider request, for
checking credentials. It needs `WEATHER_ENABLED`, `WEATHER_API_KEY`, and
`WEATHER_SMOKE_ZIP` set, and optionally `WEATHER_SMOKE_DATE` or a
`WEATHER_SMOKE_START_DATE`/`WEATHER_SMOKE_END_DATE` pair. It deliberately
prints no location, dates, forecast, key, or URL — only outcome category,
schema validity, and latency. Nothing runs it automatically.

The adapter normalizes forecast evidence and persists nothing. Before any later
work displays or stores it, confirm your Visual Crossing plan's attribution,
storage, sharing, and uncertainty requirements in its
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
`shared/configs/chain_server/config.yaml`. Model endpoints are configured
separately; see [Model Routing](#model-routing):

```yaml
retriever_port: "http://localhost:8010"
memory_port: "http://localhost:8011"
guardrails_url: "http://localhost:8012"
memory_length: 16384
deepagents_recursion_limit: 24
max_catalog_searches_per_turn: 3
max_product_detail_reads_per_turn: 2
guardrails_enabled: false
guardrails_failure_mode: closed
guardrails_timeout_seconds: 15.0
guardrails_speculative_main_model_enabled: false
guardrails_supported_modalities: [text, image]
```

The legacy routing and chatter prompt keys remain in that file for compatibility
paths; they do not configure the serving Deep Agents runtime.

### Updating Catalog Filter Metadata

The active runtime gets available product filters from the catalog retriever
after the catalog data is loaded. Do not maintain product categories in
`shared/configs/chain_server/config.yaml`.

The sidecar declares field types and uses; every value, range, coverage figure
and taxonomy scope is discovered from the rows.

The deployment-side part is two paths in
`shared/configs/catalog_retriever/config.yaml`:

```yaml
data_source: "/app/shared/data/enriched_products.jsonl"
schema_source: "/app/shared/data/enriched_products.schema.yaml"
```

Changing either of those, or the embedding model, changes the service
fingerprint, so the index must be rebuilt by the `catalog-indexer` service. A
serving container never indexes itself; until the indexer has run it answers
`/ready` with 503.

For the field-role rules, see
[Catalog Architecture](CATALOG_ARCHITECTURE.md#the-sidecar). For the
reindex-and-verify sequence, see
[Replace the catalog](CATALOG_ARCHITECTURE.md#replace-the-catalog).

### Model Routing

Model URLs and model names are set in one file, `.env.example`, or the profile
you copy from it. `shared/configs/models.yaml` holds the rest of the routing:
which roles exist, each role's `source`, the environment variables it reads for
its URL, model and key, and the locally deployed model metadata. A `base_url`
or `model` written on a role in `models.yaml` is rejected at startup, so a value
cannot hide there. Service behavior stays in each service's normal config file.

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

#### Applying a Routing Change

Whichever path you deployed with, these are the two commands to rerun after
changing your profile or `models.yaml`:

```bash
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

`show --validate` prints the resolved routing for every role without printing
any key value, and fails if a required API-key or endpoint variable is missing.
`deploy` starts only the local services that roles actually reference, then the
app stack.

Keep the env profile sourced while you do it. Without it, the endpoint roles
have no URL or model, and both commands stop and name the missing variables. A
`local_model` role ignores those variables and always uses its local service's
URL and model from `local_models.services`.

#### Adding or Changing Models

To point an existing role at another model, change its `*_BASE_URL` and
`*_MODEL` in your profile; `models.yaml` does not change. To add a role, or
change how one is served, edit its entry in `shared/configs/models.yaml`:

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

Every role takes one of those two shapes. Only the variable prefix changes —
`LLM_`, `VLM_`, `EMBED_`, and so on — so pointing media perception at its own
hosted endpoint means the same `endpoint` block with `VLM_BASE_URL`,
`VLM_MODEL`, and `VLM_API_KEY`. Set the variables in the profile you source,
then rerun the two commands above.

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

#### Guardrail Defaults

Guardrails ships disabled and spans two services, so each default lives with
whichever one reads it. [Guardrails](GUARDRAILS.md) owns that layout, along
with what a failed check costs and what the shopper is told.

The `GUARDRAILS_*` and `RAIL_API_KEY` variables above override those defaults
for one deployment, as everywhere else here.

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

# Move the local models to free GPUs (LOCAL_LLM_GPUS must hold LOCAL_LLM_TP
# IDs); for example, everything two GPUs up from the default layout:
LOCAL_LLM_GPUS=2,3,4,5 LOCAL_EMBED_GPU=6 LOCAL_CONTENT_SAFETY_GPU=6 \
LOCAL_TOPIC_CONTROL_GPU=6 LOCAL_VIDEO_SAFETY_GPU=7 \
  docker compose -f docker-compose-model-local.yaml up -d --wait $LOCAL_MODEL_SERVICES

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
