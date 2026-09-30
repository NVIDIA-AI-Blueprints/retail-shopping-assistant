<a id="top"></a>
# 🛍️ NVIDIA AI Blueprint: Retail Shopping Assistant

<div align="center">

![NVIDIA Logo](https://avatars.githubusercontent.com/u/178940881?s=200&v=4)

**A reference shopping assistant built on LangChain Deep Agents and open NVIDIA models**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://www.docker.com/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-AI-Blueprints/retail-shopping-assistant?style=social)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/NVIDIA-AI-Blueprints/retail-shopping-assistant)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/NVIDIA-AI-Blueprints/retail-shopping-assistant)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/commits)
[![Contributors](https://img.shields.io/github/contributors/NVIDIA-AI-Blueprints/retail-shopping-assistant)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/graphs/contributors)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Get Started](#get-started)
  - [Using NVIDIA Endpoints](#using-nvidia-endpoints)
  - [Locally Hosted Models](#locally-hosted-models)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Contribution Guidelines](#contribution-guidelines)
- [Community](#community)
- [References](#references)
- [License](#license)

## Overview

The Retail Shopping Assistant is a reference conversational shopping advisor
built on open tooling: LangChain's
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) as the
agent harness, and open NVIDIA Nemotron models for reasoning, perception and
safety. Shoppers type, or upload a photo or a short video.

It ships a representative catalog — apparel, footwear, bags, eyewear and
jewelry, with real images, prices, sizes, materials and colors — and that catalog declares which of its own fields become
shopper-facing filters. Point the service at your own data and the search
contract follows. Retrieval stays deterministic: the agent writes the query,
and the catalog service does the embedding search and ranking.

It is a complete reference rather than a demo. Five notebooks carry one
deployment the whole way: stand it up, read a single turn's traces, replay
conversations to tell flaky from broken, then serve the model on your own GPUs
and measure time to first token and throughput under load.

Run it on NVIDIA-hosted endpoints with no GPU, or serve the models yourself.

## Key Features

- **Natural-language product search** — the agent plans the query, the catalog retrieves deterministically
- **Photo and video understanding** — Nemotron 3.5 Super VL reads uploads in shopping context
- **Deterministic cart** — typed tools, with totals computed in code rather than generated
- **Skills with enforced tool grants** — each skill declares its tools, and the grant is rechecked at dispatch
- **Durable conversation turns** — a turn's outcome and the products it showed survive a restart
- **Representative shopper profiles** — five database-backed shoppers, or Guest
- **Optional content safety and topic control** — separate guard models, off by default
- **SSE response streaming** — products, images, text, and metrics as discrete events
- **Per-turn observability** — which skill ran, what the model was told, and why a tool was refused

## Architecture

![Assistant architecture](docs/images/shopper-agent-architecture.svg)

[Open the architecture diagram at full size](docs/images/shopper-agent-architecture.svg).

Five services, each owning one concern:

| Service | Owns |
|---------|------|
| **Chain Server** | The Deep Agents turn: skill activation, tool dispatch, grounded response assembly |
| **Catalog Retriever** | Embedding search, hard filtering, and deterministic ranking. No generative model |
| **Memory Retriever** | Durable turns, cart state, presented-product history, and shopper profiles |
| **Guardrails** | Content safety and topic control, when enabled |
| **UI** | React front end with shopper selection and per-turn detail |

A turn starts a durable record before any model work, activates the smallest
applicable skill set, dispatches only the tools those skills grant, and
assembles a reply from tool evidence. [Assistant
Architecture](docs/ASSISTANT_ARCHITECTURE.md) walks through it in full,
including the catalog data foundation and the memory boundaries.

## Get Started

Two paths. They differ only in where the models run; everything else is
identical. Start with NVIDIA endpoints unless you specifically need your own
GPUs.

### Using NVIDIA Endpoints

Every model runs on a hosted endpoint. **No GPU required.**

**You need**

- Docker 20.10+ with the Compose plugin
- Python on the host, for the deploy helpers
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)
- An [NGC account](https://ngc.nvidia.com/) to pull the UI base image

**Deploy**

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

docker login nvcr.io                 # username: $oauthtoken, password: NGC API key
python -m pip install --user -r requirements-deploy.txt

cp .env.example .env
$EDITOR .env                         # set NVIDIA_API_KEY
source .env

python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

`show --validate` prints the resolved endpoint for every model role without
printing any key, and fails if a required one is missing. Copy the template
rather than editing it: every `.env.*` is gitignored except the templates, so
your filled-in copy stays out of git along with its keys.

**Verify**

```bash
curl -s http://localhost:8010/ready   # catalog: 503 until the index is built
```

Compose indexes the catalog for you. The `catalog-indexer` service runs once,
builds the index, and exits, and `catalog-retriever` waits for it to succeed
before starting. Check `/ready` rather than `/health`: a catalog container with
no matching index is deliberately alive and serving no traffic.

Then open **http://localhost:3000**.

### Locally Hosted Models

The models run on your own GPUs, pulled from Hugging Face or NGC.

How many GPUs you need depends on which models you run and at what precision,
and each model publishes its own footprint. Taking H100 80GB as the example:

| Model | Role | On H100 80GB | Published footprint |
|-------|------|--------------|---------------------|
| Nemotron 3.5 Super | Shopping agent, photo and video | 4 GPUs at BF16 | [121B MoE, ~227 GB BF16](https://docs.nvidia.com/nemo/automodel/model-coverage/omni/nvidia/nemotron-3-5-super-vl) |
| Nemotron 3 Embed 1B | Catalog and query embedding | shares 1 GPU | [3.6 GB at FP16](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html) |
| Nemotron 3.5 Content Safety | Optional moderation | shares 1 GPU | [4B, 8 GB VRAM and up](https://huggingface.co/blog/nvidia/nemotron-3-5-content-safety) |
| Llama 3.1 NemoGuard 8B Topic Control | Optional off-topic checks | 1 GPU | [48 GB](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/support-matrix.html) |

`docker-compose-model-local.yaml` serves the first two, which is the five-GPU
default layout: four for the agent's tensor parallel group
(`LOCAL_LLM_TP=4`, `LOCAL_LLM_GPUS=0,1,2,3`) and one for embedding
(`LOCAL_EMBED_GPU=4`). The guard models run on hosted endpoints.

Those published figures assume a model has the card to itself, since vLLM
reserves most of it for the KV cache. Putting the smaller models together on
one GPU means capping `--gpu-memory-utilization` for each.

**You need**

- Docker 20.10+ with the Compose plugin, and the NVIDIA Container Toolkit
- Python on the host, for the deploy helpers
- **Five GPUs** for the default layout above, as on an 8x H100 80 GB machine
- About 240 GB of disk for the chat model's BF16 checkpoint
- A Hugging Face token with access to that checkpoint
- An NVIDIA API key, because some model roles still use hosted endpoints
- An [NGC account](https://ngc.nvidia.com/) to pull the UI base image

**Deploy**

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

docker login nvcr.io
python -m pip install --user -r requirements-deploy.txt

cp .env.local-models.example .env.local-models
$EDITOR .env.local-models            # set HF_TOKEN and NVIDIA_API_KEY
source .env.local-models
mkdir -p "$HF_CACHE"

docker compose -f docker-compose-model-local.yaml up -d --wait local-llm local-embedding
python scripts/model_config.py show --validate
docker compose up -d --build
```

Start the models before the app: `--wait` returns once both report healthy, and
the catalog indexer embeds the catalog once, at startup. The first start
downloads the checkpoint into `HF_CACHE` and can take an hour; later starts
read the cache. Follow it with
`docker compose -f docker-compose-model-local.yaml logs -f local-llm`.

To keep text embedding on the hosted endpoint, comment out the two
`TEXT_EMBED_*` lines in `.env.local-models` and start only `local-llm`.
`LOCAL_LLM_GPUS`, `LOCAL_LLM_TP`, and `LOCAL_EMBED_GPU` change the placement.

**What this gets you** that a hosted endpoint cannot: no rate limit, and the
model's own metrics. vLLM serves Prometheus metrics with nothing to enable:

```bash
curl -s http://localhost:8000/metrics | grep -E '^vllm:'
```

[docs/PERFORMANCE.md](docs/PERFORMANCE.md) covers reading them.

**Stopping**

```bash
docker compose -f docker-compose.yaml down
docker compose -f docker-compose-model-local.yaml down   # if you started them
```

No GPUs of your own? [NVIDIA Brev](https://developer.nvidia.com/brev) offers
pay-as-you-go GPU instances, and [docs/BREV.md](docs/BREV.md) has a walkthrough.

## Notebooks

Five notebooks take the deployment you just made and teach what it does. They
are the fastest way to understand this blueprint, and the best place to start
after the UI loads.

| Notebook | You will | Time | GPU |
|---|---|---|---|
| [1 · Getting Started](notebook/1_Getting_Started.ipynb) | deploy on hosted endpoints, hold a first conversation, and call each component once | ~20 min | No |
| [2 · Observability](notebook/2_Observability.ipynb) | read a turn's traces: skills, prompts, tool calls, refusals, time and tokens | ~30 min | No |
| [3 · Evaluation](notebook/3_Evaluation.ipynb) | replay conversations, investigate a failure, tell flaky from broken, run the Challenger and Judge | ~30 min | No |
| [4 · Capture Traces](notebook/4_Capture_Traces.ipynb) | record every model call of 25 journeys as an AIPerf trace, for Notebook 5 to replay | ~35 min | No |
| [5 · Performance Measurement](notebook/5_Performance_Measurement.ipynb) | serve Nemotron with vLLM and measure prefix-cache hit rate, TTFT, and throughput by concurrency | hours | Yes |

Notebook 2 is the one to read if you only read one: it shows you what the model
was actually told on a given turn, which is how you tell "the model ignored the
rule" from "the model never saw the rule".

See [notebook/README.md](notebook/README.md) for prerequisites and how to start
Jupyter.

## Documentation

[Documentation Hub](docs/README.md) indexes everything and suggests a path for
your role.

1. **[User Guide](docs/USER_GUIDE.md)** — using the assistant: chat, search, cart, uploads, FAQ
2. **[Assistant Architecture](docs/ASSISTANT_ARCHITECTURE.md)** — the catalog foundation, one shopper turn, and memory boundaries
   - **[Skills](docs/SKILLS.md)** — registered skills, runtime loading, and the markdown tuning loop
   - **[Tools](docs/TOOLS.md)** — registered tools, risk classes, and per-skill access boundaries
3. **[Observability](docs/OBSERVABILITY.md)** — read a session turn by turn, and dig into why one turn did what it did
4. **[Testing and Evaluation](tests/README.md)** — unit, integration, and Challenger/Judge workflows
5. **[Performance Measurement](docs/PERFORMANCE.md)** — latency budget, saturation point, and deployment sizing
6. **[Deployment and Configuration](docs/DEPLOYMENT.md)** — both deployment paths, and every configuration setting with its default
7. **[API Reference](docs/API.md)** — endpoints, request and response models, and streaming frames
8. **[Catalog Architecture](docs/CATALOG_ARCHITECTURE.md)** — the data format, declaring which fields become shopper-facing filters, and reindexing onto your own catalog
9. **[Guardrails](docs/GUARDRAILS.md)** — the optional safety layer: how it decides, what a failed check costs, and how to turn it on

Contributors should also read [AGENTS.md](AGENTS.md), the service map and file
layout for this codebase.

## Contribution Guidelines

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup and environment configuration
- Coding standards and best practices
- Testing guidelines and examples
- Pull request process and code review guidelines

## Community

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues)
- **Documentation**: [Comprehensive guides and references](docs/README.md)

## References

### NVIDIA AI Blueprints
- [NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints): Collection of AI application blueprints
- [NVIDIA NGC](https://ngc.nvidia.com/): AI platform and container registry
- [build.nvidia.com](https://build.nvidia.com): Hosted model endpoints and API keys

### Technologies Used
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview): Agent harness for tool and skill orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph): Runtime used underneath Deep Agents
- [vLLM](https://github.com/vllm-project/vllm): Inference server for locally hosted models
- [FastAPI](https://fastapi.tiangolo.com/): Modern Python web framework
- [React](https://reactjs.org/): JavaScript library for building user interfaces
- [Milvus](https://milvus.io/): Vector database for similarity search

### Models
- [Nemotron 3.5 Super VL](https://docs.nvidia.com/nemo/automodel/model-coverage/omni/nvidia/nemotron-3-5-super-vl): Shopping agent, and photo and video perception
- [Nemotron 3 Embed 1B](https://build.nvidia.com/nvidia/nemotron-3-embed-1b/modelcard): Catalog and query embedding
- [Nemotron 3.5 Content Safety](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3.5-content-safety): Optional moderation of shopper input and assistant output
- [Llama 3.1 NemoGuard 8B Topic Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control): Optional off-topic detection

## License

GOVERNING TERMS: Use of the blueprint software and materials and NIM containers are governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-specific Terms for AI products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/);  and the use of models is governed by the [NVIDIA Community Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/).
 
ADDITIONAL INFORMATION: [Llama 3.1 Community License Agreement](https://www.llama.com/llama3_1/license/) for Llama 3.1 70B Instruct NIM, Llama 3.1 NemoGuard 8B - Content Safety and Llama 3.1 NemoGuard 8B - Topic Control models, built with Llama.
 
This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use, found in [License-3rd-party.txt](LICENSE-3rd-party.txt).
 
Use of the product catalog data in the retail shopping assistant is governed by the terms of the [NVIDIA Data License for Retail Shopping Assistant](LICENSE-assets.txt) (15Aug2025).

---

<div align="center">

[Back to Top](#top)

</div>
