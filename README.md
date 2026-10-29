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

It is a complete reference rather than a demo. The notebooks carry one
deployment the whole way: stand it up, read a single turn's traces, replay
conversations to tell flaky from broken, then serve the model on your own GPUs
and measure time to first token and throughput under load.

Run it on NVIDIA-hosted endpoints with no local GPU, or serve the models yourself.

## Key Features

- **Natural-language product search** — the agent plans the query, the catalog retrieves deterministically
- **Photo and video understanding** — Nemotron 3.5 Super VL reads uploads in shopping context
- **Deterministic cart** — typed tools, with totals computed in code rather than generated
- **Skills with enforced tool grants** — each skill declares its tools, and the grant is rechecked at dispatch
- **Durable conversation turns** — a turn's outcome and the products it showed survive a restart
- **Optional content safety and topic control** — separate guard models, on by default
- **Optional weather-aware styling** — a destination's forecast shapes the suggestions; off by default, see [Weather Tool](docs/DEPLOYMENT.md#weather-tool)
- **SSE response streaming** — products, images, text, and metrics as discrete events
- **Per-turn observability** — which skill ran, what the model was told, and why a tool was refused

## Architecture

![Assistant architecture](docs/images/retail-diagram-shopping-assistant-blueprint-202610.jpg)

[Open the architecture diagram at full size](docs/images/retail-diagram-shopping-assistant-blueprint-202610.jpg).

Five services, each owning one concern:

| Service | Owns |
|---------|------|
| **Orchestrator** | The Deep Agents turn: skill activation, tool dispatch, grounded response assembly |
| **Catalog Retriever** | Embedding search, hard filtering, and deterministic ranking. No generative model |
| **Memory Retriever** | Durable turns, cart state, and presented-product history |
| **Guardrails** | Content safety and topic control, when enabled |
| **UI** | React front end with shopper selection and per-turn detail |

A turn starts a durable record before any model work, activates the smallest
applicable skill set, dispatches only the tools those skills grant, and
assembles a reply from tool evidence. [Assistant
Architecture](docs/ASSISTANT_ARCHITECTURE.md) walks through it in full,
including the catalog data foundation and the memory boundaries.

## Get Started

Every model runs on a hosted NVIDIA endpoint, so this path needs **no local GPU**.

**You need** Docker 20.10+ with the Compose plugin, Python on the host, an
NVIDIA API key from [build.nvidia.com](https://build.nvidia.com).

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

python -m pip install --user -r requirements-deploy.txt

cp .env.example .env
$EDITOR .env                         # set NVIDIA_API_KEY, a build.nvidia.com key
source .env

python scripts/model_config.py show --validate   # every model role resolves; no keys printed
python scripts/model_config.py deploy --build
curl -s http://localhost:8010/ready  # catalog: 503 until the index is built
```

Every model, guardrails included, is on `integrate.api.nvidia.com`, so
`NVIDIA_API_KEY` must be a [build.nvidia.com](https://build.nvidia.com) key.
If guardrails refuse every turn, the key is usually the cause; see
[Troubleshooting](docs/DEPLOYMENT.md#6-every-reply-says-i-cannot-safely-validate-this-request-right-now).

Then open **http://localhost:3000**, and continue with
[Notebook 1](notebook/1_Getting_Started.ipynb).

**Running the models on your own GPUs instead?** The default layout needs six
80 GB GPUs, such as H100s: four for the chat model, one shared by the
embedding, content safety and topic control models, and one for video safety;
five without guardrails (`GUARDRAILS_AVAILABLE=false`). See
[Locally Hosted Models](docs/DEPLOYMENT.md#-locally-hosted-models) for the GPU
sizing, setup, and model metrics. No GPUs of your own?
[NVIDIA Brev](https://developer.nvidia.com/brev) offers pay-as-you-go GPU
instances; [On NVIDIA Brev](docs/DEPLOYMENT.md#on-nvidia-brev) covers the setup.

## Notebooks

The notebooks take the deployment you just made and teach what it does. They
are the fastest way to understand this blueprint, and the best place to start
after the UI loads.

| Notebook | You will | Time | GPU |
|---|---|---|---|
| [1 · Getting Started](notebook/1_Getting_Started.ipynb) | deploy on hosted endpoints, hold a first conversation, and call each component once | ~20 min | No |
| [2 · Observability](notebook/2_Observability.ipynb) | read a turn's traces: skills, prompts, tool calls, refusals, time and tokens | ~20 min | No |
| [3 · Evaluation](notebook/3_Evaluation.ipynb) | replay conversations, investigate a failure, tell flaky from broken, run the Challenger and Judge | ~30 min | No |
| [4 · Capture Traces](notebook/4_Capture_Traces.ipynb) | record every model call of the journeys as an AIPerf trace, for Notebook 5 to replay | ~35 min | No |
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

We welcome contributions. [Contributing](CONTRIBUTING.md) covers the fork and
pull-request workflow, the sign-off we require on every commit, and the
Developer Certificate of Origin. [Code of Conduct](CODE_OF_CONDUCT.md) applies
to everyone taking part.

For development setup see [Deployment](docs/DEPLOYMENT.md), and for running the
suites see [Testing and Evaluation](tests/README.md).

## Community

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues)
- **Documentation**: [Comprehensive guides and references](docs/README.md)

## References

### NVIDIA AI Blueprints
- [NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints): Collection of AI application blueprints
- [build.nvidia.com](https://build.nvidia.com): Hosted model endpoints and API keys

### Technologies Used
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview): Agent harness for tool and skill orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph): Runtime used underneath Deep Agents
- [vLLM](https://github.com/vllm-project/vllm): Inference server for locally hosted models
- [FastAPI](https://fastapi.tiangolo.com/): Modern Python web framework
- [React](https://reactjs.org/): JavaScript library for building user interfaces
- [Milvus](https://milvus.io/): Vector database for similarity search

### Models
- [Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16): Shopping agent, and photo and video perception
- [Nemotron 3.5 Embed 1B](https://build.nvidia.com/nvidia/nemotron-3_5-embed-1b): Catalog and query embedding
- [Nemotron 3.5 Content Safety](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety): Optional moderation of shopper input and assistant output
- [Llama 3.1 NemoGuard 8B Topic Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control): Optional off-topic detection

## License

GOVERNING DOWNLOAD TERMS: Use of this blueprint software is governed by the [Apache 2.0 License](LICENSE), and enables use of separate open source and proprietary software, models, data and services governed by their respective accompanying licenses.

### Main Components and Their Terms

**Containers**

| Component | Version | When it runs | License |
|---|---|---|---|
| Blueprint services: orchestrator, catalog indexer and retriever, memory retriever, guardrails, UI (built from this repository) | This release | Default deployment | [Apache 2.0](LICENSE). Built on [python:3.11-slim](https://hub.docker.com/_/python) and [NVIDIA Base Ubuntu](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/base) ([NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license)) |
| [Milvus](https://github.com/milvus-io/milvus) | v3.0.3 | Default deployment | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [etcd](https://github.com/etcd-io/etcd) | v3.7.2 | Default deployment | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [SeaweedFS](https://github.com/seaweedfs/seaweedfs) | 4.48 | Default deployment | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [nginx](https://nginx.org/) | 1.31.7-alpine | Default deployment | [BSD-2-Clause](https://nginx.org/LICENSE) |
| [OpenTelemetry Collector Contrib](https://github.com/open-telemetry/opentelemetry-collector-contrib) | 0.163.1 | Default deployment | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | 20.23.0 | Default deployment; runs only in your own environment and is not offered as a hosted service | [Elastic License 2.0](https://github.com/Arize-ai/phoenix?tab=License-1-ov-file) |
| [vLLM](https://github.com/vllm-project/vllm) | v0.32.0, v0.31.0, v0.20.0 | Optional local model deployment | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [Prometheus](https://github.com/prometheus/prometheus) | v3.15.0 | Optional GPU monitoring | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [Grafana](https://github.com/grafana/grafana) | 13.2.4 | Optional GPU monitoring | [AGPL-3.0](https://github.com/grafana/grafana/blob/main/LICENSE) |
| [NVIDIA DCGM Exporter](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/k8s/containers/dcgm-exporter) | 3.3.5-3.4.0 | Optional GPU monitoring | As published on its NGC page |

**Models**

| Model | Used for | License |
|---|---|---|
| [Nemotron 3.5 Super VL 120B A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) | Shopping assistant LLM | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) |
| [Nemotron 3.5 Embed 1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) | Catalog text and image embeddings | [OpenMDW-1.1](https://openmdw.ai/license/1-1/). Its [Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) and [Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) base models: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [Nemotron 3.5 Content Safety](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety) | Optional content moderation | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) |
| [Llama 3.1 NemoGuard 8B Topic Control](https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control) | Optional topic control | [Llama 3.1 Community License Agreement](https://www.llama.com/llama3_1/license/), for the model, built with Llama, and its [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) base model |
| [Nemotron 3 Nano Omni 30B A3B Reasoning](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8) | Optional video safety | [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |

**Services**

| Service | When it is used | Terms |
|---|---|---|
| [Visual Crossing Weather API](https://www.visualcrossing.com/) | Optional weather tool, off by default | [Visual Crossing Terms of Service](https://www.visualcrossing.com/weather-service-terms/) |

**Datasets**

| Dataset | Location | License |
|---|---|---|
| Product catalog and product images | `shared/data/`, `shared/images/` | [NVIDIA Asset License (v. August 15, 2025)](LICENSE-assets.txt). Adobe images are also subject to the Adobe Generative AI User Guidelines, the Adobe Generative AI Product Specific Terms and the Adobe General Terms of Use, as listed in that license |

### Notices and Disclaimers

This project will download and install additional third-party open source software projects, including vLLM, Milvus, LangChain, LangGraph, Deep Agents, NeMo Guardrails, FastAPI and React. Review the license terms of these open source projects before use, found in [LICENSE-3rd-party.txt](LICENSE-3rd-party.txt).

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.

---

<div align="center">

[Back to Top](#top)

</div>
