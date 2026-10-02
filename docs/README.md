# 📚 Documentation Hub

Guides and references for the Retail Shopping Assistant, for shoppers using the
app, developers deploying it, and contributors changing it.

New here? Read the [project README](../README.md) for what the blueprint is and
what it does, then come back and pick a path below.

## Choose a deployment path

The blueprint runs against either NVIDIA-hosted model endpoints or models you
host yourself. Everything else about the deployment is identical, so pick on the
basis of whether you have GPUs.

| Path | What it needs | Start at |
|------|---------------|----------|
| **NVIDIA-hosted endpoints** | An API key, no GPU | [Deployment Guide - Hosted Endpoints](DEPLOYMENT.md#-hosted-endpoints) |
| **Self-hosted models (vLLM)** | GPUs, plus a Hugging Face token for gated checkpoints | [Deployment Guide - Locally Hosted Models](DEPLOYMENT.md#-locally-hosted-models) |
| **Managed cloud GPU** | An NVIDIA Brev account | [Deploy on Brev](BREV.md) |

The self-hosted path is a hybrid: the language and text-embedding models move
onto your GPUs, while image embedding and the guardrail models stay on hosted
endpoints. [Deployment Options](DEPLOYMENT.md#%EF%B8%8F-deployment-options)
explains which roles can move and which cannot.

## Where to go by role

**I want to use the assistant.** Start with the [User Guide](USER_GUIDE.md) for
the chat interface, product search, cart, and image upload, and check the
[FAQ](USER_GUIDE.md#-faq) if something is unclear.

**I want to turn on safety checks.** [Guardrails](GUARDRAILS.md) covers the
whole feature: how it decides, what a failed check costs, what the shopper
sees, and how to read a decision afterwards. It ships off.

**I want to deploy it.** Work through the [Deployment Guide](DEPLOYMENT.md) end
to end. It owns prerequisites, both deployment paths, every configuration
setting, and troubleshooting. Then set up
[Observability](OBSERVABILITY.md) so you can see what a turn actually did.

**I want to call it from my own client.** The [API Reference](API.md) has the
request and response contracts, the streaming event framing, and per-service
endpoints.

**I want to change how the agent behaves.** Read
[Assistant Architecture](ASSISTANT_ARCHITECTURE.md) for the turn flow,
then [Skills](SKILLS.md) and
[Tools](TOOLS.md) for what the agent can load and
call. [AGENTS.md](../AGENTS.md) is the contributor and coding-agent guide to the
codebase layout.

**I want to use my own product catalog.**
[Catalog Architecture](CATALOG_ARCHITECTURE.md) covers the data format, the
sidecar that declares which fields become shopper-facing filters, and the
reindex-and-verify sequence. For a large catalog, or to index and search on a
GPU, read [Vector Search at Catalog Scale](VECTOR_SEARCH.md).

**I want to run it faster, or size it for load.** [Performance](PERFORMANCE.md)
covers the latency budget of a turn, finding the saturation point, and sizing
hardware.

## Document index

### Deploy and operate

| Document | What it covers |
|----------|----------------|
| [Deployment Guide](DEPLOYMENT.md) | Prerequisites, both deployment paths, production deployment, the full configuration reference, monitoring, troubleshooting, security, and scaling |
| [Deploy on Brev](BREV.md) | Step-by-step deployment to a managed NVIDIA Brev GPU instance |
| [Guardrails](GUARDRAILS.md) | The optional safety layer: the three judges, failure modes, configuration, what the shopper sees, and the service API |
| [Observability](OBSERVABILITY.md) | Reading a shopper's session turn by turn, opening one turn's trace, and finding what the model was told |
| [Performance](PERFORMANCE.md) | Latency budget, load methodology, vLLM saturation, and deployment sizing |
| [Monitoring stack](../monitoring/README.md) | Prometheus and Grafana for locally hosted model metrics |
### Build against it

| Document | What it covers |
|----------|----------------|
| [API Reference](API.md) | Endpoints, request and response models, streaming frames, guardrail reporting, and error handling |
| [UI](../ui/README.md) | React app structure, local development, and build |

### Use your own catalog

| Document | What it covers |
|----------|----------------|
| [Catalog Architecture](CATALOG_ARCHITECTURE.md) | The JSONL and sidecar format, declaring field roles so filters derive from data, advertised capabilities, validated retrieval, and replacing catalog data |
| [Vector Search at Catalog Scale](VECTOR_SEARCH.md) | The Milvus and SeaweedFS stack, choosing a CPU or GPU index, sizing, GPU search limits, and indexing a large catalog |

### Understand and change the agent

| Document | What it covers |
|----------|----------------|
| [Assistant Architecture](ASSISTANT_ARCHITECTURE.md) | Published catalog, turn flow, skill-to-tool mapping, and memory boundaries |
| [Skills](SKILLS.md) | Registered skills, runtime loading, and the markdown tuning workflow |
| [Tools](TOOLS.md) | Registered tools, risk classes, and per-skill access boundaries |
| [AGENTS.md](../AGENTS.md) | Service map, turn flow, file layout, and test commands for contributors |

### Learn by running

| Document | What it covers |
|----------|----------------|
| [Notebooks](../notebook/README.md) | Ordered walkthroughs: deploy and explore, observability, evaluation, trace capture, and performance measurement |
| [Testing and Evaluation](../tests/README.md) | Unit, integration, and Challenger/Judge evaluation workflows |

### Use the assistant

| Document | What it covers |
|----------|----------------|
| [User Guide](USER_GUIDE.md) | Chat interface, product search, cart management, image upload, troubleshooting, and FAQ |

### Project and process

| Document | What it covers |
|----------|----------------|
| [Project README](../README.md) | Overview, architecture, prerequisites, and quick start |
| [Contributing](../CONTRIBUTING.md) | Fork and pull request workflow, and sign-off requirements |
| [Security](../SECURITY.md) | Reporting a security vulnerability |
| [Changelog](../CHANGELOG.md) | Release history |

## Configuration and levers

Every setting has exactly one home, and the
[Deployment Guide](DEPLOYMENT.md) documents all of them:

- [Environment Variables](DEPLOYMENT.md#environment-variables) is the full list
  of variables, what each one does, and its default.
- [Configuration File](DEPLOYMENT.md#configuration-file) covers the YAML
  defaults in `shared/configs/`.
- [Model Routing](DEPLOYMENT.md#model-routing) covers which model serves each
  role and how to point a role at a different endpoint or a local model.
- [Model Sampling and Output Limits](DEPLOYMENT.md#model-sampling-and-output-limits)
  covers temperature and token caps.
- [Guardrails](GUARDRAILS.md) covers content safety, topic control, and
  multimodal safety, which ship disabled.

Environment variables override the shipped defaults, so you do not have to edit
YAML to change a setting. An unset or empty variable leaves the default in
place.

## Troubleshooting

| Symptom | Where to look |
|---------|---------------|
| A deployment will not start, or a service is unhealthy | [Deployment - Troubleshooting](DEPLOYMENT.md#%EF%B8%8F-troubleshooting) |
| The app runs but answers look wrong | [Observability](OBSERVABILITY.md) to see what the model was told |
| Turns are slow | [Performance](PERFORMANCE.md) |
| Something in the UI misbehaves | [User Guide - Troubleshooting](USER_GUIDE.md#%EF%B8%8F-troubleshooting) |
| Authentication or API key errors | [Deployment - NVIDIA Account Setup](DEPLOYMENT.md#nvidia-account-setup) |
| Search returns nothing for your own catalog | [Catalog Architecture - Replace the catalog](CATALOG_ARCHITECTURE.md#replace-the-catalog) |

## Getting help

Report bugs and request features through GitHub Issues, and ask questions in
GitHub Discussions, on the
[project repository](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant).
For security vulnerabilities, follow [SECURITY.md](../SECURITY.md) instead of
filing a public issue.
