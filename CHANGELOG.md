# Unreleased

Substantial changes since 1.0.0. The architecture described in the 1.0.0 notes
below is historical; these are the current shapes.

## Changed

- 🤖 **Deep Agents runtime** replaces the bespoke LangGraph planner, retriever,
  cart, and chatter agents. One serving loop selects skills and calls tools,
  with grants rechecked at dispatch.
- 💬 **Nemotron 3.5 Super VL** replaces Llama 3.1 70B Instruct as the shopping
  model, and reads image and video uploads directly.
- 🐳 **vLLM with Hugging Face checkpoints** replaces NIM containers for
  self-hosted models, and covers every model role: the chat model, text
  embedding, and the content safety, topic control, and video safety
  guardrails, on six 80 GB GPUs by default.
- ⚙️ **One source for every setting**: Compose and the env templates pass
  settings through empty, and each default lives once in `shared/configs/` or in
  code. An unset or empty variable leaves the shipped default in place.

## Added

- 📚 **Skills and tools registries**, with per-skill tool grants enforced at
  dispatch rather than by prompt instruction alone.
- 💾 **Durable conversation turns**, so a turn's cart effects and raw history
  survive a restart.
- 🔒 **Optional guardrails** as an isolated service: content safety, topic
  control, and video safety. Off by default, enabled per request or per
  deployment, and reported back per turn. `GUARDRAILS_AVAILABLE=false` deploys
  without them, hosted or local: no guardrail model is started or called, and
  the UI hides the toggle.
- 📊 **Observability**: per-turn spans, Phoenix session views, and optional
  NeMo Relay for richer model spans.
- 🔍 **Data-derived catalog filters**, published from the catalog and its schema
  sidecar rather than hardcoded.
- 🧪 **Evaluation harness** with Challenger and Judge workflows.

# Retail Shopping Assistant 1.0.0 (03 September 2025)

## 🎉 First Release

This is the initial release of the NVIDIA AI Blueprint: Retail Shopping Assistant, a comprehensive AI-powered shopping experience demonstrating multi-agent architecture with LangGraph.

## 🚀 New Features

- 🤖 **Multi-agent Architecture** - LangGraph-orchestrated agents for intelligent shopping assistance
- 🖼️ **Visual Search** - Image-based product search using NVIDIA NV-CLIP model  
- 🛒 **Smart Cart Management** - Conversational shopping cart operations
- 💬 **Conversational AI** - Natural language interactions powered by Llama 3.1 70B Instruct
- 🔒 **Content Safety** - Built-in moderation with Llama 3.1 NemoGuard 8B
- ⚡ **Real-time Streaming** - Live response generation for seamless UX
- 📱 **Modern UI** - Responsive React interface with Material-UI and Tailwind CSS
- 🔍 **Vector Search** - Milvus-powered similarity search for product recommendations
- 🐳 **Containerized Deployment** - Docker compose setup with NVIDIA NIM containers
- ☁️ **Flexible Deployment** - Support for local GPU or cloud API endpoints
- 📊 **Product Catalog** - Comprehensive retail product dataset

## 📋 Pull Requests Included in v1.0.0

### Recent Improvements
- Replace langchain with OpenAI library in ChatterAgent for consistency by @antoniomtz in [#78](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/78)
- Fixed final deployment guide link by @nvidiacbrissette in [#77](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/77)
- Update welcome message and splash image by @antoniomtz in [#76](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/76)
- Made notebook documentation links point to GitHub over local files by @nvidiacbrissette in [#75](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/75)
- One last link update by @nvidiacbrissette in [#74](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/74)
- Changed some bad links by @nvidiacbrissette in [#73](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/73)
- Merge product data by @antoniomtz in [#72](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/72)
- Update license by @antoniomtz in [#71](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/71)
- Update license terms by @antoniomtz in [#70](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/70)
- Update doc requirements by @antoniomtz in [#69](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/69)
- Update notebook with latest instructions by @jahubba in [#68](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/68)
- Update notebook by @antoniomtz in [#67](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/67)
- Update url path by @antoniomtz in [#66](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/66)
- Update notebook paths by @antoniomtz in [#64](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/64)

## 👥 Contributors

- **@nvidiacbrissette** ([Christopher Brissette](https://github.com/nvidiacbrissette))
- **@jahubba** ([Jason Hubbard](https://github.com/jahubba))
- **@antoniomtz** ([Antonio Martinez](https://github.com/antoniomtz))

**Full Changelog**: https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/commits/v1.0.0

**Tag:** [v1.0.0](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/releases/tag/v1.0.0)
