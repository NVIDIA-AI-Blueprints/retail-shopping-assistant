# Unreleased

Substantial changes since 1.0.0. The architecture described in the 1.0.0 notes
below is historical; these are the current shapes.

## Changed

- 🤖 **Deep Agents runtime** replaces the bespoke LangGraph planner, retriever,
  cart, and chatter agents. One serving loop selects skills and calls tools,
  with grants rechecked at dispatch ([#123](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/123), [#210](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/210), [#255](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/255)).
- 💬 **Nemotron 3.5 Super VL** replaces Llama 3.1 70B Instruct as the shopping
  model, and reads photo and video uploads directly ([#261](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/261), [#264](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/264)).
- ☁️ **Hosted models on build.nvidia.com**: every model role defaults to
  `integrate.api.nvidia.com`, so one key runs the stack, and no NGC login is
  needed ([#290](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/290), [#319](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/319)).
- 🐳 **vLLM with Hugging Face checkpoints** replaces NIM containers for
  self-hosted models, and covers every model role: the chat model, text
  embedding, and the content safety, topic control, and video safety
  guardrails, on six 80 GB GPUs by default ([#263](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/263), [#284](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/284), [#309](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/309), [#316](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/316)).
- ⚙️ **One place for each setting**: every model URL and model name is set in
  `.env.example`; every other default lives once in `shared/configs/` or in
  code ([#264](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/264), [#276](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/276)).
- 🔍 **Catalog search** filters in the database instead of ranking the whole
  catalog, embeds with Nemotron 3 Embed 1B, and is built by a dedicated
  `catalog-indexer` service. Image embedding is off by default: uploads are
  read by the vision model and searched as text ([#188](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/188), [#189](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/189), [#208](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/208), [#214](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/214)).
- 🗄️ **Storage**: SeaweedFS replaces MinIO, Milvus moves to 3.0 with an opt-in
  GPU index, and the memory service is SQLite only ([#271](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/271), [#293](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/293), [#312](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/312), [#313](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/313)).
- 🖥️ **UI** moves from Create React App to Vite and is served as a static build
  behind nginx ([#273](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/273), [#280](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/280)).
- 🏷️ **Chain Server renamed to orchestrator** across services, paths, and docs
  ([#299](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/299)).
- 🔐 **API**: every request carries a conversation ID and a cart ID ([#285](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/285)).
- 🧹 **Code structure**: the orchestrator is split into `runtime/` and `tools/`
  packages, single-use modules are merged, and dead code is removed
  ([#243](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/243), [#249](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/249), [#294](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/294), [#296](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/296), [#297](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/297), [#304](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/304)).

## Added

- 📚 **Skills and tools registries**, with per-skill tool grants enforced at
  dispatch rather than by prompt instruction alone.
- 🧭 **Multi-item search**: several product roles, and their stock, in one
  round trip, including roles the shopper implied but did not name
  ([#146](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/146), [#149](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/149)).
- 📏 **Sizes**: a size the catalog does not stock gets an honest answer, a size
  change is one call, and the cart keeps one line per product and size
  ([#156](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/156), [#171](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/171), [#206](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/206), [#230](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/230)).
- 🚫 **Not-carried answers**: the assistant says when the shop does not carry
  something instead of substituting a near match ([#170](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/170), [#234](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/234), [#237](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/237)).
- 🔢 **Numbered products**: what the shopper sees is numbered, so "the second
  one" resolves to the product they meant ([#242](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/242), [#244](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/244)).
- 🌦️ **Weather-aware styling**, optional and off by default: a destination's
  forecast shapes what the assistant suggests ([#155](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/155), [#227](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/227), [#248](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/248)).
- 🛍️ **Storefront**: an editable cart, a compact product grid with Show all,
  and live status while a reply is pending ([#157](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/157), [#291](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/291), [#301](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/301)).
- 💾 **Durable conversation turns**, so a turn's cart effects and raw history
  survive a restart.
- 🔒 **Guardrails** as an isolated service: content safety, topic control, and
  video safety. On by default for API requests; the UI toggle starts off.
  `GUARDRAILS_AVAILABLE=false` deploys without them, hosted or local: no
  guardrail model is started or called, and the UI hides the toggle
  ([#266](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/266), [#285](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/285)).
- 📊 **Observability**: per-turn spans, whole-conversation Phoenix sessions,
  and NeMo Relay model spans, on in the example profile ([#158](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/158), [#209](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/209), [#254](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/254)).
- 🔍 **Data-derived catalog filters**, published from the catalog and its schema
  sidecar rather than hardcoded ([#192](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/192)).
- 🧪 **Evaluation**: scripted replay journeys checked against the cart, and
  Challenger and Judge workflows ([#199](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/199), [#222](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/222), [#270](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/270)).
- 📓 **Notebooks**: Getting Started, Observability, Evaluation, Capture Traces,
  and Performance Measurement ([#253](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/253), [#257](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/257), [#260](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/260)).
- 📈 **Performance measurement** tooling and a step-by-step guide
  ([#213](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/213), [#218](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/218)).
- 🏭 **Production guidance**: horizontal scaling readiness and a Taking to
  Production guide ([#207](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/207), [#289](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/289)).
- 🛡️ **Security hardening**: shopper content is not logged, text from outside
  the system is marked as data before a model reads it, dependencies are
  patched, Bandit findings are cleared, and SonarQube runs in CI ([#221](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/221), [#272](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/272), [#278](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/278), [#279](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/279), [#303](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/303)).

## Removed

- Shopper profiles ([#292](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/292)).
- The PostgreSQL memory backend ([#293](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/293)).
- Unused memory endpoints ([#294](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/294)).
- NIM containers for self-hosted models ([#284](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/pull/284)).

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
