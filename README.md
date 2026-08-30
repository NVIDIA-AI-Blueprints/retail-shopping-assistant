<a id="top"></a>
# 🛍️ NVIDIA AI Blueprint: Retail Shopping Assistant

<div align="center">

![NVIDIA Logo](https://avatars.githubusercontent.com/u/178940881?s=200&v=4)

**AI-powered retail shopping assistant with Deep Agents SDK orchestration**

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
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contribution Guidelines](#contribution-guidelines)
- [Community](#community)
- [References](#references)
- [License](#license)

## Overview

The Retail Shopping Assistant is an AI-powered blueprint that provides a comprehensive interface for an intelligent retail shopping advisor. The chain server uses the Deep Agents SDK as the assistant harness over deterministic shopping tools, with SSE-framed responses, image-based search, optional VLM media perception, and intelligent shopping cart management.

### Key Features

- 🤖 **Intelligent Product Search**: The assistant translates natural language
  into catalog queries and advertised filters; the catalog performs only
  deterministic embedding retrieval and ranking
- 🛒 **Deterministic Cart Management**: Read, add, remove, update quantities,
  and compute subtotals through typed tools
- 🧠 **Durable Turn Transcript**: A single memory-service SQLite replica
  starts every turn before agent work, finalizes its terminal outcome, and
  exactly replays finalized requests from ordered shopper/assistant records;
  rotating attempt tokens reject late finalizers after interrupted-turn recovery
- 💭 **Durable Product Continuity**: Finalized product-card output becomes
  ordered `candidate_set_presented` evidence in SQLite; a typed resolver can
  recover one exact earlier product or require clarification without another
  catalog search or model call
- 👤 **Representative Shopper Picker**: Five immutable, database-backed
  shoppers mirror the committed live-evaluation behavior profiles; the UI can
  select one or Guest and inspect its type, behavior, and ZIP. The selected ID
  is bound to the durable conversation and resolved into compact soft guidance
- 🌦️ **Dormant Weather Contract**: A disabled-by-default, directly testable
  daily forecast tool accepts a five-digit US ZIP plus today, one exact date,
  or an inclusive date range. It is not registered with the shopper agent and
  does not yet influence conversation or styling
- 📚 **Enforced Shopper Skills**: Every turn first semantically selects and
  fully loads the smallest applicable skill set; each selected `SKILL.md`
  declares its role and tool grants, only their grant union becomes
  model-visible, and dispatch rechecks the grant before execution
- 🖼️ **Visual Search**: Upload images to find similar products
- 🎥 **Optional VLM Media Perception**: Enable a VLM role to analyze image and video uploads in shopping context
- 💬 **Conversational AI**: Natural language interactions
- 🔒 **Configurable Content Safety**: Built-in moderation and safety checks are on by default and can be disabled per request or config
- ⚡ **SSE Response Stream**: Event-stream response framing for chat clients; token-level Deep Agents streaming is a follow-up after the harness migration
- 📊 **Inference Visibility**: Model names, call counts, and token usage, with
  detailed ordered agent/tool diagnostics available only when explicitly
  enabled for a trusted operator or evaluation deployment
- 📱 **Responsive UI**: Modern, mobile-friendly interface

### Architecture

![Shopper Deep Agent architecture](docs/images/shopper-agent-architecture.svg)

[Open the architecture diagram at full size](docs/images/shopper-agent-architecture.svg).

The application follows a microservices architecture:
- **Chain Server**: Deep Agents SDK orchestration with five registered shopper
  skills, a required per-turn activation phase, an eleven-tool registry with
  deterministic per-skill binding, capability-derived search schemas, bounded
  search-schema repair, a category-aware no-I/O availability stub for known
  product refs, a no-I/O active-promotions stub, typed same-conversation product
  resolution, grounded response assembly, a configurable Deep Agents execution
  deadline, a request-scoped process-local checkpointer, and an unregistered
  provider-neutral weather client/tool boundary
- **Catalog Retriever**: Generative-LLM-free text/image embedding search, hard
  filtering, normalized COSINE relevance scores, and deterministic result
  ranking
- **Memory Retriever**: Ordered durable turns with start/finalize and exact
  replay, bounded recent-turn reads, typed prior-skill continuity, presented-
  product events and a compact reference index, stable cart-line IDs, atomically
  idempotent add/remove/quantity mutations, an immutable five-row representative
  shopper registry, atomic conversation/profile binding, and request-scoped
  database sessions; standard Compose exposes its host port on loopback only
- **Guardrails**: Content safety and moderation
- **UI**: React-based frontend interface with Guest/representative-shopper
  selection

The UI sends only the selected profile ID; Guest omits it. Durable turn start
resolves the server-owned row and prevents one conversation from switching
between Guest and another shopper or between two shoppers. The model receives
one small current-turn block with `shopper_type`, exact `behavior`, and
`saved_zipcode`. This is soft interaction/style guidance only: explicit shopper
instructions win, and a profile cannot invent a budget, product requirement,
cart action, product fact, skill choice, or tool permission. Changing the
selection clears visible chat/product state and rotates the browser-scoped
session, conversation, and cart identities.

The Slice 3 weather boundary is intentionally dormant. Direct callers can
construct a typed Visual Crossing adapter for a five-digit US ZIP and an
optional exact date or inclusive date range, but the wrapper is absent from the
Deep Agents tool registry, skill grants, prompts, request context, FastAPI, and
UI. `WEATHER_ENABLED` defaults to `false`; no API key or provider request is
needed for ordinary startup, health checks, shopper turns, or offline tests.
Enabling direct construction requires `WEATHER_API_KEY` in the chain-server
environment. The key is not stored in YAML or an image, and this integration
does not require MCP.

Every turn still makes a fresh semantic skill-selection decision. The previous
turn's selected skill names are persisted with its durable output and supplied
to the next activation model step only as a read-only continuity signal; they
do not force routing or authorize tools.
If the model selects an invalid skill composition, it receives the typed reason
and one correction attempt. A second invalid selection returns a deterministic
clarifying question without running catalog or commerce tools. Multiple
activation calls in one response execute none and clarify immediately.
Conversation context still matters: a terse item-only follow-up inside an
active outfit-building or style-led single-piece thread remains an
`outfit-styling` task.
`search_catalog_tool` exposes one flat, capability-derived executable search
schema. The model cannot submit a clarification or catalog-absence result
through that tool: it must select faithful advertised values, use one faithful
advertised parent category for a shopper-named type that is not separately
advertised, or ask one concise clarification directly without a tool call.
Parent-category results are explicitly presented as closest alternatives under
their actual catalog categories.
When a catalog search needs repair, the runtime assigns one total repair to the
full normalized, model-authored `requested_product_type` phrase. It does not
reconstruct alternatives, negation, ordering, or comparisons from shopper
prose. A schema correction or a fresh
constraint-provenance review can consume that single budget; constraint feedback
returned by an in-flight schema repair closes the loop for synthesis rather than
opening another repair. Distinct advertised siblings never count as the same
repair scope. The isolated repair receives the capability-derived typed
`search_catalog_tool`, compact server-generated Catalog capabilities, the
current shopper message, bounded sanitized validator feedback, and the complete
active shopper-skill instructions. Only that search tool is available, parallel
calls are disabled, and the repair may either submit one corrected search or
signal that clarification is needed by returning no tool call. The server
discards that model prose and emits the fixed clarification `Could you clarify
the product type or requirement you want me to use?`. If another requested
search scope already succeeded, its deterministic grounded products are kept
before that clarification. If another shopping tool already completed, the
existing grounding editor preserves that evidence with the fixed clarification.
The base runtime prompt, invalid AI/tool history, and prior conversation history
are absent. For a native
tool-transport failure, the
requested scope is locked only when current or recent shopper text grounds it;
an ungrounded model-generated scope may be corrected. A rejected change to a
grounded scope is removed before execution and appears in `agent_diagnostics`
with the `repair_scope_changed` reason.
Native validation feedback contains only rejected top-level field names; raw
Pydantic `input_value` metadata and free-form `requested_product_type` text are
never copied into the authoritative repair message. After activation, the
server rejects a model response containing more than one shopping tool call,
in addition to requesting `parallel_tool_calls=false`.
When strict request validation fails while its constraint object validates
independently, the handler keeps those capability-validated advertised
constraints as a private immutable repair boundary and includes their exact
finite object in validator feedback. The repaired call must preserve them; the
strict handler rejects drift instead of overwriting model output. Free-form
rejected arguments remain excluded. Repair middleware never restores or
rewrites taxonomy, constraints, requested type, or search mode. It may restore
only the structural `scope_complete` flag, which is reported by name in bounded
`restored_fields` diagnostics.
When validation rejects an incoherent open-role search, the same repair
feedback carries the shopper-provenance rule: a shopper-named role must retain
the shopper's noun or umbrella, while a genuinely open role chooses and names
one advertised subtype.
When native validation rejects `required_constraints`, the repair receives the
typed search tool and compact Catalog capabilities needed to select valid
values. The free-form query, guidance, and requested scope remain excluded from
native validator feedback; a shopper-grounded scope is compared privately.
Middleware does not preserve unvalidated taxonomy or constraints by rewriting
the repaired call. Malformed or nonempty
free-form `unadvertised_requirements` arguments are never restored. A native
schema-invalid call containing one closes without repair. A
schema-valid, genuinely open role may still use the
bounded review for a proposed inferred requirement.

The resolved chain-server agent stack remains `deepagents==0.6.12`,
`langchain==1.3.11`, `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`.
`orjson==3.11.5` is pinned in every service requirement set that resolves it as
the last upstream release limited to the project's Apache-2.0/MIT license
policy. Redis checkpoint packages remain absent; the runtime supports only
process-local `CHECKPOINT_STORE=memory`. Each graph thread is request-scoped
with a collision-safe pair of conversation ID and request ID, deleted after
successful durable finalization, and retained only when finalization fails.
Deep Agents model-stage execution defaults to one 45-second deadline shared by
the graph and grounding editor. A graph timeout is captured as `agent_timeout`,
clears unsent products, finalizes the durable turn as failed, releases the
durable conversation turn, and then deletes its request checkpoint. The
grounding editor receives only the remaining time. Its timeout is finalized as
failed with `grounding_timeout`: search-only turns use the existing deterministic
catalog renderer, while every other turn returns a fixed retry/cart-check
response instead of the unverified draft. Editor errors and empty or whitespace-
only output follow the same fail-closed response rule with `grounding_error`.

For the serving-agent flow, see
[Shopper Agent Architecture](docs/SHOPPER_AGENT_ARCHITECTURE.md). The
[Documentation Hub](docs/README.md) links the detailed contracts and operations
guides.

### Catalog lifecycle and capability publishing

1. At startup, the catalog service loads `enriched_products.jsonl` and its
   field-role sidecar into one validated snapshot.
2. That snapshot supplies embedding documents, product details, filters, and
   the live contract at `http://localhost:8010/capabilities`.
3. On its first successful fetch, the chain server caches one process-wide
   contract shared by all sessions. Its aggregate endpoint at
   `http://localhost:8009/capabilities` returns the cached catalog contract with
   the other runtime capabilities.
4. Cached capabilities generate `search_catalog_tool`'s flat schema:
   `semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
   `required_constraints`, `scope_complete`, and optional `search_mode`.
   Taxonomy values, hard-filter properties and enum values, typed numeric range
   shape, and search-mode values come from the active contract. This typed schema
   deliberately omits cross-field validators; the handler applies a separate
   strict semantic model to the same payload. Invalid individual values fail at
   the tool boundary, while cross-field failures reach capability-aware handler
   validation and can receive one bounded repair. The agent semantically selects
   exact advertised values; deterministic chain code validates and maps the
   selection against the capability-owned
   exact category/subcategory relationships and returns corrective feedback for
   incoherent combinations. Each call covers at most one category. For a
   genuinely open request, the model selects exactly one advertised subcategory
   and names it in `requested_product_type`. A named shopper scope must retain
   the shopper's noun or umbrella. A transport repair cannot change that scope
   when it is grounded in shopper text; an ungrounded model-generated scope may
   be corrected.
   Every text search carries
   `requested_product_type`: the shortest product noun or true umbrella from
   the shopper's current turn or direct antecedent. It excludes color,
   material, fit, occasion, weather, and style modifiers. It is provenance, not
   taxonomy or ranking text, and is `null` only for image-only search.
   Validation can bind the longest exact advertised suffix in a
   modifier-bearing model phrase (`waterproof boots` to `boots`), but disables
   that shortcut for explicit alternatives containing `and`, `or`, `/`, or
   `&`.
   Thus `closed shoes or boots` remains model-owned alternative or umbrella
   reasoning rather than being collapsed to `boots`. The model owns all
   alternative, comparison, ordering, and negation semantics. When it submits
   multiple advertised subcategories from one category through the typed
   taxonomy field, the valid request remains one catalog execution; its
   candidate window expands for that selection, then rank-preserving selection
   keeps one returned candidate per selected subcategory when available before
   trimming to the configured result count. The runtime does not derive that
   selection from the shopper's raw text. Each search also requires
   `shopper_guidance`: one nonempty, product-agnostic
   sentence authored before retrieval under the active skill to connect the
   selected role to the shopper's goal or direct antecedent. Empty guidance is
   valid only for image-only search.
5. If a shopper-named type is not separately advertised but one faithful
   advertised parent category can be selected, the model searches that category
   once while preserving the shopper's type as semantic direction. The response
   discloses the broader scope and keeps each result's actual category. If
   neither a direct type nor one faithful parent can be selected, the assistant
   asks one concise clarification directly. It makes no tool call, retrieval, or
   catalog-absence claim. An unsupported
   modifier does not erase an advertised product type. A directly stated
   must-have missing from the generated schema is placed in
   `unadvertised_requirements`, while preference, styling, occasion, weather,
   and anchor context remain in the semantic query. A product type never belongs
   in `unadvertised_requirements`. Every such requirement on a shopper-stated
   product scope ranks retrieval and is returned to the shopper as unconfirmed,
   including when the model uses a synonym rather than the shopper's exact
   wording. It never becomes a hard filter and never suppresses the search. The bounded
   constraint-provenance review is reserved for a proposed inferred requirement
   on a genuinely open role when its shared repair budget remains. The review
   freezes `requested_product_type`, taxonomy, `scope_complete`, `search_mode`, and every
   advertised hard constraint. Within that preserved hard scope, it may correct
   only the soft `semantic_query`, the reviewed `unadvertised_requirements`
   lane, and its associated `shopper_guidance`; the requirement is either
   replaced with the shopper's shortest exact wording or removed. Exact wording is kept as an
   unenforceable requirement: it ranks the search and is disclosed. Removal also scrubs the corresponding
   product-attribute claim from `shopper_guidance`. When a runtime semantic
   open-role schema repair removes
   its proposed inferred requirement, runtime replaces the submitted pre-search
   guidance with neutral generic guidance for the selected role. Unresolved
   provenance after that review, or constraint feedback after the scope already
   used its schema repair, fails safe and closes the loop for synthesis. A
   successful partial search may advance to a new role with its own single
   repair opportunity; the configured
   turn cap remains three searches. When a successful or zero-result search
   consumes the final configured slot, its result records
   `SEARCH_BUDGET_EXHAUSTED`; the next model step omits only
   `search_catalog_tool`. This prevents a fourth search while preserving product
   details, availability, cart work, and honest partial synthesis.
6. The catalog validates executable requests again, generates embeddings,
   applies hard filters, and ranks results. It performs no shopper-language
   interpretation or chat/completion call.

For a singleton exact taxonomy value, deterministic validation requires
`requested_product_type` to match the advertised taxonomy value. The semantic
query is independent soft ranking direction and need not repeat the taxonomy
noun. Successful search evidence preserves it as a private ranking preference.
For a completed successful search-only turn, the runtime allows one final
tools-disabled synthesis under the active skill and then grounds that draft
against tool-role evidence. The pre-retrieval `shopper_guidance` and active
skill's static `response_guidance` support deterministic fallback when synthesis
or editing cannot produce an answer. If the shopper's goal depends on a
material, fit, comfort, durability, care, weather, or other functional property
that the evidence does not confirm, final grounding states that gap and presents
the candidates as the closest catalog or styling direction rather than as
proven suitable. Deterministic fallback ends with the same generic disclosure.
Before fallback guidance is serialized, a
narrow runtime scrub replaces documented unsupported outdoor/weather guarantee
language with neutral guidance for the selected role. This changes only response
framing; the semantic query, taxonomy, constraints, and executed search remain
unchanged. The scrub includes outdoor-surface or outdoor-walking claims and
constructions such as "handle rain," "work well for outdoor surfaces," or
"stay secure for outdoor walking," plus `wet conditions` and "works well in wet
weather/conditions." Candidate results, filters, and the assistant draft are not
rewritten into guidance after retrieval. Deterministic code
separately lists every returned candidate with its name, price, category,
and only the confirmed filters from that candidate's search. For multi-role
results, each guidance sentence is grouped with the products from the search
that produced it. Candidate groups deduplicate by `product_ref`, not display
name: the same catalog product appears once, while distinct products that share
a name remain distinct. Mixed-outcome turns retain every successful product
group when a later scope has an unsupported requirement and append the honest
gap. The fixed unsupported-requirement response is used only when that rejection
is the sole current-turn business-tool outcome;
otherwise the other outcome remains available for rendering or synthesis. If
successful search evidence remains incomplete, the renderer adds a neutral
offer to continue with the next requested piece or search scope.
Separate searches are never flattened into one global filter claim. Zero-result
evidence retains its exact taxonomy and filter scope and cannot support a claim
about a different product type or the whole catalog.
Operator diagnostics include bounded `catalog_scope_outcomes` for zero-result
scopes. The grounding boundary keeps current-turn and prior-turn
tool-role evidence separate, so earlier results can resolve references but
cannot prove that a new search or cart mutation ran. If every current-turn
business call is a rejected catalog search and no current product evidence
exists, the runtime returns a fixed retry response before model-based response
editing; prior evidence cannot be presented as results from the rejected search.

Final-response extraction ignores tool messages, assistant tool-call messages,
and internal skill-activation markers. If a completed graph contains no
shopper-facing answer, the runtime returns a safe retry response and records the
termination reason as `incomplete_agent_response` rather than exposing internal
content.

At turn start, the memory service returns a bounded set of prior raw
shopper/assistant turns eligible for model context, the authoritative cart, and
a service-issued attempt token. Blocked turns remain durable and exactly
replayable but are excluded by both the service projection and chain prompt
formatter; abandoned turns are also excluded by the formatter. Only the latest
abandoned turn can reopen; reopening retains its request identity but rotates
the attempt token, so a late finalize cannot overwrite the retry. Those recent
turns replace the legacy rolling context blob, while the
memory service also returns a compact index of products actually presented as
ordered cards on earlier turns. When a needed product is not established in the
current request, the selected discovery, styling, or cart skill may make one
typed batch resolution call. An exact single match becomes request-local
evidence for details, availability, or cart add; zero or multiple matches require
clarification and never authorize a guess. Resolution is limited to the current
conversation and does not add fuzzy matching, embeddings, cross-conversation
memory, preference/sentiment memory, or catalog-revision revalidation.

LangGraph `MemorySaver` now holds only one request's working graph state under a
collision-safe pair of conversation ID and request ID. It is deleted only after
durable finalization succeeds; a finalize failure preserves that checkpoint.
The compact historical-product index is capped at 16,384 characters, and its
typed batch resolver can run at most once per turn. Caller-supplied persona data
is not accepted as turn context. The fixed representative shoppers use a typed,
bounded server-owned registry and an atomic turn-start binding; only the
resolved three-field snapshot enters the current model input. Guest turns carry
neither that snapshot nor profile-specific prompt rules.

Catalog values are never copied into agent or catalog code. After replacing the
JSONL or sidecar, restart and verify the catalog service first, then restart and
verify the chain server so its process-lifetime cache matches the new snapshot.
See [Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md) for the complete flow
and [Catalog Schema and Filters](docs/CATALOG_FILTERS.md) for the sidecar rules.
The exact published response is documented in
[Catalog Retriever capabilities](docs/API.md#catalog-retriever-get-capabilities).

## Get Started

### Prerequisites

- **Docker**: Version 20.10+ with Docker Compose plugin
- **Python**: Host Python for deployment helpers. From the cloned repo, install
  deploy-helper dependencies with:
  ```bash
  python -m pip install --user -r requirements-deploy.txt
  ```
- **NVIDIA NGC Account**: For API access ([Get API Key](https://ngc.nvidia.com/))
- **Hardware**: 4x H100 GPUs (preferred) or 4x A100 GPUs (minimum) for local deployment, or cloud access

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
   cd retail-shopping-assistant
   ```

2. **Authenticate with NVIDIA Container Registry**:
   ```bash
   docker login nvcr.io
   ```
   Use `$oauthtoken` as the username and your NGC API key as the password.

3. **Install host deploy-helper dependencies**:
   ```bash
   python -m pip install --user -r requirements-deploy.txt
   ```

4. **Create and source an environment profile**:
   ```bash
   cp .env.example .env
   $EDITOR .env
   source .env
   ```

   Set `NVIDIA_API_KEY` in the file.

   Two templates are provided, and they differ by one role:

   | template | app LLM | everything else |
   |---|---|---|
   | `.env.example` | hosted endpoint | hosted |
   | `.env.local-llm.example` | **local NIM**, on your GPUs | hosted |

   Use the second when you need what a hosted endpoint cannot give you: no rate
   limit, and the model's own metrics. Only the LLM moves, so a run against one
   and a run against the other differ by the model serving the shopper and
   nothing else — which is what makes them comparable. See
   [Running the LLM locally](#running-the-llm-locally) below.

   Copy a template rather than editing it. Every `.env.*` is ignored except the
   two templates, so your filled-in copy stays out of git along with its keys. The env file is a sourceable shell file;
   sourcing it also sets `COMPOSE_DISABLE_ENV_FILE=1` so Docker Compose uses
   the exported shell environment instead of auto-parsing repo-root `.env`.
   `CHECKPOINT_STORE=memory` is the only supported graph-checkpoint
   configuration. Graph checkpoints disappear on chain-server restart and are
   not shared across replicas, and neither matters: a checkpoint is keyed on
   `(conversation_id, request_id)`, so it belongs to one request and two turns
   of a conversation can never share one. **No session affinity is needed** --
   any chain-server replica can serve any turn, because everything that carries
   across turns lives in the memory service. Revisit only if turns must survive
   a restart mid-turn.

   Compose stores the memory-service SQLite database at `/data/context.db` on
   the `memory-data` named volume. SQLite is a single-writer file on a
   single-mount volume, so it is the one-replica choice; PostgreSQL is selected
   by `MEMORY_DATABASE_URL` when more than one replica is needed. See the
   [Deployment Guide](docs/DEPLOYMENT.md).

   Weather remains disabled by default. Leave `WEATHER_ENABLED=false` and
   `WEATHER_API_KEY` empty for the current shopper experience. An operator
   testing the dormant client directly may set the key only in the ignored
   `.env`, process environment, or deployment secret store.

5. **Validate and deploy**:
   ```bash
   python scripts/model_config.py show --validate
   python scripts/model_config.py deploy --build
   ```

   The helper prints resolved endpoints without printing API keys. By default,
   `shared/configs/models.yaml` uses NVIDIA Build hosted endpoints for the
   app LLM, text embeddings, image embeddings, and guardrails, and starts no
   local NIM containers. The `vlm` role uses a hosted endpoint by default for
   image/video media understanding in addition to image embedding search; set it
   to `disabled` in `models.yaml` when that capability should be off.

   For local NIMs, edit the desired model roles in
   `shared/configs/models.yaml` to `source: local_nim`, then run:
   ```bash
   # Set LOCAL_NIM_CACHE in the sourced env profile first.
   mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
   python scripts/model_config.py show --validate
   python scripts/model_config.py deploy --build
   ```

   Model routing lives in `shared/configs/models.yaml`.

6. **Confirm the product catalog is indexed**:
   ```bash
   curl -s http://localhost:8010/ready
   ```

   Compose indexes for you: the `catalog-indexer` service runs once, builds the
   index and exits, and `catalog-retriever` waits for it to succeed before it
   starts. Bringing the stack up again re-runs it, so a change to the catalog
   data, schema, or embedding model is picked up by `docker compose up -d`.

   Serving containers do not index themselves: rebuilding an index begins by
   dropping the collection, which is safe when one process does it and
   destructive when two do, so it is a separate one-shot service rather than
   something that happens on start.

   To rebuild without cycling Compose, run it directly. Safe to repeat: it
   checks the catalog fingerprint first and does nothing when the index is
   already current.

   ```bash
   docker compose exec catalog-retriever python -m app.index_catalog
   ```

   Check `/ready` here, not `/health`. A catalog container with no matching
   index is alive but deliberately serving no traffic, so `/health` returns 200
   while `/ready` returns 503. Searches in that state return no products rather
   than an error, and a shopper is told the catalog is empty.

7. **Access the application**: Open your browser to `http://localhost:3000`

### Running the LLM locally

The app LLM can be served from your own GPUs as a NIM instead of a hosted
endpoint. Nothing else moves: embeddings, the VLM, guardrails, the judge and
the challenger stay where they were.

```bash
cp .env.local-llm.example .env.local-llm
$EDITOR .env.local-llm            # NVIDIA_API_KEY, and LOCAL_NIM_CACHE if not $HOME/.cache/nim
source .env.local-llm

mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
docker compose -f docker-compose-nim-local.yaml up -d nemotron

curl -s http://localhost:8000/v1/models     # ready when this answers

docker compose up -d
docker compose exec catalog-retriever python -m app.index_catalog
```

The first start pulls roughly 120GB into `LOCAL_NIM_CACHE` and takes a long
while; later starts read the cache. The NIM reserves two GPUs, `device_ids`
`'0'` and `'1'` in `docker-compose-nim-local.yaml`. Make the cache writable
before the first start — the container runs as `${UID}`, and a first start that
cannot write re-downloads on every restart.

No tracked configuration changes. The environment is read before
`shared/configs/models.yaml`, so `LLM_BASE_URL` and `LLM_MODEL` are enough and
`models.yaml` is left alone.

**What this gets you.** The NIM serves vLLM's Prometheus metrics on its own
port, with nothing to enable:

```bash
curl -s http://localhost:8000/metrics | grep -E '^vllm:'
```

Tokens in and out, time to first token, queue depth, KV-cache utilisation, and
running and waiting sequences. `vllm:num_requests_waiting` is the one that says
whether the model is the bottleneck rather than the application — a question a
hosted endpoint cannot answer at all, and one the load tooling otherwise has to
infer. Nothing scrapes these yet; the collector carries a traces pipeline only.

**Going back to the hosted endpoint** is sourcing the other profile and
restarting the chain server:

```bash
source .env
docker compose up -d --force-recreate chain-server
docker compose -f docker-compose-nim-local.yaml stop nemotron
```

8. **Stop the containers**:

   **Application services**:
   ```bash
   docker compose -f docker-compose.yaml down
   ```

   **Local NIM services, if `models.yaml` started any**:
   ```bash
   docker compose -f docker-compose-nim-local.yaml down
   ```

For detailed installation instructions, see [Deployment Guide](docs/DEPLOYMENT.md).

## Deploy on NVIDIA Brev

For a streamlined cloud deployment experience, you can deploy the Retail Shopping Assistant on **NVIDIA Brev** using GPU Environment Templates (Launchables):

**[NVIDIA Brev Deployment Guide](docs/BREV.md)** - Complete step-by-step instructions for deploying on Brev

### Why Choose NVIDIA Brev?

- **One-Click Deployment**: Pre-configured GPU environments with automatic setup
- **Managed Infrastructure**: No need to manage servers or GPU clusters
- **Secure Access**: Built-in secure tunneling for web interface access  
- **Flexible Resources**: Choose from H100, A100, and other GPU configurations
- **Cost-Effective**: Pay only for actual usage time

The Brev deployment guide walks you through the entire process from creating a Launchable to accessing your fully functional retail shopping assistant.

## Documentation

- **[Project Status](STATUS.md)**: Current implementation, verification, quality qualification, and remaining risks
- **[User Guide](docs/USER_GUIDE.md)**: How to use the application
- **[API Documentation](docs/API.md)**: Complete API reference
- **[Catalog Schema and Filters](docs/CATALOG_FILTERS.md)**: JSONL field roles and data-derived filter capabilities
- **[Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md)**: Start here for JSONL ingest, lifecycle-cached capabilities, compact agent discovery, validation, and retrieval
- **[Commerce Contracts](docs/COMMERCE_CONTRACTS.md)**: Internal product, cart, and commerce tool contracts
- **[Shopper Agent Architecture](docs/SHOPPER_AGENT_ARCHITECTURE.md)**: Clean map of the published catalog, turn flow, skills, tools, and memory boundaries
- **[Shopper Agent Leadership Note](docs/SHOPPER_AGENT_LEADERSHIP_NOTE.md)**: Concise request flow, memory ownership, worked styling example, and prioritized next steps
- **[Shopper Agent Tool Registry](docs/SHOPPER_AGENT_TOOL_REGISTRY.md)**: Registered Deep Agents tools for the shopper-serving agent
- **[Shopper Agent Skill Registry](docs/SHOPPER_AGENT_SKILL_REGISTRY.md)**: Registered Deep Agents skills and markdown tuning loop
- **[Deep Agents Migration Plan](docs/DEEP_AGENTS_MIGRATION_PLAN.md)**: SDK migration, session isolation, tools, skills, and scaling notes
- **[Deep Agents Cart Tool Goal](docs/DEEP_AGENTS_CART_TOOL_GOAL.md)**: Minimal cart-tool smoke gate and constraints
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Installation and setup instructions
- **[Testing and Evaluation](tests/README.md)**: Unit, integration, and
  Challenger/Judge workflows; multi-turn judging uses the actual generated
  conversation plus bounded current-turn catalog evidence from successful
  search and detail tools
- **[Documentation Hub](docs/README.md)**: Complete documentation index

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
- [NVIDIA NIM](https://catalog.ngc.nvidia.com/orgs/nim): Containerized AI models
- [NVIDIA NGC](https://ngc.nvidia.com/): AI platform and container registry

### Technologies Used
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview): Agent harness for tool and skill orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph): Runtime used underneath Deep Agents
- [FastAPI](https://fastapi.tiangolo.com/): Modern Python web framework
- [React](https://reactjs.org/): JavaScript library for building user interfaces
- [Milvus](https://milvus.io/): Vector database for similarity search

### Related Projects
- [Nemotron 3 Embed 1B](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3-embed-1b): Embedding model for semantic search
- [NV-CLIP](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nvclip): Visual understanding model for image retrieval
- [Nemotron 3 Super](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3-super-120b-a12b): Large language model

## License

GOVERNING TERMS: Use of the blueprint software and materials and NIM containers are governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-specific Terms for AI products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/);  and the use of models is governed by the [NVIDIA Community Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/).
 
ADDITIONAL INFORMATION: [Llama 3.1 Community License Agreement](https://www.llama.com/llama3_1/license/) for Llama 3.1 70B Instruct NIM, Llama 3.1 NemoGuard 8B - Content Safety and Llama 3.1 NemoGuard 8B - Topic Control models, built with Llama, (ii) MIT license for NV-EmbedQA-E5-v5.
 
This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use, found in [License-3rd-party.txt](/LICENSE-3rd-party.txt).
 
Use of the product catalog data in the retail shopping assistant is governed by the terms of the [NVIDIA Data License for Retail Shopping Assistant](/LICENSE-assets.txt) (15Aug2025).

---

<div align="center">

[Back to Top](#top)

</div>
