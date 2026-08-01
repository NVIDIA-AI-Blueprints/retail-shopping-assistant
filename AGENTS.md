# AGENTS.md

This file is a working guide for coding agents and contributors in this repository.

## 1) Project Summary

Retail Shopping Assistant is a multi-service application with:
- `chain_server`: FastAPI + Deep Agents SDK orchestration over deterministic catalog and cart tools.
- `catalog_retriever`: FastAPI service for text/image embedding retrieval against Milvus.
- `memory_retriever`: FastAPI + single-replica SQLite service for an immutable representative-shopper registry, ordered durable conversation turns, exact finalized-turn replay, negotiated rolling semantic summaries, stable cart-line IDs, and atomically idempotent add/remove/quantity mutations.
- `guardrails`: FastAPI wrapper around NeMo Guardrails input/output safety checks.
- `ui`: React + TypeScript chat UI using SSE streaming.
- `shared`: Shared YAML configs, JSONL catalog data/role sidecars, and image assets.

Top-level orchestration is via `docker-compose.yaml`; optional local NIM model containers are in `docker-compose-nim-local.yaml`.

## 2) Architecture and Request Flow

1. UI posts to `/api/query/stream` (nginx proxy on port `3000`).
2. Nginx routes `/api/*` to `chain-server:8009`.
3. Chain server request flow:
   - `DeepAgentsRuntime` first starts a durable turn in the memory service using negotiated response contract v2. Memory returns a durable semantic summary, bounded model-context-eligible raw turns strictly after its watermark, a separate compact historical-product index, a bounded oldest compaction source, the prior turn's selected skill names, the authoritative cart, and an optional server-resolved representative-shopper snapshot. Summary text is continuity guidance only and cannot establish exact wording, product identity/facts, cart truth, tool evidence, policy, availability, or permission. After a completed guarded response, a tools-disabled compactor receives only the prior summary and memory-owned source; a compare-and-swap summary advance commits atomically with turn finalization. Timeout, invalid output, conflict, cancellation, and failed turns retain raw source. Blocked turns remain durable and exactly replayable but are excluded from both the service projection and chain prompt formatter. Graph working state uses a request-scoped pair of `conversation_id` and `request_id`. A selected profile ID is bound immutably to that conversation and renders one compact current-turn context block containing only type, behavior, and saved ZIP. Profile precedence and non-authority rules are also present only for selected-profile turns; Guest receives neither the block nor profile-specific prompt rules. The block is soft guidance: current explicit instructions and recent explicit preferences take precedence, and it cannot establish budget, product constraints or facts, cart intent, skill selection, or tool grants. Unknown caller fields remain backward-compatibly ignored, and caller-supplied persona objects are never injected.
   - Optional input guardrails run before model/tool work; attached media is analyzed through the configured perception client.
   - Deep Agents graph execution has a configurable 45-second default deadline. A timeout captures bounded partial graph messages and clears product cards and images. When every current-request business call in that checkpoint is classified `read` by the immutable tool policy and valid typed search or detail evidence is present, a deterministic renderer may return only that evidence through output guardrails. Any pending or completed mutating call, unknown call, or unusable evidence receives the fixed retry/cart-check response. The durable turn remains failed with `agent_timeout`, and the request checkpoint is deleted only after finalization succeeds.
   - Every turn begins with a required model step that semantically selects the smallest applicable set from five registered shopper skills. The latest durable selected names are supplied as a read-only continuity hint; they never authorize tools or replace the fresh selection. Product work uses exactly one primary procedure: product discovery or outfit styling. Budget shopping is a modifier only when the shopper states a budget; cart and policy requests may use their standalone skills. An invalid composition receives its typed reason and one correction attempt; a second invalid composition ends with a deterministic clarification and runs no shopping tool. Multiple activation calls in one response execute none and clarify immediately. The runtime injects the complete selected files and exposes only the union of their declared `tools_granted`; dispatch independently rechecks the selected skills, grant union, and immutable tool policy. Pre-activation, same-batch, and ungranted shopping calls are execution-blocked.
   - Established-product comparison remains a model-owned procedure inside
     `outfit-styling`; there is no comparison skill or deterministic intent
     router. An exact opaque ref from the validated server-owned historical
     product projection may authorize its own scalar detail read directly.
     Natural, ordinal, shortened, or ambiguous earlier-product references are
     submitted together in the one batched historical-resolution call before
     separate detail reads. The default two-read cap fits one pair; an
     unauthorized, conflicting, missing, or stale ref performs no substitute
     search. Missing or ambiguous members clarify.
   - Catalog capabilities generate `search_catalog_tool`'s flat schema with exact taxonomy values and non-taxonomy required-constraint properties. Product meaning is model-owned: the active skills and tool descriptions instruct the model to author `requested_product_type`, select faithful advertised taxonomy, use a category-only scope only when it judges that category to be a faithful parent, and clarify directly when it cannot make a faithful selection. A shopper-supplied product title remains identity: the full title stays in `semantic_query`, its product noun supplies `requested_product_type`, and title words do not become hard requirements unless the shopper states them independently. Runtime does not parse current or recent shopper prose, suffix-match product phrases, classify shopper-named versus open roles, or validate a semantic relationship between `requested_product_type` and taxonomy. It validates structural completeness, capability-derived values and types, the one-category bound, category/subcategory coherence, supported search mode, and advertised hard constraints. A category-only search records the model-authored requested role and searched category separately; grounding presents category-scoped candidates under their actual catalog categories without asserting a parent relationship or catalog absence.
   - Every search also carries required pre-retrieval `shopper_guidance`: one concise, product-agnostic sentence authored under the active skill. Any nonempty `unadvertised_requirements` lane fails closed without retrieval or repair. At most one structural catalog repair is available for the entire turn, not per semantic scope. It receives the current shopper message, typed search tool, compact capabilities, sanitized validator feedback, and active skill context. Independently valid `required_constraints`, `scope_complete`, and `search_mode` are preserved; `requested_product_type` and taxonomy are not locked from shopper prose. A no-tool repair produces the fixed clarification, and another validation failure after the repair closes to synthesis. Distinct valid taxonomy-plus-hard-constraint scopes may continue within the configured three-search cap.
   - Cart mutations require explicit product/cart-line refs. The main Deep Agent
     receives the durable summary and bounded raw dialogue for semantic
     continuity. Its final evidence composer is a separate authority boundary:
     for every successfully activated turn without a fixed server response it
     receives the current shopper request, bounded
     shopper-authored continuity, the exact historical-product identity index,
     active-skill response guidance, server-owned response requirements, the
     authoritative cart, available images, and current-request typed tool
     evidence, which may be empty. It does not receive the rolling summary,
     prior assistant prose, prior-turn tool evidence, or the graph draft. This
     also prevents a no-tool follow-up from repeating an earlier assistant claim
     as current evidence.
     Successful searches preserve
     the taxonomy-independent semantic query as internal ranking evidence, the
     pre-retrieval `shopper_guidance` as product-agnostic response framing, and
     each confirmed filter set with the products from that search. If an outcome
     depends on an unconfirmed material, fit, comfort, durability, care,
     weather, or other functional property, the response discloses that gap and
     frames results as the closest catalog or styling direction rather than as
     proven suitable. Deterministic fallback uses current typed evidence,
     search guidance, static skill `response_guidance`, and verified catalog
     fields. Scoped zero-result evidence cannot establish absence outside its
     exact taxonomy and filters. The graph and composer share one execution
     deadline; a composition timeout finalizes as failed with
     `grounding_timeout`, uses the deterministic catalog renderer for
     search-only evidence, a verified-detail renderer for current successful
     detail evidence, and otherwise returns a fixed retry/cart-check response.
     Empty or invalid
     composition uses the same evidence-preserving split with `grounding_error`.
   - Optional output guardrails run, then the memory service finalizes the durable turn as completed, blocked, or failed before products, images, content, and metrics are emitted over SSE. An exact retry of a finalized request replays its stored response without model/tool work. Internal diagnostics include bounded current-turn product evidence from successful catalog search and detail results plus bounded `catalog_scope_outcomes` for zero-result scopes; each search scope remains attached to its own products. Public query responses contain an empty diagnostics object by default. `EXPOSE_AGENT_DIAGNOSTICS=true` exposes the detailed trace only for a trusted operator or evaluation deployment. Final-text extraction skips tool, tool-calling, and internal activation messages; if no shopper-facing answer exists, the runtime returns a safe fallback with `incomplete_agent_response`. On graph failure, bounded current-turn messages are captured before checkpoint cleanup.
   - A provider-neutral daily weather client and `get_weather_forecast_tool`
     factory exist as a dormant boundary. They accept only a five-digit US ZIP
     plus today, one exact date, or a complete inclusive date range.
     `WEATHER_ENABLED=false` is the default. The wrapper is not registered with
     Deep Agents, granted by a skill, mentioned in prompts, connected to shopper
     context, exposed through FastAPI, or called by the UI; startup, health
     checks, and shopper turns make no weather request.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image.
   - `/products/{product_id}` for deterministic details after a search ref is known.
   - `/capabilities` once per successful chain-server process lifecycle for the catalog-owned query contract.

## 3) Source Map (Where to Change What)

- Serving agent orchestration and registered tool wrappers:
  `chain_server/src/deepagents_runtime.py`
- Pure capability-derived catalog tool schema, structural taxonomy mapping,
  and canonical scope identity: `chain_server/src/catalog_tool_contract.py`
- Reusable model-visible catalog search rules: `chain_server/src/catalog_scope.py`
- Shopper-skill registry, frontmatter validation, and immutable tool policy: `chain_server/src/tool_policy.py`
- Per-turn skill activation, model-visible tool binding, and dispatch grant gate: `chain_server/src/skill_activation.py`
- Durable conversation-turn client and wire contracts: `chain_server/src/conversation_memory.py`
- Pure rolling-summary input planning and output validation:
  `chain_server/src/conversation_summary.py`
- Representative-shopper read client: `chain_server/src/shopper_profiles.py`
- API contract and SSE endpoint: `chain_server/src/main.py`
- Catalog capability cache/prompt projection: `chain_server/src/catalog_capabilities.py`
- Catalog intent validation/execution: `chain_server/src/catalog_request.py`, `chain_server/src/catalog_execution.py`
- Commerce service adapters: `chain_server/src/commerce_tools.py`
- Operator-managed store policy content: `shared/configs/chain_server/store_policies.yaml`
- Shopper behavior skills and references: `chain_server/skills/shopper/`
- Image/video perception: `chain_server/src/media_perception.py`
- Dormant weather request/result contract and Visual Crossing adapter:
  `chain_server/src/weather.py`
- Dormant, directly constructible weather wrapper:
  `chain_server/src/weather_tool.py`
- Shared request/state models: `chain_server/src/agenttypes.py`
- `graph.py`, `planner.py`, `retriever.py`, `cart.py`, `chatter.py`, and `summarizer.py` are legacy compatibility paths, not the serving runtime.

- Catalog API entrypoints and request validation: `catalog_retriever/src/main.py`
- JSONL loading/search-document construction: `catalog_retriever/src/catalog.py`
- Dynamic catalog capabilities: `catalog_retriever/src/capabilities.py`
- Embedding retrieval, deterministic ranking, and filtering: `catalog_retriever/src/retriever.py`
- Image/base64 helpers: `catalog_retriever/src/utils.py`

- Memory API entrypoint: `memory_retriever/src/main.py`
- SQLite configuration and schema: `memory_retriever/src/database.py`,
  `memory_retriever/src/models.py`, `memory_retriever/src/migrations.py`
- Durable turn start/finalize/replay boundary:
  `memory_retriever/src/conversations.py`
- Immutable representative-shopper bootstrap/read API:
  `memory_retriever/src/shopper_profiles.py`

- Guardrails API: `guardrails/src/main.py`
- Guardrails engine/wiring: `guardrails/src/rails.py`
- Guardrails model config helper: `guardrails/src/config_utils.py`

- UI streaming behavior: `ui/src/components/chatbox/chatbox.tsx`
- UI representative-shopper picker: `ui/src/components/ShopperPicker.tsx`
- UI API config and feature flags: `ui/src/config/config.ts`
- UI chat panel layout and footer alignment: `ui/src/chatbox.css`
- UI message/cart-toast parsing helpers: `ui/src/utils/index.ts`

- Shared config roots:
  - `shared/configs/chain_server/`
  - `shared/configs/catalog_retriever/`
  - `shared/configs/rails/`

- Local agent skills:
  - Local app runner: `skills/retail-local-runner/` plus `.agents/skills/retail-local-runner/` shim.
  - Unit/integration test runner: `skills/retail-test-runner/` plus `.agents/skills/retail-test-runner/` shim.

## 4) Runbook

### Cloud endpoint mode (no local NIM containers)

```bash
cp .env.example .env
$EDITOR .env
source .env
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

### Local NIM mode (requires multi-GPU setup)

Brings up the local LLM (`nemotron` service, image `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`), `nvclip`, `embedqa`, and the two NemoGuard guardrail containers.

```bash
cp .env.example .env.local-nim
$EDITOR .env.local-nim
source .env.local-nim
mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

Before running full local NIM mode, edit the relevant roles in
`shared/configs/models.yaml` to `source: local_nim`. The `nemotron` service is
launched with `NIM_PASSTHROUGH_ARGS=--enable-auto-tool-choice --tool-call-parser
llama3_json` so vLLM accepts `tool_choice="auto"`. Reasoning output is
suppressed via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
on the chain-server side so streamed tokens flow eagerly.

### Local app-code mode (recommended for iterative development)

Use the local runner skill when working on Python or React app code outside containers:

```bash
python skills/retail-local-runner/scripts/local_runner.py status
python skills/retail-local-runner/scripts/local_runner.py start
python skills/retail-local-runner/scripts/local_runner.py stop
```

The local runner:
- Starts app services as local processes and uses Docker only for Milvus infra (`etcd`, `minio`, `milvus`).
- Uses `shared/configs/models.yaml` plus environment overrides.
- `configure --nim-host http://HOST` writes ignored `.local-run/model-endpoints.env` with remote NIM URLs.
- Retains `WEATHER_ENABLED` and `WEATHER_API_KEY` only for the chain-server
  process and removes them from memory, guardrail, catalog, and UI processes.
- Sets `SHARED_ROOT`, `SHARED_CONFIG_ROOT`,
  `REACT_APP_API_BASE_URL=/local-api`, and `BROWSER=none`; the scoped React
  development proxy forwards that same-origin prefix to the chain server
  without the package-proxy Host restriction, so remote browsers need only
  port `3000` forwarded. Development responses use `Cache-Control: no-store`
  so a forwarding layer cannot retain a bundle with an obsolete browser API
  base URL.
- Creates runtime files under ignored `.local-run/` and links ignored `ui/public/images -> shared/images`.

If a remote NIM host is needed, ask for the base host URL and run `configure`; do not hard-code private hosts in committed files.

### Health checks

```bash
curl -sS http://localhost:3000            # UI via nginx
curl -sS http://localhost:8009/health     # chain server
curl -sS http://localhost:8010/health     # catalog retriever
curl -sS http://localhost:8011/health     # memory retriever
```

## 5) Testing and Validation

Current test assets:
- Offline unit tests under `tests/unit/`.
- Live integration scripts under `tests/integration/`, driven by endpoint calls and YAML scenario files.
- Multi-turn Judge prompts include the actual generated prior shopper and
  assistant turns plus bounded current-turn structured catalog evidence. The
  generated history is authoritative when it conflicts with a reference
  answer.
- Shopping Judge preflight requires nonempty trusted evaluation diagnostics
  before the first paid Judge request. Start the chain service with
  `EXPOSE_AGENT_DIAGNOSTICS=true`; public/default responses remain unchanged.
- Legacy/basic guardrails coverage under `guardrails/test/test_rails.py`.
- GitHub Actions runs offline Python unit tests on pull requests when backend Python files, backend requirements, or unit-test files change (`.github/workflows/python-unit-tests.yml`). This workflow intentionally uses placeholder API-key environment values and must not depend on live services or external model endpoints.
- GitHub Actions builds modified service Docker images on pull requests when service directories or compose build wiring change (`.github/workflows/docker-image-builds.yml`). This workflow is build-only and must not push images or require secrets.

Useful test workflow:
1. For offline validation, run:
   ```bash
   python skills/retail-test-runner/scripts/run_retail_tests.py unit
   ```
2. For live validation, bring services up with Docker Compose or the local runner and verify health endpoints.
3. Run integration scenarios:
   ```bash
   python skills/retail-test-runner/scripts/run_retail_tests.py integration --test-path shopping
   ```

Integration outputs are generated under `tests/integration/conversations/<TEST_PATH>/results/` and `tests/integration/conversations/<TEST_PATH>/judge/`; these are ignored artifacts and should not be committed. `tests/.coverage`, `.pytest_cache/`, `htmlcov/`, `.local-run/`, `node_modules/`, and `ui/public/images` are also ignored runtime artifacts.

## 6) Configuration Rules

- Services load configs from `SHARED_CONFIG_ROOT` when set, otherwise `/app/shared/configs`.
- Service behavior lives in `shared/configs/chain_server/config.yaml`,
  `shared/configs/catalog_retriever/config.yaml`, and
  `shared/configs/rails/config.yml`.
- Model endpoints live in `models.yaml`; each role independently uses `source: endpoint`, `source: local_nim`, or `source: disabled`.
- Catalog image helpers read assets from `SHARED_ROOT` when set, otherwise `/app/shared`.
- Catalog data and role-sidecar paths can be overridden with
  `CATALOG_DATA_SOURCE` and `CATALOG_SCHEMA_SOURCE`.
- UI API base URL defaults to `/api` for nginx. The local runner uses
  `/local-api` together with the scoped React development proxy; other local
  setups may set `REACT_APP_API_BASE_URL` to a directly reachable chain-server
  URL.
- Use `python scripts/model_config.py show --validate` to inspect resolved endpoints without printing secrets.

Key env vars:
- `LLM_API_KEY`
- `EMBED_API_KEY`
- `RAIL_API_KEY` / `NVIDIA_API_KEY` (guardrails container)
- `NGC_API_KEY` (for local NIM containers)
- `LLM_BASE_URL`, `LLM_MODEL`
- `TEXT_EMBED_BASE_URL`, `TEXT_EMBED_MODEL`
- `IMAGE_EMBED_BASE_URL`, `IMAGE_EMBED_MODEL`
- `RAILS_BASE_URL`, `RAILS_CONTENT_BASE_URL`, `RAILS_TOPIC_BASE_URL`
- `CHECKPOINT_STORE` (currently supports only `memory`)
- `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS`
- `EXPOSE_AGENT_DIAGNOSTICS` (trusted operator/evaluation deployments only)
- `MEMORY_DATABASE_URL`, `MEMORY_SQLITE_BUSY_TIMEOUT_MS`
- `MEMORY_TURN_ABANDON_SECONDS`, `MEMORY_RECENT_TURNS`
- `CATALOG_DATA_SOURCE`, `CATALOG_SCHEMA_SOURCE`
- `SHARED_CONFIG_ROOT` (local runner / non-container config root)
- `SHARED_ROOT` (local runner / non-container shared asset root)
- `REACT_APP_API_BASE_URL` (local React dev server API target)
- `WEATHER_ENABLED` (dormant direct weather-client construction only; default
  `false`)
- `WEATHER_API_KEY` (Visual Crossing server-side key; required only when
  explicitly constructing the enabled dormant client)

## 7) Important Gotchas

- Ports in docs are not always aligned with runtime wiring.
  - Actual backend service port is `8009` in compose.
  - External app entrypoint is usually `http://localhost:3000` through nginx.
  - Standard Compose binds memory port `8011` to host loopback only. Containers
    use `http://memory-retriever:8011` on the private Compose network.
- UI API base URL defaults to `/api` (nginx path). The local runner uses
  same-origin requests plus the React development proxy so the browser does not
  need direct access to port `8009`.
- The memory service stores ordered shopper/assistant turns and cart state in a
  single-replica SQLite database. Compose uses
  `sqlite:////data/context.db` on the `memory-data` named volume; deleting that
  volume deletes the stored transcript and cart state.
- The same SQLite database owns five immutable representative shoppers loaded
  from `shared/configs/memory_retriever/shopper_profiles.json`. The bundled UI
  sends only the selected ID. Turn start resolves it transactionally, binds it
  to the conversation (including Guest as `NULL`), and returns a typed
  three-field context. Switching profiles therefore requires the new
  conversation/cart identities created by the picker.
- Stale active turns are recovered at startup and atomically at the next turn
  start. An exact abandoned retry reopens only the latest conversation sequence,
  preserves its request ID for cart idempotency, and rotates its service-issued
  attempt token. Older abandoned turns remain superseded.
- Every serving turn starts durably before guardrail/model/tool work and is
  finalized exactly once as completed, blocked, or failed. An exact retry of a
  finalized request replays its response. Finalize must echo the current attempt
  token: a stale attempt is rejected and becomes a safe superseded-attempt
  response without stale products or images. A start failure runs no agent work;
  a non-fencing finalize failure preserves the grounded response, records
  `memory_finalize_error`, and keeps the graph checkpoint.
- Deep Agents graph state uses request-scoped, in-process MemorySaver threads
  keyed by conversation and request identity. A successful durable finalize
  deletes the thread; a finalize failure preserves it for recovery. Remaining
  threads disappear on chain-server restart and are not shared across workers
  or replicas.
  `CHECKPOINT_STORE=memory` is the only supported value; a compliant production
  shared graph backend remains an open decision.
- `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` defaults to 45 seconds and bounds one
  model-stage budget shared by the Deep Agents graph and final evidence
  composer. The composer receives only the remaining time. Graph and grounding
  timeouts finalize as failed using `agent_timeout` and `grounding_timeout`,
  respectively. A graph timeout may salvage only current-request typed search
  or detail text when every observed and pending business call is
  policy-classified as read-only; product cards and images remain empty. Any
  mutating or unknown call forces the retry/cart-check response. Timeout
  finalization uses the existing durable attempt fence;
  do not substitute the stale-turn abandonment setting for this live execution
  deadline.
- `conversation_summary` in chain config defaults to enabled, triggers at six
  unsummarized eligible turns, retains the newest two raw turns, caps output at
  4,096 characters, and uses a separate 15-second timeout. Its tools-disabled
  call runs only after a completed guarded response and is outside the shared
  graph/grounding deadline. When no eligible source is offered, it makes no
  model call.
- Durable raw turns contain shopper/assistant text, the nullable representative
  profile binding, selected skill names, bounded replay, and ordered event
  envelopes. They do not store the rendered shopper-context block, raw media,
  model reasoning, or the complete graph/tool transcript. Contract v2 adds a
  rolling semantic-summary projection and watermark without deleting raw rows.
  The main serving prompt keeps the summary, exact post-watermark raw tail, and
  historical-product projection in distinct lanes. The final evidence composer
  instead receives a bounded shopper-only projection and no summary or prior
  assistant text. Memory offers at most four
  exact oldest eligible turns for one compare-and-swap advance; the chain never
  selects source turns itself. Unversioned v1 responses remain available for a
  rolling deployment: deploy memory before chain and roll back chain before
  memory. Presented-product events and deterministic historical resolution are
  implemented; active anchors and effective preferences remain reserved and
  unused.
- The memory API has no service authentication. Standard Compose limits its
  host mapping to `127.0.0.1:8011`; keep it on an internal network in other
  deployments. Detailed agent diagnostics remain internal and public query
  responses return `{}` unless `EXPOSE_AGENT_DIAGNOSTICS=true` is deliberately
  set behind a trusted operator or evaluation surface.
- `CartItem` rows carry catalog `product_id`, opaque `cart_line_id`, and `price`
  fields. Startup migrations add missing fields to older SQLite databases. The
  deterministic `view_cart_total` tool uses stored prices instead of letting
  the LLM do arithmetic.
- The serving cart tools require a `PRODUCT_REF` from catalog search for adds and
  a `CART_LINE_ID` from cart state for removals. Adds revalidate the product
  against the active catalog before changing memory state.
- Cart adds use catalog `product_id`; removals and quantity changes use the
  memory service's opaque, non-reusable `cart_line_id`. All three mutations
  share one owner-scoped idempotency ledger. The cart change and replay record
  commit in the same SQLite transaction, so identical retries replay once and
  conflicting key reuse fails without mutation. Mutation records currently
  persist for the SQLite database lifetime; retention policy remains follow-up
  work. Variant-level cart identity remains future work.
- Store-policy content is cached from an operator-managed YAML file. The
  bundled template has `configured: false`, so it fails closed until every
  placeholder is replaced and an operator explicitly enables it. Product
  availability is a deliberate no-I/O stub for known conversation product
  refs. It reports sized availability for apparel and footwear and one-size
  availability for other categories; catalog presence alone is not an
  inventory signal. Active sale or promotion status is a separate no-I/O stub
  that currently reports no configured promotions; catalog search does not
  establish markdown status.
- Catalog retriever fuses candidate lists, deduplicates by product ID, applies
  explicit hard filters (including taxonomy and price), then performs
  deterministic thresholding, similarity ranking, and top-k trimming. The
  current small-catalog candidate window covers the complete active snapshot.
  The serving agent
  sends one semantic text query per catalog call. The catalog makes no
  chat/completion call or LLM interpretation. Field roles come from the catalog
  sidecar; values, ranges, and taxonomy scopes come from the JSONL and are
  exposed through `/capabilities`, not chain-server config.
- Catalog startup fingerprints the JSONL, sidecar, embedding models, image
  mode, and semantic template. Matching complete indexes are reused; changes
  rebuild all enabled catalog collections.
- The chain server caches the first successful catalog capability response for
  its process lifetime. It uses the full contract for deterministic validation
  and a compact projection for the LLM prompt. After a catalog change, wait for
  catalog health and then restart the chain server.
- Search results use source `record_id` values. Product facts are read from
  `/products/{product_id}`; current generated IDs are safe only within the
  active snapshot, so stale refs require a fresh search.
- Current-request product evidence is process-local. Finalized product cards
  also create a bounded durable same-conversation reference index. A validated
  exact opaque ref in that projection may authorize a detail read directly;
  natural, ordinal, shortened, or ambiguous references use one typed batch
  resolver. Conflicting projection entries are excluded, and an exact-ref
  detail read is checked against the active catalog name before its facts
  become evidence. That narrow path does not authorize availability or cart
  work.
  Missing, ambiguous, or stale-catalog references require clarification or a
  fresh current product choice.
- Local LLM service is named `nemotron` (was `llama`); chain-server reaches it through `shared/configs/models.yaml` when the app LLM role uses `source: local_nim`.
- Tool calling against the local NIM requires `--enable-auto-tool-choice --tool-call-parser llama3_json` passthrough args. Without them, requests with `tool_choice="auto"` 400.
- The Deep Agents model first selects shopper skills through the internal
  activation control tool. Only after the runtime injects the complete selected
  files may it choose from the union of their declared tool grants; dispatch
  rechecks that union against the immutable policy. An invalid composition gets
  one correction attempt; a repeat stops with a fixed clarification instead of
  consuming the graph recursion budget. There is no serving
  planner/retriever/chatter graph. Media analysis and the cached catalog
  contract are included in its turn context. This Slice 0 grant gate does not
  yet prove explicit current-turn cart mutation intent; that authorization
  boundary remains planned.
- The model owns semantic selection for one flat, capability-derived catalog
  search. The model-visible fields are `semantic_query`, `shopper_guidance`,
  `requested_product_type`, `taxonomy`, `required_constraints`,
  `scope_complete`, and optional `search_mode`. There is no model-authored
  taxonomy relationship or catalog-absence field. The active skills instruct
  the model to preserve the shopper's shortest product noun or umbrella in
  `requested_product_type`, choose faithful advertised taxonomy, use a
  category-only scope only when it judges that category to be a faithful
  parent, and clarify directly when no faithful selection exists. Those are
  semantic model responsibilities. Runtime does not parse shopper prose,
  suffix-match product phrases, classify shopper-named versus open roles, or
  validate `requested_product_type` against taxonomy. It validates structural
  completeness, exact capability-derived taxonomy and constraint values,
  category/subcategory coherence, the one-category bound, supported search
  mode, and text-versus-image shape. The semantic query remains independent
  soft ranking direction. A category-only search records the requested role
  and searched category separately; grounding keeps every result's actual
  category without certifying a parent relationship or catalog absence.
- Any nonempty `unadvertised_requirements` lane fails closed before retrieval
  and does not enter repair. Subjective style remains semantic direction, and
  product types never belong in the requirement lane.
- The first schema or capability validation failure in a turn may receive one
  bounded structural repair step. The isolated repair receives the current
  shopper message, typed search tool, compact capabilities, sanitized validator
  feedback, and active skill context. It may submit one corrected search or
  return no tool call; the latter is a control signal whose prose is replaced
  with `Could you clarify the product type or requirement you want me to use?`.
  Independently valid `required_constraints`, `scope_complete`, and
  `search_mode` are preserved. Runtime does not key repair to a semantic scope,
  lock `requested_product_type` or taxonomy from shopper prose, or run an
  explicit-versus-inferred requirement review. A later validation failure
  closes to synthesis. Distinct valid scopes may continue within the
  three-search cap, but the turn receives no second repair.
- Duplicate search identity is normalized taxonomy plus hard constraints;
  changing only semantic wording cannot repeat a retrieval.
- Grounding accepts factual evidence only from current-request tool-role
  messages isolated by the server request marker. Prior turns may supply
  shopper-authored continuity and exact historical identity, but prior
  assistant prose, prior tool evidence, the rolling summary, and the graph draft
  are absent from final tool-backed composition and cannot establish a search,
  detail read, mutation, or policy result.
  Successful search evidence records the model-authored semantic query as
  internal ranking direction and required pre-retrieval `shopper_guidance` as
  product-agnostic response framing. Completed successful search-only responses
  receive one tools-disabled completion under the active skill and then the
  final evidence composer. Outcomes that depend on product properties absent from
  tool evidence are disclosed as unconfirmed and framed as the closest catalog
  or styling direction, not as proven suitable. If final composition cannot
  produce an answer, static
  `response_guidance` and deterministic per-search candidate and filter groups
  provide the fallback plus the same generic unverified-property disclosure. A
  partial successful result set gets a neutral continuation. Zero-result
  evidence remains scoped to the exact advertised taxonomy and filters searched.
  Bounded `catalog_scope_outcomes` expose zero-result scopes to operator
  diagnostics.
- Final-response extraction ignores tool messages, assistant tool-call messages,
  and internal activation markers. If no shopper-facing answer remains, the
  runtime returns a safe retry response and records `incomplete_agent_response`.
- Final shopper text is grounded against tool evidence and current cart state;
  it must not claim a mutation without a successful cart result or invent facts
  absent from catalog detail evidence.
- The right chat panel is fixed between the nav bar and global footer; keep `ui/src/chatbox.css` aligned with the navbar/footer heights when changing layout.
- The weather client/tool remains deliberately dormant. Keep it out of
  `DeepAgentsRuntime` registration, `SHOPPING_TOOL_POLICIES`, shopper-skill
  grants, prompts, request/state models, FastAPI, and UI until a separate
  leveraging slice is explicitly in scope. It needs no MCP server and must never
  log the key, prepared URL, ZIP, requested dates, resolved location, provider
  body, or raw exception.

## 8) Contribution and Commit Notes

- Follow `CONTRIBUTING.md` requirements.
- Use signed commits (`git commit -s`) for contributions.
- Keep changes scoped by service; avoid cross-service behavior changes without updating related config/docs.
- Before committing, check staged changes for `.env`, `.local-run/model-endpoints.env`, private hostnames, API keys, local absolute paths, `.local-run/`, `node_modules/`, `ui/public/images`, and generated integration `results` / `judge` artifacts.

## 9) Recommended Change Workflow for Agents

1. Identify impacted service(s) and config files.
2. Implement smallest coherent change.
3. Validate via health + targeted scenario.
4. If API shape changes, update docs in `docs/API.md` and any UI assumptions.
5. Note any config/env additions in docs.
