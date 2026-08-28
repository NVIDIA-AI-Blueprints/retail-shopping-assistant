# AGENTS.md

This file is a working guide for coding agents and contributors in this repository.

## 1) Project Summary

Retail Shopping Assistant is a multi-service application with:
- `chain_server`: FastAPI + Deep Agents SDK orchestration over deterministic catalog and cart tools.
- `catalog_retriever`: FastAPI service for text/image embedding retrieval against Milvus.
- `memory_retriever`: FastAPI + single-replica SQLite service for an immutable representative-shopper registry, ordered durable conversation turns, exact finalized-turn replay, stable cart-line IDs, and atomically idempotent add/remove/quantity mutations.
- `guardrails`: FastAPI wrapper around NeMo Guardrails input/output safety checks.
- `ui`: React + TypeScript chat UI using SSE streaming.
- `shared`: Shared YAML configs, JSONL catalog data/role sidecars, and image assets.

Top-level orchestration is via `docker-compose.yaml`; optional local NIM model containers are in `docker-compose-nim-local.yaml`.

## 2) Architecture and Request Flow

1. UI posts to `/api/query/stream` (nginx proxy on port `3000`).
2. Nginx routes `/api/*` to `chain-server:8009`.
3. Chain server request flow:
   - `DeepAgentsRuntime` first starts a durable turn in the memory service, which returns bounded model-context-eligible raw turns, the prior turn's selected skill names, the authoritative cart, and an optional server-resolved representative-shopper snapshot. Blocked turns remain durable and exactly replayable but are excluded from both the service projection and chain prompt formatter. Graph working state uses a request-scoped pair of `conversation_id` and `request_id`. A selected profile ID is bound immutably to that conversation and renders one compact current-turn context block containing only type, behavior, and saved ZIP. Profile precedence and non-authority rules are also present only for selected-profile turns; Guest receives neither the block nor profile-specific prompt rules. The block is soft guidance: current explicit instructions and recent explicit preferences take precedence, and it cannot establish budget, product constraints or facts, cart intent, skill selection, or tool grants. Unknown caller fields remain backward-compatibly ignored, and caller-supplied persona objects are never injected.
   - Optional input guardrails run before model/tool work; attached media is analyzed through the configured perception client.
   - Deep Agents graph execution has a configurable 45-second default deadline. A timeout captures bounded partial graph messages, clears unsent products, finalizes the durable turn as failed, and deletes the request checkpoint only after that finalization succeeds.
   - Every turn begins with a required model step that semantically selects the smallest applicable set from five registered shopper skills. The latest durable selected names are supplied as a read-only continuity hint; they never authorize tools or replace the fresh selection. Product work uses exactly one primary procedure: product discovery or outfit styling. Budget shopping is a modifier only when the shopper states a budget; cart and policy requests may use their standalone skills. An invalid composition receives its typed reason and one correction attempt; a second invalid composition ends with a deterministic clarification and runs no shopping tool. Multiple activation calls in one response execute none and clarify immediately. The runtime injects the complete selected files and exposes only the union of their declared `tools_granted`; dispatch independently rechecks the selected skills, grant union, and immutable tool policy. Pre-activation, same-batch, and ungranted shopping calls are execution-blocked.
   - Catalog capabilities generate `search_catalog_tool`'s flat schema with exact taxonomy values and non-taxonomy required-constraint properties. The model may call it with a direct advertised scope or, when a shopper-named type is not separately advertised, one model-selected faithful advertised parent category. In the parent path, the shopper's type stays in `requested_product_type` and the semantic query, the category is the only taxonomy filter, and returned products remain closest alternatives under their actual catalog types. Model-authored catalog absence is not exposed. If neither a direct type nor one faithful parent can be selected, the assistant asks one concise clarification directly without a tool call or absence claim. Deterministic code validates and maps search values but does not interpret shopper language. Each text search carries `requested_product_type`: the shortest product noun or true umbrella from the shopper's current turn or direct antecedent, excluding color, material, fit, occasion, weather, and style modifiers. For a role the shopper did not name, it is the model's own role noun, and taxonomy carries every advertised subcategory that role covers; the evidence records the role as model-composed so the reply presents it as a suggestion and never reads a miss inside those types as the role being unavailable. It is provenance, not taxonomy or ranking text, and is `null` only for image-only search. Each call has at most one category.
   - Every search also carries required pre-retrieval `shopper_guidance`: one concise, product-agnostic sentence authored under the active skill. A directly stated unadvertised requirement ranks the search and is disclosed as unconfirmed; it never becomes a hard filter and never suppresses retrieval. Fail-closed applies to the claim, not to the shopper: price, availability, and cart mutations are never guessed, and an attribute is never asserted as confirmed without catalog evidence, but a product is never withheld for lack of evidence about it. Only a schema-valid proposed inferred requirement on a genuinely open role may consume that distinct scope's one model-owned review. Deterministic code does not classify shopper prose or rewrite malformed arguments. A repair cannot change a shopper-named scope noun. A successful partial search may continue with another valid role and its own one-repair opportunity, but no scope receives two repairs; the configured turn cap remains three successful searches.
   - Cart mutations require explicit product/cart-line refs. Grounding reads actual tool-role messages, separates current-request evidence from prior-turn evidence, and never treats an assistant draft as evidence. Successful searches preserve the taxonomy-independent semantic query as internal ranking evidence, the pre-retrieval `shopper_guidance` as product-agnostic response framing, and each confirmed filter set with the products from that search. A completed search gets one final tools-disabled model step under the active skill, followed by the grounding editor. If the requested outcome depends on an unconfirmed material, fit, comfort, durability, care, weather, or other functional property, the response must disclose that gap and frame results as the closest catalog or styling direction rather than as proven suitable. If that draft or editor is unavailable, deterministic fallback uses search guidance, static skill `response_guidance`, returned names, prices, categories, and search-scoped confirmed filters, followed by the same generic unverified-property disclosure. Scoped zero-result evidence cannot establish absence outside its exact taxonomy and filters. The graph and grounding editor share one execution deadline; a grounding timeout finalizes as failed with `grounding_timeout`, uses the deterministic catalog renderer for search-only evidence, and otherwise returns a fixed retry/cart-check response rather than the unverified draft. Editor errors and empty or whitespace-only editor output use the same fail-closed response rule with `grounding_error`.
   - Optional output guardrails run, then the memory service finalizes the durable turn as completed, blocked, or failed before products, images, content, and metrics are emitted over SSE. An exact retry of a finalized request replays its stored response without model/tool work. Internal diagnostics include bounded current-turn product evidence from successful catalog search and detail results plus bounded `catalog_scope_outcomes` for zero-result scopes; each search scope remains attached to its own products. A rejected tool call reports the gate that refused it; a multi-scope call refused for only some roles remains a completed call and reports those roles under `scope_rejections`. Public query responses contain an empty diagnostics object by default. `EXPOSE_AGENT_DIAGNOSTICS=true` exposes the detailed trace only for a trusted operator or evaluation deployment. Final-text extraction skips tool, tool-calling, and internal activation messages; if no shopper-facing answer exists, the runtime returns a safe fallback with `incomplete_agent_response`. On graph failure, bounded current-turn messages are captured before checkpoint cleanup.
   - A provider-neutral daily weather client and `get_weather_forecast_tool` factory exist as a dormant boundary. They accept only a five-digit US ZIP plus today, one exact date, or a complete inclusive date range. `WEATHER_ENABLED=false` is the default. The wrapper is not registered with Deep Agents, granted by a skill, mentioned in prompts, connected to shopper context, exposed through FastAPI, or called by the UI; startup, health checks, and shopper turns make no weather request.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image.
   - `/products/{product_id}` for deterministic details after a search ref is known.
   - `/capabilities` once per successful chain-server process lifecycle for the catalog-owned query contract.

## 3) Source Map (Where to Change What)

- Serving agent orchestration and registered tools:
  `chain_server/src/deepagents_runtime.py`
- Capability-derived model-visible tool schemas, and the stateless helpers the
  runtime calls: `chain_server/src/turn_support.py`
- Reusable model-visible catalog search rules: `chain_server/src/catalog_scope.py`
- Shopper-skill registry, frontmatter validation, and immutable tool policy: `chain_server/src/tool_policy.py`
- Per-turn skill activation, model-visible tool binding, and dispatch grant gate: `chain_server/src/skill_activation.py`
- Durable conversation-turn client and wire contracts: `chain_server/src/conversation_memory.py`
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
- The composer receives separated authority lanes. Never merge lanes with different authority into one block: dialogue carries intent, the product index carries identity, the cart is authoritative, tool evidence establishes current facts.
- Deterministic code may establish that a catalog filter is unadvertised; it must not decide the conversational move. Never substitute a fixed refusal for the model's composed answer.
- The grounding editor must run whenever any hydrated authority lane exists — tool evidence, historical product identity, or cart. Dialogue is intent only and never grounds a product claim.
- Committed commerce effects ride on the tool artifact and must be consulted before any read-only failure fallback. If the graph snapshot cannot be read, warn about the cart: absence of evidence is not evidence of absence.
- Tool-loop control outcomes are typed: tools return `(text, artifact)` via `chain_server/src/control_signals.py`, and the middleware reads the artifact. Never recover control state by parsing tool text.
- Tool-loop control prefixes are defined once in `chain_server/src/tool_loop_control.py`. Never re-declare one as a literal elsewhere; producers render from the constant and matchers key off it.
- A catalog-search gate that turns a scope back records which gate it was, as a
  `SearchRejection` on the tool artifact, one entry per searched scope in scope
  order. Many gates share one model-visible prefix, so the text can never name
  them. Diagnostics prefer the recorded code and fall back to prefix matching.
  Do not add a code to a path that hands the model an instruction to continue
  the conversation: a call whose every scope carries a code is treated as a
  refused call and receives the fixed refusal response.
- Message-shape helpers: `chain_server/src/message_shape.py`. Pure readers over LangChain messages; no runtime state.
- Request-local turn state: `chain_server/src/turn_scope.py`. Search budgets, catalog-repair bookkeeping, product evidence, and retrieved images live on one `TurnScope` per turn, not as closure variables. New per-turn mutable state belongs there.
- Prior turns are carried typed on `State.dialogue`. `State.context` is rendered prompt text only and must never be parsed back into state or authority. Dialogue establishes shopper intent, never product, policy, inventory, or cart facts.
- The pre-Deep-Agents pipeline (`graph.py`, `planner.py`, `retriever.py`, `cart.py`, `chatter.py`, `summarizer.py`, `functions.py`) has been deleted. `deepagents_runtime.py` is the only chain-server serving path.

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
- Sets `SHARED_ROOT`, `SHARED_CONFIG_ROOT`, `REACT_APP_API_BASE_URL=http://localhost:8009`, and `BROWSER=none`.
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
- UI API base URL defaults to `/api` for nginx, but local development can set `REACT_APP_API_BASE_URL` to the chain-server URL.
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
- UI API base URL defaults to `/api` (nginx path), but local runner overrides it to `http://localhost:8009`.
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
  model-stage budget shared by the Deep Agents graph and grounding editor. The
  editor receives only the remaining time. Graph and grounding timeouts finalize
  as failed using `agent_timeout` and `grounding_timeout`, respectively. Timeout
  finalization uses the existing durable attempt fence; do not substitute the
  stale-turn abandonment setting for this live execution deadline.
- Durable raw turns contain shopper/assistant text, the nullable representative
  profile binding, selected skill names, bounded replay, and ordered event
  envelopes. They do not store the rendered shopper-context block, raw media,
  model reasoning, or the complete graph/tool transcript. Presented-product
  events and deterministic historical resolution are implemented; active
  anchors and effective preferences remain reserved and unused.
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
- Search results use source `record_id` values. A search result carries the
  catalog-declared detail fields present on that product, so an attribute it
  lists is confirmed for that product and needs no further read; an attribute it
  omits is absent from the result, not evidence that the product lacks it, and
  is read from `/products/{product_id}`. That endpoint remains the source for
  fields a search does not carry. Current generated IDs are safe only within the
  active snapshot, so stale refs require a fresh search.
- Current-request product evidence is process-local. Finalized product cards
  also create a bounded durable same-conversation reference index; one unique
  exact resolution can restore a prior product after restart or on another
  worker. Missing, ambiguous, or stale-catalog references require clarification
  or a fresh search.
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
  taxonomy relationship or catalog-absence field. A shopper-named type that is
  not separately advertised may use one model-selected faithful advertised
  parent category; structured evidence forces the response to disclose the
  broader search and retain every result's actual catalog category. If neither
  a direct type nor one faithful parent can be selected, the assistant asks one
  concise clarification directly without calling the tool. Deterministic code validates and maps
  search values; it does not maintain keyword aliases or infer structured
  fields from shopper prose. Each call uses at most one category. A
  role the shopper did not name keeps the model's role noun in
  `requested_product_type` and selects every advertised subcategory that role
  covers; it is recorded as model-composed rather than refused. Alternatives, confirmations, comparisons, and
  follow-ups count as named types. For all text searches,
  `requested_product_type` is the shortest product noun or true umbrella from
  the shopper's current turn or direct antecedent, excluding color, material,
  fit, occasion, weather, and style modifiers. It is provenance rather than
  taxonomy or ranking text and is `null` only for image-only search. A
  singleton exact taxonomy value must be coherent with that provenance; the
  semantic query is independent soft ranking direction.
- An unresolved shopper-named type receives one concise clarification response
  rather than a tool call or a parent, sibling, or adjacent search. An
  unsupported modifier does not erase an advertised type. A directly stated must-have
  missing from the generated constraint schema belongs in
  `unadvertised_requirements`; subjective style remains semantic direction. A
  product type never belongs in the requirement lane.
- After one invalid search schema in a distinct scope, the model receives one
  bounded repair step. It may submit one corrected search or return no tool call
  to signal that clarification is needed. A no-tool repair is only a control
  signal: the server marks the branch, discards the model prose, and emits the
  fixed clarification
  `Could you clarify the product type or requirement you want me to use?`.
  If another requested search scope already succeeded, its deterministic
  grounded products are returned before that clarification. Other successful
  shopping-tool evidence is composed with the fixed clarification through the
  existing grounding editor.
  Only a schema-valid proposed inferred requirement on a genuinely open role
  consumes that scope's model-owned review. Explicit
  objective must-haves remain, rank the search, and are returned to the shopper
  as unconfirmed rather than suppressing retrieval. Deterministic code does not parse
  shopper prose. The repair cannot replace a shopper-stated product-scope noun.
  A successful partial search may continue to another valid role and its own
  one-repair opportunity, but no scope receives two repairs.
- Duplicate search identity is normalized taxonomy plus hard constraints;
  changing only semantic wording cannot repeat a retrieval.
- Grounding accepts product evidence only from tool-role messages. Current-turn
  evidence is isolated by the server request marker; prior-turn tool evidence
  may resolve a direct reference but cannot establish a new search or mutation.
  Successful search evidence records the model-authored semantic query as
  internal ranking direction and required pre-retrieval `shopper_guidance` as
  product-agnostic response framing. Completed successful search-only responses
  receive one tools-disabled synthesis under the active skill and then the
  grounding editor. Outcomes that depend on product properties absent from
  tool evidence are disclosed as unconfirmed and framed as the closest catalog
  or styling direction, not as proven suitable. If synthesis or editing cannot
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
- The Slice 3 weather client/tool is deliberately dormant. Keep it out of
  `DeepAgentsRuntime` registration, `SHOPPING_TOOL_POLICIES`, shopper-skill
  grants, prompts, request/state models, FastAPI, and UI until a separate
  leveraging slice defines trusted location/date precedence, grounded evidence,
  provider attribution, and forecast-uncertainty behavior. It needs no MCP
  server and must never log the key, prepared URL, ZIP, requested dates,
  resolved location, provider body, or raw exception.

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

## 10) Diagnose Before You Build

Four questions, answerable from Phoenix in minutes, before writing any fix for
assistant behaviour. Skipping them cost a day this week on a change that was
already measured as ineffective.

**1. Does the model already have the fact?** Read the prompt, do not assume it.

```
sessions(first:1, sessionFilterCondition: "session_id == '<conversation-id>'")
  { edges { node { sessionId numTraces traces(first:30)
      { edges { node { rootSpan { name startTime } } } } } } }
```

`span_kind == 'LLM'` gives the whole prompt; `span_kind == 'TOOL'` gives each
tool's input **and its return**, which the durable record does not store.

If the fact is already in the prompt, a second copy of it will not help. The
size a showing was made under was recorded, rendered into the historical index,
verified live -- and the behaviour it targeted measured 6 in 10 both before and
after, because the shopper's own words were one turn up the conversation the
whole time. That is a tie, not a gap.

**2. Is our code refusing, or is the model choosing?** Read the tool result.

A refusal of a correct action is our bug and usually has a deterministic fix. A
model that had everything and chose otherwise is a tie. These look identical in
a transcript and are opposite problems.

**3. Can you make it fail on demand?** A scenario that goes red before the fix.

Where you cannot reproduce it, say so and do not claim a fix. Two changes this
week were shipped as "unit-tested, journey-level unproven", which is honest and
useful; a third was claimed as fixed on a single green run and was not.

**4. Code or words?** Prose in a skill has a near-zero hit rate here.

Everything that measurably worked either removed a wrong behaviour or made the
runtime construct the outcome. Every rule added to a skill file this week was
inert, and one caused a regression.

### What follows from the answers

- **Our bug** -- fix it, prove red to green.
- **Tie** -- either make the runtime do it deterministically, or accept it and
  stop counting it as a defect. Do not add a rule; that has been tried.
- **Script wrong** -- fix the script. A turn with no verb should not assert a
  cart change.

### Two standing rules

**Delete a fix that does not measure.** Carrying it costs review attention and
implies evidence that does not exist.

**Never quote a single run.** The suite moves about +/-7 scenarios on identical
code -- 25% of turns take a different tool path between two runs at temperature
0. Only the intersection of repeated runs means anything, and a regression
scenario repeated ten times is worth more than one full pass.
