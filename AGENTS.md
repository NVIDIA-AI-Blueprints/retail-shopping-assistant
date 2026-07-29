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
   - Every turn begins with a required model step that semantically selects the smallest applicable set from six registered shopper skills. The latest durable selected names are supplied as a read-only continuity hint; they never authorize tools or replace the fresh selection. Product work uses exactly one primary procedure: product discovery or outfit styling. Budget shopping is a modifier only when the shopper states a budget. Event context is a modifier used only beside outfit styling; it alone grants the read-only weather tool. Select it whenever an event destination or venue is stated, a supported forecast would materially change event guidance, or the response would otherwise ask about or branch on missing destination or venue context. Cart and policy requests may use their standalone skills. An invalid composition receives its typed reason and one correction attempt; a second invalid composition ends with a deterministic clarification and runs no shopping tool. Multiple activation calls in one response execute none and clarify immediately. The runtime injects the complete selected files and exposes only the union of their declared `tools_granted`; dispatch independently rechecks the selected skills, grant union, and immutable tool policy. Pre-activation, same-batch, and ungranted shopping calls are execution-blocked.
   - Catalog capabilities generate `search_catalog_tool`'s flat schema with exact taxonomy values and non-taxonomy required-constraint properties. The model may call it with a direct advertised scope or, when a shopper-named type is not separately advertised, one model-selected faithful advertised parent category. In the parent path, the shopper's type stays in `requested_product_type` and the semantic query, the category is the only taxonomy filter, and returned products remain closest alternatives under their actual catalog types. Model-authored catalog absence is not exposed. If neither a direct type nor one faithful parent can be selected, the assistant asks one concise clarification directly without a tool call or absence claim. Deterministic code validates and maps search values but does not interpret shopper language. Each text search carries `requested_product_type`: the shortest product noun or true umbrella from the shopper's current turn or direct antecedent, excluding color, material, fit, occasion, weather, and style modifiers. For a genuinely open role, it is the one advertised subcategory selected for that role. It is provenance, not taxonomy or ranking text, and is `null` only for image-only search. Each call has at most one category.
   - Every search also carries required pre-retrieval `shopper_guidance`: one concise, product-agnostic sentence authored under the active skill. A directly stated unadvertised requirement on a shopper-named scope fails closed. Only a schema-valid proposed inferred requirement on a genuinely open role may consume that distinct scope's one model-owned review. Deterministic code does not classify shopper prose or rewrite malformed arguments. A repair cannot change a shopper-named scope noun. A successful partial search may continue with another valid role and its own one-repair opportunity, but no scope receives two repairs; the configured turn cap remains three successful searches.
   - Cart mutations require explicit product/cart-line refs. Grounding reads actual tool-role messages, separates current-request evidence from prior-turn evidence, and never treats an assistant draft as evidence. Successful searches preserve the taxonomy-independent semantic query as internal ranking evidence, the pre-retrieval `shopper_guidance` as product-agnostic response framing, and each confirmed filter set with the products from that search. A completed search gets one final tools-disabled model step under the active skill, followed by the grounding editor. Selected event-context turns also use that final editor; it receives only saved-ZIP-candidate presence, never digits, and deterministic grounded rendering restores candidates if a successful-search edit drops all of them. If the requested outcome depends on an unconfirmed material, fit, comfort, durability, care, weather, or other functional property, the response must disclose that gap and frame results as the closest catalog or styling direction rather than as proven suitable. If that draft or editor is unavailable, deterministic fallback uses search guidance, static skill `response_guidance`, returned names, prices, categories, and search-scoped confirmed filters, followed by the same generic unverified-property disclosure. Scoped zero-result evidence cannot establish absence outside its exact taxonomy and filters. The graph and grounding editor share one execution deadline; a grounding timeout finalizes as failed with `grounding_timeout`, uses the deterministic catalog renderer for search-only evidence, and otherwise returns a fixed retry/cart-check response rather than the unverified draft. Editor errors and empty or whitespace-only editor output use the same fail-closed response rule with `grounding_error`.
   - Optional output guardrails run, then the memory service finalizes the durable turn as completed, blocked, or failed before products, images, content, and metrics are emitted over SSE. An exact retry of a finalized request replays its stored response without model/tool work. Internal diagnostics include bounded current-turn product evidence from successful catalog search and detail results plus bounded `catalog_scope_outcomes` for zero-result scopes; each search scope remains attached to its own products. Weather tool arguments/output are always redacted from diagnostics and failed-turn partial graph capture. The weather trace retains only categorical candidate action, date shape, location-source kind, provider-input kind, and typed outcome; it never includes a location, ZIP, date, resolved place, URL, body, or exception. The saved profile ZIP is recursively scrubbed from diagnostic string keys and values. Public query responses contain an empty diagnostics object by default. `EXPOSE_AGENT_DIAGNOSTICS=true` exposes the detailed trace only for a trusted operator or evaluation deployment. Final-text extraction skips tool, tool-calling, and internal activation messages; if no shopper-facing answer exists, the runtime returns a safe fallback with `incomplete_agent_response`. On graph failure, bounded current-turn messages are captured before checkpoint cleanup.
   - The registered, read-only `get_weather_forecast_tool` uses a provider-neutral daily weather client with Visual Crossing as the first adapter. `WEATHER_ENABLED=false` remains the default, and startup and health checks make no provider request. Event context gets one model-visible call attempt per turn; an invalid schema consumes it. Every call requires `candidate_action`: `reuse_prior_candidates` is valid only when historical candidates exist and the current turn solely supplies event context without requesting new/refined products; once accepted, it irreversibly hides and execution-blocks catalog search and closes the remaining tool loop for synthesis before provider I/O, so lookup failure cannot reopen search. `search_new_candidates` is required for an explicit current-turn new/refined product request or when no reusable prior candidates exist. Accepted reuse asks no follow-up question and does not initiate the next product role. The one tool accepts `confirmed_saved_zip` with no model-authored location or `shopper_provided_location`, where `location` is one bounded exact span from current or recent shopper text. A city, city plus region/country, address, or postal code is sufficient. For an abbreviation or ambiguous name, `location_query` is required: it preserves the exact shopper phrase as its first component and appends only one or two comma-separated region/country qualifiers. Keep `location="NYC"` and use `location_query="NYC, NY"`; `Springfield, TX` is a valid explicit regional assumption. It must never contain an unstated ZIP or numeric component or replace the authoritative shopper phrase, and is omitted only when `location` is already sufficiently qualified. The adapter passes `location_query` when present and otherwise passes `location` unchanged to Visual Crossing's Timeline endpoint; Visual Crossing resolves the named place in the same forecast request, with no alias table, representative ZIP, or separate geocoder. Saved ZIP is released only by the narrow deterministic confirmation gate: a current location-neutral statement explicitly naming `my`/`the` usual/home area, a bare affirmative immediately after the assistant's usual/home-area question, or a strict date-only follow-up immediately after an accepted confirmation. Any explicit current location, negation, uncertainty, or override rejects saved mode; modal `may be` is uncertainty while calendar `May 5` remains valid. An explicit destination takes precedence. The date must be an exact ISO date/complete range, except that the exact shopper phrase `next week` is resolved server-side from one captured UTC date to the next Monday-through-Sunday range. An unambiguous single-day phrase such as `tomorrow` may be normalized by the model to an exact ISO date against that same prompt-visible UTC anchor; a genuinely ambiguous or unresolved relative date gets one concise clarification. Within a valid tool call, `max_provider_attempts: 2` permits one additional provider attempt only after timeout or HTTP 5xx; HTTP 400 is a generic invalid request, never proof that the shopper's location is unresolved. Only successful current-turn evidence supports forecast claims. Provider-resolved place is omitted for saved mode; for an explicit shopper location it is included in bounded evidence and final output only as a transparent, reversible provider assumption, not proof of the event location. Prior durable forecast summaries are replaced in graph/editor recent discussion with a refresh placeholder, and prior weather tool messages are excluded from prior evidence. Deterministic final rendering appends exactly one canonical block containing the server-resolved `next week` range when used, every validated daily date, condition, available temperature, precipitation fact, attribution, and uncertainty warning. Accepted reuse bypasses the grounding editor entirely. On success, the server renders the exact names from the newest historical candidate set, one bounded styling direction derived from structured forecast evidence, and the canonical forecast block. On every weather failure, it renders those prior names, one conditional weather-flexible styling direction, and the typed safe failure without asking for finer location solely because the provider failed. Weather never proves product performance or creates an unstated catalog constraint. No weather-specific FastAPI, SSE, or UI shape is added.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image.
   - `/products/{product_id}` for deterministic details after a search ref is known.
   - `/capabilities` once per successful chain-server process lifecycle for the catalog-owned query contract.

## 3) Source Map (Where to Change What)

- Serving agent orchestration, registered tools, and capability-derived
  model-visible search schema: `chain_server/src/deepagents_runtime.py`
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
- Weather request/result contract and Visual Crossing adapter:
  `chain_server/src/weather.py`
- Direct and request-bound weather tool wrappers:
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
- Focused event-context live fixtures live under
  `tests/integration/conversations/event_context/`. Their optional top-level
  `shopper_profile_id` is sent on every turn in that file; omission is Guest.
  The five-file, ten-turn gate covers Guest plan-first behavior,
  selected-profile location precedence, confirmed and overridden event ZIPs,
  selected-profile shop-now catalog behavior, and non-event weather isolation;
  it remains separate from the full shopping cohort. The selected-profile
  shop-now fixture uses Jordan's minimal occasion request and requires exactly
  one catalog-search attempt, followed by a named-place `next week` turn that
  requires one weather call and no repeated catalog search.
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
- `WEATHER_ENABLED` (enables provider calls for the registered event-weather
  capability; default `false`)
- `WEATHER_API_KEY` (Visual Crossing server-side key; required when weather is
  enabled)

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
- Startup migration 7 removes the obsolete `diagnostics_json`,
  `start_response_body`, and `finalize_response_body` derived-cache columns
  left by an earlier local schema. Authoritative turn, event, projection, cart,
  and profile data are preserved, and the one-started-turn partial unique index
  is restored; do not delete the database to resolve that upgrade.
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
- Search results use source `record_id` values. Product facts are read from
  `/products/{product_id}`; current generated IDs are safe only within the
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
  fields from shopper prose. Each call uses at most one category. A genuinely
  open role selects exactly one advertised subcategory and names it in
  `requested_product_type`. Alternatives, confirmations, comparisons, and
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
  objective must-haves remain and fail closed. Deterministic code does not parse
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
- The `event-context` modifier may use a selected profile's saved ZIP
  only as a tentative event-location candidate. Explicit shopper-stated
  destination and venue context take precedence. When a shopper explicitly
  asks to plan before products and material location/venue context is missing,
  answer in exactly two short sentences: one conditional direction and one
  destination-or-venue question. With that context complete, use one short
  paragraph and ask no further event-context question. Ordinary shop-now
  occasion-only requests run one search for one grounded core role unless the
  shopper explicitly requests a complete look or names multiple roles. If
  location is still missing and materially changes the next recommendation,
  ask only event location alongside the results; with a saved-ZIP candidate,
  that question must ask whether the event is in the shopper's usual area or
  elsewhere, never discard the candidate for a bare destination question.
  Defer venue. Saved ZIP is never shopping, shipping, availability, or current
  location. Never infer venue, weather, wind, climate, or product performance
  from a ZIP, place name, or venue setting.
- `get_weather_forecast_tool` is registered read-only and granted only by
  `event-context`, whose composition still requires `outfit-styling`. It may
  receive one model-visible attempt per turn only after either confirmed
  saved-ZIP authority or an exact bounded location span from current/recent
  shopper text is established; a schema-invalid call consumes the attempt.
  Within a valid call, `max_provider_attempts: 2` permits one internal retry
  only after timeout or HTTP 5xx. HTTP 400 maps to generic
  `weather_request_invalid`; other 4xx, connection, and response-validation
  failures are not retried. Every call requires
  `candidate_action`. Use `reuse_prior_candidates` only when historical
  candidates exist and the current turn solely supplies event context without
  asking for new/refined products. Once accepted, it irreversibly hides and
  execution-blocks catalog search and closes the remaining tool loop for
  synthesis before provider I/O, so lookup failure cannot reopen search. Use
  `search_new_candidates` for an explicit current-turn new/refined product
  request or when no reusable prior candidates exist. Accepted reuse asks no
  follow-up question and does not initiate the next product role. Saved mode
  accepts no model-authored location and reaches weather only through the narrow
  current/recent confirmation gate: a location-neutral statement explicitly
  naming `my`/`the` usual/home area, a bare affirmative immediately after the
  assistant asks about that area, or the immediately following strict date-only
  turn. Any explicit current location, negation, uncertainty, or override
  rejects saved mode; explicit shopper location always wins. Shopper-provided
  mode keeps the exact city, region/country, address, or postal phrase in
  `location`. For an abbreviation or ambiguous name, `location_query` is
  required: it must preserve that exact phrase as its first component and
  append only one or two comma-separated region/country qualifiers. Keep
  `location="NYC"` and use `location_query="NYC, NY"`; `Springfield, TX` is a
  valid explicit regional assumption. Never add an unstated ZIP or numeric
  component or replace the shopper-authored authority phrase. Omit the query
  only when `location` is already sufficiently qualified. Pass
  `location_query` when present and otherwise pass
  `location` unchanged to Visual Crossing Timeline. Do not synthesize a ZIP or
  use a separate geocoder; Visual Crossing's `resolvedAddress` becomes the
  reversible `resolved_location` assumption.
  Semantic equivalence remains model-owned rather than deterministic proof and
  is correctable through the disclosed provider resolution.
  Require an exact ISO date/range except for exact shopper-authored `next week`;
  resolve that phrase server-side from one UTC turn anchor to the next
  Monday-through-Sunday range. A current negation or different date supersedes
  an earlier use. The model may normalize an unambiguous
  single-day phrase such as `tomorrow` to an exact ISO date against that same
  prompt-visible UTC anchor; genuinely ambiguous or unresolved relative dates
  require one concise clarification.
  Weather output is current-turn evidence only. Raw arguments/output are
  redacted from tool diagnostics and failed-turn partial output; diagnostics
  retain only categorical `candidate_action`, `request_shape`,
  `location_source`, `provider_input`, and `outcome`, never a place, ZIP, date,
  resolved place, URL, body, or exception. Diagnostic string keys and values
  also scrub the saved profile ZIP, and the entire grounding-editor prompt
  replaces that ZIP before the editor call. Omit provider-resolved place in
  saved mode. For an explicit shopper location, expose it only as a
  transparent, reversible provider assumption in bounded evidence and final
  output, never as proof of event location. Prior assistant forecast summaries
  are redacted from both graph and editor recent discussion, and prior weather
  tool messages are not prior evidence. Deterministic final rendering appends
  one exact canonical block with every validated date, condition, available
  temperature, precipitation fact, Visual Crossing attribution, and forecast
  uncertainty.
  For non-reuse weather paths, grounding-editor sentences containing
  weather-domain fact language or fact-shaped dates/values are removed while
  ordinary grounded styling language remains. Accepted reuse bypasses the
  grounding editor entirely. On success, the server renders the exact names
  from the newest historical candidate set, one bounded styling direction
  derived from structured forecast evidence, and the canonical forecast block.
  On any weather failure, it preserves those prior names, adds one conditional
  weather-flexible styling/recheck direction, and appends the typed safe
  failure. Never ask for state, region, country, or a finer location solely
  because provider lookup failed.
  Never turn forecast conditions into an unstated catalog constraint or claim
  product warmth, waterproofing, comfort, safety, or other performance without
  catalog evidence.
- Before enabling `WEATHER_ENABLED` for shopper traffic, the operator must
  confirm that the selected Visual Crossing plan permits the intended
  attribution, display, storage, and sharing. That review must include durable
  final assistant summaries and forecast processing by the downstream app
  model and output guardrails. Keep the key in the named environment variable;
  never log the key, prepared URL, shopper location or location query, ZIP,
  requested dates, resolved location, provider body, or raw exception.

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
