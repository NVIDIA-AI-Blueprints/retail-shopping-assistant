# AGENTS.md

This file is a working guide for coding agents and contributors in this repository.

## 1) Project Summary

Retail Shopping Assistant is a multi-service application with:
- `chain_server`: FastAPI + Deep Agents SDK orchestration over deterministic catalog and cart tools.
- `catalog_retriever`: FastAPI service for text/image embedding retrieval against Milvus.
- `memory_retriever`: FastAPI + single-replica SQLite service for ordered durable conversation turns, exact finalized-turn replay, stable cart-line IDs, and atomically idempotent add/remove/quantity mutations.
- `guardrails`: FastAPI wrapper around NeMo Guardrails input/output safety checks.
- `ui`: React + TypeScript chat UI using SSE streaming.
- `shared`: Shared YAML configs, JSONL catalog data/role sidecars, and image assets.

Top-level orchestration is via `docker-compose.yaml`; optional local NIM model containers are in `docker-compose-nim-local.yaml`.

## 2) Architecture and Request Flow

1. UI posts to `/api/query/stream` (nginx proxy on port `3000`).
2. Nginx routes `/api/*` to `chain-server:8009`.
3. Chain server request flow:
   - `DeepAgentsRuntime` first starts a durable turn in the memory service, which returns bounded finalized raw turns and the authoritative cart. It still uses `conversation_id` as the process-local checkpoint thread. Caller-supplied persona data is not injected into model context.
   - Optional input guardrails run before model/tool work; attached media is analyzed through the configured perception client.
   - Deep Agents graph execution has a configurable 45-second default deadline. A timeout captures bounded partial graph messages, clears unsent products, finalizes the durable turn as failed, and deletes the request checkpoint only after that finalization succeeds.
   - Every turn begins with a required model step that semantically selects the smallest applicable set from five registered shopper skills. Product work uses exactly one primary procedure: product discovery or outfit styling. Budget shopping is a modifier only when the shopper states a budget; cart and policy requests may use their standalone skills. The runtime injects the complete selected files and exposes only the union of their declared `tools_granted`; dispatch independently rechecks the selected skills, grant union, and immutable tool policy. Pre-activation, same-batch, and ungranted shopping calls are execution-blocked.
   - Catalog capabilities generate the tool's exact taxonomy values and non-taxonomy required-constraint properties. The model selects from that schema; deterministic code validates and maps the structured values but does not interpret shopper language. Each text search carries `requested_product_type`: the shortest product noun or true umbrella from the shopper's current turn or direct antecedent, excluding color, material, fit, occasion, weather, and style modifiers. For `agent_selected_type`, it is the chosen advertised role noun. It is provenance, not taxonomy or ranking text, and is `null` only for `image_only`. A genuinely open role selects and names exactly one advertised subcategory. Each call has at most one category. A concrete type with no faithful advertised match stops without retrieval using empty taxonomy and no hard constraints.
   - Every search also carries required pre-retrieval `shopper_guidance`: one concise, product-agnostic sentence authored under the active skill. A directly stated unadvertised requirement on a shopper-named scope fails closed. Only a schema-valid proposed inferred requirement on a genuinely open `agent_selected_type` role may consume that distinct scope's one model-owned review. An exact duplicate of a shopper-stated unavailable concrete type receives a separate bounded validation correction. Deterministic code does not classify shopper prose or rewrite malformed arguments. A repair cannot change a shopper-named scope noun, and an open-role repair remains `agent_selected_type`. A successful partial search may continue with another valid role and its own one-repair opportunity, but no scope receives two repairs; the configured turn cap remains three successful searches.
   - Cart mutations require explicit product/cart-line refs. Grounding reads actual tool-role messages, separates current-request evidence from prior-turn evidence, and never treats an assistant draft as evidence. Successful searches preserve the taxonomy-independent semantic query as internal ranking evidence, the pre-retrieval `shopper_guidance` as product-agnostic response framing, and each confirmed filter set with the products from that search. A completed search gets one final tools-disabled model step under the active skill, followed by the grounding editor. If that draft or editor is unavailable, deterministic fallback uses search guidance, static skill `response_guidance`, returned names, prices, categories, and search-scoped confirmed filters. Scoped zero-result evidence cannot establish absence outside its exact taxonomy and filters.
   - Optional output guardrails run, then the memory service finalizes the durable turn as completed, blocked, or failed before products, images, content, metrics, and operator-facing agent diagnostics are emitted over SSE. An exact retry of a finalized request replays its stored response without model/tool work. Diagnostics include bounded current-turn product evidence from successful catalog search and detail results plus bounded `catalog_scope_outcomes` for no-direct and zero-result scopes; each search scope remains attached to its own products. Final-text extraction skips tool, tool-calling, and internal activation messages; if no shopper-facing answer exists, the runtime returns a safe fallback with `incomplete_agent_response`. On graph failure, bounded current-turn messages are captured before checkpoint cleanup.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image.
   - `/products/{product_id}` for deterministic details after a search ref is known.
   - `/capabilities` once per successful chain-server process lifecycle for the catalog-owned query contract.

## 3) Source Map (Where to Change What)

- Serving agent orchestration and registered tools: `chain_server/src/deepagents_runtime.py`
- Shopper-skill registry, frontmatter validation, and immutable tool policy: `chain_server/src/tool_policy.py`
- Per-turn skill activation, model-visible tool binding, and dispatch grant gate: `chain_server/src/skill_activation.py`
- Durable conversation-turn client and wire contracts: `chain_server/src/conversation_memory.py`
- API contract and SSE endpoint: `chain_server/src/main.py`
- Catalog capability cache/prompt projection: `chain_server/src/catalog_capabilities.py`
- Catalog intent validation/execution: `chain_server/src/catalog_request.py`, `chain_server/src/catalog_execution.py`
- Commerce service adapters: `chain_server/src/commerce_tools.py`
- Operator-managed store policy content: `shared/configs/chain_server/store_policies.yaml`
- Shopper behavior skills and references: `chain_server/skills/shopper/`
- Image/video perception: `chain_server/src/media_perception.py`
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

- Guardrails API: `guardrails/src/main.py`
- Guardrails engine/wiring: `guardrails/src/rails.py`
- Guardrails model config helper: `guardrails/src/config_utils.py`

- UI streaming behavior: `ui/src/components/chatbox/chatbox.tsx`
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
- `MEMORY_DATABASE_URL`, `MEMORY_SQLITE_BUSY_TIMEOUT_MS`
- `MEMORY_TURN_ABANDON_SECONDS`, `MEMORY_RECENT_TURNS`
- `CATALOG_DATA_SOURCE`, `CATALOG_SCHEMA_SOURCE`
- `SHARED_CONFIG_ROOT` (local runner / non-container config root)
- `SHARED_ROOT` (local runner / non-container shared asset root)
- `REACT_APP_API_BASE_URL` (local React dev server API target)

## 7) Important Gotchas

- Ports in docs are not always aligned with runtime wiring.
  - Actual backend service port is `8009` in compose.
  - External app entrypoint is usually `http://localhost:3000` through nginx.
- UI API base URL defaults to `/api` (nginx path), but local runner overrides it to `http://localhost:8009`.
- The memory service stores ordered shopper/assistant turns and cart state in a
  single-replica SQLite database. Compose uses
  `sqlite:////data/context.db` on the `memory-data` named volume; deleting that
  volume deletes the stored transcript and cart state.
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
- Deep Agents graph state still uses conversation-scoped, in-process
  MemorySaver. It disappears on chain-server restart, accumulates in heap for
  the process lifetime, and is not shared across workers or replicas.
  `CHECKPOINT_STORE=memory` is the only supported value; a compliant production
  shared graph backend remains an open decision.
- `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` defaults to 45 seconds and bounds the
  awaited Deep Agents graph invocation. Timeout finalization uses the existing
  durable attempt fence; do not substitute the stale-turn abandonment setting
  for this live execution deadline.
- Durable raw turns contain shopper/assistant text plus bounded replay and
  ordered event envelopes. They do not store raw media, model reasoning, or the
  complete graph/tool transcript. Projection tables and event vocabulary are
  reserved schema only: preference, anchor, product-reference, and historical
  resolver semantics remain unimplemented until Slice 5.
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
- Same-conversation product refs are held in a bounded process-local cache.
  They are separate from the process-local graph checkpoint; a restart, another
  replica, cache eviction, or catalog replacement requires a fresh search.
- Local LLM service is named `nemotron` (was `llama`); chain-server reaches it through `shared/configs/models.yaml` when the app LLM role uses `source: local_nim`.
- Tool calling against the local NIM requires `--enable-auto-tool-choice --tool-call-parser llama3_json` passthrough args. Without them, requests with `tool_choice="auto"` 400.
- The Deep Agents model first selects shopper skills through the internal
  activation control tool. Only after the runtime injects the complete selected
  files may it choose from the union of their declared tool grants; dispatch
  rechecks that union against the immutable policy. There is no serving
  planner/retriever/chatter graph. Media analysis and the cached catalog
  contract are included in its turn context. This Slice 0 grant gate does not
  yet prove explicit current-turn cart mutation intent; that authorization
  boundary remains planned.
- The model owns semantic selection of exact advertised taxonomy values. The
  runtime-generated schema also exposes exact advertised non-taxonomy
  constraints. Deterministic code validates and maps those values; it does not
  maintain keyword aliases or infer structured fields from shopper prose. Each
  call uses at most one category; `agent_selected_type` may include advertised
  subcategories serving one focused semantic role only when the shopper named
  no type for that role. Alternatives, confirmations, comparisons, and
  follow-ups count as named types. Every text search requires
  `requested_product_type`: the shortest product noun or true umbrella from the
  shopper's current turn or direct antecedent, excluding color, material, fit,
  occasion, weather, and style modifiers. For `agent_selected_type`, it is the
  chosen advertised role noun. It is provenance rather than taxonomy or ranking
  text and is `null` only for `image_only`. A singleton exact taxonomy value
  must be coherent with that provenance; the semantic query is independent soft
  ranking direction. A genuinely open `agent_selected_type` role is rejected
  unless its requested type names exactly one selected advertised subcategory.
- `no_direct_catalog_match` is a no-retrieval result for an explicitly requested
  concrete type and uses empty taxonomy with no hard constraints. An unsupported
  modifier does not erase an advertised type. A directly stated must-have
  missing from the generated constraint schema belongs in
  `unadvertised_requirements`; subjective style remains semantic direction. A
  product type never belongs in the requirement lane. An exact duplicate of a
  shopper-stated unavailable concrete type is rejected with one bounded
  correction requiring an empty no-direct envelope; runtime does not rewrite it.
- After one invalid search schema in a distinct scope, the model receives one
  search-only repair step. Only a schema-valid proposed inferred requirement on
  a genuinely open `agent_selected_type` role consumes that scope's model-owned
  review. Explicit objective must-haves remain and fail closed. Deterministic
  code does not parse shopper prose. The repair cannot replace a shopper-stated
  product-scope noun, and an open-role repair remains `agent_selected_type`. A
  successful partial search may continue to another valid role and its own
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
  grounding editor. If synthesis or editing cannot produce an answer, static
  `response_guidance` and deterministic per-search candidate and filter groups
  provide the fallback. A
  partial successful result set gets a neutral continuation. Zero-result
  evidence remains scoped to the exact advertised taxonomy and filters searched.
  Bounded `catalog_scope_outcomes` expose no-direct and zero-result scopes to
  operator diagnostics.
- Final-response extraction ignores tool messages, assistant tool-call messages,
  and internal activation markers. If no shopper-facing answer remains, the
  runtime returns a safe retry response and records `incomplete_agent_response`.
- Final shopper text is grounded against tool evidence and current cart state;
  it must not claim a mutation without a successful cart result or invent facts
  absent from catalog detail evidence.
- The right chat panel is fixed between the nav bar and global footer; keep `ui/src/chatbox.css` aligned with the navbar/footer heights when changing layout.

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
