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
   - `DeepAgentsRuntime` first starts a durable turn in the memory service. Turn start returns a versioned rolling summary, a bounded newest model-context raw tail strictly after its watermark, a separate memory-owned oldest compaction prefix, one typed current weather-planning scope, a derived at-most-three-turn lane containing the exact completed turns referenced by that scope, a bounded set of valid typed weather receipts, the prior turn's selected skill names, the authoritative cart, and an optional server-resolved representative-shopper snapshot. The scope-source lane is isolated, independent of compaction and raw-tail limits, and used only for subject resolution and protected styling provenance; it never enters general context or supplies current-turn `set`/`unavailable` authority. Memory returns exact durable source text, but chain hydration replaces recognized assistant forecast prose with a fixed redaction before either permitted consumer sees the lane. Summary, exact raw discussion, the historical product index, the current weather scope, its source turns, and active receipts are separate state/prompt lanes. Summary prose is semantic continuity only and cannot establish exact shopper wording, product/cart/tool evidence, location/date authority, policy, availability, current weather, skill selection, or tool grants. Only the typed singleton can carry current cross-turn location/date authority, and only an explicitly bound valid receipt can establish forecast evidence for that exact scope. Existing values and current-subject unavailability markers carry their exact shopper-turn source identities. Before normal skill activation, a request-local, tools-disabled semantic resolver compares those source-bound turns with the current query through one forced typed control call. It is neither a business tool nor a subagent. Without a pending binding its schema contains only subject continuity; a live binding adds pending disposition and the exact opaque handle. Normal activation is the sole producer of the current turn's atomic `retain`/`set`/`clear`/`unavailable` scope selection. A pure authority compiler receives the already validated current-turn proposal and applies the resolver only to prior-dependent operations: unauthorized `retain`s become `clear`, while every current-turn `set` or `unavailable` survives. Invalid, unavailable, or unclear resolver output therefore fails closed for prior authority without vetoing current facts. The zero-argument forecast tool reads only the accepted effective singleton.
   - Resolver input has one chain-local aggregate character budget across the exact scope-source lane and non-authoritative summary/recent text. The current query, typed scope, trusted date, and every source sequence remain exact. Overflow uses deterministic marked head-and-tail semantic excerpts, then removes oldest optional recent turns and the optional summary as needed. If mandatory authority still cannot fit, no resolver model call runs and the result is `unclear/not_addressed`; R1 compilation preserves current-turn facts while prior retention fails closed. That prompt-budget slice changes no durable rows, replay behavior, or memory limits; the separate exact pending-decline capability is advertised by response contract 6.
   - After a successfully guarded response, configured thresholds may run one tools-disabled summary call over the prior summary and the largest fitting contiguous part of memory's oldest prefix while retaining the configured newest raw suffix. A single oversized oldest turn uses a marked deterministic head-and-tail projection only for compactor input; durable and replay text remain exact. The compactor receives no current query, profile/ZIP, cart, product ledger, receipt projection, media, tool transcript, diagnostics, or request identity. Its closed output is applied only at atomic finalization and becomes visible on the next request. Failure, timeout, invalid input/output, and cancellation never advance the watermark; a summary-only CAS conflict gets one finalize retry without the update and no model rerun. Blocked and abandoned turns remain durable and exactly replayable but are excluded from both raw context lanes; only completed or failed turns with assistant text are eligible.
   - Graph working state uses a request-scoped pair of `conversation_id` and `request_id`. A selected profile ID is bound immutably to that conversation and renders one compact current-turn context block containing only type, behavior, and saved ZIP. Profile precedence and non-authority rules are also present only for selected-profile turns; Guest receives neither the block nor profile-specific prompt rules. The block is soft guidance: current explicit instructions and recent explicit preferences take precedence, and it cannot establish budget, product constraints or facts, cart intent, skill selection, or tool grants. Unknown caller fields remain backward-compatibly ignored, and caller-supplied persona objects are never injected.
   - Turn-start `response_contract` is the caller's maximum supported version. Memory returns the highest version it supports up to that maximum. Unversioned callers receive the exact legacy top-level/projection shape and a bounded raw tail from sequence zero, even after a summary watermark advances. A staging-era v1 tail may include abandoned turns without assistant text; the chain accepts that negotiated legacy shape and filters those rows before model context. Contract v2+ retains strict memory-owned completed/failed, post-watermark eligibility. Contract 2 adds summary/receipt lanes; contract 3 adds the legacy current-weather-scope read/write shape; contract 4 adds atomic two-component scope resolution, durable pending `event_location`/`event_date` binding, and the finalize-write capability marker; contract 5 adds the derived exact scope-source read lane; contract 6 adds source-bound component-unavailability markers plus exact pending-decline and pending-supersession finalize controls. The source lane is required, even when empty, for v5+ and must exactly match the adjacent down-projected scope's unique `(turn_id, sequence)` references. Contracts 1–5 never receive v6 marker fields or their source turns. Migration 11 stores the pending binding in defaulted `current_weather_pending_json`; migration 12 stores v6 markers in defaulted `current_weather_unavailable_json`. Each auxiliary lane is merged only when its stored scope revision matches the core revision, so a rollback-era core mutation makes stale state inert while `current_weather_scope_json` remains strict pre-v4-readable. Deploy memory first, then chain; roll back chain first, then memory. A chain negotiating v5 preserves an unsupported decline when no independent current-turn facts exist, or writes only independently validated current-turn `set` facts while preserving the live pending binding. A chain negotiating only v3 fails closed rather than issuing an atomic scope write. Fresh and upgraded SQLite schemas give all additive non-null projection columns database defaults, so rolling memory rollback can still create projections.
   - Receipt freshness is evaluated atomically once at durable turn start. The accepted receipt set is the validity snapshot for that in-flight request; the runtime performs no second wall-clock expiry check mid-turn. Before activation, the model sees only receipt ID/type, shopper location/date scope, and `valid_until`, never normalized forecast evidence. Full evidence stays server-side and becomes grounding input only after activation explicitly binds that receipt.
   - Optional input guardrails run before model/tool work; attached media is analyzed through the configured perception client.
   - Deep Agents graph execution has a configurable 45-second default deadline. A timeout captures bounded partial graph messages, clears unsent products, finalizes the durable turn as failed, and deletes the request checkpoint only after that finalization succeeds.
   - Every turn begins with a required model step that semantically selects the smallest applicable set from six registered shopper skills. The latest durable selected names are a read-only continuity hint; they never authorize tools or replace fresh selection. Product work uses exactly one primary procedure: product discovery or outfit styling. Budget shopping is a modifier only when the shopper states a budget. Event context is an additive modifier used only beside outfit styling when physical location/date/venue/weather context is actually part of the current styling subject; hypothetical relevance is insufficient. It alone grants the read-only weather tool and never suppresses non-weather tools from the grant union. Before nested validation, activations without `event-context` discard all event-context question, scope, refresh, and receipt fields, so ungranted weather controls cannot mutate state, grant weather, or reject an otherwise valid shopping activation. When a prior weather scope exists, the isolated resolver always supplies source-bound `subject_relation` before this normal activation and does not extract location or date. Its forced control schema advertises no pending fields when the singleton has no pending binding; only a live pending binding adds `pending_disposition` and its exact opaque handle. An impossible no-pending control is defensively discarded without erasing the independently valid subject relation, while a stale handle against a live binding fails closed for prior authority. Activation remains the sole producer of one current-turn atomic `weather_scope`, copying the scope revision and choosing `retain`, `set`, `clear`, or `unavailable` for each component, and owns the one shopper-facing follow-up. `clear` means missing but askable; `unavailable` records that the shopper cannot or will not provide that component for this subject. A later `set`, explicit `clear`, or new-subject replacement removes the marker. The server validates current-turn authority and applies the subject relation: `new_subject` clears every proposed retain, while `same_subject` may retain. When a live pending binding exists, a new subject also carries its exact supersession handle and may retain no old component. Exact-handle `same_subject/answered` is the pending-only completion path and requires activation to set the named component. Exact-handle `same_subject/declined` instead requires the pending component to be `unavailable`, consumes only that live question, selects no replacement location/date question, and blocks weather while the component is unavailable. Neither path is selected by deterministic phrase rules. A current-turn counterpart `set`, `clear`, or `unavailable` uses the ordinary scope-revision boundary unless it is the exact declined target. Because weather cannot become complete while either component is unavailable, both location/date follow-ups are suppressed; `event_venue` remains available as styling context. After both component actions are compiled, the same effective-state check retires any opposite-component pending binding; a later `set` or explicit `clear` may create a fresh question but never revives its old source handle. Under a negotiated v5 contract, an unsupported decline preserves the live question unless independent current-turn `set` facts can be written safely. A missing askable location or date question is persisted as a typed pending binding stamped with its originating turn ID and sequence. If the resolver is unavailable or unclear, every proposed retain is cleared, receipt/refresh reuse is rejected, and weather stays blocked only when the effective scope still depends on prior authority. A validated current-turn `set`/`set` replacement is independent of the resolver and may require weather without importing an older subject. `unchanged/not_addressed` keeps a pending question silent during intervening product work. Exact-handle `unchanged/resume_requested` re-renders it without a scope resolution or source rotation when activation supplies no current-turn scope facts. A context-only accepted scope transition uses server-owned typed-unavailability acknowledgment and the accepted question boundary, so free-form draft prose cannot obscure the transition or restore a rejected follow-up. A complete accepted scope requires the zero-argument weather call. For an unchanged complete scope, activation sets `weather_refresh=true` only for an explicit shopper refresh request; comparisons and other turns do not auto-refresh. Activation may instead bind one valid receipt only with `event_context_next_question=none`, no scope update, no refresh request, and exact equality to the effective scope. Subject continuity, pending disposition, component availability, skill selection, venue, materiality, and intent remain semantic model judgment—not a deterministic router. An invalid selected composition, question, scope, or receipt binding receives one correction attempt; a repeat ends with a deterministic clarification and no shopping tool. Multiple activation calls in one response execute none. The runtime injects complete selected files and exposes only their declared grant union; dispatch independently rechecks that union and immutable tool policy.
   - A successful event-context activation result appends a model-visible additive-boundary reminder, repeated in the catalog-search description and outfit-styling skill: a reply that only supplies the destination, venue, or date requested in the prior response is context fulfillment, so established candidates stay in play without repeated non-weather product work. Explicit same-turn comparison, refinement, replacement, search, check, cart, or policy work follows the normal selected skills. This is semantic procedural guidance, not deterministic intent classification or an execution gate; it does not change the selected grant union or dispatch policy.
   - Established-candidate comparison remains a model-owned procedure inside `outfit-styling`; there is no comparison skill or deterministic intent router. When compared products are not current-request evidence, the model submits all of them in the one batched historical-resolution call, then reads each uniquely resolved ref through separate scalar detail calls. The default two-read cap fits one pair. Missing or ambiguous required products clarify without a substitute search. Weather is optional additional evidence and never replaces resolution/details or establishes product performance.
   - The selected skills and their granted tools remain one Deep Agent's semantic procedure from activation through its final candidate answer. There is no post-answer semantic completion reviewer, operation plan, or correction trajectory that can delete the candidate and reopen tools. When event context has accepted `none`, a scope transition produces complete location/date authority, or the shopper explicitly requests a refresh of an unchanged complete scope, the selected skill directly instructs that same agent to use its one weather attempt before answering. An unchanged comparison blocks weather unless it binds an exact-scope receipt for silent grounding. Deterministic code validates activation, grants, tool arguments, evidence, and final factual grounding; it does not decide whether the shopper intended weather or comparison work.
   - Catalog capabilities generate `search_catalog_tool`'s flat schema with exact taxonomy values and non-taxonomy required-constraint properties. Product meaning is model-owned: the active skills and tool descriptions instruct the model to author `requested_product_type`, select faithful advertised taxonomy, use a category-only scope only when it judges that category to be a faithful parent, and clarify directly when it cannot make a faithful selection. Runtime does not parse current or recent shopper prose, suffix-match product phrases, classify shopper-named versus open roles, or validate a semantic relationship between `requested_product_type` and taxonomy. It validates structural completeness, capability-derived values and types, the one-category bound, category/subcategory coherence, supported search mode, and advertised hard constraints. A category-only search records the model-authored requested role and searched category separately; grounding presents category-scoped candidates under their actual catalog categories without asserting a parent relationship or catalog absence.
   - Every search also carries required pre-retrieval `shopper_guidance`: one concise, product-agnostic sentence authored under the active skill. Any nonempty `unadvertised_requirements` lane fails closed without retrieval or repair. At most one structural catalog repair is available for the entire turn, not per semantic scope. It receives the current shopper message, typed search tool, compact capabilities, sanitized validator feedback, and active skill context. Independently valid `required_constraints`, `scope_complete`, and `search_mode` are preserved; `requested_product_type` and taxonomy are not locked from shopper prose. A no-tool repair produces the fixed clarification, and another validation failure after the repair closes to synthesis. Distinct valid taxonomy-plus-hard-constraint scopes may continue within the configured three-search cap.
   - Cart mutations require explicit product/cart-line refs. Grounding reads actual tool-role messages, separates current-request evidence from prior-turn evidence, and never treats an assistant draft as evidence. Successful searches preserve the taxonomy-independent semantic query as internal ranking evidence, the pre-retrieval `shopper_guidance` as product-agnostic response framing, and each confirmed filter set with the products from that search. A completed search gets one final tools-disabled model step under the active skill, followed by the grounding editor. Event-context turns without current or explicitly bound weather evidence also use that final editor when draft text exists; it receives only saved-ZIP-candidate presence, never digits, and the accepted next-question boundary, while deterministic grounded rendering restores candidates if a successful-search edit drops all of them. Current-turn non-weather business-tool evidence always uses ordinary grounding. A protected event decision renderer is selected structurally only when event context is active, there is no current non-weather business-tool activity, and a current typed weather outcome or one explicitly bound valid receipt exists. Missing location/venue or an empty draft skips its decision editor. A separate prior-candidate fallback uses deterministic event assembly only when the draft is empty. Product comparison with current resolution/detail work guarantees ordinary grounding: a bound receipt may guide styling silently, but exact forecast facts and the prior canonical block are not repeated. Other protected weather-evidence turns with a nonempty draft give a narrow tools-disabled decision editor only bounded shopper-authored event text and the server-owned deterministic weather styling direction; it must return an exact two-key JSON object with one exact shopper-authored `venue_quote` and one or two distinct allowlisted `adjustments`. Malformed output, an ungrounded or missing quote, an unknown/duplicate adjustment, or any extra key falls back. The server alone renders fixed allowlisted phrases, exact newest prior names when present, the deterministic weather direction, only the accepted typed question, and any current typed weather failure or current canonical success block. If the requested outcome depends on an unconfirmed material, fit, comfort, durability, care, weather, or other functional property, the response must disclose that gap and frame results as the closest catalog or styling direction rather than as proven suitable. If that draft or editor is unavailable, deterministic fallback uses search guidance, static skill `response_guidance`, returned names, prices, categories, and search-scoped confirmed filters, followed by the same generic unverified-property disclosure. Scoped zero-result evidence cannot establish absence outside its exact taxonomy and filters. The graph and grounding editor share one execution deadline; a grounding timeout finalizes as failed with `grounding_timeout`, uses the deterministic catalog renderer for search-only evidence, deterministic event assembly for the protected event path, and a fixed retry/cart-check response for other non-search turns. Outside that protected path, editor errors and empty or whitespace-only output use the same fail-closed response rule with `grounding_error`; invalid protected decisions fall back deterministically.
   - Any current non-weather business-tool activity also prevents successful-weather postprocessing from appending unrelated names from the historical index. On grounding timeout, error, or empty output, a current result named `get_product_details_tool` and beginning with the canonical successful-detail marker can enter a deterministic verified-detail fallback containing only its names, prices, categories, and listed fields, plus the typed weather outcome when present; it does not manufacture a comparison judgment. The fixed retry/cart-check fallback remains for non-search turns without current detail evidence. A weather-only turn with no business activity remains on the protected event path.
   - Optional output guardrails run, then the memory service finalizes the durable turn as completed, blocked, or failed before products, images, content, and metrics are emitted over SSE. An exact retry of a finalized request replays its stored response without model/tool work. Internal diagnostics include bounded current-turn product evidence from successful catalog search and detail results plus bounded `catalog_scope_outcomes` for zero-result scopes; each search scope remains attached to its own products. Activation diagnostics retain the model-submitted question and record the server-accepted question separately when normalization applies. Weather tool arguments/output are always redacted from diagnostics and failed-turn partial graph capture. The weather trace retains only categorical date shape, location-source kind, provider-input kind, typed outcome, and receipt lifecycle state such as `promotion_prepared` or `bound`; it never includes a receipt ID, receipt scope/evidence, location, ZIP, date, resolved place, URL, body, or exception. The saved profile ZIP is recursively scrubbed from diagnostic string keys and values. Public query responses contain an empty diagnostics object by default. `EXPOSE_AGENT_DIAGNOSTICS=true` exposes the detailed trace only for a trusted operator or evaluation deployment. Final-text extraction skips tool, tool-calling, and internal activation messages; if no shopper-facing answer exists, the runtime returns a safe fallback with `incomplete_agent_response`. On graph failure, bounded current-turn messages are captured before checkpoint cleanup.
   - The registered, read-only `get_weather_forecast_tool` uses a provider-neutral daily client with Visual Crossing as the first adapter. `WEATHER_ENABLED=false` remains the default, and startup/health checks make no request. The model-visible tool has no arguments: after activation, the runtime derives the provider location and normalized date window only from the effective typed scope. Incomplete scope, an active location/venue/date question, or a bound exact-scope receipt hides and execution-blocks only weather. Saved ZIP is released only by the narrow deterministic confirmation gate; shopper-provided locations must be exact current-turn spans, with optional qualifier-preserving region/country context and no invented ZIP. Visual Crossing resolves that named place in the same request. Exact ISO dates/ranges and bounded current-turn `next week` forms retain strict validation. A valid provider call may retry once only after timeout or HTTP 5xx. A paired success may be promoted atomically as one short-lived `weather_forecast.v1` receipt; failures, secrets, saved ZIP digits, request URLs, raw provider bodies, and exceptions are never stored. Prior forecast prose and prior tool messages are non-evidence. Only an explicitly bound exact-scope receipt can be reused, and product comparison uses it silently without repeating forecast facts. Weather never proves product performance or creates an unstated catalog constraint. No weather-specific FastAPI, SSE, or UI shape is added.
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
- Rolling-summary work selection and closed output validation:
  `chain_server/src/conversation_summary.py`
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
- Source-sequence-bound weather-subject resolver and forced typed control:
  `chain_server/src/weather_scope_resolver.py`
- Pure current-turn/prior-state weather authority compilation:
  `chain_server/src/weather_scope_authority.py`
- Shared typed weather-receipt contract, exact-scope identity, TTL, and cap:
  `shared/weather_receipts.py`
- Shared singleton current-weather-scope, source-turn model/reference helper, and transition contract:
  `shared/weather_scope.py`
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
- The source-bound cross-subject regression lives under
  `tests/integration/conversations/weather_scope_continuity/`. Its three turns
  establish one complete subject, introduce a different dated subject without
  importing the old location, and complete that subject through its bound
  location question.
- The activation-capability regression lives under
  `tests/integration/conversations/weather_activation_boundary/`. Its three
  turns create a source-bound pending weather question, run intervening product
  work with weather controls inert and without repeating the question, and then
  complete the pending scope.
- The focused receipt fixture starts with the explicit natural product request
  `Show me dress options for a semi-formal wedding`, so product discovery is
  unambiguous and the later turns isolate receipt promotion/binding. Its
  comparison turn forbids every repeated forecast fact, including conditions,
  temperatures, precipitation, resolved place, attribution, and uncertainty.
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
- Startup migration 8 adds the rolling-summary text and through-sequence
  watermark to each conversation projection. An optional finalize-time
  compare-and-swap update advances both atomically with output, events, the
  product index, and replay identity. Raw context reads contain only eligible
  turns after the watermark. The memory response separates the newest raw tail
  from an oldest exact compaction prefix of up to four turns. Default chain
  policy triggers at six unsummarized turns, retains at least two raw turns,
  selects the largest fitting contiguous oldest prefix, caps output at 4,096
  characters while reserving input headroom, and gives the tools-disabled
  compactor its own five-second timeout. A single oversized oldest turn is
  represented by marked head-and-tail excerpts only inside compactor input.
- Startup migration 9 adds the bounded typed-receipt JSON projection. Memory
  validates and atomically promotes only paired successful
  `weather_forecast.v1` evidence on completed finalization, prunes expired
  receipts, supersedes an older receipt for the same exact location/date
  scope, and returns at most four. The configured receipt TTL defaults to
  3,600 seconds and may not exceed 21,600 seconds. This lane stores no saved
  ZIP, provider request/response body, prepared provider endpoint URL, key,
  exception, or failure; it retains only the pinned public attribution URL.
- Startup migration 10 adds one versioned current weather-planning scope.
  Contract 3 retains the legacy `continue`/`replace` transition for rolling
  compatibility. Contract 4 atomically resolves both location and date through
  explicit `retain`/`set`/`clear` actions, checks the expected scope revision,
  and persists one optional pending `event_location` or `event_date` question.
  Memory stamps each newly set component's source turn and stamps a pending
  binding to the current finalized turn unless the server supplies the exact
  existing pending-source handle for an unchanged unanswered question.
  Completing a pending component requires the server-only exact handle;
  shared/memory recheck the live binding and canonical completion shape.
  Memory invalidates receipts when authority
  changes and admits scope-matched promotion only. This
  singleton contains no venue, occasion, product, styling, or forecast facts
  and is not an active-anchor registry.
- Startup migration 11 stores the v4 pending question and its source fields in
  defaulted `current_weather_pending_json`, extracts a complete pre-split WIP
  binding, drops incomplete unsourceable pending fields, and leaves
  `current_weather_scope_json` strict pre-v4-readable. Memory reattaches the
  pending binding only when its stored
  scope revision equals the core revision; a rollback-era core scope mutation
  therefore makes stale pending state inert.
- Startup migration 12 adds defaulted `current_weather_unavailable_json` for
  v6 source-bound location/date unavailability markers. It is separate from
  the strict core and pending JSON lanes, and memory merges it only when its
  stored scope revision matches the core revision. A later `set`, explicit
  `clear`, new-subject replacement, or rollback-era core write therefore makes
  an older marker inapplicable.
- Response contract 5 adds no SQLite migration. Memory derives at most three
  exact completed source turns from the existing scope pointers inside the
  start transaction, independent of the summary watermark and
  `MEMORY_RECENT_TURNS`. The chain validates exact pointer equality, hydrates
  the lane separately, and exposes it only to weather-subject resolution and
  protected styling provenance. Contracts v1–v4 do not receive the field.
- Response contract 6 adds source-bound `location_unavailable` and
  `window_unavailable` fields plus exact
  `decline_pending_source_turn_id` and
  `supersede_pending_source_turn_id` finalize controls. Memory rechecks the
  live binding and scope revision atomically. Contracts through v5 strip the
  markers and derive their source-turn lane from that same down-projected
  scope.
- Preserve memory's optional receipt conflict codes across the HTTP client
  boundary. The runtime retries finalization once without the optional
  promotion, without rerunning the model. A current-scope revision, resolution,
  or status conflict is different: the runtime discards the shopper draft and
  product artifacts and terminalizes the durable turn as failed without the
  disputed scope write.
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
  envelopes. The conversation projection owns the rolling summary pair; start
  returns only completed/failed raw turns with assistant text whose sequence is
  strictly later than its watermark. Summary, newest raw discussion, the
  historical product index, current weather scope, and active typed weather
  receipts remain distinct.
  The compactor folds only a validated contiguous boundary in memory's oldest
  exact prefix after a successful response, redacts prior canonical forecast
  blocks, and never receives the receipt projection or complete graph/tool
  transcript. Durable raw turns do not store the rendered shopper-context
  block, raw media, or model reasoning.
  Presented-product events, deterministic historical resolution, the singleton
  current weather scope, and bounded exact-scope weather receipts are
  implemented; active anchors and effective preferences remain reserved and
  unused.
- The memory API has no service authentication. Standard Compose limits its
  host mapping to `127.0.0.1:8011`; keep it on an internal network in other
  deployments. Detailed agent diagnostics remain internal and public query
  responses return `{}` unless `EXPOSE_AGENT_DIAGNOSTICS=true` is deliberately
  set behind a trusted operator or evaluation surface.
- Chain-server `/query/stream` and `/query/timing` payload logs retain only
  request/response character lengths and media/image counts. They never log raw
  shopper text, response content, ZIP/location/date values, receipt evidence,
  or weather-provider material.
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
  answer in exactly two short sentences and include only the
  activation-selected location, venue, or date question. With that context
  complete, use one short paragraph and ask no further event-context question.
  Ordinary shop-now
  occasion-only requests run one search for one grounded core role unless the
  shopper explicitly requests a complete look or names multiple roles. If
  location is still missing and materially changes the next recommendation,
  activation selects only event location alongside the results; with a saved-ZIP candidate,
  that question must ask whether the event is in the shopper's usual area or
  elsewhere, never discard the candidate for a bare destination question.
  After destination is established, activation may instead select one material
  venue/setting question. Saved ZIP is never shopping, shipping, availability,
  or current location. Never infer beach, outdoor/indoor setting, terrain,
  weather, wind, climate, or product performance from a ZIP or place name.
- `get_weather_forecast_tool` is registered read-only and granted only by
  `event-context`, whose composition still requires `outfit-styling`. It may
  receive one zero-argument model-visible attempt when the effective typed
  scope contains both location and date. Without that complete scope, the
  runtime hides and execution-blocks the tool. Scope compilation additionally
  requires either confirmed saved-area authority or an exact location span from
  the current shopper turn. Within a scope-valid call,
  `max_provider_attempts: 2` permits one internal retry
  only after timeout or HTTP 5xx. HTTP 400 maps to generic
  `weather_request_invalid`; other 4xx, connection, and response-validation
  failures are not retried. The mandatory activation call owns
  `event_context_next_question`, required exactly when `event-context` is
  selected and omitted otherwise. It is the model's semantic choice from
  conversation plus the typed current scope:
  `event_location` only when destination is missing and material,
  `event_venue` only after destination is established when venue or setting is
  missing and material, `event_date` only after destination and any material
  venue are established when weather is enabled and material and the typed
  scope has no bounded date, and `none` otherwise. When a prior scope exists,
  an isolated tools-disabled resolver compares its exact source-sequence-bound
  shopper turns with the current query and emits one forced typed control call.
  It is not a registered business tool or a subagent. Activation then consumes
  or proposes one atomic resolution that copies the scope revision and chooses
  `retain`, `set`, `clear`, or `unavailable` for both components; only the
  current turn can supply `set` or `unavailable` authority. `clear` is ordinary
  askable absence, while `unavailable` is a source-bound current-subject state
  that is reset by `set`, explicit `clear`, or subject replacement. Invalid,
  timed-out, unavailable, or unclear resolver
  output fails closed. An approved `same_subject/not_addressed` outcome
  authorizes activation's explicit retains, while `new_subject` clears them.
  `same_subject/answered` is accepted only for the exact typed binding: the
  resolver echoes its opaque source-turn handle and activation sets the named
  component. `unchanged/resume_requested` uses that same exact binding only to
  ask the stored question again without a scope write when activation supplies
  no current-turn scope facts. Any independently validated current-turn action
  still survives.
  `same_subject/declined` uses the exact binding and a target
  `unavailable` action to consume only that pending question, select no
  replacement location/date question, and continue without weather. A
  `new_subject` resolution against a live pending binding uses the exact
  supersession handle and may retain no old component. These semantic choices
  are model-owned; deterministic code does not recognize phrases or route
  intent. Either unavailability marker suppresses location/date follow-ups but
  not the independent styling-only venue question.
  The accepted
  location/date question is stored as a pending binding even when no authority
  value changes. An unavailable or unclear resolver clears every proposed
  retain and blocks prior-dependent weather, but a validated current-turn
  `set`/`set` replacement remains independent authority.
  An explicitly shopper-stated outdoor patio, beach, garden, rooftop, or
  open-air setting makes enabled live weather material; with destination and
  that setting but no bounded date, select `event_date`. Skill selection,
  location, venue, materiality, and intent remain model-owned semantic
  guidance. The dynamic enum is typed argument consistency, not an intent
  router or keyword routing layer.
  Only the value from a successfully
  completed activation is trusted; the server does not infer a question from
  weather configuration or a missing date. The same activation may update the
  typed singleton with an atomic resolution, or bind one currently valid
  `weather_receipt_id` only with `event_context_next_question=none`, no scope
  update, no refresh request, and exact equality to the effective scope.
  Unbound receipts never ground, and binding one blocks another weather call.
  A scope update that produces a complete effective scope requires the
  zero-argument weather call and rejects a text-only model response. For an
  unchanged complete scope, only explicit `weather_refresh=true` does so;
  comparisons and other turns do not auto-refresh.
  Accepted `event_location` or `event_venue` also hides and
  execution-blocks weather. Event context is additive and may gate only
  weather: all non-weather tools in the selected grant union remain available,
  and consuming the one forecast attempt does not close business-tool work.
  Only accepted `event_date` may render a date question. Current-turn product,
  comparison, cart, and policy evidence always uses ordinary grounding. A
  protected event decision renderer is selected structurally only when event
  context is active, there is no current non-weather business-tool activity,
  and a current typed weather outcome (success or failure) or explicitly bound
  valid receipt exists. Missing
  location/venue or an empty draft skips that editor. A separate prior-candidate
  fallback uses deterministic event assembly only when the draft is empty.
  Product comparison with current resolution/details remains on ordinary
  grounding and uses a bound receipt silently without repeating forecast
  facts. The execution tool accepts no arguments. Saved mode reaches weather
  only through the narrow current-turn confirmation gate: a location-neutral statement explicitly
  naming `my`/`the` usual/home area, a bare affirmative immediately after the
  assistant asks about that area, or the immediately following strict date-only
  turn. Any explicit current location, negation, uncertainty, or override
  rejects saved mode; explicit shopper location always wins. Shopper-provided
  mode keeps the exact current-turn city, region/country, address, or postal
  phrase in the scope. For an abbreviation or ambiguous name, a provider
  qualifier is
  required: it must preserve that exact phrase as its first component and
  append only one or two comma-separated region/country qualifiers. Keep
  `location="NYC"` and use `location_query="NYC, NY"`; `Springfield, TX` is a
  valid explicit regional assumption. Never add an unstated ZIP or numeric
  component or replace the shopper-authored authority phrase. Omit the query
  only when the location is already sufficiently qualified. The adapter passes
  the compiled provider location to Visual Crossing Timeline. Do not synthesize a ZIP or
  use a separate geocoder; Visual Crossing's `resolvedAddress` becomes the
  reversible `resolved_location` assumption.
  Semantic equivalence remains model-owned rather than deterministic proof and
  is correctable through the disclosed provider resolution.
  Require an exact ISO date/range except for exact shopper-authored `next week`.
  Exact `<weekday> next week` requires the matching lowercase `weekday` and is
  resolved server-side to that day inside the next Monday-through-Sunday
  window; bare `next week` omits `weekday` and resolves to the full range.
  Omitted, mismatched, mixed, negated, or superseded weekday authority fails
  closed. The model may normalize an unambiguous
  single-day phrase such as `tomorrow` to an exact ISO date against that same
  prompt-visible UTC anchor. This UTC anchor is an explicit current limitation:
  caller- or shopper-local timezone anchoring is not yet implemented. For a
  genuinely ambiguous or unresolved relative
  date, the activation model may choose `event_date` only when weather is
  enabled and material and the date is neither established nor explicitly
  unavailable.
  Current successful weather output has precedence. A later turn may reuse
  only one still-valid `weather_forecast.v1` receipt explicitly bound during
  activation for the exact same location/date scope; an unbound receipt
  is never evidence, and a bound receipt blocks another weather call. Changed,
  uncertain, or refresh-requested scope requires fresh evidence. Raw
  arguments/output are redacted from tool diagnostics and failed-turn partial
  output; diagnostics retain only categorical `request_shape`,
  `location_source`, `provider_input`, `outcome`, and receipt lifecycle state,
  never a receipt ID or scope/evidence, place, ZIP, date, resolved place, URL,
  body, or exception. Diagnostic string keys and values
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
  For ordinary-grounding weather paths, grounding-editor sentences containing
  weather-domain fact language or fact-shaped dates/values are removed while
  ordinary grounded styling language remains. Missing location/venue or an empty
  draft skips the protected weather-outcome decision editor. The separate
  prior-candidate-only empty-draft branch assembles deterministically. Other
  protected weather-outcome turns expose only bounded shopper-authored event
  text and a server-owned deterministic weather direction, and accept only
  exact JSON with a grounded `venue_quote` plus one or two distinct allowlisted
  adjustment codes. Invalid output falls back; the server maps codes to fixed
  phrases and assembles exact newest names when present, its deterministic
  weather direction, only the accepted question, and a current typed weather
  outcome/current canonical block. A bound receipt used beside current product
  resolution or details guides comparison silently; exact forecast facts and
  the prior canonical block are not repeated. Never ask for state, region,
  country, or a finer location solely because provider lookup failed.
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
