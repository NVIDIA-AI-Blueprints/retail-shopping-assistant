# AGENTS.md

This file is a working guide for coding agents and contributors in this repository.

## 1) Project Summary

Retail Shopping Assistant is a multi-service application with:
- `chain_server`: FastAPI + Deep Agents SDK orchestration over deterministic catalog and cart tools.
- `catalog_retriever`: FastAPI service for text/image embedding retrieval against Milvus.
- `memory_retriever`: FastAPI + SQLite service for per-user context and cart state.
- `guardrails`: FastAPI wrapper around NeMo Guardrails input/output safety checks.
- `ui`: React + TypeScript chat UI using SSE streaming.
- `shared`: Shared YAML configs, JSONL catalog data/role sidecars, and image assets.

Top-level orchestration is via `docker-compose.yaml`; optional local NIM model containers are in `docker-compose-nim-local.yaml`.

## 2) Architecture and Request Flow

1. UI posts to `/api/query/stream` (nginx proxy on port `3000`).
2. Nginx routes `/api/*` to `chain-server:8009`.
3. Chain server request flow:
   - `DeepAgentsRuntime` loads scoped context and the authoritative cart from the memory service.
   - Optional input guardrails run before model/tool work; attached media is analyzed through the configured perception client.
   - The runtime supplies the Deep Agents model with a compact cached catalog-capability projection and registered catalog, product-detail, and cart tools.
   - Catalog constraints are validated deterministically before retrieval, cart mutations require explicit product/cart-line refs, and a grounding editor removes unsupported product claims.
   - Optional output guardrails run, updated context is persisted, and products, images, content, and metrics are emitted over SSE.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image.
   - `/products/{product_id}` for deterministic details after a search ref is known.
   - `/capabilities` once per successful chain-server process lifecycle for the catalog-owned query contract.

## 3) Source Map (Where to Change What)

- Serving agent orchestration and registered tools: `chain_server/src/deepagents_runtime.py`
- API contract and SSE endpoint: `chain_server/src/main.py`
- Catalog capability cache/prompt projection: `chain_server/src/catalog_capabilities.py`
- Catalog intent validation/execution: `chain_server/src/catalog_request.py`, `chain_server/src/catalog_execution.py`
- Commerce service adapters: `chain_server/src/commerce_tools.py`
- Image/video perception: `chain_server/src/media_perception.py`
- Shared request/state models: `chain_server/src/agenttypes.py`
- `graph.py`, `planner.py`, `retriever.py`, `cart.py`, `chatter.py`, and `summarizer.py` are legacy compatibility paths, not the serving runtime.

- Catalog API entrypoints and request validation: `catalog_retriever/src/main.py`
- JSONL loading/search-document construction: `catalog_retriever/src/catalog.py`
- Dynamic catalog capabilities: `catalog_retriever/src/capabilities.py`
- Embedding retrieval, deterministic ranking, and filtering: `catalog_retriever/src/retriever.py`
- Image/base64 helpers: `catalog_retriever/src/utils.py`

- Memory API and SQLite schema (`CartItem` includes `price`, with idempotent migration): `memory_retriever/src/main.py`

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
- Service behavior lives in `chain_server/config.yaml`, `catalog_retriever/config.yaml`, and `rails/config.yml`.
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
- `CATALOG_DATA_SOURCE`, `CATALOG_SCHEMA_SOURCE`
- `SHARED_CONFIG_ROOT` (local runner / non-container config root)
- `SHARED_ROOT` (local runner / non-container shared asset root)
- `REACT_APP_API_BASE_URL` (local React dev server API target)

## 7) Important Gotchas

- Ports in docs are not always aligned with runtime wiring.
  - Actual backend service port is `8009` in compose.
  - External app entrypoint is usually `http://localhost:3000` through nginx.
- UI API base URL defaults to `/api` (nginx path), but local runner overrides it to `http://localhost:8009`.
- Memory store is SQLite in-container (`context.db`); data lifecycle depends on container persistence.
- `CartItem` rows carry a `price` column; the deterministic `view_cart_total` tool uses these prices instead of letting the LLM do arithmetic. Older DBs are auto-migrated by `_ensure_price_column`.
- The serving cart tools require a `PRODUCT_REF` from catalog search for adds and
  a `CART_LINE_ID` from cart state for removals. Adds revalidate the product
  against the active catalog before changing memory state.
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
- Local LLM service is named `nemotron` (was `llama`); chain-server reaches it through `shared/configs/models.yaml` when the app LLM role uses `source: local_nim`.
- Tool calling against the local NIM requires `--enable-auto-tool-choice --tool-call-parser llama3_json` passthrough args. Without them, requests with `tool_choice="auto"` 400.
- The Deep Agents model chooses among registered tools directly; there is no
  serving planner/retriever/chatter graph. Media analysis and the cached catalog
  contract are included in its turn context.
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
