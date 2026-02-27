# AGENTS.md

This file is a working guide for coding agents and contributors in this repository.

## 1) Project Summary

Retail Shopping Assistant is a multi-service application with:
- `chain_server`: FastAPI + LangGraph orchestration for planner/retriever/cart/chatter/summary agents.
- `catalog_retriever`: FastAPI service for text/image embedding retrieval against Milvus.
- `memory_retriever`: FastAPI + SQLite service for per-user context and cart state.
- `guardrails`: FastAPI wrapper around NeMo Guardrails input/output safety checks.
- `ui`: React + TypeScript chat UI using SSE streaming.
- `shared`: Shared YAML configs, product CSV data, and image assets.

Top-level orchestration is via `docker-compose.yaml`; optional local NIM model containers are in `docker-compose-nim-local.yaml`.

## 2) Architecture and Request Flow

1. UI posts to `/api/query/stream` (nginx proxy on port `3000`).
2. Nginx routes `/api/*` to `chain-server:8009`.
3. Chain server graph flow:
   - `memory_node` pulls context/cart from memory service.
   - `planner_node` selects `cart`, `retriever`, or `chatter`.
   - `rails_input_node` runs guardrails input check in parallel.
   - Selected agent runs, then chatter produces streamed response.
   - `rails_output_node` checks final response safety.
   - `summary_node` persists summarized context back to memory service.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image.

## 3) Source Map (Where to Change What)

- Agent orchestration: `chain_server/src/graph.py`
- API contract and SSE endpoint: `chain_server/src/main.py`
- Routing logic: `chain_server/src/planner.py`
- Catalog query logic from chain side: `chain_server/src/retriever.py`
- Cart tool behavior: `chain_server/src/cart.py`
- Streamed generation: `chain_server/src/chatter.py`
- Context summarization/persistence: `chain_server/src/summarizer.py`
- Shared chain models/tools: `chain_server/src/agenttypes.py`, `chain_server/src/functions.py`

- Catalog API entrypoints: `catalog_retriever/src/main.py`
- Embedding/retrieval/reranking/category filtering: `catalog_retriever/src/retriever.py`
- Image/base64 helpers: `catalog_retriever/src/utils.py`

- Memory API and SQLite schema: `memory_retriever/src/main.py`

- Guardrails API: `guardrails/src/main.py`
- Guardrails engine/wiring: `guardrails/src/rails.py`
- Guardrails config override helper: `guardrails/src/config_utils.py`

- UI streaming behavior: `ui/src/components/chatbox/chatbox.tsx`
- UI API config and feature flags: `ui/src/config/config.ts`
- UI message/cart-toast parsing helpers: `ui/src/utils/index.ts`

- Shared config roots:
  - `shared/configs/chain_server/`
  - `shared/configs/catalog_retriever/`
  - `shared/configs/rails/`

## 4) Runbook

### Cloud endpoint mode (no local NIM containers)

```bash
export NGC_API_KEY=<your_key>
export LLM_API_KEY=$NGC_API_KEY
export EMBED_API_KEY=$NGC_API_KEY
export RAIL_API_KEY=$NGC_API_KEY
export CONFIG_OVERRIDE=config-build.yaml
docker compose -f docker-compose.yaml up -d --build
```

### Local NIM mode (requires multi-GPU setup)

```bash
export NGC_API_KEY=<your_key>
export LLM_API_KEY=$NGC_API_KEY
export EMBED_API_KEY=$NGC_API_KEY
export RAIL_API_KEY=$NGC_API_KEY
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
docker compose -f docker-compose-nim-local.yaml up -d
docker compose -f docker-compose.yaml up -d --build
```

### Health checks

```bash
curl -sS http://localhost:3000            # UI via nginx
curl -sS http://localhost:8009/health     # chain server
curl -sS http://localhost:8010/health     # catalog retriever
curl -sS http://localhost:8011/health     # memory retriever
```

## 5) Testing and Validation

There is no comprehensive unit-test suite across all services yet.

Current test assets:
- `guardrails/test/test_rails.py` (basic/stub-like unittest coverage)
- `tests/` scripts for conversation/timing/quality evaluation, driven by live endpoints and YAML scenario files.

Useful test workflow:
1. Bring services up with Docker Compose.
2. Verify health endpoints.
3. Run conversation eval scripts in `tests/` (requires `TEST_PATH` and expected conversation folders).

## 6) Configuration Rules

- Chain server loads `/app/shared/configs/chain_server/config.yaml` and optionally merges `CONFIG_OVERRIDE` from the same directory.
- Catalog retriever and guardrails use the same override pattern.
- Override files are shallow-merged (top-level keys); nested structures are not deep-merged.

Key env vars:
- `LLM_API_KEY`
- `EMBED_API_KEY`
- `RAIL_API_KEY` / `NVIDIA_API_KEY` (guardrails container)
- `CONFIG_OVERRIDE`
- `NGC_API_KEY` (for local NIM containers)

## 7) Important Gotchas

- Ports in docs are not always aligned with runtime wiring.
  - Actual backend service port is `8009` in compose.
  - External app entrypoint is usually `http://localhost:3000` through nginx.
- UI API base URL is hard-coded to `/api` (nginx path), not direct service URLs.
- Memory store is SQLite in-container (`context.db`); data lifecycle depends on container persistence.
- Cart add/remove uses catalog similarity checks before memory mutation.
- For image search, catalog retriever bypasses category filtering and relies on similarity ranking.

## 8) Contribution and Commit Notes

- Follow `CONTRIBUTING.md` requirements.
- Use signed commits (`git commit -s`) for contributions.
- Keep changes scoped by service; avoid cross-service behavior changes without updating related config/docs.

## 9) Recommended Change Workflow for Agents

1. Identify impacted service(s) and config files.
2. Implement smallest coherent change.
3. Validate via health + targeted scenario.
4. If API shape changes, update docs in `docs/API.md` and any UI assumptions.
5. Note any config/env additions in docs.

