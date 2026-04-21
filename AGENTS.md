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
   - `memory_node` pulls context/cart (with prices) from memory service.
   - `planner_node` selects `cart`, `retriever`, or `chatter`. The user query is prefixed with `IMAGE ATTACHED: yes/no` so deictic image queries route to `retriever`.
   - `rails_input_node` runs guardrails input check in parallel.
   - Selected agent runs, then chatter produces streamed response. Chatter sees structured grounding fields (`PRECEDING AGENT`, `PRECEDING AGENT RESULT`, `CURRENT CART`, `AVAILABLE CATALOG`, `RECENT DISCUSSION`) and must not invent cart actions.
   - `rails_output_node` checks final response safety.
   - `summary_node` persists summarized context back to memory service.
4. For product discovery, chain server calls catalog retriever:
   - `/query/text` for text-only.
   - `/query/image` for text + image. With an image attached, the extractor returns empty `search_entities` for filter-only refinements but still emits price filters.

## 3) Source Map (Where to Change What)

- Agent orchestration: `chain_server/src/graph.py`
- API contract and SSE endpoint: `chain_server/src/main.py`
- Routing logic + image-attached signal: `chain_server/src/planner.py`
- Catalog extraction (image-aware rules, filter sanitization): `chain_server/src/retriever.py`
- Cart tools (`add_to_cart`, `remove_from_cart`, `view_cart_total`) + deterministic name/pronoun resolver: `chain_server/src/cart.py`
- Streamed generation with structured grounding: `chain_server/src/chatter.py`
- Context summarization/persistence: `chain_server/src/summarizer.py`
- Shared chain models/tools + XML/JSON tool-call fallback parser: `chain_server/src/agenttypes.py`, `chain_server/src/functions.py`

- Catalog API entrypoints: `catalog_retriever/src/main.py`
- Embedding/retrieval/reranking/category filtering: `catalog_retriever/src/retriever.py`
- Image/base64 helpers: `catalog_retriever/src/utils.py`

- Memory API and SQLite schema (`CartItem` includes `price`, with idempotent migration): `memory_retriever/src/main.py`

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

Brings up the local LLM (`nemotron` service, image `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`), `nvclip`, `embedqa`, and the two NemoGuard guardrail containers.

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

The `nemotron` service is launched with `NIM_PASSTHROUGH_ARGS=--enable-auto-tool-choice --tool-call-parser llama3_json` so vLLM accepts `tool_choice="auto"`. Reasoning output is suppressed via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on the chain-server side so streamed tokens flow eagerly.

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
- `CartItem` rows carry a `price` column; the deterministic `view_cart_total` tool uses these prices instead of letting the LLM do arithmetic. Older DBs are auto-migrated by `_ensure_price_column`.
- Cart add/remove uses catalog name matching (with normalization + Jaccard fallback) before memory mutation; pure semantic similarity on descriptions is no longer used.
- The cart agent deterministically resolves pronouns (`it`, `this`) against the most recent product in `RECENT DISCUSSION` and overrides the LLM's `item_name` if they disagree.
- For image search, catalog retriever bypasses category filtering and relies on similarity ranking. Top-k is applied before price filters, so a tight budget on a high-priced image-similarity cluster can legitimately return zero matches.
- Local LLM service is named `nemotron` (was `llama`); chain-server reaches it at `http://nemotron:8000/v1` per `shared/configs/chain_server/config.yaml`. Cloud override `config-build.yaml` still uses `meta/llama-3.1-70b-instruct` on `build.nvidia.com`.
- Tool calling against the local NIM requires `--enable-auto-tool-choice --tool-call-parser llama3_json` passthrough args. Without them, requests with `tool_choice="auto"` 400.
- Nemotron sometimes returns tool calls as XML/JSON inside the assistant `content` field instead of `message.tool_calls`. `chain_server/src/functions.py::parse_tool_call_fallback` handles both shapes; `_coerce_value` parses stringified Python literals (`"[]"`, `"{'k':'v'}"`) so list/dict args don't reach downstream code as strings.
- Planner LLM input is prefixed with `IMAGE ATTACHED: yes/no`. With an image attached, deictic queries ("do you have this under $X", "find similar") route to `retriever`, not `chatter`. Only explicit cart operations (`add this`, `buy this`) still go to `cart_node`.
- Chatter is strictly grounded in `CURRENT CART` + `AVAILABLE CATALOG` + `PRECEDING AGENT RESULT`; it must not claim a cart mutation unless the cart agent ran this turn, and it must not invent product names absent from those fields.

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

