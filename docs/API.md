# 🛍️ Retail Shopping Assistant API Documentation

## 📋 Table of Contents

- [Overview](#-overview)
- [Base URL](#-base-url)
- [Authentication](#-authentication)
- [Data Models](#-data-models)
- [Endpoints](#-endpoints)
- [Error Handling](#-error-handling)
- [Rate Limiting](#-rate-limiting)
- [Examples](#-examples)
- [Client Integration](#-client-integration)
- [Notes](#-notes)

## 🎯 Overview

The Retail Shopping Assistant API provides a comprehensive interface for an AI-powered retail shopping advisor. The API is built on a microservices architecture and currently uses the Deep Agents SDK as the chain-server assistant harness.

### Key Features

- **Real-time Streaming**: Server-Sent Events (SSE) for live responses
- **Multi-modal Input**: Text queries, image uploads, and optional VLM-backed video/image understanding
- **Shopping Cart Management**: Add, remove, and view cart items
- **Content Safety**: Built-in guardrails for safe interactions
- **Performance Monitoring**: Detailed timing information

## 🌐 Base URL

```
http://localhost:3000/api   # normal UI/nginx entrypoint
http://localhost:8009       # direct chain-server development endpoint
```

The catalog retriever is an internal service at `http://localhost:8010`. See
[Catalog Architecture](CATALOG_REFACTOR_PLAN.md) for the ingest, capability,
agent-discovery, and validation flow.

The memory retriever is an internal single-replica service at
`http://localhost:8011`. Its durable-turn endpoints are documented below for
operators and service integrations; they are not browser-facing APIs.

## 🔐 Authentication

Currently, the API does not require authentication for local deployments. For production deployments, consider implementing API key authentication or OAuth2.

## 📊 Data Models

### Catalog Capabilities

The catalog retriever owns its filter and metadata capability contract. Chain
server request-building code should consume this contract instead of inferring
filterability from product text or hard-coded category lists.

Field roles come from `shared/data/enriched_products.schema.yaml`. Enum/list
values, numeric ranges, taxonomy nodes, and field coverage are discovered from
the configured JSONL. See [Catalog Schema and Filters](CATALOG_FILTERS.md).

The chain server caches the first successfully fetched full contract for its
process lifetime and uses that object for deterministic request validation.
The LLM receives only a compact projection of the fields, values, taxonomy
keys/scopes, and semantic/filter roles it needs; the full response below is not
copied into every prompt.

```typescript
interface CatalogCapabilities {
  catalog_id: string;
  product_count: number;
  retrieval_modes: Array<'text' | 'image' | 'hybrid'>;
  image_search_enabled: boolean;
  fields: Record<string, {
    type: 'enum' | 'enum_list' | 'number' | 'text' | 'unclassified';
    observed_type?: string;
    filterable: boolean;
    searchable: boolean;
    detail: boolean;
    taxonomy: boolean;
    operators: string[];
    source_fields: string[];
    coverage: {present: number; total: number};
    values: Array<{value: string; count: number}>;
    min_value?: number;
    max_value?: number;
  }>;
  taxonomy: {
    category_field?: string;
    subcategory_field?: string;
    categories: Record<string, {
      product_count: number;
      filters: Record<string, unknown>;
      semantic_fields: Record<string, unknown>;
      subcategories: Record<string, {
        product_count: number;
        filters: Record<string, unknown>;
        semantic_fields: Record<string, unknown>;
      }>;
    }>;
  };
  // Flat compatibility projection of fields where filterable=true.
  filters: Record<string, {
    type: 'enum' | 'enum_list' | 'number' | 'text';
    operators: string[];
    source_fields: string[];
    values?: string[];
    min_value?: number;
    max_value?: number;
    request_aliases?: Record<string, string>;
  }>;
}
```

`fields` and nested `taxonomy` are authoritative. `filters` remains for
compatibility and is generated from the same field roles. Do not add static
catalog values to chain-server config, UI code, prompts, or the sidecar.
Catalog ingestion requires each ordered taxonomy field to be a scalar `enum`
with `filter` use so the agent's generic taxonomy envelope can always be
enforced; the field names and observed values remain catalog-owned.

### QueryRequest

The main request model for all shopping queries.

```typescript
interface QueryRequest {
  user_id: number;                    // Unique user identifier
  query: string;                      // User's text query
  image?: string;                     // Legacy base64/data-URL image (optional)
  media?: MediaAttachment[];          // Image/video attachments (optional)
  session_id?: string;                // Optional website/browser session identifier
  conversation_id?: string;           // Optional chat thread identifier
  cart_id?: string;                   // Optional cart identifier
  request_id?: string;                // Optional stable ID for exact whole-turn replay
  context?: string;                   // Previous conversation context
  cart?: Cart;                        // Current shopping cart state
  retrieved?: Record<string, string>; // Previously retrieved products
  guardrails?: boolean;               // Enable content safety (default: chain-server config; true by default)
  image_bool?: boolean;               // Indicate if image is provided (default: false)
}

interface MediaAttachment {
  type: 'image' | 'video';
  data: string;                       // Base64 data URL, or raw base64 with mime_type
  mime_type: string;                  // image/jpeg, image/png, or video/mp4 by default
  filename?: string;
}
```

**Example:**
```json
{
  "user_id": 123,
  "query": "Show me red dresses under $100",
  "image": "",
  "media": [],
  "session_id": "session_abc",
  "conversation_id": "conversation_abc",
  "cart_id": "cart_abc",
  "request_id": "request_abc",
  "context": "Previous conversation about summer clothing",
  "cart": {
    "contents": [
      {
        "item": "blue_shirt",
        "amount": 2
      }
    ]
  },
  "retrieved": {
    "product1": "https://example.com/product1.jpg"
  },
  "guardrails": true,
  "image_bool": false
}
```

`session_id`, `conversation_id`, `cart_id`, and `request_id` are optional for
backward compatibility. When the scoped IDs are omitted, the server maps the
legacy `user_id` to internal compatibility identifiers; when `request_id` is
omitted, it generates a new UUID for the turn. A caller retrying the same exact
turn should reuse its request ID. The memory service stores the request digest
with the durable turn: an identical finalized retry replays the stored response,
products, retrieved images, and diagnostics without model/tool work or another
finalize call. Reusing the request ID with different shopper text or media is a
conflict. The same request ID also derives stable cart-mutation idempotency keys.

The bundled UI creates browser-session identifiers and sends them on every
turn. When supplied, `conversation_id` scopes durable raw turns, presented-
product evidence, and historical resolution; `cart_id` scopes cart reads/writes.
The Deep Agents working graph is request-scoped under a collision-safe pair of
conversation ID and request ID, not used as durable shopper memory, and deleted
after successful turn finalization. Production website integrations should move these IDs to a
server-owned session/thread service before broad rollout so customer context
and cart state cannot bleed across sessions.

Durable ordered shopper/assistant turns live in the single-replica
memory-service SQLite database. At turn start the runtime consumes a bounded set
of finalized recent turns and a compact product-reference projection in place
of the legacy rolling context blob. Products enter durable reference evidence
only when they appear in the finalized ordered `product_results` sent as product
cards. The selected discovery, styling, or cart skill may conditionally resolve
typed references against those events. Exactly one match becomes request-local
evidence; zero or multiple matches require clarification and do not authorize a
detail, availability, or cart tool. The deterministic resolver adds no separate
model or catalog call.

`MemorySaver` remains process-local and is not shared across workers or
replicas, but its graph thread now exists only for one request. It is deleted
after successful durable finalization and preserved if finalization fails. The
durable resolver is same-conversation only and does not implement fuzzy or
embedding matching, preferences, sentiment, active anchors, cross-conversation
lookup, or stale-catalog-revision handling.

Caller-supplied persona data is not part of `QueryRequest` and is not injected
into model context. Persona support remains deferred until the API has a typed,
bounded schema, authenticated ownership, input-safety validation, and an
explicit untrusted-data boundary.

`image` remains supported for backward compatibility and is normalized into the
same internal media list as `media[]`. New clients should use `media[]` for
video uploads. The bundled UI calls `/capabilities` on load and enforces the
configured media counts, MIME types, byte limits, and video duration limit. That
same endpoint also exposes non-secret model names and catalog filter metadata
for future UI controls.

### Multi-modal Input

Uploaded images can be used in two ways:

- Image embedding search through the catalog retriever when image embeddings
  are configured.
- Optional VLM media perception when the `vlm` model role is enabled in
  `shared/configs/models.yaml`.

Uploaded videos require VLM media perception. If VLM is disabled, video
understanding is unavailable and the assistant should not invent visual
details. The VLM analysis is passed to the Deep Agents runtime as concise text
context; raw media is not persisted into conversation memory.

If VLM media perception is unavailable and the user request depends on the
attached media, for example "shoes she is wearing in this video", the assistant
returns a direct explanation and asks for a text description instead of running
a long catalog or LLM turn with unsupported visual assumptions.

Descriptive media requests such as "what's in this look" or "describe this
outfit" are answered from VLM media analysis and should not trigger catalog
retrieval or product image cards. Catalog retrieval is reserved for explicit
shopping intent, such as finding similar items, checking availability or price,
asking for recommendations, or adding an item to the cart.

### QueryResponse

The response model for non-streaming queries.

```typescript
interface QueryResponse {
  response: string;                   // Generated response text
  images: Record<string, string>;     // Product images
  cart: Cart;                         // Authoritative cart snapshot after the turn
  timings: Record<string, number>;    // Performance timing data
  token_usage?: TokenUsage;           // LLM token usage summary
  model_usage?: ModelUsage;           // Per-role model usage summary
  agent_diagnostics?: AgentDiagnostics; // Ordered agent/tool trace
}
```

`AgentDiagnostics` is operator-facing turn metadata. It is additive and must
not be rendered as assistant prose.

```typescript
interface AgentDiagnostics {
  skill_files_read: string[];
  tool_calls: Array<{
    sequence: number;
    tool_name: string;
    arguments: Record<string, unknown>;
    status: 'completed' | 'rejected' | 'error' | 'pending';
    rejection_reason?: string;
    duplicate?: boolean;
    restored_fields?: string[];       // Bounded names of server-restored locks
  }>;
  rejected_tool_calls: number[];       // Sequence numbers in tool_calls
  duplicate_tool_calls: number[];      // Rejected duplicate-call subset
  product_evidence: Array<{
    product_ref: string;
    product_name: string;
    source_tool: 'search_catalog_tool' | 'get_product_details_tool';
    evidence_type: 'search_result' | 'product_detail';
    facts: Record<string, unknown>;
    search_scope?: {
      taxonomy: Record<string, unknown>;
      confirmed_filters: Record<string, unknown>;
    };
  }>;
  product_evidence_truncated: boolean;
  catalog_scope_outcomes: Array<{
    outcome: 'no_direct_catalog_match' | 'zero_results';
    requested_product_type?: string | null;
    taxonomy?: Record<string, unknown>;
    confirmed_filters?: Record<string, unknown>;
  }>;
  final_termination_reason: string;
  partial_graph_messages: Array<{
    type: string;
    content: string;
    name?: string;
    tool_call_id?: string;
    tool_calls?: Array<Record<string, unknown>>;
    truncated?: boolean;
  }>;
  partial_graph_messages_truncated?: boolean;
  partial_graph_capture_error?: string;
  diagnostic_collection_error?: string;
  memory_finalize_error?: string;
}
```

`final_termination_reason: "memory_start_failed"` means the required durable
start failed before guardrail/model/tool work. A generic finalize transport or
service failure does not replace grounded shopper text; it sets
`memory_finalize_error` and preserves the request checkpoint for recovery.
A superseded attempt is the deliberate exception: its stale response is replaced
with the safe attempt-fencing response described below.

Successful turns leave `partial_graph_messages` empty. Before a failed graph
checkpoint is deleted, the runtime reads its latest state and preserves up to
the final 24 current-turn assistant/tool messages, with each content field
bounded to 2,000 characters. Tool calls remain in model-issued order even when
parallel tool results finish in another order. `skill_files_read` contains only
successfully activated/injected `/shopper/.../SKILL.md` paths and successful
explicit reads of those files; other reference-file reads remain visible in
`tool_calls`. Every Deep Agents turn starts with an internal
`activate_shopper_skills_tool` call. A pre-activation or same-batch shopping
call is rejected with `rejection_reason: "skill_activation_required"`.
For a one-shot catalog repair, independently valid finite locked fields are
restored before execution rather than delegated back to the model.
`restored_fields` contains only the bounded field names restored on that tool
call, never a second copy of their values.

`product_evidence` contains only structured records derived from successful
current-turn `search_catalog_tool` and `get_product_details_tool` result
messages. It is limited to 24 records and 32,000 serialized characters;
`product_evidence_truncated` is `true` when either bound omits evidence. Search
records keep their taxonomy and confirmed filters in `search_scope`, attached
only to products returned by that search. The list excludes semantic queries,
raw tool messages, model reasoning, and other diagnostic fields.
`catalog_scope_outcomes` contains at most eight server-authored, product-free
outcomes. Its only accepted outcomes are `no_direct_catalog_match` and
`zero_results`; records may include only `requested_product_type`, `taxonomy`,
and `confirmed_filters` in addition to `outcome`. Evaluation consumers default
missing legacy list fields to `[]` and `product_evidence_truncated` to `false`.

Successful internal search-tool results carry `SEARCH_DIRECTION_EVIDENCE`, the
model-authored semantic query used as an independent private ranking preference,
and required pre-retrieval `shopper_guidance` authored under the active skill.
For completed search-only turns, the runtime runs one tools-disabled synthesis
under the active skill and grounds the draft against tool-role evidence. Static
skill `response_guidance` and pre-retrieval guidance support deterministic
fallback, which separately renders every returned candidate
with its name, price, category, and the confirmed-filter group from its own
search. A partial successful result set receives a neutral continuation. A zero-result tool response
carries its exact advertised taxonomy and confirmed filters; that scoped miss
cannot prove absence for a different type or the whole catalog.

Final-response extraction skips tool messages, assistant messages that still
contain tool calls, and internal skill-activation markers. If a completed graph
contains no shopper-facing response, the API emits
`"I could not complete that shopping request. Please try again."` and reports
`final_termination_reason: "incomplete_agent_response"` in diagnostics.

**Example:**
```json
{
  "response": "I found several red dresses under $100 that might interest you...",
  "images": {
    "product1": "https://cdn.shop.com/dress1.jpg",
    "product2": "https://cdn.shop.com/dress2.jpg"
  },
  "cart": {
    "contents": []
  },
  "timings": {
    "total": 3.48,
    "memory": 0.12,
    "deepagents": 3.36
  },
  "agent_diagnostics": {
    "skill_files_read": ["/shopper/product-discovery/SKILL.md"],
    "tool_calls": [{
      "sequence": 1,
      "tool_name": "activate_shopper_skills_tool",
      "arguments": {"skill_names": ["product-discovery"]},
      "status": "completed"
    }],
    "rejected_tool_calls": [],
    "duplicate_tool_calls": [],
    "product_evidence": [],
    "product_evidence_truncated": false,
    "catalog_scope_outcomes": [],
    "final_termination_reason": "completed",
    "partial_graph_messages": []
  }
}
```

### Cart

Shopping cart data model.

```typescript
interface Cart {
  contents: CartItem[];
}

interface CartItem {
  cart_line_id?: string;              // Opaque, non-reusable line ID on authoritative reads
  product_id?: string;                // Catalog product ref when available
  item: string;                       // Product display name
  amount: number;                     // Quantity
  price?: number;                     // Cached unit price
}
```

### Streaming Response

For streaming endpoints, responses are sent as Server-Sent Events (SSE) with the following format:

```typescript
interface StreamingChunk {
  type: 'content' | 'images' | 'products' | 'metrics' | 'error' | 'done';
  payload:
    | string
    | Record<string, string>
    | ProductSummary[]
    | {
        timings: Record<string, number>;
        total_seconds: number;
        token_usage: {
          input_tokens: number;
          output_tokens: number;
          total_tokens: number;
          model_calls: number;
        };
        model_usage: Record<string, {
          status: 'used' | 'failed' | 'disabled' | 'not_used';
          calls: number;
          detail?: string;
        }>;
        agent_diagnostics?: AgentDiagnostics;
      };
  timestamp: number;
}
```

## 🔄 Endpoints

### POST `/query/stream`

Returns a Server-Sent Events (SSE) response stream for shopping assistant
responses. In the current Deep Agents harness migration slice, the stream is
SSE-framed but does not yet emit token-level model chunks while the agent is
running. The endpoint currently emits product metadata, image payloads,
completed turn response text, and timing/token-usage/agent diagnostic metrics
after the Deep Agents turn finishes.

Before guardrails or agent work, the chain server starts a durable conversation
turn and receives bounded finalized recent turns plus the authoritative cart.
It finalizes that turn as `completed`, `blocked`, or `failed` before emitting the
terminal SSE frames. An exact finalized retry replays the stored output without
another model/tool turn. The public SSE frame shapes are unchanged.

Every unblocked Deep Agents turn includes one bounded activation model step
before normal shopping-tool selection. That step selects registered shopper
skills; the runtime injects their complete instructions before exposing the
ten shopping tools. It is included in `token_usage.model_calls` and
`agent_diagnostics`.

Token-level Deep Agents streaming is a known limitation for this PR and is
planned as a follow-up after the harness migration is stable.

**Request Body:** `QueryRequest`

**Response:** Server-Sent Events (SSE) stream

**Headers:**
```
Content-Type: application/json
Accept: text/event-stream
```

**Example Request:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "user_id": 123,
    "query": "Show me red dresses under $100"
  }'
```

**Example Response:**
```
data: {"type": "products", "payload": [{"product_id": "dress-1", "display_name": "Red Wrap Dress", "price": {"amount": 89.0, "currency": "USD"}, "image_url": "https://..."}], "timestamp": 1716400001.1}

data: {"type": "images", "payload": {"Red Wrap Dress": "https://..."}, "timestamp": 1716400001.2}

data: {"type": "content", "payload": "I found several red dresses...", "timestamp": 1716400001.2}

data: {"type": "metrics", "payload": {"timings": {"memory": 0.03, "catalog_search": 0.41, "deepagents": 1.92}, "total_seconds": 2.36, "token_usage": {"input_tokens": 1260, "output_tokens": 180, "total_tokens": 1440, "model_calls": 3}, "model_usage": {"text_embedding": {"status": "used", "calls": 1, "detail": "Catalog text/vector retrieval"}, "content_safety": {"status": "used", "calls": 2, "detail": "Input and output safety checks"}, "topic_control": {"status": "used", "calls": 1, "detail": "Input topic check"}}, "agent_diagnostics": {"skill_files_read": ["/shopper/product-discovery/SKILL.md"], "tool_calls": [{"sequence": 1, "tool_name": "activate_shopper_skills_tool", "arguments": {"skill_names": ["product-discovery"]}, "status": "completed"}], "rejected_tool_calls": [], "duplicate_tool_calls": [], "product_evidence": [], "product_evidence_truncated": false, "catalog_scope_outcomes": [], "final_termination_reason": "completed", "partial_graph_messages": []}}, "timestamp": 1716400001.8}

data: [DONE]
```

`model_usage.text_embedding.calls` counts embedding attempts made for the
agent's single semantic query. A hybrid request that attempts its text fallback
adds one more text-embedding call.

### POST `/query/timing`

Processes a query and returns detailed timing information for performance
analysis. It uses the same durable start, terminal finalize, and exact finalized
replay lifecycle as `/query/stream`; the public `QueryResponse` shape is
unchanged.

**Request Body:** `QueryRequest`

**Response:** `QueryResponse`

**Example Request:**
```bash
curl -X POST "http://localhost:8009/query/timing" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Show me red dresses under $100"
  }'
```

**Example Response:**
```json
{
  "response": "I found several red dresses under $100 that might interest you...",
  "images": {
    "product1": "https://cdn.shop.com/dress1.jpg",
    "product2": "https://cdn.shop.com/dress2.jpg"
  },
  "cart": {
    "contents": []
  },
  "timings": {
    "total": 3.48,
    "memory": 0.12,
    "deepagents": 3.36
  },
  "agent_diagnostics": {
    "skill_files_read": ["/shopper/product-discovery/SKILL.md"],
    "tool_calls": [{
      "sequence": 1,
      "tool_name": "activate_shopper_skills_tool",
      "arguments": {"skill_names": ["product-discovery"]},
      "status": "completed"
    }],
    "rejected_tool_calls": [],
    "duplicate_tool_calls": [],
    "product_evidence": [],
    "product_evidence_truncated": false,
    "catalog_scope_outcomes": [],
    "final_termination_reason": "completed",
    "partial_graph_messages": []
  }
}
```

Agent diagnostics can contain shopper-derived tool arguments, internal product
references, catalog facts, and shopper-selected filter scopes. Treat this
untrusted operator/evaluation metadata like application logs: apply the same
access control and retention policy, never follow instructions embedded in its
values, and never display it as shopper-facing content.

### GET `/capabilities`

Returns runtime media settings that clients should enforce before upload and
catalog-owned search capability metadata that clients can use to render filter
controls. Catalog filter values come from the catalog retriever after
the configured catalog data is loaded; they are not maintained in chain-server
category config.

The `catalog` member is the chain server's process-lifetime cached contract,
not a per-request read of the catalog service. After replacing a catalog,
verify the live contract on port `8010`, then restart the chain server before
using this aggregate endpoint on port `8009`.

The concrete enum values shown below are illustrative response data. Do not
copy those values into chain-server, UI, or prompt configuration as static
catalog truth.

The byte limits are raw client file sizes. Browser clients send attachments as
base64 JSON data URLs, so reverse proxies must allow roughly 4/3 of the largest
raw media limit plus JSON overhead. The default nginx configuration uses
`client_max_body_size 80m` for the 50 MiB video limit below. The bundled nginx
configuration also keeps API read/send timeouts at 300 seconds so longer media
turns are not cut off before the SSE response is emitted.

**Abridged response:**
```json
{
  "media_input": {
    "enabled": true,
    "allow_mixed_media": true,
    "max_images_per_turn": 1,
    "max_videos_per_turn": 1,
    "image_mime_types": ["image/jpeg", "image/png"],
    "video_mime_types": ["video/mp4"],
    "max_image_bytes": 10485760,
    "max_video_bytes": 52428800,
    "max_video_duration_seconds": 120,
    "vlm_enabled": true
  },
  "models": {
    "app_llm": {
      "label": "Language reasoning",
      "model": "nvidia/nemotron-3-super-120b-a12b",
      "source": "endpoint",
      "enabled": true
    },
    "vlm": {
      "label": "Vision-language inference",
      "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
      "source": "endpoint",
      "enabled": true
    },
    "text_embedding": {
      "label": "Text embedding",
      "model": "nvidia/nv-embedqa-e5-v5",
      "source": "endpoint",
      "enabled": true
    }
  },
  "catalog": {
    "catalog_id": "fashion_products",
    "product_count": 205,
    "retrieval_modes": ["text", "image", "hybrid"],
    "image_search_enabled": true,
    "filters": {
      "category": {
        "type": "enum",
        "operators": ["in"],
        "source_fields": ["category"],
        "values": ["apparel", "bags", "eyewear", "footwear", "jewelry"]
      },
      "price": {
        "type": "number",
        "operators": ["gte", "lte"],
        "source_fields": ["price"],
        "min_value": 39.9,
        "max_value": 269.99,
        "request_aliases": {"min": "min_price", "max": "max_price"}
      }
    },
    "fields": {
      "care": {
        "type": "text",
        "observed_type": null,
        "filterable": false,
        "searchable": true,
        "detail": true,
        "taxonomy": false,
        "operators": [],
        "source_fields": ["care"],
        "coverage": {"present": 24, "total": 205},
        "values": [],
        "min_value": null,
        "max_value": null
      }
    },
    "taxonomy": {"category_field": "category", "subcategory_field": "subcategory", "categories": {}}
  }
}
```

### Catalog Retriever GET `/capabilities`

Catalog retriever exposes a separate capability endpoint on the catalog service
port, usually `http://localhost:8010/capabilities`. This endpoint describes
which fields are searchable, filterable, or available as details, including
their observed taxonomy scopes. It is the live contract for the catalog
snapshot loaded by that catalog-service process; the chain-server aggregate on
port `8009` is its cached copy.

**Response:** `CatalogCapabilities`

**Example Response:**
The concrete values shown below are abridged observed response data. The
catalog retriever derives them from the loaded JSONL.

```json
{
  "catalog_id": "fashion_products",
  "product_count": 205,
  "retrieval_modes": ["text", "image", "hybrid"],
  "image_search_enabled": true,
  "filters": {
    "category": {
      "type": "enum",
      "operators": ["in"],
      "source_fields": ["category"],
      "values": ["apparel", "bags", "eyewear", "footwear", "jewelry"]
    },
    "price": {
      "type": "number",
      "operators": ["gte", "lte"],
      "source_fields": ["price"],
      "min_value": 39.9,
      "max_value": 269.99,
      "request_aliases": {"min": "min_price", "max": "max_price"}
    }
  },
  "fields": {
    "care": {
      "type": "text",
      "filterable": false,
      "searchable": true,
      "detail": true,
      "taxonomy": false,
      "operators": [],
      "source_fields": ["care"],
      "coverage": {"present": 24, "total": 205},
      "values": []
    }
  },
  "taxonomy": {
    "category_field": "category",
    "subcategory_field": "subcategory",
    "categories": {
      "apparel": {
        "product_count": 96,
        "filters": {},
        "semantic_fields": {},
        "subcategories": {
          "dresses": {
            "product_count": 32,
            "filters": {"neckline": {"values": [{"value": "v_neck", "count": 19}]}},
            "semantic_fields": {}
          }
        }
      }
    }
  }
}
```

### Catalog Retriever POST `/query/text`

Executes a structured text catalog search on the catalog service port, usually
`http://localhost:8010/query/text`.

The agent-facing search tool requires one `semantic_query`, one pre-retrieval
product-agnostic `shopper_guidance`, one `requested_product_type`, one
`taxonomy_status`, one capability-derived `taxonomy` envelope, one
capability-derived `required_constraints` object, and `scope_complete`.
`requested_product_type` is the shortest product noun or true
umbrella from the shopper's current turn or direct antecedent. It excludes
color, material, fit, occasion, weather, and style modifiers; for
`agent_selected_type`, it is the chosen advertised role noun. It is `null` only
for `image_only`. The semantic query supplies soft ranking direction
independently of taxonomy; it need not repeat the selected taxonomy noun.
Taxonomy and hard constraints are enforced through their structured fields.
`shopper_guidance` is authored under the active skill before results are known
and is not sent to the catalog service.
Each call accepts at most one category. For a broad request that names no type,
`agent_selected_type` selects exactly one advertised subcategory as the focused
starting role. It is forbidden for a role whose type the shopper named,
including an alternative, confirmation, comparison, or follow-up. Invalid
open-role provenance is rejected rather than silently reinterpreted, and a
genuinely open-role selection must name its selected advertised subcategory in
`requested_product_type`. A repair of an open role remains
`agent_selected_type`.
`no_direct_catalog_match` is a no-retrieval result with
empty taxonomy and no hard constraints; an unsupported modifier does not erase
an advertised type, and subjective style remains semantic. The duplicate-search
identity is normalized taxonomy plus hard constraints, so paraphrasing cannot
repeat a retrieval while a genuinely different hard-filter scope may run within
`max_catalog_searches_per_turn`. The chain maps generic category/subcategory
selections to advertised field names and sends the semantic query as a singleton
`text` list.

One invalid agent search may receive one search-only repair per distinct scope.
Malformed or nonempty free-form `unadvertised_requirements` arguments on a
native schema-invalid call fail closed. A schema-valid, genuinely open
`agent_selected_type` role may consume that scope's repair for model-owned
review: preserve an explicit objective must-have, or remove only an inferred or
subjective requirement. Deterministic code does not parse shopper prose. The
repair cannot replace a shopper-stated product-scope noun. A successful partial
search may continue to another valid role with its own one-repair opportunity;
no scope receives two repairs. Completed scopes and deterministic stop results
close the loop, and the configured turn cap remains three successful searches.
For multi-role output, each pre-retrieval guidance sentence remains grouped with
products from its originating search. Completed turns get one tools-disabled
synthesis from collected evidence; search-only drafts pass through grounding,
with deterministic rendering as fail-closed fallback.

The chain server also bounds product-detail reads with
`max_product_detail_reads_per_turn`. Detail reads are intended for direct
product fact questions and shortlisted comparisons, not for enriching every
initial outfit recommendation.

When a cart item matches current-request product evidence, the chain server can
return that product image in a cart or comparison turn. A later turn can obtain
that evidence from one unique durable same-conversation resolution without
forcing another catalog search; missing, ambiguous, or stale active-catalog refs
require clarification or a fresh search.

**Request Body:**
```json
{
  "text": ["practical structured office tote"],
  "categories": [],
  "filters": {"subcategory": ["tote_bags"], "price": {"max": 60}},
  "k": 4,
  "candidate_k": 205
}
```

The catalog endpoint retains a text-list shape for direct/internal compatibility;
the serving agent uses one entry. The catalog performs embedding generation,
vector search, candidate fusion, product-ID deduplication, hard filtering,
thresholding, and deterministic similarity ordering. It performs no
shopper-language interpretation, query expansion, chat/completion call, or
learned reranking.

`candidate_k` is optional. When omitted, the current small-catalog default
covers the complete active snapshot before hard filtering and final trimming to
`k`.

Unknown filter fields, values, taxonomy values, or operators return HTTP 422
with the catalog's validation message. The chain server treats that response as
non-retryable and does not silently rerun a weakened query. Numeric bounds must
be finite numbers, not booleans; if any supplied bound is invalid, the complete
filter request is rejected instead of partially applied. `min`/`gte` are lower-
bound aliases and `max`/`lte` are upper-bound aliases. Supplying both aliases
for the same bound is ambiguous and returns HTTP 422. The same rule applies
across representations: for example, `price.min` cannot be combined with the
top-level `min_price` compatibility alias. Legacy `categories` may accompany
non-taxonomy filters, but combining it with an explicit taxonomy filter is
also ambiguous and returns HTTP 422.

**Response:**
```json
{
  "texts": ["Work Bag | structured tote | accessories,bag\nPRICE: 59.0"],
  "ids": ["generated:abc123"],
  "similarities": [0.91],
  "names": ["Work Bag"],
  "images": ["/images/work_bag.jpg"],
  "products": [
    {
      "product_id": "generated:abc123",
      "display_name": "Work Bag",
      "description": "structured tote",
      "category": "tote_bags",
      "price": {"amount": 59.0, "currency": "USD"},
      "image_url": "/images/work_bag.jpg",
      "attributes": {"similarity": 0.91}
    }
  ],
  "diagnostics": {
    "requested_top_k": 4,
    "candidate_k": 205,
    "after_filter_count": 1,
    "returned_count": 1
  },
  "no_result_reason": null
}
```

### Catalog Retriever POST `/query/image`

Accepts the same fields as `/query/text`, plus `image_base64`. Explicit category
and price filters are hard filters for image and hybrid retrieval too.
Image and hybrid results retain pooled similarity-score ordering.
When the active capabilities do not advertise image or hybrid retrieval, an
image-only assistant request asks the shopper for a text description instead
of issuing an empty text search. An explicit image/hybrid mode is never silently
downgraded to text and requires an attached image; unsupported or incomplete
mode requests stop before retrieval.

Request models reject unknown fields, including client-supplied embedding
vectors.

### Catalog Retriever GET `/products/{product_id}`

Returns deterministic details for one source product ID from the active
snapshot. Core mapped roles such as identity, name, description, image, and
price are returned in top-level contract fields; other source fields marked
`detail` in the sidecar appear under `attributes`. Accepted source IDs are
nonempty canonical strings safe for one URL path segment, so the ID returned by
search round-trips exactly through this endpoint. A missing ID returns HTTP 404.

```json
{
  "product_id": "generated:abc123",
  "display_name": "Classic Black Patent Leather Purse",
  "description": "A structured black patent leather tote...",
  "category": "tote_bags",
  "price": {"amount": 49.9, "currency": "USD"},
  "image_url": "/images/Classic_Black_Patent_Leather_Purse.jpg",
  "attributes": {
    "care": "spot clean with a damp cloth",
    "composition": "patent leather",
    "primary_color": "black",
    "structure": "structured"
  },
  "variants": [],
  "source_uri": null
}
```

### Memory Retriever POST `/conversations/{conversation_id}/turn/start`

Starts one durable turn transaction before guardrail, model, or tool work. The
memory service accepts one active turn per conversation and assigns its ordered
`sequence`. It returns the authoritative cart and at most
`MEMORY_RECENT_TURNS` prior finalized raw turns. Raw media is not stored; the
request digest includes ordered media content hashes so exact retries can be
distinguished safely.

```json
{
  "request_id": "request_abc",
  "shopper_text": "Show me black flats",
  "cart_user_id": 123,
  "request_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "catalog_revision": "catalog-fingerprint"
}
```

```json
{
  "turn_id": "8e40575d5e5a4dbca34e1d08a2cb1692",
  "attempt_id": "bd77b851b3494e37a764e3dfa7500208",
  "sequence": 4,
  "replayed": false,
  "status": "started",
  "recent_turns": [
    {
      "sequence": 3,
      "shopper_text": "Show me a beige top",
      "assistant_text": "Here are the grounded beige options.",
      "status": "completed"
    }
  ],
  "projection": {
    "version": 3,
    "active_anchors": [],
    "effective_preferences": [],
    "product_reference_index": [
      {
        "candidate_set_id": "candidate-set-event-id",
        "turn_seq": 3,
        "catalog_revision": "catalog-fingerprint",
        "products": [
          {
            "ref": "product-123",
            "name": "Beige Ribbed Top",
            "category": "apparel",
            "position": 1
          }
        ]
      }
    ],
    "last_turn_id": "prior-turn-id"
  },
  "cart": [],
  "assistant_text": null,
  "termination_reason": null,
  "output": null
}
```

`projection.product_reference_index` is a compact, bounded index of ordered
product-card sets derived from durable `candidate_set_presented` events. The
runtime uses it as read-only context for typed historical resolution; the
authoritative full product payload remains in the event. The active-anchor and
preference lanes remain reserved and unused. Its serialized value is capped at
16,384 characters by retaining the newest complete candidate sets. On an exact retry of a finalized
turn, `replayed` is `true` and the response includes stored `assistant_text`,
`termination_reason`, and `output` (`product_results`, `retrieved`, and
`agent_diagnostics`). The chain server returns that stored output without agent
execution. Reusing a request ID with different input, retrying a still-started
turn, or starting another turn while the conversation has an active turn returns
HTTP 409. A start transport or conflict failure prevents agent work.

At memory-service startup and atomically before each new turn start, turns left
in `started` longer than `MEMORY_TURN_ABANDON_SECONDS` are changed to
`abandoned`. An exact retry may reopen an abandoned turn only while it is the
latest sequence in that conversation. The service keeps its turn ID, sequence,
and request ID but rotates the opaque `attempt_id`; an older abandoned turn is
superseded and cannot be reopened. This is not a continuous background
expiration process.

### Memory Retriever POST `/conversations/{conversation_id}/turns/{turn_id}/finalize`

Finalizes a started turn transaction as `completed`, `blocked`, or `failed`.
The request, any ordered event envelopes, and the compact projection commit
atomically with the replay output. When finalized `output.product_results`
contains referenceable product cards, the service derives one ordered
`candidate_set_presented` event from that server-controlled output. Empty,
failed, or non-presented candidates do not enter the reference index.
The `runtime-presented-products` event key is reserved; caller-supplied events
using it are rejected with HTTP 422.

```json
{
  "request_id": "request_abc",
  "attempt_id": "bd77b851b3494e37a764e3dfa7500208",
  "assistant_text": "Here are the black flats I found.",
  "status": "completed",
  "termination_reason": "completed",
  "events": [],
  "output": {
    "product_results": [],
    "retrieved": {},
    "agent_diagnostics": {
      "final_termination_reason": "completed"
    }
  }
}
```

```json
{
  "turn_id": "8e40575d5e5a4dbca34e1d08a2cb1692",
  "attempt_id": "bd77b851b3494e37a764e3dfa7500208",
  "sequence": 4,
  "replayed": false,
  "status": "completed",
  "assistant_text": "Here are the black flats I found.",
  "termination_reason": "completed"
}
```

The caller must echo the `attempt_id` returned by start. An identical finalize
retry for the current attempt returns the same receipt with `replayed: true`.
After an abandoned turn is reopened, a late finalize from its old attempt is
rejected with HTTP 409 `turn_attempt_superseded`; the chain server returns a safe
superseded-attempt response instead of exposing the stale answer or products. A
generic finalize transport or service failure still preserves the already
grounded response, its request-scoped graph checkpoint, and records
`memory_finalize_error`. After successful durable finalization, the chain server
deletes the request-scoped checkpoint identified by that collision-safe
conversation/request pair. Different final data, a
request-ID mismatch, duplicate event keys, or an invalid status transition also
returns a conflict and rolls back the transaction. Event envelopes are stored
in logical order. Active anchors, preferences, and selections remain reserved;
only presented-product candidate sets currently update a projection.

### Memory Retriever POST `/conversations/{conversation_id}/products/resolve`

Deterministically resolves a nonempty batch of typed product descriptors against
that conversation's durable `candidate_set_presented` events. This is an
internal memory-service API. It performs no catalog, embedding, or model call.

```json
{
  "references": [
    {
      "reference_id": "chosen_top",
      "display_name": "Beige Ribbed Top",
      "turn_sequence": 3
    }
  ]
}
```

Selectors may include `product_ref`, exact `display_name`, exact `category`,
`turn_sequence`, or `candidate_set_id`. `ordinal` is one-based and requires a
turn sequence or candidate-set ID. Matching is case-insensitive after trimming;
there is no fuzzy or semantic matching. The service accepts at most 20
descriptors per request.

```json
{
  "results": [
    {
      "reference_id": "chosen_top",
      "status": "resolved",
      "match_count": 1,
      "matches": [
        {
          "product": {
            "product_id": "product-123",
            "display_name": "Beige Ribbed Top",
            "category": "apparel",
            "image_url": null,
            "price": null,
            "availability": "unknown"
          },
          "candidate_set_id": "candidate-set-event-id",
          "turn_sequence": 3,
          "position": 1,
          "catalog_revision": "catalog-fingerprint"
        }
      ]
    }
  ]
}
```

Each result is `resolved`, `ambiguous`, or `not_found`. Only a unique match is
added to the chain server's request-local product evidence. Ambiguous and
missing results return bounded candidates for a concise clarification and never
authorize a guessed product. The shopper runtime permits at most one batched
resolver-tool call per turn; a second call is stopped. The current slice does
not invalidate stored evidence when `catalog_revision` changes.

### Memory Retriever DELETE `/conversations/{conversation_id}`

Deletes that conversation's durable turns, cascaded event envelopes, and
reserved projection row. It deliberately does not delete cart rows, cart
mutation replay rows, or the legacy user/context row.

```json
{
  "conversation_id": "conversation_abc",
  "deleted_turns": 4,
  "deleted_events": 0,
  "deleted_projection": true
}
```

### GET `/health`

Health check endpoint to verify service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1716400000.0,
  "version": "1.0.0",
  "services": {
    "chain_server": "healthy",
    "catalog_retriever": "healthy",
    "memory_retriever": "healthy",
    "guardrails": "healthy"
  }
}
```

### GET `/`

Root endpoint with API information.

**Response:**
```json
{
  "message": "Shopping Assistant API",
  "version": "1.0.0",
  "endpoints": {
    "query": "/query",
    "stream": "/query/stream",
    "timing": "/query/timing",
    "capabilities": "/capabilities",
    "health": "/health",
    "docs": "/docs"
  }
}
```

## ❌ Error Handling

### Error Response Format

```typescript
interface ErrorResponse {
  detail: string;                     // Error message
  status_code: number;                // HTTP status code
  timestamp: string;                  // Error timestamp
}
```

### Common Error Codes

| Status Code | Description | Example |
|-------------|-------------|---------|
| 400 | Bad Request | Invalid request format |
| 422 | Validation Error | Missing fields or unsupported catalog constraint |
| 500 | Internal Server Error | Service unavailable |
| 503 | Service Unavailable | NIM containers not ready |

**Example Error Response:**
```json
{
  "detail": "Invalid request format: missing required field 'user_id'",
  "status_code": 422,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## ⚡ Rate Limiting

Currently, the API does not implement rate limiting. For production deployments, consider implementing rate limiting based on:

- Requests per minute per user
- Concurrent connections per user
- Total requests per hour

## 💡 Examples

### Product Search

**Find dresses by description:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Show me summer dresses with floral patterns"
  }'
```

**Search by price range:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Find shoes under $50"
  }'
```

### Shopping Cart Operations

**Add item to cart:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Add the black polka dot dress to my cart",
    "cart": {
      "contents": []
    }
  }'
```

**View cart contents:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "What is in my shopping cart?",
    "cart": {
      "contents": [
        {
          "item": "black_polka_dot_dress",
          "amount": 1
        }
      ]
    }
  }'
```

**Remove item from cart:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Remove the black polka dot dress from my cart",
    "cart": {
      "contents": [
        {
          "item": "black_polka_dot_dress",
          "amount": 1
        }
      ]
    }
  }'
```

### Image-based Search

**Search by uploaded image:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Find products similar to this image",
    "image": "base64_encoded_image_data",
    "image_bool": true
  }'
```

### Conversational Queries

**General questions:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "What accessories would go well with a red dress?",
    "context": "Previous conversation about summer clothing"
  }'
```

**Style advice:**
```bash
curl -X POST "http://localhost:8009/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Help me build an outfit for a summer wedding"
  }'
```

### Performance Analysis

**Get detailed timing information:**
```bash
curl -X POST "http://localhost:8009/query/timing" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Show me red dresses under $100"
  }'
```

## 🔧 Client Integration

### JavaScript/TypeScript Example

```typescript
class ShoppingAssistantAPI {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8009') {
    this.baseUrl = baseUrl;
  }

  async streamQuery(request: QueryRequest): Promise<ReadableStream> {
    const response = await fetch(`${this.baseUrl}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.body!;
  }

  async queryWithTiming(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query/timing`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Usage example
const api = new ShoppingAssistantAPI();

// Stream query
const stream = await api.streamQuery({
  user_id: 123,
  query: "Show me red dresses under $100"
});

const reader = stream.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = new TextDecoder().decode(value);
  console.log('Received:', chunk);
}
```

### Python Example

```python
import requests
import json
import sseclient

class ShoppingAssistantAPI:
    def __init__(self, base_url: str = "http://localhost:8009"):
        self.base_url = base_url

    def stream_query(self, request: dict):
        """Stream a query and yield response chunks."""
        response = requests.post(
            f"{self.base_url}/query/stream",
            json=request,
            headers={"Accept": "text/event-stream"},
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP error! status: {response.status_code}")
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break
            yield json.loads(event.data)

    def query_with_timing(self, request: dict) -> dict:
        """Send a query and get timing information."""
        response = requests.post(
            f"{self.base_url}/query/timing",
            json=request
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP error! status: {response.status_code}")
        
        return response.json()

    def health_check(self) -> dict:
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage example
api = ShoppingAssistantAPI()

# Stream query
request = {
    "user_id": 123,
    "query": "Show me red dresses under $100"
}

for chunk in api.stream_query(request):
    print(f"Received: {chunk}")

# Get timing information
response = api.query_with_timing(request)
print(f"Response: {response['response']}")
print(f"Timing: {response['timings']}")
```

## 📝 Notes

- All timestamps are in Unix timestamp format (seconds since epoch)
- Image data may be raw base64 or a `data:` URL; video media should include
  `mime_type: "video/mp4"` and is sent through `media[]`
- The API supports both local and cloud-based NIM deployments
- The `vlm` model role is enabled by default for image/video media perception
  and can be set to `disabled`; image embedding search remains separately controlled by the
  `image_embedding` model role and `CATALOG_IMAGE_EMBEDDING_ENABLED`.
- Content safety is enabled by default but can be disabled per request
- `/query/stream` uses SSE framing. Token-level Deep Agents streaming is a
  known follow-up after the harness migration; this slice emits completed turn
  events rather than live model chunks. The stream includes `products` frames
  for structured catalog summaries and `metrics` frames for per-turn inference
  timing summaries.

---

For more information, see the [main README](../README.md) or [GitHub repository](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant). 
