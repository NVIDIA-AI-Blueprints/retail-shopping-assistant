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
http://localhost:8000
```

## 🔐 Authentication

Currently, the API does not require authentication for local deployments. For production deployments, consider implementing API key authentication or OAuth2.

## 📊 Data Models

### Catalog Capabilities

The catalog retriever owns its filter and metadata capability contract. Chain
server request-building code should consume this contract instead of inferring
filterability from product text or hard-coded category lists.

Filter fields are configured in `shared/configs/catalog_retriever/config.yaml`
under `filter_registry`. Enum values are never configured statically; they are
discovered from the configured CSV `source_fields` after the catalog data is
loaded. See [Catalog Filter Configuration](CATALOG_FILTERS.md) for the operator
workflow.

```typescript
interface CatalogCapabilities {
  catalog_id: string;
  retrieval_modes: Array<'text' | 'image' | 'hybrid'>;
  image_search_enabled: boolean;
  filters: Record<string, {
    type: 'enum' | 'number' | 'text';
    operators: string[];
    source_fields: string[];
    values?: string[];
    min_value?: number;
    max_value?: number;
    request_aliases?: Record<string, string>;
  }>;
}
```

The default fashion catalog declares `category` as an enum filter sourced from
the product `subcategory` column, `price` as a numeric filter with `min_price`
and `max_price` request aliases. Catalogs that can strictly filter by color,
material, size, or other metadata should declare those fields under
`filter_registry` too. Do not add static enum values to chain-server config,
UI code, or prompts. Example enum values in documentation are illustrative
only; runtime values come from `/capabilities`.

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

`session_id`, `conversation_id`, and `cart_id` are optional for backward
compatibility. When they are omitted, the server maps the legacy `user_id` to
internal compatibility identifiers. The bundled UI creates browser-session
identifiers and sends them on every turn. When supplied, `conversation_id`
scopes the Deep Agents thread and conversation memory, while `cart_id` scopes
cart reads/writes. Production website integrations should move these IDs to a
server-owned session/thread service before broad rollout so customer context and
cart state cannot bleed across sessions.

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
}
```

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
  item: string;                       // Product identifier
  amount: number;                     // Quantity
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
completed turn response text, and timing/token-usage metrics after the Deep
Agents turn finishes.

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
curl -X POST "http://localhost:8000/query/stream" \
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

data: {"type": "metrics", "payload": {"timings": {"memory": 0.03, "catalog_search": 0.41, "deepagents": 1.92}, "total_seconds": 2.36, "token_usage": {"input_tokens": 1260, "output_tokens": 180, "total_tokens": 1440, "model_calls": 1}, "model_usage": {"text_embedding": {"status": "used", "calls": 1, "detail": "Catalog text/vector retrieval"}, "content_safety": {"status": "used", "calls": 2, "detail": "Input and output safety checks"}, "topic_control": {"status": "used", "calls": 1, "detail": "Input topic check"}}}, "timestamp": 1716400001.8}

data: [DONE]
```

### POST `/query/timing`

Processes a query and returns detailed timing information for performance analysis.

**Request Body:** `QueryRequest`

**Response:** `QueryResponse`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/query/timing" \
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
  }
}
```

### GET `/capabilities`

Returns runtime media settings that clients should enforce before upload and
catalog-owned search capability metadata that clients can use to render filter
controls. Catalog filter values come from the catalog retriever after
the configured catalog data is loaded; they are not maintained in chain-server
category config.

The concrete enum values shown below are illustrative response data. Do not
copy those values into chain-server, UI, or prompt configuration as static
catalog truth.

The byte limits are raw client file sizes. Browser clients send attachments as
base64 JSON data URLs, so reverse proxies must allow roughly 4/3 of the largest
raw media limit plus JSON overhead. The default nginx configuration uses
`client_max_body_size 80m` for the 50 MiB video limit below. The bundled nginx
configuration also keeps API read/send timeouts at 300 seconds so longer media
turns are not cut off before the SSE response is emitted.

**Response:**
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
    "catalog_id": "fashion_products_extended",
    "retrieval_modes": ["text", "image", "hybrid"],
    "image_search_enabled": true,
    "filters": {
      "category": {
        "type": "enum",
        "operators": ["in"],
        "source_fields": ["subcategory"],
        "values": ["bag", "dress", "shoes"]
      },
      "price": {
        "type": "number",
        "operators": ["gte", "lte"],
        "source_fields": ["price"],
        "min_value": 39.9,
        "max_value": 269.99,
        "request_aliases": {"min": "min_price", "max": "max_price"}
      },
      "color": {
        "type": "enum",
        "operators": ["in"],
        "source_fields": ["color"],
        "values": ["black", "green"]
      }
    }
  }
}
```

### Catalog Retriever GET `/capabilities`

Catalog retriever exposes a separate capability endpoint on the catalog service
port, usually `http://localhost:8010/capabilities`. This endpoint describes
which catalog metadata fields are valid hard filters.

For enum filters, `values` is generated from the loaded CSV rows. For numeric
filters, `min_value` and `max_value` are generated from the loaded CSV rows.
`filter_registry` specifies field names and types only.

**Response:** `CatalogCapabilities`

**Example Response:**
The concrete enum values shown below are illustrative response data. The
catalog retriever derives them from the loaded CSV rows.

```json
{
  "catalog_id": "fashion_products_extended",
  "retrieval_modes": ["text", "image", "hybrid"],
  "image_search_enabled": true,
  "filters": {
    "category": {
      "type": "enum",
      "operators": ["in"],
      "source_fields": ["subcategory"],
      "values": ["bag", "dress", "shoes"]
    },
    "price": {
      "type": "number",
      "operators": ["gte", "lte"],
      "source_fields": ["price"],
      "min_value": 39.9,
      "max_value": 269.99,
      "request_aliases": {"min": "min_price", "max": "max_price"}
    },
    "color": {
      "type": "enum",
      "operators": ["in"],
      "source_fields": ["color"],
      "values": ["black", "green"]
    }
  }
}
```

### Catalog Retriever POST `/query/text`

Executes a structured text catalog search on the catalog service port, usually
`http://localhost:8010/query/text`.

The chain server bounds Deep Agents catalog tool loops with
`max_catalog_searches_per_turn` so one shopping turn cannot keep probing the
catalog indefinitely. Multi-item outfit requests should fit within the default
cap by running one focused search per required item type and then synthesizing
the response from those results.

The chain server also bounds product-detail reads with
`max_product_detail_reads_per_turn`. Detail reads are intended for direct
product fact questions and shortlisted comparisons, not for enriching every
initial outfit recommendation.

When a cart item matches a product ref cached in the active conversation, the
chain server can return that product image again on later cart or comparison
turns without forcing another catalog search.

**Request Body:**
```json
{
  "text": ["practical work bag"],
  "categories": ["bag"],
  "filters": {"price": {"max": 60}},
  "k": 4,
  "candidate_k": 20
}
```

`candidate_k` is optional. When omitted, the catalog retriever searches a wider
candidate window than `k`, applies hard filters over that wider window, and then
trims the final response to `k`.

**Response:**
```json
{
  "texts": ["Work Bag | structured tote | accessories,bag\nPRICE: 59.0"],
  "ids": ["123"],
  "similarities": [0.91],
  "names": ["Work Bag"],
  "images": ["/images/work_bag.jpg"],
  "products": [
    {
      "product_id": "123",
      "display_name": "Work Bag",
      "description": "structured tote",
      "category": "bag",
      "price": {"amount": 59.0, "currency": "USD"},
      "image_url": "/images/work_bag.jpg",
      "attributes": {"similarity": 0.91}
    }
  ],
  "diagnostics": {
    "requested_top_k": 4,
    "candidate_k": 20,
    "after_filter_count": 1,
    "returned_count": 1
  },
  "no_result_reason": null
}
```

### Catalog Retriever POST `/query/image`

Accepts the same fields as `/query/text`, plus `image_base64`. Explicit category
and price filters are hard filters for image and hybrid retrieval too.

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
| 422 | Validation Error | Missing required fields |
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
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Show me summer dresses with floral patterns"
  }'
```

**Search by price range:**
```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Find shoes under $50"
  }'
```

### Shopping Cart Operations

**Add item to cart:**
```bash
curl -X POST "http://localhost:8000/query/stream" \
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
curl -X POST "http://localhost:8000/query/stream" \
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
curl -X POST "http://localhost:8000/query/stream" \
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
curl -X POST "http://localhost:8000/query/stream" \
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
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "What accessories would go well with a red dress?",
    "context": "Previous conversation about summer clothing"
  }'
```

**Style advice:**
```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Help me build an outfit for a summer wedding"
  }'
```

### Performance Analysis

**Get detailed timing information:**
```bash
curl -X POST "http://localhost:8000/query/timing" \
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

  constructor(baseUrl: string = 'http://localhost:8000') {
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
    def __init__(self, base_url: str = "http://localhost:8000"):
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
