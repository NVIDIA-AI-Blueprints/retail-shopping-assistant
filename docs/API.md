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

The Retail Shopping Assistant API provides a comprehensive interface for an AI-powered retail shopping advisor. The API is built on a microservices architecture using LangGraph for agent orchestration and supports both streaming and non-streaming responses.

### Key Features

- **Real-time Streaming**: Server-Sent Events (SSE) for live responses
- **Multi-modal Input**: Text queries and image uploads
- **Shopping Cart Management**: Add, remove, and view cart items
- **Content Safety**: Built-in guardrails for safe interactions
- **Performance Monitoring**: Detailed timing information

## 🌐 Base URL

```
http://localhost:3000/api
```

## 🔐 Authentication

The reference API does not implement end-user authentication or authorization.
It is available through the loopback-bound nginx entrypoint for local,
single-operator evaluation only; internal service APIs are not published to the
host. The client-provided `user_id` separates demo state but is not an
authenticated identity.

Do not expose this API to untrusted or multiple users. A broader deployment must
place authenticated TLS ingress in front of nginx and derive user identity and
authorization server-side. See [SECURITY.md](../SECURITY.md) for the supported
deployment boundary.

## 📊 Data Models

### QueryRequest

The main request model for all shopping queries.

```typescript
interface QueryRequest {
  user_id: number;                    // Unique user identifier
  query: string;                      // User's text query
  image?: string;                     // Base64 encoded image (optional)
  context?: string;                   // Previous conversation context
  cart?: Cart;                        // Current shopping cart state
  retrieved?: Record<string, string>; // Previously retrieved products
  guardrails?: boolean;               // Enable content safety (default: true)
  image_bool?: boolean;               // Indicate if image is provided (default: false)
}
```

**Example:**
```json
{
  "user_id": 123,
  "query": "Show me red dresses under $100",
  "image": "",
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

### QueryResponse

The response model for non-streaming queries.

```typescript
interface QueryResponse {
  response: string;                   // Generated response text
  images: Record<string, string>;     // Product images
  timings: Record<string, number>;    // Performance timing data
}
```

**Example:**
```json
{
  "response": "I found several red dresses under $100 that might interest you...",
  "images": {},
  "timings": {
    "total": 3.48,
    "planner": 0.12,
    "retriever": 1.23,
    "chatter": 2.13
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
  type: 'content' | 'images' | 'error' | 'done';
  payload: string | Record<string, string>;
  timestamp: number;
}
```

## 🔄 Endpoints

### POST `/query/stream`

Streams real-time responses back to the client as the shopping assistant generates them.

**Request Body:** `QueryRequest`

**Response:** Server-Sent Events (SSE) stream

**Headers:**
```
Content-Type: application/json
Accept: text/event-stream
```

**Example Request:**
```bash
curl -X POST "http://localhost:3000/api/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "user_id": 123,
    "query": "Show me red dresses under $100"
  }'
```

**Example Response:**
```
data: {"type": "content", "payload": "I found several red dresses...", "timestamp": 1716400001.2}

data: {"type": "images", "payload": {"product1": "https://..."}, "timestamp": 1716400001.5}

data: {"type": "content", "payload": " that might interest you...", "timestamp": 1716400001.8}

data: [DONE]
```

### POST `/query/timing`

Processes a query and returns detailed timing information for performance analysis.

**Request Body:** `QueryRequest`

**Response:** `QueryResponse`

**Example Request:**
```bash
curl -X POST "http://localhost:3000/api/query/timing" \
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
  "images": {},
  "timings": {
    "total": 3.48,
    "planner": 0.12,
    "retriever": 1.23,
    "chatter": 2.13,
    "guardrails": 0.05
  }
}
```

### GET `/health`

Health check endpoint to verify service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1716400000.0,
  "version": "1.0.0"
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
    "health": "/health",
    "docs": "/docs"
  }
}
```

## ❌ Error Handling

### Error Response Format

```typescript
interface ErrorResponse {
  detail: string | Array<Record<string, unknown>>;
}
```

FastAPI validation failures return a list in `detail`; processing failures
raised before streaming begins return a string. A failure after an SSE stream
has started is sent as `{"type":"error","payload":"..."}` within the stream.

### Common Error Codes

| Status Code | Description | Example |
|-------------|-------------|---------|
| 422 | Validation Error | Missing required fields |
| 500 | Internal Server Error | Query processing failed |

**Example Error Response:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "user_id"],
      "msg": "Field required"
    }
  ]
}
```

## ⚡ Rate Limiting

The blueprint does not implement rate limiting. Do not expose it to untrusted
clients. Any broader deployment must implement limits such as:

- Requests per minute per user
- Concurrent connections per user
- Total requests per hour

## 💡 Examples

### Product Search

**Find dresses by description:**
```bash
curl -X POST "http://localhost:3000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Show me summer dresses with floral patterns"
  }'
```

**Search by price range:**
```bash
curl -X POST "http://localhost:3000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Find shoes under $50"
  }'
```

### Shopping Cart Operations

**Add item to cart:**
```bash
curl -X POST "http://localhost:3000/api/query/stream" \
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
curl -X POST "http://localhost:3000/api/query/stream" \
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
curl -X POST "http://localhost:3000/api/query/stream" \
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
curl -X POST "http://localhost:3000/api/query/stream" \
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
curl -X POST "http://localhost:3000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "What accessories would go well with a red dress?",
    "context": "Previous conversation about summer clothing"
  }'
```

**Style advice:**
```bash
curl -X POST "http://localhost:3000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Help me build an outfit for a summer wedding"
  }'
```

### Performance Analysis

**Get detailed timing information:**
```bash
curl -X POST "http://localhost:3000/api/query/timing" \
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

  constructor(baseUrl: string = 'http://localhost:3000/api') {
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
    def __init__(self, base_url: str = "http://localhost:3000/api"):
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
- Image data may be a base64 data URI (as sent by the bundled UI) or raw base64;
  remote URLs and filesystem paths are rejected
- The API supports both local and cloud-based NIM deployments
- Content safety is enabled by default but can be disabled per request
- Streaming responses provide real-time feedback for better user experience

---

For more information, see the [main README](../README.md) or [GitHub repository](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant). 
