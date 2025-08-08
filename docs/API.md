# 🛍️ Retail Shopping Assistant API Documentation

## 📋 Table of Contents

- [Overview](#🎯-overview)
- [Base URL](#🌐-base-url)
- [Authentication](#🔐-authentication)
- [Data Models](#📊-data-models)
- [Endpoints](#🔄-endpoints)
- [Error Handling](#❌-error-handling)
- [Rate Limiting](#⚡-rate-limiting)
- [Examples](#💡-examples)
- [Client Integration](#🔧-client-integration)
- [Notes](#📝-notes)

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
http://localhost:8000
```

## 🔐 Authentication

Currently, the API does not require authentication for local deployments. For production deployments, consider implementing API key authentication or OAuth2.

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
  "images": {
    "product1": "https://cdn.shop.com/dress1.jpg",
    "product2": "https://cdn.shop.com/dress2.jpg"
  },
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
    "health": "/health",
    "docs": "/docs"
  },
  "agents": [
    "planner",
    "retriever",
    "cart",
    "chatter",
    "summary"
  ]
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
- Image data should be base64 encoded without the data URL prefix
- The API supports both local and cloud-based NIM deployments
- Content safety is enabled by default but can be disabled per request
- Streaming responses provide real-time feedback for better user experience

---

For more information, see the [main README](../README.md) or [GitHub repository](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant). 
