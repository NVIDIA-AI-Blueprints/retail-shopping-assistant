# 🛍️ Shopping Assistant API Documentation

## 📄 License / Disclaimer

[Add license here]

## Overview

The Shopping Assistant API provides a comprehensive interface for an AI-powered retail shopping advisor. Built with LangGraph for agent orchestration. 

## 🏗️ Architecture

The API is built on a microservices architecture with the following components:

- **Chain Server**: Main API server using LangGraph for agent orchestration
- **Catalog Retriever**: Product search and recommendation service
- **Memory Retriever**: User context and shopping cart management
- **Guardrails**: Content safety and moderation service
- **UI**: React-based frontend for user interaction

### Software Components
- [Llama 3.1 70B Instruct NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.1-70b-instruct)
- [Llama 3.1 NemoGuard 8B - Content Safety](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemoguard-8b-content-safety)
- [Llama 3.1 NemoGuard 8B - Topic Control](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemoguard-8b-topic-control)
- [NVIDIA Retrieval QA E5 Embedding v5 ](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nv-embedqa-e5-v5)
- [NV-CLIP](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nvclip)

#### Hardware Requirements
- 4xH100

## 📘 Data Models

### `QueryRequest`

```json
{
  "user_id": 123,
  "query": "Show me red dresses under $100",
  "image": "base64_encoded_image_data",
  "context": "Previous conversation context",
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

### `QueryResponse`

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

## 🔄 API Endpoints

### POST `/query/stream`

Streams real-time responses back to the client as the shopping assistant generates them.

**Request Body**: `QueryRequest`

**Response**: Server-Sent Events (SSE) stream

**Example**:
```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "query": "Show me red dresses under $100"
  }'
```

**Streaming Response**:
```
data: {"type": "images", "payload": {"product1": "https://..."}, "timestamp": 1716400000.0}

data: {"type": "content", "payload": "I found several red dresses...", "timestamp": 1716400001.2}

data: {"type": "content", "payload": " that might interest you...", "timestamp": 1716400001.5}

data: [DONE]
```

### POST `/query/timing`

Processes a query and returns detailed timing information for performance analysis.

**Request Body**: `QueryRequest`

**Response**: `QueryResponse` with detailed timing breakdown

### GET `/health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1716400000.0,
  "version": "1.0.0"
}
```

### GET `/`

Root endpoint with API information.

**Response**:
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

## 🎯 Agent Types

The shopping assistant uses specialized agents for different tasks:

### Planner Agent
- **Purpose**: Routes user queries to appropriate specialized agents
- **Input**: User query and context
- **Output**: Next agent to handle the query

### Cart Agent
- **Purpose**: Manages shopping cart operations
- **Capabilities**: Add/remove items, view cart contents
- **Tools**: `add_to_cart`, `remove_from_cart`, `view_cart`

### Retriever Agent
- **Purpose**: Searches and retrieves product information
- **Capabilities**: Product search, image-based search, recommendations
- **Input**: Text queries or images

### Visualizer Agent
- **Purpose**: Generates visual content and visualizations
- **Capabilities**: Product visualization, scene generation
- **Use Cases**: "What would this look like in my room?"

### Chatter Agent
- **Purpose**: Generates natural language responses
- **Capabilities**: Conversational responses, context-aware replies
- **Features**: Streaming response generation

### Summary Agent
- **Purpose**: Summarizes and finalizes responses
- **Capabilities**: Response refinement, context summarization

## 🔒 Content Safety

The API includes built-in content safety through guardrails:

- **Input Safety**: Checks user queries for inappropriate content
- **Output Safety**: Validates generated responses
- **Fallback**: Returns safe default messages for flagged content

## ⚡ Performance

The API provides detailed timing information:

- **Total Time**: Complete request processing time
- **Agent Timings**: Individual agent processing times
- **Memory Access**: Context and cart retrieval time
- **Safety Checks**: Guardrails processing time

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- NVIDIA AI Endpoints access (for LLM)

### Quick Start

#### Environment Variables
You will need to set the nim cache directory and set a number of API keys to pull down the required NIM containers. Note that there are many keys here in case a user wants to use other OpenAI API compliant models for specific tasks, but a single NVIDIA NGC key will work if you are not using different services. In the below code block you can simply replace the NGC_API_KEY with your key, and the rest will be automatically filled in the same way.
```bash
# Set the NIM cache.
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod a+w "$LOCAL_NIM_CACHE"

export NGC_API_KEY=[YOUR API KEY]
export LLM_API_KEY=$NGC_API_KEY
export EMBED_API_KEY=$NGC_API_KEY
export RAIL_API_KEY=$NGC_API_KEY
```

#### Launching
```bash
# Clone the repository
git clone <repository-url>
cd shopping-assistant-demo

# Launch the local NIMs
docker compose -f docker-compose-nim-local.yaml up -d

# Build and launch the service-containters.
docker compose -f docker-compose.yaml up -d --build
```

#### Accessing

Once launched, you can access the solution by navigating to `http://localhost:3000`.

#### Configuration

You can change many important properties such as context length, top_k for vector retrieval, and various prompts within the configuration file in `langgraph/chain_server/app/config.yaml`.


