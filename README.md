# 🛍️ NVIDIA AI Blueprint: Retail Shopping Assistant

<div align="center">

![NVIDIA Logo](https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f)

**AI-powered retail shopping assistant with multi-agent architecture**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://www.docker.com/)

</div>

## Overview

The Retail Shopping Assistant is an AI-powered blueprint that provides a comprehensive interface for an intelligent retail shopping advisor. Built with LangGraph for agent orchestration, it features multi-agent architecture, real-time streaming responses, image-based search, and intelligent shopping cart management.

### Key Features

- 🤖 **Intelligent Product Search**: Find products using natural language or images
- 🛒 **Smart Cart Management**: Add, remove, and manage shopping cart items
- 🖼️ **Visual Search**: Upload images to find similar products
- 💬 **Conversational AI**: Natural language interactions
- 🔒 **Content Safety**: Built-in moderation and safety checks
- ⚡ **Real-time Streaming**: Live response generation
- 📱 **Responsive UI**: Modern, mobile-friendly interface

### Architecture

The application follows a microservices architecture with specialized agents for different tasks:
- **Chain Server**: Main API with LangGraph orchestration
- **Catalog Retriever**: Product search and recommendations
- **Memory Retriever**: User context and cart management
- **Guardrails**: Content safety and moderation
- **UI**: React-based frontend interface

For detailed architecture information, see [Architecture Overview](docs/README.md#architecture-overview).

## Get Started

### Prerequisites

- **Docker**: Version 20.10+ with Docker Compose plugin
- **NVIDIA NGC Account**: For API access ([Get API Key](https://ngc.nvidia.com/))
- **Hardware**: 4x H100 GPUs (for local deployment) or cloud access

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
   cd retail-shopping-assistant
   ```

2. **Set up environment**:
   ```bash
   export NGC_API_KEY=your_nvapi_key_here
   export LLM_API_KEY=$NGC_API_KEY
   export EMBED_API_KEY=$NGC_API_KEY
   export RAIL_API_KEY=$NGC_API_KEY
   export LOCAL_NIM_CACHE=~/.cache/nim
   mkdir -p "$LOCAL_NIM_CACHE"
   chmod a+w "$LOCAL_NIM_CACHE"
   ```

3. **Launch the application**:
   ```bash
   # Start local NIMs (requires 4x H100 GPUs)
   docker compose -f docker-compose-nim-local.yaml up -d
   
   # Build and launch the application
   docker compose -f docker-compose.yaml up -d --build
   ```

4. **Access the application**: Open your browser to `http://localhost:3000`

For detailed installation instructions, see [Deployment Guide](docs/DEPLOYMENT.md).

## Documentation

- **[User Guide](docs/USER_GUIDE.md)**: How to use the application
- **[API Documentation](docs/API.md)**: Complete API reference
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Installation and setup instructions
- **[Documentation Hub](docs/README.md)**: Complete documentation index

## Contribution Guidelines

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup and environment configuration
- Coding standards and best practices
- Testing guidelines and examples
- Pull request process and code review guidelines

## Community

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/discussions)
- **Documentation**: [Comprehensive guides and references](docs/README.md)

## References

### NVIDIA AI Blueprints
- [NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints): Collection of AI application blueprints
- [NVIDIA NIM](https://catalog.ngc.nvidia.com/orgs/nim): Containerized AI models
- [NVIDIA NGC](https://ngc.nvidia.com/): AI platform and container registry

### Technologies Used
- [LangGraph](https://github.com/langchain-ai/langgraph): Agent orchestration framework
- [FastAPI](https://fastapi.tiangolo.com/): Modern Python web framework
- [React](https://reactjs.org/): JavaScript library for building user interfaces
- [Milvus](https://milvus.io/): Vector database for similarity search

### Related Projects
- [NVIDIA Retrieval QA](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nv-embedqa-e5-v5): Embedding model for semantic search
- [NV-CLIP](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nvclip): Visual understanding model
- [Llama 3.1](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.1-70b-instruct): Large language model

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by NVIDIA AI Blueprints**

[Back to Top](#nvidia-ai-blueprint-retail-shopping-assistant)

</div>


