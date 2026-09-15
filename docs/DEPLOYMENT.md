# 🚀 Deployment Guide

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Deployment Options](#%EF%B8%8F-deployment-options)
- [Local Deployment](#-local-deployment)
- [Cloud Deployment](#%EF%B8%8F-cloud-deployment)
- [Production Deployment](#-production-deployment)
- [Configuration](#%EF%B8%8F-configuration)
- [Monitoring](#-monitoring)
- [Troubleshooting](#%EF%B8%8F-troubleshooting)

## 🎯 Overview

This guide covers local development and evaluation deployments of the Retail
Shopping Assistant. The supplied artifacts are a reference blueprint, not a
production-ready or multi-tenant service. Host ports bind to loopback by default;
see [SECURITY.md](../SECURITY.md) before introducing any remote access.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 20.04+ or equivalent Linux distribution
- **CPU**: 8+ cores
- **RAM**: 32GB system memory
- **Storage**: 50GB available disk space
- **Network**: Stable internet connection

#### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 16+ cores
- **RAM**: 128GB+ system memory
- **Storage**: 100GB+ available disk space
- **GPUs**: 4x H100 (for local NIM deployment)
- **Network**: High-speed internet connection

### Software Dependencies

#### Required Software
- **Docker Engine**: Version 28.3.3+ with Docker Compose plugin (required for
  reliable loopback-only port publishing)
- **NVIDIA Container Toolkit**: For GPU acceleration
- **NVIDIA Drivers**: Latest compatible drivers
- **Git**: For repository cloning

#### Optional Software
- **Kubernetes**: For downstream product adaptation (not supplied here)
- **Helm**: For downstream Kubernetes adaptation (not supplied here)
- **Prometheus**: For monitoring
- **Grafana**: For visualization

### NVIDIA Account Setup

1. **Create NVIDIA Account**:
   - Visit [NVIDIA NGC](https://ngc.nvidia.com/)
   - Sign up for a free account

2. **Generate API Key**:
   - Navigate to **API Keys** in your account settings
   - Generate a new API key
   - Copy the key (starts with `nvapi-`)

3. **Accept Terms**:
   - Accept the terms of service for required NIM containers
   - Ensure you have access to the NVIDIA Container Registry

## 🎛️ Deployment Options

### Option 1: Local NIM Deployment (Recommended)

**Best for**: Local development and evaluation with GPU resources

**Pros**:
- Maximum performance and low latency
- Complete privacy and data control
- No ongoing cloud costs
- Full customization capabilities

**Cons**:
- Requires significant GPU resources (4x H100)
- Higher initial hardware investment
- More complex setup and maintenance

### Option 2: Cloud NIM Deployment

**Best for**: Local development and evaluation without local GPUs

**Pros**:
- No local GPU requirements
- Faster initial setup
- Pay-per-use pricing
- Managed infrastructure

**Cons**:
- Ongoing cloud costs
- Network latency
- Data privacy considerations
- API rate limits

Cloud mode uses `nvidia/llama-nemotron-embed-vl-1b-v2` for image embeddings.
Local NIM mode continues to use NV-CLIP.

> **⚠️ Embedding model migration — existing deployments must reset Milvus.**
> The text embedder is now `nvidia/nemotron-3-embed-1b`, which emits **2048-dim**
> vectors (the previous `nv-embedqa-e5-v5` emitted 1024). A collection created by
> the old model will reject every query with
> `vector dimension mismatch, expected vector size(byte) 4096, actual 8192`.
>
> `docker compose down -v` is **not** sufficient: Milvus data here lives in a host
> bind-mount (`./catalog_retriever/volumes/milvus`), which survives `-v`. Preserve
> the directory and force a re-seed at the new dimension:
> ```bash
> docker compose -f docker-compose.yaml down -v
> mv ./catalog_retriever/volumes ./catalog_retriever/volumes.previous
> ```
> Fresh checkouts are unaffected. The text collection is 2048-dimensional in
> both modes. The image collection is 1024-dimensional with local NV-CLIP and
> 2048-dimensional with the public vision embedder, so switching modes also
> requires fresh Milvus state.

### Option 3: Hybrid Deployment

**Best for**: Advanced local evaluation with mixed model-hosting requirements

**Pros**:
- Flexibility in resource allocation
- Cost optimization
- Scalability options

**Cons**:
- More complex configuration
- Network management overhead

## 🏠 Local Deployment

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

# Create NIM cache directory
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod a+w "$LOCAL_NIM_CACHE"

# Set environment variables
export NGC_API_KEY=your_nvapi_key_here
export LLM_API_KEY=$NGC_API_KEY
export EMBED_API_KEY=$NGC_API_KEY
export RAIL_API_KEY=$NGC_API_KEY
```

### Step 2: Verify GPU Setup

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Step 3: Authenticate with NVIDIA Registry

```bash
# Login to NVIDIA Container Registry
docker login nvcr.io

# Username: oauthtoken
# Password: your_nvapi_key_here
```

### Step 4: Deploy NIMs

```bash
# Start local NIMs
docker compose -f docker-compose-nim-local.yaml up -d

# Monitor NIM startup
docker compose -f docker-compose-nim-local.yaml logs -f

# Wait for all NIMs to be ready (check logs for "ready" messages)
```

### Step 5: Deploy Application

```bash
# Build and start application services
docker compose -f docker-compose.yaml up -d --build

# Monitor application startup
docker compose -f docker-compose.yaml logs -f
```

### Step 6: Verify Deployment

```bash
# Check service status
docker compose -f docker-compose.yaml ps

# Test API endpoints
curl http://localhost:3000
curl http://localhost:3000/api/health

# Check NIM status
docker compose -f docker-compose-nim-local.yaml ps
```

## ☁️ Cloud Deployment

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

# Authenticate with NVIDIA Container Registry
docker login nvcr.io
# Use oauthtoken as the username and your NGC API key as the password

# Set environment variables for cloud NIMs
export NGC_API_KEY=your_nvapi_key_here
export LLM_API_KEY=$NGC_API_KEY
export EMBED_API_KEY=$NGC_API_KEY
export RAIL_API_KEY=$NGC_API_KEY
```

### Step 2: Configure Cloud Endpoints

# Set environment variable
export CONFIG_OVERRIDE=config-build.yaml

### Step 3: Deploy Application

```bash
# Start application services only
docker compose -f docker-compose.yaml up -d --build

# Monitor startup
docker compose -f docker-compose.yaml logs -f
```

### Step 4: Verify Deployment

```bash
# Check service status
docker compose -f docker-compose.yaml ps
```

## 🏭 Production Deployment

The repository does not ship a supported production deployment. Kubernetes,
Docker Swarm, public ingress, and multi-user operation require a separate
security architecture and deployment implementation.

At minimum, production adopters must provide authenticated TLS ingress,
server-derived identity, per-user authorization, tenant isolation, private
service networking, secrets management, rate limiting, audit logging, monitoring,
data-retention controls, dependency and image maintenance, backup/recovery, and a
deployment-specific threat analysis. Do not adapt the loopback reference Compose
file by merely publishing its ports.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NGC_API_KEY` | NVIDIA NGC API key | Yes | - |
| `LLM_API_KEY` | Language model API key | Yes | - |
| `EMBED_API_KEY` | Embedding model API key | Yes | - |
| `RAIL_API_KEY` | Guardrails API key | Yes | - |
| `LOCAL_NIM_CACHE` | NIM cache directory | Local only | `~/.cache/nim` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `NODE_ENV` | Node environment | No | `production` |

### Configuration File

The main configuration is in `shared/configs/chain_server/config.yaml`:

```yaml
# NIM Endpoints
llm_port: "http://nemotron:8000/v1"  # or cloud endpoint
llm_name: "nvidia/nemotron-3-super-120b-a12b"
retriever_port: "http://catalog-retriever:8010"
memory_port: "http://memory-retriever:8011"
rails_port: "http://rails:8012"

# Agent Prompts
routing_prompt: |
  You are a retail store assistant that routes customer queries...

chatter_prompt: |
  You are a helpful shopping assistant specializing in...
```

### Updating Categories

The system uses a static list of product categories for classification and retrieval. These categories are defined in the configuration file and should be updated when new product types are added to the system.

#### Current Categories

The following categories are currently supported:
- **Bags**: Handbags, purses, clutches
- **Sunglasses**: Eyewear and sun protection
- **Dresses**: Various dress styles and lengths
- **Skirts**: Different skirt types and lengths
- **Top/Blouse/Sweater**: Upper body garments
- **Shoes**: Footwear including heels, flats, and sandals
- **Earrings**: Jewelry worn on the lobe or edge of the ear
- **Bracelets**: Jewelry worn on the wrist or arm
- **Necklaces**: Jawelry wrong around the neck

#### How to Update Categories

1. **Edit Configuration Files**: Update the categories list in `shared/configs/chain_server/config.yaml`
2. **Restart Services**: After updating categories, restart the chain server and catalog retriever services
3. **Update Product Data**: Ensure new products in your catalog are tagged with the appropriate categories
4. **Test Classification**: Verify that the LLM can properly classify queries into the new categories

#### Configuration File Location

```yaml
# shared/configs/chain_server/config.yaml
categories: [
    "bag",
    "sunglasses", 
    "dress",
    "skirt",
    "top blouse sweater",
    "shoes",
    "earrings",
    "bracelet",
    "necklace"
]
```

### Configuration Override System

The application supports a flexible configuration override system that allows you to switch between different deployment scenarios without modifying the base configuration files.

#### How It Works

1. **Base Configuration**: Services load their base file from
   `shared/configs/<service>/`
2. **Override Detection**: If the `CONFIG_OVERRIDE` environment variable is set, the system looks for an override file
3. **Merge Process**: The override file values are merged into the base configuration, with override values taking precedence

#### Environment Variable

| Variable | Description | Example Values |
|----------|-------------|----------------|
| `CONFIG_OVERRIDE` | Specifies the override config file name | `config-build.yaml`, `config-custom.yaml` |

#### Default Configuration (Local NIMs)

By default, the application uses local NIM endpoints. No environment variable is needed:

```bash
# Deploy with local NIMs (default)
docker compose -f docker-compose-nim-local.yaml up -d
docker compose -f docker-compose.yaml up -d --build
```

**Chain Server Default** (`shared/configs/chain_server/config.yaml`):
```yaml
# LLM endpoint for local NIM deployment
llm_port: "http://nemotron:8000/v1"
llm_name: "nvidia/nemotron-3-super-120b-a12b"
```

**Catalog Retriever Default** (`shared/configs/catalog_retriever/config.yaml`):
```yaml
# Text embedding endpoint for local NIM deployment
text_embed_port: "http://embedqa:8000/v1"
text_model_name: "nvidia/nemotron-3-embed-1b"

# Image embedding endpoint for local NIM deployment
image_embed_port: "http://nvclip:8000/v1"
image_model_name: "nvidia/nvclip"
```

**Guardrails Default** (`shared/configs/rails/config.yml`):
```yaml
models:
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    parameters:
      base_url: http://content:8000/v1

  - type: topic_control
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-topic-control
    parameters:
      base_url: http://topic_control:8000/v1
```

#### Cloud NIM Deployment (`config-build.yaml`)

Use this when using NVIDIA API Catalog hosted endpoints:

```bash
# Set environment variable
export CONFIG_OVERRIDE=config-build.yaml

# Deploy without local NIMs
docker compose -f docker-compose.yaml up -d --build
```

**Chain Server Override** (`shared/configs/chain_server/config-build.yaml`):
```yaml
# Public NVIDIA API endpoint
llm_port: "https://integrate.api.nvidia.com/v1"
llm_name: "nvidia/nemotron-3-super-120b-a12b"
```

**Catalog Retriever Override** (`shared/configs/catalog_retriever/config-build.yaml`):
```yaml
# Public NVIDIA API endpoint
text_embed_port: "https://integrate.api.nvidia.com/v1"
text_model_name: "nvidia/nemotron-3-embed-1b"

# Image embedding endpoint for build.nvidia.com
image_embed_port: "https://integrate.api.nvidia.com/v1"
image_model_name: "nvidia/llama-nemotron-embed-vl-1b-v2"
```

The public vision embedder requires image inputs to be sent with
`input_type: passage` and `modality: image`; the catalog retriever adds those
parameters automatically for this model.

**Guardrails Override** (`shared/configs/rails/config-build.yaml`):
```yaml
models:
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemotron-safety-guard-8b-v3
    parameters:
      base_url: https://integrate.api.nvidia.com/v1

  - type: topic_control
    engine: nim
    model: nvidia/nemotron-3.5-lightning-30b-a3b
    parameters:
      base_url: https://integrate.api.nvidia.com/v1
      chat_template_kwargs:
        enable_thinking: false
```

Cloud mode uses Llama 3.1 Nemotron Safety Guard 8B V3 plus Nemotron 3.5
Lightning as the topic classifier. Reasoning is disabled for topic control so
the rail receives the required `on-topic` or `off-topic` label. Local NIM mode
continues to use the dedicated content-safety and topic-control NIMs.

#### Creating Custom Override Files

You can create your own override files for custom configurations:

1. **Create the override file** in the same directory as the base config:
   ```bash
   # For chain server
   cp shared/configs/chain_server/config.yaml shared/configs/chain_server/config-custom.yaml
   
   # For catalog retriever
   cp shared/configs/catalog_retriever/config.yaml shared/configs/catalog_retriever/config-custom.yaml
   
   # For guardrails
   cp shared/configs/rails/config.yml shared/configs/rails/config-custom.yaml
   ```

2. **Modify the override file** with your custom values:
   ```yaml
   # Example: Custom LLM endpoint
   llm_port: "https://your-custom-endpoint.com/v1"
   llm_name: "your-custom-model"
   
   # Example: Custom embedding endpoint
   text_embed_port: "https://your-embedding-service.com/v1"
   text_model_name: "your-embedding-model"
   
   # Example: Custom guardrails endpoints
   models:
     - type: content_safety
       engine: nim
       model: nvidia/llama-3.1-nemoguard-8b-content-safety
       parameters:
         base_url: https://your-custom-endpoint.com/v1
   ```

3. **Use the custom override**:
   ```bash
   export CONFIG_OVERRIDE=config-custom.yaml
   docker compose -f docker-compose.yaml up -d --build
   ```

#### Switching Between Configurations

Changing `CONFIG_OVERRIDE` requires Compose to recreate the affected containers;
`docker compose restart` does not apply changed environment values.

```bash
# Use local NIMs (default - no environment variable needed)
unset CONFIG_OVERRIDE
docker compose -f docker-compose.yaml up -d

# Switch to cloud NIMs
export CONFIG_OVERRIDE=config-build.yaml
docker compose -f docker-compose.yaml up -d

# Use custom configuration
export CONFIG_OVERRIDE=config-custom.yaml
docker compose -f docker-compose.yaml up -d
```

#### Docker Compose Integration

You can also set the override in your docker-compose files:

```yaml
# In docker-compose.yaml
services:
  chain-server:
    environment:
      - CONFIG_OVERRIDE=${CONFIG_OVERRIDE:-config-local.yaml}
  
  catalog-retriever:
    environment:
      - CONFIG_OVERRIDE=${CONFIG_OVERRIDE:-config-local.yaml}
  
  rails:
    environment:
      - CONFIG_OVERRIDE=${CONFIG_OVERRIDE:-config-local.yaml}
```

Then use it:
```bash
# Use local config
CONFIG_OVERRIDE=config-local.yaml docker compose up -d

# Use cloud config
CONFIG_OVERRIDE=config-build.yaml docker compose up -d
```

### Performance Tuning

#### GPU Memory Optimization

```yaml
# In docker-compose-nim-local.yaml
environment:
  - NIM_KVCACHE_PERCENT=.5  # Adjust based on GPU memory
  - NIM_MAX_BATCH_SIZE=1    # Reduce for memory constraints
```

#### System Resource Limits

```yaml
# In docker-compose.yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

## 📊 Monitoring

### Health Checks

```bash
# Docker Compose: check the public entrypoint and routed chain-server health
curl http://localhost:3000
curl http://localhost:3000/api/health

# Inspect internal container health without publishing service ports
docker compose -f docker-compose.yaml exec chain-server \
  python -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8009/health").read().decode())'
docker compose -f docker-compose.yaml exec catalog-retriever \
  python -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8010/health").read().decode())'
docker compose -f docker-compose.yaml exec memory-retriever \
  python -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8011/health").read().decode())'
```

### Logging

```bash
# View application logs
docker compose -f docker-compose.yaml logs -f

# View NIM logs
docker compose -f docker-compose-nim-local.yaml logs -f

# View specific service logs
docker compose -f docker-compose.yaml logs -f chain-server
```

### Metrics and Alerting

The blueprint does not expose Prometheus `/metrics` endpoints or ship Grafana
dashboards and alerts. Downstream deployments must add application
instrumentation and monitoring appropriate to their environment.

## 🛠️ Troubleshooting

### Common Issues

#### 1. NIM Container Pull Failures

**Symptoms**: Docker pull errors for nvcr.io containers

**Solutions**:
```bash
# Verify that the NGC API key is present without printing it
test -n "$NGC_API_KEY" && echo "NGC_API_KEY is set" || echo "NGC_API_KEY is missing"

# Re-authenticate
docker login nvcr.io

# Clear Docker cache
docker system prune -a

# Check network connectivity
curl -I https://nvcr.io
```

#### 2. GPU Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch sizes in config
# Edit docker-compose-nim-local.yaml
environment:
  - NIM_KVCACHE_PERCENT=.3
  - NIM_MAX_BATCH_SIZE=1

# Restart NIMs
docker compose -f docker-compose-nim-local.yaml restart
```

#### 3. Service Startup Failures

**Symptoms**: Services fail to start or crash

**Solutions**:
```bash
# Check service logs
docker compose -f docker-compose.yaml logs

# Check resource usage
docker stats

# Verify dependencies
docker compose -f docker-compose.yaml ps

# Check port conflicts
sudo netstat -tulpn | grep :8000
```

#### 4. Performance Issues

**Symptoms**: Slow response times

**Solutions**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor system resources
htop

# Check public endpoint reachability
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' https://integrate.api.nvidia.com/v1/models

# Optimize configuration
# Edit shared/configs/chain_server/config.yaml
top_k_retrieve: 2  # Reduce for faster responses
```

#### 5. Authentication Issues

**Symptoms**: API key errors

**Solutions**:
```bash
# Check key permissions
# Ensure key has access to required NIMs

# Test API key
curl -H "Authorization: Bearer $NGC_API_KEY" \
  https://integrate.api.nvidia.com/v1/models
```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Restart services
docker compose -f docker-compose.yaml restart

# View debug logs
docker compose -f docker-compose.yaml logs -f
```

### Recovery Procedures

#### Service Recovery

```bash
# Restart specific service
docker compose -f docker-compose.yaml restart chain-server

# Restart all services
docker compose -f docker-compose.yaml restart

# Rebuild and restart
docker compose -f docker-compose.yaml up -d --build
```

#### Data Recovery

```bash
# Back up the host bind-mounted Milvus state
tar -czf milvus_backup.tar.gz -C catalog_retriever volumes

# Restore it only while the stack is stopped
tar -xzf milvus_backup.tar.gz -C catalog_retriever
```

## 🔒 Security Considerations

### Network Security

- Keep the supplied loopback bindings and private service network intact
- Never bind internal application, database, object-store, or model ports to a
  non-loopback address or place them behind a remote link
- Add authenticated TLS ingress before any remote or multi-user access
- Derive user identity and authorization server-side

### Data Security

- Encrypt sensitive data at rest
- Use secure API keys
- Implement access controls
- Regular security updates

### Container Security

- Scan images for vulnerabilities
- Use non-root users
- Implement resource limits
- Regular image updates

## 📈 Scaling

Scaling is outside the supplied reference deployment. It requires the production
controls described above, including authenticated ingress, server-derived
identity, shared-state and tenant-isolation design, network policy, and a
deployment-specific threat analysis.

---

For more information, see the [main README](../README.md) or [API documentation](API.md). 
