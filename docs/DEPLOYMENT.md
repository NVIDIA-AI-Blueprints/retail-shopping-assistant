# 🚀 Deployment Guide

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Fresh Deployment](#-fresh-deployment)
- [Deployment Options](#%EF%B8%8F-deployment-options)
- [Local Deployment](#-local-deployment)
- [Cloud Deployment](#%EF%B8%8F-cloud-deployment)
- [Production Deployment](#-production-deployment)
- [Configuration](#%EF%B8%8F-configuration)
- [Monitoring](#-monitoring)
- [Troubleshooting](#%EF%B8%8F-troubleshooting)

## 🎯 Overview

This guide covers deploying the Retail Shopping Assistant. Model routing lives
in one file, `shared/configs/models.yaml`. Each model role can independently
use an external endpoint, a local NIM container, or be disabled.

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
- **Docker**: Version 20.10+ with Docker Compose plugin
- **NVIDIA Container Toolkit**: For GPU acceleration
- **NVIDIA Drivers**: Latest compatible drivers
- **Git**: For repository cloning
- **Python deploy helper dependencies**: From the cloned repo, install on the
  host with the same Python interpreter used to run `scripts/model_config.py`:
  ```bash
  python -m pip install --user -r requirements-deploy.txt
  ```

#### Optional Software
- **Kubernetes**: For production orchestration
- **Helm**: For Kubernetes deployments
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

## 🚀 Fresh Deployment

This is the shortest path for a new environment with hosted NVIDIA endpoints.

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

docker login nvcr.io
# Username: $oauthtoken
# Password: your NVIDIA API key

python -m pip install --user -r requirements-deploy.txt

cp .env.example .env
$EDITOR .env
source .env

python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

Open `http://localhost:3000`.

The deploy helper resolves models from `shared/configs/models.yaml`, starts
only local NIM containers referenced by roles with `source: local_nim`, and then
starts the app stack from `docker-compose.yaml`.

The env file is a sourceable shell profile. Source the profile you want before
validation or deployment; `COMPOSE_DISABLE_ENV_FILE=1` keeps Docker Compose
from auto-parsing repo-root `.env` as dotenv and mixing environments.

For an existing deployment, rebuild the Milvus catalog embeddings when changing
the text/image embedding model or catalog data source. The catalog service skips
collections that are already populated, so stale vectors can otherwise remain
from the previous model configuration.

## 🎛️ Deployment Options

Model routing is per role:

| `source` | Meaning | Local NIMs started |
|----------|---------|--------------------|
| `endpoint` | Use the role's `base_url`/`model` or env overrides | none |
| `local_nim` | Start and use the referenced local NIM service | that service only |
| `disabled` | Capability is intentionally unavailable | none |

Use `shared/configs/models.yaml` to choose the source for each role. Copy
`.env.example` to a private env profile such as `.env`, `.env.hosted`, or
`.env.local-nim`, edit it, then `source` the profile before running validation,
deployment, or raw Docker Compose commands.

Set `CATALOG_IMAGE_EMBEDDING_ENABLED=false` in the sourced profile when a
deployment should skip image embedding clients and image collection population.
Text retrieval remains enabled.

## 🏠 Local Deployment

Use this only when this machine will run local NIM containers.

### Step 1: Environment Setup

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

cp .env.example .env.local-nim
$EDITOR .env.local-nim
source .env.local-nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod a+w "$LOCAL_NIM_CACHE"
```

Then edit `shared/configs/models.yaml` and set each local role to
`source: local_nim` with the matching `local_service`.

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

### Step 4: Validate and Deploy

```bash
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
docker compose -f docker-compose.yaml logs -f
```

The helper starts only the NIM services referenced by roles with
`source: local_nim`, then starts the application services.

### Step 5: Verify Deployment

```bash
docker compose -f docker-compose.yaml ps
docker compose -f docker-compose-nim-local.yaml ps
curl http://localhost:8009/health
curl http://localhost:3000
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

# Create and source an environment profile for hosted endpoints
cp .env.example .env.hosted
$EDITOR .env.hosted
source .env.hosted
```

### Step 2: Validate Model Routing

```bash
python scripts/model_config.py show --validate
```

### Step 3: Deploy Application

```bash
# Start application services only
python scripts/model_config.py deploy --build

# Monitor startup
docker compose -f docker-compose.yaml logs -f
```

### Step 4: Verify Deployment

```bash
# Check service status
docker compose -f docker-compose.yaml ps
```

## 🏭 Production Deployment

### Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.24+)
- Helm (3.0+)
- NVIDIA GPU Operator installed
- Ingress controller configured

#### Step 1: Create Namespace

```bash
kubectl create namespace retail-assistant
kubectl config set-context --current --namespace=retail-assistant
```

#### Step 2: Create ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: retail-assistant-config
data:
  config.yaml: |
    llm_port: "https://api.nvcf.nvidia.com/v1/chat/completions"
    llm_name: "meta/llama-3.1-70b-instruct"
    retriever_port: "https://api.nvcf.nvidia.com/v1/embeddings"
    memory_port: "http://memory-retriever:8011"
    rails_port: "https://api.nvcf.nvidia.com/v1/chat/completions"
    memory_length: 16384
    top_k_retrieve: 4
    multimodal: true
```

#### Step 3: Create Secret

```bash
kubectl create secret generic nvidia-api-keys \
  --from-literal=ngc-api-key=your_nvapi_key_here \
  --from-literal=llm-api-key=your_nvapi_key_here \
  --from-literal=embed-api-key=your_nvapi_key_here \
  --from-literal=rail-api-key=your_nvapi_key_here
```

#### Step 4: Deploy with Helm

```bash
# Add Helm repository (if using a chart)
helm repo add retail-assistant https://charts.example.com
helm repo update

# Deploy the application
helm install retail-assistant retail-assistant/retail-assistant \
  --namespace retail-assistant \
  --set nvidiaApiKey=your_nvapi_key_here
```

### Docker Swarm Deployment

#### Step 1: Initialize Swarm

```bash
docker swarm init
```

#### Step 2: Create Secrets

```bash
echo "your_nvapi_key_here" | docker secret create ngc-api-key -
echo "your_nvapi_key_here" | docker secret create llm-api-key -
echo "your_nvapi_key_here" | docker secret create embed-api-key -
echo "your_nvapi_key_here" | docker secret create rail-api-key -
```

#### Step 3: Deploy Stack

```bash
docker stack deploy -c docker-compose.prod.yaml retail-assistant
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NGC_API_KEY` | NVIDIA NGC API key | Yes | - |
| `LLM_API_KEY` | Language model API key | Yes | - |
| `VLM_API_KEY` | Optional VLM media perception API key | When `vlm` uses an authenticated endpoint | - |
| `EMBED_API_KEY` | Embedding model API key | Yes | - |
| `RAIL_API_KEY` | Guardrails API key | Yes | - |
| `CATALOG_SEARCH_TIMEOUT_SECONDS` | Optional chain-server timeout for catalog search requests | No | no timeout |
| `LOCAL_NIM_CACHE` | NIM cache directory | Local only | `~/.cache/nim` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `NODE_ENV` | Node environment | No | `production` |

### Configuration File

The main configuration is in `chain_server/config/config.yaml`:

```yaml
# NIM Endpoints
llm_port: "http://localhost:8000/v1"  # or cloud endpoint
llm_name: "meta/llama-3.1-70b-instruct"
retriever_port: "http://localhost:8010"
memory_port: "http://localhost:8011"
rails_port: "http://localhost:8012"

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

### Model Routing

Model endpoints are selected from one file: `shared/configs/models.yaml`.
Service behavior stays in each service's normal config file, while model base
URLs, model names, API-key environment variables, and local NIM service metadata
live in `models.yaml`.

Each role has a `source`:

| Source | Use case |
|--------|----------|
| `endpoint` | Hosted NVIDIA endpoint, remote NIM endpoint, or any OpenAI-compatible HTTP endpoint |
| `local_nim` | A NIM service started from `docker-compose-nim-local.yaml` by the deploy helper |
| `disabled` | Optional capability intentionally turned off for a deployment |

If `api_key_env` is set, `show --validate` requires that environment variable
to be present and the runtime sends it to the model endpoint. For local NIM
roles that do not need request-time auth, use `api_key_env: null`. Local NIM
container startup credentials are separate and are listed once under
`local_nims.required_env`.

The `vlm` role controls image/video media perception for user uploads. It uses
a hosted endpoint by default and can be set to `disabled` when media perception
should be off. Image embedding search remains controlled separately by the
`image_embedding` role and `CATALOG_IMAGE_EMBEDDING_ENABLED`.

#### Standard Deployment Flow

```bash
python -m pip install --user -r requirements-deploy.txt
cp .env.example .env
$EDITOR .env
source .env

python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

`show --validate` prints the resolved model routing without printing key values.
It fails if a required API-key variable or endpoint variable is missing.

For fully local NIMs:

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

Before running that command, edit each desired role in
`shared/configs/models.yaml` to use `source: local_nim`.

For a single remote NIM host in local app-code mode:

```bash
python skills/retail-local-runner/scripts/local_runner.py configure --nim-host http://HOST
python skills/retail-local-runner/scripts/local_runner.py start
```

The local runner writes ignored `.local-run/model-endpoints.env` with the
derived per-role base URLs.

#### Adding or Changing Models

Edit `shared/configs/models.yaml` and update one role entry:

```yaml
models:
  app_llm:
    source: endpoint
    provider: openai_compatible
    base_url_env: LLM_BASE_URL
    model_env: LLM_MODEL
    api_key_env: LLM_API_KEY
```

For a Compose-managed local NIM, use `source: local_nim` and reference a local
service:

```yaml
models:
  image_embedding:
    source: local_nim
    provider: openai_compatible
    local_service: nvclip
    api_key_env: null
```

For VLM media perception through a hosted endpoint:

```yaml
models:
  vlm:
    source: endpoint
    provider: openai_compatible
    base_url_env: VLM_BASE_URL
    model_env: VLM_MODEL
    api_key_env: VLM_API_KEY
```

For VLM media perception through the Compose-managed local Omni NIM:

```yaml
models:
  vlm:
    source: local_nim
    provider: openai_compatible
    local_service: nemotron_omni
    api_key_env: null
```

Then deploy with:

```bash
export LLM_BASE_URL=https://your-endpoint/v1
export LLM_MODEL=your-model-name
export LLM_API_KEY=...
python scripts/model_config.py show --validate
python scripts/model_config.py deploy --build
```

For locally deployed roles, reference a `local_service` in `models.yaml`. The
deploy helper starts only those local NIM services.

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
# Check individual services
curl http://localhost:8009/health  # Chain server
curl http://localhost:8010/health  # Catalog retriever
curl http://localhost:8011/health  # Memory retriever
curl http://localhost:8012/health  # Guardrails
curl http://localhost:3000         # UI
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

### Metrics Collection

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'retail-assistant'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8010', 'localhost:8011']
```

#### Grafana Dashboard

Create a Grafana dashboard with the following metrics:
- Request rate and latency
- GPU utilization
- Memory usage
- Error rates
- Response times by agent

### Alerting

Set up alerts for:
- Service health status
- High error rates
- GPU memory usage
- Response time degradation
- API key expiration

## 🛠️ Troubleshooting

### Common Issues

#### 1. NIM Container Pull Failures

**Symptoms**: Docker pull errors for nvcr.io containers

**Solutions**:
```bash
# Verify NGC API key
echo $NGC_API_KEY

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

# Check network latency (for cloud deployment)
ping api.nvcf.nvidia.com

# Optimize configuration
# Edit chain_server/app/config.yaml
top_k_retrieve: 2  # Reduce for faster responses
```

#### 5. Authentication Issues

**Symptoms**: API key errors

**Solutions**:
```bash
# Verify API key format
echo $NGC_API_KEY | head -c 10

# Check key permissions
# Ensure key has access to required NIMs

# Test API key
curl -H "Authorization: Bearer $NGC_API_KEY" \
  https://api.nvcf.nvidia.com/v1/models
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
# Backup volumes
docker run --rm -v retail-shopping-assistant_milvus_data:/data \
  -v $(pwd):/backup alpine tar czf /backup/milvus_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v retail-shopping-assistant_milvus_data:/data \
  -v $(pwd):/backup alpine tar xzf /backup/milvus_backup.tar.gz -C /data
```

## 🔒 Security Considerations

### Network Security

- Use HTTPS in production
- Implement API authentication
- Configure firewall rules
- Use VPN for remote access

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

### Horizontal Scaling

```yaml
# In docker-compose.yaml
deploy:
  replicas: 3
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

### Load Balancing

The bundled UI sends uploaded media as base64 JSON. Keep any reverse proxy
request-body limit aligned with `media_input.max_video_bytes` after base64
expansion. With the default 50 MiB raw video cap, `nginx.conf` uses
`client_max_body_size 80m`.

```yaml
# nginx.conf
upstream retail_assistant {
    server chain-server:8000;
    server chain-server:8001;
    server chain-server:8002;
}
```

### Auto-scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: retail-assistant-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: retail-assistant
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

For more information, see the [main README](../README.md) or [API documentation](API.md). 
