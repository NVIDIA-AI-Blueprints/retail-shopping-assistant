# 🚀 Deployment Guide

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This guide covers deploying the Retail Shopping Assistant in various environments, from local development to production. The application supports multiple deployment strategies to accommodate different hardware configurations and use cases.

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

## 🎛️ Deployment Options

### Option 1: Local NIM Deployment (Recommended)

**Best for**: Development, testing, production with GPU resources

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

**Best for**: Development, testing, production without local GPUs

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

### Option 3: Hybrid Deployment

**Best for**: Production with mixed requirements

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
curl http://localhost:8000/health
curl http://localhost:3000

# Check NIM status
docker compose -f docker-compose-nim-local.yaml ps
```

## ☁️ Cloud Deployment

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
cd retail-shopping-assistant

# Set environment variables for cloud NIMs
export NGC_API_KEY=your_nvapi_key_here
export LLM_API_KEY=$NGC_API_KEY
export EMBED_API_KEY=$NGC_API_KEY
export RAIL_API_KEY=$NGC_API_KEY
```

### Step 2: Configure Cloud Endpoints

Edit `chain_server/app/config.yaml`:

```yaml
# Cloud NIM endpoints
llm_port: "https://api.nvcf.nvidia.com/v1/chat/completions"
llm_name: "meta/llama-3.1-70b-instruct"
retriever_port: "https://api.nvcf.nvidia.com/v1/embeddings"
memory_port: "http://localhost:8011"
rails_port: "https://api.nvcf.nvidia.com/v1/chat/completions"
```

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

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:3000
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
| `EMBED_API_KEY` | Embedding model API key | Yes | - |
| `RAIL_API_KEY` | Guardrails API key | Yes | - |
| `LOCAL_NIM_CACHE` | NIM cache directory | Local only | `~/.cache/nim` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `NODE_ENV` | Node environment | No | `production` |

### Configuration File

The main configuration is in `chain_server/app/config.yaml`:

```yaml
# NIM Endpoints
llm_port: "http://localhost:8000/v1"  # or cloud endpoint
llm_name: "meta/llama-3.1-70b-instruct"
retriever_port: "http://localhost:8010"
memory_port: "http://localhost:8011"
rails_port: "http://localhost:8012"

# Agent Configuration
memory_length: 16384
top_k_retrieve: 4
multimodal: true

# Product Categories
categories: [
    "bag",
    "sunglasses", 
    "dress",
    "skirt",
    "top blouse sweater",
    "shoes"
]

# Agent Prompts
routing_prompt: |
  You are a retail store assistant that routes customer queries...

chatter_prompt: |
  You are a helpful shopping assistant specializing in...
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
# Check service health
curl http://localhost:8000/health

# Check individual services
curl http://localhost:8010/health  # Catalog retriever
curl http://localhost:8011/health  # Memory retriever
curl http://localhost:8012/health  # Guardrails
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