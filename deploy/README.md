# Retail Shopping Assistant Helm Chart

This Helm chart deploys the NVIDIA Retail Shopping Assistant on OpenShift. The application is an AI-powered multi-agent system that provides intelligent product search, shopping cart management, visual search capabilities, and conversational AI interactions.

## Table of Contents

- [What Gets Deployed](#what-gets-deployed)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Deployment Guide](#detailed-deployment-guide)
- [Configuration Reference](#configuration-reference)
- [Upgrading](#upgrading)
- [Uninstalling](#uninstalling)
- [Troubleshooting](#troubleshooting)

## What Gets Deployed

This Helm chart deploys the following components:

### Application Services

| Component | Description | Port | Purpose |
|-----------|-------------|------|---------|
| **Chain Server** | FastAPI-based orchestration service | 8009 | Central coordinator using LangGraph for multi-agent workflows, handles conversation flow, planning, and tool execution |
| **Catalog Retriever** | Product search microservice | 8010 | Vector similarity search for products using Milvus, supports text and image-based queries |
| **Memory Retriever** | Conversation memory service | 8011 | Stores and retrieves conversation history, supports both SQLite and PostgreSQL backends |
| **Guardrails (Rails)** | Content moderation service | 8012 | NVIDIA NeMo Guardrails for input/output filtering and safety checks |
| **Frontend** | React-based web UI | 8080 | User interface for chat interactions, product browsing, and cart management |
| **Nginx** | Reverse proxy | 8080 | Routes traffic between frontend and backend services, handles CORS |

### Infrastructure Services

| Component | Description | Port | Purpose |
|-----------|-------------|------|---------|
| **Milvus** | Vector database | 19530 | Stores product embeddings for similarity search |
| **etcd** | Key-value store | 2379 | Metadata storage for Milvus |
| **MinIO** | Object storage | 9000/9001 | Blob storage for Milvus data |
| **PostgreSQL** | Relational database | 5432 | (Optional) Production-ready storage for conversation memory |

### Additional Resources Created

- **ConfigMaps**: Application configuration for each service
- **Secrets**: API keys, database credentials, MinIO credentials
- **PersistentVolumeClaims**: Storage for Milvus, etcd, MinIO, and shared data
- **Services**: ClusterIP services for internal communication
- **Route**: OpenShift Route for external HTTPS access
- **NetworkPolicy**: Network segmentation for security
- **ServiceAccount**: Dedicated service account for the application

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OpenShift Route (TLS)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Nginx Reverse Proxy                          │
└─────────────────────────────────────────────────────────────────────┘
                          │                    │
              ┌───────────┘                    └───────────┐
              ▼                                            ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│   Frontend (React UI)   │              │     Chain Server (FastAPI)  │
└─────────────────────────┘              └─────────────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
     ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
     │   Catalog Retriever      │    │   Memory Retriever       │    │   Guardrails (Rails)     │
     └──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                    │                           │                                 │
                    ▼                           ▼                                 ▼
     ┌──────────────────────────┐    ┌──────────────────┐       ┌──────────────────────────────────┐
     │   Milvus Vector DB       │    │  PostgreSQL/     │       │    NVIDIA NIM Services           │
     └──────────────────────────┘    │  SQLite          │       │    (Cloud or Local)              │
                    │                └──────────────────┘       └──────────────────────────────────┘
          ┌────────┴────────┐
          ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│   etcd          │  │   MinIO         │
└─────────────────┘  └─────────────────┘
```

### Data Flow

1. **User Request**: Users interact through the React frontend, which communicates with the Nginx proxy
2. **Request Routing**: Nginx routes API requests to the Chain Server and static content to the Frontend
3. **Agent Orchestration**: The Chain Server uses LangGraph to coordinate multiple AI agents (Planner, Chatter, Summarizer)
4. **Product Search**: The Catalog Retriever performs vector similarity search in Milvus using NVIDIA embeddings
5. **Memory Management**: The Memory Retriever stores/retrieves conversation context
6. **Content Safety**: All LLM interactions pass through Guardrails for content moderation
7. **AI Inference**: NVIDIA NIM services (cloud or local) provide LLM and embedding capabilities

## Prerequisites

- OpenShift 4.12 or later
- Helm 3.x installed
- `oc` CLI configured with cluster access
- NVIDIA NGC API key with access to:
  - LLM models (e.g., `meta/llama-3.1-70b-instruct`)
  - Embedding models (e.g., `nvidia/nv-embedqa-e5-v5`, `nvidia/nvclip`)
- Sufficient cluster resources (recommended minimum):
  - 8 CPU cores
  - 16 GB RAM
  - 50 GB storage

## Quick Start

### 1. Create a namespace

> **Important:** The default `values.yaml` assumes the namespace is `retail-shopping`. If you use a different namespace, you must update the image repository paths in your values file to match (e.g., `image-registry.openshift-image-registry.svc:5000/<your-namespace>/chain-server`).

```bash
oc new-project retail-shopping
```

### 2. Create the NVIDIA API key secret

> **Note:** If you prefer to let Helm manage the secret, skip this step and set `secrets.ngcApiKey` in your values file instead.

```bash
# Secret name must match what the Helm chart expects: <release-name>-retail-shopping-assistant-nvidia-api-keys
# For a release named "retail-shopping", use:
oc create secret generic retail-shopping-retail-shopping-assistant-nvidia-api-keys \
  --from-literal=ngc-api-key='<YOUR_NGC_API_KEY>' \
  --from-literal=llm-api-key='<YOUR_NGC_API_KEY>' \
  --from-literal=embed-api-key='<YOUR_NGC_API_KEY>' \
  --from-literal=nvidia-api-key='<YOUR_NGC_API_KEY>'
```

### 3. Build and push application images

```bash
# Navigate to repository root first
cd /path/to/retail-shopping-assistant

# Create BuildConfigs and build images
oc new-build --name=chain-server --binary --strategy=docker
oc start-build chain-server --from-dir=./chain_server --follow

# IMPORTANT: catalog-retriever must be built from repo root to include shared/ data
oc new-build --name=catalog-retriever --binary --strategy=docker
oc start-build catalog-retriever --from-dir=. --follow

oc new-build --name=memory-retriever --binary --strategy=docker
oc start-build memory-retriever --from-dir=./memory_retriever --follow

oc new-build --name=rails --binary --strategy=docker
oc start-build rails --from-dir=./guardrails --follow

oc new-build --name=frontend --binary --strategy=docker
oc start-build frontend --from-dir=./ui --follow
```

### 4. Install the chart

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  --set secrets.create=false \
  --set secrets.ngcApiKey="some_real_key"
```

Or with values file:

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  -f my-values.yaml
```

## Detailed Deployment Guide

### Step 1: Prepare Your Environment

```bash
# Verify OpenShift connectivity
oc whoami
oc cluster-info

# Create and switch to the deployment namespace
oc new-project retail-shopping

# Verify Helm is installed
helm version
```

### Step 2: Configure Secrets

**Option A: Let Helm create secrets (for development)**

Create a `values.yaml` file:

```yaml
secrets:
  create: true
  ngcApiKey: "nvapi-your-api-key-here"
  minioAccessKey: "minioadmin"
  minioSecretKey: "your-secure-minio-password"
```

**Option B: Create secrets manually (recommended for production)**

> **Important:** Secret names must match the Helm chart's naming convention: `<release-name>-retail-shopping-assistant-<secret-type>`. The examples below assume a release name of `retail-shopping`.

```bash
# NVIDIA API keys (name format: <release-name>-retail-shopping-assistant-nvidia-api-keys)
oc create secret generic retail-shopping-retail-shopping-assistant-nvidia-api-keys \
  --from-literal=ngc-api-key='<YOUR_NGC_API_KEY>' \
  --from-literal=llm-api-key='<YOUR_NGC_API_KEY>' \
  --from-literal=embed-api-key='<YOUR_NGC_API_KEY>' \
  --from-literal=nvidia-api-key='<YOUR_NGC_API_KEY>'

# MinIO credentials (name format: <release-name>-retail-shopping-assistant-minio-credentials)
oc create secret generic retail-shopping-retail-shopping-assistant-minio-credentials \
  --from-literal=access-key='minioadmin' \
  --from-literal=secret-key='your-secure-password'
```

Then set `secrets.create=false` in your values.

### Step 3: Build Application Images

The application requires custom images built from the source code:

```bash
# Navigate to repository root
cd /path/to/retail-shopping-assistant

# Chain Server - Core orchestration service
oc new-build --name=chain-server --binary --strategy=docker
oc start-build chain-server --from-dir=./chain_server --follow

# Catalog Retriever - Product search service
# Build from repo root to include product data in shared/ folder
oc new-build --name=catalog-retriever --binary --strategy=docker
oc start-build catalog-retriever --from-dir=. --follow

# Memory Retriever - Conversation memory service
oc new-build --name=memory-retriever --binary --strategy=docker
oc start-build memory-retriever --from-dir=./memory_retriever --follow

# Guardrails - Content moderation service
oc new-build --name=rails --binary --strategy=docker
oc start-build rails --from-dir=./guardrails --follow

# Frontend - React UI
oc new-build --name=frontend --binary --strategy=docker
oc start-build frontend --from-dir=./ui --follow
```

Verify images are built:

```bash
oc get imagestreams
```

### Step 4: Install the Helm Chart

**Basic installation:**

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping
```

**Installation with custom values:**

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  --set secrets.create=false \
  --set nim.llmModel="meta/llama-3.1-70b-instruct" \
  --set postgresql.enabled=true
```

**Installation with values file:**

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  -f production-values.yaml
```

**Dry-run to preview resources:**

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  --dry-run --debug
```

### Step 5: Verify Deployment

> **Note:** Resource names follow the pattern `<release-name>-retail-shopping-assistant-<component>`. The examples below assume a release named `retail-shopping`.

```bash
# Check all pods are running
oc get pods -l app.kubernetes.io/instance=retail-shopping

# Check services
oc get svc -l app.kubernetes.io/instance=retail-shopping

# Get the application URL (route name is <release-name>-retail-shopping-assistant)
oc get route retail-shopping-retail-shopping-assistant -o jsonpath='{.spec.host}'

# Check pod logs if troubleshooting
oc logs -f deployment/retail-shopping-retail-shopping-assistant-chain-server
```

### Step 6: Access the Application

```bash
# Get the route URL (route name is <release-name>-retail-shopping-assistant)
echo "https://$(oc get route retail-shopping-retail-shopping-assistant -o jsonpath='{.spec.host}')"
```

## Configuration Reference

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.create` | Create secrets from values | `true` |
| `secrets.ngcApiKey` | NVIDIA NGC API key | `""` |
| `nim.useCloudNims` | Use NVIDIA API Catalog | `true` |
| `nim.llmModel` | LLM model name | `meta/llama-3.1-70b-instruct` |
| `nim.textEmbedModel` | Text embedding model | `nvidia/nv-embedqa-e5-v5` |
| `nim.imageEmbedModel` | Image embedding model | `nvidia/nvclip` |
| `customLLM.enabled` | Use custom LLM endpoint | `false` |
| `customLLM.endpoint` | Custom LLM endpoint URL | `""` |
| `customLLM.model` | Custom LLM model name | `""` |
| `customLLM.apiKey` | Custom LLM API key | `""` |
| `chainServer.replicaCount` | Chain Server replicas | `1` |
| `milvus.useOperator` | Use Milvus Operator | `false` |
| `milvus.external.enabled` | Use external Milvus | `false` |
| `postgresql.enabled` | Deploy PostgreSQL | `false` |
| `route.enabled` | Create OpenShift Route | `true` |
| `networkPolicy.enabled` | Create NetworkPolicy | `true` |

### Using Cloud NIMs (Default)

The chart defaults to using NVIDIA API Catalog for LLM and embedding services. This requires:

1. An active NGC API key with API Catalog access
2. Network access to `integrate.api.nvidia.com`

```yaml
nim:
  useCloudNims: true
  cloudEndpoint: "https://integrate.api.nvidia.com/v1"
  llmModel: "meta/llama-3.1-70b-instruct"
  textEmbedModel: "nvidia/nv-embedqa-e5-v5"
  imageEmbedModel: "nvidia/nvclip"
```

### Using Local NIMs

For air-gapped or on-premise deployments:

1. Install NVIDIA GPU Operator
2. Deploy local NIM services
3. Update values:

```yaml
nim:
  useCloudNims: false
chainServer:
  env:
    CONFIG_OVERRIDE: ""  # Use default config
```

### Using Custom LLM Endpoints

You can configure the application to use your own hosted LLM endpoint (e.g., vLLM, RHOAI Model Serving, TGI, or any OpenAI-compatible API) instead of NVIDIA NIMs.

#### Configuration

Add the `customLLM` section to your values file:

```yaml
customLLM:
  # Enable custom LLM endpoint
  enabled: true
  # Full URL to your LLM endpoint (include /v1 path if required by your API)
  endpoint: "https://your-llm-endpoint.example.com:443/v1"
  # Model name as expected by your LLM server
  model: "your-model-name"
  # API key for authentication (leave empty if not required)
  apiKey: "your-api-key-here"
```

#### Example: Red Hat OpenShift AI (RHOAI) Model Serving

```yaml
customLLM:
  enabled: true
  endpoint: "https://llama-4-scout-17b-16e-w4a16-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1"
  model: "llama-4-scout-17b-16e-instruct"
  apiKey: "your-rhoai-api-key"
```

#### Example: Self-hosted vLLM

```yaml
customLLM:
  enabled: true
  endpoint: "http://vllm-service:8000/v1"
  model: "meta-llama/Llama-3.1-70B-Instruct"
  apiKey: ""  # No auth for internal service
```

#### Via Helm Command Line

```bash
helm install retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  --set customLLM.enabled=true \
  --set customLLM.endpoint="https://your-llm-endpoint.example.com:443/v1" \
  --set customLLM.model="your-model-name" \
  --set customLLM.apiKey="your-api-key"
```

#### Important Notes

- **Endpoint Format**: Ensure your endpoint URL includes the correct path (e.g., `/v1` for OpenAI-compatible APIs)
- **Embedding Services**: The `customLLM` setting only overrides the LLM endpoint. Embedding services (text and image) still use NVIDIA NIMs configured in the `nim` section
- **API Compatibility**: Your LLM endpoint must be compatible with the OpenAI chat completions API format
- **API Key Security**: The API key is stored in a Kubernetes Secret. For production, consider using external secret management

### Using PostgreSQL (Recommended for Production)

SQLite is used by default but is not recommended for production. Enable PostgreSQL:

> **Note:** The PostgreSQL service name follows the pattern `<release-name>-retail-shopping-assistant-postgresql`. Update the `postgresUrl` accordingly.

```yaml
postgresql:
  enabled: true
secrets:
  postgres:
    enabled: true
    password: "your-secure-password"
memoryRetriever:
  database:
    usePostgres: true
    # Service name format: <release-name>-retail-shopping-assistant-postgresql
    postgresUrl: "postgresql://memoryuser:your-secure-password@retail-shopping-retail-shopping-assistant-postgresql:5432/memorydb"
```

### Using Milvus Operator

For production deployments, use the Milvus Operator instead of standalone Milvus:

```bash
# Install Milvus Operator first
helm repo add milvus-operator https://zilliztech.github.io/milvus-operator/
helm install milvus-operator milvus-operator/milvus-operator
```

```yaml
milvus:
  useOperator: true
```

### Using External Milvus (Zilliz Cloud)

```yaml
milvus:
  enabled: true
  external:
    enabled: true
    endpoint: "https://your-instance.zillizcloud.com:19530"
etcd:
  enabled: false
minio:
  enabled: false
```

### Resource Configuration

Customize resource requests and limits for each component:

```yaml
chainServer:
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"

catalogRetriever:
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
```

### Storage Configuration

```yaml
global:
  storageClass: ""  # Use cluster default
  storageClassRWX: "ocs-storagecluster-cephfs"  # For ReadWriteMany volumes

persistence:
  sharedData:
    enabled: true
    size: 10Gi
    accessMode: ReadWriteOnce  # Use ReadWriteMany with RWX storage class

milvus:
  persistence:
    enabled: true
    size: 20Gi
```

## Upgrading

```bash
# Upgrade with new values
helm upgrade retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  -f my-values.yaml

# Upgrade with specific overrides
helm upgrade retail-shopping ./deploy/retail-shopping-assistant \
  --namespace retail-shopping \
  --set chainServer.replicaCount=3

# Check upgrade history
helm history retail-shopping --namespace retail-shopping

# Rollback if needed
helm rollback retail-shopping 1 --namespace retail-shopping
```

## Uninstalling

```bash
# Uninstall the release
helm uninstall retail-shopping --namespace retail-shopping

# Clean up PVCs (optional - this deletes all data!)
oc delete pvc -l app.kubernetes.io/instance=retail-shopping

# Delete the namespace (optional)
oc delete project retail-shopping
```

## Troubleshooting

### Pods not starting

Check pod status and logs:

```bash
oc get pods
oc describe pod <pod-name>
oc logs <pod-name>
```

### Image pull errors

Ensure ImagePullSecrets are configured:

```bash
oc create secret docker-registry nvidia-registry \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='<NGC_API_KEY>' \
  --docker-email=your-email@example.com

oc secrets link default nvidia-registry --for=pull
```

### Milvus issues

If using standalone Milvus and encountering issues:

> **Note:** Deployment names follow the pattern `<release-name>-retail-shopping-assistant-<component>`. The examples below assume a release named `retail-shopping`.

```bash
# Check Milvus logs
oc logs deployment/retail-shopping-retail-shopping-assistant-milvus

# Check etcd and MinIO status
oc logs deployment/retail-shopping-retail-shopping-assistant-etcd
oc logs deployment/retail-shopping-retail-shopping-assistant-minio
```

Consider using the Milvus Operator or Zilliz Cloud for production.

### Network connectivity

Test internal connectivity:

> **Note:** Service names follow the pattern `<release-name>-retail-shopping-assistant-<component>`. The examples below assume a release named `retail-shopping`.

```bash
# Test from chain-server to catalog-retriever
oc exec -it $(oc get pod -l app.kubernetes.io/component=chain-server -o name | head -1) \
  -- curl http://retail-shopping-retail-shopping-assistant-catalog-retriever:8010/health

# Test from chain-server to memory-retriever
oc exec -it $(oc get pod -l app.kubernetes.io/component=chain-server -o name | head -1) \
  -- curl http://retail-shopping-retail-shopping-assistant-memory-retriever:8011/health
```

### API Key issues

Verify secrets are created correctly:

> **Note:** Secret names follow the pattern `<release-name>-retail-shopping-assistant-nvidia-api-keys`. The example below assumes a release named `retail-shopping`.

```bash
oc get secrets
oc describe secret retail-shopping-retail-shopping-assistant-nvidia-api-keys
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Pods stuck in `Pending` | Insufficient resources | Check node resources with `oc describe nodes` |
| `ImagePullBackOff` | Missing registry credentials | Configure ImagePullSecret |
| Chain Server `CrashLoopBackOff` | Missing API keys | Verify `<release-name>-retail-shopping-assistant-nvidia-api-keys` secret exists |
| Catalog Retriever failing | Milvus not ready | Wait for Milvus pod to be Running |
| Route not accessible | TLS issues | Check route TLS settings |

## Additional Resources

- **[OpenShift Technical Reference](../OPENSHIFT_DEPLOYMENT_GUIDE.md)** - Detailed service specifications, hardware requirements, compatibility assessments, and migration checklists

## License

Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Apache-2.0
