# OpenShift Technical Reference - Retail Shopping Assistant

This document provides detailed technical specifications, compatibility assessments, and advanced configuration guidance for deploying the Retail Shopping Assistant on OpenShift.

> **For deployment instructions, see the [Helm Chart README](./deploy/README.md)** which provides the recommended deployment method using the Helm chart.

## Table of Contents

- [Service Components](#service-components)
- [OpenShift Compatibility Assessment](#openshift-compatibility-assessment)
- [Hardware Requirements](#hardware-requirements)
- [Deployment Strategy Recommendations](#deployment-strategy-recommendations)
- [Security Considerations](#security-considerations)
- [Known Limitations](#known-limitations)
- [Migration Checklist](#migration-checklist)
- [References](#references)

---

## Service Components

Detailed technical specifications for each service component.

### 1. Chain Server (Main API)

| Property | Value |
|----------|-------|
| **Image Base** | `python:3.11-slim` |
| **Port** | 8009 |
| **Framework** | FastAPI with Uvicorn |
| **Key Dependencies** | LangGraph, LangChain, OpenAI SDK, Pydantic |
| **Config Path** | `/app/shared/configs/chain_server/config.yaml` |

**Key Features:**
- Multi-agent orchestration with LangGraph (Planner, Cart, Retriever, Chatter, Summary agents)
- Streaming responses via Server-Sent Events (SSE) at `/query/stream`
- Health check endpoint at `/health`

**Environment Variables:**
| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_API_KEY` | Yes | API key for LLM access |
| `CONFIG_OVERRIDE` | No | Override config file (e.g., `config-build.yaml`) |
| `CATALOG_RETRIEVER_URL` | No | Catalog service URL (default: `http://catalog-retriever:8010`) |
| `MEMORY_RETRIEVER_URL` | No | Memory service URL (default: `http://memory-retriever:8011`) |
| `RAILS_URL` | No | Guardrails service URL (default: `http://rails:8012`) |

### 2. Catalog Retriever

| Property | Value |
|----------|-------|
| **Image Base** | `python:3.11-slim` |
| **Port** | 8010 |
| **Framework** | FastAPI with Uvicorn |
| **Key Dependencies** | LangChain-Milvus, PyMilvus, Pandas, Pillow, OpenAI SDK |
| **Config Path** | `/app/shared/configs/catalog_retriever/config.yaml` |

**Key Features:**
- Text-based product search via `/query/text`
- Image + text search via `/query/image`
- Integration with Milvus vector database
- Loads product catalog from `/app/shared/data/products_extended.csv` on startup

**Environment Variables:**
| Variable | Required | Description |
|----------|----------|-------------|
| `EMBED_API_KEY` | Yes | API key for embedding models |
| `MILVUS_HOST` | Yes | Milvus service hostname |
| `MILVUS_PORT` | Yes | Milvus port (typically `19530`) |
| `CONFIG_OVERRIDE` | No | Override config file |

### 3. Memory Retriever

| Property | Value |
|----------|-------|
| **Image Base** | `python:3.11-slim` |
| **Port** | 8011 |
| **Framework** | FastAPI with Uvicorn |
| **Key Dependencies** | SQLAlchemy, Pydantic, FastAPI |
| **Database** | SQLite (default) or PostgreSQL |

**Key Features:**
- User session context management via `/user/{user_id}/context/*` endpoints
- Shopping cart persistence via `/user/{user_id}/cart/*` endpoints
- Health check at `/health`

**API Endpoints:**
- `GET /user/{user_id}/context` - Get conversation context
- `POST /user/{user_id}/context/add` - Append to context
- `POST /user/{user_id}/context/replace` - Replace context
- `GET /user/{user_id}/cart` - Get cart contents
- `POST /user/{user_id}/cart/add` - Add item to cart
- `POST /user/{user_id}/cart/remove` - Remove item from cart
- `POST /user/{user_id}/cart/clear` - Clear entire cart

> **Note:** SQLite is used by default. For production deployments with multiple replicas, enable PostgreSQL via the Helm chart.

### 4. Guardrails Service

| Property | Value |
|----------|-------|
| **Image Base** | `python:3.11-slim` |
| **Port** | 8012 |
| **Framework** | FastAPI with NeMo Guardrails |
| **Key Dependencies** | nemoguardrails, langchain-nvidia-ai-endpoints, OpenAI SDK |
| **Config Path** | `/app/shared/configs/rails/config.yml` |

**Key Features:**
- Input content safety checks via `/rail/input/check`
- Output content safety checks via `/rail/output/check`
- Uses NVIDIA NeMo Guardrails framework

**Environment Variables:**
| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | API key for guardrails models |
| `CONFIG_OVERRIDE` | No | Override config file |

### 5. Frontend (React UI)

| Property | Value |
|----------|-------|
| **Port** | 8080 |
| **Framework** | React 18 with TypeScript |
| **Key Dependencies** | Material-UI, TailwindCSS, DOMPurify |

**Key Features:**
- Responsive chat interface with streaming support
- Image upload for visual search (drag & drop or file picker)
- Guardrails toggle in UI
- Message download and chat reset functionality

### 6. Infrastructure Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **Milvus** | `milvusdb/milvus:v2.4.13-hotfix` | 19530 | Vector database for product embeddings |
| **etcd** | `quay.io/coreos/etcd:v3.5.5` | 2379 | Milvus metadata storage |
| **MinIO** | `minio/minio:RELEASE.2023-03-20T20-16-18Z` | 9000/9001 | Milvus object storage |
| **PostgreSQL** | `postgres:15` | 5432 | (Optional) Memory retriever database |

**Milvus Collections:**
- `shopping_advisor_text_db` - Text embeddings (1024 dimensions)
- `shopping_advisor_image_db` - Image embeddings (1024 dimensions)

---

## OpenShift Compatibility Assessment

### Compatible Components

| Component | Status | Notes |
|-----------|--------|-------|
| Chain Server | ✅ Compatible | Standard Python container |
| Catalog Retriever | ✅ Compatible | Standard Python container |
| Memory Retriever | ✅ Compatible | Use PostgreSQL for production |
| Guardrails | ✅ Compatible | Standard Python container |
| Frontend | ✅ Compatible | Standard Node.js container |
| Nginx | ✅ Compatible | Uses unprivileged nginx image |
| etcd | ✅ Compatible | Available on quay.io |
| MinIO | ✅ Compatible | Standard deployment |

### Components Requiring Special Handling

| Component | Issue | Resolution |
|-----------|-------|------------|
| **Milvus** | Standalone requires `seccomp:unconfined` | Use Milvus Operator or Zilliz Cloud |
| **NVIDIA NIMs (Local)** | GPU scheduling, privileged mode | Use Cloud NIMs or NVIDIA GPU Operator |

### Security Context Constraints (SCC)

The Helm chart configures all containers to run with restricted security contexts:

```yaml
securityContext:
  runAsNonRoot: true
  seccompProfile:
    type: RuntimeDefault

containerSecurityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

---

## Hardware Requirements

### Minimum Requirements (Cloud NIMs)

| Resource | Requirement |
|----------|-------------|
| CPU | 8 cores |
| RAM | 16 GB |
| Storage | 50 GB |
| GPU | Not required |
| Network | Stable internet for API calls |

### Recommended Requirements (Local NIMs)

| Resource | Requirement |
|----------|-------------|
| CPU | 16+ cores |
| RAM | 128 GB |
| Storage | 100 GB+ SSD |
| GPU | 4x NVIDIA H100 (80GB) or 4x A100 (80GB) |
| Network | High-speed internal network |

### GPU Distribution (Local NIM Deployment)

| NIM Service | GPU Assignment | Shared Memory |
|-------------|----------------|---------------|
| Llama 3.1 70B | GPU 0, GPU 1 | 16 GB |
| NV-EmbedQA-E5-v5 | GPU 2 | Default |
| NV-CLIP | GPU 2 (shared) | Default |
| Content Safety | GPU 2 (shared) | Default |
| Topic Control | GPU 3 | Default |

---

## Deployment Strategy Recommendations

### Strategy 1: Cloud-First (Recommended)

**Best for:** Quick deployment, reduced operational complexity

- Use NVIDIA API Catalog for all NIM services
- Deploy application services on OpenShift via Helm chart
- Use managed PostgreSQL for production memory storage
- Use Milvus Operator or Zilliz Cloud for vector database

**Pros:**
- No GPU infrastructure required
- Faster time to deployment
- Reduced operational overhead

**Cons:**
- Ongoing cloud API costs
- Network latency
- Data privacy considerations

### Strategy 2: Hybrid (Balanced)

**Best for:** Production workloads with cost optimization

- Deploy application services on OpenShift
- Use NVIDIA GPU Operator for local NIM deployment
- Deploy Milvus using the Milvus Operator
- Use OpenShift Data Foundation for storage

### Strategy 3: Fully On-Premise

**Best for:** Air-gapped environments, strict data privacy requirements

- Full local NIM deployment with NVIDIA GPU Operator
- All services containerized on OpenShift
- OpenShift Data Foundation for all storage
- Requires GPU nodes in the cluster

---

## Security Considerations

### Secrets Management

The Helm chart creates secrets for:
- NVIDIA API keys (NGC, LLM, Embedding)
- MinIO credentials
- PostgreSQL credentials (if enabled)

For production, consider:
- External secret management (HashiCorp Vault, AWS Secrets Manager)
- OpenShift Secrets Store CSI Driver
- Sealed Secrets for GitOps workflows

### Network Policies

The Helm chart includes NetworkPolicies that:
- Allow ingress from OpenShift ingress controller
- Allow internal pod-to-pod communication within the namespace
- Allow egress to external NVIDIA API endpoints (port 443)

### Pod Security

All pods are configured with:
- Non-root user execution
- Read-only root filesystem (where possible)
- Dropped capabilities
- Seccomp profile enforcement

---

## Known Limitations

### Platform Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Milvus standalone requires elevated privileges | Cannot use default Milvus container | Use Milvus Operator or Zilliz Cloud |
| Local NIMs require GPU nodes | Complex infrastructure | Use Cloud NIMs (NVIDIA API Catalog) |
| SQLite not suitable for HA | Single replica only | Enable PostgreSQL in Helm values |
| RWO storage limits scaling | Pods must be on same node | Use RWX storage class (CephFS, EFS) |

### Feature Limitations

| Feature | Limitation | Notes |
|---------|------------|-------|
| Image Upload | 10MB max size | Configurable in frontend |
| Supported Categories | Fixed list in config | Update `categories` in config.yaml |
| Concurrent Users | Limited by resource allocation | Scale pods horizontally |

### API Compatibility

| Requirement | Notes |
|-------------|-------|
| LLM Endpoint | Must be OpenAI chat completions API compatible |
| Embedding Endpoint | Must support NVIDIA embedding model format |

---

## Migration Checklist

### Pre-Migration

- [ ] Obtain NVIDIA NGC API key with appropriate entitlements
- [ ] Verify OpenShift cluster has sufficient resources
- [ ] Install NVIDIA GPU Operator (if using local NIMs)
- [ ] Configure storage provisioner (RWX recommended)
- [ ] Create namespace

### Deployment

- [ ] Create ImagePullSecret for nvcr.io (if using NVIDIA images)
- [ ] Build and push application images to internal registry
- [ ] Configure Helm values for your environment
- [ ] Deploy using Helm chart
- [ ] Verify all pods are running

### Post-Deployment

- [ ] Verify health check endpoints
- [ ] Test product search functionality
- [ ] Test image upload and visual search
- [ ] Test shopping cart operations
- [ ] Verify guardrails are functioning
- [ ] Configure monitoring and alerting

### Production Readiness

- [ ] Enable PostgreSQL for memory retriever
- [ ] Configure horizontal pod autoscaling
- [ ] Set up log aggregation
- [ ] Document backup and recovery procedures
- [ ] Load test with expected user volume

---

## References

- [Helm Chart README](./deploy/README.md) - Primary deployment documentation
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
- [OpenShift Security Context Constraints](https://docs.openshift.com/container-platform/latest/authentication/managing-security-context-constraints.html)
- [Milvus Operator for Kubernetes](https://milvus.io/docs/install_cluster-milvusoperator.md)
- [NeMo Guardrails Documentation](https://github.com/NVIDIA/NeMo-Guardrails)
- [NVIDIA API Catalog](https://build.nvidia.com/) - Cloud-hosted NIMs

---

*This document provides technical reference information. For deployment instructions, see the [Helm Chart README](./deploy/README.md).*
