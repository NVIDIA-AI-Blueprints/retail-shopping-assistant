# Security Policy

## Reporting a Vulnerability

Do not report suspected vulnerabilities through a public GitHub issue or pull
request. Report them through one of these private channels:

- [NVIDIA Product Security](https://www.nvidia.com/en-us/security/) (preferred)
- Email [psirt@nvidia.com](mailto:psirt@nvidia.com); use the
  [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key) when
  appropriate

Include the affected version or commit, vulnerability type, reproduction steps,
proof-of-concept information when available, and the potential impact. NVIDIA
PSIRT will acknowledge and assess reports and coordinate fixes or advisories as
appropriate.

## Supported Deployment Scope

Retail Shopping Assistant is a public NVIDIA AI Blueprint: reference software
for local development, evaluation, and customization. It is not a
production-hardened, multi-user, or multi-tenant retail service and does not
provide built-in end-user authentication or authorization.

The supported default deployment is for a single trusted user on a local
workstation. The web entry point is bound to loopback. The chain server, catalog
retriever, memory retriever, guardrails, and supporting data services communicate
on the private application network and are not intended to be exposed directly.

The client-supplied `user_id` is a convenience key for demo session state. It is
randomized to make accidental collision or guessing unlikely, but it is not an
authenticated identity or an authorization boundary.

Do not expose the bundled deployment to a LAN, the Internet, or a tunnel that
does not enforce authenticated access for a single trusted operator. Do not use
it with real customer, personal, payment, confidential, or production data.

Any broader deployment must add authenticated TLS ingress, server-derived user
identity, per-user authorization, network policy, secrets management, rate
limiting, monitoring, audit controls, and an appropriate data-retention policy.

## Security Architecture and Context

Retail Shopping Assistant is a containerized demonstration application composed
of a React UI, an nginx entry point, an orchestration API, catalog and memory
services, content guardrails, and local data services.

**Repository Exposure Classification:** Public.

Basis: the source repository is publicly readable on GitHub.

**Service Exposure Classification:** External / Regulated (high confidence).

Basis: the software is publicly distributed for user-operated deployment as a
non-production reference blueprint. This describes distribution and exposure
context, not production readiness or vulnerability severity.

The intended trust boundaries are:

- Nginx and the UI form the only host-reachable application entry point. The
  supplied Compose configuration binds it to `127.0.0.1:3000`.
- The chain server, catalog retriever, memory retriever, guardrails, and data
  services are trusted internal components on the private application network.
- The memory service stores demonstration conversation summaries and cart state
  in SQLite, indexed by the client-supplied demo session identifier.
- Cloud configuration can send prompts or images to operator-selected model and
  embedding providers. Operators are responsible for confirming that those
  providers and their data-handling terms are appropriate.
- Content guardrails provide model-content moderation. They do not authenticate
  callers or authorize access to data.

### Threat Model

1. **Unauthorized demo-session access:** If the UI or an API is exposed to
   multiple users, a caller-controlled `user_id` could be used to access or
   modify another session's cart or conversation context.
2. **Internal API bypass:** Publishing internal service or data-store ports would
   let callers bypass the intended nginx and chain-server request path.
3. **Resource exhaustion and cost abuse:** Untrusted queries and image payloads
   can consume local resources or invoke model and embedding endpoints. The
   blueprint does not provide production rate limiting or quotas.
4. **Data disclosure:** Queries, images, conversation summaries, cart contents,
   and diagnostic logs may be stored locally or sent to configured model
   providers.
5. **Model-input manipulation:** Untrusted text and images influence agent
   routing and cart-tool selection. Content moderation reduces some unsafe model
   output but does not establish caller identity or data ownership.

### Critical Security Assumptions

- The host, local browser, and operator are trusted.
- Loopback-bound ports are not re-published through a proxy, tunnel, or port
  forwarding rule unless it enforces authenticated access for the single trusted
  operator described above.
- The private application network is not reachable by untrusted clients.
- Only public catalog data and synthetic demonstration session data are used.
- API keys are supplied through appropriate local secret handling and are never
  committed to source control.
- Configured model and embedding endpoints are trusted and approved for the data
  submitted to them.
- Deployers introducing remote or multi-user access provide authentication,
  authorization, TLS, rate limiting, audit, retention, and server-derived
  identity controls outside or in front of this blueprint.

## Production Use

This repository is a starting point, not a production security architecture.
Teams adapting it for a product must perform their own threat analysis and add
identity, authorization, tenant isolation, secure secrets management, encrypted
transport, abuse controls, auditability, data governance, dependency management,
and deployment-specific hardening before handling real users or sensitive data.
