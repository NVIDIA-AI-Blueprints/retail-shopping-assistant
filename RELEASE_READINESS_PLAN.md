# Release Readiness Plan

**Companion to:** [`RELEASE_READINESS_AUDIT.md`](RELEASE_READINESS_AUDIT.md)
**Starting point:** `staging` at `1853f94`
**Date:** 17 September 2026

This plan turns every finding in the audit into ordered steps. Each step says what to do, why it matters, which files are involved, and how to prove it worked.

---

## How to use this plan

The work is grouped into four phases. **The phases are ordered deliberately** — each one is safe to stop after, and each leaves the repository in a better and still-consistent state.

| Phase | What it achieves | Rough effort | Can you ship after this? |
|---|---|---|---|
| **Phase 0** | Fast, zero-risk corrections | Half a day | No, but the repo stops misleading people |
| **Phase 1** | Fixes the four blockers | 2–3 days | **Yes — as a developer blueprint** |
| **Phase 2** | Security, correctness, process hardening | 1–2 weeks | Yes, with more confidence |
| **Phase 3** | Structural cleanup and long-term health | Ongoing | Not release-gating |

**Recommended release cut: after Phase 1, with Phase 2 items either done or explicitly written down as known limitations.**

Two ground rules carried over from the project's own `AGENTS.md`, which are worth honouring while working through this:

- Make the smallest coherent change, and validate it before moving on.
- Never quote a single evaluation run as proof. If a change could affect answer quality, repeated runs are the only meaningful signal.

### Before you start

Confirm you are on a clean tree and that the baseline is green, so that any later failure is attributable to your change and not to something pre-existing.

```bash
cd /home/ubuntu/retail-shopping-assistant
git status --porcelain                                    # expect no output
python skills/retail-test-runner/scripts/run_retail_tests.py unit
```

Baseline to compare against for the rest of this plan: **1,485 passed, 13 skipped, 1 xfailed**.

---

## Phase 0 — Stop the documentation from misleading people

Nothing here touches application code. These are corrections to files that currently state things that are not true. Do this first because every later step is easier when the docs describe reality, and because a contributor following today's `AGENTS.md` will actively break the weather feature.

### Step 0.1 — Correct the weather section in `AGENTS.md`

**Why:** `AGENTS.md` tells coding agents and contributors to keep the weather tool out of registration, policy, skill grants, and prompts. All four statements are now false, and following them means deleting working code. *(Audit: B3)*

**Edit** `AGENTS.md` at lines 30 and 423–428 to state what is actually true:

- The tool **is** registered — `deepagents_runtime.py:2927`
- It **is** in the immutable policy — `tool_policy.py:143-146`
- It **is** granted by the `destination-weather` skill
- It **is** described in prompts — `deepagents_runtime.py:3002`
- It ships **disabled by default**, and that part should be preserved and stated clearly

**Verify:**
```bash
grep -n "dormant" AGENTS.md          # remaining hits should describe history, not current rules
python scripts/check_agents_md_config_refs.py
```

### Step 0.2 — Fix the same inverted claim in the other three documents

**Why:** the identical wrong statement appears in several places, and fixing only `AGENTS.md` leaves three more traps. *(Audit: B3)*

| File | Lines |
|---|---|
| `STATUS.md` | 31–41 |
| `docs/API.md` | 488–495 |
| `docs/USER_GUIDE.md` | 78–82 |

For `USER_GUIDE.md`, the user-facing framing should be that weather is an operator-enabled capability which is off by default — not that the assistant cannot do weather.

### Step 0.3 — Correct the tool and skill counts everywhere

**Why:** the docs say eleven tools and five skills; the code has twelve and seven. *(Audit: M1)*

The twelve commerce tools are listed at `deepagents_runtime.py:2908-2921`. The missing one in the docs is `describe_catalog_tool`. Update:

- `docs/SHOPPER_AGENT_TOOL_REGISTRY.md` lines 23–24, plus the skill-access matrix at 1007–1014 and the risk-class prose at 271
- `STATUS.md` lines 69–70 and 101–102
- `README.md` line 83
- `docs/API.md` line 541
- `AGENTS.md` — the "eleven-tool" wording in the architecture sections

The seven skills are the directories under `chain_server/skills/shopper/`. Note that `docs/SHOPPER_AGENT_SKILL_REGISTRY.md` already lists all seven correctly — use it as the reference, but fix its two smaller errors: the `product-discovery` row omits `describe_catalog_tool`, and the `catalog-questions` row omits `exclusive_group`.

**Verify:** count the list in the source and compare to each document.
```bash
sed -n '2908,2921p' chain_server/src/deepagents_runtime.py
ls chain_server/skills/shopper/
```

### Step 0.4 — Refresh `STATUS.md`

**Why:** dated 2026-08-02, roughly seventy commits behind, and its headline test number is wrong. *(Audit: M2)*

- Update the date.
- Replace "953 passed with 1 xfailed" with the current **1,485 passed, 13 skipped, 1 xfailed**.
- Correct `langgraph==1.2.7` to `1.2.11` (see `chain_server/requirements.txt:27`).
- Summarise the work landed since 2026-08-02, including weather (#227), the catalog-indexer service (#208), horizontal-scaling readiness (#207), and per-turn token usage (#222).

Consider adding a short note at the top recording when it was last reconciled, so future staleness is visible at a glance.

### Step 0.5 — Mark completed planning documents as historical

**Why:** these read as descriptions of current, unfinished work. `DEEP_AGENTS_MIGRATION_PLAN.md:18-22` claims legacy graph files still exist; they were deleted. *(Audit: Minor)*

Add a short banner at the top of `docs/DEEP_AGENTS_MIGRATION_PLAN.md`, `docs/CATALOG_REFACTOR_PLAN.md`, and `docs/DEEP_AGENTS_CART_TOOL_GOAL.md` stating that the work is complete, when it completed, and pointing to `docs/SHOPPER_AGENT_ARCHITECTURE.md` as the current reference. Do not delete them — the history is valuable.

Also update `.cursor/rules/codebase.mdc`, which still references the deleted `graph.py` and `planner.py`.

### Step 0.6 — Fix the two incorrect statements in `CONTRIBUTING.md`

**Why:** line 15 says "there is no CI/CD process in place yet" while three workflows exist. *(Audit: M7)*

Replace it with a description of what CI actually runs: unit tests, changed-file lint, the AGENTS config-reference check, and Docker image builds. Leave the DCO requirement as-is — Step 2.6 makes it true rather than removing it.

### Step 0.7 — Correct the Prometheus example port

`docs/DEPLOYMENT.md:822` scrapes chain-server on `8000`; the service runs on `8009`. *(Audit: Minor)*

---

**Phase 0 exit check:** documentation describes the system that exists. No code changed, so the suite must still be at baseline.

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py unit
git diff --check
```

---

## Phase 1 — The release blockers

These four items are what stand between the current state and a defensible blueprint release.

### Step 1.1 — Build the UI as a production image

**Why:** the shipped image runs the Create React App development server. *(Audit: B1)*

**Current state** — `ui/Dockerfile:34-36`:
```
RUN npm install
CMD ["npm", "run", "start"]
```

**Target:** a multi-stage build that produces static assets and serves them from a real web server.

1. **Build stage** — use a pinned Node image, copy `package.json` and `package-lock.json` first for layer caching, then run `npm ci` (not `npm install`, so the committed lockfile is actually honoured), then `npm run build`. The `build` script already exists at `ui/package.json:28`.
2. **Serve stage** — copy `build/` into nginx and serve statically. Set `NODE_ENV=production`.
3. **Preserve the image symlink.** The current Dockerfile does `ln -s /app/shared/images /app/public/images` at line 32. CRA resolves `public/` at build time, so this must happen **before** `npm run build` in the build stage, or the product images will 404. This is the one genuine trap in this step.
4. Add a non-root `USER` and a `HEALTHCHECK`.

**Verify:**
```bash
docker compose build frontend
docker compose up -d
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:3000        # expect 200
```
Then in a browser: load the UI, send a message, confirm the SSE stream renders and product images display. Confirm the served HTML/JS is minified.

### Step 1.2 — Close the default network exposure

**Why:** internal services and an object store with default credentials are published on all interfaces. *(Audit: B2)*

In `docker-compose.yaml`, prefix the host side of these port mappings with `127.0.0.1:`, following the pattern already used correctly for memory-retriever at line 167:

| Service | Lines | Ports |
|---|---|---|
| chain-server | 15–16 | 8009 |
| catalog-retriever | 103–104 | 8010 |
| rails | 200–201 | 8012 |
| MinIO | 256–257 | 9000, 9001 |
| Milvus | 280–281 | 19530, 9091 |
| Phoenix | 343–347 | 6006 |

Leave nginx on `3000` publicly bound — that is the intended entrypoint.

Phoenix matters more than it looks: its traces contain conversation content, so it deserves the same treatment the team already gave the memory service.

Replace the MinIO defaults at lines 260–261 and 286–287 with values sourced from the environment, and document the new variables in `.env.example`.

**Verify:**
```bash
docker compose up -d
ss -ltnp | grep -E '8009|8010|8012|9000|19530|6006'    # expect 127.0.0.1, not 0.0.0.0
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:3000        # still 200
```
Confirm the app still works end to end — the services talk over the internal Compose network, so loopback host binding should not affect them.

### Step 1.3 — Write the release notes

**Why:** the changelog describes v1.0.0 from September 2025 and an architecture that no longer exists, across 195 unreleased commits. *(Audit: B4)*

1. Choose a version number. The architecture was replaced, so **2.0.0** is the honest choice.
2. Add an entry to `CHANGELOG.md` covering at minimum:
   - **Breaking:** the LangGraph multi-agent pipeline was removed; `deepagents_runtime.py` is the only serving path
   - Deep Agents SDK harness with a skill/tool registry and an immutable tool policy
   - Durable conversation turns, cart, and product references in the memory service
   - Representative shopper profiles
   - Schema-driven catalog with capability-derived search
   - Optional destination-weather capability, disabled by default
   - The new `catalog-indexer` service
   - Horizontal-scaling readiness and Postgres support for the memory service
3. Generate the PR list mechanically so nothing is missed:
   ```bash
   git log v1.0.0..HEAD --format='- %s' --reverse > /tmp/since_v1.md
   ```
4. Tag the release once the cut is agreed.

### Step 1.4 — Align the container Python version with CI

**Why:** services run 3.11 while tests and lint target 3.12, so you validate on a different interpreter than you ship. *(Audit: M5)*

Change `python:3.11-slim` to a **digest-pinned** 3.12 image in `catalog_retriever/Dockerfile:4`, `chain_server/Dockerfile:4`, `guardrails/Dockerfile:5`, and `memory_retriever/Dockerfile:5`.

Digest-pin rather than using the floating tag, so rebuilds are reproducible.

**Verify:**
```bash
docker compose build
docker compose run --rm chain-server python -V     # expect 3.12.x
python skills/retail-test-runner/scripts/run_retail_tests.py unit
```
Watch for dependency wheels that resolve differently on 3.12. If anything breaks, it is better to find it now than after release.

---

**Phase 1 exit check — the release gate:**

- [ ] UI image serves a production build, and product images load
- [ ] Only port 3000 is publicly bound; no default credentials remain
- [ ] `CHANGELOG.md` describes this release; a version tag exists
- [ ] All services run Python 3.12 and the suite is green on it
- [ ] Unit suite at or above baseline (1,485 passed)
- [ ] `git diff --check` clean and `docker compose config` valid

---

## Phase 2 — Security, correctness, and process

Strongly recommended before a public release. Anything you choose not to do here should be written down as a known limitation rather than left silent.

### Step 2.1 — Fence shopper text and decide the guardrails default

**Why:** the fencing mechanism exists and is applied to vision output, but the shopper's own query and conversation history reach the instruction channel unfenced — and guardrails are off by default. *(Audit: M3)*

Apply the existing `fencing.py` treatment to the user query and the `RECENT DISCUSSION` block in `deepagents_runtime.py:3742-3755` and `:3774`, mirroring the `MEDIA_FENCE` usage at `:3751-3752`.

Then make a deliberate decision about `guardrails_enabled` defaulting to `False` (`config.py:202-208`). Either default it on, or document prominently in the README that the default configuration has no input filtering.

**Careful:** this changes prompt bytes, so it can move answer quality. Follow the project's own rule — run the judged evaluation more than once and compare the intersection, never a single run.

**Verify:** extend `tests/unit/chain_server/test_untrusted_text_is_fenced.py` to cover the query and dialogue lanes.

### Step 2.2 — Redact logging and stop returning raw exceptions

**Why:** full shopper queries, complete response state, cart contents, and diagnostics are logged by default; clients receive `str(e)`. *(Audit: M4)*

- `chain_server/src/main.py:202` — do not log the raw query; log a correlation ID and metadata.
- `chain_server/src/main.py:251,277` — never log the whole `out_state_dict` at INFO.
- `catalog_retriever/src/main.py:209` — stop logging the full request object.
- `main.py:231-233,239-241,298-300` — return a stable error code to the client, log detail server-side only.
- `catalog_retriever/src/main.py:36` — drop the startup directory listing.

### Step 2.3 — Enforce the advertised video duration limit

**Why:** `/capabilities` advertises `max_video_duration_seconds`, and `_validate_media` never checks it. *(Audit: M6)*

Either enforce duration in `main.py:619-691`, or remove the field from the advertised capabilities until it is enforced. Do not leave an API promising a limit it does not apply.

### Step 2.4 — Make compose wait for readiness, and restart on failure

**Why:** `depends_on` has no `condition: service_healthy`, so traffic can arrive before schema migration completes. *(Audit: M10)*

- Add healthchecks and `condition: service_healthy` for memory (`/ready`) and catalog (`/ready`) at `docker-compose.yaml:43-45`.
- Add `restart: unless-stopped` to long-running application services.
- Remove the unused `MILVUS_*` environment and the Milvus `depends_on` from memory-retriever (lines 169–170, 180–182) — that service never uses Milvus.

**Verify:** `docker compose down -v && docker compose up -d`, then confirm chain-server does not log connection errors during startup.

### Step 2.5 — Document or fix the process-lifetime caches

**Why:** a catalog redeploy without a chain-server restart silently produces stale validation behaviour, and `_SCHEMA_CACHE` is unlocked. *(Audit: M9)*

- Add a lock around `_SCHEMA_CACHE` (`turn_support.py:1242-1258`) matching the pattern already used in `CatalogCapabilitiesClient`.
- Document the catalog-capability cache (`catalog_capabilities.py:43-69`) in the `AGENTS.md` gotchas section alongside the existing checkpoint note, and state the operational rule: **after a catalog change, restart the chain server**.
- Consider a short negative-cache or circuit breaker for the catalog-unavailable path, which currently retries every turn.

### Step 2.6 — Enforce DCO in CI

**Why:** `CONTRIBUTING.md` says unsigned commits will not be accepted; none of the last 50 are signed and nothing checks. *(Audit: M7)*

Add a DCO check to the pull-request workflow. Decide explicitly what to do about the existing unsigned history — for a repository that reproduces the full Developer Certificate of Origin, this is worth a deliberate answer rather than a silent one.

### Step 2.7 — Complete `docs/API.md`

**Why:** live endpoints are undocumented and the file contradicts itself. *(Audit: M8)*

- Document `GET /ready` on all three services.
- Document `GET /cart` and `PATCH /cart/lines/{cart_line_id}` (`main.py:436-504`) — the UI uses them.
- Add the guardrails service section: `/rail/input/check`, `/rail/input/timing`, `/rail/output/check`, `/rail/output/timing`.
- Correct the `GET /health` shape at lines 1258–1274 to the actual `{status, timestamp, version}`.
- Resolve the guardrails-default contradiction between line 142 and lines 1603–1604 in favour of the code, which is `false`.
- Decide whether the legacy `/user/{user_id}/…` memory routes are supported or deprecated, and say so.

### Step 2.8 — Harden the container images

*(Audit: Minor)*

Add a non-root `USER` and a `HEALTHCHECK` to each service Dockerfile. Clean apt lists in the same layer in `guardrails/Dockerfile:12-13`. Pin the `:latest` tags in `docker-compose-nim-local.yaml:99,122` and `monitoring/docker-compose.yaml:21,40`. Add resource limits to application services.

### Step 2.9 — Fix the broken CI cleanup step

`ci.yml` runs `cd retail-shopping-assistant`, but checkout places the repo at the workspace root. The `cd` fails, `|| true` masks it, and teardown never runs — leaking containers and images on the runner. Remove the `cd`. *(Audit: Minor)*

### Step 2.10 — Restrict CORS

`chain_server/src/main.py:108-114` sets `allow_origins=["*"]` with `allow_credentials=True`. Restrict origins for production profiles. *(Audit: Minor)*

---

**Phase 2 exit check:**

- [ ] Untrusted text lanes fenced; guardrails default decided and documented
- [ ] No shopper PII in logs; no raw exceptions returned to clients
- [ ] Advertised media limits match enforced limits
- [ ] Compose waits on readiness; restart policies set
- [ ] Caches locked and documented
- [ ] DCO enforced in CI
- [ ] `docs/API.md` complete and self-consistent
- [ ] Unit suite green; judged evaluation shows no regression across repeated runs

---

## Phase 3 — Structural health

Not release-gating. This is the work that keeps the project maintainable, and it is deliberately last because it is the highest-churn and lowest-urgency.

### Step 3.1 — Pay down lint and formatting debt in one dedicated pass

**Why:** 516 findings and 134 unformatted files mean the repo-wide gate can never be switched on, and CI is stuck scoping to changed files. *(Audit: M13)*

Do this as **its own pull request touching nothing else**, because it produces a very large diff. The CI comments already explain exactly why it has been deferred — this step is what retires that explanation.

1. Apply the mechanical fixes:
   ```bash
   ruff check . --fix          # 439 of 516 are auto-fixable
   ruff format .
   ```
2. Review the remainder by hand — these are real smells, not style: `B008` (18), `E402` (13), `F401` (10), `B904` (4), `B905` (4), `SIM115` (2), `F841` (1).
3. Confirm the suite is still green.
4. **Then** switch CI to repo-wide enforcement: change the lint job in `.github/workflows/python-unit-tests.yml` from changed-file scoping to `ruff check .` and `ruff format --check .`, and delete the two explanatory comments, which will no longer be true.

### Step 3.2 — Split `_create_agent`

**Why:** a 1,537-line function (`deepagents_runtime.py:1547-3083`) rebuilt every turn, containing every tool as an inline closure. It is the main reason business logic cannot be unit-tested without the full runtime. *(Audit: M12)*

The groundwork is already done — `TurnScope` was extracted precisely so tools could move independently.

Approach: extract each tool implementation into a module-level function taking `TurnScope` plus its clients, leaving the `@tool` decorator as a thin wrapper. Suggested targets: `runtime_tools/catalog.py`, `runtime_tools/cart.py`, `runtime_tools/conversation_products.py`.

Do it **one tool at a time**, proving byte-identical behaviour at each step. The team has done this kind of mechanical extraction before and verified it by normalising and diffing function bodies — reuse that technique.

### Step 3.3 — Split `turn_support.py`

**Why:** 5,368 lines spanning at least a dozen responsibilities. *(Audit: M12)*

The file header already documents the cyclic-import problem between its themes. Lowest-risk first cut: pull a thin `turn_support_types.py` for shared models, then separate the response/evidence region (roughly lines 2422–4224) from the scope-reasoning region (roughly 245–911).

Other natural seams: static search schemas (914–1243), dynamic schema building (1245–2036), agent diagnostics (2255–2420), fail-closed responses (2422–2614), and cart reference resolution (4660–5270).

### Step 3.4 — Add a type checker

No `mypy.ini` or `pyrightconfig.json` exists. Introduce one on `chain_server/src` with gradual strictness, then add it to CI. Start with the boundaries: `config: Any` on the runtime constructor (`deepagents_runtime.py:1104`) should be `ChainServerConfig`. *(Audit: M13)*

### Step 3.5 — Run the Postgres suite in CI

`test_the_schema_works_on_postgres.py` is a real suite that skips without `MEMORY_TEST_POSTGRES_URL`, so Postgres regressions are invisible on every PR. Add a CI job with a Postgres service container. Then address the concurrent-migration race (`migrations.py`, `main.py:166-168`) and the SQLite-specific `INSERT OR IGNORE` at `migrations.py:133-138`. *(Audit: M11)*

### Step 3.6 — Strengthen the UI

Split `chatbox.tsx` (1,045 lines) by concern. Make the SSE parser buffer across chunks and add an `AbortController` on unmount (`useChat.ts:165-218`). Add tests for SSE error paths and session reset — three test files is thin for this surface. *(Audit: Minor)*

### Step 3.7 — Remove duplication and dead code

Centralise `_REFERENCE_WRAPPERS` and the `_NON_ATTRIBUTE_*` frozensets; replace the hard-coded `"STOP_TOOL_USE: …"` literal at `turn_support.py:2682` with the shared prefix constant. Resolve `weather_tool.py`, which is now a dead second implementation of the inline tool at `deepagents_runtime.py:2683` — either make it the single registration path or delete it. *(Audit: Minor)*

### Step 3.8 — Raise coverage on the weak spots

`vocabulary_judge.py` sits at 54% against an 89% project average. The catalog retriever and memory service are also thin relative to their complexity — 7 and 10 test files against `retriever.py` at 1,472 lines and `conversations.py` at 875.

### Step 3.9 — Repository presentation

Add a pull-request template. Add a short README section distinguishing what ships (services, `shared/`, compose) from development tooling (`skills/`, `.agents/`, `.cursor/`, `benchmarks/`, `STATUS.md`), so external consumers are not confused. Change `python` to `python3` in copy-paste commands at `AGENTS.md:148-150` and `187-189`.

---

## Quick reference: findings to steps

| Finding | Severity | Step |
|---|---|---|
| B1 UI dev server in image | Blocker | 1.1 |
| B2 Open ports + default credentials | Blocker | 1.2 |
| B3 `AGENTS.md` weather inverted | Blocker | 0.1, 0.2 |
| B4 Changelog frozen at v1.0.0 | Blocker | 1.3 |
| M1 Tool/skill counts wrong | Major | 0.3 |
| M2 `STATUS.md` stale | Major | 0.4 |
| M3 Unfenced shopper text | Major | 2.1 |
| M4 PII in logs | Major | 2.2 |
| M5 Python 3.11 vs 3.12 | Major | 1.4 |
| M6 Video duration unenforced | Major | 2.3 |
| M7 DCO unenforced | Major | 0.6, 2.6 |
| M8 `docs/API.md` gaps | Major | 2.7 |
| M9 Process-lifetime caches | Major | 2.5 |
| M10 Compose readiness | Major | 2.4 |
| M11 Migration race | Major | 3.5 |
| M12 Oversized modules | Major | 3.2, 3.3 |
| M13 Lint debt, no type checker | Major | 3.1, 3.4 |
| Obsolete planning docs | Minor | 0.5 |
| Prometheus port | Minor | 0.7 |
| Dockerfile hardening | Minor | 2.8 |
| CI cleanup bug | Minor | 2.9 |
| Open CORS | Minor | 2.10 |
| UI size and SSE parser | Minor | 3.6 |
| Duplication, dead code | Minor | 3.7 |
| Coverage gaps | Minor | 3.8 |
| Repo presentation | Minor | 3.9 |
