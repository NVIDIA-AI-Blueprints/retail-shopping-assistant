# Release Readiness Audit

**Repository:** NVIDIA AI Blueprint — Retail Shopping Assistant
**Branch audited:** `staging` at commit `1853f94` ("Stop substituting a near match for a garment the shop does not carry", #237)
**Date:** 17 September 2026
**Type:** Read-only audit. No application code, configuration, or documentation was changed in producing this report.

The companion document [`RELEASE_READINESS_PLAN.md`](RELEASE_READINESS_PLAN.md) turns every finding below into an ordered, step-by-step plan.

---

## 1. The short version

**This is a well-engineered codebase with a release-packaging and documentation problem.**

The engineering itself is in good shape. The full offline test suite passes, coverage across the four Python services is 89%, no secrets are committed, and the team has clearly invested in a disciplined "fail-closed" design where the assistant refuses to invent product facts. None of that is where the risk is.

The risk is in three places:

1. **The container image you would ship the UI in runs a development server.** This is the single clearest thing standing between the current state and a defensible release.
2. **The documentation actively contradicts the code** in ways that would mislead a new contributor into breaking working features. The weather section of `AGENTS.md` is the worst case: it instructs people to remove code that is now live and correct.
3. **The default deployment exposes internal services and default credentials** on all network interfaces.

None of these require re-architecting anything. They are finite, well-understood pieces of work. A realistic estimate is that the blocking set is a few days of focused effort, with a longer tail of structural cleanup that can safely happen after the release.

### Verdict by audience

| If you are releasing this as… | Verdict |
|---|---|
| A **developer blueprint / reference demo** (its stated purpose) | **Nearly ready.** Fix the blockers in Phase 1 of the plan, and be explicit in the README about what is demo-grade. |
| A **production-grade deployable** | **Not ready.** The whole plan applies, including the authentication and multi-replica items that are currently documented as open questions rather than solved. |

The repository describes itself as a blueprint, so the first column is the fair bar. This audit is written against that bar, and it flags separately the places where the docs currently *imply* the second.

---

## 2. What is genuinely good

It is worth being specific about this, because the finding list below is long and would otherwise give a misleading impression.

**The tests are real and they pass.** Running the project's own runner (`python skills/retail-test-runner/scripts/run_retail_tests.py unit`) produced **1,485 passed, 13 skipped, 1 expected failure, in 99 seconds**. The 13 skips are Postgres tests that correctly skip when no Postgres URL is configured, and the single expected failure is a deliberately documented open boundary, not a broken test.

**Coverage is high.** Measured across `chain_server`, `catalog_retriever`, `memory_retriever`, and `guardrails`: **89% of 10,071 statements**. Several critical modules are at or near complete coverage, including `fencing.py` (100%), `tool_evidence.py` (100%), `turn_scope.py` (100%), `models.py` (100%), `migrations.py` (99%), `weather.py` (96%), and `memory_retriever/src/main.py` (96%).

**No secrets are committed.** Only `.env.example` and `.env.local-llm.example` are tracked, and both contain shell-indirection placeholders rather than values. A scan of all tracked files for API-key patterns and private hostnames or IPs returned nothing. `.gitignore` correctly excludes `.env*`, `**/context.db`, evaluation artifacts, and build output.

**The safety design is thoughtful and deliberate.** The codebase separates what the model decides from what deterministic code enforces, and it prefers refusing to answer over inventing a product fact. Concrete examples: typed tool-loop control signals travel as LangChain artifacts rather than being recovered by parsing prose (`control_signals.py`), untrusted vision-model output is fenced before it reaches the agent's instruction channel (`fencing.py`), and a grounding-editor failure falls back to a fixed safe response rather than emitting an unverified draft (`deepagents_runtime.py:3317-3333`).

**Operational details have had real attention.** The memory service separates liveness (`/health`) from readiness (`/ready`, which gates on schema migration). The catalog indexer runs as a single-flight job so replicas cannot race each other building Milvus indexes. Connection-pool, threadpool, and uvicorn concurrency settings are aligned and documented. The memory service's host port is bound to loopback only.

**Dependency pinning is disciplined.** Every service `requirements.txt` uses exact `==` pins, `ui/package-lock.json` is committed, and specific pins carry documented reasons (`orjson==3.11.5` is held at the last release under the project's license policy).

**CI is honest about its own limits.** The lint job scopes Ruff to changed files and says why in a code comment: the repo carries hundreds of pre-existing findings, and failing every unrelated PR would be counterproductive. That is a defensible engineering decision, clearly recorded. Most repositories in this state have no comment at all.

---

## 3. Findings

Severity means:

- **Blocker** — should not ship in this state.
- **Major** — should be fixed for the release, or consciously accepted and written down.
- **Minor** — real, worth fixing, safe to defer.

### 3.1 Blockers

---

**B1. The UI container image runs the development server.**

`ui/Dockerfile:36` ends with `CMD ["npm", "run", "start"]`. That is the Create React App development server. The image never runs `npm run build`, so it ships unminified source with hot-reload machinery attached, and it serves through the dev server rather than a static file server. The same file uses `npm install` at line 34 rather than `npm ci`, which means the committed `package-lock.json` is not honoured and two builds of the same commit can resolve different dependency trees.

This is the most visible release defect in the repository. Anyone who pulls the blueprint and runs `docker compose up` gets a development server as their front end.

*Fix direction:* multi-stage build — `npm ci` then `npm run build`, then serve the static output from nginx or an equivalent. Set `NODE_ENV=production`.

---

**B2. The default compose stack publishes internal services on all interfaces, with default credentials.**

In `docker-compose.yaml`, the following are bound to `0.0.0.0` rather than loopback: chain-server (`8009`), catalog-retriever (`8010`), rails (`8012`), MinIO (`9000`, `9001`), Milvus (`19530`, `9091`), and Phoenix (`6006`). None of the application services implement authentication.

MinIO additionally ships with `MINIO_ROOT_USER: minioadmin` and `MINIO_ROOT_PASSWORD: minioadmin` (lines 260–261), repeated as `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` at lines 286–287. Anyone on the same network as a machine running the default stack has unauthenticated access to the object store, the vector database, the catalog API, and the Phoenix trace UI — and Phoenix traces contain conversation content.

The team clearly understands this class of problem, because the memory service *is* correctly bound to `127.0.0.1:8011` (line 167) and a previous commit removed a published etcd port (#216). The remaining services simply have not had the same treatment.

*Fix direction:* bind everything except the nginx entrypoint to loopback, or drop the host publish entirely and rely on the internal Compose network. Generate MinIO credentials rather than shipping defaults.

---

**B3. `AGENTS.md` instructs contributors to remove a working, shipped feature.**

`AGENTS.md:30` and `AGENTS.md:423-428` state that the weather tool is dormant and must be kept out of runtime registration, `SHOPPING_TOOL_POLICIES`, shopper-skill grants, and prompts.

All four of those statements are now false. As of #227 ("A forecast is a task of its own"):

| Claim in `AGENTS.md` | Reality |
|---|---|
| Not registered with Deep Agents | Registered at `deepagents_runtime.py:2927` |
| Absent from `SHOPPING_TOOL_POLICIES` | Present at `tool_policy.py:143-146` |
| Not granted by a skill | Granted by `chain_server/skills/shopper/destination-weather/SKILL.md` |
| Not mentioned in prompts | Injected at `deepagents_runtime.py:3002` and described at `:3039` |

One part of the claim remains true and should be preserved: the feature is **off by default**. `WEATHER_ENABLED` defaults to `false` in `docker-compose.yaml:35`, `.env.example:153`, and `shared/configs/chain_server/config.yaml:54`, and the tool is only appended to the registry when enabled.

This is a blocker because `AGENTS.md` is the instruction file for coding agents and new contributors working in this repo. Following it as written means deleting correct code. The same inverted claim also appears in `STATUS.md:31-41`, `docs/API.md:488-495`, and `docs/USER_GUIDE.md:78-82`.

---

**B4. The changelog does not describe the software being released.**

`CHANGELOG.md` contains exactly one entry: version 1.0.0, dated 3 September 2025. It describes "LangGraph-orchestrated agents", "Multi-agent Architecture", and Llama 3.1 70B — an architecture that has since been deleted and replaced by the Deep Agents harness.

**195 commits have landed since that tag**, and the only tags in the repository are `v1.0.0` and two dated `archive/` tags. There are no release notes for any of the work that constitutes this release: the Deep Agents migration, the durable memory service, the skill and tool registries, representative shoppers, the catalog schema refactor, horizontal-scaling work, or the weather feature.

For a public NVIDIA blueprint, the changelog is the first thing an evaluator reads to understand what changed. Right now it describes different software.

---

### 3.2 Major

---

**M1. Documented tool and skill counts are wrong throughout.**

The runtime registers **twelve** commerce tools. The list at `deepagents_runtime.py:2908-2921` contains `search_catalog_tool`, `get_product_details_tool`, `resolve_conversation_products_tool`, `get_cart_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool`, `view_cart_total_tool`, `get_store_policy_tool`, `check_product_availability_tool`, `describe_catalog_tool`, and `check_active_promotions_tool`. With the activation control tool that is thirteen bound tools, or fourteen when weather is enabled.

Documentation says eleven in `docs/SHOPPER_AGENT_TOOL_REGISTRY.md:23-24`, `STATUS.md:101-102`, `README.md:83`, `docs/API.md:541`, and `AGENTS.md`. The missing one is `describe_catalog_tool`.

Similarly, there are **seven** shopper skills on disk — `budget-shopping`, `cart-management`, `catalog-questions`, `destination-weather`, `outfit-styling`, `product-discovery`, `store-policy-answers` — while `AGENTS.md` and `STATUS.md:69-70` both say five. `docs/SHOPPER_AGENT_SKILL_REGISTRY.md` is the one document that correctly lists all seven.

---

**M2. `STATUS.md` is six weeks and roughly seventy commits stale.**

It is dated 2026-08-02. Its verification section claims "953 passed with 1 xfailed"; the actual current figure is **1,485 passed**. It pins `langgraph==1.2.7` while `chain_server/requirements.txt:27` specifies `1.2.11`. Its entire weather subsection is inverted per B3, and it describes five skills and eleven tools per M1.

A status document that is wrong is worse than no status document, because readers trust it.

---

**M3. Shopper text reaches the agent's instruction channel unfenced, and guardrails are off by default.**

The repository has a well-designed fencing mechanism in `fencing.py`, and it is applied to vision-model output at `deepagents_runtime.py:3751-3752`. But the shopper's own query and the `RECENT DISCUSSION` history block are assembled into the prompt without it (`deepagents_runtime.py:3742-3755`, `:3774`).

Meanwhile `guardrails_enabled` defaults to `False` (`config.py:202-208`). So in a default deployment there is neither input filtering nor prompt fencing on the untrusted lane most likely to carry an injection attempt.

The mitigation already exists in the codebase; it is simply not applied to this lane.

---

**M4. Shopper queries and full response state are written to logs.**

`chain_server/src/main.py:202` logs the complete shopper query. Lines 251 and 277 log the entire `out_state_dict`, which includes the assistant response, cart contents, timings, and `agent_diagnostics`. `catalog_retriever/src/main.py:209` logs the full `TextQueryRequest`.

For a shopping assistant, that is customer intent data and cart contents flowing into centralized logs by default. The same routes also return `str(e)` to clients on error (`main.py:231-233`, `239-241`, `298-300`), leaking internal exception text.

---

**M5. Container Python version does not match CI or the lint target.**

All four Python services build on `python:3.11-slim` (`catalog_retriever/Dockerfile:4`, `chain_server/Dockerfile:4`, `guardrails/Dockerfile:5`, `memory_retriever/Dockerfile:5`). CI runs tests and lint on Python **3.12** (`.github/workflows/python-unit-tests.yml`), and `ruff.toml:3` sets `target-version = "py312"`.

Tests therefore pass on a different interpreter than the one that runs in production. The tag is also floating rather than digest-pinned, so rebuilds are not reproducible.

---

**M6. An advertised media limit is not enforced.**

`config.py:45` defines `max_video_duration_seconds` and `main.py:348` advertises it on `/capabilities`. The validator at `main.py:619-691` checks file count, MIME type, and byte size — but never duration. The API promises a limit it does not apply.

---

**M7. The contribution process described in `CONTRIBUTING.md` is not followed or enforced.**

`CONTRIBUTING.md:21` states: "Any contribution which contains commits that are not Signed-Off will not be accepted." **Zero of the last 50 commits carry a `Signed-off-by` line**, and no CI job checks for one. `AGENTS.md:434` repeats the `git commit -s` requirement.

The same file at line 15 says "there is no CI/CD process in place yet". There are three GitHub Actions workflows.

For an Apache-2.0 NVIDIA repository that reproduces the full Developer Certificate of Origin text, an unenforced and unfollowed DCO is a legal-process gap, not just a tidiness issue.

---

**M8. `docs/API.md` has real gaps and one self-contradiction.**

At 1,613 lines it is the largest document in the repository, and it is missing live endpoints:

- `GET /ready` on chain, catalog, and memory — undocumented.
- `GET /cart` and `PATCH /cart/lines/{cart_line_id}` on chain-server (`main.py:436-504`), used by the UI — undocumented.
- The entire guardrails service surface (`/rail/input/check`, `/rail/input/timing`, `/rail/output/check`, `/rail/output/timing`) — undocumented.
- The documented aggregate `GET /health` shape with nested `services` (lines 1258–1274) is not what any service returns; all three return `{status, timestamp, version}`.

It also contradicts itself on the guardrails default: line 142 says "true by default", lines 1603–1604 say off, and the code default is `false`.

---

**M9. Process-lifetime caches are undocumented scaling hazards.**

`catalog_capabilities.py:43-69` caches the catalog capability contract for the entire process lifetime. If the catalog is redeployed without restarting the chain server, the agent validates against stale taxonomy and enum values, which changes fail-closed behaviour in a way that is hard to diagnose.

`turn_support.py:1242-1258` holds `_SCHEMA_CACHE` with **no lock**, unlike the locked pattern used elsewhere in the same codebase.

`AGENTS.md` carefully documents that graph checkpoints are process-local, but says nothing about either of these.

---

**M10. Compose does not wait for dependencies to become ready.**

`docker-compose.yaml:43-45` declares `depends_on` for memory, catalog, and rails without `condition: service_healthy`. The memory service exposes `/ready` precisely so that callers can wait for schema migration to finish — and nothing waits for it. Early traffic after `docker compose up` can hit a partially initialised stack.

Relatedly, no application service declares a `restart` policy, so a crashed service stays down.

---

**M11. Database migrations can race under Postgres.**

Every replica runs `run_schema_migrations` at startup (`memory_retriever/src/main.py:166-168`). The version-row insert is guarded (`migrations.py:313-318`), but two replicas can still enter `migrate(connection)` and run DDL concurrently before either commits. There are no down-migrations.

Additionally `migrations.py:133-138` uses SQLite-specific `INSERT OR IGNORE`, which would fail on a SQLite-to-Postgres upgrade carrying idempotency rows. Postgres support is genuine and opt-in — `scripts/copy_memory_to_postgres.py` exists and `test_the_schema_works_on_postgres.py` is a real suite — but it is **skipped in CI**, so Postgres regressions are invisible on every pull request.

---

**M12. Two modules are far past a maintainable size, and the largest function is rebuilt every turn.**

`turn_support.py` is **5,368 lines** and `deepagents_runtime.py` is **4,310**. Within the latter, `_create_agent` spans lines **1547–3083 — a single 1,537-line function** that defines every agent tool as an inline closure, and it runs on every turn.

The practical cost is testability: cart, search, availability, and weather business rules live inside nested closures, so exercising them requires standing up the full runtime. That is why `tests/unit/chain_server/` has 55 files, many of which are integration-shaped.

The team has already done the hard prerequisite work here — extracting per-turn state into `TurnScope` specifically so tools could be relocated independently — so the seam is prepared but not yet cut. Other long functions: `_reviewed_provenance` (482 lines), `_rendered_evidence` (279), `_add_cart_items_impl` (258), `_rewrite_response_for_grounding` (249), `_execute_turn` (238), `_validated_request` (226, nesting depth 5), `_start_conversation_turn` (nesting depth 7).

---

**M13. There is no repo-wide lint gate and no type checker.**

`ruff check .` reports **516 findings** (439 auto-fixable) and `ruff format --check` would reformat **134 of 174 files**. By area: `tests/` 208, `catalog_retriever/` 145, `chain_server/` 104, `memory_retriever/` 33, `benchmarks/` 14, `guardrails/` 10, `skills/` 8, `scripts/` 6, `shared/` 5.

The dominant rules are mechanical and safe to fix: `UP006` non-PEP-585 annotations (175), `I001` unsorted imports (98), `W293` whitespace on blank lines (46), `UP045` non-PEP-604 `Optional` (41), `UP035` deprecated imports (35).

A smaller set are genuine code smells worth reviewing by hand rather than auto-fixing: `B008` function call in default argument (18), `E402` imports not at top of file (13), `F401` unused imports (10), `B904` raise-without-from inside except (4), `B905` zip without explicit strict (4), `SIM115` open without context handler (2), `F841` unused variable (1).

There is no `mypy.ini` or `pyrightconfig.json` anywhere in the repository.

---

### 3.3 Minor

**Obsolete planning documents read as current architecture.** `docs/DEEP_AGENTS_MIGRATION_PLAN.md:18-22` says legacy graph files "still exist" — they were deleted. Lines 37–43 cite eleven tools and five skills. `docs/CATALOG_REFACTOR_PLAN.md` and `docs/DEEP_AGENTS_CART_TOOL_GOAL.md` describe shipped work as pending. These are useful history but dangerous onboarding material.

**`.cursor/rules/codebase.mdc` references deleted modules** (`graph.py`, `planner.py`), so coding agents get stale guidance from a second source.

**`ci.yml` has a broken cleanup step.** It runs `cd retail-shopping-assistant` in the cleanup block, but `actions/checkout` places the repository at the workspace root. The `cd` fails and the subsequent `docker compose down` calls are masked by `|| true`, so teardown silently does nothing.

**Dockerfiles run as root** and declare no `HEALTHCHECK`. `guardrails/Dockerfile:12-13` installs build tooling without clearing apt lists in the same layer.

**Unpinned `:latest` image tags** in `docker-compose-nim-local.yaml:99,122` and `monitoring/docker-compose.yaml:21,40`.

**The memory service is coupled to Milvus for no reason** — `docker-compose.yaml:169-170,180-182` passes `MILVUS_*` environment and a `depends_on: milvus: service_healthy`, but `memory_retriever/src` never uses Milvus. This slows startup and couples unrelated failure domains.

**No resource limits** on any application service in the main compose file.

**The UI is thin on tests and heavy in one file.** `ui/src/components/chatbox/chatbox.tsx` is 1,045 lines; there are three test files in the whole UI. The SSE parser (`useChat.ts:165-218`) assumes `data:` lines arrive whole and has no `AbortController` on unmount, so a split chunk can mis-parse and an unmounted component keeps reading.

**Duplicated constants** that violate the project's own centralization pattern: `_REFERENCE_WRAPPERS` in both `deepagents_runtime.py:606` and `conversation_products.py:361`; `_NON_ATTRIBUTE_SEARCH_KEYS` / `_NON_ATTRIBUTE_KEYS` in `turn_support.py:4446` and `conversation_products.py:396`; a hard-coded `"STOP_TOOL_USE: …"` literal at `turn_support.py:2682` beside correct `STOP_TOOL_USE_PREFIX` usage elsewhere.

**`weather_tool.py` is now dead.** It is documented as the dormant wrapper, but production registers an inline tool in `deepagents_runtime.py:2683`. Only unit tests import the wrapper, so there are two implementations that can diverge.

**`vocabulary_judge.py` is the coverage outlier** at 54%, well below the 89% project average.

**CORS is fully open.** `chain_server/src/main.py:108-114` sets `allow_origins=["*"]` together with `allow_credentials=True`.

**Prometheus example config points at the wrong port** — `docs/DEPLOYMENT.md:822` scrapes chain-server on `8000`; it runs on `8009`.

**No pull-request template**, though `CODEOWNERS` and a full set of issue templates exist.

**Copy-paste commands assume `python` is on PATH** (`AGENTS.md:148-150`, `187-189`); minimal Linux images provide only `python3`.

---

## 4. How this was verified

Everything in this report was checked against the working tree at `1853f94`. Specifically:

- The full offline unit suite was executed twice, once plain and once under coverage, using the repository's own test runner and dev virtualenv.
- Coverage was measured with `pytest-cov` across all four Python service source trees.
- `ruff check` and `ruff format --check` were run repo-wide and broken down by rule and directory.
- Tracked-file inventory, secret scanning, and `.gitignore` behaviour were checked with `git ls-files` and `git grep` so that only *shipped* files were assessed.
- Commit history, tags, and DCO sign-off rates were read from `git log`.
- Documentation claims were checked one by one against the cited source lines.

Three parallel reviews contributed detailed findings, which I spot-verified on the highest-stakes claims before including them here — the tool count, the UI Dockerfile, the Python base images, the MinIO credentials, and the weather registration path were each confirmed by direct reading rather than taken on report.

Two limits are worth stating plainly. **No live or judged evaluation run was performed**, so this audit says nothing about answer quality; `STATUS.md` and the evaluation archive remain the source for that. And **no container image was built and no stack was brought up**, so the deployment findings are from reading compose files, Dockerfiles, and nginx configuration rather than from observing a running system.
