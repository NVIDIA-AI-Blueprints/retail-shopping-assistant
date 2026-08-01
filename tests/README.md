# Tests

This directory hosts the Retail Shopping Assistant test assets, split into
two independent suites.

## Layout

```
tests/
├── conftest.py             # Shared fixtures for all unit tests
├── pytest.ini              # Pytest configuration (asyncio, paths, markers)
├── requirements-dev.txt    # Dependencies needed to run the unit suite
├── requirements.txt        # Dependencies for the integration scripts
├── unit/                   # Offline, hermetic unit tests (see below)
│   ├── chain_server/
│   ├── catalog_retriever/
│   ├── memory_retriever/
│   └── guardrails/
├── integration/            # End-to-end scripts driving live services
│   ├── conversations/
│   │   └── shopping/
│   │       └── conv_*.yaml # Committed shopping golden conversations
│   ├── conversation_collector.py
│   ├── response_quality.py
│   ├── time_breakdown.py
│   └── quality_plots.py
└── evaluation/             # Challenger/Judge evaluation workflows
    ├── PLAN.md
    ├── eval_config.yaml
    ├── judge_rules.md
    └── datasets/
```

## Unit tests

The `unit/` tree mirrors the production service layout. Every service module
has at least one matching `test_*.py` file. All tests run fully offline: no
Docker, no network, no LLM/Milvus/nemoguardrails services required. External
dependencies (OpenAI clients, HTTP calls, Milvus, LangGraph streaming, the
SQLite file database used by `memory_retriever`) are stubbed inside each
test or via fixtures in `conftest.py`.

### Running the unit suite

The repository runner selects the local development/test environment and sets
host-side shared paths:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py unit
```

Pass focused pytest arguments after `--pytest-args`:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py unit \
  --pytest-args -q unit/catalog_retriever
```

### Writing new unit tests

Fixtures available from `conftest.py`:

- `base_config` / `valid_config_dict`: representative chain-server configs.
- `fake_response_cls`: a tiny `requests.Response` stand-in for HTTP stubs.
- `make_openai_chat_response`: factory for fake OpenAI chat responses.
- `stream_writer_capture`: intercepts `langgraph.config.get_stream_writer`
  so assertions can inspect streamed payloads.

Guidelines:

- Keep tests hermetic. Mock every external service; never call a live API.
- Prefer behavioural assertions (what the agent does/emits) over internal
  implementation details.
- Name test files `test_<module>.py` and test classes `Test<Feature>`.
- When adding a new service module, add a matching `__init__.py` to
  `<service>/src/` if one does not already exist so the module is
  importable as a package.

## Integration scripts

The `integration/` folder contains scripts that exercise the full system
via its public HTTP endpoints. They assume all services are running
(typically via Docker Compose or the local runner). The committed source of
truth is `integration/conversations/shopping/conv_*.yaml`.

Run endpoint conversations and timing without the paid Judge stage:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path shopping \
  --skip-quality
```

The files under `integration/conversations/<TEST_PATH>/conv_*.yaml` are the
committed golden reference for integration quality checks. Each file contains
fixed queries plus expected answers. Generated response output, judge output,
plots, timing summaries, and comparisons are ignored artifacts and should not
be committed unless that is an explicit project decision.

Endpoint results are written to the ignored directory
`integration/conversations/<TEST_PATH>/results/`. LLM-as-judge
outputs are written to the ignored directory
`integration/conversations/<TEST_PATH>/quality/results/`. Archive completed
shopping runs and their quality/timing comparisons outside the repository under
`~/exec-briefs/retail-shopping-assistant/quality/shopping/`; the canonical
archive stores judged output under `judge/`. In-repo generated folders are
scratch output, not the reported quality artifact.

The focused `product_comparison` dataset proves the prior-product path. Two
shortlist turns establish one exact product apiece; its committed diagnostic
expectations are checked before any Judge call. The comparison turn must use
one batched historical resolver call, two product-detail reads, and no catalog
search. Run it with `--test-path product_comparison`; each collector run
generates a fresh conversation identity.
The Judge receives the generated prior turns plus a bounded, ref-free projection
of current-turn structured catalog evidence; product facts present in that
projection are authoritative for scoring.

The judge stage requires explicit environment configuration:

```bash
export JUDGE_BASE_URL="<openai-compatible-base-url>"
export JUDGE_MODEL="<judge-model-name>"
export JUDGE_API_KEY_ENV="JUDGE_API_KEY"
export JUDGE_API_KEY="<judge-api-key>"
```

These are *not* part of the unit suite and are not collected by `pytest`
by default. They are intended for periodic quality/performance evaluation
against a deployed stack.

## Evaluation scaffold

The `evaluation/` folder contains the Challenger/Judge evaluation workflows. It captures
the planned directory shape, model configuration references, judge rules,
text/image/style-guide shopping scenario briefs, generated image-shopping
assets, runnable Challenger/Judge helpers, and ignored generated-results
location.

Start with:

- `evaluation/PLAN.md`: design and first implementation slice.
- `evaluation/eval_config.yaml`: non-secret environment variable references
  for Challenger and Judge models.
- `evaluation/judge_rules.md`: optional qualitative scoring rubric.
- `evaluation/datasets/text_shopping/scenarios.yaml`: text-only scenario
  briefs with shopper behavior and language cues.
- `evaluation/datasets/image_shopping/scenarios.yaml`: image-driven scenario
  briefs that reference asset sidecar ids.
- `evaluation/datasets/image_shopping/assets/`: generated product images and
  YAML sidecars for image-shopping evaluation.
- `evaluation/datasets/style_guide/scenarios.yaml`: styling-skill behavior
  scenarios covering anchor product, no-anchor discovery, cart styling,
  mid-browse styling, budget/style blending, comparison, wardrobe gap, and
  refinement behavior.
- `evaluation/datasets/style_guide/README.md`: catalog-coupling levels and
  refresh steps for deployments with a changed catalog.
- `evaluation/src/challenger.py`: scenario-driven live conversation runner.
- `evaluation/src/judge.py`: optional saved-run Judge phase.
- `evaluation/src/report.py`: simple HTML/text report generation.

Typical validation and run commands from the repo root. Source
`tests/evaluation/.env` first for live Challenger/Judge model calls:

```bash
PYTHONPATH=tests/evaluation python -m src.challenger --dry-run
PYTHONPATH=tests/evaluation python -m src.challenger
PYTHONPATH=tests/evaluation python -m src.challenger --scenario-id text_budget_work_bag
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios --dataset style_guide
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios --dataset text_shopping --dataset image_shopping --dataset style_guide
PYTHONPATH=tests/evaluation python -m src.judge --latest --enable-judge
PYTHONPATH=tests/evaluation python -m src.judge tests/evaluation/results/runs/<run_id>/run.yaml
```

Generated outputs should go under `evaluation/results/` and are ignored except
for `evaluation/results/.gitkeep`.
