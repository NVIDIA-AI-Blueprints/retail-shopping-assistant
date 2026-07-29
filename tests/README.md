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
│   │   ├── shopping/
│   │   │   └── conv_*.yaml # Full shopping golden conversations
│   │   ├── event_context/
│   │   │   └── conv_*.yaml # Broader profile/Guest event-context gate
│   │   ├── event_context_weather_guidance/
│   │   │   └── conv_*.yaml # Thin location→venue→date weather regression
│   │   └── event_context_comparison/
│   │       └── conv_*.yaml # Search→weather→prior-product comparison proof
│   ├── conversation_collector.py
│   ├── diagnostic_validation.py
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

The files under `integration/conversations/<TEST_PATH>/conv_*.yaml` are
committed scenario references for integration checks; the canonical shopping
golden remains `integration/conversations/shopping/conv_*.yaml`. Each file
contains fixed queries plus expected answers. A file may set one optional top-level
`shopper_profile_id`; the collector sends that server-owned ID on every turn in
the file, while omission or `null` remains Guest. The focused `event_context/`
set covers selected-profile location precedence, Guest isolation, shop-now
behavior, saved-location override, and non-event weather isolation without
running the full shopping cohort. The thinner
`event_context_weather_guidance/` set contains two three-turn regressions: NYC
plus an outdoor patio plus an exact relative weekday, and Cancun followed by an
explicit beach setting and then the date. Both assert one initial search, no
product/weather reads while collecting the one missing context field, and one
weather call with no repeated product read after the date.
The `event_context_comparison/` fixture follows one complete three-turn
occasion conversation: initial wedding candidates, live NYC patio weather, and
a later comparison of two prior products. Its deterministic expectations
require exactly search; then weather; then conversation-product resolution plus
two product-detail reads, with the expected grounded product names carried
through the responses. It also verifies the accepted event-question decision
on every turn and the categorical, redacted named-place weather trace.

Run only that targeted feature set with:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path event_context_weather_guidance \
  --skip-quality
```

Run only the focused event-to-comparison architecture proof with:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path event_context_comparison \
  --skip-quality
```

After collection, `diagnostic_validation.py` always checks the committed
skill, tool, sequence, evidence, weather, and stable-response expectations
before timing or optional Judge work. `--skip-quality` disables
the paid Judge stage and its derived quality plots; it does not disable this
deterministic live gate or timing. A diagnostic failure stops the run and
prevents archiving a false success. This is a structural architecture,
redaction, tool-sequence, and evidence proof rather than a semantic styling
quality score; use the saved transcript, manual review, or the optional Judge
for that separate assessment.

The collector records client elapsed time, application token and model-call
usage, model-usage summaries, and trusted diagnostics when the endpoint exposes
them. Generated response output, judge output, plots, timing summaries, and
comparisons are ignored artifacts and should not be committed unless that is
an explicit project decision.

Endpoint results are written to the ignored directory
`integration/conversations/<TEST_PATH>/results/`. LLM-as-judge
outputs are written to the ignored directory
`integration/conversations/<TEST_PATH>/quality/results/`. Archive completed
shopping runs and their quality/timing comparisons outside the repository under
`~/exec-briefs/retail-shopping-assistant/quality/shopping/`; the canonical
archive stores judged output under `judge/`. In-repo generated folders are
scratch output, not the reported quality artifact.

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
