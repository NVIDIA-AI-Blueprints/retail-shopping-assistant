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
│   ├── conversation_collector.py
│   ├── output_collector.py
│   ├── response_quality.py
│   ├── time_breakdown.py
│   ├── quality_plots.py
│   └── run_tests.sh
├── evaluation/             # Challenger/Judge evaluation workflows
│   ├── PLAN.md
│   ├── eval_config.yaml
│   ├── judge_rules.md
│   └── datasets/
└── examples/               # YAML conversation scenarios consumed by integration scripts
```

## Unit tests

The `unit/` tree mirrors the production service layout. Every service module
has at least one matching `test_*.py` file. All tests run fully offline: no
Docker, no network, no LLM/Milvus/nemoguardrails services required. External
dependencies (OpenAI clients, HTTP calls, Milvus, LangGraph streaming, the
SQLite file database used by `memory_retriever`) are stubbed inside each
test or via fixtures in `conftest.py`.

### Running the unit suite

Install the development dependencies into a virtual environment, then run
`pytest` from the repo root:

```bash
python3 -m venv .venv-tests
source .venv-tests/bin/activate
pip install -r tests/requirements-dev.txt

cd tests
pytest -q
```

Common invocations:

```bash
pytest unit/chain_server                 # single service
pytest unit/chain_server/test_cart.py    # single file
pytest -k "test_add_to_cart"             # keyword match
pytest --cov=chain_server --cov=catalog_retriever --cov=memory_retriever --cov=guardrails
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
(typically via `docker compose up`) and are driven by YAML scenario files
under `examples/`.

Typical flow:

```bash
export TEST_PATH="2025_08_16"
bash integration/run_tests.sh
```

The files under `integration/conversations/<TEST_PATH>/conv_*.yaml` are the
committed golden reference for integration quality checks. Each file contains
fixed queries plus expected answers. Generated response output, judge output,
plots, timing summaries, and comparisons are ignored artifacts and should not
be committed unless that is an explicit project decision.

For commit-scoped quality and timing tracking, run the deterministic retail
test runner with an explicit result directory. To capture a previous-commit
baseline, run from that commit or a clean checkout and use a stable label such
as the short SHA:

```bash
BASELINE_SHA="$(git rev-parse --short HEAD)"
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path shopping \
  --result-directory "$BASELINE_SHA" \
  --archive-label "$BASELINE_SHA"
```

To capture the latest working-tree run and generate a local comparison:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path shopping \
  --result-directory results \
  --archive-label latest \
  --compare-with "$BASELINE_SHA"
```

Endpoint results are written to the ignored directory
`integration/conversations/<TEST_PATH>/<RESULT_DIRECTORY>/`. LLM-as-judge
outputs are written to the ignored directory
`integration/conversations/<TEST_PATH>/quality/<RESULT_DIRECTORY>/`.

The runner also archives each integration run under the ignored local archive
root `integration/conversations/<TEST_PATH>/quality_progress/`. Set
`RETAIL_TEST_ARCHIVE_ROOT` or pass `--archive-root` to choose a different local
location. The archive contains the golden YAML snapshot, actual result YAML,
copied `timing_summary.png`, computed `timing_summary.json`, judge summaries
when quality runs, metadata, and a short `summary.md`.

For the default `results` directory, the runner automatically preserves the
existing `results/` output as `quality_progress/runs/previous/` before a new
integration run starts. After the run succeeds, the new output is archived as
`quality_progress/runs/latest/`, and a comparison report is written under
`quality_progress/comparisons/`. Quality progress is appended to
`quality_progress/progress.jsonl` and summarized in
`quality_progress/progress.md`.

Use `--no-local-archive` only when intentionally skipping the ignored local
quality/timing trail.

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
