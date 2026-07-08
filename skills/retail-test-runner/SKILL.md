---
name: retail-test-runner
description: Run the Retail Shopping Assistant test suites and evaluation workflows under tests/, including offline pytest unit tests, live integration scripts, and tests/evaluation Challenger/Judge runs with one-scenario, all-scenario, latest-run, and report-result workflows.
metadata:
  short-description: Run retail unit and integration tests
---

# Retail Test Runner

Use this skill when the user asks to run, validate, debug, or explain the Retail Shopping Assistant tests or evaluator workflows under this repository's `tests/` directory.

## Default Runner

Use the deterministic runner from the repo root:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py all --test-path shopping
```

Common invocations:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py unit
python skills/retail-test-runner/scripts/run_retail_tests.py integration --test-path shopping
python skills/retail-test-runner/scripts/run_retail_tests.py integration --test-path shopping --disable-guardrails --request-timeout 60
python skills/retail-test-runner/scripts/run_retail_tests.py integration --test-path rails --skip-quality
```

The runner:

- Loads repo-root `.env` before checking integration credentials.
- Uses `.local-run/dev-venv/bin/python` when it exists, then `.venv-tests/bin/python`, then the current Python.
- Runs unit tests from `tests/` so `tests/pytest.ini` is applied.
- Runs integration scripts from `tests/integration/` so their relative `conversations/<TEST_PATH>` paths resolve correctly.
- Targets the chain-server timing endpoint at `http://localhost:8009/query/timing` by default.
- Sets `TEST_PATH` for integration runs.
- Can send `guardrails=false` on live integration requests with `--disable-guardrails`.
- Bounds each live request with `--request-timeout <seconds>`.
- Uses `--result-directory <name>` consistently across collection, timing plots, and response-quality judging.

## Unit Tests

The unit suite lives under `tests/unit` and is offline. It should not require Docker, Milvus, NIMs, guardrails services, or network calls.

If Python dev dependencies are missing, install them into the repo-local dev venv:

```bash
python skills/retail-local-runner/scripts/local_runner.py install-dev
```

Targeted unit examples:

```bash
cd tests
../.local-run/dev-venv/bin/python -m pytest -q unit/chain_server
../.local-run/dev-venv/bin/python -m pytest -q unit/chain_server/test_cart.py
../.local-run/dev-venv/bin/python -m pytest -q -k test_add_to_cart
```

## Integration Tests

The integration suite lives under `tests/integration` and drives live HTTP endpoints. Before running it:

- Ensure the chain server is running and reachable at `http://localhost:8009`.
- Choose an existing scenario directory under `tests/integration/conversations/`, usually `shopping` or `rails`.
- For LLM-as-judge scoring, set explicit judge configuration in the repo-root
  `.env` or launching shell: `JUDGE_BASE_URL`, `JUDGE_MODEL`,
  `JUDGE_API_KEY_ENV`, and the API key variable named by `JUDGE_API_KEY_ENV`.
  Use `--skip-quality` to run endpoint integration without the judge stage.

The runner preflights the selected conversation directory and service URL. Use `--no-preflight` only when intentionally testing a nonstandard setup.

Integration outputs are written under:

- `tests/integration/conversations/<TEST_PATH>/<RESULT_DIRECTORY>/`
- `tests/integration/conversations/<TEST_PATH>/quality/<RESULT_DIRECTORY>/`

The committed scenario files under
`tests/integration/conversations/<TEST_PATH>/conv_*.yaml` are the golden
reference: each file contains fixed queries plus expected answers. Generated
run output is ignored and should not be committed unless explicitly requested.

By default, integration runs are also archived under the ignored local archive
root `tests/integration/conversations/<TEST_PATH>/quality_progress/`. Set
`RETAIL_TEST_ARCHIVE_ROOT` or pass `--archive-root` to choose a different local
location. The archive contains:

- `golden/`: snapshot of the committed scenario YAML used for the run.
- `results/`: actual response YAML plus `timing_summary.png` when generated.
- `quality/`: judge YAML, `quality_summary.json`, `quality_summary.md`, and
  `response_quality.png` when the judge stage runs.
- `timing_summary.json`: computed timing averages from the result YAML.
- `metadata.json` and `summary.md`.

For the default `results` directory, the runner automatically preserves the
existing `results/` output as `quality_progress/runs/previous/` before a new
integration run starts. After the run succeeds, the new output is archived as
`quality_progress/runs/latest/`, and a comparison report is written under
`quality_progress/comparisons/`.

To capture a previous-commit baseline with an explicit label, run integration
from that commit or a clean checkout and use a stable label such as the short
SHA:

```bash
BASELINE_SHA="$(git rev-parse --short HEAD)"
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path shopping \
  --result-directory "$BASELINE_SHA" \
  --archive-label "$BASELINE_SHA"
```

To capture the latest working-tree run and compare it with that baseline:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py integration \
  --test-path shopping \
  --result-directory results \
  --archive-label latest \
  --compare-with "$BASELINE_SHA"
```

The comparison report is written under
`<archive-root>/comparisons/`. Timing is included for both the previous and
latest runs through `timing_summary.json` and copied `timing_summary.png` files.
Quality progress is appended to `<archive-root>/progress.jsonl` and summarized
in `<archive-root>/progress.md`. Use `--no-local-archive` only when you
intentionally do not want the ignored local quality/timing trail.

Do not delete stable baseline archives unless the user asks for a clean run.
The `latest` archive label is intentionally reusable and may be overwritten by
the newest working-tree run.

## Evaluation Challenger and Judge

For live evaluation under `tests/evaluation`, source the repo-root `.env`
without printing values. This is the canonical place for Challenger and Judge
model names:

```bash
set -a && source .env && set +a
export CHALLENGER_MODEL_API_KEY="${CHALLENGER_MODEL_API_KEY:-${NVIDIA_API_KEY:-}}"
export JUDGE_MODEL_API_KEY="${JUDGE_MODEL_API_KEY:-${NVIDIA_API_KEY:-}}"
```

Run one live Challenger scenario:

```bash
PYTHONPATH=tests/evaluation python -m src.challenger --scenario-id text_budget_work_bag
```

Run all selected evaluation scenarios:

```bash
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios
```

Judge the latest saved run while keeping `judge_model.enabled: false` in config:

```bash
PYTHONPATH=tests/evaluation python -m src.judge --latest --enable-judge
```

The Judge appends scores to `results/runs/<run_id>/run.yaml` and regenerates
`results/latest.html`.

## Validation

After editing this skill, validate both skill folders:

```bash
uv run --with pyyaml python "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/retail-test-runner
uv run --with pyyaml python "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/retail-test-runner
```
