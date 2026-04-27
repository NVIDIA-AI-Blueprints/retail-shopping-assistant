---
name: retail-test-runner
description: Run the Retail Shopping Assistant test suites under tests/, including offline pytest unit tests in tests/unit and live integration scripts in tests/integration for conversation, quality, timing, and plots.
metadata:
  short-description: Run retail unit and integration tests
---

# Retail Test Runner

Use this skill when the user asks to run, validate, debug, or explain the Retail Shopping Assistant tests under this repository's `tests/` directory.

## Default Runner

Use the deterministic runner from the repo root:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py all --test-path shopping
```

Common invocations:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py unit
python skills/retail-test-runner/scripts/run_retail_tests.py integration --test-path shopping
python skills/retail-test-runner/scripts/run_retail_tests.py integration --test-path rails --skip-quality
```

The runner:

- Loads repo-root `.env` before checking integration credentials.
- Uses `.local-run/dev-venv/bin/python` when it exists, then `.venv-tests/bin/python`, then the current Python.
- Runs unit tests from `tests/` so `tests/pytest.ini` is applied.
- Runs integration scripts from `tests/integration/` so their relative `conversations/<TEST_PATH>` paths resolve correctly.
- Targets the chain-server timing endpoint at `http://localhost:8009/query/timing` by default.
- Sets `TEST_PATH` for integration runs.

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
- Put `NVIDIA_API_KEY` in the repo-root `.env` or export it in the launching shell, unless using `--skip-quality`; `response_quality.py` requires it for LLM-as-judge scoring.

The runner preflights the selected conversation directory and service URL. Use `--no-preflight` only when intentionally testing a nonstandard setup.

Integration outputs are written under:

- `tests/integration/conversations/<TEST_PATH>/results/`
- `tests/integration/conversations/<TEST_PATH>/judge/`

Do not delete or overwrite these outputs unless the user asks for a clean run.

## Validation

After editing this skill, validate both skill folders:

```bash
uv run --with pyyaml python "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/retail-test-runner
uv run --with pyyaml python "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/retail-test-runner
```
