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

These are *not* part of the unit suite and are not collected by `pytest`
by default. They are intended for periodic quality/performance evaluation
against a deployed stack.
