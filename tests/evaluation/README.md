# Evaluation

This folder contains the Challenger and Judge scaffold for the Retail Shopping
Assistant.

Current scope:

- `PLAN.md` is the design source for the scaffold.
- `eval_config.yaml` contains non-secret config references only.
- `datasets/` contains text/image shopping scenario briefs for Challenger runs.
- `datasets/image_shopping/assets/` contains generated product-photo inputs
  plus YAML sidecars that own image descriptions.
- `results/` is reserved for generated run output and is ignored except for
  `.gitkeep`.
- `src/` contains evaluation-only Challenger, Judge, and report helpers.

## How It Works

The Challenger loads scenarios from `datasets/`, uses the configured
`challenger_model` to generate one shopper turn at a time, sends each turn to
the live Shopping Assistant, then feeds the assistant response back into the
next Challenger prompt. Image scenarios send the asset image on the first turn
only; sidecar descriptions stay in evaluation metadata.

The Judge reads a saved `run.yaml`, applies `judge_rules.md` with the configured
`judge_model`, and appends scenario-level scores, reasons, criteria, and
critical failures back into the run record.

## Running

Start the target Shopping Assistant, then source the evaluation environment file:

```bash
set -a
source tests/evaluation/.env
set +a
```

That file should provide the Challenger model variables referenced by
`eval_config.yaml`:

```bash
export CHALLENGER_MODEL_BASE_URL="<openai-compatible-base-url>"
export CHALLENGER_MODEL_NAME="<challenger-model-name>"
export CHALLENGER_MODEL_API_KEY="<challenger-api-key>"
```

For a locally hosted Hugging Face, TGI, vLLM, or other OpenAI-compatible model,
the chat request still needs a `model` value. Use the server's served model id
or alias. If the local endpoint does not require auth, omit the API key:

```bash
export CHALLENGER_MODEL_BASE_URL="http://localhost:8000/v1"
export CHALLENGER_MODEL_NAME="<local-model-name>"
unset CHALLENGER_MODEL_API_KEY
```

You can also put local defaults directly in `eval_config.yaml` and leave the env
vars unset:

```yaml
challenger_model:
  base_url: "http://localhost:8000/v1"
  model: "<local-model-name-or-alias>"
  api_key_required: false
```

Env vars override `base_url` and `model` when both are set. For a cloud endpoint
where missing auth should fail before the run starts, set `api_key_required:
true`.

For reasoning-capable local chat templates, keep `disable_thinking: true` so the
Challenger and Judge request final answers without `<think>` output. Keep
`json_mode: true` so model responses are requested as JSON objects. Set either
flag to `false` only for endpoints that reject that OpenAI-compatible option.

## Challenger Commands

The dry run validates scenario selection and does not call a model or the target
Shopping Assistant:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --dry-run
```

The live Challenger uses `CHALLENGER_MODEL_*` and sends turns to
`target_agent.base_url` plus `target_agent.endpoint` from `eval_config.yaml`:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger
```

Run one specified live scenario by id:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --scenario-id text_budget_work_bag
```

Run every scenario in the configured dataset list, ignoring
`scenario_limit_per_dataset`:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios
```

Run every text and image scenario explicitly:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios --dataset text_shopping --dataset image_shopping
```

Useful selection flags:

- `--scenario-id <id>`: run one scenario id. Repeat the flag to run multiple
  named scenarios.
- `--dataset <name>`: select a dataset such as `text_shopping` or
  `image_shopping`. Repeat the flag to include multiple datasets.
- `--all-scenarios`: ignore `scenario_limit_per_dataset` and run every selected
  scenario.
- `--scenario-limit <n>`: override `scenario_limit_per_dataset` for this run.
- `--judge`: run the Judge after the Challenger finishes when the Judge model is
  enabled in config.

Set `target_agent.guardrails` in `eval_config.yaml` to choose whether eval
requests run with guardrails:

```yaml
target_agent:
  guardrails: true   # use false to disable guardrails for eval requests
```

The Challenger generates shopper turns with `challenger_model`, sends each turn
to `target_agent.endpoint`, sends image assets only on the first image-scenario
turn, and writes `results/runs/<run_id>/run.yaml`.

## Judge Commands

To run the optional Judge, set:

```bash
export JUDGE_MODEL_BASE_URL="<openai-compatible-base-url>"
export JUDGE_MODEL_NAME="<judge-model-name>"
export JUDGE_MODEL_API_KEY="<judge-api-key>"
```

For a no-auth local Judge model, set `JUDGE_MODEL_BASE_URL` and
`JUDGE_MODEL_NAME`, then leave `JUDGE_MODEL_API_KEY` unset.

Judge the latest saved run:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.judge --latest --enable-judge
```

Judge a specific run:

```bash
set -a && source tests/evaluation/.env && set +a
PYTHONPATH=tests/evaluation python -m src.judge tests/evaluation/results/runs/<run_id>/run.yaml --enable-judge
```

`--enable-judge` enables the configured Judge model only for that command. If
`judge_model.enabled: true` is set in `eval_config.yaml`, the flag is not needed,
and Challenger runs will also auto-run the Judge unless disabled in config.

## Results

Reports are written as `results/runs/<run_id>/index.html`, `results/latest.html`,
and `results/latest.txt`.

Open `results/latest.html` from the local filesystem to browse clickable run
reports. The report also prints absolute local paths beside the links so the
same outputs can be opened directly from JupyterLab or VS Code over SSH. These
files are not served by the Shopping Assistant UI on `http://localhost:3000`;
using that app URL for evaluator result files can return `403 Forbidden`.

Key files:

- `results/latest.html`: clickable index of runs.
- `results/latest.txt`: latest run id.
- `results/runs/<run_id>/index.html`: one run's visual report.
- `results/runs/<run_id>/run.yaml`: full machine-readable run record, including
  Judge output after `src.judge` runs.

Do not put runtime Shopping Assistant code here.
