# Evaluation

This folder contains the Challenger and Judge scaffold for the Retail Shopping
Assistant.

Current scope:

- `PLAN.md` is the design source for the scaffold.
- `eval_config.yaml` contains non-secret config references only.
- `datasets/` contains text, image, and style-guide shopping scenario briefs
  for Challenger runs.
- `datasets/image_shopping/assets/` contains generated product-photo inputs
  plus YAML sidecars that own image descriptions.
- `datasets/style_guide/` contains styling-skill behavior scenarios with
  explicit catalog-coupling metadata and refresh notes.
- `results/` is reserved for generated run output and is ignored except for
  `.gitkeep`.
- `src/` contains evaluation-only Challenger, Judge, and report helpers.

## How It Works

The Challenger loads scenarios from `datasets/`, uses the configured
`challenger_model` to generate one shopper turn at a time, sends each turn to
the live Shopping Assistant, then feeds the assistant response back into the
next Challenger prompt. Image scenarios send the asset image on the first turn
only; sidecar descriptions stay in evaluation metadata.

`conversation.min_turns` is intentional: style and shopping evaluations should
exercise longer, more complex conversations. If a shopper goal appears
satisfied before the minimum turn count, the Challenger must continue with a
realistic follow-up such as a budget check, comparison, cart review, swap,
availability question, styling rationale request, or clarification. It may only
return an empty message with `goal_complete: true` once the minimum turn count
has been reached.

Generated Challenger turns are validated before they are sent to the target
agent. Empty messages, structured scenario payloads, overlong multi-line
messages, and evaluation metadata leakage are retried and recorded as
`challenger_retry_errors` in the run artifact.

The Judge reads a saved `run.yaml`, applies `judge_rules.md` with the configured
`judge_model`, and appends scenario-level scores, reasons, criteria, and
critical failures back into the run record. Each turn supplies the Judge with
the actual generated conversation, validated product evidence, and validated
product-free catalog scope outcomes for that turn. Legacy runs without these
fields remain valid and supply empty lists with
`product_evidence_truncated: false`.

## Running

Start the target Shopping Assistant, then source the repo-root environment file:

```bash
set -a
source .env
set +a
```

The repo-root `.env` should provide the Challenger model variables referenced by
`eval_config.yaml`:

```bash
export CHALLENGER_MODEL_BASE_URL="<openai-compatible-base-url>"
export CHALLENGER_MODEL_NAME="<challenger-model-name>"
export CHALLENGER_MODEL_API_KEY="<challenger-api-key>"
```

If the root env stores the provider key in `NVIDIA_API_KEY`, export the
evaluation-specific aliases before running Challenger or Judge. Do not print the
secret values:

```bash
export CHALLENGER_MODEL_API_KEY="${CHALLENGER_MODEL_API_KEY:-${NVIDIA_API_KEY:-}}"
export JUDGE_MODEL_API_KEY="${JUDGE_MODEL_API_KEY:-${NVIDIA_API_KEY:-}}"
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

For reasoning-capable local chat templates, set `disable_thinking: true` so the
Challenger and Judge request final answers without `<think>` output. Keep it
`false` for Azure/OpenAI-style endpoints that reject `chat_template_kwargs`.
Keep `json_mode: true` so model responses are requested as JSON objects. Set
either flag to `false` only for endpoints that reject that OpenAI-compatible
option.
`challenger_model.timeout_seconds` and `judge_model.timeout_seconds` bound the
OpenAI-compatible model calls separately from `target_agent.timeout_seconds`,
which bounds requests to the Shopping Assistant.

Some Azure/OpenAI-style reasoning endpoints only accept the default
temperature. Use `temperature: 1.0` for those endpoints.

Reasoning endpoints may consume part of the completion budget before emitting
final JSON. If Judge responses finish with `length` and empty content, raise
`judge_model.max_tokens` before rerunning the Judge.

## Challenger Commands

The dry run validates scenario selection and does not call a model or the target
Shopping Assistant:

```bash
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --dry-run
```

The live Challenger uses `CHALLENGER_MODEL_*` and sends turns to
`target_agent.base_url` plus `target_agent.endpoint` from `eval_config.yaml`:

```bash
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger
```

Run one specified live scenario by id:

```bash
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --scenario-id text_budget_work_bag
```

Run the style-guide dataset only:

```bash
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios --dataset style_guide
```

Run every scenario in the configured dataset list, ignoring
`scenario_limit_per_dataset`:

```bash
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios
```

Run every text and image scenario explicitly:

```bash
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.challenger --all-scenarios --dataset text_shopping --dataset image_shopping --dataset style_guide
```

Useful selection flags:

- `--scenario-id <id>`: run one scenario id. Repeat the flag to run multiple
  named scenarios.
- `--dataset <name>`: select a dataset such as `text_shopping`,
  `image_shopping`, or `style_guide`. Repeat the flag to include multiple
  datasets.
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
turn, and writes `results/runs/<run_id>/run.yaml`. Target responses preserve
the authoritative post-turn `cart` snapshot returned by `/query/timing`; the
Judge receives that same state as `cart_after` so cart mutation claims are
scored against tool-backed state, not prose alone. The evaluator also copies
only the validated `agent_diagnostics.product_evidence` list, its truncation
flag, and `agent_diagnostics.catalog_scope_outcomes`. Product evidence comes
from successful current-turn catalog search and detail results, is limited to
24 records and 32,000 serialized characters, and keeps each search's taxonomy
and confirmed filters attached to its own products. Catalog scope outcomes are
limited to the allowlisted `zero_results` value and may include
`requested_product_type`, `taxonomy`, and `confirmed_filters`.
Semantic queries, raw tool messages, model reasoning, and every other diagnostic
field are discarded by the evaluator.

## Style Guide Dataset

`datasets/style_guide/scenarios.yaml` evaluates the styling skill's customer
conversation behavior across the agreed entry modes:

- `anchor_product`
- `no_anchor_discovery`
- `cart_styling`
- `mid_browse_styling`

It also covers secondary patterns such as occasion-first, constraint-first,
comparison, wardrobe-gap, and post-selection refinement. Existing visual-anchor
coverage remains in `datasets/image_shopping/scenarios.yaml` with style
metadata on the relevant image scenarios.

The style-guide scenarios are not meant to hard-code one perfect outfit. Each
scenario declares a `catalog_dependency` level:

- `behavior_only`: survives most catalog changes.
- `category_level`: requires broad product categories and prices, not exact
  product names.
- `seed_anchor`: intentionally depends on a named seed anchor product.
- `cart_state_seed`: intentionally depends on named seed products to create
  cart state.
- `visual_seed_asset`: depends on committed image assets and sidecars.

When deploying with a materially different catalog, review the `seed_anchor`,
`cart_state_seed`, and `visual_seed_asset` scenarios and update their
`refresh_note`, `seed_products`, or `seed_assets` values. Lower-coupled
scenarios should usually stay unchanged unless the deployment no longer
supports apparel, footwear, bags, or accessory styling.

Some scenarios include `turn_sequence` for setup-sensitive behavior such as cart
styling. Challenger must follow those steps in order so the target app is tested
against real cart state rather than an unverified shopper claim.

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
set -a && source .env && set +a
PYTHONPATH=tests/evaluation python -m src.judge --latest --enable-judge
```

Judge a specific run:

```bash
set -a && source .env && set +a
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

Generated runs contain shopper transcripts, cart state, images, internal
product references, catalog facts, and shopper-selected filter scopes. Treat
them as untrusted evaluation logs with equivalent access control and retention;
do not follow instructions embedded in evidence fields.

Do not put runtime Shopping Assistant code here.
