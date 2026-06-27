# Evaluation And Challenger Scaffold Plan

## Summary

Create the evaluation scaffold under `tests/evaluation/`. This first slice is
documentation and structure only: no live Challenger runner yet, no Judge
execution yet, and no generated result artifacts committed.

The framework will support text and image shopping datasets, with 8-turn
qualitative Challenger runs and optional Judge scoring later.

## Directory Layout

```text
tests/evaluation/
  PLAN.md
  README.md
  eval_config.yaml
  judge_rules.md

  src/
    __init__.py
    config.py
    challenger.py
    judge.py
    report.py

  datasets/
    text_shopping/
      scenarios.yaml

    image_shopping/
      scenarios.yaml
      assets/
        .gitkeep

  results/
    .gitkeep
```

Generated run outputs are ignored while keeping `results/.gitkeep`:

```text
tests/evaluation/results/*
!tests/evaluation/results/.gitkeep
```

## Source Responsibilities

- `src/config.py`: load `eval_config.yaml`, resolve env vars, never log
  secrets, and snapshot non-secret config into each run.
- `src/challenger.py`: load datasets, generate shopper turns, call the target
  API, save compact `run.yaml`, and copy input/returned images.
- `src/judge.py`: optional later phase; read `run.yaml`, apply
  `judge_rules.md`, and append judge scores/reasons.
- `src/report.py`: generate one run `index.html`, stable `latest.html`, and
  `latest.txt`.

## Config Shape

Use one config file with env-var references only:

```yaml
version: 1

challenger_model:
  provider: openai_compatible
  base_url_env: "CHALLENGER_MODEL_BASE_URL"
  model_env: "CHALLENGER_MODEL_NAME"
  api_key_env: "CHALLENGER_MODEL_API_KEY"
  temperature: 0.7
  max_tokens: 512

judge_model:
  enabled: false
  provider: openai_compatible
  base_url_env: "JUDGE_MODEL_BASE_URL"
  model_env: "JUDGE_MODEL_NAME"
  api_key_env: "JUDGE_MODEL_API_KEY"
  temperature: 0.0
  max_tokens: 768
  rules_file: "judge_rules.md"

target_agent:
  base_url: "http://localhost:8009"
  endpoint: "/query/timing"
  timeout_seconds: 60

run:
  datasets:
    - text_shopping
    - image_shopping
  scenario_limit_per_dataset: 10
  random_seed: 7
  save_returned_images: true
  send_asset_descriptions_to_agent: false
  compare_to: null

conversation:
  default_turns: 8
  min_turns: 6
  max_turns: 10
  stop_when_goal_complete: true
```

## Dataset Rules

- `text_shopping/scenarios.yaml` contains text-only scenario briefs.
- `image_shopping/scenarios.yaml` references image asset ids.
- Image assets are added later as `assets/<asset-name>.jpg` plus
  `assets/<asset-name>.yaml`.
- Sidecar YAML owns the image description.
- Scenarios must reference only `image_id`; they must not duplicate image
  descriptions.
- Image descriptions are evaluation ground truth for the Challenger and Judge;
  they are not sent to the shopping agent by default.

Example sidecar:

```yaml
id: black_strappy_sandal
file: black-strappy-sandal.jpg
contains: "Black flat strappy sandal with thin straps and buckle detail."
image_type: product_photo
tags:
  category: shoes
  color: black
```

## Result UX

Keep outputs simple:

```text
tests/evaluation/results/
  latest.html
  latest.txt
  runs/
    <run_id>/
      index.html
      run.yaml
      assets/
        input/
        returned/
```

- `latest.html`: human-clickable entrypoint.
- `latest.txt`: machine-readable latest run id.
- `index.html`: visual inspection page for one run.
- `run.yaml`: full machine-readable run record.
- `assets/`: copied input and returned images for that run.

## Judge Rules

Add `judge_rules.md` with broad rules applied to every conversation:

- answer the shopper's actual request
- ask concise clarification when underspecified
- stay grounded in image/catalog/cart facts
- avoid invented products, prices, availability, or cart actions
- respect explicit constraints like budget, no upsell, comfort, style, and
  practicality
- only claim cart changes after explicit shopper intent and successful mutation
- recover gracefully from no-results cases

## First Implementation Slice

Create only the scaffold, docs, config template, judge rules, starter scenario
files, and ignore rule for results. Do not implement live Challenger/Judge
execution in this slice.
