# Evaluation on NeMo Gym — operational guide

Runs the Retail Shopping Assistant evaluation under **NeMo Gym**: a **Challenger**
(simulated shopper) drives the live assistant for a multi-turn conversation, and a
**Judge** scores it against the rubric in [`../judge_rules.md`](../judge_rules.md).
Output is a standardized rollout + metrics artifact you can trend commit-to-commit.

**Native, no wrappers.** The Challenger loop and the Judge live directly in these
Gym servers (logic ported from `../src/`); they don't import the framework at
runtime. The LLMs run as Gym **model servers**.

- Concepts + why-Gym: [`../GYM_MIGRATION.md`](../GYM_MIGRATION.md)
- Rubric: [`../judge_rules.md`](../judge_rules.md)

---

## Layout

```text
gym/
  config.yaml            # the Gym "environment": model servers + agent + verifier + dataset
  setup_gym.sh           # register these servers into a NeMo Gym checkout (symlinks)
  run_eval.sh            # build dataset -> gym eval run -> rollouts + metrics
  build_dataset.py       # datasets/<name>/scenarios.yaml -> Gym JSONL (+ image assets)
  responses_api_agents/challenger_agent/app.py   # agent harness: native shopper loop
  resources_servers/challenger_judge/app.py      # verifier: native judge (judge_rules.md)
  build/                 # generated Gym JSONL (gitignored)
  results/               # rollouts + aggregate metrics (gitignored)
```

---

## 1. Install NeMo Gym (one time)

```bash
# NeMo Gym lives in its own checkout; GYM_DIR points at it.
cd /path/to/Gym
uv venv && uv sync --extra dev
.venv/bin/gym --help                     # sanity check

# register this repo's servers with that Gym checkout (symlinks; code stays here)
cd /path/to/retail-shopping-assistant
GYM_DIR=/path/to/Gym bash tests/evaluation/gym/setup_gym.sh
```

## 2. Bring up the assistant + credentials

- **Assistant running** (target of the eval). Local: `docker compose up -d` (use the
  `staging` branch for the VLM/media build). Or point at a deployment (step 3).
  - Start it with **`EXPOSE_AGENT_DIAGNOSTICS=true`** so the assistant returns per-turn
    `agent_diagnostics` (which skill was active, which tools ran, rejected/duplicate
    calls, termination reason). Without it those fields come back **absent** — the run
    still works, but the recorded turns can't explain a failure after the fact.
- **`.env`** at the repo root with:
  - `CHALLENGER_MODEL_BASE_URL`, `CHALLENGER_MODEL_NAME` — the shopper-simulator LLM
  - `JUDGE_MODEL_BASE_URL`, `JUDGE_MODEL_NAME` — the judge LLM
  - `NVIDIA_API_KEY` — the key for both (the `*_MODEL_API_KEY` vars are empty;
    `run_eval.sh` falls back to `NVIDIA_API_KEY`).

## 3. Run the eval

```bash
# default dataset (text_shopping), Challenger only (no scoring)
GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh

# choose a dataset:  text_shopping | visual_uploads (image+video) | style_guide
DATASET=visual_uploads GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh

# turn the Judge ON (score with judge_rules.md)
JUDGE_ENABLED=true GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh

# one scenario (smoke test)
GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh --limit 1

# point at a remote/staging assistant instead of localhost:8009
RETAIL_ASSISTANT_URL=https://your-staging GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh
```

`run_eval.sh` runs **one dataset** (env `DATASET`, default `text_shopping`) and writes
`results/<DATASET>_rollouts.jsonl`. Run it once per dataset.

`run_eval.sh` sources `.env`, builds the dataset, clears stale caches, and runs
`gym eval run` (Gym starts the model servers + agent + verifier, runs each scenario,
writes results, tears down).

### Concurrency (parallel scenarios)

Gym runs **scenarios in parallel**, bounded by a semaphore. `run_eval.sh` defaults to
**`CONCURRENCY=3`** (three conversations at once). Turns *within* a scenario stay
sequential — each turn depends on the last; only whole scenarios run side by side.

```bash
# override the default of 3
CONCURRENCY=6 DATASET=text_shopping GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh

# fully serial (useful for clean logs / debugging one scenario at a time)
CONCURRENCY=1 GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh
```

**Impact:** wall-clock scales roughly linearly — a 10-scenario suite at `CONCURRENCY=3`
finishes in ~⅓ the time of serial. Raising it shortens a full-suite run.

**Attention — this is bounded by shared backends, so tune it deliberately:**
- **Safe against state collisions.** Each scenario uses a distinct deterministic
  `user_id` (derived from the scenario id) and resets its session, so parallel
  conversations don't corrupt each other's memory/cart on the shared assistant.
- **Hosted-model rate limits are the real ceiling.** Every parallel scenario multiplies
  concurrent calls to the challenger, judge, *and* the assistant's own LLM. Too high →
  **429 / rate-limit errors** that fail scenarios (reward 0) and skew the run. If you see
  429s, lower `CONCURRENCY`.
- **Shared assistant throughput.** All scenarios hit the same assistant instance; past a
  point you queue on it, not gain speed. `3` is a conservative default — raise gradually
  (e.g. 3 → 6) and watch for errors/latency before going higher.
- A value passed after `--` still wins: `run_eval.sh --concurrency 8` overrides the env
  default (last flag wins in `gym eval run`).

## 4. Study the results

Written to `tests/evaluation/gym/results/`:

| File | What |
|---|---|
| `<dataset>_rollouts.jsonl` | one record per scenario — the transcript + score |
| `<dataset>_rollouts_aggregate_metrics.json` | run summary (mean reward, etc.) — **the commit-trackable artifact** |
| `<dataset>_rollouts_materialized_inputs.jsonl` | the exact inputs used |

Each rollout row (JSON):
- `reward` — **1.0 if the Judge passed, else 0.0** (0.0 when the Judge is off).
- `score` (1–5), `passed`, `criteria` (the 11), `critical_failures` — present when judging is on.
- `response.retail_record.turns[]` — the conversation: each turn has `shopper` (the
  generated message) and `target`, a **curated** record of the assistant's reply:
  `status_code`, `response`, `images`, `cart`, `timings`, plus the configured
  **diagnostics** from `agent_diagnostics` — `product_evidence`,
  `product_evidence_truncated`, `catalog_scope_outcomes`, `tool_calls`,
  `skill_files_read`, `rejected_tool_calls`, `duplicate_tool_calls`,
  `final_termination_reason` (present only when the target ran with
  `EXPOSE_AGENT_DIAGNOSTICS=true`; a field the target didn't send is absent, not empty).
  It records these named fields, **not** a dump of the whole response — the recorded
  turn is the contract the Judge adjudicates against.

Quick read:

```bash
python - <<'PY'
import json
rows=[json.loads(l) for l in open('tests/evaluation/gym/results/text_shopping_rollouts.jsonl') if l.strip()]
for d in rows:
    rec=d['response']['retail_record']
    print(rec['id'], '| reward', d['reward'], '| score', d.get('score'), '| turns', len(rec['turns']))
    for t in rec['turns']:
        print('  shopper  :', t['shopper'][:90])
        print('  assistant:', (t['target'].get('response','') or '')[:110])
PY
```

The framework's `../src/report.py` can also render HTML from a `run.yaml`.

---

## The Challenger (agent harness)

`responses_api_agents/challenger_agent/app.py`. For each scenario it:
1. builds a prompt from the scenario (`shopper_goal`, `constraints`,
   `shopper_behavior`, `language_cues`) + the transcript so far,
2. asks the **`challenger_model`** server for the next shopper turn (JSON:
   `message` + `goal_complete`),
3. sends it to the assistant, records the turn,
4. stops on `goal_complete` after `min_turns`, or at `target_turns`/`max_turns`.

Each turn's `target` is a **curated** record (not a raw response dump): top-level
`response`/`images`/`cart`/`timings` plus the named `recorded_diagnostics` from the
assistant's `agent_diagnostics` — a faithful port of the framework's `TargetAgentClient`
(tests/evaluation, #140).

Knobs (in `config.yaml` under `challenger_agent`): `target_agent_url`, `guardrails`,
`default_turns` (8), `min_turns` (6), `max_turns` (10), `recorded_diagnostics` (the
8 diagnostic fields kept per turn — needs `EXPOSE_AGENT_DIAGNOSTICS=true` on the target).

## The Judge + criteria (verifier)

`resources_servers/challenger_judge/app.py`. When `JUDGE_ENABLED=true`, `verify()`
sends the transcript + [`../judge_rules.md`](../judge_rules.md) to the **`judge_model`**
server and parses the verdict. **Reward = `pass ? 1 : 0`**; `score`/`criteria`/
`critical_failures` ride along.

**Rubric (from `judge_rules.md`):** 1–5 score, **pass threshold ≥ 4**.

11 criteria: `goal_completion`, `relevance_helpfulness`, `groundedness`,
`constraint_following`, `multi_turn_context`, `tool_state_correctness`,
`clarification_recovery`, `safety_scope`, `communication_quality`,
`style_composition_quality`, `decision_boundary_quality`.

Critical failures (force `pass:false`): invented product/price/availability/cart;
cart change without request+success; ignored budget/no-upsell; image ignored when
referenced; valid request refused; unsafe/out-of-scope; internal names leaked; style
advice as catalog fact.

---

## Image & video tests ("shop this look")

A scenario can hand the assistant an **image or a video** on the first turn (the
staging assistant's VLM understands both). Layout:

```text
datasets/<dataset>/
  scenarios.yaml
  assets/
    <id>.jpg | <id>.mp4     # the media file
    <id>.yaml               # sidecar: id, file, `contains` (ground-truth description)
```

A scenario references the asset by `image_id`:

```yaml
- id: video_shop_this_look_evening_gown
  image_id: lady_black_dress          # -> assets/lady_black_dress.mp4  (video/mp4)
  shopper_goal: "Find a formal evening gown similar to the one in the video."
  constraints: ["Use the attached video as the main reference."]
  shopper_behavior: { type: occasion_focused, notes: "..." }
  language_cues: ["Shop this look for me."]
```

- `build_dataset.py` resolves `image_id` to `assets/<id>.{jpg,png,mp4,mov}` and embeds
  it as `media: [{type: image|video, data: <base64>, mime_type}]`.
- The **Challenger sends the media on turn 1** (as the staging `media` list); the
  sidecar `contains` description is **not** sent to the assistant — it stays in eval
  metadata for grounding.
- The **`visual_uploads`** dataset is a worked example (one menswear image, two outfit
  videos). Run it: `DATASET=visual_uploads ... run_eval.sh`.

> Prereq: the assistant must be the **staging (VLM)** build for media to be understood.

## How to change things

| Want to… | Do this |
|---|---|
| **Add a text test** | add a scenario to `datasets/<dataset>/scenarios.yaml` (`id`, `shopper_goal`, `constraints`, `shopper_behavior`, `language_cues`). Rerun with `DATASET=<dataset>`. |
| **Add an image/video test** | drop the file in `datasets/<dataset>/assets/<id>.{jpg,mp4}`, add a `<id>.yaml` sidecar (`id`, `file`, `contains`), and a scenario with `image_id: <id>` (see [Image & video tests](#image--video-tests-shop-this-look)). |
| **Choose which dataset to run** | `DATASET=visual_uploads bash run_eval.sh` (default `text_shopping`). |
| **Change the criteria / rubric** | edit [`../judge_rules.md`](../judge_rules.md) — the Judge reads it verbatim; no code change. |
| **Change the pass threshold or reward** | reward mapping is in `challenger_judge/app.py` (`reward = 1 if passed else 0`); the threshold text is in `judge_rules.md`. |
| **Swap the challenger/judge model** | change `CHALLENGER_MODEL_NAME` / `JUDGE_MODEL_NAME` (and `*_BASE_URL`) in `.env`, or the `model`/`base_url` in `config.yaml`'s model servers. |
| **Change conversation length** | set `default_turns` / `min_turns` / `max_turns` under `challenger_agent` in `config.yaml`. |
| **Change recorded diagnostics** | edit `recorded_diagnostics` under `challenger_agent` in `config.yaml` (needs `EXPOSE_AGENT_DIAGNOSTICS=true` on the target). |
| **Change parallelism** | `CONCURRENCY=N bash run_eval.sh` (default `3`; `1` = serial). Mind hosted-model rate limits — see [Concurrency](#concurrency-parallel-scenarios). |
| **Eval a different assistant** | `RETAIL_ASSISTANT_URL=...` (env) or `target_agent_url` in `config.yaml`. |
| **Add a whole dataset** | create `datasets/<name>/scenarios.yaml`, add it to `build_dataset.py` / a `--dataset <name>` build, and a `datasets:` entry in `config.yaml`. |

## Per-commit tracking (CI)

Run the eval on each agent commit and archive the metrics artifact; the trend across
commits is what you diff. The headline number is `mean/reward` (= pass-rate when the
Judge is on).

```bash
# per commit, in CI:
JUDGE_ENABLED=true DATASET=text_shopping GYM_DIR=/opt/Gym \
  bash tests/evaluation/gym/run_eval.sh

# archive the metric keyed by commit, then compare to the previous one
mkdir -p eval_metrics
cp tests/evaluation/gym/results/text_shopping_rollouts_aggregate_metrics.json \
   "eval_metrics/$(git rev-parse --short HEAD).json"
# a regression = mean/reward drops vs the last commit's file
```

Gym gives standardized, reproducible numbers; wire the archived files into a
dashboard or W&B if you want charts.

## Terminology

`challenger_agent` is a Gym **`responses_api_agent`** (agent harness); `challenger_judge`
is a **resources server** (verifier); the LLMs are **model servers**. ("forwarder" is
not a Gym term.)
