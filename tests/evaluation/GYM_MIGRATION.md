# Running the Evaluation under NeMo Gym — Migration Guide

This folder's **Challenger + Judge** framework is the source of truth. This guide
explains how we run it **under NeMo Gym** so evaluation produces standardized,
reproducible metrics you can track **commit-to-commit** (and get the Gym ecosystem:
common format, concurrency, experiment-tracker/leaderboard hooks, RL-training path).

> **What Gym adds, plainly:** it doesn't replace the Challenger/Judge — it *runs*
> them and emits a standard `reward` + metrics artifact per run. Run it in CI on each
> agent commit and the numbers are apples-to-apples, so regressions are visible.

---

## 1. Install NeMo Gym

```bash
# in the NeMo Gym checkout (separate repo/dir; referenced by GYM_DIR)
cd /path/to/Gym
uv venv && uv sync --extra dev
.venv/bin/gym --help          # sanity check
```

Then register this repo's Gym adapter with that Gym checkout (symlinks only — the
code stays here):

```bash
cd /path/to/retail-shopping-assistant
GYM_DIR=/path/to/Gym bash tests/evaluation/gym/setup_gym.sh
```

Env vars come from the repo `.env` (`CHALLENGER_MODEL_*`, `JUDGE_MODEL_*`,
`NVIDIA_API_KEY`, target agent URL).

---

## 2. Gym concepts (and how our framework maps in)

A Gym **environment** is four decoupled pieces wired by a Hydra YAML config, run by
`gym eval run`:

| Gym concept | Gym term / location | Our piece |
|---|---|---|
| **Dataset** | JSONL, one task/row (OpenAI *Responses* format): `responses_create_params.input` + `verifier_metadata` | `datasets/*/scenarios.yaml`, converted |
| **Agent harness** | *responses API agent* in `responses_api_agents/` (`SimpleResponsesAPIAgent.run()`) | `challenger_agent` — **native** shopper loop (logic ported from `challenger.py`) |
| **Verifier (+ state)** | *resources server* in `resources_servers/` (`SimpleResourcesServer.verify()` → **`reward`**) | `challenger_judge` — **native** verify applying `judge_rules.md` |
| **Model** | *model server* in `responses_api_models/` | `challenger_model` + `judge_model` (Gym model servers) |

> **Note on "forwarder":** that was a throwaway nickname from an earlier prototype —
> **not** a Gym term. The Gym slot the Challenger fills is the **agent harness /
> responses API agent**.

**Run command:** `gym eval run --config <config> --agent challenger_agent
--input <dataset>.jsonl --output results/<run>.jsonl --split validation`. Gym starts
the servers, runs each task, and writes rollouts + aggregate metrics.

---

## 3. How the Challenger runs as a Gym agent (native)

`challenger_agent` implements the loop **directly** (no `src/` import at runtime;
prompt/parse ported from `challenger.py`):

- Gym hands the agent one task = one scenario (`verifier_metadata` carries the
  scenario brief/goal/constraints/behavior + any image asset).
- The agent asks the **`challenger_model`** (Gym model server) for each shopper turn,
  sends it to the live assistant, and continues until `goal_complete`/`max_turns`.
- Each turn records a **curated** assistant reply — `response`/`images`/`cart`/`timings`
  plus the configured `recorded_diagnostics` from `agent_diagnostics` (active skill,
  tool calls, rejected/duplicate calls, termination reason; #140). Needs
  `EXPOSE_AGENT_DIAGNOSTICS=true` on the target.
- The transcript is attached to the Gym rollout via `extra="allow"` and handed to the
  verifier.

---

## 4. How the Judge criteria map to Gym (nothing in `judge_rules.md` changes)

Gym is **agnostic** to judging criteria. Its only hard requirement from the verifier
is a single **`reward` float**. So:

- `verify()` sends the transcript + `judge_rules.md` to the **`judge_model`** server
  and parses your **1–5 score, 11 criteria, `pass`, `critical_failures`** — the rubric
  file is used verbatim, unchanged. Enable with `JUDGE_ENABLED=true`.
- **Mapping to Gym:**
  - `reward = 1.0 if pass else 0.0` (headline; a critical failure forces `pass:false`
    → `reward 0`). *(Alternative: `reward = score/5`.)*
  - `score`, the 11 `criteria`, and `critical_failures` are attached as extra fields
    (Gym persists them) and surfaced via Gym's **`aggregate_metrics`** hook as
    **pass-rate, mean score, and per-criterion means** — the numbers you trend across
    commits.

So the criteria align to Gym by *addition*, not change: Gym reads your rubric; your
rubric drives the reward.

---

## 5. Directory layout of the Gym adapter

Repo-owned; Gym only gets symlinks.

```text
tests/evaluation/
  ... (Challenger/Judge framework — source of truth) ...
  gym/
    README.md  config.yaml  setup_gym.sh  run_eval.sh  build_dataset.py
    responses_api_agents/                 # Gym concept: AGENT HARNESS
      challenger_agent/app.py             #   SimpleResponsesAPIAgent -> wraps run_scenario
    resources_servers/                    # Gym concept: VERIFIER
      challenger_judge/app.py             #   SimpleResourcesServer.verify() -> wraps judge_scenario
```

The subfolders deliberately mirror NeMo Gym's own tree (`responses_api_agents/`,
`resources_servers/`) so the layout itself reflects the Gym concepts.

---

## 6. Prerequisites to run

1. **Assistant running from `staging`** (has the VLM/omni media path). `docker compose up`.
2. **NeMo Gym installed** + `setup_gym.sh` run (section 1).
3. **`.env`** with `CHALLENGER_MODEL_*`, `JUDGE_MODEL_*` (Judge optional), `NVIDIA_API_KEY`.

## 7. Run + read metrics

```bash
GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh          # all configured datasets
GYM_DIR=/path/to/Gym bash tests/evaluation/gym/run_eval.sh --limit 1  # one scenario (smoke)
```

Outputs: Gym rollouts (transcripts) + aggregate metrics (pass-rate, mean score,
per-criterion) under `tests/evaluation/results/`. Your `report.py` HTML can run as a
post-step on the same records.

> Status: guide first; the `gym/` adapter is built in phases — Phase 1 (Challenger
> agent + text_shopping), Phase 2 (Judge verifier), Phase 3 (image/style + CI metrics).
