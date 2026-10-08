---
name: retail-local-runner
description: Start, stop, configure, inspect, redeploy, and troubleshoot the Retail Shopping Assistant locally or through Docker Compose, covering local app processes, local Milvus infra containers, and locally deployed models from docker-compose-model-local.yaml.
metadata:
  short-description: Run Retail Shopping Assistant locally
---

# Retail Local Runner

Use this skill when the user wants to run, stop, configure, inspect, redeploy, or troubleshoot the Retail Shopping Assistant from this repository.

## Core Workflow

The deterministic runner is:

```bash
python skills/retail-local-runner/scripts/local_runner.py <command>
```

Available commands:

- `configure`: create ignored `.local-run/model-endpoints.env` values that point app services to a remote host serving the models from `docker-compose-model-local.yaml`.
- `install-dev`: create `.local-run/dev-venv` and install Python dev/test packages there.
- `start`: start local Milvus infra containers, then local memory, guardrails, catalog, chain-server, and UI processes.
- `stop`: stop only tracked local app/UI processes and local Milvus infra containers. Never stop the remote models.
- `status`: show tracked process, port, health, and Milvus infra status.
- `logs`: print recent logs from `.local-run/logs`.

## Compose Redeploy

When the user asks to redeploy the whole stack with Compose, source their env
profile and rebuild:

```bash
set -a && source .env && set +a
docker compose -f docker-compose.yaml up -d --build --force-recreate --remove-orphans
```

If the locally hosted models are running (`.env.local-models`), source that
profile instead and leave out `--remove-orphans`: both compose files share one
project, so it would stop the models.

Then verify:

```bash
docker compose -f docker-compose.yaml ps
curl -fsS --max-time 5 http://localhost:8009/health
curl -fsS --max-time 5 http://localhost:8010/health
curl -fsS --max-time 5 http://localhost:8011/health
curl -fsS --max-time 5 http://localhost:3000 >/dev/null && echo ui_ok
```

Do not print raw env values. If validation is needed before deployment, print
only variable names with values redacted. Do not invent endpoint values; ask
which profile to source if it is not obvious.

## Stop Workflow

When the user asks to stop, shut down, tear down, or restart the local Retail Shopping Assistant, run `stop` first. Do not ask for the model host and do not check or create model endpoint files for stop-only requests.

```bash
python skills/retail-local-runner/scripts/local_runner.py stop
```

The stop command only kills PID files tracked under `.local-run/pids/` and stops local Milvus infra containers from `docker-compose.yaml`:

- `milvus`
- `seaweedfs`
- `etcd`

It must not stop or modify the remote model host running `docker-compose-model-local.yaml`. If ports are still occupied after `stop`, use `status` and `lsof` to report the untracked owner instead of killing unrelated processes.

## Remote Model Host

Before running `configure` or `start`, check whether the ignored local model endpoint env exists:

```bash
test -f .local-run/model-endpoints.env && sed -n 's/=.*/=<redacted>/p' .local-run/model-endpoints.env
```

If the model endpoint env is missing and the user wants to point app services at a remote model host, ask for the remote model host URL before running the script. Do not treat the runner's missing-host error as the final answer. Ask exactly:

```text
What is the remote model host URL? Use the base host without per-model ports, for example http://MODEL_HOST.
```

Use one host URL such as `http://MODEL_HOST`; the runner derives these endpoints, which match the ports in `docker-compose-model-local.yaml`:

- LLM and VLM: `:8000/v1`
- text embeddings: `:8001/v1`
- content safety: `:8003/v1`
- dedicated topic control: `:8004/v1`
- complete-video submission safety: `:8005/v1`

Run:

```bash
python skills/retail-local-runner/scripts/local_runner.py configure --model-host http://HOST
```

For a fresh start when the URL is already known, this is enough:

```bash
python skills/retail-local-runner/scripts/local_runner.py start --model-host http://HOST
```

## Local Services

The runner starts these local app processes:

- memory retriever: `http://localhost:8011`
- guardrails: `http://localhost:8012`
- catalog retriever: `http://localhost:8010`
- chain server: `http://localhost:8009`
- UI: `http://localhost:3000`

Only Milvus infra remains containerized:

```bash
docker compose -f docker-compose.yaml up -d etcd seaweedfs milvus
```

If another local Milvus is already healthy at `localhost:19530` with health on `http://localhost:9091/healthz`, the runner reuses that local endpoint instead of creating conflicting Docker containers. `stop` still only stops this repo's tracked app processes and this repo's Compose infra; it does not stop Milvus containers owned by another project.

## Operational Notes

- Runtime files live under `.local-run/`; service logs are in `.local-run/logs/`.
- Service virtual environments are created under each service's `venv/` directory.
- Existing local Milvus on `localhost:19530` is reused when healthy.
- Python dev/test packages must be installed into `.local-run/dev-venv`, never the global interpreter:
  `python skills/retail-local-runner/scripts/local_runner.py install-dev`.
- Use `.local-run/dev-venv/bin/python -m pytest ...` for local unit tests after `install-dev`.
- UI dependencies are installed into `ui/node_modules` when missing.
- The runner creates `ui/public/images -> shared/images` so the Vite dev server can serve catalog images from `/images/...`, matching the UI Dockerfile behavior.
- The runner sets `SHARED_ROOT`, `SHARED_CONFIG_ROOT`, and `VITE_API_BASE_URL=/api`; the Vite dev server proxies `/api` to the chain server.
- `configure --model-host` writes `.local-run/model-endpoints.env` with the per-role base URLs and model names.
- When `NVIDIA_API_KEY` or `NGC_API_KEY` is present, the runner uses it as the default for `LLM_API_KEY`, `EMBED_API_KEY`, and `RAIL_API_KEY`; generated local endpoint envs use `local` as a no-auth placeholder when no key is set.
- `WEATHER_ENABLED` and `WEATHER_API_KEY` remain available only to the local
  chain-server process; the runner removes them from memory, guardrails,
  catalog, and UI environments and never writes them to
  `.local-run/model-endpoints.env`.
- Use `status` before deciding whether to start or stop.
- Use `logs --service catalog-retriever --lines 120` when catalog startup is slow; first startup may populate Milvus embeddings through the remote embedding model.

## Validation

After editing this skill, validate both folders:

```bash
uv run --with pyyaml python "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/retail-local-runner
uv run --with pyyaml python "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/retail-local-runner
```
