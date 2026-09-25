# Notebooks

Three notebooks, in order. Each one is a sequence of steps: it says what to
**Run** or **Do**, shows what **You should see**, and says where to look when
you see something else.

| Notebook | You will | Time |
|---|---|---|
| [1 · Getting Started](1_Getting_Started.ipynb) | deploy on hosted endpoints, hold a first conversation, and call each component once | ~20 min |
| [2 · Observability](2_Observability.ipynb) | read a turn's traces in Phoenix: skills, prompts, tool calls, refusals, time and tokens | ~30 min |
| [3 · Evaluation](3_Evaluation.ipynb) | replay fixed conversations, read failures, repeat, compare runs, write a scenario | ~30 min |

## Before you start

- Docker with the Compose plugin
- Python 3.10 or later, with Jupyter
- An NVIDIA API key, from [build.nvidia.com](https://build.nvidia.com)
- No GPU: every model runs on a hosted endpoint

Start Jupyter from a shell that has the key, at the repo root:

```bash
docker login nvcr.io                   # username: $oauthtoken   password: your API key
export NVIDIA_API_KEY="nvapi-..."
export EXPOSE_AGENT_DIAGNOSTICS=true   # development only: each turn reports its tools
jupyter lab notebook/
```

`1_Deploy_Retail_Shopping_Assistant.ipynb` is the previous deploy notebook,
kept unchanged because the QA workflow runs it. Start with Getting Started.

`helpers.py` holds the shared plumbing: HTTP calls, the chat stream, and paging
through Phoenix. The notebooks keep the code that teaches in their own cells.
Service addresses default to `localhost`. Override one with an environment
variable of the same name (`CHAIN_SERVER`, `CATALOG`, `MEMORY`, `PHOENIX`) to
use a remote deployment.

## Going further

- [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md): local NIMs on your own GPUs, and production settings
- [docs/API.md](../docs/API.md): every endpoint
- [docs/OBSERVABILITY.md](../docs/OBSERVABILITY.md): tracing layers and NeMo Relay
- [tests/evaluation/README.md](../tests/evaluation/README.md): replay, Challenger and Judge
