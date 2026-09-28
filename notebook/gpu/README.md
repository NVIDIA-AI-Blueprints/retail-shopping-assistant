# Serving Nemotron for Notebook 5

Nemotron 3.5 Super (BF16) on vLLM, prefix caching on for attention and Mamba
layers, bound to `127.0.0.1` behind an API key. Defaults: **4x H100 80 GB,
TP=4**.

## Quick start

On the GPU machine, from the repo root:

```bash
export HF_TOKEN=hf_...                    # access to the nvidia org on Hugging Face
notebook/gpu/deploy.sh                    # first run downloads ~240 GB
notebook/gpu/verify_prefix_cache.sh       # want a speedup above ~2x
```

Then AIPerf and Jupyter, once, with [uv](https://docs.astral.sh/uv/) (no
`sudo`; stock Ubuntu images lack `python3-venv`, so `python3 -m venv` fails):

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/aiperf --python 3.12
uv pip install --python ~/aiperf/bin/python aiperf==0.13.0 jupyterlab matplotlib
```

Then [Notebook 5](../5_Performance_Measurement.ipynb), in `~/aiperf/bin/jupyter lab`.

## What the machine needs

- NVIDIA driver, Docker, and the NVIDIA Container Toolkit (`docker run --gpus all` works).
- About 280 GB of GPU memory in total, and about 250 GB free disk for the weights.
- `curl`, `bc`, `openssl`, `python3` (3.10 or later).
- This repo. No clone access? Copy `notebook/` alone: Notebook 5 needs only
  `stress_replay.py`, `traces/`, and this folder.

## Adapting to another machine

Every setting is an environment variable to `deploy.sh`. Notebook 5 and
`verify_prefix_cache.sh` read `BASE_URL` (default `http://localhost:8000`).

| Your machine | Set |
|---|---|
| 8x H100 | `TP=8`. Adds expert parallel; much more cache room. |
| Some GPUs busy | `GPUS=4,5,6,7 TP=4` |
| Less than ~280 GB of GPU memory | Won't fit. BF16 is ~240 GB of weights; no smaller quantization exists for this checkpoint yet. |
| Port 8000 taken | `PORT=8001`, and `BASE_URL=http://localhost:8001` for the notebook and the check. |
| Weights on a bigger disk | `HF_CACHE=/data/huggingface` |
| Another checkpoint | `MODEL=...`, and possibly `VLLM_IMAGE=...`. Keep `SERVED_NAME` or pass it to the check as `MODEL`. |
| Out of memory at load | `MAX_NUM_SEQS=64`, or `GPU_MEM_UTIL=0.80` |
| A vLLM-based NIM instead | Skip `deploy.sh`. Start it with `NIM_ENABLE_KV_CACHE_REUSE=1` and point `BASE_URL` at it. |

Other settings:

| Variable | Default | What it does |
|---|---|---|
| `PREFIX_CACHE` | `1` | `0` turns caching off: Notebook 5's Step 5. |
| `MAMBA_PREFIX` | `1` | `0` caches attention only. Use it if vLLM rejects `--mamba-cache-mode align`. |
| `BLOCK_SIZE` | `64` | Match granularity. Try `32` or `16` if prompts part early. |
| `MAX_MODEL_LEN` | `32768` | The trace's longest call is ~28k tokens. Shorter leaves more room for cache. Below ~25k, pass `verify_prefix_cache.sh` a smaller `PREFIX_WORDS`. |
| `MAX_NUM_SEQS` | `128` | Notebook 5's top concurrency. More than this queues. |
| `EXPOSE` | `loopback` | `lan` binds `0.0.0.0`: plain HTTP, and the key goes in the clear. Prefer SSH or Tailscale. |
| `KEYFILE` | `~/.nemotron_api_key` | Generated once. Notebook 5 reads it; `NEMOTRON_API_KEY` overrides. |
| `CONTAINER_NAME` | `nemotron35` | Also read by `verify_prefix_cache.sh`. |

## Reaching it from a laptop

Run Jupyter on the GPU machine and tunnel only the browser:

```bash
ssh -L 8888:localhost:8888 <gpu-host>
~/aiperf/bin/jupyter lab --no-browser --port 8888 notebook/
```

Don't run AIPerf through a tunnel to port 8000: at high concurrency the
tunnel, not the model, sets the numbers.

## Worth knowing

- **Hit rate can lie.** Without `--mamba-cache-mode align`, vLLM counts prefix
  hits but recomputes the Mamba state anyway. `verify_prefix_cache.sh` and
  Notebook 5's caching-off runs judge by TTFT.
- **Mamba reuse is unvalidated on this checkpoint.** If startup rejects
  `--mamba-cache-mode align`, `MAMBA_PREFIX=0` and note it in the results.
- **Pinned image.** vLLM `v0.30.0`; earlier releases don't load this
  architecture, and nightly tags are deleted after a few days.
- **Crashes at high concurrency** (a known MoE kernel issue): lower `MAX_NUM_SEQS`.
- **Watch it:** `docker logs -f nemotron35`, `nvidia-smi`.
  Stop: `docker rm -f nemotron35`.
