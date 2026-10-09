# Guardrails

An optional safety layer that runs as its own service and decides whether a
turn may proceed. It ships enabled for API requests that do not say otherwise.
The UI's Guardrails toggle starts off, to show the same turn before and after,
and turns it on for a session; with `GUARDRAILS_CLIENT_CAN_DISABLE=false` it
starts on.

## How it decides

Three judges, each a separate model, each answering one question.

| Judge | Model role | Question it answers | Default model |
| --- | --- | --- | --- |
| Content safety | `content_safety` | Is this text or image unsafe? | `nvidia/nemotron-3.5-content-safety` |
| Topic control | `topic_control` | Is this a retail shopping request? | `nvidia/llama-3.1-nemoguard-8b-topic-control` |
| Multimodal safety | `multimodal_safety` | Is this attachment unsafe? | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |

They run at two stages. On **input**, shopper text goes to content safety and
topic control, and any attachment goes to multimodal safety. On **output**, the
assistant's reply goes to content safety alone — a reply cannot be off-topic in
a way the input check did not already settle.

Topic control is a retail policy, not a general one. It is loaded from
`shared/configs/rails/` rather than compiled in, so the definition of
"on topic" is yours to edit.

## Three outcomes, and why the third matters

Every check returns `allow`, `block`, or `error`, aggregated with a fixed
precedence: any block blocks, otherwise any error is an error, otherwise allow.

`block` always stops the turn. `error` means a judge did not reach a verdict —
the provider timed out, returned nothing usable, or failed — and what that
costs is a deployment decision:

| `GUARDRAILS_FAILURE_MODE` | An `error` means |
| --- | --- |
| `closed` (default) | The turn stops. Safety checks you cannot run are treated as checks you did not pass. |
| `open` | The turn proceeds unchecked. |

An explicit unsafe verdict blocks in either mode. `open` only relaxes the
unknown.

Keeping `error` distinct from `block` is deliberate, and the code goes out of
its way to preserve it. NeMo's action dispatcher catches an exception inside a
guard action and reports the rail as *blocked*, so a provider outage would
otherwise arrive as a refusal — the shopper sees a polite decline, the turn
looks safely handled, and `GUARDRAILS_FAILURE_MODE` never applies. The rails
evaluator therefore reports action failures out of band and checks them before
the dispatcher's status, because a judge whose model never answered has not
found anything unsafe.

## Turning it on and off

```bash
export RAIL_API_KEY="..."          # the guard models' API key
export GUARDRAILS_ENABLED=false    # only to make off the default; or send `guardrails` per request
docker compose up -d rails
```

The service listens on port 8012 and the chain server reaches it at
`GUARDRAILS_URL`. A request may carry its own `guardrails` value, as the UI
toggle does, and by default it overrides the deployment default either way, so
you can show the same turn with and without guardrails. Set
`GUARDRAILS_CLIENT_CAN_DISABLE=false` to lock them on: `true` still turns them
on, and `false` is ignored.
`/capabilities` reports this as `request_override_supported` and
`client_can_disable`.

## Deploying without guardrails

Set `GUARDRAILS_AVAILABLE=false` for a deployment that has no guardrails at
all, hosted or local: to save the GPUs the judges need, or because no key for
them is available.

```bash
export GUARDRAILS_AVAILABLE=false
```

Then:

- The guardrail service starts without building its rails, so no guard model
  endpoint or key is needed, and every check returns `error` with
  `diagnostic_code: guardrails_unavailable`. The chain server sends none.
- With the locally hosted models, `.env.local-models` points no role at the
  three guardrail models and leaves them out of `LOCAL_MODEL_SERVICES`, so they
  are not started.
- `/capabilities` reports `guardrails.available: false`, the UI hides the
  Guardrails toggle, and the content safety, topic control and multimodal
  safety models show as off.
- A request with `guardrails: true` gets a 400. Requests that omit it, or send
  `false`, run unguarded as before.
- `scripts/model_config.py show --validate` does not check the three guardrail
  roles.

`GUARDRAILS_ENABLED=true` with `GUARDRAILS_AVAILABLE=false` stops the chain
server at startup, as the two contradict each other.

## Configuration

| Variable | What it controls | Default |
| --- | --- | --- |
| `GUARDRAILS_AVAILABLE` | `false`: no guardrails in this deployment ([above](#deploying-without-guardrails)) | true |
| `RAIL_API_KEY` | API key for the guard models | Required |
| `GUARDRAILS_URL` | Where the chain server reaches the service | `http://rails:8012` |
| `GUARDRAILS_ENABLED` | Deployment default for requests that omit `guardrails` | on (off when `GUARDRAILS_AVAILABLE=false`) |
| `GUARDRAILS_FAILURE_MODE` | What an `error` costs | `closed` |
| `GUARDRAILS_TIMEOUT_SECONDS` | Deadline for one decision | `15.0` |
| `GUARDRAILS_INPUT_EXECUTION_MODE` | Run the two text input rails `parallel` or `sequential` | `parallel` |
| `GUARDRAILS_SUPPORTED_MODALITIES` | Which modalities are covered | `text,image` |
| `GUARDRAILS_SPECULATIVE_MAIN_MODEL_ENABLED` | Overlap the first model step with input rails | off |
| `MULTIMODAL_SAFETY_VIDEO_FPS` | Frames per second sampled from a video | `2.0` |

Guardrails spans two services, so each default lives with whichever service
reads it: the chain-server ones in `shared/configs/chain_server/config.yaml`,
the service's own in `guardrails/src/rails.py`. `.env.example` and
`docker-compose.yaml` only pass these through, empty by default, so no setting
has two sources that can drift apart.

`GUARDRAILS_TIMEOUT_SECONDS` is the one both services read. The service gives
itself one second less than the number you set, so the judges are abandoned and
a `policy_timeout` returned *before* the chain server's own deadline cancels
the request — otherwise a slow judge would surface as a dropped connection
rather than a decision your failure mode can act on.

### Speculative execution

With `GUARDRAILS_SPECULATIVE_MAIN_MODEL_ENABLED`, a guarded text-only turn
starts the first model step while the input rails are still running, holding
every tool call behind a gate that opens only on `allow`. It trades a little
waste for latency: a turn that ends up blocked may still have cost one model
request, which is recorded as discarded rather than hidden. Turns with
attachments stay sequential.

## What the shopper sees

Two sentences, both in `shared/configs/chain_server/config.yaml`, neither with
an environment variable — a shopper-facing sentence belongs in reviewed
configuration rather than a deployment variable.

`unsafe_message` answers a block. `guardrails_unavailable_message` answers an
error that stopped the turn, and it asks the shopper to check their cart,
because a cart change may already have committed before the check failed.

The UI surfaces each stage's result inline, labelled by the judge that stopped
it, so a refusal is legible rather than mysterious.

## Media

Images are judged by default. **Video is not.**

The video judge is fully implemented — it samples frames at
`MULTIMODAL_SAFETY_VIDEO_FPS`, includes embedded audio in the judgement, and
prompts the model to ignore any instruction carried inside the video's own
text, speech, or symbols. But `video` is absent from the default
`GUARDRAILS_SUPPORTED_MODALITIES`, so an uploaded video returns
`unsupported_video_safety`, which is an `error` — and under the default
`closed` failure mode the turn stops.

Video judging is not vetted for this release. Adding `video` to
`GUARDRAILS_SUPPORTED_MODALITIES` enables it, and validating it is then yours.

## Reading a decision afterwards

Each decision emits a `guardrails.decision` span carrying `guardrails.stage`,
`guardrails.policy`, `guardrails.status`, `guardrails.modalities`,
`guardrails.latency_ms`, and `guardrails.failure_code` when something failed.
Tracing is best-effort: if the exporter is missing, checks still run, they just
go unpublished. See [Observability](OBSERVABILITY.md) for reading spans.

### What is never recorded

Request bodies are never logged or traced. Shopper text, assistant replies, and
attachments do not appear in a span, a log line, or an error path. What leaves
the service is a status, a policy name, sanitized category names, a latency,
and a diagnostic code — provider errors are reduced to codes like
`content_safety_check_failed` rather than passed through.

## The service API

`POST /v1/checks` takes a `stage` of `input` or `output` and returns the
decision. The two stages are mutually exclusive by validation: an input check
rejects `assistant_text`, and an output check rejects everything else. Up to 16
prior messages may accompany an input check, so topic control can judge a
follow-up in context rather than in isolation.

The response carries `status`, `stage`, `policy`, `violated_categories`,
`latency_ms`, `diagnostic_code`, `modalities`, and `model_calls` — the last
being a per-judge call count, which is how the UI reports which judge ran and
how you attribute cost.

`GET /health` is liveness. `GET /capabilities` reports the resolved
`input_execution_mode`, since that is decided at startup.

## Where it lives

The service is [`guardrails/src/main.py`](../guardrails/src/main.py) for the
API and [`rails.py`](../guardrails/src/rails.py) for the judges and their
prompts. The chain-server side — when to call, what a failure costs, and what
the shopper is told — is
[`chain_server/src/guardrails.py`](../chain_server/src/guardrails.py).
Deployment settings are catalogued in [Deployment](DEPLOYMENT.md).
