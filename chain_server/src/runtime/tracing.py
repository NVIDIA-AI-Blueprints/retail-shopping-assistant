# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-turn trace spans, and NeMo Relay's lifecycle events beside them."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import sys
from typing import Any

from ..agenttypes import State
from .identity import RequestIdentity

logger = logging.getLogger(__name__)


def _turn_trace_session(identity: RequestIdentity):
    """Bind this turn's spans to its conversation, not to its graph thread.

    The LangChain instrumentor derives a session from LangGraph's ``thread_id``,
    which is ``[conversation_id, request_id]`` -- unique per turn. Left alone,
    a twenty-turn conversation becomes twenty single-turn sessions and the whole
    point of a session view is lost. Binding ``conversation_id`` explicitly is
    what makes a trace session and a durable conversation the same set of turns.

    Returns a null context when tracing is not installed, so this costs nothing
    in a deployment that never exports a span.
    """

    try:
        from openinference.instrumentation import using_attributes
    except Exception:  # noqa: BLE001 - tracing must never break a turn.
        return contextlib.nullcontext()

    try:
        return using_attributes(
            session_id=identity.conversation_id,
            user_id=str(identity.context_user_id),
        )
    except Exception as exc:  # noqa: BLE001 - same.
        logger.warning("Could not bind trace session: %s", type(exc).__name__)
        return contextlib.nullcontext()


@contextlib.contextmanager
def _turn_span(identity: RequestIdentity):
    """Open one span per turn, parenting the graph and the grounding editor.

    The graph and the grounding editor are two separate top-level invocations,
    so without this a turn arrives as two unrelated traces. This also gives the
    finished diagnostics somewhere to hang: every span the instrumentor raises
    has closed by the time the blob settles.
    """

    try:
        from opentelemetry import trace
    except Exception:  # noqa: BLE001 - tracing must never break a turn.
        yield None
        return

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("turn") as span:
        try:
            # The instrumentor's context attributes reach the spans it raises,
            # not one opened by hand, so the root states its own kind and
            # session or it renders as UNKNOWN and sits outside the session.
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("session.id", identity.conversation_id)
            span.set_attribute("conversation.id", identity.conversation_id)
            span.set_attribute("request.id", identity.request_id)
        except Exception:  # noqa: BLE001 # nosec B110 - same.
            pass
        yield span


def _record_turn_diagnostics(span: Any, state: State) -> None:
    """Attach the settled diagnostics to the turn span as metadata.

    Everything rides the one supported ``metadata`` channel rather than a
    private attribute namespace, including the scalar counts -- viewers filter
    on metadata subkeys, so the countables stay filterable without inventing
    names nothing else understands.

    ``partial_graph_messages`` is excluded: it is the largest field by far and
    the span tree already holds what it would say.
    """

    if span is None:
        return
    try:
        diagnostics = dict(getattr(state, "agent_diagnostics", None) or {})
        diagnostics.pop("partial_graph_messages", None)
        tool_calls = diagnostics.get("tool_calls") or []
        payload = {
            "termination_reason": diagnostics.get("final_termination_reason"),
            "tool_calls": len(tool_calls),
            "tool_calls_rejected": len(diagnostics.get("rejected_tool_calls") or []),
            "products_shown": len(diagnostics.get("product_evidence") or []),
            "zero_result_scopes": len(
                diagnostics.get("catalog_scope_outcomes") or []
            ),
            "skills": diagnostics.get("skill_files_read") or [],
            "tools": [call.get("tool_name") for call in tool_calls],
            # Serialised, not nested: trace viewers flatten nested metadata
            # into one attribute per leaf, which buries the useful fields.
            "diagnostics_json": json.dumps(diagnostics, default=str),
        }
        span.set_attribute("metadata", json.dumps(payload, default=str))
    except Exception as exc:  # noqa: BLE001 - diagnostics never break a turn.
        logger.warning("Could not record turn diagnostics: %s", type(exc).__name__)


_RELAY_SUBSCRIBER = "chain-server"


_relay_warnings_said: set[str] = set()


def _relay_warn_once(key: str, message: str, *args: Any) -> None:
    """Say a Relay problem once, not once a turn.

    The agent is rebuilt every turn, so a warning raised on the build path is a
    warning per turn -- and the conditions here (a missing package, a release
    that changed the arguments) are settled at startup and never change.
    """

    if key in _relay_warnings_said:
        return
    _relay_warnings_said.add(key)
    logger.warning(message, *args)


def configure_relay_tracing(config: Any) -> bool:
    """Export NeMo Relay's lifecycle events, or export nothing and say why.

    Relay's middleware emits into an in-process runtime, and nothing leaves the
    service until a subscriber is registered -- so attaching the middleware
    alone is observable to nobody. This is the other half, and it lives beside
    ``_relay_instrumented`` because neither half is any use without the other.

    It speaks ``openinference`` rather than ``gen_ai`` because that is the
    dialect Phoenix reads. Relay carries its own exporter, so this adds a second
    OTLP client to the process, not a second backend.

    ``RELAY_OTLP_ENDPOINT`` is where it sends, falling back to
    ``OTEL_EXPORTER_OTLP_ENDPOINT``. Relay refuses plain HTTP to anything but
    localhost or a loopback address, so under compose it sends to the
    ``relay-collector`` that shares this container's network and forwards to
    ``otel-collector``; run as a local process, the shared endpoint is already
    loopback.

    Returns whether events are being exported, so a caller can say so.
    """

    if not getattr(config, "relay_enabled", False):
        return False

    endpoint = (
        os.environ.get("RELAY_OTLP_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )
    if not endpoint:
        logger.warning(
            "RELAY_ENABLED is set but neither RELAY_OTLP_ENDPOINT nor "
            "OTEL_EXPORTER_OTLP_ENDPOINT is; Relay events have nowhere to go "
            "and will not be exported."
        )
        return False

    try:
        from nemo_relay import OpenTelemetryConfig, OpenTelemetrySubscriber

        relay_config = OpenTelemetryConfig(
            "openinference", f"{endpoint.rstrip('/')}/v1/traces"
        )
        relay_config.service_name = os.environ.get("OTEL_SERVICE_NAME") or "chain-server"
        subscriber = OpenTelemetrySubscriber(relay_config)
        # Registration is global and refuses a duplicate name, so a second
        # runtime would raise and read as "tracing broke" when the first
        # subscriber is still exporting perfectly well.
        subscriber.deregister(_RELAY_SUBSCRIBER)
        subscriber.register(_RELAY_SUBSCRIBER)
        # Spans are batched, so a container stopped without a flush loses the
        # tail of the last conversation -- the part worth reading.
        atexit.register(subscriber.force_flush)
    except ImportError:
        _relay_warn_once(
            "not-installed",
            "RELAY_ENABLED is set but nemo-relay is not installed; "
            "install requirements-relay.txt to trace.",
        )
        return False
    except Exception as exc:  # noqa: BLE001 - tracing must never break startup.
        logger.warning("Could not configure Relay tracing: %s", type(exc).__name__)
        return False

    logger.info("Relay tracing enabled, exporting to %s", endpoint)
    return True


@contextlib.contextmanager
def _relay_turn_scope(config: Any, conversation_id: str):
    """Open one Relay scope around the turn, so its events have somewhere to go.

    Relay does not read OpenTelemetry's context. Its events go into a runtime
    with its own scope stack, so a session set with ``using_attributes`` reaches
    the LangChain spans and never reaches Relay's -- and with no Relay scope
    open, every Relay span is emitted as its own root.

    Both of those are the same absence. One scope here gives Relay's spans a
    parent to nest under and a place to carry the conversation, and it is the
    supported way to do it: no LangGraph internals are touched.

    A no-op when Relay is off or absent, and a no-op when the scope will not
    open -- a trace must never cost a turn.
    """

    if not getattr(config, "relay_enabled", False):
        yield
        return

    try:
        from nemo_relay import ScopeType
        from nemo_relay import scope as relay_scope

        # Not "turn": our own OpenTelemetry span already owns that name, and two
        # different producers emitting a span called "turn" is unreadable.
        opened = relay_scope.scope(
            "relay-turn",
            ScopeType.Agent,
            metadata={"session.id": conversation_id},
        )
        opened.__enter__()
    except ImportError:
        yield
        return
    except Exception as exc:  # noqa: BLE001 - never break a turn for a trace
        _relay_warn_once("scope", "NeMo Relay could not open a turn scope: %s", exc)
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            opened.__exit__(*sys.exc_info())


def _relay_may_observe_but_not_decide(middleware: Any) -> None:
    """Put Relay beside the call rather than inside it.

    Relay's integration hooks LangGraph middleware, so it wraps every model call
    and every tool call: it receives the request, decides what the handler is
    given, and decides what the agent is told came back. That is three chances
    to change an outcome it is only supposed to record, and one of them has
    already been taken -- its tool wrapper re-encoded the arguments and handed
    the tool a copy in which every unset optional had become an explicit null,
    so a search the model sent as ``{"requested_product_type": "dress"}`` arrived
    carrying a price constraint the shopper never gave. This service rejects
    invented constraints, so the tool refused its own call.

    Fixing that one path is not enough, because the position is the problem
    rather than the bug. What follows makes three things true whatever Relay
    does, including raising:

    * the handler is called with the request that arrived, not a copy;
    * it is called **exactly once**, so a wrapper that retries or abandons
      cannot double a cart write or drop one;
    * the agent receives the handler's own return value, not the wrapper's.

    Relay still sees the call and still writes its span -- the handler it is
    given runs the real work at the real moment, so nesting and timings are
    unaffected. It has no say in the result.

    An exception from the wrapper is contained: the work is already done, and the
    agent gets its result. A trace must never cost a turn.
    """

    for name in ("wrap_tool_call", "wrap_model_call"):
        wrapped = getattr(middleware, name, None)
        if wrapped is not None:
            setattr(middleware, name, _observing_only(wrapped, name))

    for name in ("awrap_tool_call", "awrap_model_call"):
        wrapped = getattr(middleware, name, None)
        if wrapped is not None:
            setattr(middleware, name, _observing_only_async(wrapped, name))


def _observing_only(wrapped: Any, name: str) -> Any:
    """The synchronous half of putting Relay beside the call."""

    def observed(request: Any, handler: Any) -> Any:
        done: list[Any] = []

        def run_once(_relays_version: Any = None) -> Any:
            # Relay offers its own copy of the request; the handler answers the
            # one that actually arrived. Called through Relay so its span still
            # wraps the real work.
            if not done:
                done.append(handler(request))
            return done[0]

        try:
            wrapped(request, run_once)
        except Exception as exc:  # noqa: BLE001 - a trace may not cost a turn
            _relay_warn_once(f"raised:{name}", "NeMo Relay %s raised: %s", name, exc)
        if not done:
            # Relay never reached the handler -- it failed early, or chose not
            # to. The call still has to happen.
            done.append(handler(request))
        return done[0]

    return observed


def _observing_only_async(wrapped: Any, name: str) -> Any:
    """The asynchronous half, which is the live path: the agent runs async."""

    async def observed(request: Any, handler: Any) -> Any:
        done: list[Any] = []

        async def run_once(_relays_version: Any = None) -> Any:
            if not done:
                done.append(await handler(request))
            return done[0]

        try:
            await wrapped(request, run_once)
        except Exception as exc:  # noqa: BLE001 - a trace may not cost a turn
            _relay_warn_once(f"raised:{name}", "NeMo Relay %s raised: %s", name, exc)
        if not done:
            done.append(await handler(request))
        return done[0]

    return observed


def _relay_instrumented(
    agent_kwargs: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Add NeMo Relay's instrumentation, or hand back exactly what came in.

    Off unless RELAY_ENABLED says otherwise, and absent when the package is
    not installed. An observability layer that changes behaviour by being
    absent is not observability, so every failure here falls back to the
    untouched arguments and says so once.

    What we hand create_deep_agent carries the parts that make this agent
    correct: middleware is the tool-loop control and the skill gate, backend is
    the skills filesystem, checkpointer is the within-turn state that is
    deleted at every exit. So the wrapper's output is checked rather than
    trusted -- it appends its own middleware to ours, which is the behaviour we
    want, and a release that replaced ours instead would silently remove both
    gates.
    """

    if not getattr(config, "relay_enabled", False):
        return agent_kwargs
    try:
        from nemo_relay.integrations.deepagents import add_nemo_relay_integration
    except ImportError:
        _relay_warn_once(
            "not-installed",
            "RELAY_ENABLED is set but nemo-relay is not installed; "
            "install requirements-relay.txt to trace.",
        )
        return agent_kwargs

    try:
        instrumented = add_nemo_relay_integration(agent_kwargs)
    except Exception as exc:  # pragma: no cover - never break a turn for a trace
        _relay_warn_once("failed", "NeMo Relay instrumentation failed: %s", exc)
        return agent_kwargs

    for name, value in agent_kwargs.items():
        if name not in instrumented:
            _relay_warn_once(
                f"dropped:{name}",
                "NeMo Relay dropped %s from the agent arguments; "
                "running without it.", name
            )
            return agent_kwargs
        if name != "middleware" and instrumented[name] is not value:
            _relay_warn_once(
                f"replaced:{name}",
                "NeMo Relay replaced %s rather than preserving it; "
                "running without it.", name
            )
            return agent_kwargs
    ours = agent_kwargs["middleware"]
    for added in list(instrumented["middleware"])[len(ours):]:
        _relay_may_observe_but_not_decide(added)
    if list(instrumented["middleware"])[: len(ours)] != list(ours):
        _relay_warn_once(
            "middleware-order",
            "NeMo Relay did not preserve our middleware order; "
            "running without it.",
        )
        return agent_kwargs
    return instrumented
