# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay cannot change a turn, whatever it does.

Its integration hooks LangGraph middleware, so it wraps every model call and
every tool call: it receives the request, decides what the handler is given, and
decides what the agent is told came back. Three chances to change an outcome it
is only supposed to record.

One of them was already taken. Relay's tool wrapper re-encoded the arguments and
handed the tool a copy in which every unset optional had become an explicit
null, so a search sent as ``{"requested_product_type": "dress"}`` arrived
carrying a price constraint the shopper never gave, the tool refused its own
call, and the turn answered "I couldn't complete a valid catalog search".

These tests are not about that bug. They are about the position being the
problem, so each one takes a different way a wrapper could change an outcome --
rewriting the request, rewriting the result, calling twice, not calling at all,
raising -- and asserts the turn is unaffected.
"""

from __future__ import annotations

from typing import Any

import pytest

from chain_server.src.deepagents_runtime import _relay_may_observe_but_not_decide


class _Request:
    def __init__(self, args: dict[str, Any]) -> None:
        self.tool_call = {"name": "search_catalog_tool", "args": args}

    def override(self, tool_call: dict[str, Any]) -> "_Request":
        replacement = _Request({})
        replacement.tool_call = tool_call
        return replacement


def _guarded(behaviour: Any, *, attr: str = "wrap_tool_call") -> Any:
    """A middleware whose wrapper misbehaves in one specific way."""

    middleware = type("_M", (), {attr: staticmethod(behaviour)})()
    _relay_may_observe_but_not_decide(middleware)
    return middleware


ORIGINAL = {"requested_product_type": "dress"}


def test_a_wrapper_that_rewrites_the_request_does_not_reach_the_tool() -> None:
    """The bug that happened, kept as one case among several."""

    def rewrites(request: Any, handler: Any) -> Any:
        poisoned = {**request.tool_call["args"], "price": {"min": None, "max": None}}
        return handler(request.override(tool_call={**request.tool_call, "args": poisoned}))

    seen: dict[str, Any] = {}
    _guarded(rewrites).wrap_tool_call(
        _Request(ORIGINAL), lambda r: seen.update(r.tool_call["args"])
    )

    assert seen == ORIGINAL


def test_a_wrapper_that_rewrites_the_result_does_not_reach_the_agent() -> None:
    """A tool that added to the cart must not be reported as one that did not."""

    def rewrites_result(request: Any, handler: Any) -> Any:
        handler(request)
        return "something else entirely"

    truth = object()
    result = _guarded(rewrites_result).wrap_tool_call(_Request(ORIGINAL), lambda r: truth)

    assert result is truth


def test_a_wrapper_that_calls_twice_runs_the_tool_once() -> None:
    """Doubling a cart write is worse than losing a trace."""

    def calls_twice(request: Any, handler: Any) -> Any:
        handler(request)
        return handler(request)

    calls = []
    _guarded(calls_twice).wrap_tool_call(_Request(ORIGINAL), lambda r: calls.append(1))

    assert len(calls) == 1


def test_a_wrapper_that_never_calls_the_handler_still_runs_the_tool() -> None:
    """The work has to happen whether or not the trace wants it to."""

    def never_calls(request: Any, handler: Any) -> Any:
        return "traced, but nothing ran"

    calls = []
    result = _guarded(never_calls).wrap_tool_call(
        _Request(ORIGINAL), lambda r: calls.append(1) or "real"
    )

    assert len(calls) == 1
    assert result == "real"


def test_a_wrapper_that_raises_does_not_cost_the_turn() -> None:
    """A trace must never be the reason a shopper's turn fails."""

    def explodes(request: Any, handler: Any) -> Any:
        handler(request)
        raise RuntimeError("exporter unreachable")

    result = _guarded(explodes).wrap_tool_call(_Request(ORIGINAL), lambda r: "real")

    assert result == "real"


def test_a_wrapper_that_raises_before_calling_still_runs_the_tool() -> None:
    def explodes_early(request: Any, handler: Any) -> Any:
        raise RuntimeError("failed before it started")

    calls = []
    result = _guarded(explodes_early).wrap_tool_call(
        _Request(ORIGINAL), lambda r: calls.append(1) or "real"
    )

    assert len(calls) == 1
    assert result == "real"


def test_relay_still_sees_the_call() -> None:
    """Containing it must not silence it, or this trades one blindness for another."""

    traced = []

    def records(request: Any, handler: Any) -> Any:
        traced.append(dict(request.tool_call["args"]))
        return handler(request)

    _guarded(records).wrap_tool_call(_Request(ORIGINAL), lambda r: "real")

    assert traced == [ORIGINAL]


def test_the_real_work_runs_inside_the_wrapper() -> None:
    """So the span still covers it, and timings still mean something.

    Calling the handler outside Relay's wrapper would be simpler and would
    reduce the trace to a note that a call happened, somewhere, for some time.
    """

    order = []

    def records(request: Any, handler: Any) -> Any:
        order.append("span opened")
        handler(request)
        order.append("span closed")

    _guarded(records).wrap_tool_call(_Request(ORIGINAL), lambda r: order.append("tool ran"))

    assert order == ["span opened", "tool ran", "span closed"]


def test_model_calls_are_guarded_too() -> None:
    """Relay wraps the model as well; the same three chances apply there."""

    def rewrites_result(request: Any, handler: Any) -> Any:
        handler(request)
        return "a completion the model never produced"

    truth = object()
    guarded = _guarded(rewrites_result, attr="wrap_model_call")

    assert guarded.wrap_model_call(_Request(ORIGINAL), lambda r: truth) is truth


@pytest.mark.parametrize("attr", ["awrap_tool_call", "awrap_model_call"])
@pytest.mark.asyncio
async def test_the_async_wrappers_are_guarded(attr: str) -> None:
    """The live path: the agent runs async, so these are the ones that run.

    Guarding only the synchronous wrappers would leave the defect exactly where
    it happens and pass every synchronous test above.
    """

    async def misbehaves(request: Any, handler: Any) -> Any:
        poisoned = {**request.tool_call["args"], "price": {"min": None}}
        await handler(request.override(tool_call={**request.tool_call, "args": poisoned}))
        return "a result the handler never returned"

    seen: dict[str, Any] = {}

    async def handler(request: Any) -> str:
        seen.update(request.tool_call["args"])
        return "real"

    guarded = _guarded(misbehaves, attr=attr)
    result = await getattr(guarded, attr)(_Request(ORIGINAL), handler)

    assert seen == ORIGINAL
    assert result == "real"


def test_middleware_without_wrappers_is_left_alone() -> None:
    """Relay may add middleware that wraps nothing; it must not break."""

    class _Bare:
        pass

    bare = _Bare()
    _relay_may_observe_but_not_decide(bare)  # must not raise

    assert not hasattr(bare, "wrap_tool_call")


def test_the_middleware_relay_adds_is_actually_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, not just the guard.

    Everything above would keep passing if nothing ever called
    `_relay_may_observe_but_not_decide`. This drives `_relay_instrumented`,
    which is the only place that does.
    """

    import sys
    import types

    from chain_server.src import deepagents_runtime as runtime

    class _Added:
        @staticmethod
        def wrap_tool_call(request: Any, handler: Any) -> Any:
            handler(request)
            return "relay's answer, not the tool's"

    added = _Added()

    module = types.ModuleType("nemo_relay.integrations.deepagents")
    module.add_nemo_relay_integration = lambda kwargs: {
        **kwargs, "middleware": [*kwargs["middleware"], added]
    }
    monkeypatch.setitem(sys.modules, "nemo_relay", types.ModuleType("nemo_relay"))
    monkeypatch.setitem(
        sys.modules, "nemo_relay.integrations", types.ModuleType("nemo_relay.integrations")
    )
    monkeypatch.setitem(sys.modules, "nemo_relay.integrations.deepagents", module)

    ours = object()
    instrumented = runtime._relay_instrumented(
        {"middleware": [ours], "backend": object(), "checkpointer": object()},
        types.SimpleNamespace(relay_enabled=True),
    )

    assert instrumented["middleware"][0] is ours

    truth = object()
    result = instrumented["middleware"][1].wrap_tool_call(
        _Request(ORIGINAL), lambda request: truth
    )

    assert result is truth
