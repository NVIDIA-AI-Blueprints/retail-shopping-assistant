# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tool receives what the model sent, whether or not it is being traced.

Relay's tool wrapper does not hand the model's arguments to the tool. It encodes
them for the span with ``BestEffortAnyCodec`` and passes the decoded copy
onward, and the copy is not the original: the codec tries ``model_dump()``
first, which materialises every optional field that was never set. A search sent
as ``{"requested_product_type": "dress"}`` arrives as
``{"requested_product_type": "dress", "price": {"min": null, "max": null}}`` --
a constraint the shopper never gave.

This service rejects invented constraints deliberately, so the tool refused its
own call and the turn answered "I couldn't complete a valid catalog search".
Measured on J01: two failures with tracing on, five passes with it off, on the
same image, differing only by RELAY_ENABLED.

The tests are about the property, not the workaround: tracing may observe a tool
call and may not decide what the tool receives.
"""

from __future__ import annotations

from typing import Any

import pytest

from chain_server.src.deepagents_runtime import _relay_must_not_rewrite_arguments


class _Request:
    """Enough of a ToolCallRequest to see which arguments arrive."""

    def __init__(self, args: dict[str, Any]) -> None:
        self.tool_call = {"name": "search_catalog_tool", "args": args}

    def override(self, tool_call: dict[str, Any]) -> "_Request":
        replacement = _Request({})
        replacement.tool_call = tool_call
        return replacement


class _RelayMiddleware:
    """Relay's wrapper, in the shape that matters: it rewrites the arguments."""

    def __init__(self) -> None:
        self.traced: list[dict[str, Any]] = []

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        # What Relay does: record the call, then hand the handler a re-encoded
        # copy in which unset optionals have become explicit nulls.
        self.traced.append(dict(request.tool_call["args"]))
        rewritten = {**request.tool_call["args"], "price": {"min": None, "max": None}}
        return handler(request.override(tool_call={**request.tool_call, "args": rewritten}))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        # Its own body, not a call to the sync one. Relay's is independent, and
        # a fake that delegates would be patched for free -- hiding a fix that
        # only ever touched the sync wrapper, which is not the live path.
        self.traced.append(dict(request.tool_call["args"]))
        rewritten = {**request.tool_call["args"], "price": {"min": None, "max": None}}
        return handler(request.override(tool_call={**request.tool_call, "args": rewritten}))


def test_the_tool_receives_the_arguments_the_model_sent() -> None:
    middleware = _RelayMiddleware()
    _relay_must_not_rewrite_arguments(middleware)
    seen: dict[str, Any] = {}

    def handler(request: Any) -> str:
        seen.update(request.tool_call["args"])
        return "ok"

    middleware.wrap_tool_call(_Request({"requested_product_type": "dress"}), handler)

    assert seen == {"requested_product_type": "dress"}


def test_a_constraint_the_shopper_never_gave_never_reaches_the_tool() -> None:
    """The defect, named as itself.

    An invented `price` is not a harmless extra field here: constraints are
    validated against what the shopper actually asked for, so the tool rejects
    the call and the turn says it could not search.
    """

    middleware = _RelayMiddleware()
    _relay_must_not_rewrite_arguments(middleware)
    seen: dict[str, Any] = {}

    def handler(request: Any) -> str:
        seen.update(request.tool_call["args"])
        return "ok"

    middleware.wrap_tool_call(_Request({"requested_product_type": "dress"}), handler)

    assert "price" not in seen


def test_the_trace_still_sees_the_call() -> None:
    """Preserving the arguments must not cost the observability.

    A fix that stopped Relay recording the call would trade one silent failure
    for another.
    """

    middleware = _RelayMiddleware()
    _relay_must_not_rewrite_arguments(middleware)

    middleware.wrap_tool_call(_Request({"requested_product_type": "dress"}), lambda r: "ok")

    assert middleware.traced == [{"requested_product_type": "dress"}]


def test_the_tool_result_is_returned_unchanged() -> None:
    middleware = _RelayMiddleware()
    _relay_must_not_rewrite_arguments(middleware)

    sentinel = object()
    result = middleware.wrap_tool_call(_Request({"a": 1}), lambda r: sentinel)

    assert result is sentinel


@pytest.mark.asyncio
async def test_the_async_wrapper_is_covered_too() -> None:
    """Both are patched: the agent runs async, so the async one is the live path.

    Patching only the sync wrapper would leave the defect exactly where it
    happens and pass every synchronous test.
    """

    middleware = _RelayMiddleware()
    _relay_must_not_rewrite_arguments(middleware)
    seen: dict[str, Any] = {}

    def handler(request: Any) -> str:
        seen.update(request.tool_call["args"])
        return "ok"

    await middleware.awrap_tool_call(_Request({"requested_product_type": "dress"}), handler)

    assert seen == {"requested_product_type": "dress"}


def test_middleware_without_tool_wrappers_is_left_alone() -> None:
    """Relay may add middleware that does not wrap tools; it must not break."""

    class _Bare:
        pass

    bare = _Bare()
    _relay_must_not_rewrite_arguments(bare)  # must not raise

    assert not hasattr(bare, "wrap_tool_call")


def test_the_middleware_relay_adds_is_actually_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, not just the helper.

    Everything above tests `_relay_must_not_rewrite_arguments` in isolation and
    would keep passing if nothing ever called it. This drives
    `_relay_instrumented`, which is the only place that does.
    """

    import sys
    import types

    from chain_server.src import deepagents_runtime as runtime

    added = _RelayMiddleware()

    def add_nemo_relay_integration(kwargs):
        return {**kwargs, "middleware": [*kwargs["middleware"], added]}

    module = types.ModuleType("nemo_relay.integrations.deepagents")
    module.add_nemo_relay_integration = add_nemo_relay_integration
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

    # Ours still first and unreplaced -- the PR's existing guarantee.
    assert instrumented["middleware"][0] is ours

    seen: dict[str, Any] = {}
    instrumented["middleware"][1].wrap_tool_call(
        _Request({"requested_product_type": "dress"}),
        lambda request: seen.update(request.tool_call["args"]),
    )

    assert seen == {"requested_product_type": "dress"}
