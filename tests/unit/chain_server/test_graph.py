# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.graph``.

The graph module ships two kinds of artifacts:

* ``GraphNodes`` and ``GraphRouting`` - small async callables that the
  LangGraph runtime invokes for each node/edge. They are pure Python except
  for a few HTTP calls, which we stub.
* ``create_graph`` - wires everything together and compiles a LangGraph. We
  smoke-test it to make sure the factory does not raise and returns a
  compiled runnable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import requests

from chain_server.src import graph as graph_mod
from chain_server.src.agenttypes import Cart, Rail, State
from chain_server.src.graph import GraphNodes, GraphRouting, create_graph


@dataclass
class _HttpRecorder:
    gets: List[Dict[str, Any]] = field(default_factory=list)
    posts: List[Dict[str, Any]] = field(default_factory=list)


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def install_config(monkeypatch: pytest.MonkeyPatch, base_config) -> SimpleNamespace:
    """Install a known config into the graph module's private global.

    The module stores config as a module-level ``_config`` that is populated
    by ``create_graph``. Unit tests that call ``GraphNodes`` directly must
    set this explicitly.
    """
    monkeypatch.setattr(graph_mod, "_config", base_config)
    return base_config


@pytest.fixture
def http_recorder(monkeypatch: pytest.MonkeyPatch) -> _HttpRecorder:
    recorder = _HttpRecorder()

    def _fake_get(url: str, timeout: int = 10, **_: Any):
        recorder.gets.append({"url": url, "timeout": timeout})
        if url.endswith("/context"):
            return _FakeResponse({"context": "prior chat"})
        if url.endswith("/cart"):
            return _FakeResponse(
                {"cart": [{"item": "Silk Dress", "amount": 1, "price": 49.99}]}
            )
        return _FakeResponse({})

    def _fake_post(url: str, json: Dict[str, Any], timeout: int = 10, **_: Any):
        recorder.posts.append({"url": url, "json": json, "timeout": timeout})
        if url.endswith("/rail/input/check"):
            # Echo query back → is_safe True.
            return _FakeResponse(
                {"response": [{"role": "assistant", "content": json["query"]}]}
            )
        if url.endswith("/rail/output/check"):
            return _FakeResponse(
                {"response": [{"role": "assistant", "content": json["query"]}]}
            )
        return _FakeResponse({})

    monkeypatch.setattr(graph_mod.requests, "get", _fake_get)
    monkeypatch.setattr(graph_mod.requests, "post", _fake_post)
    return recorder


# ---------------------------------------------------------------------------
# GraphNodes.get_memory
# ---------------------------------------------------------------------------


class TestGetMemory:
    async def test_populates_context_and_cart_on_success(
        self, install_config, http_recorder: _HttpRecorder
    ) -> None:
        state = State(user_id=42, query="hi")

        result = await GraphNodes.get_memory(state)

        assert result.context == "prior chat"
        assert result.cart.contents == [
            {"item": "Silk Dress", "amount": 1, "price": 49.99}
        ]
        assert "memory" in result.timings

        urls = [call["url"] for call in http_recorder.gets]
        assert any(u.endswith("/user/42/context") for u in urls)
        assert any(u.endswith("/user/42/cart") for u in urls)

    async def test_http_error_leaves_empty_context_and_cart(
        self,
        install_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_: Any, **__: Any) -> None:
            raise requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(graph_mod.requests, "get", _boom)

        state = State(user_id=7, query="hi")
        result = await GraphNodes.get_memory(state)

        assert result.context == ""
        assert result.cart.contents == []
        assert "memory" in result.timings


# ---------------------------------------------------------------------------
# Rails nodes
# ---------------------------------------------------------------------------


class TestRailsNodes:
    async def test_input_check_returns_safe_when_content_matches(
        self, install_config, http_recorder: _HttpRecorder
    ) -> None:
        state = State(user_id=1, query="what's for dinner")

        result = await GraphNodes.check_input_safety(state)

        assert result["is_safe"] is True
        assert "rails_input_check" in result["rail_timings"]

    async def test_input_check_flags_mismatched_response_as_unsafe(
        self,
        install_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _post_unsafe(url: str, json: Dict[str, Any], timeout: int = 10, **_: Any):
            return _FakeResponse(
                {"response": [{"role": "assistant", "content": "BLOCKED"}]}
            )

        monkeypatch.setattr(graph_mod.requests, "post", _post_unsafe)
        state = State(user_id=1, query="what's for dinner")

        result = await GraphNodes.check_input_safety(state)

        assert result["is_safe"] is False

    async def test_input_check_bypassed_when_guardrails_disabled(
        self, install_config
    ) -> None:
        state = State(user_id=1, query="q", guardrails=False)

        result = await GraphNodes.check_input_safety(state)

        assert result == {"is_safe": True}

    async def test_input_check_defaults_to_safe_on_http_error(
        self, install_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_: Any, **__: Any) -> None:
            raise requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(graph_mod.requests, "post", _boom)
        state = State(user_id=1, query="q")

        result = await GraphNodes.check_input_safety(state)

        assert result["is_safe"] is True
        assert "rails_input_check" in result["rail_timings"]

    async def test_output_check_matches_response(
        self, install_config, http_recorder: _HttpRecorder
    ) -> None:
        state = State(user_id=1, query="q", response="safe reply")

        result = await GraphNodes.check_output_safety(state)

        assert result["is_safe"] is True

    async def test_output_check_missing_response_structure_defaults_safe(
        self, install_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _post(url: str, json: Dict[str, Any], timeout: int = 10, **_: Any):
            return _FakeResponse({})

        monkeypatch.setattr(graph_mod.requests, "post", _post)
        state = State(user_id=1, query="q", response="anything")

        result = await GraphNodes.check_output_safety(state)

        assert result["is_safe"] is True


# ---------------------------------------------------------------------------
# Check-rail / unsafe-output nodes
# ---------------------------------------------------------------------------


class TestCheckRailAndUnsafe:
    async def test_check_rail_node_returns_timings_dict(self) -> None:
        rail = Rail(is_safe=False, rail_timings={"rails_input_check": 0.25})

        result = await GraphNodes.check_rail_node(rail)

        assert result == {"timings": {"rails_input_check": 0.25}}

    async def test_unsafe_output_streams_safe_message(
        self,
        install_config,
        stream_writer_capture: list,
    ) -> None:
        rail = Rail(is_safe=False)

        result = await GraphNodes.unsafe_output(rail)

        assert result["response"] == install_config.unsafe_message
        # Exactly one payload frame should have been emitted.
        assert len(stream_writer_capture) == 1
        payload = json.loads(stream_writer_capture[0])
        assert payload["type"] == "content"
        assert payload["payload"] == install_config.unsafe_message
        assert "timestamp" in payload


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestGraphRouting:
    @pytest.mark.parametrize(
        "is_safe,expected",
        [(True, "chatter_node"), (False, "unsafe_output")],
    )
    def test_decide_if_input_safe(self, is_safe: bool, expected: str) -> None:
        assert GraphRouting.decide_if_input_safe(Rail(is_safe=is_safe)) == expected

    @pytest.mark.parametrize(
        "is_safe,expected",
        [(True, "summarize_node"), (False, "unsafe_output")],
    )
    def test_decide_if_output_safe(self, is_safe: bool, expected: str) -> None:
        assert GraphRouting.decide_if_output_safe(Rail(is_safe=is_safe)) == expected


# ---------------------------------------------------------------------------
# create_graph smoke test
# ---------------------------------------------------------------------------


class TestCreateGraphSmoke:
    def test_create_graph_compiles_without_error(self, base_config) -> None:
        # Any object with ``invoke`` / ``decide_function`` suffices — LangGraph
        # only inspects the callables, not their types.
        class _Noop:
            async def invoke(self, state: State, verbose: bool = False) -> State:
                return state

            def decide_function(self, state: State) -> str:
                return "chatter"

        compiled = create_graph(
            cart_agent=_Noop(),
            retriever_agent=_Noop(),
            planner_agent=_Noop(),
            chatter_agent=_Noop(),
            summary_agent=_Noop(),
            config=base_config,
        )

        # A compiled LangGraph exposes ``invoke``/``astream`` among others; we
        # only need to prove the factory produced a runnable callable.
        assert callable(getattr(compiled, "ainvoke", None)) or callable(
            getattr(compiled, "astream", None)
        )
        assert graph_mod._config is base_config
