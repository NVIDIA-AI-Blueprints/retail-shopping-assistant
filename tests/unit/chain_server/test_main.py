# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.main``.

The module does expensive work at import time: it calls ``load_config``,
constructs every agent, and compiles a LangGraph. For unit tests we replace
each of those with lightweight stubs before importing the module.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from chain_server.src.agenttypes import Cart, State


class _NoopAgent:
    """Constructor-only stub used for every agent class under test."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.invoked_with: List[Dict[str, Any]] = []

    async def invoke(self, state: State, verbose: bool = False) -> State:
        self.invoked_with.append({"state": state, "verbose": verbose})
        return state

    def decide_function(self, state: State) -> str:
        return "chatter"


class _StubCompiledGraph:
    """Replacement for the compiled LangGraph runnable."""

    def __init__(self, response_text: str = "ok") -> None:
        self.response_text = response_text
        self.astream_calls: List[Any] = []
        self.ainvoke_calls: List[Any] = []

    async def astream(self, state: State, stream_mode: str = "custom"):
        self.astream_calls.append((state, stream_mode))
        for piece in ["hello ", "world"]:
            yield piece

    async def ainvoke(self, state: State) -> Dict[str, Any]:
        self.ainvoke_calls.append(state)
        return {
            "response": self.response_text,
            "timings": {"chatter": 0.1, "memory": 0.01},
        }


@pytest.fixture
def main_module(
    monkeypatch: pytest.MonkeyPatch, base_config
) -> Iterator[Any]:
    """Import ``chain_server.src.main`` with all heavy deps stubbed."""
    from chain_server.src import cart as cart_mod
    from chain_server.src import chatter as chatter_mod
    from chain_server.src import config as config_mod
    from chain_server.src import graph as graph_mod
    from chain_server.src import planner as planner_mod
    from chain_server.src import retriever as retriever_mod
    from chain_server.src import summarizer as summarizer_mod

    # Config loader returns our pre-baked config rather than reading YAML.
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: base_config)

    # Every agent class replaced with a noop stub.
    monkeypatch.setattr(cart_mod, "CartAgent", _NoopAgent)
    monkeypatch.setattr(chatter_mod, "ChatterAgent", _NoopAgent)
    monkeypatch.setattr(planner_mod, "PlannerAgent", _NoopAgent)
    monkeypatch.setattr(retriever_mod, "RetrieverAgent", _NoopAgent)
    monkeypatch.setattr(summarizer_mod, "SummaryAgent", _NoopAgent)

    compiled = _StubCompiledGraph()
    monkeypatch.setattr(graph_mod, "create_graph", lambda **_: compiled)

    # Force a fresh import so our stubs are actually used.
    sys.modules.pop("chain_server.src.main", None)
    main_module = importlib.import_module("chain_server.src.main")
    main_module._test_compiled = compiled  # type: ignore[attr-defined]

    yield main_module

    sys.modules.pop("chain_server.src.main", None)


@pytest.fixture
def client(main_module) -> TestClient:
    return TestClient(main_module.app)


# ---------------------------------------------------------------------------
# create_initial_state
# ---------------------------------------------------------------------------


class TestCreateInitialState:
    def test_defaults_fill_empty_strings_and_empty_cart(
        self, main_module
    ) -> None:
        request = main_module.QueryRequest(user_id=1, query="hi")
        state = main_module.create_initial_state(request)

        assert state.user_id == 1
        assert state.query == "hi"
        assert state.context == ""
        assert state.image == ""
        assert isinstance(state.cart, Cart)
        assert state.cart.is_empty()
        assert state.guardrails is True

    def test_cart_passthrough(self, main_module) -> None:
        cart = Cart(contents=[{"item": "X", "amount": 2, "price": 9.99}])
        request = main_module.QueryRequest(user_id=1, query="hi", cart=cart)
        state = main_module.create_initial_state(request)

        assert state.cart.contents == cart.contents

    def test_none_context_becomes_empty(self, main_module) -> None:
        request = main_module.QueryRequest(user_id=1, query="hi", context=None)
        state = main_module.create_initial_state(request)
        assert state.context == ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TestHealthAndRoot:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"] == "1.0.0"

    def test_root_describes_endpoints(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

        body = response.json()
        assert body["message"] == "Shopping Assistant API"
        assert body["version"] == "1.0.0"
        for key in ["query", "stream", "timing", "health", "docs"]:
            assert key in body["endpoints"]


class TestTimingEndpoint:
    def test_returns_response_and_timings(
        self, main_module, client: TestClient
    ) -> None:
        response = client.post(
            "/query/timing",
            json={"user_id": 1, "query": "hello"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == main_module._test_compiled.response_text
        assert "total" in body["timings"]
        assert body["timings"]["total"] > 0


class TestStreamEndpoint:
    def test_stream_returns_sse_body_with_done_marker(
        self, main_module, client: TestClient
    ) -> None:
        with client.stream(
            "POST",
            "/query/stream",
            json={"user_id": 1, "query": "hi"},
        ) as stream_response:
            assert stream_response.status_code == 200
            chunks: List[str] = []
            for line in stream_response.iter_lines():
                if line:
                    chunks.append(line)

        joined = "\n".join(chunks)
        assert "data: hello " in joined
        assert "data: world" in joined
        assert "[DONE]" in joined

    def test_image_only_query_populates_placeholder(
        self, main_module, client: TestClient
    ) -> None:
        # Image-only requests should get a placeholder query injected so that
        # the graph has something to work with.
        compiled = main_module._test_compiled
        compiled.astream_calls.clear()

        with client.stream(
            "POST",
            "/query/stream",
            json={"user_id": 1, "query": "", "image": "data:image/jpeg;base64,AAA"},
        ) as stream_response:
            # Drain the stream so the generator actually runs.
            for _ in stream_response.iter_lines():
                pass

        assert compiled.astream_calls
        state_arg, _ = compiled.astream_calls[-1]
        assert state_arg.image.startswith("data:image/jpeg")
        assert "image" in state_arg.query.lower()


class TestValidation:
    def test_missing_user_id_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/query/timing",
            json={"query": "hi"},
        )
        assert response.status_code == 422

    def test_bad_payload_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/query/timing",
            json={"user_id": "not-an-int", "query": "hi"},
        )
        assert response.status_code == 422
