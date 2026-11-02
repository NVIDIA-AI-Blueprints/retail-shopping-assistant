# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Repository-wide pytest configuration for the offline unit test suite.

Goals
-----
- Make service source packages importable without bringing up Docker or any
  live backend. Each service lives in ``<service>/src/`` and is importable as
  ``<service>.src.<module>`` once the repo root is on ``sys.path``.
- Provide shared fixtures (``base_config``, ``mocked_openai`` etc.) that all
  suites can depend on without knowing service-specific wiring.
- Ensure every service module that reads API keys at import time finds
  harmless placeholder values so tests can import them directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import pytest

# Ensure the repo root is importable so tests can `import chain_server.src...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Several service modules read API keys at import time. Set harmless defaults
# before any test imports them so we never trip on KeyError during collection.
for _key in ("LLM_API_KEY", "EMBED_API_KEY", "RAIL_API_KEY", "NVIDIA_API_KEY"):
    os.environ.setdefault(_key, f"test-{_key.lower()}")


@pytest.fixture
def base_config() -> SimpleNamespace:
    """Return a valid chain-server-shaped config object.

    Agent classes in ``chain_server`` accept any object with matching
    attributes; using ``SimpleNamespace`` keeps the fixture decoupled from
    the concrete ``ChainServerConfig`` pydantic model so we can test agents
    in isolation.
    """
    return SimpleNamespace(
        llm_port="http://localhost:8000/v1",
        llm_name="test-model",
        retriever_port="http://localhost:8010",
        memory_port="http://localhost:8011",
        rails_port="http://localhost:8012",
        routing_prompt="You are a routing assistant.",
        chatter_prompt="You are a helpful shopping assistant.",
        categories=[
            "bag",
            "sunglasses",
            "dress",
            "shoes",
            "top blouse sweater",
        ],
        agent_choices=["cart", "retriever", "chatter"],
        memory_length=16384,
        top_k_retrieve=4,
        multimodal=True,
        unsafe_message="Sorry, I can only help with shopping questions.",
    )


@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    """Dict counterpart of :func:`base_config` for testing pydantic validation."""
    return {
        "llm_port": "http://localhost:8000/v1",
        "llm_name": "test-model",
        "retriever_port": "http://localhost:8010",
        "memory_port": "http://localhost:8011",
        "rails_port": "http://localhost:8012",
        "routing_prompt": "You are a routing assistant.",
        "chatter_prompt": "You are a helpful shopping assistant.",
        "categories": ["bag", "shoes"],
        "agent_choices": ["cart", "retriever", "chatter"],
        "memory_length": 16384,
        "top_k_retrieve": 4,
        "multimodal": True,
        "unsafe_message": "Sorry, I can only help with shopping questions.",
    }


class FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by service tests.

    Providing a tiny shim here (rather than in each test file) keeps the
    service tests readable and avoids subtle coupling to the ``requests``
    implementation. Supports the subset of the interface exercised by the
    chain-server agents: ``status_code``, ``text``, ``json``,
    ``raise_for_status``.
    """

    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
        text: str | None = None,
    ) -> None:
        self._json = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text if text is not None else __import__("json").dumps(self._json)

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise __import__("requests").exceptions.HTTPError(
                f"HTTP {self.status_code}"
            )


@pytest.fixture
def fake_response_cls() -> type:
    """Expose ``FakeResponse`` to tests as a fixture-friendly handle."""
    return FakeResponse


@pytest.fixture
def make_openai_chat_response():
    """Factory for a fake OpenAI chat completion response.

    The OpenAI client used across agents consumes
    ``response.choices[0].message`` (content + optional tool_calls). This
    fixture builds that nested structure from primitive inputs so tests don't
    repeat the ceremony.
    """

    def _build(
        content: str | None = None,
        tool_name: str | None = None,
        tool_arguments: str | None = None,
    ) -> SimpleNamespace:
        tool_calls = None
        if tool_name is not None:
            tool_calls = [
                SimpleNamespace(
                    function=SimpleNamespace(
                        name=tool_name,
                        arguments=tool_arguments or "{}",
                    )
                )
            ]
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return _build


@pytest.fixture
def stream_writer_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Capture LangGraph stream writer payloads emitted by agent ``invoke``.

    Several chain-server agents call ``get_stream_writer()`` and push events
    through it. For unit tests we intercept the factory and redirect writes
    into a plain list so assertions can inspect what the agent streamed
    without standing up the real LangGraph runtime.
    """
    from chain_server.src import chatter as chatter_mod
    from chain_server.src import graph as graph_mod

    captured: list[str] = []

    def _fake_writer() -> Any:
        def _write(payload: str) -> None:
            captured.append(payload)

        return _write

    monkeypatch.setattr(chatter_mod, "get_stream_writer", _fake_writer)
    monkeypatch.setattr(graph_mod, "get_stream_writer", _fake_writer)
    yield captured
