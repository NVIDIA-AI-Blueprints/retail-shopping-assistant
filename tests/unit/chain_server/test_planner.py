# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.planner``.

The planner agent is a thin wrapper over an LLM call with normalization and
defaulting. We pin the contract of each public surface without making any
real network requests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chain_server.src import planner as planner_mod
from chain_server.src.agenttypes import State
from chain_server.src.planner import PlannerAgent


@pytest.fixture
def planner_agent(base_config, monkeypatch: pytest.MonkeyPatch) -> PlannerAgent:
    """Construct a PlannerAgent with the OpenAI client stubbed out."""

    class _FakeOpenAI:
        def __init__(self, *_, **__) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: None)
            )

    monkeypatch.setattr(planner_mod, "OpenAI", _FakeOpenAI)
    return PlannerAgent(config=base_config)


def _stub_llm(planner_agent: PlannerAgent, raw_content: str) -> None:
    """Swap the planner's chat completion stub to return ``raw_content``."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw_content))]
    )
    planner_agent.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )


class TestInitialization:
    def test_constructor_copies_config_fields(self, planner_agent: PlannerAgent, base_config) -> None:
        assert planner_agent.llm_name == base_config.llm_name
        assert planner_agent.llm_port == base_config.llm_port
        assert planner_agent.agent_choices == base_config.agent_choices
        assert planner_agent.system_prompt == base_config.routing_prompt

    def test_openai_client_constructor_failure_is_surfaced(
        self, base_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom:
            def __init__(self, *_, **__) -> None:
                raise RuntimeError("boom")

        monkeypatch.setattr(planner_mod, "OpenAI", _Boom)
        with pytest.raises(RuntimeError):
            PlannerAgent(config=base_config)


class TestCreateRoutingMessages:
    def test_message_shape(self, planner_agent: PlannerAgent) -> None:
        messages = planner_agent._create_routing_messages("find a bag")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == planner_agent.system_prompt
        assert messages[1]["role"] == "user"
        assert "find a bag" in messages[1]["content"]


class TestCallLlmForRouting:
    def test_returns_lowercased_trimmed_content(self, planner_agent: PlannerAgent) -> None:
        _stub_llm(planner_agent, "  RETRIEVER  \n")
        assert planner_agent._call_llm_for_routing("anything") == "retriever"

    def test_defaults_to_chatter_on_exception(
        self, planner_agent: PlannerAgent
    ) -> None:
        def _raise(**_: Any) -> Any:
            raise RuntimeError("llm down")

        planner_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_raise))
        )

        assert planner_agent._call_llm_for_routing("anything") == "chatter"


class TestNormalizeAgentName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("cart", "cart"),
            ("retriever", "retriever"),
            ("chatter", "chatter"),
            ("search", "retriever"),  # mapping alias
            ("cart_node", "cart"),
            ("product_finder", "retriever"),
            ("general", "chatter"),
            ("assistant", "chatter"),
        ],
    )
    def test_known_names_and_aliases(
        self, planner_agent: PlannerAgent, raw: str, expected: str
    ) -> None:
        assert planner_agent._normalize_agent_name(raw) == expected

    def test_unknown_name_falls_back_to_chatter(
        self, planner_agent: PlannerAgent
    ) -> None:
        assert planner_agent._normalize_agent_name("weather_bot") == "chatter"


class TestInvoke:
    def test_routes_query_via_llm_and_normalizes(
        self, planner_agent: PlannerAgent
    ) -> None:
        _stub_llm(planner_agent, "search")  # alias for retriever
        state = State(user_id=1, query="show me dresses")

        out = planner_agent.invoke(state, verbose=False)

        assert out.next_agent == "retriever"
        assert "planner" in out.timings

    def test_image_only_query_bypasses_llm_and_routes_to_retriever(
        self, planner_agent: PlannerAgent
    ) -> None:
        def _explode(**_: Any) -> Any:  # any LLM call would be a bug here
            raise AssertionError("LLM must not be called for image-only queries")

        planner_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_explode))
        )
        state = State(user_id=1, query="", image="data:image/png;base64,AAA")

        out = planner_agent.invoke(state, verbose=False)

        assert out.next_agent == "retriever"

    def test_invalid_agent_choice_defaults_to_chatter(
        self, planner_agent: PlannerAgent
    ) -> None:
        _stub_llm(planner_agent, "weather_bot")
        state = State(user_id=1, query="what is the weather")

        out = planner_agent.invoke(state, verbose=False)

        assert out.next_agent == "chatter"


class TestDecideFunction:
    def test_returns_next_agent_when_set(self, planner_agent: PlannerAgent) -> None:
        state = State(user_id=1, query="q", next_agent="cart")
        assert planner_agent.decide_function(state) == "cart"

    def test_defaults_to_chatter_when_next_agent_empty(
        self, planner_agent: PlannerAgent
    ) -> None:
        state = State(user_id=1, query="q")
        assert planner_agent.decide_function(state) == "chatter"
