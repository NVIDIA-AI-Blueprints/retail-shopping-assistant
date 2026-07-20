# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the enforced shopper-skill activation gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import Field, PrivateAttr

from chain_server.src.agenttypes import Cart, State
from chain_server.src.deepagents_runtime import (
    DeepAgentsRuntime,
    RequestIdentity,
    _skill_activation_input_model,
)
from chain_server.src.skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    ShopperSkillActivationError,
    ShopperSkillActivationMiddleware,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogTaxonomySubcategory,
)


REQUEST_ID = "request-a"
GATED_TOOLS = frozenset({"search_catalog_tool"})


class _RecordingToolModel(BaseChatModel):
    """Deterministic chat model that records each dynamically bound request."""

    model_name: str
    responses: list[AIMessage]
    calls: list[dict[str, Any]] = Field(default_factory=list)
    _response_index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "recording-openai-compatible"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "ls_provider": "openai",
            "ls_model_name": self.model_name,
            "ls_model_type": "chat",
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.bind(
            recording_tool_names=[_bound_tool_name(candidate) for candidate in tools],
            recording_tool_choice=tool_choice,
            recording_tool_settings=dict(kwargs),
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_names = list(kwargs.pop("recording_tool_names", []))
        tool_choice = kwargs.pop("recording_tool_choice", None)
        tool_settings = dict(kwargs.pop("recording_tool_settings", {}))
        system_prompt = "\n\n".join(
            message.text
            for message in messages
            if isinstance(message, SystemMessage)
        )
        self.calls.append(
            {
                "tools": tool_names,
                "tool_choice": tool_choice,
                "settings": tool_settings,
                "system_prompt": system_prompt,
                "messages": [
                    {"type": message.type, "text": message.text}
                    for message in messages
                ],
            }
        )
        response = self.responses[self._response_index]
        self._response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def _bound_tool_name(candidate: Any) -> str:
    if isinstance(candidate, dict):
        function = candidate.get("function") or {}
        return str(candidate.get("name") or function.get("name") or "")
    return str(getattr(candidate, "name", ""))


@tool
def activate_shopper_skills_tool(skill_names: list[str]) -> str:
    """Select shopper skills."""

    return ", ".join(skill_names)


@tool
def search_catalog_tool(query: str) -> str:
    """Search products."""

    return query


def _middleware() -> ShopperSkillActivationMiddleware:
    return ShopperSkillActivationMiddleware(
        request_id=REQUEST_ID,
        gated_tools=GATED_TOOLS,
        skill_descriptions={
            "outfit-styling": (
                "Use for outfit completion and conversational mid-browse styling."
            ),
            "product-discovery": "Use for browsing without styling intent.",
        },
    )


def test_activation_schema_rejects_two_primary_procedures() -> None:
    activation_input = _skill_activation_input_model(
        ("budget-shopping", "outfit-styling", "product-discovery")
    )

    with pytest.raises(ValueError, match="select exactly one primary procedure"):
        activation_input(
            skill_names=["outfit-styling", "product-discovery"],
        )

    selected = activation_input(
        skill_names=["outfit-styling", "budget-shopping"],
    )
    assert selected.skill_names == ["outfit-styling", "budget-shopping"]


def test_activation_schema_requires_primary_for_budget_only() -> None:
    activation_input = _skill_activation_input_model(
        ("budget-shopping", "outfit-styling", "product-discovery")
    )

    with pytest.raises(
        ValueError,
        match="budget-shopping requires exactly one primary procedure",
    ):
        activation_input(
            skill_names=["budget-shopping"],
        )


def test_activation_schema_allows_standalone_cart_and_policy_skills() -> None:
    activation_input = _skill_activation_input_model(
        (
            "cart-management",
            "outfit-styling",
            "product-discovery",
            "store-policy-answers",
        )
    )

    assert activation_input(
        skill_names=["cart-management"],
    ).skill_names == ["cart-management"]
    assert activation_input(
        skill_names=["store-policy-answers"],
    ).skill_names == ["store-policy-answers"]


def _model_request(messages: list[Any] | None = None) -> ModelRequest:
    messages = messages or [HumanMessage(content=f"REQUEST ID: {REQUEST_ID}")]
    return ModelRequest(
        model=cast(Any, object()),
        messages=messages,
        tools=[activate_shopper_skills_tool, search_catalog_tool],
        state={"messages": messages},
    )


def _capture_request(
    middleware: ShopperSkillActivationMiddleware,
    request: ModelRequest,
) -> ModelRequest:
    captured: list[ModelRequest] = []

    def handler(prepared: ModelRequest) -> ModelResponse:
        captured.append(prepared)
        if prepared.tool_choice == SKILL_ACTIVATION_TOOL_NAME:
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "activation-call",
                                "name": SKILL_ACTIVATION_TOOL_NAME,
                                "args": {
                                    "skill_names": ["outfit-styling"],
                                },
                            }
                        ],
                    )
                ]
            )
        return ModelResponse(result=[AIMessage(content="done")])

    middleware.wrap_model_call(request, handler)
    return captured[0]


def _tool_request(name: str, messages: list[Any]) -> ToolCallRequest:
    tool = (
        activate_shopper_skills_tool
        if name == SKILL_ACTIVATION_TOOL_NAME
        else search_catalog_tool
    )
    return ToolCallRequest(
        tool_call={
            "id": f"{name}-call",
            "name": name,
            "args": {},
            "type": "tool_call",
        },
        tool=tool,
        state={"messages": messages},
        runtime=cast(Any, None),
    )


def _activated_messages(request_id: str = REQUEST_ID) -> list[Any]:
    return [
        HumanMessage(content=f"REQUEST ID: {request_id}"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "activation-call",
                    "name": SKILL_ACTIVATION_TOOL_NAME,
                    "args": {
                        "skill_names": ["outfit-styling"],
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                f"{SKILL_ACTIVATION_COMPLETE} "
                "/shopper/outfit-styling/SKILL.md"
            ),
            name=SKILL_ACTIVATION_TOOL_NAME,
            tool_call_id="activation-call",
        ),
    ]


def test_pending_phase_forces_only_the_activation_tool() -> None:
    prepared = _capture_request(_middleware(), _model_request())

    assert [candidate.name for candidate in prepared.tools] == [
        SKILL_ACTIVATION_TOOL_NAME
    ]
    assert prepared.tool_choice == SKILL_ACTIVATION_TOOL_NAME
    assert prepared.model_settings["parallel_tool_calls"] is False
    assert "Required Shopper Skill Selection" in prepared.system_prompt
    assert "outfit-styling: Use for outfit completion" in prepared.system_prompt
    assert "product-discovery: Use for browsing" in prepared.system_prompt
    assert "never select both" in prepared.system_prompt
    assert "budget shopping may accompany either" in prepared.system_prompt
    assert (
        "Do not switch to product discovery merely because the current turn asks"
        in prepared.system_prompt
    )


def test_pending_phase_exposes_prior_skill_as_continuity_signal() -> None:
    messages = [
        *_activated_messages("old-request"),
        HumanMessage(content=f"REQUEST ID: {REQUEST_ID}"),
    ]

    prepared = _capture_request(_middleware(), _model_request(messages))

    assert "Previous turn's selected shopper skills: outfit-styling" in (
        prepared.system_prompt
    )
    assert "change it only when the shopper changes tasks" in (
        prepared.system_prompt
    )


def test_pending_phase_rejects_a_direct_model_answer() -> None:
    middleware = _middleware()

    with pytest.raises(
        ShopperSkillActivationError,
        match="did not complete required shopper skill activation",
    ):
        middleware.wrap_model_call(
            _model_request(),
            lambda _: ModelResponse(result=[AIMessage(content="Here are skirts.")]),
        )


def test_active_phase_injects_complete_skill_and_exposes_commerce() -> None:
    middleware = _middleware()
    middleware.activate(
        {
            "/shopper/outfit-styling/SKILL.md": (
                "# Outfit Styling\n"
                "## Conversational Mid-Browse\n"
                "Preserve the accepted beige top.\n"
                "## Unsupported Commerce Details"
            )
        }
    )

    prepared = _capture_request(middleware, _model_request(_activated_messages()))

    assert [candidate.name for candidate in prepared.tools] == [
        "search_catalog_tool"
    ]
    assert prepared.tool_choice is None
    assert prepared.model_settings["parallel_tool_calls"] is False
    assert "# Outfit Styling" in prepared.system_prompt
    assert "## Conversational Mid-Browse" in prepared.system_prompt
    assert "## Unsupported Commerce Details" in prepared.system_prompt


def test_active_phase_rejects_multiple_shopping_tools_in_one_model_step() -> None:
    middleware = _middleware()
    middleware.activate(
        {"/shopper/product-discovery/SKILL.md": "# Product Discovery"}
    )

    with pytest.raises(
        ShopperSkillActivationError,
        match="multiple shopping tools in one step",
    ):
        middleware.wrap_model_call(
            _model_request(_activated_messages()),
            lambda _: ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "search-a",
                                "name": "search_catalog_tool",
                                "args": {"query": "tops"},
                            },
                            {
                                "id": "search-b",
                                "name": "search_catalog_tool",
                                "args": {"query": "skirts"},
                            },
                        ],
                    )
                ]
            ),
        )


def test_failed_activation_exposes_no_tools() -> None:
    middleware = _middleware()
    middleware.fail()

    prepared = _capture_request(middleware, _model_request())

    assert prepared.tools == []
    assert prepared.tool_choice is None
    assert "Shopper Skill Activation Failed" in prepared.system_prompt


def test_same_batch_activation_does_not_unlock_commerce() -> None:
    messages = [
        HumanMessage(content=f"REQUEST ID: {REQUEST_ID}"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "activation-call",
                    "name": SKILL_ACTIVATION_TOOL_NAME,
                    "args": {
                        "skill_names": ["outfit-styling"],
                    },
                },
                {
                    "id": "search-call",
                    "name": "search_catalog_tool",
                    "args": {"query": "bottoms for a beige top"},
                },
            ],
        ),
    ]
    handled: list[ToolCallRequest] = []

    result = _middleware().wrap_tool_call(
        _tool_request("search_catalog_tool", messages),
        lambda request: handled.append(request),
    )

    assert handled == []
    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert str(result.content).startswith(SKILL_ACTIVATION_REQUIRED)


def test_previous_turn_activation_does_not_unlock_current_turn() -> None:
    messages = [
        *_activated_messages("old-request"),
        HumanMessage(content=f"REQUEST ID: {REQUEST_ID}"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "search-call",
                    "name": "search_catalog_tool",
                    "args": {"query": "skirts"},
                }
            ],
        ),
    ]

    result = _middleware().wrap_tool_call(
        _tool_request("search_catalog_tool", messages),
        lambda _: ToolMessage(content="should not run", tool_call_id="search-call"),
    )

    assert isinstance(result, ToolMessage)
    assert str(result.content).startswith(SKILL_ACTIVATION_REQUIRED)


def test_current_turn_activation_unlocks_commerce() -> None:
    request = _tool_request("search_catalog_tool", _activated_messages())
    expected = ToolMessage(content="catalog result", tool_call_id="search-call")
    handled: list[ToolCallRequest] = []

    result = _middleware().wrap_tool_call(
        request,
        lambda prepared: handled.append(prepared) or expected,
    )

    assert handled == [request]
    assert result is expected


def test_activation_tool_is_always_allowed() -> None:
    request = _tool_request(SKILL_ACTIVATION_TOOL_NAME, [])
    expected = ToolMessage(content="loaded", tool_call_id="activation-call")

    result = _middleware().wrap_tool_call(request, lambda _: expected)

    assert result is expected


@pytest.mark.asyncio
async def test_compiled_agent_loads_full_skill_before_exposing_commerce(
    base_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real agent loop across activation and commerce phases."""

    model_name = "compiled-skill-activation-test"
    base_config.llm_name = model_name
    model = _RecordingToolModel(
        model_name=model_name,
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "invalid-activation",
                        "name": SKILL_ACTIVATION_TOOL_NAME,
                        "args": {
                            "skill_names": [
                                "outfit-styling",
                                "product-discovery",
                            ],
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "activate-skill",
                        "name": SKILL_ACTIVATION_TOOL_NAME,
                        "args": {
                            "skill_names": ["outfit-styling"],
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "read-cart",
                        "name": "get_cart_tool",
                        "args": {},
                    }
                ],
            ),
            AIMessage(content="Your cart is empty."),
        ],
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    monkeypatch.setattr(runtime, "_read_cart", lambda _: Cart())
    identity = RequestIdentity(
        session_id="session-a",
        conversation_id="conversation-a",
        cart_id="cart-a",
        context_user_id=1,
        cart_user_id=1,
        request_id=REQUEST_ID,
    )
    state = State(user_id=1, query="What bottoms go with the beige top?")
    agent = runtime._create_agent(
        state,
        identity,
        CatalogCapabilities(
            catalog_id="test-catalog",
            retrieval_modes=["text"],
            filters={},
        ),
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"REQUEST ID: {REQUEST_ID}\n"
                        "USER QUERY: What bottoms go with the beige top?"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": identity.conversation_id}},
    )

    assert len(model.calls) == 4
    first_call, retry_call, shopping_call, _ = model.calls
    assert first_call["tools"] == [SKILL_ACTIVATION_TOOL_NAME]
    assert first_call["tool_choice"] == SKILL_ACTIVATION_TOOL_NAME
    assert first_call["settings"]["parallel_tool_calls"] is False
    assert "## Active Shopper Skills" not in first_call["system_prompt"]
    assert "outfit-styling: Customer-facing fashion styling" in (
        first_call["system_prompt"]
    )
    assert "product-discovery: General product search" in (
        first_call["system_prompt"]
    )

    assert retry_call["tools"] == [SKILL_ACTIVATION_TOOL_NAME]
    assert retry_call["tool_choice"] == SKILL_ACTIVATION_TOOL_NAME
    assert SKILL_ACTIVATION_TOOL_NAME not in shopping_call["tools"]
    assert "search_catalog_tool" in shopping_call["tools"]
    assert "get_cart_tool" in shopping_call["tools"]
    assert "read_file" in shopping_call["tools"]
    assert "## Active Shopper Skills" in shopping_call["system_prompt"]
    assert "# Outfit Styling" in shopping_call["system_prompt"]
    assert "# Product Discovery" not in shopping_call["system_prompt"]
    assert "## Conversational Mid-Browse" in shopping_call["system_prompt"]
    assert "## Unsupported Commerce Details" in shopping_call["system_prompt"]
    assert result["messages"][-1].content == "Your cart is empty."


@pytest.mark.asyncio
async def test_compiled_agent_allows_one_invalid_taxonomy_repair_then_synthesizes(
    base_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid catalog enum cannot consume the turn in a retry loop."""

    model_name = "compiled-tool-loop-control-test"
    base_config.llm_name = model_name
    invalid_search = {
        "semantic_query": "pants to coordinate with a beige top",
        "requested_product_type": "pants",
        "taxonomy_status": "exact_requested_type",
        "taxonomy": {"category": ["apparel"], "subcategory": ["pants"]},
        "required_constraints": {},
        "scope_complete": True,
    }
    model = _RecordingToolModel(
        model_name=model_name,
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "activate-skill",
                        "name": SKILL_ACTIVATION_TOOL_NAME,
                        "args": {
                            "skill_names": ["outfit-styling"],
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "invalid-search",
                        "name": "search_catalog_tool",
                        "args": invalid_search,
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "invalid-repair",
                        "name": "search_catalog_tool",
                        "args": invalid_search,
                    }
                ],
            ),
            AIMessage(
                content=(
                    "I don't see pants in this catalog. Would you like me to "
                    "look at skirts instead?"
                )
            ),
        ],
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    identity = RequestIdentity(
        session_id="session-loop",
        conversation_id="conversation-loop",
        cart_id="cart-loop",
        context_user_id=1,
        cart_user_id=1,
        request_id="request-loop",
    )
    state = State(user_id=1, query="What pants go with the beige top?")
    capabilities = CatalogCapabilities(
        catalog_id="test-catalog",
        retrieval_modes=["text"],
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category",
            subcategory_field="subcategory",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=2,
                    subcategories={
                        "skirts": CatalogTaxonomySubcategory(product_count=2)
                    },
                )
            },
        ),
    )
    agent = runtime._create_agent(state, identity, capabilities)

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "REQUEST ID: request-loop\n"
                        "USER QUERY: What pants go with the beige top?"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": identity.conversation_id}},
    )

    assert len(model.calls) == 4
    assert model.calls[2]["tools"] == ["search_catalog_tool"]
    assert model.calls[2]["settings"]["parallel_tool_calls"] is False
    assert "## Catalog Search Repair" in model.calls[2]["system_prompt"]
    assert "## Active Shopper Skills" in model.calls[2]["system_prompt"]
    assert "# Outfit Styling" in model.calls[2]["system_prompt"]
    assert "You are a shopping assistant" not in model.calls[2]["system_prompt"]
    repair_messages = model.calls[2]["messages"]
    assert not any(
        message["type"] in {"ai", "tool"} for message in repair_messages
    )
    assert sum(message["type"] == "human" for message in repair_messages) == 2
    assert "USER QUERY: What pants go with the beige top?" in (
        repair_messages[-2]["text"]
    )
    assert "CATALOG VALIDATOR FEEDBACK" in repair_messages[-1]["text"]
    assert model.calls[3]["tools"] == []
    assert result["messages"][-1].content.startswith("I don't see pants")
