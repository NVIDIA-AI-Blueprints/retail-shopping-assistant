# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the enforced shopper-skill activation gate."""

from __future__ import annotations

from collections.abc import Sequence
import json
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
from chain_server.src.catalog_execution import CatalogSearchExecution
from chain_server.src.deepagents_runtime import (
    DeepAgentsRuntime,
    RequestIdentity,
    _current_turn_weather_date_available,
    _skill_activation_input_model,
)
from chain_server.src.skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_TOOL_NOT_GRANTED,
    _ACTIVATION_PROMPT,
    ShopperSkillActivationError,
    ShopperSkillActivationMiddleware,
    selected_skill_names_for_turn,
)
from chain_server.src.tool_loop_control import SERVER_CATALOG_CLARIFICATION
from chain_server.src.weather_tool import weather_date_context_available
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogTaxonomySubcategory,
    SearchCatalogResult,
)


REQUEST_ID = "request-a"
SKILL_TOOL_GRANTS = {
    "budget-shopping": frozenset(),
    "cart-management": frozenset(
        {
            "add_cart_items_tool",
            "get_cart_tool",
            "remove_cart_item_tool",
            "resolve_conversation_products_tool",
            "update_cart_items_tool",
            "view_cart_total_tool",
        }
    ),
    "event-context": frozenset({"get_weather_forecast_tool"}),
    "outfit-styling": frozenset(
        {
            "check_active_promotions_tool",
            "check_product_availability_tool",
            "get_product_details_tool",
            "resolve_conversation_products_tool",
            "search_catalog_tool",
        }
    ),
    "product-discovery": frozenset(
        {
            "check_active_promotions_tool",
            "check_product_availability_tool",
            "get_product_details_tool",
            "resolve_conversation_products_tool",
            "search_catalog_tool",
        }
    ),
    "store-policy-answers": frozenset({"get_store_policy_tool"}),
}


def test_activation_prompt_makes_explicit_outdoor_weather_material() -> None:
    compact_prompt = " ".join(_ACTIVATION_PROMPT.split())
    assert "outdoor patio, beach, garden, rooftop, or" in compact_prompt
    assert "choose `event_date`, not `none`" in compact_prompt


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


@tool
def get_product_details_tool(product_ref: str) -> str:
    """Read product details."""

    return product_ref


@tool
def resolve_conversation_products_tool(references: list[str]) -> str:
    """Resolve products shown earlier in the conversation."""

    return ", ".join(references)


@tool
def add_cart_items_tool(product_ref: str) -> str:
    """Add a product to the cart."""

    return product_ref


@tool
def get_weather_forecast_tool(location_source: str) -> str:
    """Read event-weather evidence."""

    return location_source


@tool
def get_store_policy_tool(topic: str) -> str:
    """Read one configured store policy."""

    return topic


def _middleware(
    *,
    previous_selected_skills: Sequence[str] = (),
) -> ShopperSkillActivationMiddleware:
    return ShopperSkillActivationMiddleware(
        request_id=REQUEST_ID,
        skill_descriptions={
            "budget-shopping": "Use as a budget modifier.",
            "cart-management": "Use for cart operations.",
            "event-context": (
                "Use with outfit styling when event location or venue matters."
            ),
            "outfit-styling": (
                "Use for outfit completion and conversational mid-browse styling."
            ),
            "product-discovery": "Use for browsing without styling intent.",
            "store-policy-answers": "Use for store policies.",
        },
        skill_tool_grants=SKILL_TOOL_GRANTS,
        previous_selected_skills=previous_selected_skills,
    )


def test_activation_schema_rejects_two_primary_procedures() -> None:
    activation_input = _skill_activation_input_model(
        (
            "budget-shopping",
            "event-context",
            "outfit-styling",
            "product-discovery",
        ),
    )

    with pytest.raises(ValueError, match="select exactly one primary procedure"):
        activation_input(
            skill_names=["outfit-styling", "product-discovery"],
        )

    selected = activation_input(
        skill_names=["outfit-styling", "budget-shopping"],
    )
    assert selected.skill_names == ["outfit-styling", "budget-shopping"]

    selected_with_event = activation_input(
        skill_names=[
            "outfit-styling",
            "event-context",
            "budget-shopping",
        ],
        event_context_next_question="event_location",
    )
    assert selected_with_event.skill_names == [
        "outfit-styling",
        "event-context",
        "budget-shopping",
    ]
    assert selected_with_event.event_context_next_question == "event_location"


def test_activation_schema_binds_only_event_context_question() -> None:
    activation_input = _skill_activation_input_model(
        ("cart-management", "event-context", "outfit-styling"),
    )

    with pytest.raises(
        ValueError,
        match="event_context_next_question is required exactly",
    ):
        activation_input(
            skill_names=["outfit-styling", "event-context"],
        )

    with pytest.raises(
        ValueError,
        match="event_context_next_question is required exactly",
    ):
        activation_input(
            skill_names=["outfit-styling"],
            event_context_next_question="none",
        )

    selected = activation_input(
        skill_names=[
            "outfit-styling",
            "event-context",
            "cart-management",
        ],
        event_context_next_question="event_date",
    )
    assert selected.skill_names == [
        "outfit-styling",
        "event-context",
        "cart-management",
    ]
    assert selected.event_context_next_question == "event_date"


def test_bounded_weather_date_removes_event_date_from_activation_schema() -> None:
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling"),
        weather_date_available=True,
    )

    accepted = activation_input(
        skill_names=["outfit-styling", "event-context"],
        event_context_next_question="none",
    )

    assert accepted.event_context_next_question == "none"
    with pytest.raises(ValueError):
        activation_input(
            skill_names=["outfit-styling", "event-context"],
            event_context_next_question="event_date",
        )
    question_schema = activation_input.model_json_schema()["properties"][
        "event_context_next_question"
    ]
    assert "event_date" not in json.dumps(question_schema)


def test_bare_next_week_shapes_activation_as_a_bounded_date() -> None:
    date_available = weather_date_context_available(
        ("NYC, on an outdoor patio next week.",)
    )
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling"),
        weather_date_available=date_available,
    )

    assert date_available is True
    with pytest.raises(ValueError):
        activation_input(
            skill_names=["outfit-styling", "event-context"],
            event_context_next_question="event_date",
        )


def test_prior_event_date_cannot_narrow_a_new_event_activation() -> None:
    state = State(
        user_id=1,
        query="Now help with a different wedding in Cancun on the beach.",
        context=(
            "User: The first wedding is August 3 in NYC.\n"
            "Assistant: I can plan around that date."
        ),
    )

    assert _current_turn_weather_date_available(state) is False
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling"),
        weather_date_available=_current_turn_weather_date_available(state),
    )
    accepted = activation_input(
        skill_names=["outfit-styling", "event-context"],
        event_context_next_question="event_date",
    )
    assert accepted.event_context_next_question == "event_date"


def test_missing_weather_date_keeps_event_date_in_activation_schema() -> None:
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling"),
        weather_date_available=False,
    )

    accepted = activation_input(
        skill_names=["outfit-styling", "event-context"],
        event_context_next_question="event_date",
    )

    assert accepted.event_context_next_question == "event_date"


def test_dynamic_event_question_gets_typed_activation_feedback() -> None:
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling"),
    )
    middleware = _middleware()

    with pytest.raises(ValueError) as captured:
        activation_input(
            skill_names=["outfit-styling", "event-context"],
            event_context_next_question="destination",
        )

    first = middleware.handle_activation_validation_error(captured.value)
    second = middleware.handle_activation_validation_error(captured.value)
    response = middleware.wrap_model_call(
        _model_request(),
        lambda _: pytest.fail("clarification must not call the model"),
    )

    assert first.startswith("SHOPPER_SKILL_ACTIVATION_INVALID:")
    assert "event_context_next_question" in first
    assert second.startswith(
        "SHOPPER_SKILL_ACTIVATION_CLARIFICATION_REQUIRED:"
    )
    assert response.result[0].content == "What event detail should I plan around?"


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


@pytest.mark.parametrize(
    "skill_names",
    [
        ["event-context"],
        ["event-context", "product-discovery"],
    ],
)
def test_activation_schema_requires_outfit_styling_for_event_context(
    skill_names: list[str],
) -> None:
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling", "product-discovery")
    )

    with pytest.raises(ValueError, match="event-context requires outfit-styling"):
        activation_input(skill_names=skill_names)


def test_repeated_invalid_event_context_selection_clarifies_without_tools() -> None:
    activation_input = _skill_activation_input_model(
        ("event-context", "outfit-styling", "product-discovery")
    )
    middleware = _middleware()

    with pytest.raises(ValueError) as captured:
        activation_input(skill_names=["event-context"])

    first = middleware.handle_activation_validation_error(captured.value)
    second = middleware.handle_activation_validation_error(captured.value)
    response = middleware.wrap_model_call(
        _model_request(),
        lambda _: pytest.fail("clarification must not call the model"),
    )

    assert first.startswith("SHOPPER_SKILL_ACTIVATION_INVALID:")
    assert "requires outfit-styling" in first
    assert "Pair it with outfit-styling for occasion-led fashion guidance" in first
    assert second.startswith(
        "SHOPPER_SKILL_ACTIVATION_CLARIFICATION_REQUIRED:"
    )
    assert response.result[0].content == (
        "What outfit or event would you like help styling?"
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


def _model_request(
    messages: list[Any] | None = None,
    *,
    tools: list[BaseTool] | None = None,
) -> ModelRequest:
    messages = messages or [HumanMessage(content=f"REQUEST ID: {REQUEST_ID}")]
    return ModelRequest(
        model=cast(Any, object()),
        messages=messages,
        tools=tools
        or [
            activate_shopper_skills_tool,
            search_catalog_tool,
            get_product_details_tool,
            resolve_conversation_products_tool,
            add_cart_items_tool,
            get_weather_forecast_tool,
            get_store_policy_tool,
        ],
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


def _tool_request(
    name: str,
    messages: list[Any],
    args: dict[str, Any] | None = None,
) -> ToolCallRequest:
    tool = (
        activate_shopper_skills_tool
        if name == SKILL_ACTIVATION_TOOL_NAME
        else {
            "add_cart_items_tool": add_cart_items_tool,
            "remove_cart_item_tool": add_cart_items_tool,
            "update_cart_items_tool": add_cart_items_tool,
            "get_product_details_tool": get_product_details_tool,
            "get_weather_forecast_tool": get_weather_forecast_tool,
            "get_store_policy_tool": get_store_policy_tool,
            "resolve_conversation_products_tool": (
                resolve_conversation_products_tool
            ),
            "search_catalog_tool": search_catalog_tool,
        }[name]
    )
    return ToolCallRequest(
        tool_call={
            "id": f"{name}-call",
            "name": name,
            "args": args or {},
            "type": "tool_call",
        },
        tool=tool,
        state={"messages": messages},
        runtime=cast(Any, None),
    )


def _activated_messages(
    request_id: str = REQUEST_ID,
    *,
    skill_name: str = "outfit-styling",
    query: str | None = None,
) -> list[Any]:
    request_text = f"REQUEST ID: {request_id}"
    if query:
        request_text += f"\nUSER QUERY: {query}"
    return [
        HumanMessage(content=request_text),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "activation-call",
                    "name": SKILL_ACTIVATION_TOOL_NAME,
                    "args": {
                        "skill_names": [skill_name],
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                f"{SKILL_ACTIVATION_COMPLETE} "
                f"/shopper/{skill_name}/SKILL.md"
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
    assert "event-context: Use with outfit styling" in prepared.system_prompt
    assert "never select both" in prepared.system_prompt
    assert "budget shopping may accompany either" in prepared.system_prompt
    assert "Event context may accompany outfit" in prepared.system_prompt
    normalized_prompt = " ".join(prepared.system_prompt.split())
    assert "one permitted `event_context_next_question` decision" in (
        normalized_prompt
    )
    assert "Event context is additive" in normalized_prompt
    assert "`event_venue` only after destination is established" in (
        normalized_prompt
    )
    assert "Never infer a beach, outdoor, indoor, or terrain setting" in (
        normalized_prompt
    )
    assert (
        "Select it whenever an event destination or venue is stated, or when "
        "the response would otherwise ask about or branch on missing destination"
        in normalized_prompt
    )
    assert "generic advice is not a reason to omit it" in normalized_prompt
    assert (
        "Do not switch to product discovery merely because the current turn asks"
        in normalized_prompt
    )


def test_pending_phase_exposes_prior_skill_as_continuity_signal() -> None:
    prepared = _capture_request(
        _middleware(previous_selected_skills=["outfit-styling"]),
        _model_request(),
    )

    assert "Previous turn's selected shopper skills: outfit-styling" in (
        prepared.system_prompt
    )
    assert "change it only when the shopper changes tasks" in (
        prepared.system_prompt
    )


def test_completed_current_turn_activation_exposes_selected_skill_names() -> None:
    messages = _activated_messages(
        REQUEST_ID,
        skill_name="outfit-styling",
    )

    assert selected_skill_names_for_turn(messages, REQUEST_ID) == (
        "outfit-styling",
    )
    assert selected_skill_names_for_turn(messages, "other-request") == ()


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
        },
        ["outfit-styling"],
    )

    prepared = _capture_request(middleware, _model_request(_activated_messages()))

    assert [candidate.name for candidate in prepared.tools] == [
        "search_catalog_tool",
        "get_product_details_tool",
        "resolve_conversation_products_tool",
    ]
    assert prepared.tool_choice is None
    assert prepared.model_settings["parallel_tool_calls"] is False
    assert "# Outfit Styling" in prepared.system_prompt
    assert "## Conversational Mid-Browse" in prepared.system_prompt
    assert "## Unsupported Commerce Details" in prepared.system_prompt


def test_event_context_injects_beside_styling_and_exposes_weather() -> None:
    middleware = _middleware()
    middleware.activate(
        {
            "/shopper/event-context/SKILL.md": "# Event Context",
            "/shopper/outfit-styling/SKILL.md": "# Outfit Styling",
        },
        ["outfit-styling", "event-context"],
    )

    prepared = _capture_request(middleware, _model_request(_activated_messages()))

    assert [candidate.name for candidate in prepared.tools] == [
        "search_catalog_tool",
        "get_product_details_tool",
        "resolve_conversation_products_tool",
        "get_weather_forecast_tool",
    ]
    assert "# Event Context" in prepared.system_prompt
    assert "# Outfit Styling" in prepared.system_prompt
    assert middleware._granted_tools == (
        SKILL_TOOL_GRANTS["outfit-styling"]
        | SKILL_TOOL_GRANTS["event-context"]
    )


def test_weather_result_does_not_hide_or_block_later_business_tools() -> None:
    middleware = _middleware()
    middleware.activate(
        {
            "/shopper/cart-management/SKILL.md": "# Cart Management",
            "/shopper/event-context/SKILL.md": "# Event Context",
            "/shopper/outfit-styling/SKILL.md": "# Outfit Styling",
            "/shopper/store-policy-answers/SKILL.md": "# Store Policy Answers",
        },
        [
            "outfit-styling",
            "event-context",
            "cart-management",
            "store-policy-answers",
        ],
    )
    messages = [
        HumanMessage(content=f"REQUEST ID: {REQUEST_ID}"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "activation-call",
                    "name": SKILL_ACTIVATION_TOOL_NAME,
                    "args": {
                        "skill_names": [
                            "outfit-styling",
                            "event-context",
                            "cart-management",
                            "store-policy-answers",
                        ],
                        "event_context_next_question": "none",
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                f"{SKILL_ACTIVATION_COMPLETE} "
                "/shopper/outfit-styling/SKILL.md, "
                "/shopper/event-context/SKILL.md, "
                "/shopper/cart-management/SKILL.md, "
                "/shopper/store-policy-answers/SKILL.md"
            ),
            name=SKILL_ACTIVATION_TOOL_NAME,
            tool_call_id="activation-call",
        ),
    ]
    weather_request = _tool_request(
        "get_weather_forecast_tool",
        messages,
        {"location_source": "shopper_provided_location"},
    )
    weather_result = ToolMessage(
        content="weather evidence",
        name="get_weather_forecast_tool",
        tool_call_id="get_weather_forecast_tool-call",
    )

    assert middleware.wrap_tool_call(
        weather_request,
        lambda prepared: weather_result,
    ) is weather_result

    messages_after_weather = [*messages, weather_result]
    prepared = _capture_request(
        middleware,
        _model_request(messages_after_weather),
    )
    assert [candidate.name for candidate in prepared.tools] == [
        "search_catalog_tool",
        "get_product_details_tool",
        "resolve_conversation_products_tool",
        "add_cart_items_tool",
        "get_weather_forecast_tool",
        "get_store_policy_tool",
    ]

    for tool_name, args in (
        (
            "resolve_conversation_products_tool",
            {"references": ["Intricate Lace Gown", "Wavy Hem Satin Dress"]},
        ),
        ("get_product_details_tool", {"product_ref": "dress-1"}),
        ("add_cart_items_tool", {"product_ref": "dress-1"}),
        ("get_store_policy_tool", {"topic": "returns"}),
    ):
        request = _tool_request(tool_name, messages_after_weather, args)
        expected = ToolMessage(
            content=f"{tool_name} result",
            name=tool_name,
            tool_call_id=f"{tool_name}-call",
        )
        handled: list[ToolCallRequest] = []
        result = middleware.wrap_tool_call(
            request,
            lambda candidate: handled.append(candidate) or expected,
        )
        assert handled == [request]
        assert result is expected


def test_runtime_denial_hides_only_weather_and_preserves_product_tools() -> None:
    middleware = _middleware()
    middleware.activate(
        {
            "/shopper/event-context/SKILL.md": "# Event Context",
            "/shopper/outfit-styling/SKILL.md": "# Outfit Styling",
        },
        ["outfit-styling", "event-context"],
    )
    rejection = "STOP_TOOL_USE: Event date authority is incomplete."

    middleware.deny_tool_for_turn("get_weather_forecast_tool", rejection)
    prepared = _capture_request(
        middleware,
        _model_request(_activated_messages()),
    )
    handled: list[ToolCallRequest] = []
    blocked = middleware.wrap_tool_call(
        _tool_request("get_weather_forecast_tool", _activated_messages()),
        handled.append,
    )

    assert [candidate.name for candidate in prepared.tools] == [
        "search_catalog_tool",
        "get_product_details_tool",
        "resolve_conversation_products_tool",
    ]
    assert handled == []
    assert isinstance(blocked, ToolMessage)
    assert blocked.content == rejection


def test_weather_dispatch_requires_event_context_beside_outfit_styling() -> None:
    messages = [
        HumanMessage(content=f"REQUEST ID: {REQUEST_ID}"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "activation-call",
                    "name": SKILL_ACTIVATION_TOOL_NAME,
                    "args": {
                        "skill_names": ["outfit-styling", "event-context"],
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                f"{SKILL_ACTIVATION_COMPLETE} "
                "/shopper/outfit-styling/SKILL.md, "
                "/shopper/event-context/SKILL.md"
            ),
            name=SKILL_ACTIVATION_TOOL_NAME,
            tool_call_id="activation-call",
        ),
    ]
    request = _tool_request(
        "get_weather_forecast_tool",
        messages,
        {"location_source": "confirmed_saved_zip"},
    )

    styling_only = _middleware()
    styling_only.activate(
        {"/shopper/outfit-styling/SKILL.md": "# Outfit Styling"},
        ["outfit-styling"],
    )
    blocked_calls: list[ToolCallRequest] = []
    blocked = styling_only.wrap_tool_call(request, blocked_calls.append)

    assert blocked_calls == []
    assert isinstance(blocked, ToolMessage)
    assert str(blocked.content).startswith(SKILL_TOOL_NOT_GRANTED)

    event_styling = _middleware()
    event_styling.activate(
        {
            "/shopper/outfit-styling/SKILL.md": "# Outfit Styling",
            "/shopper/event-context/SKILL.md": "# Event Context",
        },
        ["outfit-styling", "event-context"],
    )
    expected = ToolMessage(
        content="weather result",
        name="get_weather_forecast_tool",
        tool_call_id="get_weather_forecast_tool-call",
    )
    allowed_calls: list[ToolCallRequest] = []
    allowed = event_styling.wrap_tool_call(
        request,
        lambda prepared: allowed_calls.append(prepared) or expected,
    )

    assert allowed_calls == [request]
    assert allowed is expected


def test_active_phase_rejects_multiple_shopping_tools_in_one_model_step() -> None:
    middleware = _middleware()
    middleware.activate(
        {"/shopper/product-discovery/SKILL.md": "# Product Discovery"},
        ["product-discovery"],
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
    middleware = _middleware()
    middleware.activate(
        {"/shopper/outfit-styling/SKILL.md": "# Outfit Styling"},
        ["outfit-styling"],
    )
    request = _tool_request("search_catalog_tool", _activated_messages())
    expected = ToolMessage(content="catalog result", tool_call_id="search-call")
    handled: list[ToolCallRequest] = []

    result = middleware.wrap_tool_call(
        request,
        lambda prepared: handled.append(prepared) or expected,
    )

    assert handled == [request]
    assert result is expected


@pytest.mark.parametrize(
    "tool_name",
    [
        "add_cart_items_tool",
        "remove_cart_item_tool",
        "update_cart_items_tool",
    ],
)
def test_outfit_styling_rejects_cart_mutation_before_execution(
    tool_name: str,
) -> None:
    middleware = _middleware()
    middleware.activate(
        {"/shopper/outfit-styling/SKILL.md": "# Outfit Styling"},
        ["outfit-styling"],
    )
    request = _tool_request(tool_name, _activated_messages())
    handled: list[ToolCallRequest] = []

    result = middleware.wrap_tool_call(request, handled.append)

    assert handled == []
    assert isinstance(result, ToolMessage)
    assert str(result.content).startswith(SKILL_TOOL_NOT_GRANTED)


def test_browse_only_product_discovery_rejects_cart_mutation() -> None:
    middleware = _middleware()
    middleware.activate(
        {"/shopper/product-discovery/SKILL.md": "# Product Discovery"},
        ["product-discovery"],
    )
    messages = _activated_messages(
        skill_name="product-discovery",
        query="Show me casual black shoes",
    )

    prepared = _capture_request(middleware, _model_request(messages))

    assert [candidate.name for candidate in prepared.tools] == [
        "search_catalog_tool",
        "get_product_details_tool",
        "resolve_conversation_products_tool",
    ]
    request = _tool_request(
        "add_cart_items_tool",
        messages,
        {"items": [{"product_ref": "shoe-a"}]},
    )
    handled: list[ToolCallRequest] = []
    result = middleware.wrap_tool_call(request, handled.append)
    assert handled == []
    assert isinstance(result, ToolMessage)
    assert str(result.content).startswith(SKILL_TOOL_NOT_GRANTED)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Constraint provenance remains an unresolved semantic-assurance boundary.",
)
def test_invented_catalog_constraint_is_rejected_before_execution() -> None:
    """An advertised filter is not authorized merely because it is valid."""

    middleware = _middleware()
    middleware.activate(
        {"/shopper/outfit-styling/SKILL.md": "# Outfit Styling"},
        ["outfit-styling"],
    )
    request = _tool_request(
        "search_catalog_tool",
        _activated_messages(query="Heels or flats for this look?"),
        {"required_constraints": {"color": ["black"]}},
    )
    handled: list[ToolCallRequest] = []

    result = middleware.wrap_tool_call(request, handled.append)

    assert handled == []
    assert isinstance(result, ToolMessage)
    assert str(result.content).startswith("CATALOG_CALL_NOT_AUTHORIZED:")


def test_cart_management_exposes_cart_mutation_but_not_catalog_search() -> None:
    middleware = _middleware()
    middleware.activate(
        {"/shopper/cart-management/SKILL.md": "# Cart Management"},
        ["cart-management"],
    )
    messages = _activated_messages()
    prepared = _capture_request(middleware, _model_request(messages))

    assert [candidate.name for candidate in prepared.tools] == [
        "resolve_conversation_products_tool",
        "add_cart_items_tool",
    ]
    request = _tool_request("add_cart_items_tool", messages)
    expected = ToolMessage(content="cart updated", tool_call_id="add-call")
    result = middleware.wrap_tool_call(request, lambda _: expected)
    assert result is expected


def test_activation_tool_is_always_allowed() -> None:
    request = _tool_request(SKILL_ACTIVATION_TOOL_NAME, [])
    expected = ToolMessage(content="loaded", tool_call_id="activation-call")

    result = _middleware().wrap_tool_call(request, lambda _: expected)

    assert result is expected


def test_pending_phase_rejects_multiple_activation_calls() -> None:
    """Competing activation calls execute neither selection."""

    response = _middleware().wrap_model_call(
        _model_request(),
        lambda _: ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "activate-discovery",
                            "name": SKILL_ACTIVATION_TOOL_NAME,
                            "args": {"skill_names": ["product-discovery"]},
                        },
                        {
                            "id": "activate-styling",
                            "name": SKILL_ACTIVATION_TOOL_NAME,
                            "args": {"skill_names": ["outfit-styling"]},
                        },
                    ],
                )
            ]
        ),
    )

    assert response.result == [
        AIMessage(content="What product or shopping task would you like help with?")
    ]


@pytest.mark.asyncio
async def test_compiled_agent_bounds_repeated_invalid_budget_activation(
    base_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated modifier-only selection clarifies instead of exhausting the graph."""

    model_name = "compiled-budget-activation-recovery-test"
    base_config.llm_name = model_name
    model = _RecordingToolModel(
        model_name=model_name,
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"budget-only-{attempt}",
                        "name": SKILL_ACTIVATION_TOOL_NAME,
                        "args": {"skill_names": ["budget-shopping"]},
                    }
                ],
            )
            for attempt in (1, 2)
        ],
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    identity = RequestIdentity(
        session_id="session-budget",
        conversation_id="conversation-budget",
        cart_id="cart-budget",
        context_user_id=1,
        cart_user_id=1,
        request_id="request-budget",
    )
    agent = runtime._create_agent(
        State(user_id=1, query="I need to stay under $100 total."),
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
                        "REQUEST ID: request-budget\n"
                        "USER QUERY: I need to stay under $100 total."
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": identity.conversation_id}},
    )

    assert len(model.calls) == 2
    activation_results = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == SKILL_ACTIVATION_TOOL_NAME
    ]
    assert len(activation_results) == 2
    assert activation_results[0].status == "error"
    assert str(activation_results[0].content).startswith(
        "SHOPPER_SKILL_ACTIVATION_INVALID:"
    )
    assert activation_results[1].status == "error"
    assert str(activation_results[1].content).startswith(
        "SHOPPER_SKILL_ACTIVATION_CLARIFICATION_REQUIRED:"
    )
    assert result["messages"][-1].content == (
        "What product or outfit would you like to find within your budget?"
    )
    assert not any(
        isinstance(message, ToolMessage)
        and message.name != SKILL_ACTIVATION_TOOL_NAME
        for message in result["messages"]
    )


@pytest.mark.asyncio
async def test_compiled_agent_loads_skill_and_blocks_ungranted_tool(
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
            AIMessage(content="I can help with products for this outfit."),
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
    state = State(
        user_id=1,
        query="What bottoms go with the beige top?",
        previous_selected_skill_names=["outfit-styling"],
    )
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
    assert "Previous turn's selected shopper skills: outfit-styling" in (
        first_call["system_prompt"]
    )

    assert retry_call["tools"] == [SKILL_ACTIVATION_TOOL_NAME]
    assert retry_call["tool_choice"] == SKILL_ACTIVATION_TOOL_NAME
    assert SKILL_ACTIVATION_TOOL_NAME not in shopping_call["tools"]
    assert "search_catalog_tool" in shopping_call["tools"]
    assert "get_product_details_tool" in shopping_call["tools"]
    assert "check_product_availability_tool" in shopping_call["tools"]
    assert "check_active_promotions_tool" in shopping_call["tools"]
    assert "get_cart_tool" not in shopping_call["tools"]
    assert "add_cart_items_tool" not in shopping_call["tools"]
    assert "read_file" in shopping_call["tools"]
    assert "## Active Shopper Skills" in shopping_call["system_prompt"]
    rejected = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "get_cart_tool"
    ]
    assert len(rejected) == 1
    assert str(rejected[0].content).startswith(SKILL_TOOL_NOT_GRANTED)
    assert result["messages"][-1].content == (
        "I can help with products for this outfit."
    )


@pytest.mark.asyncio
async def test_compiled_agent_answers_promotions_without_catalog_search(
    base_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promotions lookup uses its granted stub instead of catalog retrieval."""

    from chain_server.src import deepagents_runtime as runtime_mod

    model_name = "compiled-promotions-test"
    base_config.llm_name = model_name
    model = _RecordingToolModel(
        model_name=model_name,
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "activate-product-discovery",
                        "name": SKILL_ACTIVATION_TOOL_NAME,
                        "args": {"skill_names": ["product-discovery"]},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "check-promotions",
                        "name": "check_active_promotions_tool",
                        "args": {},
                    }
                ],
            ),
            AIMessage(
                content=(
                    "No active sale or promotion is available through the assistant "
                    "right now."
                )
            ),
        ],
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)

    def fail_catalog_search(*_args, **_kwargs):
        raise AssertionError("promotions lookup must not execute catalog search")

    monkeypatch.setattr(
        runtime_mod,
        "execute_catalog_search",
        fail_catalog_search,
    )
    identity = RequestIdentity(
        session_id="session-promotions",
        conversation_id="conversation-promotions",
        cart_id="cart-promotions",
        context_user_id=1,
        cart_user_id=1,
        request_id=REQUEST_ID,
    )
    agent = runtime._create_agent(
        State(user_id=1, query="Any sales on shoes?"),
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
                        "USER QUERY: Any sales on shoes?"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": identity.conversation_id}},
    )

    assert len(model.calls) == 3
    assert model.calls[0]["tools"] == [SKILL_ACTIVATION_TOOL_NAME]
    assert "check_active_promotions_tool" in model.calls[1]["tools"]
    assert "search_catalog_tool" in model.calls[1]["tools"]
    assert "get_cart_tool" not in model.calls[1]["tools"]
    promotion_results = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "check_active_promotions_tool"
    ]
    assert len(promotion_results) == 1
    expected = (
        "No active sale or promotion is available through the assistant right now."
    )
    assert expected in str(promotion_results[0].content)
    assert not any(
        isinstance(message, ToolMessage) and message.name == "search_catalog_tool"
        for message in result["messages"]
    )
    assert result["messages"][-1].content == expected


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
                content=(
                    "I can’t map pants to an advertised catalog type without "
                    "substituting something else. Would you like me to search "
                    "skirts, or did you mean another kind of bottom?"
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

    assert len(model.calls) == 3
    assert model.calls[2]["tools"] == ["search_catalog_tool"]
    assert model.calls[2]["tool_choice"] == "auto"
    assert model.calls[2]["settings"]["parallel_tool_calls"] is False
    assert "## Catalog Search Repair" in model.calls[2]["system_prompt"]
    assert "CATALOG CAPABILITIES (server-generated data)" in (
        model.calls[2]["system_prompt"]
    )
    assert "category=apparel" in model.calls[2]["system_prompt"]
    assert "subcategory=skirts" in model.calls[2]["system_prompt"]
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
    assert "category=apparel" not in repair_messages[-1]["text"]
    assert result["messages"][-1].content.startswith("I can’t map pants")
    assert result["messages"][-1].additional_kwargs[
        SERVER_CATALOG_CLARIFICATION
    ] is True


@pytest.mark.asyncio
async def test_compiled_agent_executes_capability_valid_repair(
    base_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repair may replace unvalidated catalog fields before execution."""

    from chain_server.src import deepagents_runtime as runtime_mod

    model_name = "compiled-capability-repair-test"
    base_config.llm_name = model_name
    model = _RecordingToolModel(
        model_name=model_name,
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "activate-skill",
                        "name": SKILL_ACTIVATION_TOOL_NAME,
                        "args": {"skill_names": ["product-discovery"]},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "invalid-search",
                        "name": "search_catalog_tool",
                        "args": {
                            "semantic_query": "black shoes",
                            "shopper_guidance": "Finding black shoes.",
                            "requested_product_type": "shoes",
                            "taxonomy": {
                                "category": ["footwear"],
                                "subcategory": ["shoes"],
                            },
                            "required_constraints": {"color": ["black"]},
                            "scope_complete": True,
                            "search_mode": "typo-mode",
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "valid-repair",
                        "name": "search_catalog_tool",
                        "args": {
                            "semantic_query": "black shoes",
                            "shopper_guidance": "Finding black shoes.",
                            "requested_product_type": "shoes",
                            "taxonomy": {
                                "category": ["footwear"],
                                "subcategory": ["flats"],
                            },
                            "required_constraints": {
                                "primary_color": ["black"]
                            },
                            "scope_complete": True,
                            "search_mode": "text",
                        },
                    }
                ],
            ),
            AIMessage(content="I found no black flats in the current catalog."),
        ],
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    executed_plans = []

    def execute_catalog_search(plan, *_args, **_kwargs):
        executed_plans.append(plan)
        return CatalogSearchExecution(
            result=SearchCatalogResult(ok=True, products=[])
        )

    monkeypatch.setattr(
        runtime_mod,
        "execute_catalog_search",
        execute_catalog_search,
    )
    identity = RequestIdentity(
        session_id="session-capability-repair",
        conversation_id="conversation-capability-repair",
        cart_id="cart-capability-repair",
        context_user_id=1,
        cart_user_id=1,
        request_id="request-capability-repair",
    )
    capabilities = CatalogCapabilities(
        catalog_id="test-catalog",
        retrieval_modes=["text"],
        filters={
            "category": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["category"],
                values=["footwear"],
            ),
            "subcategory": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["subcategory"],
                values=["flats"],
            ),
            "primary_color": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["primary_color"],
                values=["black"],
            )
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category",
            subcategory_field="subcategory",
            categories={
                "footwear": CatalogTaxonomyCategory(
                    product_count=1,
                    subcategories={
                        "flats": CatalogTaxonomySubcategory(product_count=1)
                    },
                )
            },
        ),
    )
    agent = runtime._create_agent(
        State(user_id=1, query="Show me black shoes."),
        identity,
        capabilities,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "REQUEST ID: request-capability-repair\n"
                        "USER QUERY: Show me black shoes."
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": identity.conversation_id}},
    )

    assert len(executed_plans) == 1
    assert executed_plans[0].hard_filters["primary_color"] == ["black"]
    assert "color" not in executed_plans[0].hard_filters
    assert executed_plans[0].search_mode == "text"
    assert model.calls[2]["tools"] == ["search_catalog_tool"]
    assert model.calls[3]["tools"] == []
    assert result["messages"][-1].content.startswith("I found no black flats")
