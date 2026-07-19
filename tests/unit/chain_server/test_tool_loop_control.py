# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deterministic Deep Agents tool-loop control."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from chain_server.src.tool_loop_control import (
    CONSTRAINT_REVIEW_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
    UNSUPPORTED_CONSTRAINT_PREFIX,
    UNSUPPORTED_TAXONOMY_PREFIX,
    ToolLoopControlMiddleware,
)


@tool
def activate_shopper_skills_tool(skill_names: list[str]) -> str:
    """Select shopper skills."""

    return ", ".join(skill_names)


@tool
def search_catalog_tool(semantic_query: str) -> str:
    """Search products."""

    return semantic_query


@tool
def get_cart_tool() -> str:
    """Read the cart."""

    return "empty"


TOOLS = [activate_shopper_skills_tool, search_catalog_tool, get_cart_tool]


def _model_request(messages: list[Any] | None = None) -> ModelRequest:
    messages = messages or [HumanMessage(content="shopper request")]
    return ModelRequest(
        model=cast(Any, object()),
        messages=messages,
        tools=TOOLS,
        state={"messages": messages},
    )


def _capture_model_request(
    middleware: ToolLoopControlMiddleware,
    messages: list[Any] | None = None,
) -> ModelRequest:
    captured: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    middleware.wrap_model_call(_model_request(messages), handler)
    return captured[0]


def _tool_result(
    content: str,
    *,
    tool_call_id: str = "call-a",
    status: str = "success",
) -> ToolMessage:
    return ToolMessage(
        content=content,
        name="search_catalog_tool",
        tool_call_id=tool_call_id,
        status=cast(Any, status),
    )


def _messages_with_result(result: ToolMessage) -> list[Any]:
    return [HumanMessage(content="shopper request"), result]


def test_normal_phase_preserves_activation_and_shopping_tools() -> None:
    prepared = _capture_model_request(ToolLoopControlMiddleware())

    assert prepared.tools == TOOLS


@pytest.mark.parametrize(
    "content",
    [
        "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already searched.",
        "STOP_TOOL_USE: Catalog search limit reached for this turn.",
        "STOP_TOOL_USE: Product-detail read limit reached for this turn.",
    ],
)
def test_stop_result_removes_tools_from_next_model_step(content: str) -> None:
    middleware = ToolLoopControlMiddleware()
    result = _tool_result(content)

    assert _capture_model_request(
        middleware,
        _messages_with_result(result),
    ).tools == []


def test_completed_search_scope_removes_tools_from_next_model_step() -> None:
    middleware = ToolLoopControlMiddleware()
    result = _tool_result(
        "SEARCH_RESULT_GROUNDING_NOTE: grounded candidates\n\n"
        "SEARCH_SCOPE_COMPLETE: This search covers every requested role."
    )

    prepared = _capture_model_request(
        middleware,
        _messages_with_result(result),
    )

    assert prepared.tools == []
    assert "## Tool Loop Closed" in prepared.system_prompt


def test_partial_search_scope_keeps_tools_available() -> None:
    middleware = ToolLoopControlMiddleware()
    result = _tool_result("SEARCH_RESULT_GROUNDING_NOTE: grounded candidates")

    assert _capture_model_request(
        middleware,
        _messages_with_result(result),
    ).tools == TOOLS


def test_closed_loop_strips_a_model_emitted_tool_call() -> None:
    middleware = ToolLoopControlMiddleware()
    messages = _messages_with_result(
        _tool_result("STOP_TOOL_USE: Catalog search limit reached.")
    )
    captured: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "ignored-search",
                            "name": "search_catalog_tool",
                            "args": {"semantic_query": "retry"},
                        }
                    ],
                )
            ]
        )

    response = middleware.wrap_model_call(_model_request(messages), handler)

    assert captured[0].tools == []
    assert captured[0].tool_choice == "none"
    assert "## Tool Loop Closed" in captured[0].system_prompt
    assert response.result[0].tool_calls == []
    assert "couldn't establish a reliable catalog match" in (
        response.result[0].content
    )


def test_no_direct_match_fallback_does_not_claim_grounded_products() -> None:
    middleware = ToolLoopControlMiddleware()
    messages = _messages_with_result(
        _tool_result(
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy matches "
            "casual sneakers."
        )
    )

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "ignored-search",
                            "name": "search_catalog_tool",
                            "args": {"semantic_query": "sandals"},
                        }
                    ],
                )
            ]
        )

    response = middleware.wrap_model_call(_model_request(messages), handler)

    assert "grounded options" not in response.result[0].content
    assert "reliable catalog match" in response.result[0].content


def test_closed_loop_strips_a_model_emitted_invalid_tool_call() -> None:
    middleware = ToolLoopControlMiddleware()
    messages = _messages_with_result(
        _tool_result("STOP_TOOL_USE: Catalog search limit reached.")
    )

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    invalid_tool_calls=[
                        {
                            "id": "invalid-search",
                            "name": "search_catalog_tool",
                            "args": "{not-json}",
                            "error": "invalid tool arguments",
                        }
                    ],
                )
            ]
        )

    response = middleware.wrap_model_call(_model_request(messages), handler)

    assert response.result[0].invalid_tool_calls == []
    assert "reliable catalog match" in response.result[0].content


def test_one_search_schema_repair_is_exposed_then_tools_are_removed() -> None:
    middleware = ToolLoopControlMiddleware()
    error = _tool_result(
        SEARCH_VALIDATION_ERROR_PREFIX + "{} with error: invalid taxonomy",
        status="error",
    )
    repair_messages = _messages_with_result(error)
    repair_request = _capture_model_request(middleware, repair_messages)
    repair_result = _tool_result(
        "catalog results\nSEARCH_SCOPE_COMPLETE: complete",
        tool_call_id="call-b",
    )
    completed_messages = [*repair_messages, repair_result]

    assert [tool.name for tool in repair_request.tools] == ["search_catalog_tool"]
    assert repair_request.model_settings["parallel_tool_calls"] is False
    assert "## Catalog Search Repair" in repair_request.system_prompt
    assert "exactly one category" in repair_request.system_prompt
    assert "agent_selected_type" in repair_request.system_prompt
    assert "separate the requested product type from its modifiers" in (
        repair_request.system_prompt
    )
    normalized_prompt = " ".join(repair_request.system_prompt.split())
    assert "subjective style stays in the semantic query" in normalized_prompt
    assert "supported alternative branch must still be searched" in normalized_prompt
    assert "every advertised subcategory that serves that role" in normalized_prompt
    assert "wet-weather outfit is context" in normalized_prompt
    assert _capture_model_request(middleware, completed_messages).tools == []


def test_incomplete_successful_repair_continues_without_second_repair() -> None:
    middleware = ToolLoopControlMiddleware()
    error = _tool_result(
        SEARCH_VALIDATION_ERROR_PREFIX + "{} with error: invalid taxonomy",
        status="error",
    )
    repair_messages = _messages_with_result(error)
    _capture_model_request(middleware, repair_messages)
    repaired_search = _tool_result(
        "SEARCH_RESULT_GROUNDING_NOTE: grounded candidates",
        tool_call_id="call-b",
    )
    continued_messages = [*repair_messages, repaired_search]

    assert _capture_model_request(middleware, continued_messages).tools == TOOLS

    second_error = _tool_result(
        SEARCH_VALIDATION_ERROR_PREFIX + "{} with error: still invalid",
        tool_call_id="call-c",
        status="error",
    )
    assert _capture_model_request(
        middleware,
        [*continued_messages, second_error],
    ).tools == []


def test_inferred_constraint_review_allows_one_correction() -> None:
    middleware = ToolLoopControlMiddleware()
    review = _tool_result(
        CONSTRAINT_REVIEW_PREFIX + " remove an inferred weather requirement"
    )

    prepared = _capture_model_request(middleware, _messages_with_result(review))

    assert [tool.name for tool in prepared.tools] == ["search_catalog_tool"]
    assert "preserve a product attribute directly stated" in prepared.system_prompt


@pytest.mark.parametrize(
    "content",
    [
        UNSUPPORTED_TAXONOMY_PREFIX + " pants is not advertised",
        UNSUPPORTED_CONSTRAINT_PREFIX + " water resistance is not enforceable",
    ],
)
def test_deterministic_search_refusal_closes_the_loop(content: str) -> None:
    middleware = ToolLoopControlMiddleware()

    assert _capture_model_request(
        middleware,
        _messages_with_result(_tool_result(content)),
    ).tools == []


def test_non_schema_error_does_not_change_loop_control() -> None:
    middleware = ToolLoopControlMiddleware()
    result = _tool_result("Catalog service unavailable.", status="error")

    assert _capture_model_request(
        middleware,
        _messages_with_result(result),
    ).tools == TOOLS


@pytest.mark.asyncio
async def test_async_stop_result_removes_tools() -> None:
    middleware = ToolLoopControlMiddleware()
    messages = _messages_with_result(
        _tool_result("STOP_TOOL_USE: Catalog search limit reached.")
    )
    captured: list[ModelRequest] = []

    async def model_handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    await middleware.awrap_model_call(_model_request(messages), model_handler)

    assert captured[0].tools == []
