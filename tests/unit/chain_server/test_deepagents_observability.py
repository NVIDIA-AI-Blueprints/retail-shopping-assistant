# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Deep Agents turn diagnostics."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from chain_server.src.agenttypes import State
from chain_server.src.deepagents_runtime import (
    DeepAgentsRuntime,
    RequestIdentity,
    _collect_agent_diagnostics,
)
from chain_server.src.skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
)
from chain_server.src.tool_loop_control import SEARCH_VALIDATION_ERROR_PREFIX


def test_tool_trace_preserves_model_order_arguments_skills_and_duplicates() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: old-request"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "old-search",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "old"},
                }
            ],
        ),
        ToolMessage(content="old result", tool_call_id="old-search"),
        HumanMessage(content="REQUEST ID: request-a\nUSER QUERY: What bottoms?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "skill-activation",
                    "name": SKILL_ACTIVATION_TOOL_NAME,
                    "args": {"skill_names": ["outfit-styling"]},
                }
            ],
        ),
        ToolMessage(
            content=(
                f"{SKILL_ACTIVATION_COMPLETE} "
                "/shopper/outfit-styling/SKILL.md"
            ),
            name=SKILL_ACTIVATION_TOOL_NAME,
            tool_call_id="skill-activation",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "bottoms-search",
                    "name": "search_catalog_tool",
                    "args": {
                        "semantic_query": "bottoms to coordinate with a beige top",
                        "taxonomy": {
                            "category": ["bottoms"],
                            "subcategory": ["pants", "shorts", "skirts"],
                        },
                        "required_constraints": {},
                    },
                },
            ],
        ),
        ToolMessage(content="catalog results", tool_call_id="bottoms-search"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "duplicate-search",
                    "name": "search_catalog_tool",
                    "args": {
                        "semantic_query": "more bottoms",
                        "taxonomy": {
                            "category": ["bottoms"],
                            "subcategory": ["pants", "shorts", "skirts"],
                        },
                        "required_constraints": {},
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                "STOP_TOOL_USE: This catalog taxonomy and constraint scope was "
                "already searched in this turn."
            ),
            tool_call_id="duplicate-search",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-a",
        final_termination_reason="completed",
    )

    assert diagnostics["skill_files_read"] == [
        "/shopper/outfit-styling/SKILL.md"
    ]
    assert [call["tool_name"] for call in diagnostics["tool_calls"]] == [
        SKILL_ACTIVATION_TOOL_NAME,
        "search_catalog_tool",
        "search_catalog_tool",
    ]
    assert diagnostics["tool_calls"][1]["arguments"]["semantic_query"] == (
        "bottoms to coordinate with a beige top"
    )
    assert diagnostics["tool_calls"][2] == {
        "sequence": 3,
        "tool_name": "search_catalog_tool",
        "arguments": {
            "semantic_query": "more bottoms",
            "taxonomy": {
                "category": ["bottoms"],
                "subcategory": ["pants", "shorts", "skirts"],
            },
            "required_constraints": {},
        },
        "status": "rejected",
        "rejection_reason": "duplicate_catalog_scope",
        "duplicate": True,
    }
    assert diagnostics["rejected_tool_calls"] == [3]
    assert diagnostics["duplicate_tool_calls"] == [3]
    assert diagnostics["final_termination_reason"] == "completed"
    assert diagnostics["partial_graph_messages"] == []


def test_tool_trace_records_pre_activation_execution_rejection() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: request-gated"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "premature-search",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "bottoms for a beige top"},
                }
            ],
        ),
        ToolMessage(
            content=SKILL_ACTIVATION_REQUIRED,
            name="search_catalog_tool",
            tool_call_id="premature-search",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-gated",
        final_termination_reason="completed",
    )

    assert diagnostics["tool_calls"][0]["status"] == "rejected"
    assert diagnostics["tool_calls"][0]["rejection_reason"] == (
        "skill_activation_required"
    )
    assert diagnostics["rejected_tool_calls"] == [1]
    assert diagnostics["skill_files_read"] == []


def test_tool_trace_distinguishes_rejected_error_and_pending_calls() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: request-b"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "limited-search",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "another category"},
                },
                {
                    "id": "failed-skill",
                    "name": "read_file",
                    "args": {"file_path": "/shopper/missing/SKILL.md"},
                },
                {
                    "id": "pending-detail",
                    "name": "get_product_details_tool",
                    "args": {"product_ref": "prod-1"},
                },
            ],
        ),
        ToolMessage(
            content="STOP_TOOL_USE: Catalog search limit reached for this turn.",
            tool_call_id="limited-search",
        ),
        ToolMessage(
            content="Error reading file: file not found",
            tool_call_id="failed-skill",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-b",
        final_termination_reason="recursion_limit",
        preserve_partial_messages=True,
    )

    assert [call["status"] for call in diagnostics["tool_calls"]] == [
        "rejected",
        "error",
        "pending",
    ]
    assert diagnostics["tool_calls"][0]["rejection_reason"] == (
        "catalog_search_limit"
    )
    assert diagnostics["skill_files_read"] == []
    assert diagnostics["rejected_tool_calls"] == [1]
    assert diagnostics["duplicate_tool_calls"] == []
    assert [message["type"] for message in diagnostics["partial_graph_messages"]] == [
        "ai",
        "tool",
        "tool",
    ]


def test_tool_trace_classifies_search_schema_errors_as_rejected() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: request-schema"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "invalid-search",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "pants"},
                }
            ],
        ),
        ToolMessage(
            content=(
                SEARCH_VALIDATION_ERROR_PREFIX
                + "{'taxonomy': {'subcategory': ['pants']}} with error: invalid"
            ),
            name="search_catalog_tool",
            tool_call_id="invalid-search",
            status="error",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-schema",
        final_termination_reason="completed",
    )

    assert diagnostics["tool_calls"][0]["status"] == "rejected"
    assert diagnostics["tool_calls"][0]["rejection_reason"] == (
        "invalid_catalog_request"
    )
    assert diagnostics["rejected_tool_calls"] == [1]


@pytest.mark.asyncio
async def test_stream_metrics_include_agent_diagnostics(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = DeepAgentsRuntime(base_config)
    state = State(user_id=1, query="hello", guardrails=False)
    state.response = "done"
    state.agent_diagnostics = {
        "skill_files_read": [],
        "tool_calls": [],
        "rejected_tool_calls": [],
        "duplicate_tool_calls": [],
        "final_termination_reason": "completed",
        "partial_graph_messages": [],
    }
    identity = RequestIdentity(
        session_id="session-a",
        conversation_id="conversation-a",
        cart_id="cart-a",
        context_user_id=1,
        cart_user_id=1,
        request_id="request-a",
    )

    async def fake_run_turn(*args, **kwargs):
        return state

    monkeypatch.setattr(runtime, "_run_turn", fake_run_turn)

    chunks = [json.loads(chunk) async for chunk in runtime.astream(state, identity)]

    assert [chunk["type"] for chunk in chunks] == ["images", "content", "metrics"]
    assert chunks[-1]["payload"]["agent_diagnostics"] == state.agent_diagnostics
