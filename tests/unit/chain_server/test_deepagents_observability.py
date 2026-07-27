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
from chain_server.src.tool_loop_control import (
    SEARCH_VALIDATION_ERROR_PREFIX,
    SERVER_RESTORED_TOOL_CALL_FIELDS,
)


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


def test_tool_trace_records_native_repair_scope_rejection() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: request-repair-scope"),
        AIMessage(
            content="I couldn't establish a reliable catalog match.",
            additional_kwargs={
                "server_rejected_tool_calls": [
                    {
                        "id": "changed-repair",
                        "name": "search_catalog_tool",
                        "args": {"requested_product_type": "tote_bags"},
                        "rejection_reason": "repair_scope_changed",
                    }
                ]
            },
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-repair-scope",
        final_termination_reason="completed",
    )

    assert diagnostics["tool_calls"] == [
        {
            "sequence": 1,
            "tool_name": "search_catalog_tool",
            "arguments": {"requested_product_type": "tote_bags"},
            "status": "rejected",
            "rejection_reason": "repair_scope_changed",
        }
    ]
    assert diagnostics["rejected_tool_calls"] == [1]


def test_tool_trace_records_bounded_server_restored_fields() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: request-repair-restore"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "restored-repair",
                    "name": "search_catalog_tool",
                    "args": {
                        "requested_product_type": "sneakers",
                        "scope_complete": True,
                    },
                }
            ],
            additional_kwargs={
                SERVER_RESTORED_TOOL_CALL_FIELDS: [
                    {
                        "tool_call_id": "restored-repair",
                        "fields": ["scope_complete"],
                    }
                ]
            },
        ),
        ToolMessage(
            content="catalog results",
            name="search_catalog_tool",
            tool_call_id="restored-repair",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-repair-restore",
        final_termination_reason="completed",
    )

    assert diagnostics["tool_calls"] == [
        {
            "sequence": 1,
            "tool_name": "search_catalog_tool",
            "arguments": {
                "requested_product_type": "sneakers",
                "scope_complete": True,
            },
            "status": "completed",
            "restored_fields": ["scope_complete"],
        }
    ]


def test_tool_trace_preserves_bounded_catalog_scope_outcome() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: request-scope"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "zero-results",
                    "name": "search_catalog_tool",
                    "args": {
                        "semantic_query": "formal skirts",
                        "requested_product_type": "skirts",
                        "taxonomy": {
                            "category": ["apparel"],
                            "subcategory": ["skirts"],
                        },
                        "required_constraints": {},
                        "scope_complete": True,
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                "SEARCH_NO_MATCH_GROUNDING_NOTE: Zero products matched this "
                "exact advertised taxonomy and filter scope.\n\n"
                'CATALOG_SCOPE_OUTCOME: {"outcome": '
                '"zero_results", "requested_product_type": "skirts", '
                '"taxonomy": {"category": ["apparel"], '
                '"subcategory": ["skirts"]}, "confirmed_filters": {}}'
            ),
            name="search_catalog_tool",
            tool_call_id="zero-results",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="request-scope",
        final_termination_reason="completed",
    )

    assert diagnostics["catalog_scope_outcomes"] == [
        {
            "outcome": "zero_results",
            "requested_product_type": "skirts",
            "taxonomy": {
                "category": ["apparel"],
                "subcategory": ["skirts"],
            },
            "confirmed_filters": {},
        }
    ]


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


def test_product_evidence_contains_only_successful_current_turn_product_tools() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: old-request"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "old-search",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "old products"},
                }
            ],
        ),
        ToolMessage(
            content=(
                "SEARCH_RESULT_GROUNDING_NOTE\n"
                "PRODUCT_REF: old-product\n"
                "NAME: Old Product"
            ),
            tool_call_id="old-search",
        ),
        HumanMessage(content="REQUEST ID: evidence-request"),
        AIMessage(
            content="private reasoning must not be copied",
            tool_calls=[
                {
                    "id": "search",
                    "name": "search_catalog_tool",
                    "args": {
                        "semantic_query": "coordinate with beige top from last week"
                    },
                },
                {
                    "id": "detail",
                    "name": "get_product_details_tool",
                    "args": {"product_ref": "prod-1"},
                },
                {
                    "id": "failed-detail",
                    "name": "get_product_details_tool",
                    "args": {"product_ref": "error-product"},
                },
                {
                    "id": "cart",
                    "name": "get_cart_tool",
                    "args": {},
                },
            ],
        ),
        ToolMessage(
            content=(
                "SEARCH_RESULT_GROUNDING_NOTE\n"
                "SEARCH_DIRECTION_EVIDENCE: "
                '"coordinate with beige top from last week"\n'
                "SEARCH_FILTER_EVIDENCE: "
                '{"color": ["beige"], "price": {"max": 100}}\n'
                "SEARCH_TAXONOMY_EVIDENCE: "
                '{"category": ["bottoms"], "subcategory": ["pants"]}\n'
                "PRODUCT_REF: prod-1\n"
                "NAME: Sand Trousers\n"
                "CATEGORY: bottoms\n"
                "PRICE: $49.00 USD\n"
                "IMAGE_URL: https://catalog.invalid/prod-1.png\n"
                "PRODUCT_REF: prod-2\n"
                "NAME: Cream Pants\n"
                "CATEGORY: bottoms\n"
                "PRICE: $59.00 USD"
            ),
            name="search_catalog_tool",
            tool_call_id="search",
        ),
        ToolMessage(
            content=(
                "PRODUCT_DETAIL_GROUNDING_NOTE\n"
                "PRODUCT_REF: prod-1\n"
                "NAME: Sand Trousers\n"
                "CATEGORY: bottoms\n"
                "BRAND: Example Brand\n"
                "PRICE: $49.00 USD\n"
                "DETAILS:\n"
                "- material: cotton\n"
                "- care: machine wash"
            ),
            name="get_product_details_tool",
            tool_call_id="detail",
        ),
        ToolMessage(
            content=(
                "PRODUCT_DETAIL_GROUNDING_NOTE\n"
                "PRODUCT_REF: error-product\n"
                "NAME: Error Product"
            ),
            name="get_product_details_tool",
            tool_call_id="failed-detail",
            status="error",
        ),
        ToolMessage(
            content=(
                "SEARCH_RESULT_GROUNDING_NOTE\n"
                "PRODUCT_REF: ignored-cart-product\n"
                "NAME: Ignored Cart Product"
            ),
            name="get_cart_tool",
            tool_call_id="cart",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="evidence-request",
        final_termination_reason="completed",
    )

    assert diagnostics["product_evidence"] == [
        {
            "product_ref": "prod-1",
            "product_name": "Sand Trousers",
            "source_tool": "search_catalog_tool",
            "evidence_type": "search_result",
            "facts": {
                "category": "bottoms",
                "price": "$49.00 USD",
                "image_available": True,
            },
            "search_scope": {
                "taxonomy": {
                    "category": ["bottoms"],
                    "subcategory": ["pants"],
                },
                "confirmed_filters": {
                    "color": ["beige"],
                    "price": {"max": 100},
                },
            },
        },
        {
            "product_ref": "prod-2",
            "product_name": "Cream Pants",
            "source_tool": "search_catalog_tool",
            "evidence_type": "search_result",
            "facts": {
                "category": "bottoms",
                "price": "$59.00 USD",
                "image_available": False,
            },
            "search_scope": {
                "taxonomy": {
                    "category": ["bottoms"],
                    "subcategory": ["pants"],
                },
                "confirmed_filters": {
                    "color": ["beige"],
                    "price": {"max": 100},
                },
            },
        },
        {
            "product_ref": "prod-1",
            "product_name": "Sand Trousers",
            "source_tool": "get_product_details_tool",
            "evidence_type": "product_detail",
            "facts": {
                "category": "bottoms",
                "brand": "Example Brand",
                "price": "$49.00 USD",
                "image_available": False,
                "material": "cotton",
                "care": "machine wash",
            },
        },
    ]
    serialized = json.dumps(diagnostics["product_evidence"])
    assert "coordinate with beige top from last week" not in serialized
    assert "private reasoning must not be copied" not in serialized
    assert "https://catalog.invalid" not in serialized
    assert "old-product" not in serialized
    assert "error-product" not in serialized
    assert "ignored-cart-product" not in serialized
    assert diagnostics["product_evidence_truncated"] is False


def test_product_evidence_keeps_each_search_scope_with_its_products() -> None:
    messages = [
        HumanMessage(content="REQUEST ID: scoped-searches"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "red-apparel",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "red tops"},
                },
                {
                    "id": "black-footwear",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "black shoes"},
                },
            ],
        ),
        ToolMessage(
            content=(
                "SEARCH_RESULT_GROUNDING_NOTE\n"
                'SEARCH_FILTER_EVIDENCE: {"color": ["red"]}\n'
                "SEARCH_TAXONOMY_EVIDENCE: "
                '{"category": ["apparel"], "subcategory": ["tops"]}\n'
                "PRODUCT_REF: red-top\n"
                "NAME: Red Top\n"
                "CATEGORY: apparel"
            ),
            tool_call_id="red-apparel",
        ),
        ToolMessage(
            content=(
                "SEARCH_RESULT_GROUNDING_NOTE\n"
                'SEARCH_FILTER_EVIDENCE: {"color": ["black"]}\n'
                "SEARCH_TAXONOMY_EVIDENCE: "
                '{"category": ["footwear"], "subcategory": ["shoes"]}\n'
                "PRODUCT_REF: black-shoe\n"
                "NAME: Black Shoe\n"
                "CATEGORY: footwear"
            ),
            tool_call_id="black-footwear",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="scoped-searches",
        final_termination_reason="completed",
    )

    evidence = diagnostics["product_evidence"]
    assert [record["product_ref"] for record in evidence] == [
        "red-top",
        "black-shoe",
    ]
    assert evidence[0]["search_scope"] == {
        "taxonomy": {"category": ["apparel"], "subcategory": ["tops"]},
        "confirmed_filters": {"color": ["red"]},
    }
    assert evidence[1]["search_scope"] == {
        "taxonomy": {"category": ["footwear"], "subcategory": ["shoes"]},
        "confirmed_filters": {"color": ["black"]},
    }
    assert diagnostics["product_evidence_truncated"] is False


def test_product_evidence_bounds_facts_and_strings() -> None:
    long_value = "v" * 600
    detail_fields = "\n".join(
        f"- field-{index}: {long_value}" for index in range(45)
    )
    messages = [
        HumanMessage(content="REQUEST ID: bounded-request"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "bounded-detail",
                    "name": "get_product_details_tool",
                    "args": {"product_ref": "long-product"},
                },
                {
                    "id": "bounded-search",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "products"},
                },
            ],
        ),
        ToolMessage(
            content=(
                "PRODUCT_DETAIL_GROUNDING_NOTE\n"
                f"PRODUCT_REF: {'r' * 600}\n"
                f"NAME: {'n' * 600}\n"
                f"CATEGORY: {long_value}\n"
                "BRAND: Example\n"
                "PRICE: $1.00 USD\n"
                "DETAILS:\n"
                f"{detail_fields}"
            ),
            tool_call_id="bounded-detail",
        ),
        ToolMessage(
            content=(
                "SEARCH_RESULT_GROUNDING_NOTE\n"
                f'SEARCH_FILTER_EVIDENCE: {{"style": "{long_value}"}}\n'
                f'SEARCH_TAXONOMY_EVIDENCE: {{"category": ["{long_value}"]}}\n'
                "PRODUCT_REF: scoped-product\n"
                "NAME: Scoped Product"
            ),
            tool_call_id="bounded-search",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="bounded-request",
        final_termination_reason="completed",
    )

    evidence = diagnostics["product_evidence"]
    assert len(evidence) == 2
    assert len(evidence[0]["product_ref"]) == 500
    assert len(evidence[0]["product_name"]) == 500
    assert len(evidence[0]["facts"]) == 40
    assert len(evidence[0]["facts"]["category"]) == 500
    assert len(evidence[0]["facts"]["field-0"]) == 500
    assert len(evidence[1]["search_scope"]["taxonomy"]["category"][0]) == 500
    assert len(evidence[1]["search_scope"]["confirmed_filters"]["style"]) == 500
    assert diagnostics["product_evidence_truncated"] is False


def test_product_evidence_reports_record_limit_truncation() -> None:
    products = "\n".join(
        f"PRODUCT_REF: product-{index}\nNAME: Product {index}"
        for index in range(25)
    )
    messages = [
        HumanMessage(content="REQUEST ID: record-limit"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "many-products",
                    "name": "search_catalog_tool",
                    "args": {"semantic_query": "products"},
                }
            ],
        ),
        ToolMessage(
            content=f"SEARCH_RESULT_GROUNDING_NOTE\n{products}",
            tool_call_id="many-products",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="record-limit",
        final_termination_reason="completed",
    )

    assert len(diagnostics["product_evidence"]) == 24
    assert diagnostics["product_evidence_truncated"] is True


def test_product_evidence_reports_serialized_size_truncation() -> None:
    long_value = "v" * 600
    detail_fields = "\n".join(
        f"- field-{index}: {long_value}" for index in range(45)
    )
    products = "\n".join(
        (
            f"PRODUCT_REF: large-{index}\n"
            f"NAME: Large Product {index}\n"
            "CATEGORY: apparel\n"
            "BRAND: Example\n"
            "PRICE: $1.00 USD\n"
            "DETAILS:\n"
            f"{detail_fields}"
        )
        for index in range(2)
    )
    messages = [
        HumanMessage(content="REQUEST ID: aggregate-limit"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "large-details",
                    "name": "get_product_details_tool",
                    "args": {"product_ref": "large-0"},
                }
            ],
        ),
        ToolMessage(
            content=f"PRODUCT_DETAIL_GROUNDING_NOTE\n{products}",
            tool_call_id="large-details",
        ),
    ]

    diagnostics = _collect_agent_diagnostics(
        messages,
        request_id="aggregate-limit",
        final_termination_reason="completed",
    )

    evidence = diagnostics["product_evidence"]
    assert len(json.dumps(evidence, sort_keys=True, default=str)) <= 32_000
    assert [record["product_ref"] for record in evidence] == ["large-0"]
    assert diagnostics["product_evidence_truncated"] is True


@pytest.mark.asyncio
async def test_query_responses_hide_agent_diagnostics_by_default(
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
        "product_evidence": [],
        "product_evidence_truncated": False,
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
    result = await runtime.ainvoke(state, identity)

    assert [chunk["type"] for chunk in chunks] == ["images", "content", "metrics"]
    assert chunks[-1]["payload"]["agent_diagnostics"] == {}
    assert result["agent_diagnostics"] == {}


@pytest.mark.asyncio
async def test_trusted_query_responses_can_expose_agent_diagnostics(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config.expose_agent_diagnostics = True
    runtime = DeepAgentsRuntime(base_config)
    state = State(user_id=1, query="hello", guardrails=False)
    state.response = "done"
    state.agent_diagnostics = {
        "skill_files_read": ["/shopper/product-discovery/SKILL.md"],
        "tool_calls": [],
        "final_termination_reason": "completed",
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
    result = await runtime.ainvoke(state, identity)

    assert chunks[-1]["payload"]["agent_diagnostics"] == state.agent_diagnostics
    assert result["agent_diagnostics"] == state.agent_diagnostics
