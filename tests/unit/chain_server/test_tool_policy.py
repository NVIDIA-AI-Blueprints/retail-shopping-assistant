# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deterministic shopper-skill tool policy."""

from __future__ import annotations

import pytest

from chain_server.src.tool_policy import (
    SHOPPING_TOOL_POLICIES,
    granted_tools_for_skills,
    tool_is_granted,
    validate_registered_tool_names,
    validate_skill_tool_grants,
)


def _skill_tool_grants() -> dict[str, frozenset[str]]:
    grants = {
        skill_name: frozenset(
            tool_name
            for tool_name, policy in SHOPPING_TOOL_POLICIES.items()
            if skill_name in policy.allowed_skills_any_of
        )
        for skill_name in {
            skill_name
            for policy in SHOPPING_TOOL_POLICIES.values()
            for skill_name in policy.allowed_skills_any_of
        }
    }
    grants["budget-shopping"] = frozenset()
    return grants


def test_policy_covers_all_registered_shopping_tools() -> None:
    assert set(SHOPPING_TOOL_POLICIES) == {
        "add_cart_items_tool",
        "check_active_promotions_tool",
        "check_product_availability_tool",
        "get_cart_tool",
        "get_product_details_tool",
        "get_store_policy_tool",
        "remove_cart_item_tool",
        "resolve_conversation_products_tool",
        "search_catalog_tool",
        "update_cart_items_tool",
        "view_cart_total_tool",
    }
    assert {
        name
        for name, policy in SHOPPING_TOOL_POLICIES.items()
        if policy.risk == "mutating"
    } == {
        "add_cart_items_tool",
        "remove_cart_item_tool",
        "update_cart_items_tool",
    }
    resolver = SHOPPING_TOOL_POLICIES["resolve_conversation_products_tool"]
    assert resolver.risk == "read"
    assert resolver.allowed_skills_any_of == frozenset(
        {"cart-management", "outfit-styling", "product-discovery"}
    )
    promotions = SHOPPING_TOOL_POLICIES["check_active_promotions_tool"]
    assert promotions.risk == "read"
    assert promotions.allowed_skills_any_of == frozenset(
        {"outfit-styling", "product-discovery"}
    )


def test_frontmatter_grants_and_execution_policy_must_match() -> None:
    grants = _skill_tool_grants()

    validate_skill_tool_grants(grants)

    drifted = dict(grants)
    drifted["outfit-styling"] = frozenset(
        {*drifted["outfit-styling"], "add_cart_items_tool"}
    )
    with pytest.raises(ValueError, match="unexpected=.*add_cart_items_tool"):
        validate_skill_tool_grants(drifted)


def test_registered_tool_names_must_match_policy() -> None:
    validate_registered_tool_names(set(SHOPPING_TOOL_POLICIES))

    with pytest.raises(ValueError, match="missing=.*view_cart_total_tool"):
        validate_registered_tool_names(
            set(SHOPPING_TOOL_POLICIES).difference({"view_cart_total_tool"})
        )


def test_selected_skills_receive_only_their_declared_union() -> None:
    grants = _skill_tool_grants()

    tools = granted_tools_for_skills(
        ["outfit-styling", "budget-shopping"],
        grants,
    )

    assert tools == frozenset(
        {
            "check_active_promotions_tool",
            "check_product_availability_tool",
            "get_product_details_tool",
            "resolve_conversation_products_tool",
            "search_catalog_tool",
        }
    )
    assert tool_is_granted(
        "search_catalog_tool",
        ["outfit-styling", "budget-shopping"],
        tools,
    )
    assert not tool_is_granted(
        "add_cart_items_tool",
        ["outfit-styling", "budget-shopping"],
        tools,
    )
