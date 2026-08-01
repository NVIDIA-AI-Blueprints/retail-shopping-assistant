# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-visible catalog search boundary."""


def test_model_catalog_search_has_no_semantic_relation_label() -> None:
    from chain_server.src.catalog_tool_contract import SearchCatalogToolArguments

    assert "taxonomy_status" not in SearchCatalogToolArguments.model_fields


def test_catalog_search_scope_identity_is_order_independent() -> None:
    from chain_server.src.catalog_tool_contract import catalog_search_scope

    first = catalog_search_scope(
        {"subcategory": ["heels", "flats"], "category": ["footwear"]},
        {"price": {"max": 100, "min": 25}, "primary_color": ["navy", "black"]},
    )
    second = catalog_search_scope(
        {"category": ["footwear"], "subcategory": ["flats", "heels"]},
        {"primary_color": ["black", "navy"], "price": {"min": 25, "max": 100}},
    )

    assert first == second


def test_catalog_search_rules_allow_model_selected_parent_category() -> None:
    from chain_server.src.catalog_scope import CATALOG_SEARCH_RULES

    assert "not separately advertised" in CATALOG_SEARCH_RULES
    assert "one faithful advertised parent category" in CATALOG_SEARCH_RULES
    assert "keep the shopper's product type" in CATALOG_SEARCH_RULES.lower()
    assert "never put a product type in `unadvertised_requirements`" in (
        CATALOG_SEARCH_RULES
    )
    assert "ask one concise clarification question directly" in CATALOG_SEARCH_RULES
    assert "treat the title as product identity" in CATALOG_SEARCH_RULES
    assert "do not split words inside the title" in CATALOG_SEARCH_RULES
    assert "taxonomy_status" not in CATALOG_SEARCH_RULES
    assert "no_direct_catalog_match" not in CATALOG_SEARCH_RULES
