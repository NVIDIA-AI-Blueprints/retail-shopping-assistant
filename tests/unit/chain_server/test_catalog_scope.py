# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-visible catalog search boundary."""


def test_model_catalog_search_has_no_semantic_relation_label() -> None:
    from chain_server.src.deepagents_runtime import SearchCatalogToolArguments

    assert "taxonomy_status" not in SearchCatalogToolArguments.model_fields


def test_catalog_search_rules_require_direct_clarification() -> None:
    from chain_server.src.catalog_scope import CATALOG_SEARCH_RULES

    assert "ask one concise clarification question directly" in CATALOG_SEARCH_RULES
    assert "taxonomy_status" not in CATALOG_SEARCH_RULES
    assert "no_direct_catalog_match" not in CATALOG_SEARCH_RULES
