# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-visible catalog search boundary."""


def test_model_catalog_search_has_no_semantic_relation_label() -> None:
    from chain_server.src.turn_support import SearchCatalogToolArguments

    assert "taxonomy_status" not in SearchCatalogToolArguments.model_fields


def test_catalog_search_rules_allow_model_selected_parent_category() -> None:
    from chain_server.src.catalog_scope import CATALOG_SEARCH_RULES

    assert "not separately advertised" in CATALOG_SEARCH_RULES
    assert "one faithful advertised parent category" in CATALOG_SEARCH_RULES
    assert "keep the shopper's product type" in CATALOG_SEARCH_RULES.lower()
    assert "never put a product type in `unadvertised_requirements`" in (
        CATALOG_SEARCH_RULES
    )
    assert "ask one concise clarification question directly" in CATALOG_SEARCH_RULES
    assert "taxonomy_status" not in CATALOG_SEARCH_RULES
    assert "no_direct_catalog_match" not in CATALOG_SEARCH_RULES


class TestDetailReadRedundancy:
    """A search returns what a detail read does -- but prove it per catalog.

    Measured on the fashion catalog: across all five advertised categories and
    20 products, a detail read returned no field the search had not already
    supplied. That justifies skipping the round trip, but not assuming it for
    every catalog -- silently dropping a field is worse than a redundant call,
    so the check asks the capability contract rather than trusting the measurement.
    """

    def _capabilities(self):
        """A contract advertising two detail fields on `bags`."""

        from types import SimpleNamespace

        detail = SimpleNamespace(detail=True)
        plain = SimpleNamespace(detail=False)
        bags = SimpleNamespace(
            filters={"bag_closure": detail, "structure": detail, "price": plain},
            subcategories={"tote_bags": object()},
        )
        return SimpleNamespace(
            taxonomy=SimpleNamespace(categories={"bags": bags})
        )

    def _product(self, category, attributes):
        from types import SimpleNamespace

        return SimpleNamespace(category=category, attributes=attributes)

    def test_full_evidence_makes_the_read_redundant(self) -> None:
        from chain_server.src.turn_support import _detail_fields_already_held

        product = self._product("bags", {"bag_closure": "zip", "structure": "soft"})
        assert _detail_fields_already_held(product, self._capabilities())

    def test_a_subcategory_is_matched_to_its_category(self) -> None:
        from chain_server.src.turn_support import _detail_fields_already_held

        product = self._product(
            "tote_bags", {"bag_closure": "zip", "structure": "soft"}
        )
        assert _detail_fields_already_held(product, self._capabilities())

    def test_one_missing_field_still_reads(self) -> None:
        """Any gap must fetch. This is the property that keeps it safe."""

        from chain_server.src.turn_support import _detail_fields_already_held

        product = self._product("bags", {"bag_closure": "zip"})
        assert not _detail_fields_already_held(product, self._capabilities())

    def test_a_product_recovered_from_history_still_reads(self) -> None:
        """The historical index stores identity only, so it has no attributes."""

        from chain_server.src.turn_support import _detail_fields_already_held

        assert not _detail_fields_already_held(
            self._product("bags", {}), self._capabilities()
        )

    def test_an_unknown_category_still_reads(self) -> None:
        """Never skip a read for a product whose category cannot be checked."""

        from chain_server.src.turn_support import _detail_fields_already_held

        product = self._product("cookware", {"bag_closure": "zip"})
        assert not _detail_fields_already_held(product, self._capabilities())
