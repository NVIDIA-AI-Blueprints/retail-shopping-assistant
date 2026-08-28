# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-visible catalog search boundary."""


def test_model_catalog_search_has_no_semantic_relation_label() -> None:
    from chain_server.src.turn_support import SearchCatalogToolArguments

    assert "taxonomy_status" not in SearchCatalogToolArguments.model_fields


def test_catalog_search_rules_allow_model_selected_parent_category() -> None:
    from chain_server.src.catalog_scope import CATALOG_SEARCH_RULES

    assert "not separately advertised" in CATALOG_SEARCH_RULES
    assert "denotes the same kind of thing" in CATALOG_SEARCH_RULES
    assert "keep the shopper's product type" in CATALOG_SEARCH_RULES.lower()
    assert "never put a product type in `unadvertised_requirements`" in (
        CATALOG_SEARCH_RULES
    )
    # A kind no advertised subcategory denotes is not carried, and saying so
    # is a fact off the published taxonomy -- not the banned guess from a
    # search that returned little.
    assert "it in `not_covered`" in CATALOG_SEARCH_RULES
    assert "do not call the tool at" in CATALOG_SEARCH_RULES
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


class TestResolutionReturnsStoredAttributes:
    """A product recovered from history carries the facts it was shown with.

    The presented-product event stores the whole ProductSummary, attributes
    included. Rendering only identity told the model to spend a round trip
    fetching what the lane already held, and the old wording claimed material
    and care *required* a detail read -- which was never true of this data.
    """

    def _product(self, attributes):
        from shared.commerce_contracts import Money, ProductSummary

        return ProductSummary(
            product_id="generated:abc",
            display_name="Ravenna Crossbody Bag",
            category="crossbody_bags",
            price=Money(amount=49.99, currency="USD"),
            attributes=attributes,
        )

    def test_stored_attributes_are_rendered(self) -> None:
        from chain_server.src.conversation_products import (
            _presented_attribute_facts,
        )

        facts = _presented_attribute_facts(
            self._product({"bag_closure": "zip", "composition": "leather"})
        )
        assert facts == {"bag_closure": "zip", "composition": "leather"}

    def test_marketing_copy_and_scores_never_reach_the_model(self) -> None:
        """catalog_text is marketing prose; similarity is a retrieval score."""

        from chain_server.src.conversation_products import (
            _presented_attribute_facts,
        )

        facts = _presented_attribute_facts(
            self._product(
                {
                    "bag_closure": "zip",
                    "catalog_text": "The only bag you will ever need!",
                    "similarity": 0.83,
                    "taxonomy": "bags > crossbody",
                }
            )
        )
        assert facts == {"bag_closure": "zip"}

    def test_a_product_with_no_attributes_renders_nothing(self) -> None:
        from chain_server.src.conversation_products import (
            _presented_attribute_facts,
        )

        assert _presented_attribute_facts(self._product({})) == {}

    def test_the_renderer_actually_emits_them(self) -> None:
        """The unit above can pass while the renderer drops the result.

        Caught by mutation: replacing the renderer's lookup with an empty dict
        left every other test green, so nothing asserted the attributes reach
        what the model reads.
        """

        from chain_server.src.conversation_products import (
            ConversationProductMatch,
            ProductReferenceResolution,
            ResolveConversationProductsResult,
            format_product_resolution,
        )

        product = self._product({"bag_closure": "zip", "composition": "leather"})
        result = ResolveConversationProductsResult(
            results=[
                ProductReferenceResolution(
                    reference_id="bag1",
                    status="resolved",
                    match_count=1,
                    matches=[
                        ConversationProductMatch(
                            product=product,
                            turn_sequence=1,
                            position=1,
                            candidate_set_id="set-1",
                        )
                    ],
                )
            ]
        )
        rendered = format_product_resolution(result)
        assert "CONFIRMED WHEN SHOWN:" in rendered
        assert "bag_closure: zip" in rendered
        assert "composition: leather" in rendered


class TestEveryDescriptionReachesTheModel:
    """A field description the model never reads is worse than none.

    `_search_catalog_tool_input_model` redefines `taxonomy` and
    `required_constraints` with the active catalog's enums, and redefining a
    field replaces the whole `FieldInfo` -- description included. Two long
    descriptions sat on the base class saying things the live ones did not
    say, for months, invisible: mutation-tested by replacing both with a
    placeholder, which left all 1343 tests green and never appeared in the
    rendered schema.

    So the rule is structural rather than textual. Any field the builder
    overrides must carry no description on the base class, because that text
    cannot reach the model and nothing else will notice when it drifts.
    """

    _OVERRIDDEN = ("taxonomy", "required_constraints")

    def _capabilities(self):
        from shared.commerce_contracts import (
            CatalogCapabilities,
            CatalogFilterCapability,
            CatalogTaxonomyCapabilities,
            CatalogTaxonomyCategory,
            CatalogTaxonomySubcategory,
        )

        return CatalogCapabilities(
            catalog_id="fashion",
            retrieval_modes=["text"],
            filters={
                "category": CatalogFilterCapability(
                    type="enum",
                    operators=["in"],
                    source_fields=["category"],
                    values=["apparel"],
                ),
                "subcategory": CatalogFilterCapability(
                    type="enum",
                    operators=["in"],
                    source_fields=["subcategory"],
                    values=["dresses"],
                ),
            },
            taxonomy=CatalogTaxonomyCapabilities(
                category_field="category",
                subcategory_field="subcategory",
                categories={
                    "apparel": CatalogTaxonomyCategory(
                        product_count=1,
                        subcategories={
                            "dresses": CatalogTaxonomySubcategory(product_count=1)
                        },
                    )
                },
            ),
        )

    def test_an_overridden_field_carries_no_dead_description(self) -> None:
        from chain_server.src.turn_support import SearchCatalogToolArguments

        for name in self._OVERRIDDEN:
            field = SearchCatalogToolArguments.model_fields[name]
            assert not field.description, (
                f"'{name}' is redefined by _search_catalog_tool_input_model, so "
                "a description here never reaches the model. Put it on the "
                "override instead."
            )

    def test_every_rendered_description_is_non_empty(self) -> None:
        """The other half: the override must actually supply one."""

        from chain_server.src.turn_support import (
            _search_catalog_scopes_input_model,
        )

        schema = _search_catalog_scopes_input_model(
            self._capabilities(), max_scopes=4
        ).model_json_schema()
        scope = schema["$defs"]["CatalogSearchToolArguments"]["properties"]
        for name in self._OVERRIDDEN:
            assert scope[name].get("description", "").strip(), (
                f"'{name}' reaches the model with no description at all"
            )

    def test_the_salvaged_taxonomy_rules_reach_the_model(self) -> None:
        """Both clauses were only ever in the dead copy, so assert the channel.

        Not the source: asserting the constant would have passed throughout the
        months the text was unreachable.
        """

        from chain_server.src.turn_support import (
            _search_catalog_scopes_input_model,
        )

        schema = _search_catalog_scopes_input_model(
            self._capabilities(), max_scopes=4
        ).model_json_schema()
        taxonomy = schema["$defs"]["CatalogSearchToolArguments"]["properties"][
            "taxonomy"
        ]["description"]
        assert "parent or sibling" in taxonomy
        assert "Pumps are heels" in taxonomy

    def test_advertised_values_never_enter_the_unadvertised_field(self) -> None:
        """It told the model to do the one thing the field is not for.

        "Before calling the tool, include every exact advertised filter value
        that modifies the target product" belongs to `required_constraints`,
        where it still is. On `unadvertised_requirements` it contradicted the
        field's own opening line.
        """

        from chain_server.src.turn_support import (
            _search_catalog_scopes_input_model,
        )

        schema = _search_catalog_scopes_input_model(
            self._capabilities(), max_scopes=4
        ).model_json_schema()
        unadvertised = schema["$defs"]["CatalogRequiredConstraints"]["properties"][
            "unadvertised_requirements"
        ]["description"]
        assert "advertised filter" not in unadvertised
        required = schema["$defs"]["CatalogSearchToolArguments"]["properties"][
            "required_constraints"
        ]["description"]
        assert "every exact matching filter value" in required
