# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a search leaves out is part of what it found.

Three turns of J10 said something false, and all three said it by reading a
narrowed result as the whole catalog. One reported a product the shop sells
as not stocked, because a price filter removed it. One introduced four bags
as everything in the shop under fifty dollars, when forty-three products
qualified across five departments. One called four jewellery pieces the ones
the shop carries under $150, when seventeen do.

The assertions here run against the evidence the *grounding editor* reads,
not the tool result the agent reads. They are two different renderings --
the editor's is rebuilt from the typed artifact, deliberately without
parsing prose -- and the editor is the one that writes what the shopper
gets. A disclosure added only to the tool result reaches the component that
chooses tools and never the component that chooses words, which is how all
three of these survived being told.
"""

from __future__ import annotations

from types import SimpleNamespace

from chain_server.src.catalog_format import _format_excluded_near_miss
from chain_server.src.catalog_search import _a_category_the_shopper_did_not_name
from chain_server.src.grounding_evidence import _customer_safe_search_evidence
from shared.commerce_contracts import (
    CatalogTaxonomyCapabilities,
    Money,
    ProductSummary,
)


def _near_miss(name: str, amount: float) -> ProductSummary:
    return ProductSummary(
        product_id="generated:1",
        display_name=name,
        price=Money(amount=amount),
    )


def test_the_excluded_product_is_named_with_its_price() -> None:
    """The price is the answer. "Is it within $150" is answered by $169.99."""

    line = _format_excluded_near_miss(_near_miss("Southwest Bracelet", 169.99))

    assert "Southwest Bracelet" in line
    assert "169.99" in line


def test_the_excluded_product_is_evidence_and_not_an_offer() -> None:
    """It fails the search it came from, so it is not one of the results.

    Said plainly here because the turn reading it is holding a budget the
    product breaks, and the journey's rule is that an over-budget product
    must never arrive as though it qualified.
    """

    line = _format_excluded_near_miss(_near_miss("Southwest Bracelet", 169.99))

    assert "must not be offered" in line
    assert "never say this shop has no such product" in line


def test_nothing_excluded_says_nothing() -> None:
    assert _format_excluded_near_miss(None) == ""


def _browse(category: list[str], subcategory: list[str]) -> tuple:
    """A browse whose evidence is keyed the way the catalog names its fields.

    `department` and `product_type` rather than `category` and `subcategory`:
    the taxonomy evidence carries catalog field names, and reading it by the
    generic role names finds nothing on any catalog that calls them something
    else.
    """

    evidence = SimpleNamespace(
        taxonomy={"department": category, "product_type": subcategory}
    )
    attempt = SimpleNamespace(
        taxonomy_status="agent_selected_type",
        capabilities=SimpleNamespace(
            taxonomy=CatalogTaxonomyCapabilities(
                category_field="department",
                subcategory_field="product_type",
            )
        ),
    )
    return evidence, attempt


def test_a_department_chosen_for_the_shopper_is_named() -> None:
    evidence, attempt = _browse(["bags"], [])

    assert _a_category_the_shopper_did_not_name(evidence, attempt) == "bags"


def test_naming_the_subcategories_does_not_hide_the_department() -> None:
    """The bug behind J10 turn 1.

    The category and subcategory lists were flattened together and the
    disclosure wanted exactly one value across both, so it fired for a bare
    category and fell silent as soon as the scope named what was under it --
    which is what browsing a department looks like. "Nothing over $50"
    searched two subcategories of bags, disclosed nothing, and the reply
    called four bags everything in the shop.
    """

    evidence, attempt = _browse(["bags"], ["tote_bags", "crossbody_bags"])

    assert _a_category_the_shopper_did_not_name(evidence, attempt) == "bags"


def test_a_search_spanning_departments_narrowed_nothing() -> None:
    evidence, attempt = _browse(["bags", "jewelry"], [])

    assert _a_category_the_shopper_did_not_name(evidence, attempt) == ""


def test_a_type_the_shopper_named_is_not_a_department_chosen_for_them() -> None:
    evidence, _ = _browse(["bags"], ["tote_bags"])
    asked_for = SimpleNamespace(taxonomy_status="exact_requested_type")

    assert _a_category_the_shopper_did_not_name(evidence, asked_for) == ""


def _editor_reads(**payload: object) -> str:
    """The evidence the grounding editor is given for a search with results."""

    return _customer_safe_search_evidence(
        {
            "outcome": "results",
            "taxonomy": {"category": ["jewelry"]},
            "confirmed_filters": {"price": {"max": 110.01}},
            "products": [{"name": "Pearl Bracelet"}],
            **payload,
        }
    )


def test_the_editor_is_told_what_the_filter_removed() -> None:
    """J10 turn 6. The agent was told; the editor writes the reply."""

    lane = _editor_reads(
        excluded_near_miss={
            "display_name": "Southwest Bracelet",
            "price": 169.99,
        }
    )

    assert "Southwest Bracelet" in lane
    assert "169.99" in lane
    assert "Never say this shop has no such product" in lane


def test_the_editor_is_told_the_products_are_a_slice() -> None:
    """J10 turn 5. Four of seventeen, described as what the shop carries."""

    lane = _editor_reads(
        how_many_matched=17,
        products=[{"name": f"piece {index}"} for index in range(4)],
    )

    assert "at least 17 products matched" in lane
    assert "4 are shown" in lane
    assert "do not present them as everything" in lane


def test_a_complete_result_set_is_not_called_a_slice() -> None:
    """J10 turn 2 is true and has to stay sayable.

    All four tote bags under $50 really are all four. Only the count tells
    that sentence apart from the identical one over four of seventeen, so a
    rule that fired on both would make a correct turn worse.
    """

    lane = _editor_reads(
        how_many_matched=0,
        products=[{"name": f"tote {index}"} for index in range(4)],
    )

    assert "PARTIAL_RESULT_SET" not in lane


def test_a_search_that_excluded_nothing_says_nothing_about_exclusions() -> None:
    assert "EXCLUDED_BY_THIS_SEARCHS_FILTER" not in _editor_reads()
