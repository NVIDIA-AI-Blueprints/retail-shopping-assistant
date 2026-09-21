# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a search leaves out is part of what it found.

Two turns of J10 said something false, and both said it by reading a narrowed
result as the whole catalog. One reported a product the shop sells as not
stocked, because a price filter removed it. The other introduced four bags as
everything in the shop under fifty dollars, when forty-three products
qualified across five departments.
"""

from __future__ import annotations

from types import SimpleNamespace

from chain_server.src.catalog_search import _a_category_the_shopper_did_not_name
from chain_server.src.response_format import _format_excluded_near_miss
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
