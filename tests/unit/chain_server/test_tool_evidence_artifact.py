# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalog evidence contract: what the model reads, what the composer gets.

Catalog tools build typed evidence and render the model-visible text *from* it,
carrying the payload on the tool artifact. Nothing parses catalog facts back out
of that prose.

The expectations here are written out in full rather than derived, so that any
change to what the model reads -- or to what the composer is allowed to say --
shows up as a visible diff in review rather than as a quietly different string.
That matters most for the composer summaries: they are the boundary between
"the catalog established this" and "the assistant asserted this", and a fact
that silently drops out of one of them becomes an unsupported claim.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from chain_server.src import grounding_evidence as grounding_evidence_mod
from chain_server.src import response_format
from chain_server.src import turn_support as runtime_mod_support
from chain_server.src.tool_evidence import (
    DETAIL_EVIDENCE_KEY,
    EVIDENCE_KEY,
    ProductDetailEvidence,
    SearchEvidence,
    detail_evidence_of,
    evidence_of,
)
from shared.commerce_contracts import Money, ProductDetail, ProductSummary


# The shipped module composes these two calls where it needs the text. It
# used to also carry a wrapper for each, which only these tests called.
def _rendered_product(product):
    return response_format._format_product_record(
        runtime_mod_support._search_product_record(product)
    )


def _rendered_product_details(detail):
    return response_format._format_product_detail_record(
        runtime_mod_support._product_detail_record(detail)
    )


# A product record carries the product and nothing else. The attribute-limit
# note is a fact about the search, so it is said once per result rather than
# once per hit, and the image URL is not said at all -- the model cannot open
# it and the UI reads it from the typed artifact.


class _StubToolMessage:
    """Minimal stand-in for the tool message the composer reads."""

    def __init__(
        self,
        evidence: SearchEvidence | None = None,
        *,
        artifact: dict[str, Any] | None = None,
    ) -> None:
        self.artifact = artifact if evidence is None else evidence.as_artifact()


# --------------------------------------------------------------------------
# What the model reads
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("product", "expected"),
    [
        pytest.param(
            ProductSummary(
                product_id="prod_1",
                display_name="Ravenna Crossbody Bag",
                description="structured",
                category="Bags",
                price=Money(amount=49.99),
                image_url="https://example.invalid/a.jpg",
            ),
            "PRODUCT_REF: prod_1\n"
            "NAME: Ravenna Crossbody Bag\n"
            "CATEGORY: Bags\n"
            "PRICE: $49.99 USD",
            id="all-fields",
        ),
        pytest.param(
            ProductSummary(
                product_id="prod_2", display_name="Plain Tee", description=""
            ),
            "PRODUCT_REF: prod_2\nNAME: Plain Tee",
            id="no-optional-fields",
        ),
        pytest.param(
            ProductSummary(
                product_id="prod_3",
                display_name="Café Blazer — élan",
                description="",
                category="Outerwear",
                price=Money(amount=1234.5, currency="EUR"),
            ),
            "PRODUCT_REF: prod_3\n"
            "NAME: Café Blazer — élan\n"
            "CATEGORY: Outerwear\n"
            "PRICE: $1234.50 EUR",
            id="non-ascii-name-and-currency",
        ),
        pytest.param(
            ProductSummary(
                product_id="prod_4",
                display_name="Rounding Check",
                description="",
                price=Money(amount=7.0),
                image_url="https://example.invalid/b.png",
            ),
            "PRODUCT_REF: prod_4\n"
            "NAME: Rounding Check\n"
            "PRICE: $7.00 USD",
            id="price-keeps-two-decimals",
        ),
    ],
)
def test_search_result_text_the_model_reads(
    product: ProductSummary, expected: str
) -> None:
    assert _rendered_product(product) == expected


@pytest.mark.parametrize(
    ("detail", "expected_body"),
    [
        pytest.param(
            ProductDetail(
                product_id="det_1",
                display_name="Zephyr Linen Skirt",
                description="",
                category="skirt",
                brand="Example Brand",
                price=Money(amount=39.99),
                image_url="/images/zephyr.jpg",
                attributes={
                    "care": "Machine wash cold.",
                    "composition": "100% linen",
                },
            ),
            "PRODUCT_REF: det_1\n"
            "NAME: Zephyr Linen Skirt\n"
            "CATEGORY: skirt\n"
            "BRAND: Example Brand\n"
            "PRICE: $39.99 USD\n"
            "IMAGE_URL: /images/zephyr.jpg\n"
            "DETAILS:\n"
            "- care: Machine wash cold.\n"
            "- composition: 100% linen",
            id="all-fields",
        ),
        pytest.param(
            ProductDetail(
                product_id="det_2", display_name="Bare Item", description=""
            ),
            "PRODUCT_REF: det_2\nNAME: Bare Item\nNO_ADDITIONAL_STRUCTURED_DETAILS",
            id="no-attributes",
        ),
        pytest.param(
            ProductDetail(
                product_id="det_3",
                display_name="Trail Boot",
                description="",
                category="boots",
                price=Money(amount=120.0, currency="GBP"),
                attributes={
                    "outer_material": ["leather", "mesh"],
                    "care_instructions": "Wipe clean",
                    "dimensions": {"height": 12, "width": 4},
                },
            ),
            # Attributes are sorted, underscores become spaces, and list and
            # dict values keep their own formatting.
            "PRODUCT_REF: det_3\n"
            "NAME: Trail Boot\n"
            "CATEGORY: boots\n"
            "PRICE: $120.00 GBP\n"
            "DETAILS:\n"
            "- care instructions: Wipe clean\n"
            "- dimensions: height=12, width=4\n"
            "- outer material: leather, mesh",
            id="sorted-underscored-and-structured-values",
        ),
    ],
)
def test_product_detail_text_the_model_reads(
    detail: ProductDetail, expected_body: str
) -> None:
    expected = f"{response_format._PRODUCT_DETAIL_GROUNDING_NOTE}\n{expected_body}"

    assert _rendered_product_details(detail) == expected


# --------------------------------------------------------------------------
# What the composer is given
# --------------------------------------------------------------------------


def _results_evidence() -> SearchEvidence:
    return SearchEvidence(
        outcome="results",
        taxonomy={"category": ["Bags"], "product_type": ["crossbody bag"]},
        confirmed_filters={"primary_color": ["black"]},
        semantic_query="black work bag",
        shopper_guidance="prefers structured shapes",
        requested_product_type="crossbody bag",
        advertised_category="Bags",
        products=[
            {
                "product_ref": "prod_1",
                "name": "Ravenna Crossbody Bag",
                "category": "Bags",
                "price": "$49.99 USD",
                "image_url": "https://example.invalid/a.jpg",
            },
            {
                "product_ref": "prod_2",
                "name": "Plain Tee",
                "category": "",
                "price": "",
                "image_url": "",
            },
        ],
    )


def test_composer_summary_for_search_results() -> None:
    """Every fact the composer may repeat, and the limits placed on it."""

    summary = grounding_evidence_mod._customer_safe_tool_evidence(
        "", _StubToolMessage(_results_evidence())
    )

    assert summary.split('SPEAK AS A SHOP ASSISTANT')[0].rstrip() == (
        "CUSTOMER_SAFE_SEARCH_EVIDENCE: Search results support product names, "
        "prices, categories, image availability, confirmed search filters, any "
        "attribute listed as confirmed for that specific product, and a modest "
        "styling role. An attribute confirmed for one product is not evidence "
        "about another. They do not support care, construction, fit, comfort, "
        "weather, grass, gravel, heat, or best-in-category claims, nor any "
        "attribute not listed for that product. Treat names as display names, "
        "not attribute evidence; group claims require the attribute confirmed "
        "on every item.\n"
        "CONFIRMED_SEARCH_FILTERS: Every product below passed each filter "
        "predicate. A one-value list confirms that value; a multi-value list "
        "confirms only membership in the set, not which value matched: "
        '{"primary_color": ["black"]}\n'
        "ADVERTISED_SEARCH_TAXONOMY: This search used only these advertised "
        "taxonomy values. Lists are inclusive scopes; they do not mean every "
        "product has every value. Do not describe an unlisted product type as "
        'advertised: {"category": ["Bags"], "product_type": ["crossbody bag"]}\n'
        "- Ravenna Crossbody Bag | category: Bags | price: $49.99 USD | "
        "image: available\n"
        "- Plain Tee\n"
        "REQUESTED_SCOPE_RELATION: crossbody bag is not separately advertised. "
        "The search used the broader advertised category Bags. Present these "
        "as closest options and keep every returned product's actual catalog "
        "category; do not relabel them as the requested type."
    )
    # The shopper's own wording and internal identifiers never reach the composer.
    assert "black work bag" not in summary
    assert "PRODUCT_REF" not in summary
    assert "prod_1" not in summary


def test_search_results_carry_the_attributes_the_catalog_confirmed() -> None:
    """The catalog sends these with every result; dropping them cost a read.

    Before this, the runtime discarded the structured attributes and told the
    model they were available only from a product-detail read -- one of two per
    turn. When that budget ran out the assistant reported a confirmed attribute
    as unknown.
    """

    product = SimpleNamespace(
        product_id="prod_9",
        display_name="Black Satin Lace-Up Dress",
        category="dresses",
        price=Money(amount=69.99),
        image_url="/images/d.jpg",
        attributes={
            "composition": "100% satin",
            "garment_length": "maxi",
            "neckline": "off shoulder",
            # not attributes: retrieval bookkeeping and the prose serialisation
            "similarity": 0.67,
            "taxonomy": {"category": "apparel"},
            "catalog_text": "name: ...\nsummary: An elegant black satin gown",
        },
    )

    record = runtime_mod_support._search_product_record(product)

    assert record["attributes"] == {
        "composition": "100% satin",
        "garment_length": "maxi",
        "neckline": "off shoulder",
    }
    text = response_format._format_product_record(record)
    assert "CONFIRMED_ATTRIBUTES:" in text
    assert "- composition: 100% satin" in text
    assert "- garment length: maxi" in text
    # The marketing summary is never forwarded.
    assert "elegant" not in text
    assert "catalog_text" not in text
    assert "similarity" not in text


def test_composer_may_state_a_confirmed_attribute() -> None:
    """Widening what search carries is useless if the editor still strips it."""

    evidence = _results_evidence()
    evidence.products[0]["attributes"] = {"composition": "100% satin"}

    summary = grounding_evidence_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert "confirmed: composition: 100% satin" in summary
    assert "any attribute listed as confirmed for that specific product" in summary
    # And it must not become licence to claim it for the others.
    assert "not evidence about another" in summary


def test_composer_summary_for_zero_results() -> None:
    """Zero matches bound a scope; they never support catalog-wide absence."""

    evidence = SearchEvidence(
        outcome="zero_results",
        taxonomy={"category": ["Bags"]},
        confirmed_filters={"primary_color": ["black"]},
        requested_product_type="crossbody bag",
        advertised_category="Bags",
    )

    summary = grounding_evidence_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert summary.split('SPEAK AS A SHOP ASSISTANT')[0].rstrip() == (
        "CUSTOMER_SAFE_SCOPED_NO_MATCH_EVIDENCE: Zero products matched only "
        "the exact advertised search scope below. This does not establish that "
        "a different, unsearched, or unadvertised product type is absent, and "
        "it does not support a catalog-wide availability claim.\n"
        'ADVERTISED_SEARCH_TAXONOMY: {"category": ["Bags"]}\n'
        'CONFIRMED_SEARCH_FILTERS: {"primary_color": ["black"]}\n'
        "REQUESTED_SCOPE_RELATION: crossbody bag is not separately advertised. "
        "The broader advertised category Bags returned zero products for this "
        "search, so do not claim that the requested type is absent from the "
        "whole catalog.\n"
        # Absence alone made the model ask which alternative the shopper
        # wanted, showing none of them. The retry that answered that ran here,
        # on the server, and kept the size while dropping the product type --
        # boots, for a tote bag in a size 8. So the retry is named as the
        # model's to make, along with which filter may give and which may not.
        #
        # "Keep the size" then had to earn an exception. Asked for a tote in a
        # size 8, the model had no optional filter to give up and obeyed: it
        # showed nothing and asked whether to show the totes it had. A size
        # that is the whole request is the one thing left to relax, and the
        # sizes actually stocked are what the shopper is owed instead.
        "NEXT: nothing in the catalog matched all of these at once. Search "
        "again yourself, now, with one optional requirement dropped -- colour, "
        "pattern, style or price. Keep the product type and keep the size: a "
        "shopper asking for a dress is not answered with a skirt, and a size "
        "is a fact about a body, not a preference. Then show what that finds "
        "and say plainly which requirement could not be met. If the size is "
        "the only requirement there is, drop the size instead and search "
        "again: say first that nothing comes in the size they asked for, name "
        "the sizes these do come in -- one size, or a range that excludes "
        "theirs -- and never present them as the size they asked for. If the "
        "product type itself is one this shop does not carry, say that instead "
        "and search no further. Do not answer with a question alone, and never "
        "offer a choice between things the shopper cannot see: asking \"shall "
        "I show you the ones we do have\" is that refusal wearing a question "
        "mark. Show them."
    )


def test_composer_summary_for_product_detail() -> None:
    """Detail reads carry their fields, and a warning that nothing else is known."""

    record = {
        "product_ref": "det_1",
        "name": "Zephyr Linen Skirt",
        "category": "skirt",
        "brand": "Example Brand",
        "price": "$39.99 USD",
        "image_url": "/images/zephyr.jpg",
        "details": ["care: Machine wash cold.", "composition: 100% linen"],
    }

    summary = grounding_evidence_mod._customer_safe_tool_evidence(
        "",
        _StubToolMessage(
            artifact=ProductDetailEvidence(products=[record]).as_artifact()
        ),
    )

    assert summary.split('SPEAK AS A SHOP ASSISTANT')[0].rstrip() == (
        "CUSTOMER_SAFE_PRODUCT_DETAIL_EVIDENCE: Product details were read for "
        "these products, but the available detail data contains only the "
        "listed facts. Do not state material, care, dimensions, closures, fit, "
        "sizing, colorways, or outdoor performance unless the field appears in "
        "this evidence summary.\n"
        "- Zephyr Linen Skirt | category: skirt | price: $39.99 USD | "
        "image: available | details: care: Machine wash cold.; "
        "composition: 100% linen"
    )


def test_no_direct_catalog_match_is_a_refusal_not_an_empty_search() -> None:
    """Running no retrieval and finding nothing are different claims.

    Summarising this as an empty search result would let the composer imply the
    catalog was checked and came back empty.
    """

    evidence = SearchEvidence(
        outcome="no_direct_catalog_match",
        requested_product_type="casual sneakers",
    )

    summary = grounding_evidence_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert summary.startswith("CUSTOMER_SAFE_NO_MATCH_EVIDENCE:")
    assert "No retrieval ran" in summary
    assert "Do not name alternatives" in summary
    # The requested type is withheld so the composer cannot echo it as advertised.
    assert "casual sneakers" not in summary
    assert "CUSTOMER_SAFE_SEARCH_EVIDENCE" not in summary


def test_scope_relation_is_absent_when_no_parent_was_substituted() -> None:
    """A plain search must not gain a parent-scope caveat it never earned."""

    evidence = _results_evidence()
    evidence.advertised_category = None

    summary = grounding_evidence_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert "REQUESTED_SCOPE_RELATION" not in summary


def test_empty_results_do_not_append_a_blank_line() -> None:
    """With no products there is no prose to fall back to, so none is faked."""

    evidence = _results_evidence()
    evidence.products = []

    summary = grounding_evidence_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert not summary.endswith("\n")


# --------------------------------------------------------------------------
# How the payload travels
# --------------------------------------------------------------------------


def test_evidence_is_readable_from_object_and_dict_shaped_messages() -> None:
    """Messages reach this code both ways; both must be readable.

    Failing to read one shape would silently fall through to a summary built
    from something other than the search's own facts.
    """

    artifact = _results_evidence().as_artifact()

    assert evidence_of(_StubToolMessage(_results_evidence())) is not None
    assert evidence_of({"type": "tool", "artifact": artifact}) is not None
    assert evidence_of(None) is None
    assert evidence_of({"type": "tool"}) is None
    assert evidence_of(_StubToolMessage(artifact={"other": 1})) is None


def test_search_and_detail_evidence_never_read_each_other() -> None:
    """A detail read must never be summarised as a search, or vice versa."""

    detail_artifact = ProductDetailEvidence(products=[]).as_artifact()
    search_artifact = SearchEvidence(outcome="results").as_artifact()

    assert set(detail_artifact) == {DETAIL_EVIDENCE_KEY}
    assert set(search_artifact) == {EVIDENCE_KEY}
    assert evidence_of(_StubToolMessage(artifact=detail_artifact)) is None
    assert detail_evidence_of(_StubToolMessage(artifact=search_artifact)) is None


def test_artifact_is_json_round_trippable() -> None:
    """The payload rides a checkpointed ToolMessage, so it must survive serde."""

    import json

    artifact = _results_evidence().as_artifact()

    assert json.loads(json.dumps(artifact)) == artifact


def test_every_search_summary_tells_the_composer_to_drop_the_jargon() -> None:
    """The evidence is internal bookkeeping and a model will read it aloud.

    A live reply came back as "I checked the broader apparel category with the
    adult all-genders filter, and that search returned no matches under that
    scope" -- which is the evidence block paraphrased. The shopper is standing
    in a shop, not in front of a query planner.
    """

    evidence = SearchEvidence(outcome="zero_results")
    message = SimpleNamespace(artifact=evidence.as_artifact())

    summary = grounding_evidence_mod._customer_safe_tool_evidence("", message)

    assert "SPEAK AS A SHOP ASSISTANT" in summary
    for banned in ("search", "filter", "scope", "taxonomy", "query", "results"):
        assert banned in summary.split("SPEAK AS A SHOP ASSISTANT")[1]
    assert "adult_all_genders" in summary
    assert "pieces anyone can wear" in summary


def test_a_zero_result_tells_the_model_to_relax_its_own_search() -> None:
    """Absence alone produced a menu of things the shopper could not see.

    "No green dress in a size 2 -- would you like size 4, or another colour?"
    on a turn where the catalog held plenty of size 2 dresses in other
    colours. That was answered by running the search again on the server and
    handing over what it found -- a retry that kept the size and dropped the
    product type with everything else, so a tote bag in a size 8 came back as
    boots. The model issues this retry correctly itself; what it needed was
    to be told to, and told which filter may give.
    """

    from chain_server.src.grounding_evidence import _customer_safe_search_evidence

    summary = _customer_safe_search_evidence(
        {
            "outcome": "zero_results",
            "taxonomy": {"subcategory": ["dresses"]},
            "confirmed_filters": {"primary_color": ["green"], "sizes": ["2"]},
        }
    )

    assert "Search again yourself" in summary
    assert "Keep the product type and keep the size" in summary
    assert "do not answer with a question alone" in summary.casefold()
    # The two substitutions this exists to prevent, named rather than implied.
    assert "not answered with a skirt" in summary
    assert "not a preference" in summary


def test_a_zero_result_does_not_hand_over_products_of_its_own() -> None:
    """No second search runs here, so nothing can arrive that the reply
    disowns. The heading its results used to land under is gone with it."""

    from chain_server.src.grounding_evidence import _customer_safe_search_evidence

    summary = _customer_safe_search_evidence(
        {
            "outcome": "zero_results",
            "taxonomy": {"subcategory": ["tote_bags"]},
            "confirmed_filters": {"sizes": ["8"]},
        }
    )

    assert "CUSTOMER_SAFE_RELAXED_SEARCH_EVIDENCE" not in summary
    # An uncarried type is not a filter to relax, and saying so here is what
    # keeps "dark blue jeans" from becoming a navy skirt on the retry.
    assert "search no further" in summary
