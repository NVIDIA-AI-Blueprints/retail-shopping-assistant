# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract for the typed search-evidence payload (slice D2).

Two obligations are proved here, and they are different claims:

1. The model-visible search text is byte-identical to what the previous
   text-only implementation produced. Rendering moved behind a record, so this
   is proved against a reference copy of the old renderer rather than against a
   golden string that could be edited to match.
2. The composer receives the *same evidence* from the typed payload as it
   previously reconstructed by parsing that text. A byte-identical tool result
   does not by itself establish this -- it is a separate path with its own
   summary -- so it gets its own equivalence check.
"""

from __future__ import annotations

from typing import Any

import pytest

from chain_server.src import deepagents_runtime as runtime_mod
from shared.commerce_contracts import Money, ProductSummary
from chain_server.src.search_evidence import EVIDENCE_KEY, SearchEvidence, evidence_of


def _reference_format_product(product: Any) -> str:
    """The renderer exactly as it stood before the record was introduced."""

    lines = [
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if getattr(product, "category", None):
        lines.append(f"CATEGORY: {product.category}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    lines.append(
        "DETAILS: Call get_product_details_tool with this PRODUCT_REF before "
        "stating materials, dimensions, pockets, closures, care, comfort, or "
        "outdoor-practicality claims."
    )
    return "\n".join(lines)


_PRODUCT_CASES = [
    ProductSummary(
        product_id="prod_1",
        display_name="Ravenna Crossbody Bag",
        description="structured",
        category="Bags",
        price=Money(amount=49.99),
        image_url="https://example.invalid/a.jpg",
    ),
    # No category, no image, no price: every optional line absent.
    ProductSummary(product_id="prod_2", display_name="Plain Tee", description=""),
    # Non-ASCII display name and a non-default currency.
    ProductSummary(
        product_id="prod_3",
        display_name="Café Blazer — élan",
        description="",
        category="Outerwear",
        price=Money(amount=1234.5, currency="EUR"),
    ),
    # A price that must keep two decimal places rather than collapsing.
    ProductSummary(
        product_id="prod_4",
        display_name="Rounding Check",
        description="",
        price=Money(amount=7.0),
        image_url="https://example.invalid/b.png",
    ),
]


@pytest.mark.parametrize("product", _PRODUCT_CASES, ids=lambda p: p.product_id)
def test_product_text_is_byte_identical_to_the_previous_renderer(
    product: ProductSummary,
) -> None:
    """Routing rendering through a record must not change one byte the model sees."""

    assert runtime_mod._format_product(product) == _reference_format_product(product)


def test_product_record_carries_the_fields_the_text_shows() -> None:
    record = runtime_mod._search_product_record(_PRODUCT_CASES[0])

    assert record == {
        "product_ref": "prod_1",
        "name": "Ravenna Crossbody Bag",
        "category": "Bags",
        "price": "$49.99 USD",
        "image_url": "https://example.invalid/a.jpg",
    }


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
            runtime_mod._search_product_record(product) for product in _PRODUCT_CASES
        ],
    )


def _render_results_text(evidence: SearchEvidence) -> str:
    """Reproduce the tool's model-visible text from a payload."""

    lines = [
        runtime_mod._SEARCH_RESULT_GROUNDING_NOTE,
        runtime_mod._format_search_direction_evidence(evidence.semantic_query),
        runtime_mod._format_search_guidance_evidence(evidence.shopper_guidance),
        runtime_mod._format_search_taxonomy_evidence(evidence.taxonomy),
    ]
    if evidence.advertised_category:
        lines.append(
            runtime_mod._format_search_scope_relation_evidence(
                requested_product_type=evidence.requested_product_type or "",
                advertised_category=evidence.advertised_category,
            )
        )
    if evidence.confirmed_filters:
        lines.append(
            runtime_mod._format_search_filter_evidence(evidence.confirmed_filters)
        )
    for record in evidence.products:
        lines.append(runtime_mod._format_product_record(record))
    return "\n\n".join(lines)


def test_typed_results_summary_matches_what_parsing_the_text_produced() -> None:
    """The composer must get the same evidence from the payload as from the prose.

    This is the claim the byte-identity test cannot make: the payload and the
    text are two routes into the composer, and only comparing their outputs
    shows the runtime learned nothing extra -- and lost nothing -- by reading
    data instead of re-reading its own prose.
    """

    evidence = _results_evidence()
    from_text = runtime_mod._customer_safe_tool_evidence(_render_results_text(evidence))
    from_payload = runtime_mod._customer_safe_tool_evidence(
        "", _StubToolMessage(evidence)
    )

    assert from_payload == from_text


def test_typed_zero_results_summary_matches_what_parsing_the_text_produced() -> None:
    evidence = SearchEvidence(
        outcome="zero_results",
        taxonomy={"category": ["Bags"]},
        confirmed_filters={"primary_color": ["black"]},
        requested_product_type="crossbody bag",
        advertised_category="Bags",
    )
    text = "\n\n".join(
        [
            runtime_mod._SEARCH_NO_MATCH_GROUNDING_NOTE,
            runtime_mod._format_search_taxonomy_evidence(evidence.taxonomy),
            runtime_mod._format_search_scope_relation_evidence(
                requested_product_type=evidence.requested_product_type or "",
                advertised_category=evidence.advertised_category or "",
            ),
            runtime_mod._format_search_filter_evidence(evidence.confirmed_filters),
        ]
    )

    from_text = runtime_mod._customer_safe_tool_evidence(text)
    from_payload = runtime_mod._customer_safe_tool_evidence(
        "", _StubToolMessage(evidence)
    )

    assert from_payload == from_text


def test_scope_relation_is_absent_when_no_parent_was_substituted() -> None:
    """A plain search must not gain a parent-scope caveat it never earned."""

    evidence = _results_evidence()
    evidence.advertised_category = None

    summary = runtime_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert "REQUESTED_SCOPE_RELATION" not in summary


def test_empty_typed_results_do_not_append_a_blank_line() -> None:
    """With no products there is no prose to fall back to, so none is faked."""

    evidence = _results_evidence()
    evidence.products = []

    summary = runtime_mod._customer_safe_tool_evidence("", _StubToolMessage(evidence))

    assert summary == summary.rstrip("\n")
    assert not summary.endswith("\n")


def test_evidence_of_ignores_messages_without_the_payload() -> None:
    assert evidence_of(None) is None
    assert evidence_of(_StubToolMessage(None)) is None
    assert evidence_of(_StubToolMessage(None, artifact={"other": 1})) is None

    evidence = _results_evidence()
    payload = evidence_of(_StubToolMessage(evidence))
    assert payload is not None
    assert payload["outcome"] == "results"
    assert payload["products"][0]["name"] == "Ravenna Crossbody Bag"


def test_evidence_is_readable_from_a_dict_shaped_message() -> None:
    """Messages reach this code as objects or as dicts; both must be readable.

    Falling back to the text path for a dict-shaped message would work, but
    only by re-parsing prose -- the exact round trip this slice removes.
    """

    artifact = _results_evidence().as_artifact()

    payload = evidence_of({"type": "tool", "artifact": artifact})

    assert payload is not None
    assert payload["outcome"] == "results"
    assert evidence_of({"type": "tool"}) is None


def test_artifact_is_json_round_trippable() -> None:
    """The payload rides a checkpointed ToolMessage, so it must survive serde."""

    import json

    artifact = _results_evidence().as_artifact()
    restored = json.loads(json.dumps(artifact))

    assert restored == artifact
    assert set(restored) == {EVIDENCE_KEY}


class _StubToolMessage:
    """Minimal stand-in for the tool message the composer reads."""

    def __init__(
        self,
        evidence: SearchEvidence | None,
        *,
        artifact: dict[str, Any] | None = None,
    ) -> None:
        self.artifact = (
            artifact if evidence is None else evidence.as_artifact()
        )
