# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import requests
from pydantic import ValidationError

from chain_server.src.conversation_products import (
    ConversationProductMatch,
    ConversationProductsClient,
    ConversationProductsError,
    ProductEvidence,
    ProductReferenceDescriptor,
    ProductReferenceResolution,
    ResolveConversationProductsRequest,
    ResolveConversationProductsResult,
    format_historical_product_index,
    format_product_resolution,
)
from shared.commerce_contracts import Money, ProductSummary


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload


class _Session:
    def __init__(
        self,
        response: _Response | None = None,
        error: requests.RequestException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: Any, timeout: float) -> _Response:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _product(product_id: str, display_name: str) -> ProductSummary:
    return ProductSummary(
        product_id=product_id,
        display_name=display_name,
        category="crossbody_bags",
        price=Money(amount=79),
        image_url=f"/images/{product_id}.png",
    )


def _match(product_id: str, display_name: str, position: int) -> dict[str, Any]:
    return {
        "product": _product(product_id, display_name).model_dump(mode="json"),
        "candidate_set_id": "set-2",
        "turn_sequence": 2,
        "position": position,
        "catalog_revision": "catalog-v1",
    }


def test_reference_descriptors_require_selectors_and_scope_ordinals() -> None:
    with pytest.raises(ValidationError, match="selector"):
        ProductReferenceDescriptor(reference_id="bag")
    with pytest.raises(ValidationError, match="ordinal requires"):
        ProductReferenceDescriptor(
            reference_id="bag",
            display_name="Cobalt Crossbody",
            ordinal=2,
        )
    with pytest.raises(ValidationError):
        ResolveConversationProductsRequest(references=[])

    descriptor = ProductReferenceDescriptor(
        reference_id="bag",
        candidate_set_id="set-2",
        ordinal=2,
    )

    assert descriptor.ordinal == 2


def test_client_posts_one_batch_and_parses_typed_products() -> None:
    payload = {
        "results": [
            {
                "reference_id": "bag",
                "status": "resolved",
                "matches": [_match("bag-2", "Cobalt Crossbody", 2)],
                "match_count": 1,
            }
        ]
    }
    session = _Session(_Response(payload))
    client = ConversationProductsClient(
        "http://memory:8011/",
        timeout_seconds=4,
        session=session,
    )
    descriptors = [
        ProductReferenceDescriptor(
            reference_id="bag",
            display_name="Cobalt Crossbody",
        ),
        ProductReferenceDescriptor(
            reference_id="second",
            turn_sequence=2,
            ordinal=2,
        ),
    ]

    result = client.resolve("conversation/a", descriptors)

    assert len(session.calls) == 1
    assert session.calls[0] == {
        "url": ("http://memory:8011/conversations/conversation%2Fa/products/resolve"),
        "json": {
            "references": [
                {
                    "reference_id": "bag",
                    "display_name": "Cobalt Crossbody",
                },
                {
                    "reference_id": "second",
                    "turn_sequence": 2,
                    "ordinal": 2,
                },
            ]
        },
        "timeout": 4,
    }
    match = result.results[0].matches[0]
    assert match.product.product_id == "bag-2"
    assert match.candidate_set_id == "set-2"
    assert match.turn_sequence == 2


@pytest.mark.parametrize(
    "response,error_code,retryable",
    [
        (_Response({}, status_code=503), "conversation_products_unavailable", True),
        (
            _Response({}, status_code=422),
            "conversation_products_request_rejected",
            False,
        ),
        (_Response({"results": []}), "conversation_products_response_invalid", False),
    ],
)
def test_client_errors_are_stable(
    response: _Response,
    error_code: str,
    retryable: bool,
) -> None:
    client = ConversationProductsClient(
        "http://memory",
        session=_Session(response),
    )

    with pytest.raises(ConversationProductsError) as caught:
        client.resolve(
            "conversation-a",
            [ProductReferenceDescriptor(reference_id="bag", category="bags")],
        )

    assert caught.value.code == error_code
    assert caught.value.retryable is retryable


def test_transport_error_is_retryable() -> None:
    client = ConversationProductsClient(
        "http://memory",
        session=_Session(error=requests.Timeout("timeout")),
    )

    with pytest.raises(ConversationProductsError) as caught:
        client.resolve(
            "conversation-a",
            [ProductReferenceDescriptor(reference_id="bag", category="bags")],
        )

    assert caught.value.code == "conversation_products_request_failed"
    assert caught.value.retryable is True


def test_the_index_carries_the_colour_a_reference_is_made_of() -> None:
    """"The black one" is unanswerable from a list of names.

    Four dresses were shown together and all four were black; two had "Black"
    in the name and two did not. The model guessed by name, reached back
    fourteen turns for a navy dress, and put it in the cart. Nothing refused it
    -- the descriptor it sent was internally consistent and named a real
    product.
    """

    from chain_server.src.conversation_products import (
        format_historical_product_index,
    )

    rendered = format_historical_product_index(
        [
            {
                "candidate_set_id": "set-9",
                "turn_seq": 9,
                "products": [
                    {
                        "ref": "a",
                        "name": "Belle Noir Satin Gown",
                        "category": "dresses",
                        "color": "black",
                        "position": 1,
                    },
                    {
                        "ref": "b",
                        "name": "Ivory Sheath Dress",
                        "category": "dresses",
                        "color": "white",
                        "position": 2,
                    },
                ],
            }
        ]
    )

    # The one that is black does not say so in its name.
    assert "Belle Noir Satin Gown [dresses|black]" in rendered
    assert "Ivory Sheath Dress [dresses|white]" in rendered


def test_the_most_recent_showing_is_listed_first() -> None:
    """"The black one" means the most recent black thing, not the oldest.

    The list opened with turn 1 and buried the latest showing at the bottom of
    a long prompt. Asked for "the black one in a 2" one turn after four black
    dresses were shown, the assistant reached back fourteen turns for a navy
    dress and put it in the cart.
    """

    from chain_server.src.conversation_products import (
        format_historical_product_index,
    )

    rendered = format_historical_product_index(
        [
            {
                "candidate_set_id": "old",
                "turn_seq": 1,
                "products": [
                    {"ref": "a", "name": "Navy Dress", "category": "dresses",
                     "position": 1}
                ],
            },
            {
                "candidate_set_id": "new",
                "turn_seq": 9,
                "products": [
                    {"ref": "b", "name": "Black Dress", "category": "dresses",
                     "position": 1}
                ],
            },
        ]
    )

    lines = [line for line in rendered.splitlines() if line.startswith("- set=")]
    assert lines[0].startswith("- set=new turn=9")
    assert lines[-1].startswith("- set=old turn=1")
    assert "most recently shown first" in rendered


def test_historical_index_formatter_is_compact_and_ignores_bad_rows() -> None:
    rendered = format_historical_product_index(
        [
            {"malformed": True},
            {
                "candidate_set_id": "set-2",
                "turn_seq": 2,
                "catalog_revision": "catalog-v1",
                "products": [
                    {
                        "ref": "bag-1",
                        "name": "Structured\nTote",
                        "category": "tote_bags",
                        "position": 1,
                    },
                    {
                        "ref": "bag-2",
                        "name": "Cobalt Crossbody",
                        "category": "crossbody_bags",
                        "position": 2,
                    },
                ],
            },
        ]
    )

    assert rendered.startswith(
        "HISTORICAL PRODUCT INDEX (read-only, most recently shown first):"
    )
    assert "set=set-2 turn=2" in rendered
    assert "1:Structured Tote [tote_bags] <bag-1>" in rendered
    assert "2:Cobalt Crossbody [crossbody_bags] <bag-2>" in rendered
    assert format_historical_product_index([]) == ""


def test_historical_index_bound_keeps_newest_sets() -> None:
    reference_sets = [
        {
            "candidate_set_id": f"set-{index}",
            "turn_seq": index,
            "products": [
                {
                    "ref": f"product-{index}",
                    "name": f"Product {index} " + ("x" * 80),
                    "position": 1,
                }
            ],
        }
        for index in range(1, 5)
    ]

    rendered = format_historical_product_index(reference_sets, max_chars=256)

    assert "set=set-4" in rendered
    assert "set=set-1" not in rendered
    assert "earlier historical products omitted" in rendered
    assert len(rendered) <= 256


def test_evidence_adds_only_unique_resolved_products() -> None:
    resolved = ProductReferenceResolution(
        reference_id="unique",
        status="resolved",
        matches=[ConversationProductMatch.model_validate(_match("bag-1", "Tote", 1))],
        match_count=1,
    )
    ambiguous = ProductReferenceResolution(
        reference_id="ambiguous",
        status="ambiguous",
        matches=[
            ConversationProductMatch.model_validate(_match("bag-2", "Blue Bag", 1)),
            ConversationProductMatch.model_validate(_match("bag-3", "Tan Bag", 2)),
        ],
        match_count=2,
    )
    missing = ProductReferenceResolution(
        reference_id="missing",
        status="not_found",
        matches=[],
        match_count=0,
    )
    evidence = ProductEvidence()

    evidence.add_resolutions([resolved, ambiguous, missing])

    assert evidence.get("bag-1") == resolved.matches[0].product
    assert evidence.get("bag-2") is None
    assert evidence.get("bag-3") is None


def test_result_formatter_resolves_one_and_requires_clarification_otherwise() -> None:
    result = ResolveConversationProductsResult(
        results=[
            ProductReferenceResolution(
                reference_id="unique",
                status="resolved",
                matches=[
                    ConversationProductMatch.model_validate(
                        _match("bag-1", "Structured Tote", 1)
                    )
                ],
                match_count=1,
            ),
            ProductReferenceResolution(
                reference_id="ambiguous",
                status="ambiguous",
                matches=[
                    ConversationProductMatch.model_validate(
                        _match("bag-2", "Cobalt Crossbody", 1)
                    ),
                    ConversationProductMatch.model_validate(
                        _match("bag-3", "Tan Shoulder Bag", 2)
                    ),
                ],
                match_count=2,
            ),
            ProductReferenceResolution(
                reference_id="missing",
                status="not_found",
                matches=[],
                match_count=0,
            ),
        ]
    )

    rendered = format_product_resolution(result)

    assert "REFERENCE unique: RESOLVED\nPRODUCT_REF: bag-1" in rendered
    assert "REFERENCE ambiguous: CLARIFICATION REQUIRED" in rendered
    assert "Cobalt Crossbody, Tan Shoulder Bag" in rendered
    assert "REFERENCE missing: NOT FOUND" in rendered
    # An ambiguous reference is a question; nothing found is a search. Both
    # used to say "do not guess" and stop, which left a shopper who named a
    # product that was never shown with no path forward at all.
    assert "Do not guess" in rendered.split("REFERENCE missing")[0]
    assert "search the catalog" in rendered.split("REFERENCE missing")[1]
    # Neither may authorise adding something the shopper has not seen.
    assert "Never add a product the shopper has not been shown" in rendered
