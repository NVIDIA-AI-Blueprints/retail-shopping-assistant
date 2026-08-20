# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A choice the record made is a fact about the conversation, not the message.

Live, `conversation-e01a4fee`: the shopper said "add the first pairing to cart",
was asked for a shoe size, gave it -- and was then refused three turns running,
each time being asked to confirm products they had already chosen. The fourth
attempt succeeded only because they typed both catalog names in full.

The tool call was byte-identical in all four turns. Only the words in the
shopper's message differed, and the gate could see nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace

from chain_server.src.conversation_products import ProductEvidence
from chain_server.src.turn_support import (
    _cart_product_provenance_issue,
    _identified_in_the_current_showing,
    _system_identification_events,
)


def _product(ref: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(product_id=ref, display_name=name)


DRESS = _product("generated:dress", "Ocean Blue Chiffon Maxi Dress")
SHOES = _product("generated:shoes", "Elegant Embroidered Espadrilles")
OTHER = _product("generated:other", "Coral Silk Maxi Dress")


def _showing(turn_seq: int, *, identified: list[str] | None = None) -> dict:
    entry: dict = {
        "candidate_set_id": f"set-{turn_seq}",
        "turn_seq": turn_seq,
        "products": [
            {"ref": DRESS.product_id, "name": DRESS.display_name},
            {"ref": SHOES.product_id, "name": SHOES.display_name},
            {"ref": OTHER.product_id, "name": OTHER.display_name},
        ],
    }
    if identified is not None:
        entry["system_identified"] = identified
    return entry


def test_a_product_chosen_in_an_earlier_turn_is_still_established() -> None:
    """Turn 3 chose the pairing. Turn 4 only answered a question about size."""

    state = SimpleNamespace(
        historical_product_sets=[
            _showing(2, identified=[DRESS.product_id, SHOES.product_id])
        ]
    )

    issue = _cart_product_provenance_issue(
        DRESS,
        "I need size 6 for the shoes",
        ProductEvidence(),
        state.historical_product_sets[0]["products"],
        _identified_in_the_current_showing(state),
    )

    assert issue == ""


def test_a_product_nobody_chose_is_still_refused() -> None:
    """The gate's whole purpose, and it must survive the change."""

    state = SimpleNamespace(
        historical_product_sets=[_showing(2, identified=[DRESS.product_id])]
    )

    issue = _cart_product_provenance_issue(
        OTHER,
        "I need size 6 for the shoes",
        ProductEvidence(),
        state.historical_product_sets[0]["products"],
        _identified_in_the_current_showing(state),
    )

    assert "PRODUCT NOT ESTABLISHED" in issue
    assert OTHER.display_name in issue


def test_a_newer_showing_retires_the_earlier_choice() -> None:
    """A new set of products is a new choice to make.

    Without this the gate would be permanently satisfied by something the
    shopper picked once, twenty turns ago, and "add it" would mean whatever the
    model decided it meant.
    """

    state = SimpleNamespace(
        historical_product_sets=[
            _showing(2, identified=[DRESS.product_id, SHOES.product_id]),
            _showing(5),
        ]
    )

    assert _identified_in_the_current_showing(state) == set()

    issue = _cart_product_provenance_issue(
        DRESS,
        "add it",
        ProductEvidence(),
        state.historical_product_sets[1]["products"],
        _identified_in_the_current_showing(state),
    )

    assert "PRODUCT NOT ESTABLISHED" in issue


def test_the_newest_showing_carries_its_own_choices() -> None:
    """The lapse is not amnesia: a choice made against the new set still holds."""

    state = SimpleNamespace(
        historical_product_sets=[
            _showing(2, identified=[OTHER.product_id]),
            _showing(5, identified=[DRESS.product_id]),
        ]
    )

    assert _identified_in_the_current_showing(state) == {DRESS.product_id}


def test_the_turn_records_what_the_record_picked() -> None:
    """Nothing is durable unless the turn writes it down."""

    events = _system_identification_events(
        SimpleNamespace(
            system_identified_products=[DRESS.product_id, SHOES.product_id]
        ),
        SimpleNamespace(request_id="request-1"),
    )

    assert len(events) == 1
    assert events[0].event_type == "historical_reference_resolved"
    assert events[0].payload["product_refs"] == [
        DRESS.product_id,
        SHOES.product_id,
    ]


def test_a_turn_that_picked_nothing_records_nothing() -> None:
    """An empty event would retire the previous choice for no reason."""

    assert (
        _system_identification_events(
            SimpleNamespace(system_identified_products=[]),
            SimpleNamespace(request_id="request-1"),
        )
        == []
    )
