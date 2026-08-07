# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A size is part of what the shopper chose, so it is part of the cart line.

Turn 14 of the fifteen-turn script asked "what size should I add?" and then
added nothing. The question was invented: no column, no field, nowhere for the
answer to go. It sounded like retail, which is what made it convincing.

The fix was not to stop asking -- a dress entering a cart with no size reads as
a toy -- but to make the question real. These hold the parts that quietly go
wrong once it is.
"""

from __future__ import annotations

from chain_server.src.turn_support import _normalize_cart_add_tool_items


def test_two_sizes_of_one_product_are_two_lines() -> None:
    """Merging them would halve the order without saying so.

    A 6 and an 8 of one dress are two things a shopper owns, not one line of
    quantity two, so the normalizer keys on the size as well as the reference.
    """

    normalized = _normalize_cart_add_tool_items(
        [
            {"product_ref": "p1", "quantity": 1, "size": "6"},
            {"product_ref": "p1", "quantity": 1, "size": "8"},
        ]
    )

    assert set(normalized) == {("p1", "6"), ("p1", "8")}
    assert [entry["quantity"] for entry in normalized.values()] == [1, 1]


def test_the_same_size_twice_still_merges() -> None:
    """Two of the same size is a quantity, and must not become two lines."""

    normalized = _normalize_cart_add_tool_items(
        [
            {"product_ref": "p1", "quantity": 1, "size": "8"},
            {"product_ref": "p1", "quantity": 2, "size": "8"},
        ]
    )

    assert set(normalized) == {("p1", "8")}
    assert normalized[("p1", "8")]["quantity"] == 3


def test_one_size_goods_carry_no_size() -> None:
    """A handbag has no size to record, and blank is not a size."""

    normalized = _normalize_cart_add_tool_items(
        [
            {"product_ref": "bag", "quantity": 1},
            {"product_ref": "bag2", "quantity": 1, "size": "   "},
        ]
    )

    assert set(normalized) == {("bag", None), ("bag2", None)}
    assert all(entry["size"] is None for entry in normalized.values())


def test_the_size_reaches_the_cart_payload() -> None:
    """The adapter must send it, or the answer is discarded at the last step --
    which is exactly what happened before this existed."""

    from shared.commerce_contracts import AddCartItemInput

    request = AddCartItemInput(
        user_id="1",
        product_id="p1",
        quantity=1,
        idempotency_key="k",
        size="8",
    )

    assert request.size == "8"
    assert AddCartItemInput(
        user_id="1", product_id="p1", quantity=1, idempotency_key="k"
    ).size is None


def test_the_size_survives_reading_the_cart_back() -> None:
    """Recording it is not enough if the shopper never sees it.

    Live, a dress went in as a size 8 and the cart read back as "The Office
    A-line Dress — $179.99 (qty 1)" with no size anywhere: the column held it,
    and every layer above dropped it.
    """

    from chain_server.src.commerce_tools import _cart_line_from_memory_item

    line = _cart_line_from_memory_item(
        {
            "cart_line_id": "abc",
            "product_id": "p1",
            "item": "The Office A-line Dress",
            "amount": 1,
            "price": 179.99,
            "size": "8",
        }
    )

    assert line is not None
    assert line.size == "8"

    one_size = _cart_line_from_memory_item(
        {"cart_line_id": "b", "product_id": "p2", "item": "A bag", "amount": 1}
    )
    assert one_size is not None and one_size.size is None
