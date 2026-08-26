# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J06 t9, "actually make those a 7".

`update_cart_items_tool` declares a `size` field for one reason: to turn a size
change into the add-then-remove sequence instead of a silent no-op. The model
sent `size: 7` rather than `size: "7"`, and pydantic refused the call on the
type before that guidance could run:

    size: 8   -> "Input should be a valid string"     (no mention of carts)
    size: "8" -> CART_UPDATE_REFUSED: add the new size first, confirm it,
                 then remove the old line...

Three attempts, three type errors, and then it gave up, sent the quantity
alone, and told the shopper it had updated a dress it was never asked about.
The one message that would have told it what to do was never delivered.
"""

from __future__ import annotations

import pytest

from chain_server.src.deepagents_runtime import _UpdateCartItemsInput


def _size(value):
    return _UpdateCartItemsInput(
        cart_line_id="line-1", quantity=1, size=value
    ).size


def test_a_size_written_as_a_number_is_taken() -> None:
    assert _size(7) == "7"


def test_a_float_that_is_a_whole_number_loses_its_point_zero() -> None:
    """`7.0` must not become the size "7.0", which matches nothing."""

    assert _size(7.0) == "7"


def test_a_size_written_as_a_string_is_untouched() -> None:
    assert _size("7") == "7"
    assert _size("onesize") == "onesize"


def test_no_size_stays_absent() -> None:
    assert _size(None) is None


def test_a_boolean_is_not_a_size() -> None:
    """True would otherwise coerce to "1", which is a plausible-looking size."""

    with pytest.raises(Exception):
        _size(True)
