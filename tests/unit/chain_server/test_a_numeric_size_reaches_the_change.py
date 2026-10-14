# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J06 t9, "actually make those a 7".

The `size` field on `update_cart_items_tool` is how a size change is asked for,
and the model sent `size: 7` rather than `size: "7"`. Pydantic refused the call
on the type before any of it ran:

    size: 8   -> "Input should be a valid string"     (no mention of carts)
    size: "8" -> the line moves to the size they asked for

Three attempts, three type errors, and then it gave up, sent the quantity
alone, and told the shopper it had updated a dress it was never asked about.
Sizes are "2" and "onesize" in this catalog, so a bare number is the obvious
slip, and coercing it costs nothing.
"""

from __future__ import annotations

import pytest
from chain_server.src.tools.cart import _UpdateCartItemsInput
from pydantic import ValidationError


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

    with pytest.raises(ValidationError):
        _size(True)
