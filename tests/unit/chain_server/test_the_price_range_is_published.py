# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J10 t7, "what's the most expensive thing you have".

Two runs in five refused outright. The other three answered confidently and
wrongly -- "the most expensive item in the catalog is the Polished Pebble Grain
Purse at $189.99" -- having searched one category and reported its dearest item
as the shop's. This catalog runs to $269.99.

The range was published in capabilities the whole time. The tool schema
described every filter as "Advertised hard filter 'price'." and dropped it.
"""

from __future__ import annotations

from chain_server.src.turn_support import _advertised_range, _search_catalog_tool_input_model
from shared.commerce_contracts import CatalogCapabilities, CatalogFilterCapability


def _capability(**kw):
    return CatalogFilterCapability(
        type="number", operators=["gte", "lte"], source_fields=["price"], **kw
    )


def test_the_range_reads_as_the_catalog_states_it() -> None:
    assert _advertised_range(_capability(min_value=39.9, max_value=269.99)) == (
        "39.9 to 269.99"
    )


def test_a_whole_number_keeps_no_trailing_zeros() -> None:
    assert _advertised_range(_capability(min_value=10.0, max_value=200.0)) == (
        "10 to 200"
    )


def test_one_bound_is_still_worth_saying() -> None:
    assert _advertised_range(_capability(min_value=39.9)) == "from 39.9"
    assert _advertised_range(_capability(max_value=269.99)) == "up to 269.99"


def test_no_bounds_says_nothing() -> None:
    assert _advertised_range(_capability()) == ""


def test_the_range_reaches_the_field_the_model_reads() -> None:
    """Through the built schema, not the helper. The helper existing is worth
    nothing if the description still says only "Advertised hard filter"."""

    capabilities = CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=["text"],
        filters={"price": _capability(min_value=39.9, max_value=269.99)},
    )

    model = _search_catalog_tool_input_model(capabilities)
    constraints = model.model_fields["required_constraints"].annotation
    description = constraints.model_fields["price"].description

    assert "39.9 to 269.99" in description
