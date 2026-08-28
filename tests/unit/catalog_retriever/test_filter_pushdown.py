# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The database decides the filters it can decide exactly, and no others."""

from __future__ import annotations

import json

import pytest

from catalog_retriever.src.retriever import Retriever
from shared.commerce_contracts import CatalogFilterCapability


class _Stub(Retriever):
    """A retriever with only the state the expression builder reads."""

    def __init__(self, capabilities):
        self.filter_capabilities = capabilities


def _capabilities():
    return {
        "primary_color": CatalogFilterCapability(
            type="enum", operators=["in"], source_fields=["primary_color"],
            values=["black", "beige"],
        ),
        "sizes": CatalogFilterCapability(
            type="enum_list", operators=["in"], source_fields=["sizes"],
            values=["2", "4"],
        ),
        "price": CatalogFilterCapability(
            type="number", operators=["gte", "lte"], source_fields=["price"],
        ),
        "audience": CatalogFilterCapability(
            type="enum", operators=["in"],
            source_fields=["target_audience", "audience"], values=["womens"],
        ),
    }


def test_an_enum_becomes_a_membership_test() -> None:
    expr, covered = _Stub(_capabilities())._filter_expression(
        {"primary_color": "black"}
    )

    assert expr == '(primary_color in ["black"])'
    assert covered == {"primary_color"}


def test_a_list_valued_enum_asks_whether_the_list_contains_it() -> None:
    """`sizes` is a list per product, so equality would never match."""

    expr, _ = _Stub(_capabilities())._filter_expression({"sizes": ["2", "4"]})

    assert expr == '(json_contains_any(sizes, ["2", "4"]))'


def test_a_number_is_never_pushed_down() -> None:
    """Prices are stored as text.

    `price >= 100` against a text field does not fail -- it returns nothing.
    Silently excluding every product is worse than filtering in Python, so
    numbers stay where they can be compared as numbers.
    """

    expr, covered = _Stub(_capabilities())._filter_expression(
        {"price": {"min": 100}}
    )

    assert expr == ""
    assert covered == set()


def test_several_source_fields_satisfy_the_filter_if_any_of_them_match() -> None:
    """Mirrors the Python matcher, which unions the values across source fields."""

    expr, _ = _Stub(_capabilities())._filter_expression({"audience": "womens"})

    assert expr == (
        '(target_audience in ["womens"] or audience in ["womens"])'
    )


def test_filters_combine_with_and() -> None:
    expr, covered = _Stub(_capabilities())._filter_expression(
        {"primary_color": "black", "sizes": "2"}
    )

    # Ordered by filter name, so the same request always builds the same
    # expression -- which makes it comparable between runs.
    assert expr == (
        '(primary_color in ["black"]) and (json_contains_any(sizes, ["2"]))'
    )
    assert covered == {"primary_color", "sizes"}


def test_a_mixed_request_pushes_down_only_what_it_can() -> None:
    """The number stays behind, and the caller is told so."""

    expr, covered = _Stub(_capabilities())._filter_expression(
        {"primary_color": "black", "price": {"max": 150}}
    )

    assert expr == '(primary_color in ["black"])'
    assert covered == {"primary_color"}


def test_a_value_that_cannot_be_written_as_a_literal_is_left_to_python() -> None:
    """Rather than escaping a quote into an expression nobody can read."""

    expr, covered = _Stub(_capabilities())._filter_expression(
        {"primary_color": 'bl"ack'}
    )

    assert expr == ""
    assert covered == set()


def test_an_unknown_filter_is_ignored_by_the_builder() -> None:
    """Validation refuses those earlier; the builder must not invent a clause."""

    expr, covered = _Stub(_capabilities())._filter_expression({"nonesuch": "x"})

    assert expr == ""
    assert covered == set()
