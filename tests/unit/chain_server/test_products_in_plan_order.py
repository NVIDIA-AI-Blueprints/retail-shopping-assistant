# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shopper sees products in the order the roles were asked for.

Retrieval fans out across a thread pool, and the publishing used to happen
inside each worker. So the order products reached the screen was the order the
scopes finished, which is not stable: one request for "a cream knit sweater and
brown ankle boots" arrived as four sweaters then two boots on three runs, and as
sweater, boot, boot, sweater, sweater, sweater on a fourth.

The evidence the model reads was never affected -- it is rendered from the
attempt list as `SCOPE 1`, `SCOPE 2`. Measured against a live stack, the reply
names products in exactly that order, whether it names two of six or all six.
So the prose was ordered by the plan while the pictures beside it were ordered
by the race, and "the first one" meant different garments to the shopper and to
the record.
"""

from __future__ import annotations

from types import SimpleNamespace

from chain_server.src.agenttypes import State
from chain_server.src.catalog_search import (
    SearchContext,
    _Attempt,
    _published_in_plan_order,
)
from chain_server.src.turn_scope import TurnScope
from shared.commerce_contracts import ProductSummary


def _context() -> SearchContext:
    return SearchContext(
        config=SimpleNamespace(
            top_k_retrieve=4,
            search_products_per_call=36,
            max_catalog_searches_per_turn=3,
            retriever_port="http://catalog-retriever:8010",
            catalog_search_timeout_seconds=5,
        ),
        state=State(user_id=1, query="a cream knit sweater and brown ankle boots"),
        scope=TurnScope(),
        capabilities=None,
        search_input_model=None,
        constraint_input_model=None,
    )


def _attempt(role: str) -> _Attempt:
    return _Attempt(
        semantic_query=role,
        requested_product_type=role,
        taxonomy={},
        required_constraints={},
        shopper_guidance="",
    )


def _finished(role: str, *names: str) -> _Attempt:
    """One scope that retrieved products, as the pool leaves it."""

    attempt = _attempt(role)
    attempt.result = SimpleNamespace(
        ok=True,
        products=[
            ProductSummary(
                product_id=name.lower().replace(" ", "-"),
                display_name=name,
                image_url=f"http://images/{name.lower().replace(' ', '-')}.jpg",
            )
            for name in names
        ],
    )
    return attempt


def _published(attempts: list[_Attempt]) -> list[str]:
    ctx = _context()
    _published_in_plan_order(ctx, attempts)
    return [
        str(product.get("display_name"))
        for product in ctx.state.product_results
        if isinstance(product, dict)
    ]


def test_products_are_published_in_plan_order() -> None:
    sweaters = _finished("sweaters", "Polished Peplum", "Qute Cashmere")
    boots = _finished("boots", "Yantra Boots", "Zestful Boots")

    assert _published([sweaters, boots]) == [
        "Polished Peplum",
        "Qute Cashmere",
        "Yantra Boots",
        "Zestful Boots",
    ]


def test_the_order_the_scopes_finished_in_does_not_show() -> None:
    """The same two scopes, completed the other way round, publish the same.

    Nothing here simulates a race, and that is the point: the function reads the
    plan, so there is no path by which completion order could reach the shopper.
    """

    boots = _finished("boots", "Yantra Boots", "Zestful Boots")
    sweaters = _finished("sweaters", "Polished Peplum", "Qute Cashmere")

    # Boots finished first, and are still asked for second.
    assert _published([sweaters, boots]) == _published(
        [_finished("sweaters", "Polished Peplum", "Qute Cashmere"), boots]
    )


def test_the_pictures_are_published_in_the_same_order_as_the_products() -> None:
    """One order, not two. The chat renders these, the panel renders the other."""

    ctx = _context()
    _published_in_plan_order(
        ctx,
        [
            _finished("sweaters", "Polished Peplum", "Qute Cashmere"),
            _finished("boots", "Yantra Boots"),
        ],
    )

    assert list(ctx.scope.retrieved) == [
        "Polished Peplum",
        "Qute Cashmere",
        "Yantra Boots",
    ]


def test_a_scope_that_found_nothing_takes_no_position() -> None:
    empty = _attempt("jeans")
    empty.result = SimpleNamespace(ok=True, products=[])
    refused = _attempt("hats")

    assert _published(
        [empty, _finished("sweaters", "Polished Peplum"), refused]
    ) == ["Polished Peplum"]


def test_the_same_product_from_two_scopes_keeps_its_first_position() -> None:
    """Deduplication is by product id, and the earlier role owns the ordinal."""

    both: list[_Attempt] = [
        _finished("sweaters", "Polished Peplum", "Qute Cashmere"),
        _finished("knitwear", "Polished Peplum", "Gentle Meadow"),
    ]

    assert _published(both) == [
        "Polished Peplum",
        "Qute Cashmere",
        "Gentle Meadow",
    ]
