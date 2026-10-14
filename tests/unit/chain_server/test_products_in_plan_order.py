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
from chain_server.src.runtime.runtime import the_showing
from chain_server.src.runtime.turn_scope import TurnScope
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


def test_each_scope_becomes_one_headed_group() -> None:
    """The shopper reads two headed lists, so the turn records two groups.

    The boundary is the scope that answered, taken where it is already known.
    Recovering it afterwards would mean comparing category strings between
    products, which two scopes answering out of one category defeat.
    """

    ctx = _context()
    _published_in_plan_order(
        ctx,
        [
            _finished("sweaters", "Polished Peplum", "Qute Cashmere"),
            _finished("boots", "Yantra Boots"),
        ],
    )

    showing = the_showing(ctx.state.product_results, ctx.state.product_groups)

    assert [
        (group["heading"], [p["display_name"] for p in group["products"]])
        for group in showing
    ] == [
        ("sweaters", ["Polished Peplum", "Qute Cashmere"]),
        ("boots", ["Yantra Boots"]),
    ]


def test_the_number_restarts_inside_each_group() -> None:
    """Because that is how the shopper counts: "the first boots" is the boots.

    Numbered straight through, the boots began at three and the phrase had no
    answer the structure could give.
    """

    ctx = _context()
    _published_in_plan_order(
        ctx,
        [
            _finished("sweaters", "Polished Peplum", "Qute Cashmere"),
            _finished("boots", "Yantra Boots"),
        ],
    )

    showing = the_showing(ctx.state.product_results, ctx.state.product_groups)

    assert [
        (group["heading"], [(p["position"], p["display_name"]) for p in group["products"]])
        for group in showing
    ] == [
        ("sweaters", [(1, "Polished Peplum"), (2, "Qute Cashmere")]),
        ("boots", [(1, "Yantra Boots")]),
    ]


def test_a_group_headed_with_one_of_its_own_products_loses_the_heading() -> None:
    """Asked to add one tote by name, the model sends that name as the type.

    All four totes would then be headed "Ombre Canvas Tote Bag", three of which
    are not that. Settled by comparing the heading with the showing's own
    products -- data in hand, not a list of words.
    """

    ctx = _context()
    _published_in_plan_order(
        ctx,
        [_finished("Polished Peplum", "Polished Peplum", "Qute Cashmere")],
    )

    showing = the_showing(ctx.state.product_results, ctx.state.product_groups)

    assert [group["heading"] for group in showing] == [""]
    assert [p["display_name"] for p in showing[0]["products"]] == [
        "Polished Peplum",
        "Qute Cashmere",
    ]


def test_a_product_no_scope_claimed_is_still_shown() -> None:
    """A name lookup puts products on screen without going through a scope."""

    ctx = _context()
    _published_in_plan_order(ctx, [_finished("sweaters", "Polished Peplum")])
    ctx.state.product_results.append(
        ProductSummary(
            product_id="yantra-boots", display_name="Yantra Boots"
        ).model_dump(mode="json")
    )

    showing = the_showing(ctx.state.product_results, ctx.state.product_groups)

    assert [
        (group["heading"], [p["display_name"] for p in group["products"]])
        for group in showing
    ] == [("sweaters", ["Polished Peplum"]), ("", ["Yantra Boots"])]


def test_the_position_never_reaches_the_product_record() -> None:
    """`ProductSummary` forbids unknown fields, and the memory path re-parses it.

    Stamping the position into `state.product_results` would fail validation
    there, so it is added only to the copy handed to the client.
    """

    ctx = _context()
    _published_in_plan_order(ctx, [_finished("sweaters", "Polished Peplum")])
    the_showing(ctx.state.product_results, ctx.state.product_groups)

    assert "position" not in ctx.state.product_results[0]
    ProductSummary(**ctx.state.product_results[0])
