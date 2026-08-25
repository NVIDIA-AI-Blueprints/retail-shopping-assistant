"""A price the catalog's published range rules out is answered, not relaxed.

"Anything for $5 to $10" in a shop whose cheapest item is $39.90 came back
with four products -- a $139.99 bag among them -- sitting under a reply that
said we had nothing in that range. The search was correct and found nothing;
the zero-result relaxation then dropped the price bound and returned whatever
the department held.

That relaxation earns its keep when there is a near miss to offer: no green
dress in a 2, so here are the size 2 dresses in other colours. It has nothing
to offer when the requested span and the advertised span do not touch. The
floor is the answer, and the model already reads it off the field.
"""

from __future__ import annotations

from typing import Any

import pytest

from chain_server.src import catalog_search
from chain_server.src.catalog_search import (
    SearchContext,
    _Attempt,
    _outside_everything_the_shop_sells,
    _relaxed_alternatives,
)
from chain_server.src.catalog_request import CatalogSearchPlan
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
)


def _capabilities() -> CatalogCapabilities:
    return CatalogCapabilities(
        filters={
            "price": CatalogFilterCapability(
                type="number",
                operators=["range"],
                source_fields=["price"],
                min_value=39.9,
                max_value=269.99,
            ),
            "primary_color": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["primary_color"],
                values=["black", "green"],
            ),
        },
    )


class _Config:
    """Only what building a relaxed search reads.

    A None here is not a neutral stub: the arguments are evaluated before the
    call, so `ctx.config.retriever_port` raised inside the loop's own
    `except Exception: continue` and no search ran whatever the guard did.
    The first version of this test passed against a deleted guard for exactly
    that reason.
    """

    retriever_port = 8000
    catalog_search_timeout_seconds = 5
    top_k_retrieve = 8


class _State:
    image = None


def _context() -> SearchContext:
    return SearchContext(
        config=_Config(),
        state=_State(),
        scope=None,
        capabilities=_capabilities(),
        search_input_model=None,
        constraint_input_model=None,
    )


@pytest.mark.parametrize(
    "filters, unreachable",
    [
        # The whole ask sits below the floor, and above the ceiling.
        ({"price": {"min": 5, "max": 10}}, True),
        ({"price": {"min": 400}}, True),
        # These overlap the advertised span. One department may hold nothing
        # under $60, but another does, and the relaxation should go and look.
        ({"price": {"max": 60}}, False),
        ({"price": {"min": 30, "max": 45}}, False),
        ({"price": {"min": 39.9, "max": 269.99}}, False),
        # Exactly on the floor and exactly on the ceiling: an item sits at
        # each, so both are reachable and neither may be ruled out.
        ({"price": {"max": 39.9}}, False),
        ({"price": {"min": 269.99}}, False),
        # Nothing numeric to compare, so nothing is ruled out here.
        ({"primary_color": ["green"]}, False),
        ({}, False),
    ],
)
def test_only_a_span_that_misses_the_catalog_entirely_is_unreachable(
    filters: dict[str, Any], unreachable: bool
) -> None:
    assert _outside_everything_the_shop_sells(_context(), filters) is unreachable


def test_an_unreachable_price_does_not_run_a_relaxed_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The call site, not just the reading -- the retry must not be issued.

    Asserted here rather than on the helper alone because the bug was four
    products reaching the shopper's screen, and only the search running puts
    them there.
    """

    calls: list[Any] = []

    def _record(plan: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(plan)
        raise AssertionError("a relaxed search must not run for this bound")

    monkeypatch.setattr(catalog_search, "execute_catalog_search", _record)

    attempt = _Attempt(
        semantic_query="items under $10",
        requested_product_type="any",
        taxonomy={"category": ["bags"]},
        required_constraints={"price": {"min": 5, "max": 10}},
        shopper_guidance="Looking for bags within your $5 to $10 budget",
    )
    # The real plan, because a stand-in without `model_copy` throws where the
    # relaxation builds its retry -- inside the same `except Exception` -- and
    # the test then passes whether the guard is there or not.
    attempt.plan = CatalogSearchPlan(
        should_search=True,
        semantic_queries=["items under $10"],
        hard_filters={"price": {"min": 5, "max": 10}},
    )

    assert _relaxed_alternatives(_context(), attempt) == []
    assert calls == []
