# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A role the shopper never named is searched, and recorded as the model's.

Dressing someone means "a top", not "a sweater specifically". The catalog's
split into blouses and sweaters is its own business, so a role that spans both
used to be turned back until the model collapsed it to one -- costing a round
trip and, every time it fired, a narrower answer than the shopper asked for.

It now searches. What these tests hold is the other half of the bargain: the
reply must be able to say the role was the assistant's idea, and must not read
a miss inside two advertised types as the role being unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chain_server.src import catalog_search as catalog_search_mod
from chain_server.src.agenttypes import State
from chain_server.src.catalog_search import SearchContext, search_catalog
from chain_server.src.control_signals import REJECTIONS_KEY
from chain_server.src.tool_evidence import EVIDENCE_KEY
from chain_server.src.turn_scope import TurnScope
from chain_server.src import turn_support
from chain_server.src.turn_support import (
    _scope_relation_line,
    _scope_relation_payload,
    _search_catalog_tool_input_model,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogTaxonomySubcategory,
    Money,
    ProductSummary,
    SearchCatalogResult,
)


def _subcategory() -> CatalogTaxonomySubcategory:
    return CatalogTaxonomySubcategory(product_count=3)


def _capabilities() -> CatalogCapabilities:
    return CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=["text"],
        filters={
            "department": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["department"],
                values=["apparel", "footwear"],
            ),
            "product_type": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["product_type"],
                values=["blouses", "sweaters", "camisoles", "flats", "sandals"],
            ),
            "price": CatalogFilterCapability(
                type="number",
                operators=["gte", "lte"],
                source_fields=["price"],
            ),
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="department",
            subcategory_field="product_type",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=9,
                    subcategories={
                        "blouses": _subcategory(),
                        "sweaters": _subcategory(),
                        "camisoles": _subcategory(),
                    },
                ),
                "footwear": CatalogTaxonomyCategory(
                    product_count=6,
                    subcategories={
                        "flats": _subcategory(),
                        "sandals": _subcategory(),
                    },
                ),
            },
        ),
    )


def _context(query: str) -> SearchContext:
    capabilities = _capabilities()
    model = _search_catalog_tool_input_model(capabilities)
    return SearchContext(
        config=SimpleNamespace(
            top_k_retrieve=4,
            search_products_per_call=36,
            max_catalog_searches_per_turn=3,
            retriever_port="http://catalog-retriever:8010",
            catalog_search_timeout_seconds=5,
        ),
        state=State(user_id=1, query=query),
        scope=TurnScope(),
        capabilities=capabilities,
        search_input_model=model,
        constraint_input_model=model.model_fields[
            "required_constraints"
        ].annotation,
    )


def _role(rpt: str, subcategories: list[str], **overrides: Any) -> dict[str, Any]:
    scope = {
        "semantic_query": f"casual {rpt}",
        "shopper_guidance": f"Finding a {rpt} for this look.",
        "requested_product_type": rpt,
        "taxonomy": {"category": [], "subcategory": subcategories},
        "required_constraints": {},
    }
    scope.update(overrides)
    return scope


@pytest.fixture
def retrieval(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what each role actually sent to retrieval, and return products."""

    seen: dict[str, Any] = {"filters": [], "products": True}

    def execute(plan, *_args, **_kwargs):
        seen["filters"].append(dict(plan.hard_filters))
        products = (
            [
                ProductSummary(
                    product_id=f"p{index}",
                    display_name=f"Product {index}",
                    price=Money(amount=20.0),
                    category="sweaters",
                )
                for index in range(3)
            ]
            if seen["products"]
            else []
        )
        return SimpleNamespace(
            result=SearchCatalogResult(ok=True, products=products),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", execute)
    return seen


def _evidence(result: Any) -> dict[str, Any]:
    return (result[1] or {})[EVIDENCE_KEY]


def test_a_role_the_shopper_never_named_is_searched_not_refused(
    retrieval: dict[str, Any],
) -> None:
    """The whole point: no round trip spent teaching the model a taxonomy."""

    ctx = _context("make the outfit more casual and cheaper, keep the jeans")

    result = search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])

    assert (result[1] or {}).get(REJECTIONS_KEY) is None
    assert _evidence(result)["outcome"] == "results"
    assert retrieval["filters"][0]["product_type"] == ["blouses", "sweaters"]


def test_the_evidence_records_that_the_model_proposed_the_role(
    retrieval: dict[str, Any],
) -> None:
    ctx = _context("make the outfit more casual and cheaper, keep the jeans")

    payload = _evidence(
        search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])
    )

    assert payload["composed_role"] is True
    assert payload["role_advertised_types"] == ["blouses", "sweaters"]
    assert payload["requested_product_type"] == "top"


def test_a_role_the_shopper_did_name_is_not_recorded_as_composed(
    retrieval: dict[str, Any],
) -> None:
    """Over-claiming here would tell shoppers we invented what they asked for."""

    ctx = _context("show me some blouses")

    payload = _evidence(search_catalog(ctx, [_role("blouses", ["blouses"])]))

    assert payload["composed_role"] is False
    assert payload["role_advertised_types"] == []


def test_several_composed_roles_each_retrieve_under_their_own_filters(
    retrieval: dict[str, Any],
) -> None:
    """The whole-look call: one round trip, every role answered.

    This is the shape that used to cost three round trips and end with one
    role answered -- the shoes forced down to flats, sandals discarded.
    """

    ctx = _context("build me the whole look under $130")

    result = search_catalog(
        ctx,
        [
            _role("top", ["blouses", "sweaters"]),
            _role("shoes", ["flats", "sandals"]),
        ],
    )

    assert (result[1] or {}).get(REJECTIONS_KEY) is None
    assert ctx.scope.catalog_searches == 2
    assert [f["product_type"] for f in retrieval["filters"]] == [
        ["blouses", "sweaters"],
        ["flats", "sandals"],
    ]


def test_the_composer_is_told_the_role_was_proposed(
    retrieval: dict[str, Any],
) -> None:
    ctx = _context("make the outfit more casual, keep the jeans")

    payload = _evidence(
        search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])
    )
    relation = _scope_relation_payload(payload)
    line = _scope_relation_line(payload, has_products=True)

    assert relation["relation"] == "model_composed_role"
    assert relation["requested_product_type"] == "top"
    assert "did not ask for top" in line
    assert "proposed by the assistant" in line
    # Naming the range alone is an inventory note. It lists what is on the
    # shelves and still hands the shopper an outfit built on an assumption
    # nobody stated, which is what a live guest turn actually did.
    assert "who the catalog serves" in line
    assert "assumed the pieces are for the shopper" in line
    assert "Never ask who they are" in line


def test_a_composed_role_with_no_matches_names_what_was_searched(
    retrieval: dict[str, Any],
) -> None:
    """Absence inside two of five advertised types is not absence of the role.

    Without this the composer says "there are no tops", which is a claim the
    search never made and the catalog does not support.
    """

    retrieval["products"] = False
    ctx = _context("make the outfit more casual, keep the jeans")

    payload = _evidence(
        search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])
    )
    line = _scope_relation_line(payload, has_products=False)

    assert payload["outcome"] == "zero_results"
    assert "blouses, sweaters" in line
    assert "do not claim the role is unavailable" in line


def test_the_model_visible_text_carries_the_relation(
    retrieval: dict[str, Any],
) -> None:
    ctx = _context("make the outfit more casual, keep the jeans")

    text, _ = search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])

    assert "SEARCH_SCOPE_RELATION_EVIDENCE:" in text
    assert "model_composed_role" in text


def _priced(name: str, amount: float, category: str) -> ProductSummary:
    return ProductSummary(
        product_id=f"ref-{name}",
        display_name=name,
        price=Money(amount=amount),
        category=category,
    )


def test_each_product_carries_the_filters_its_own_role_was_searched_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filter belonging to one role must never be stated about another's.

    Reproduces a real turn: the model capped the shoes at $59.99 and left the
    layer uncapped. The merged evidence stated the union against every product,
    so a $179.99 sweater was recorded as confirmed under the shoes' cap -- and
    the reply repeated it, listing a $149.99 item under a "$59.99 max" heading.
    """

    def execute(plan, *_args, **_kwargs):
        shoes = "flats" in str(plan.hard_filters.get("product_type") or [])
        products = (
            [_priced("Navy Flats", 59.99, "flats")]
            if shoes
            else [
                _priced("Gentle Meadow Sweater", 49.99, "sweaters"),
                _priced("Jade Serenity Sweater", 179.99, "sweaters"),
            ]
        )
        return SimpleNamespace(
            result=SearchCatalogResult(ok=True, products=products),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", execute)
    ctx = _context("brighten the layer and swap to a cheaper black flat")

    result = search_catalog(
        ctx,
        [
            _role("layer", ["sweaters"]),
            _role(
                "flats",
                ["flats"],
                required_constraints={"price": {"max": 59.99}},
            ),
        ],
    )

    by_name = {
        product["name"]: product for product in _evidence(result)["products"]
    }
    layer_scope = by_name["Jade Serenity Sweater"]["search_scope"]
    shoe_scope = by_name["Navy Flats"]["search_scope"]

    assert layer_scope["confirmed_filters"] == {}
    assert shoe_scope["confirmed_filters"] == {"price": {"max": 59.99}}
    assert layer_scope["taxonomy"] != shoe_scope["taxonomy"]


def test_the_composer_never_states_one_role_s_filter_about_another_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The filter statements the shopper reads are grouped by the real scope."""

    payload = {
        "outcome": "results",
        "confirmed_filters": {"price": {"max": 59.99}},
        "products": [
            {
                "name": "Jade Serenity Sweater",
                "search_scope": {"taxonomy": {}, "confirmed_filters": {}},
            },
            {
                "name": "Navy Flats",
                "search_scope": {
                    "taxonomy": {},
                    "confirmed_filters": {"price": {"max": 59.99}},
                },
            },
        ],
    }

    groups = turn_support._products_by_confirmed_filters(payload)

    assert [
        ([p["name"] for p in products], filters) for filters, products in groups
    ] == [
        (["Jade Serenity Sweater"], {}),
        (["Navy Flats"], {"price": {"max": 59.99}}),
    ]


def test_a_proposed_role_that_found_nothing_is_visible_to_an_operator(
    retrieval: dict[str, Any],
) -> None:
    """The case the disclosure exists for must be countable, not just handled.

    A role nobody named that returned nothing is indistinguishable, in a trace,
    from a shopper asking for something the catalog lacks -- unless the outcome
    says which it was. Every zero-result scope observed in evaluation so far has
    been shopper-named, so this path has never run outside a test.
    """

    retrieval["products"] = False
    ctx = _context("make the outfit more casual, keep the jeans")

    payload = _evidence(
        search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])
    )

    assert payload["scope_outcome"]["outcome"] == "zero_results"
    assert payload["scope_outcome"]["composed_role"] is True
    assert payload["scope_outcome"]["requested_product_type"] == "top"


def test_a_shopper_named_role_that_found_nothing_is_not_marked_proposed(
    retrieval: dict[str, Any],
) -> None:
    retrieval["products"] = False
    ctx = _context("show me some blouses")

    payload = _evidence(search_catalog(ctx, [_role("blouses", ["blouses"])]))

    assert payload["scope_outcome"]["composed_role"] is False


def test_the_composer_summary_carries_the_proposed_role_disclosure(
    retrieval: dict[str, Any],
) -> None:
    """The line has to survive the hop from evidence into the composer's brief.

    The equivalent parent-category disclosure is observed reaching shoppers in
    real runs, so this channel works; what was untested is that a composed role
    reaches it too.
    """

    retrieval["products"] = False
    ctx = _context("make the outfit more casual, keep the jeans")
    result = search_catalog(ctx, [_role("top", ["blouses", "sweaters"])])
    message = SimpleNamespace(artifact=result[1], content=result[0])

    summary = turn_support._customer_safe_tool_evidence(result[0], message)

    assert "did not ask for top" in summary
    assert "blouses, sweaters" in summary
    assert "do not claim the role is unavailable" in summary


def test_each_role_in_one_call_keeps_its_own_proposed_flag(
    retrieval: dict[str, Any],
) -> None:
    """One call can mix a role the shopper named with one the model added."""

    ctx = _context("show me some blouses")

    result = search_catalog(
        ctx,
        [_role("blouses", ["blouses"]), _role("shoes", ["flats", "sandals"])],
    )

    # Products are appended in scope order, so the named role's results come
    # first. They are keyed by position rather than name because the fake
    # retrieval returns the same names for every role.
    stamps = [
        product["search_scope"]["composed_role"]
        for product in _evidence(result)["products"]
    ]
    assert stamps == [False, False, False, True, True, True]


def _sweater_scope(**constraints: Any) -> dict[str, Any]:
    return {
        "semantic_query": "black sweater under $60",
        "shopper_guidance": "Finding a sweater for this request.",
        "requested_product_type": "sweater",
        "taxonomy": {"category": [], "subcategory": ["sweaters"]},
        "required_constraints": constraints,
    }


def test_two_looks_at_one_role_in_one_call_both_retrieve(
    retrieval: dict[str, Any],
) -> None:
    """A request and its fallback are not retries of each other.

    "black crew neck, or if not, any black one under $60" is one call with two
    differently filtered roles. The first used to reserve the shopper scope and
    the second was then refused as a duplicate of its own sibling, so half a
    correctly formed request never left the building and the answer arrived a
    turn late.
    """

    ctx = _context(
        "show me sweaters between $40 and $60; if nothing matches show me "
        "anything under $60"
    )

    result = search_catalog(
        ctx,
        [
            _sweater_scope(price={"min": 40, "max": 60}),
            _sweater_scope(price={"max": 60}),
        ],
    )

    assert (result[1] or {}).get(REJECTIONS_KEY) is None
    assert len(retrieval["filters"]) == 2
    assert retrieval["filters"][0]["price"] == {"min": 40.0, "max": 60.0}
    assert retrieval["filters"][1]["price"] == {"max": 60.0}


def test_an_identical_sibling_in_one_call_still_retrieves_once(
    retrieval: dict[str, Any],
) -> None:
    """Relaxing the sibling rule must not let the same retrieval run twice."""

    ctx = _context("show me black sweaters under $60")

    result = search_catalog(
        ctx,
        [
            _sweater_scope(price={"max": 60}),
            _sweater_scope(price={"max": 60}),
        ],
    )

    assert (result[1] or {})[REJECTIONS_KEY] == [
        None,
        "duplicate_catalog_scope",
    ]
    assert len(retrieval["filters"]) == 1


def test_the_same_role_in_a_later_call_is_still_refused(
    retrieval: dict[str, Any],
) -> None:
    """The rule still does its real job: stopping a paraphrased retry."""

    ctx = _context("show me black sweaters under $60")

    search_catalog(ctx, [_sweater_scope(price={"max": 60})])
    result = search_catalog(
        ctx,
        [{**_sweater_scope(price={"max": 60}), "semantic_query": "dark knitwear"}],
    )

    assert (result[1] or {})[REJECTIONS_KEY] == ["duplicate_shopper_scope"]
    assert len(retrieval["filters"]) == 1
