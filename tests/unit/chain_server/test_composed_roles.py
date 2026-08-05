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
