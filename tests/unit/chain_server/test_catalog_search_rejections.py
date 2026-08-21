# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every catalog-search gate names itself when it turns a scope back.

Nine of these gates render one model-visible prefix, so a refusal used to reach
diagnostics as a single undifferentiated reason. Each test below drives one real
``search_catalog`` call into one gate and asserts the code that comes back on
the artifact, so a gate that stops recording -- or two gates that start sharing
one code -- fails here rather than quietly making refusals uncountable again.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import json
import pytest

from chain_server.src import catalog_search as catalog_search_mod
from chain_server.src.agenttypes import State
from chain_server.src.catalog_search import SearchContext, search_catalog
from chain_server.src.control_signals import (
    NOT_CARRIED_KEY,
    REJECTIONS_KEY,
    SearchRejection,
)
from chain_server.src.turn_scope import TurnScope
from chain_server.src.turn_support import _search_catalog_tool_input_model
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


def _capabilities(
    *,
    retrieval_modes: tuple[str, ...] = ("text",),
    price_operators: tuple[str, ...] = ("gte", "lte"),
) -> CatalogCapabilities:
    """A small two-category catalog, with the two knobs the gates need moved."""

    return CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=list(retrieval_modes),
        filters={
            "department": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["department"],
                values=["apparel", "bags"],
            ),
            "product_type": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["product_type"],
                values=["dresses", "tote_bags", "crossbody_bags"],
            ),
            "color": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["color"],
                values=["black", "blue"],
            ),
            "price": CatalogFilterCapability(
                type="number",
                operators=list(price_operators),
                source_fields=["price"],
            ),
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="department",
            subcategory_field="product_type",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=1,
                    subcategories={
                        "dresses": CatalogTaxonomySubcategory(product_count=1),
                    },
                ),
                "bags": CatalogTaxonomyCategory(
                    product_count=2,
                    subcategories={
                        "tote_bags": CatalogTaxonomySubcategory(product_count=1),
                        "crossbody_bags": CatalogTaxonomySubcategory(
                            product_count=1
                        ),
                    },
                ),
            },
        ),
    )


def _context(
    query: str,
    capabilities: CatalogCapabilities | None = None,
) -> SearchContext:
    capabilities = capabilities or _capabilities()
    search_input_model = _search_catalog_tool_input_model(capabilities)
    return SearchContext(
        config=SimpleNamespace(
            top_k_retrieve=8,
            top_k_retrieve_broad=12,
            search_products_per_call=36,
            max_catalog_searches_per_turn=3,
            retriever_port="http://catalog-retriever:8010",
            catalog_search_timeout_seconds=5,
        ),
        state=State(user_id=1, query=query),
        scope=TurnScope(),
        capabilities=capabilities,
        search_input_model=search_input_model,
        constraint_input_model=search_input_model.model_fields[
            "required_constraints"
        ].annotation,
    )


def _scope(**overrides: Any) -> dict[str, Any]:
    scope = {
        "semantic_query": "tote bags for work",
        "shopper_guidance": "Looking for tote bags for this request.",
        "requested_product_type": "tote bags",
        "taxonomy": {"category": [], "subcategory": ["tote_bags"]},
        "required_constraints": {},
    }
    scope.update(overrides)
    return scope


def _no_products(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for retrieval, which no gate below is meant to reach."""

    return SimpleNamespace(
        result=SearchCatalogResult(ok=True, products=[]),
        fallback_attempted=False,
        fallback_used=False,
    )


@pytest.fixture(autouse=True)
def _offline_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        catalog_search_mod,
        "execute_catalog_search",
        _no_products,
    )


def _rejection_codes(result: Any) -> list[Any]:
    if not isinstance(result, tuple):
        return []
    return (result[1] or {}).get(REJECTIONS_KEY, [])


#: One case per gate reachable through the tool: the shopper turn, whatever
#: turn state the gate needs, the scope the model sent, and the code that gate
#: must record. Deriving these from real arguments rather than from a stubbed
#: attempt is deliberate -- a code recorded on a path the model cannot reach
#: attributes nothing.
GateCase = tuple[
    str,
    str,
    CatalogCapabilities | None,
    Callable[[SearchContext], None],
    dict[str, Any],
]

GATE_CASES: tuple[GateCase, ...] = (
    (
        SearchRejection.REPAIR_CHANGED_PRODUCT_SCOPE,
        "show me crossbody bags",
        None,
        lambda ctx: setattr(
            ctx.scope.repair,
            "failed_repair_scope_key",
            "crossbody bag",
        ),
        _scope(),
    ),
    (
        SearchRejection.CAPABILITIES_SCHEMA_MISMATCH,
        "show me tote bags",
        None,
        lambda ctx: None,
        _scope(taxonomy={"category": ["bags"], "subcategory": ["hatboxes"]}),
    ),
    (
        SearchRejection.REPAIR_CHANGED_CONSTRAINTS,
        "show me tote bags",
        None,
        lambda ctx: setattr(
            ctx.scope.repair,
            "pending_taxonomy_constraints",
            {"color": ["blue"]},
        ),
        _scope(required_constraints={"color": ["black"]}),
    ),
    (
        SearchRejection.TAXONOMY_NOT_ADVERTISED_FOR_SCOPE,
        "show me tote bags",
        None,
        lambda ctx: None,
        _scope(taxonomy={"category": ["bags"], "subcategory": ["dresses"]}),
    ),
    (
        SearchRejection.CONSTRAINT_REPAIR_CHANGED_REQUEST,
        "show me tote bags",
        None,
        lambda ctx: ctx.scope.repair.pending_constraint_reviews.update(
            {
                "tote bag": {
                    "requirements": ["laptop sleeve"],
                    "taxonomy": {
                        "category": [],
                        "subcategory": ["crossbody_bags"],
                    },
                    "scope_complete": True,
                    "search_mode": None,
                    "required_constraints": {},
                }
            }
        ),
        _scope(),
    ),
    (
        # The shopper named the role; the model answered it with a narrower one.
        SearchRejection.SHOPPER_SCOPE_TAXONOMY_MISMATCH,
        "show me tote bags",
        None,
        lambda ctx: None,
        _scope(
            semantic_query="handbags",
            requested_product_type="handbags",
        ),
    ),
    (
        SearchRejection.CONSTRAINT_REVIEW_REQUIRED,
        "put together a work outfit",
        None,
        lambda ctx: None,
        _scope(
            required_constraints={
                "unadvertised_requirements": ["waterproof lining"]
            },
        ),
    ),
    (
        SearchRejection.REQUIREMENT_PROVENANCE_UNESTABLISHED,
        "put together a work outfit",
        None,
        lambda ctx: ctx.scope.repair.constraint_reviewed_scopes.add("tote bag"),
        _scope(
            required_constraints={
                "unadvertised_requirements": ["waterproof lining"]
            },
        ),
    ),
    (
        SearchRejection.UNSUPPORTED_CATALOG_TAXONOMY,
        "show me handbags",
        None,
        lambda ctx: None,
        _scope(
            semantic_query="handbags",
            requested_product_type="handbags",
            taxonomy={"category": ["apparel"], "subcategory": ["tote_bags"]},
        ),
    ),
    (
        # An advertised mode this catalog names but retrieval cannot run.
        SearchRejection.UNSUPPORTED_SEARCH_MODE,
        "show me tote bags",
        _capabilities(retrieval_modes=("text", "sparse")),
        lambda ctx: None,
        _scope(search_mode="sparse"),
    ),
    (
        # An advertised filter used with an operator it does not advertise.
        SearchRejection.UNSUPPORTED_CATALOG_CONSTRAINT,
        "show me tote bags",
        _capabilities(price_operators=("gte",)),
        lambda ctx: None,
        _scope(required_constraints={"price": {"max": 100}}),
    ),
    (
        SearchRejection.CATALOG_SEARCH_LIMIT,
        "show me tote bags",
        None,
        lambda ctx: setattr(ctx.scope, "catalog_searches", 3),
        _scope(),
    ),
)


@pytest.mark.parametrize(
    ("expected_code", "query", "capabilities", "prepare", "scope"),
    GATE_CASES,
    ids=[case[0] for case in GATE_CASES],
)
def test_each_gate_records_which_gate_refused_the_scope(
    expected_code: str,
    query: str,
    capabilities: CatalogCapabilities | None,
    prepare: Callable[[SearchContext], None],
    scope: dict[str, Any],
) -> None:
    ctx = _context(query, capabilities)
    prepare(ctx)

    result = search_catalog(ctx, [scope])

    assert _rejection_codes(result) == [expected_code]


def test_repeated_shopper_scope_is_attributed_to_the_shopper_scope_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paraphrase of a search that already found something is refused.

    The first search has to return products. The rule exists to stop a retry
    rewording an *answered* search, and a scope that came back empty has not
    been answered -- relaxing a filter and looking again is the honest next
    move there, so that case is deliberately allowed.
    """

    def _with_products(plan, *_args, **_kwargs):
        return SimpleNamespace(
            result=SearchCatalogResult(
                ok=True,
                products=[
                    ProductSummary(
                        product_id="p1",
                        display_name="A Tote",
                        price=Money(amount=49.0),
                        category="tote_bags",
                    )
                ],
            ),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(
        catalog_search_mod, "execute_catalog_search", _with_products
    )
    ctx = _context("show me tote bags")

    first = search_catalog(ctx, [_scope()])
    second = search_catalog(ctx, [_scope(semantic_query="roomy tote bags")])

    assert _rejection_codes(first) == []
    assert _rejection_codes(second) == [
        SearchRejection.DUPLICATE_SHOPPER_SCOPE
    ]


def test_an_empty_scope_may_be_searched_again_with_a_filter_relaxed() -> None:
    """"No green dress in a 2" must be able to look again without the size.

    The duplicate gate keyed on the shopper's words and the product type,
    ignoring filters, so the relaxed retry was refused and three live runs
    answered with a numbered menu of things they could have searched for.
    """

    ctx = _context("show me tote bags")

    first = search_catalog(ctx, [_scope()])
    second = search_catalog(ctx, [_scope(semantic_query="roomy tote bags")])

    assert _rejection_codes(first) == []
    assert SearchRejection.DUPLICATE_SHOPPER_SCOPE not in _rejection_codes(second)


def test_repeated_catalog_scope_is_attributed_to_the_catalog_scope_gate() -> None:
    """An open role repeats the taxonomy without repeating the shopper's noun.

    The shopper never named the type, so the shopper-scope gate does not fire
    and the repeat has to be caught -- and named -- by the taxonomy-and-
    constraints gate instead.
    """

    ctx = _context("put together a work outfit")

    first = search_catalog(ctx, [_scope(semantic_query="structured work tote")])
    second = search_catalog(ctx, [_scope(semantic_query="roomy work tote")])

    assert _rejection_codes(first) == []
    assert _rejection_codes(second) == [SearchRejection.DUPLICATE_CATALOG_SCOPE]


#: Three gates keyed on a ``taxonomy_status`` the server no longer derives.
#: ``_catalog_execution_taxonomy_status`` returns six statuses and
#: ``no_direct_catalog_match`` is not one of them, which strands the two gates
#: that require it; the third needs ``exact_requested_type`` for a product type
#: the catalog does not advertise, and every route to that combination is
#: refused by the schema first. They keep their codes so that a status change
#: that revives them is attributable on the day it happens.
UNREACHABLE_GATES = frozenset(
    {
        SearchRejection.NO_ADVERTISED_TAXONOMY_MATCH,
        SearchRejection.ADVERTISED_MATCH_REPORTED_AS_GAP,
        SearchRejection.EXACT_TAXONOMY_NOT_ADVERTISED,
    }
)


def test_every_reachable_gate_code_is_exercised() -> None:
    """A new gate with no case here would be unattributable in production."""

    exercised = {case[0] for case in GATE_CASES} | {
        SearchRejection.DUPLICATE_SHOPPER_SCOPE,
        SearchRejection.DUPLICATE_CATALOG_SCOPE,
    }

    assert set(SearchRejection) - exercised == UNREACHABLE_GATES


def test_a_scope_that_runs_records_no_code_beside_one_that_was_refused() -> None:
    """One refused role in a multi-role call must not be lost or overstated.

    The codes are positional, so a reader can tell which role was turned back
    and that the other one searched -- which is what keeps a partly refused
    call from being counted as a refused call.
    """

    ctx = _context("show me tote bags and a dress")

    result = search_catalog(
        ctx,
        [
            _scope(taxonomy={"category": ["bags"], "subcategory": ["hatboxes"]}),
            _scope(
                semantic_query="dresses",
                requested_product_type="dress",
                taxonomy={"category": [], "subcategory": ["dresses"]},
            ),
        ],
    )

    assert _rejection_codes(result) == [
        SearchRejection.CAPABILITIES_SCHEMA_MISMATCH,
        None,
    ]


def test_a_search_that_runs_carries_no_rejection_key_at_all() -> None:
    ctx = _context("show me tote bags")

    result = search_catalog(ctx, [_scope()])

    assert isinstance(result, tuple)
    assert REJECTIONS_KEY not in result[1]


def test_recording_a_gate_code_leaves_the_control_signal_intact() -> None:
    """The codes ride beside the artifact the tool loop already reads."""

    ctx = _context("show me tote bags")
    ctx.scope.catalog_searches = 3

    text, artifact = search_catalog(ctx, [_scope()])

    assert text.startswith("STOP_TOOL_USE: Catalog search limit reached")
    assert artifact["control_signals"] == ["stop_tool_use"]
    assert artifact[REJECTIONS_KEY] == [SearchRejection.CATALOG_SEARCH_LIMIT]


def test_a_shopper_who_shows_you_a_garment_has_stated_its_colour() -> None:
    """Provenance was computed from the typed query alone.

    A shopper who attaches a photo and says "I like the top" never types
    "cream", so every attribute the camera conveyed read as model-invented and
    was refused -- returning nothing for a request the catalog could answer.
    """

    from chain_server.src.turn_support import stated_media_terms

    analysis = json.dumps(
        {
            "fashion_items": ["cable-knit sweater", "blue jeans"],
            "colors": ["cream", "beige"],
            "materials_or_textures": ["cable knit"],
            "style_terms": ["boho-chic"],
            "occasion": "casual fall outing",
            "search_queries": ["cable knit sweater women"],
        }
    )

    terms = stated_media_terms(analysis)

    assert "cream" in terms
    assert "cable-knit sweater" in terms
    # The model's reading of the image is not the shopper speaking.
    assert "boho-chic" not in terms
    assert "casual fall outing" not in terms
    assert "cable knit sweater women" not in terms


def test_stated_media_terms_survives_the_vlm_changing_shape() -> None:
    """The same key comes back as a string one turn and a list the next."""

    from chain_server.src.turn_support import stated_media_terms

    assert "cream" in stated_media_terms(json.dumps({"colors": "cream"}))
    assert "cream" in stated_media_terms(json.dumps({"colors": ["cream"]}))
    assert stated_media_terms(json.dumps({"colors": {"x": 1}})) == ""
    assert stated_media_terms("not json at all") == ""
    assert stated_media_terms("") == ""


def test_a_product_type_the_catalog_lacks_is_recorded_as_a_fact() -> None:
    """A rejection said the arguments were wrong. It never said aprons.

    Asked to compare two aprons in a catalog that carries none, the model sent
    an empty taxonomy -- the only honest one -- and got a schema error back.
    Whether the catalog carries aprons is not in doubt at that point: nothing it
    advertises matches the word. So the tool states it, rather than leaving the
    model to volunteer ``not_covered`` on a retry it does not always make.
    """

    ctx = _context("compare the Everyday Cotton Apron and the Linen Kitchen Apron")

    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="Everyday Cotton Apron",
                requested_product_type="apron",
                taxonomy={"category": [], "subcategory": []},
            )
        ],
    )

    text, artifact = result if isinstance(result, tuple) else (result, None)
    assert (artifact or {}).get(NOT_CARRIED_KEY) == ["apron"]
    # The model is told in the same call, not only through the artifact.
    assert "NOT_CARRIED" in text
    assert _rejection_codes(result) == [SearchRejection.CAPABILITIES_SCHEMA_MISMATCH]


def test_an_advertised_type_rejected_on_its_taxonomy_is_not_called_uncarried() -> None:
    """The catalog carries tote bags; only these arguments were wrong.

    Recording every schema mismatch as "not carried" would tell shoppers a
    catalog lacks what it sells, and would disarm the guard that stops a turn
    whose searches were all refused.
    """

    ctx = _context("show me tote bags")

    result = search_catalog(
        ctx,
        [_scope(taxonomy={"category": ["bags"], "subcategory": ["hatboxes"]})],
    )

    text, artifact = result if isinstance(result, tuple) else (result, None)
    assert NOT_CARRIED_KEY not in (artifact or {})
    assert "NOT_CARRIED" not in text


def test_one_rejected_scope_does_not_cancel_the_scopes_beside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shopper asked for a look and got nothing, over one word.

    The model composed three roles and wrote an unadvertised colour on the
    third. The whole call was refused at the tool boundary, so the two sound
    roles never reached this function -- which had been built all along to judge
    each role on its own and run the ones that stand up.
    """

    searched: list[str] = []

    def _record(plan: Any, *_args: Any, **_kwargs: Any) -> Any:
        searched.extend(plan.semantic_queries)
        return SimpleNamespace(
            result=SearchCatalogResult(ok=True, products=[]),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", _record)

    ctx = _context("a black dress and a tan tote bag")
    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="black dresses",
                requested_product_type="dresses",
                taxonomy={"category": ["apparel"], "subcategory": ["dresses"]},
                required_constraints={"color": ["black"]},
            ),
            _scope(
                semantic_query="tan tote bags",
                required_constraints={"color": ["tan"]},
            ),
        ],
    )

    # The sound role ran; the unsound one did not. The repeat is that role's
    # relaxed retry: it found nothing, so the same search runs again without
    # the constraints that may give, and the reply shows what the shop has.
    assert searched[0] == "black dresses"
    assert set(searched) == {"black dresses"}
    assert SearchRejection.CAPABILITIES_SCHEMA_MISMATCH in _rejection_codes(result)


def test_a_type_the_catalog_does_not_list_is_disclosed_not_swapped_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shown a video with jeans in it, the model searched skirts.

    Nothing said so. The disclosure only fired when a bare parent category was
    chosen, so a scope naming real subcategories looked exactly like a direct
    search for the thing that was asked for.

    Whether the swap is sound is a judgement nothing here can make -- a pump is
    a heel, a jean is not a skirt. That the shopper's type is not advertised is
    certain, and is what gets recorded.
    """

    def _one_product(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            result=SearchCatalogResult(
                ok=True,
                products=[
                    ProductSummary(
                        product_id="skirt-1",
                        display_name="A Skirt",
                        category="skirts",
                        price=Money(amount=49.99, currency="USD"),
                    )
                ],
            ),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", _one_product)

    ctx = _context("do you have jeans")
    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="dark blue jeans",
                requested_product_type="jeans",
                taxonomy={"category": ["apparel"], "subcategory": ["dresses"]},
            )
        ],
    )

    text = result[0] if isinstance(result, tuple) else result
    assert '"requested_product_type": "jeans"' in text
    assert '"requested_type_is_advertised": false' in text
    assert '"searched_types": ["dresses"]' in text


def test_an_advertised_type_is_not_reported_as_a_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for tote bags and searching tote_bags substitutes nothing."""

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", _no_products)

    ctx = _context("show me tote bags")
    result = search_catalog(ctx, [_scope()])

    text = result[0] if isinstance(result, tuple) else result
    assert "requested_type_is_advertised" not in text
