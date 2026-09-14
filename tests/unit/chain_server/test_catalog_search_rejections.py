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

import json
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

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
        # Shaped wrongly, rather than worded wrongly. A value outside the
        # advertised vocabulary no longer reaches this gate -- it is set aside
        # before validation and the search runs -- so the case that proves the
        # gate is one no reconciliation can rescue.
        _scope(required_constraints={"unadvertised_requirements": "waterproof"}),
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


def test_a_role_the_shopper_never_typed_is_still_searched_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule above holds for a role read off a video, not just a typed one.

    It did not. The key it turns on was set only when the shopper's own words
    held the product type, so a look lifted from a video had no key at all and
    no retry of one of its roles was ever a duplicate.

    That is the whole of a reported failure. Sent a video of a sweater, jeans
    and boots, the model searched a role for jeans -- which this catalog does
    not carry -- under one advertised subcategory after another, five of them,
    every search succeeding because each was a correct hard-filtered slice of
    a catalog with no jeans in it. Nine model calls and 122k tokens of prompt
    to end up where the first search already was.

    Identical to its sibling above but for the one thing that matters: the
    shopper said nothing about the type.
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
    ctx = _context("I want to shop this look")

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
            _scope(
                required_constraints={
                    "unadvertised_requirements": "waterproof",
                }
            ),
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


def test_a_look_with_a_role_this_shop_does_not_stock_still_shops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A look of three where one role names a garment this shop does not sell.

    Shown a video of a sweater, jeans and boots and asked to shop it, the
    reply came back with boots alone: the sweater role carried a colour this
    catalog does not advertise, so it was refused alongside the jeans, and one
    role of three survived.

    The sweater searches now. Its colour is set aside and ranked on, because
    what remains is a search for the right garment.

    The jeans do not, and that is the point of this test as much as the
    sweater is. Setting a type aside would leave the department, and a
    department is not a family -- ranking "dark blue straight leg jeans"
    across 39 skirts and 33 dresses returns skirts, and offering those as the
    jeans substitutes a garment the shopper named. So the type stays a
    vocabulary error, said at once and not searched around.
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

    ctx = _context("I want to shop this look")
    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="cream knit sweater",
                requested_product_type="sweater",
                taxonomy={"category": ["apparel"], "subcategory": []},
                required_constraints={"color": ["cream"]},
            ),
            _scope(
                semantic_query="dark blue straight leg jeans",
                requested_product_type="jeans",
                taxonomy={"category": ["apparel"], "subcategory": ["jeans"]},
                required_constraints={"color": ["blue"]},
            ),
            _scope(
                semantic_query="brown ankle boots",
                requested_product_type="tote bags",
                taxonomy={"category": ["bags"], "subcategory": ["tote_bags"]},
            ),
        ],
    )

    # The colour was set aside, so the garment is still searched for.
    assert "cream knit sweater" in searched
    assert "brown ankle boots" in searched
    # The type was not: no skirt is offered as the jeans.
    assert "dark blue straight leg jeans" not in searched

    # Twice each at most, not thirty-two: the second is the relaxed retry of
    # a role that found nothing here, which is the tool looking again on its
    # own rather than the model being sent back to rewrite its arguments.
    assert max(Counter(searched).values()) <= 2

    text = result[0] if isinstance(result, tuple) else result
    assert "SEARCH_WORDS_RANKED_NOT_FILTERED" in text
    assert "cream" in text
    assert "jeans" in text


def test_a_carried_type_over_an_uncarried_query_is_not_the_garment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relabelling the role does not make a skirt into the jeans.

    Checking the declaration caught the model filing jeans under one
    subcategory, so it stopped declaring jeans. Live, a look whose jeans this
    shop does not carry came back as ``requested_product_type: "skirts"`` with
    ``subcategory: ["skirts"]`` -- every field advertised, every field
    agreeing -- against ``semantic_query: "dark wash straight-leg jeans"``.
    The substitution had already happened in the model's own head, and the
    garment the shopper had named survived only in the ranking text. A navy
    fitted skirt was offered as the dark bottom.

    So the query is read too: a scope may not answer a garment this shop has
    no value for with one advertised name that is a different garment.
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

    ctx = _context("I want to shop this look")
    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="dark wash straight leg jeans",
                requested_product_type="dresses",
                taxonomy={"category": ["apparel"], "subcategory": ["dresses"]},
            ),
            _scope(semantic_query="roomy tote bags"),
        ],
    )

    assert "dark wash straight leg jeans" not in searched
    assert "roomy tote bags" in searched
    assert SearchRejection.TAXONOMY_NOT_ADVERTISED_FOR_SCOPE in (
        _rejection_codes(result)
    )


def test_the_same_request_twice_is_not_run_a_second_time() -> None:
    """A repair that changed nothing is not a repair, and stops here.

    The locks that police a repair compare scope keys and constraints, so a
    retry identical to the call they turned back reads to them as a faithful
    repair and is judged again -- to the same verdict, for the same reason.
    That is the shape the look failure took: the same two payloads, over and
    over, until the turn ran out of recursion.

    Reconciliation removes the reason those two were rejected at all. This
    removes the loop, which was never specific to them.
    """

    ctx = _context("show me tote bags")
    scope = _scope(required_constraints={"unadvertised_requirements": "wet"})

    first = search_catalog(ctx, [scope])
    assert _rejection_codes(first) == [
        SearchRejection.CAPABILITIES_SCHEMA_MISMATCH
    ]

    second = search_catalog(ctx, [scope])
    text = second[0] if isinstance(second, tuple) else second
    assert "SEARCH_NOT_REPAIRED" in text
    assert _rejection_codes(second) == []


def test_a_repair_that_changed_something_is_judged_on_its_merits() -> None:
    """The backstop ends identical retries, not repair itself."""

    ctx = _context("show me tote bags")

    search_catalog(
        ctx,
        [_scope(required_constraints={"unadvertised_requirements": "wet"})],
    )
    repaired = search_catalog(ctx, [_scope()])

    text = repaired[0] if isinstance(repaired, tuple) else repaired
    assert "SEARCH_NOT_REPAIRED" not in text


def test_a_payload_the_catalog_can_honour_is_left_exactly_as_it_came() -> None:
    """Nothing is set aside, and nothing is disclosed, on the ordinary path.

    The disclosure is only true when a word could not be honoured. Emitting it
    on a search that filtered on everything it was given would tell the model
    its own filters had not been applied, and cost every turn tokens for the
    privilege.
    """

    ctx = _context("show me black dresses")

    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="black dresses",
                requested_product_type="dresses",
                taxonomy={"category": ["apparel"], "subcategory": ["dresses"]},
                required_constraints={"color": ["black"]},
            )
        ],
    )

    text = result[0] if isinstance(result, tuple) else result
    assert "SEARCH_WORDS_RANKED_NOT_FILTERED" not in text
    assert _rejection_codes(result) == []


def test_a_word_the_catalog_cannot_filter_on_does_not_cost_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shopper asked for a look and got nothing, over one word.

    The model composed the roles and wrote an unadvertised colour on one of
    them. That role was refused, and a refused role is a role in repair --
    policed by locks that hold one scope at a time, so with two bad roles the
    retries were judged against each other's lock and the turn spent its whole
    budget on them.

    Nothing about "tan" needed the model. It is not one of the colours this
    catalog advertises, so it cannot be a filter, and it is already in the
    query where the index can rank on it. So the filter drops it, the search
    runs, and the result says the word was ranked on rather than filtered by.
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

    # Both roles ran, and neither was refused.
    assert set(searched) == {"black dresses", "tan tote bags"}
    assert _rejection_codes(result) == []

    # The colour could not be honoured as a filter, so the results do not
    # promise it. That difference is said out loud rather than left for the
    # shopper to find on a product page.
    text = result[0] if isinstance(result, tuple) else result
    assert "SEARCH_WORDS_RANKED_NOT_FILTERED" in text
    assert "tan" in text


def test_the_advertised_half_of_a_colour_list_still_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the word this shop does not use goes; the rest stays a filter.

    Dropping the whole field was tried here first, and it is worse. The field
    *is* the filter, so losing it leaves no colour constraint at all: asked
    for a cream sweater as `["cream", "beige"]`, the search ranked on
    "cable-knit" alone and came back with sweaters in any colour, red among
    them.

    Keeping half can still be narrower than the model meant -- `["cream",
    "white"]` filters to white in a shop whose cream is beige -- which is why
    the scope prompt asks for every advertised value the shopper's word could
    be rather than the single nearest, and why what could not be honoured is
    disclosed either way.
    """

    filters: list[dict[str, Any]] = []

    def _record(plan: Any, *_args: Any, **_kwargs: Any) -> Any:
        filters.append(dict(plan.hard_filters))
        return SimpleNamespace(
            result=SearchCatalogResult(ok=True, products=[]),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", _record)

    # This catalog advertises black and blue, so "black" is the half that
    # survives the vocabulary check and "cream" is the half that cannot.
    ctx = _context("I want to shop this look, the cream sweater")
    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="cream cable-knit sweater",
                required_constraints={"color": ["cream", "black"]},
            )
        ],
    )

    # The first search is the one this test is about. A later entry is the
    # relaxed retry the tool runs on its own when a scope finds nothing, and
    # dropping the filter is the whole point of that one.
    assert filters
    first = filters[0]
    # Black is advertised, so it still filters -- which is the point: without
    # it the colour stops constraining anything at all.
    assert "black" in str(first.get("color"))
    # Cream is not, so it never reaches the database.
    assert "cream" not in str(first.get("color")).casefold()
    assert _rejection_codes(result) == []

    text = result[0] if isinstance(result, tuple) else result
    assert "SEARCH_WORDS_RANKED_NOT_FILTERED" in text
    assert "cream" in text


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


def _sized(name, values):
    from shared.commerce_contracts import (
        CatalogCoverage,
        CatalogFieldCapability,
        CatalogTaxonomySubcategory,
        CatalogValueCapability,
    )
    return name, CatalogTaxonomySubcategory(
        product_count=1,
        filters={"sizes": CatalogFieldCapability(
            type="enum",
            operators=["in"],
            source_fields=["sizes"],
            coverage=CatalogCoverage(present=1, total=1),
            values=[CatalogValueCapability(value=v, count=1) for v in values],
        )},
    )


def _catalog_with_sizes():
    from shared.commerce_contracts import (
        CatalogCapabilities,
        CatalogTaxonomyCapabilities,
        CatalogTaxonomyCategory,
    )
    bags = dict([_sized("tote_bags", ["onesize"]), _sized("clutches", ["onesize"])])
    apparel = dict([_sized("dresses", ["2", "4", "6", "8", "10", "12"])])
    from shared.commerce_contracts import CatalogFilterCapability
    return CatalogCapabilities(
        catalog_id="fashion", retrieval_modes=["text"],
        filters={
            "sizes": CatalogFilterCapability(
                type="enum", operators=["in"], source_fields=["sizes"],
                values=["onesize", "2", "4", "6", "8", "10", "12"],
            )
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category", subcategory_field="subcategory",
            categories={
                "bags": CatalogTaxonomyCategory(product_count=2, subcategories=bags),
                "apparel": CatalogTaxonomyCategory(product_count=1, subcategories=apparel),
            },
        ),
    )


def _asked(subcategories, sizes):
    from types import SimpleNamespace

    from chain_server.src.catalog_search import _size_that_cannot_apply
    return _size_that_cannot_apply(
        SimpleNamespace(subcategory=subcategories),
        {"sizes": sizes},
        _catalog_with_sizes(),
    )


def test_a_number_asked_of_a_onesize_scope_cannot_apply() -> None:
    """J01 t11, "do you have a tote bag in a size 8".

    A turn was spent on "there aren't any in that size -- would you like me to
    show you tote bags in their standard one size?" with four tote bags already
    in the result. The assistant was obeying the zero-result rule, which says a
    size is never the filter you give up and to offer rather than show. That is
    right for a garment and meaningless for a tote: every bags subcategory
    advertises onesize and nothing else.
    """

    assert _asked(["tote_bags"], ["8"]) == "8"


def test_a_size_a_garment_really_comes_in_is_kept() -> None:
    """The rule must not start dropping sizes from clothes."""

    assert _asked(["dresses"], ["8"]) == ""


def test_a_mixed_scope_keeps_the_size() -> None:
    """One sized subcategory in the scope and the size still means something."""

    assert _asked(["tote_bags", "dresses"], ["8"]) == ""


def test_onesize_asked_of_a_onesize_scope_is_not_dropped() -> None:
    assert _asked(["tote_bags"], ["onesize"]) == ""


def test_no_subcategory_narrows_nothing() -> None:
    assert _asked([], ["8"]) == ""


# The wiring for the inapplicable-size drop is proved by J01 t11 live, not
# here. A unit attempt kept tripping a different gate: a per-subcategory sizes
# capability makes "8" an unsupported constraint before the drop is reached,
# which is not the path production takes -- there the size is advertised
# catalog-wide, passes validation, and matches nothing. Fighting the fixture
# into the production shape tests the fixture.


def test_several_categories_fan_out_with_their_subcategories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """J10 t1, "I'm shopping on a tight budget, nothing over $50".

    The shopper names no category and no product type, so every category is in
    scope. The model asks for exactly that -- all of them with a price ceiling
    -- and one category per scope turned the whole call back.

    Each split scope carries its own category's advertised subcategories, which
    is what the refusal asks for in as many words: "select every advertised
    subcategory that role covers". That satisfies the open-role rule rather
    than relaxing it.
    """

    seen: list = []

    def _capture(plan, url, **kw):
        seen.append(dict(plan.hard_filters or {}))
        return SimpleNamespace(
            result=SearchCatalogResult(ok=True, products=[]),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", _capture)
    ctx = _context("nothing over $50")
    ctx.config.max_search_scopes_per_call = 10

    search_catalog(
        ctx,
        [
            _scope(
                semantic_query="anything under fifty",
                requested_product_type="items",
                taxonomy={"category": ["apparel", "bags"], "subcategory": []},
                required_constraints={"price": {"max": 50}},
            )
        ],
    )

    # Two scopes, each with its own category and that category's advertised
    # subcategories. Further calls are the zero-result relaxation retrying,
    # which is a different mechanism.
    scoped = [f for f in seen if f.get("department")]
    assert [f["department"] for f in scoped] == [["apparel"], ["bags"]], seen
    assert scoped[0]["product_type"] == ["dresses"]
    assert scoped[1]["product_type"] == ["crossbody_bags", "tote_bags"]
    assert all(f["price"] == {"max": 50.0} for f in scoped)


def test_one_category_and_no_subcategory_is_left_alone() -> None:
    """"Rainy outfit under $60" arrives as apparel with a price and nothing
    else. Filling in every apparel subcategory would answer an invented role
    with the whole department, so this shape keeps its refusal."""

    from chain_server.src.catalog_search import _one_scope_per_category

    ctx = SimpleNamespace(
        capabilities=_capabilities(),
        config=SimpleNamespace(max_search_scopes_per_call=10),
    )
    scope = {
        "semantic_query": "rainy outfit under $60",
        "taxonomy": {"category": ["apparel"], "subcategory": []},
    }

    assert _one_scope_per_category(ctx, [scope]) == [scope]


def test_a_scope_that_names_subcategories_is_left_alone() -> None:
    from chain_server.src.catalog_search import _one_scope_per_category

    ctx = SimpleNamespace(
        capabilities=_capabilities(),
        config=SimpleNamespace(max_search_scopes_per_call=10),
    )
    scope = {
        "semantic_query": "dresses and totes",
        "taxonomy": {
            "category": ["apparel", "bags"],
            "subcategory": ["dresses", "tote_bags"],
        },
    }

    assert _one_scope_per_category(ctx, [scope]) == [scope]


def test_a_scopeless_browse_is_shown_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """J10 t1, "I'm shopping on a tight budget, nothing over $50".

    This used to be refused: the model named apparel, the shopper had named no
    product type, and the open-role rule demanded subcategories to justify the
    choice. Measured, that produced "could you clarify the product type" three
    runs in five, on a request that was already complete.

    The worst case being guarded against was a PARTIAL answer -- clothes under
    $50 rather than everything under $50. The cost of guarding it was no answer
    at all. So it runs, and the narrowing is disclosed the way a product chosen
    from a description is.
    """

    def _one(*_a, **_k):
        return SimpleNamespace(
            result=SearchCatalogResult(
                ok=True,
                products=[
                    ProductSummary(
                        product_id="d1",
                        display_name="Black Polka-Dotted Slip Dress",
                        category="dresses",
                        price=Money(amount=49.90, currency="USD"),
                    )
                ],
            ),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", _one)
    ctx = _context("I'm shopping on a tight budget, nothing over $50")

    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="anything under fifty",
                # A generic noun, not a product type that binds to a category:
                # "apparel" as a requested_product_type is turned back by a
                # different check, and that one is still right.
                requested_product_type="items",
                taxonomy={"category": ["apparel"], "subcategory": []},
                required_constraints={"price": {"max": 50}},
            )
        ],
    )

    assert _rejection_codes(result) == []


def test_a_narrowed_role_is_not_told_to_widen() -> None:
    """A role with no filter behind it gets the narrowing advice only. The
    wider shapes are for a request that named no type at all."""

    ctx = _context("something for a rainy day")

    result = search_catalog(
        ctx,
        [
            _scope(
                semantic_query="rainy outfit",
                requested_product_type="apparel",
                taxonomy={"category": ["apparel"], "subcategory": []},
                required_constraints={},
            )
        ],
    )

    text = result[0] if isinstance(result, tuple) else result
    assert "leave the category out" not in text
