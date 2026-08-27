# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A search nobody scoped by audience still comes back with one; say so.

A catalog that is almost entirely womenswear answers "I need a work outfit"
with womenswear no matter who asked. The shopper is the only party in the
exchange who cannot see that nobody chose that, so the reply owes them the
sentence.

This started life bolted onto the composed-role disclosure, which was wrong in
a way that took a live turn to expose: a shopper who says "work casual outfit"
in their own words has named the role, so nothing is composed, so the sentence
never fired -- on exactly the turn that needed it. The trigger is the search
not filtering on audience, and nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chain_server.src import catalog_search as catalog_search_mod
from chain_server.src.agenttypes import State
from chain_server.src.catalog_search import (
    SearchContext,
    _assumed_audience,
    search_catalog,
)
from chain_server.src.tool_evidence import EVIDENCE_KEY
from chain_server.src.turn_scope import TurnScope
from chain_server.src.turn_support import (
    _assumed_audience_line,
    _required_constraints_input_model,
    _audience_assumption_events,
    _turn_audience_events,
    _customer_safe_search_evidence,
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

AUDIENCE_FIELD = "target_audience"


def _capabilities() -> CatalogCapabilities:
    return CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=["text"],
        filters={
            "department": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["department"],
                values=["apparel"],
            ),
            "product_type": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["product_type"],
                values=["blouses", "sweaters", "skirts"],
            ),
            "pattern": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["pattern"],
                values=["solid", "striped"],
            ),
            AUDIENCE_FIELD: CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=[AUDIENCE_FIELD],
                values=["womens", "adult_all_genders"],
            ),
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="department",
            subcategory_field="product_type",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=9,
                    subcategories={
                        "blouses": CatalogTaxonomySubcategory(product_count=3),
                        "sweaters": CatalogTaxonomySubcategory(product_count=3),
                        "skirts": CatalogTaxonomySubcategory(product_count=3),
                    },
                ),
            },
        ),
    )


def _context(query: str, *, already_disclosed: list[str] | None = None) -> SearchContext:
    capabilities = _capabilities()
    model = _search_catalog_tool_input_model(capabilities)
    return SearchContext(
        config=SimpleNamespace(
            top_k_retrieve=4,
            search_products_per_call=36,
            max_catalog_searches_per_turn=3,
            retriever_port="http://catalog-retriever:8010",
            catalog_search_timeout_seconds=5,
            wearer_audience_field=AUDIENCE_FIELD,
        ),
        state=State(
            user_id=1,
            query=query,
            assumed_audience=list(already_disclosed or []),
        ),
        scope=TurnScope(),
        capabilities=capabilities,
        search_input_model=model,
        constraint_input_model=model.model_fields[
            "required_constraints"
        ].annotation,
    )


def _role(rpt: str, subcategories: list[str], **overrides: Any) -> dict[str, Any]:
    scope = {
        "semantic_query": f"work casual {rpt}",
        "shopper_guidance": f"Finding a {rpt} for this look.",
        "requested_product_type": rpt,
        "taxonomy": {"category": [], "subcategory": subcategories},
        "required_constraints": {},
    }
    scope.update(overrides)
    return scope


@pytest.fixture
def retrieval(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return products carrying the audience the catalog tagged them with."""

    seen: dict[str, Any] = {"audiences": ["womens"]}

    def execute(plan, *_args, **_kwargs):
        products = [
            ProductSummary(
                product_id=f"p{index}",
                display_name=f"Product {index}",
                price=Money(amount=20.0),
                category="sweaters",
                attributes={AUDIENCE_FIELD: audience},
            )
            for index, audience in enumerate(seen["audiences"])
        ]
        return SimpleNamespace(
            result=SearchCatalogResult(ok=True, products=products),
            fallback_attempted=False,
            fallback_used=False,
        )

    monkeypatch.setattr(catalog_search_mod, "execute_catalog_search", execute)
    return seen


def _evidence(result: Any) -> dict[str, Any]:
    return (result[1] or {})[EVIDENCE_KEY]


def test_a_role_the_shopper_named_still_discloses_the_audience(
    retrieval: dict[str, Any],
) -> None:
    """The live regression, in one test.

    "I have a conference next week and am looking for a work casual outfit for
    less than $40" produced a single role the shopper had named, so no role was
    composed -- and the reply opened "Here are work-casual conference pieces
    I'm seeing that stay under $40", having silently decided the shopper wanted
    womenswear.
    """

    ctx = _context("looking for a work casual outfit for less than $40")

    payload = _evidence(
        search_catalog(ctx, [_role("work casual outfit", ["blouses", "skirts"])])
    )

    assert payload["composed_role"] is False
    assert payload["assumed_audience"] == ["womens"]
    assert "ASSUMED_AUDIENCE" in _customer_safe_search_evidence(payload)


def test_a_stated_audience_is_not_an_assumption(
    retrieval: dict[str, Any],
) -> None:
    """Filtering on it makes it the shopper's constraint, not the shop's guess.

    Disclosing it anyway would tell shoppers who asked for menswear that we
    have assumed they want menswear, which reads as not having listened.
    """

    ctx = _context("something for my husband")

    payload = _evidence(
        search_catalog(
            ctx,
            [
                _role(
                    "sweaters",
                    ["sweaters"],
                    required_constraints={
                        AUDIENCE_FIELD: ["adult_all_genders"]
                    },
                )
            ],
        )
    )

    assert payload["confirmed_filters"].get(AUDIENCE_FIELD)
    assert payload["assumed_audience"] == []
    assert "ASSUMED_AUDIENCE" not in _customer_safe_search_evidence(payload)


def test_every_audience_that_came_back_is_named(
    retrieval: dict[str, Any],
) -> None:
    """A mixed result set is a mixed disclosure, not the majority value."""

    retrieval["audiences"] = ["womens", "adult_all_genders", "womens"]
    ctx = _context("a work casual outfit")

    payload = _evidence(
        search_catalog(ctx, [_role("work casual outfit", ["blouses"])])
    )

    assert sorted(payload["assumed_audience"]) == ["adult_all_genders", "womens"]


def test_the_whole_look_is_disclosed_once(retrieval: dict[str, Any]) -> None:
    """Several roles, one sentence -- not one apology per role."""

    ctx = _context("build me a work outfit")

    payload = _evidence(
        search_catalog(
            ctx,
            [_role("top", ["blouses"]), _role("bottom", ["skirts"])],
        )
    )

    assert payload["assumed_audience"] == ["womens"]
    assert _customer_safe_search_evidence(payload).count("ASSUMED_AUDIENCE") == 1


def test_a_catalog_without_the_field_discloses_nothing() -> None:
    """The field is configured, so a catalog that lacks it must stay silent.

    Not every catalog tags an audience. Inventing the sentence from an absent
    field would have the assistant assume out loud on no evidence at all.
    """

    products = [{"attributes": {"primary_color": "navy"}}]

    assert _assumed_audience(AUDIENCE_FIELD, {}, products) == []
    assert _assumed_audience("", {}, products) == []


def test_the_disclosure_is_an_assumption_about_the_shopper() -> None:
    """The wording took four live attempts; hold the one that worked.

    Aimed at the range it produced an inventory note. Aimed at the result set:
    "I'm currently seeing women's dress options" -- the limit of one search.
    Aimed at the shop: "this is a workwear-friendly shop", which named the
    style and dropped the audience entirely. Only the assumption about the
    shopper reads as something they can correct.
    """

    line = _assumed_audience_line({"assumed_audience": ["womens"]})

    assert "assuming you're looking for" in line
    assert "invite them to correct it" in line
    assert "not a note about the shop's style" in line
    assert "about who the shopper is" in line
    # Live, one clause after the shopper said "I need something to wear", the
    # reply asked "(I can adjust if you're shopping for someone else)".
    assert "already said it is for them" in line
    # The shopper stands in a shop, not in front of a query planner. A live
    # reply read "I tried the closest available search in apparel with the
    # filter adult all-genders, and that search returned zero results".
    assert "never a catalog label" in line
    # And says nothing about which pieces suit a wider audience. That clause
    # produced "I also have a few bags that anyone can wear (the crossbody
    # styles)" as the second sentence of a wedding-outfit reply -- all 38
    # adult_all_genders products are bags and sunglasses, and sunglasses are
    # tagged both ways, so it can only ever report the catalog's own tagging.
    assert "Say nothing about which pieces suit a wider audience" in line
    assert "pieces anyone can wear" not in line


def test_a_conversation_already_told_is_not_told_again(
    retrieval: dict[str, Any],
) -> None:
    """Three live replies in one conversation all opened with the assumption.

    "I need a work outfit", then "something in navy instead", then "what shoes
    would go with that" -- every one of them began "Assuming you're looking for
    women's ...". The trigger is true on nearly every turn, so a disclosure
    that does not remember having been made turns into a verbal tic.
    """

    ctx = _context("show me something in navy instead", already_disclosed=["womens"])

    payload = _evidence(search_catalog(ctx, [_role("blouses", ["blouses"])]))

    assert payload["assumed_audience"] == []
    assert "ASSUMED_AUDIENCE" not in _customer_safe_search_evidence(payload)


def test_the_turn_that_discloses_records_it_for_the_next_one(
    retrieval: dict[str, Any],
) -> None:
    """Suppression is only honest if the first turn writes down what it said."""

    ctx = _context("a work casual outfit")

    search_catalog(ctx, [_role("work casual outfit", ["blouses", "skirts"])])

    assert ctx.state.disclosed_audience == ["womens"]
    events = _audience_assumption_events(
        ctx.state, SimpleNamespace(request_id="req-1")
    )
    assert [event.event_type for event in events] == [
        "audience_assumption_disclosed"
    ]
    assert events[0].payload == {"audience": ["womens"]}


def test_a_turn_that_disclosed_nothing_records_nothing(
    retrieval: dict[str, Any],
) -> None:
    """Silence must leave an earlier disclosure standing, not overwrite it.

    Recording an empty audience every quiet turn would clear the memory and
    bring the sentence straight back on the turn after.
    """

    ctx = _context("show me navy", already_disclosed=["womens"])

    search_catalog(ctx, [_role("blouses", ["blouses"])])

    assert ctx.state.disclosed_audience == []
    assert (
        _audience_assumption_events(
            ctx.state, SimpleNamespace(request_id="req-2")
        )
        == []
    )


def test_a_declaration_outranks_an_assumption_from_the_same_turn(
    retrieval: dict[str, Any],
) -> None:
    """The self-correcting turn must not record the guess it overturned.

    Unscoped search returns womenswear, the evidence says so, the model
    recognises a husband was named and searches again with the values that
    suit him. Recording both would leave the conversation carrying an
    assumption the turn had already corrected.
    """

    state = SimpleNamespace(
        disclosed_audience=["womens"],
        agent_diagnostics={
            "product_evidence": [
                {
                    "search_scope": {
                        "confirmed_filters": {
                            AUDIENCE_FIELD: ["adult_all_genders"]
                        }
                    }
                }
            ]
        },
    )

    events = _turn_audience_events(
        state, SimpleNamespace(request_id="req-3"), field_name=AUDIENCE_FIELD
    )

    assert [event.event_type for event in events] == ["wearer_audience_declared"]
    assert events[0].payload == {"audience": ["adult_all_genders"]}


def test_a_turn_that_only_assumed_records_the_assumption() -> None:
    """The other half: no declaration means the assumption is what stands."""

    state = SimpleNamespace(
        disclosed_audience=["womens"], agent_diagnostics={"product_evidence": []}
    )

    events = _turn_audience_events(
        state, SimpleNamespace(request_id="req-4"), field_name=AUDIENCE_FIELD
    )

    assert [event.event_type for event in events] == [
        "audience_assumption_disclosed"
    ]


def test_the_audience_filter_carries_the_rule_the_model_needs() -> None:
    """The generic description was the whole bug.

    Every advertised filter got "Advertised hard filter '<name>'.", so the
    model had nothing to read but the enum value names. "sunglasses for men"
    and "my husband" landed on the right value 3/3; "shades for hubby" sent no
    filter at all and returned women's sunglasses. The rule below already
    existed in the system prompt, which is the channel that gets ignored.
    """

    model = _required_constraints_input_model(
        _capabilities(),
        wearer_audience_field=AUDIENCE_FIELD,
    )
    described = model.model_fields[AUDIENCE_FIELD].description or ""

    # The rule, which is what makes it portable to a catalog with more values.
    # It is built in two ordered steps rather than as a judgment about which
    # values "suit" the person. Measured on "my husband is coming too, he needs
    # sunglasses": the suits-them phrasing scored 0 of 10, and asking the model
    # to weigh inclusion against exclusion only moved the failure between the
    # man and the woman. The covers-everyone value being unconditional, and the
    # gendered value being opt-in, is what stopped it oscillating.
    # Two ordered steps with nothing to weigh, and the meaning of each value
    # supplied by the catalog rather than enumerated here. Measured across six
    # wordings on the running service: asking which values "suit" the person
    # scored a man 0/10, and every variant that fixed one gender broke the
    # other. This one scored men 14/14 and adult women 8/10.
    assert "covering all genders is always in the list" in described
    assert "only when its published meaning fits the person named" in described
    assert "closest value available is not the test" in described
    # The vocabulary, which is what "for men" had and "hubby" did not.
    # Measured: cutting the female terms out of this list dropped "sister" to
    # 1/4 because the filter stopped firing at all.
    for word in ("hubby", "my wife", "my sister", "my daughter", "my dad"):
        assert word in described, word
    # The vocabulary, which is what "for men" had and "hubby" did not.
    assert "hubby" in described
    # The clause protecting the case a required field broke: answering
    # "nobody named" with a covers-everyone value returned no clothing and no
    # shoes at all, only bags.
    assert "When nobody is named, omit this filter entirely" in described
    # A child in an adult catalog gets nothing, not adult substitutes.
    assert "never substitute what does not suit them" in described
    # The decision point owns the rule about carried audiences; the SHOPPING
    # FOR block only reports the value. Stated as a positive trigger, because
    # enumerating the ways a shopper moves on cannot be completed.
    assert "Only this turn's words count" in described
    assert "naming someone is what turns the filter on" in described
    # The person-words are for parsing, not a menu. A live reply offered
    # "if you meant men's or kids' pieces instead" in a catalog that
    # stocks neither -- vocabulary added for recognition leaking into
    # what the assistant proposes.
    assert "They are not audiences to offer back" in described
    assert "may name only audiences this catalog advertises" in described


def test_every_other_filter_keeps_the_generic_description() -> None:
    """Only the configured audience field is special-cased."""

    model = _required_constraints_input_model(
        _capabilities(),
        wearer_audience_field=AUDIENCE_FIELD,
    )

    assert model.model_fields["pattern"].description == (
        "Advertised hard filter 'pattern'."
    )


def test_a_catalog_with_no_configured_audience_field_is_untouched() -> None:
    """A deployment that never set the field name sees no change at all."""

    model = _required_constraints_input_model(_capabilities())

    assert model.model_fields[AUDIENCE_FIELD].description == (
        f"Advertised hard filter '{AUDIENCE_FIELD}'."
    )


def test_the_dropped_event_cap_matches_what_a_finalize_can_carry() -> None:
    """A receipt must never be rejected by the field that exists to prevent it.

    The cap was written as 32 while the memory service accepts up to 128 events
    and can therefore report up to 128 dropped types. Exceeding a pydantic
    max_length raises rather than truncates, chain-server reads that as a
    failed finalize, and the turn stays `started` -- the exact dead-conversation
    bug the dropped-event mechanism was added to remove, reachable through it.
    """

    from chain_server.src.conversation_memory import (
        MAX_FINALIZE_EVENTS,
        TurnFinalizeResult,
    )

    # Read the other side's bound rather than restating it. A test that used
    # MAX_FINALIZE_EVENTS for both the cap and the sample passes for any value
    # and proves nothing; this one fails if either service moves.
    from memory_retriever.src.conversations import TurnFinalizeRequest

    server_bound = next(
        rule.max_length
        for rule in TurnFinalizeRequest.model_fields["events"].metadata
        if getattr(rule, "max_length", None) is not None
    )
    assert MAX_FINALIZE_EVENTS == server_bound

    receipt = TurnFinalizeResult.model_validate(
        {
            "turn_id": "t1",
            "attempt_id": "a1",
            "sequence": 1,
            "replayed": False,
            "status": "completed",
            "assistant_text": "ok",
            "termination_reason": None,
            "dropped_event_types": [
                f"unknown_{index}" for index in range(MAX_FINALIZE_EVENTS)
            ],
        }
    )

    assert len(receipt.dropped_event_types) == MAX_FINALIZE_EVENTS
