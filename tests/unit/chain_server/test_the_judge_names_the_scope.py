# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge names where a word lives; it does not grade the model's guess.

The reply is parsed here, not produced, so these are tests about the contract
between the two: a word maps to catalogue values, an absent word maps to an
empty list, and an invented value never becomes a filter.
"""

from __future__ import annotations

from chain_server.src.vocabulary_judge import (
    ScopeQuestion,
    VocabularyVerdict,
    _read,
)

_SUBCATEGORIES = [
    "blouses",
    "boots",
    "camisoles",
    "dresses",
    "flats",
    "heels",
    "sandals",
    "skirts",
    "sweaters",
    "tote_bags",
]
_COLOURS = ["beige", "black", "blue", "green", "navy", "white"]


def _verdict(reply: str, *words: str) -> VocabularyVerdict:
    return _read(
        reply,
        [ScopeQuestion(word) for word in words],
        _SUBCATEGORIES,
        _COLOURS,
    )


def test_an_umbrella_word_names_every_subcategory_it_covers() -> None:
    """"Shoes" is not a subcategory here, but four of them are shoes."""

    verdict = _verdict(
        '{"a": {"shoes": ["flats", "heels", "sandals", "boots"]}}', "shoes"
    )

    assert verdict.subcategories_for("shoes") == [
        "flats",
        "heels",
        "sandals",
        "boots",
    ]


def test_a_garment_this_shop_does_not_sell_names_nothing() -> None:
    """The verdict that mattered. Across the archive the model filed jeans
    under skirts 106 times; asked where jeans live, the answer is nowhere."""

    verdict = _verdict('{"a": {"jeans": []}}', "jeans")

    assert verdict.subcategories_for("jeans") == []


def test_an_empty_list_and_an_unasked_word_are_not_the_same() -> None:
    """Empty is "we do not sell it". None is "nobody asked, or nobody answered".

    Conflating them turns an endpoint timeout into a claim about the shop's
    stock, so the caller reads one as an answer for the shopper and the other
    as a reason to search what the model sent.
    """

    verdict = _verdict('{"a": {"jeans": []}}', "jeans")

    assert verdict.subcategories_for("jeans") == []
    assert verdict.subcategories_for("belts") is None


def test_a_value_this_catalogue_lacks_is_discarded() -> None:
    """A filter naming a value the catalogue does not hold matches nothing.

    Left in, one invented "denim" would return zero products -- and zero
    products is indistinguishable from "we do not sell that", so a
    hallucination would read as a fact about the shop.
    """

    verdict = _verdict('{"a": {"jeans": ["denim", "skirts"]}}', "jeans")

    assert verdict.subcategories_for("jeans") == ["skirts"]


def test_a_word_left_with_nothing_after_filtering_is_still_answered() -> None:
    """Every value invented, so there is nowhere real to search.

    Reported as the same empty verdict as a deliberate one, because searching
    for it is equally pointless either way.
    """

    verdict = _verdict('{"a": {"jeans": ["denim", "jeans"]}}', "jeans")

    assert verdict.subcategories_for("jeans") == []


def test_an_answer_to_a_word_nobody_asked_about_is_dropped() -> None:
    """The reply is the turn's, but a role may only read its own answer."""

    verdict = _verdict('{"a": {"shoes": ["heels"], "hats": ["blouses"]}}', "shoes")

    assert verdict.subcategories_for("shoes") == ["heels"]
    assert verdict.subcategories_for("hats") is None


def test_a_colour_word_widens_to_every_near_match() -> None:
    """"Cream" mapped to beige alone once, and an off-white sweater the shop
    does stock never reached the shopper."""

    verdict = _verdict('{"b": {"cream": ["beige", "white"]}}')

    assert verdict.colours_for("cream") == ["beige", "white"]


def test_a_reply_that_is_not_json_is_unavailable_not_empty() -> None:
    """Unreadable is unjudged. Read as empty, it would close every role."""

    verdict = _verdict("the model refused to answer", "jeans")

    assert verdict.unavailable is True
    assert verdict.subcategories_for("jeans") is None


def test_nothing_here_reads_the_word_itself() -> None:
    """No string matching stands behind the judge, deliberately.

    A degraded rule that stemmed the word against the subcategory list sat
    here briefly and was removed. It is the same idea as the eighteen-garment
    list it would have replaced: it knows "dresses" and not "pumps", so it
    quietly narrows umbrella roles during an outage and nobody finds out until
    a shopper asks for shoes and is shown nothing.

    When the judge cannot be reached the search runs on what the model sent,
    unchanged, and this asserts there is no lexical path left to fall into.
    """

    import chain_server.src.vocabulary_judge as judge_module

    assert not hasattr(judge_module, "by_name_only")
    assert not hasattr(judge_module, "exact_identity_only")
