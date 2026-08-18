# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A size is part of what the shopper chose, so it is part of the cart line.

Turn 14 of the fifteen-turn script asked "what size should I add?" and then
added nothing. The question was invented: no column, no field, nowhere for the
answer to go. It sounded like retail, which is what made it convincing.

The fix was not to stop asking -- a dress entering a cart with no size reads as
a toy -- but to make the question real. These hold the parts that quietly go
wrong once it is.
"""

from __future__ import annotations

import pathlib

from chain_server.src.turn_support import _normalize_cart_add_tool_items
from types import SimpleNamespace
from chain_server.src.turn_support import _cart_size_issue
import inspect


def test_two_sizes_of_one_product_are_two_lines() -> None:
    """Merging them would halve the order without saying so.

    A 6 and an 8 of one dress are two things a shopper owns, not one line of
    quantity two, so the normalizer keys on the size as well as the reference.
    """

    normalized = _normalize_cart_add_tool_items(
        [
            {"product_ref": "p1", "quantity": 1, "size": "6"},
            {"product_ref": "p1", "quantity": 1, "size": "8"},
        ]
    )

    assert set(normalized) == {("p1", "6"), ("p1", "8")}
    assert [entry["quantity"] for entry in normalized.values()] == [1, 1]


def test_the_same_size_twice_still_merges() -> None:
    """Two of the same size is a quantity, and must not become two lines."""

    normalized = _normalize_cart_add_tool_items(
        [
            {"product_ref": "p1", "quantity": 1, "size": "8"},
            {"product_ref": "p1", "quantity": 2, "size": "8"},
        ]
    )

    assert set(normalized) == {("p1", "8")}
    assert normalized[("p1", "8")]["quantity"] == 3


def test_one_size_goods_carry_no_size() -> None:
    """A handbag has no size to record, and blank is not a size."""

    normalized = _normalize_cart_add_tool_items(
        [
            {"product_ref": "bag", "quantity": 1},
            {"product_ref": "bag2", "quantity": 1, "size": "   "},
        ]
    )

    assert set(normalized) == {("bag", None), ("bag2", None)}
    assert all(entry["size"] is None for entry in normalized.values())


def test_the_size_reaches_the_cart_payload() -> None:
    """The adapter must send it, or the answer is discarded at the last step --
    which is exactly what happened before this existed."""

    from shared.commerce_contracts import AddCartItemInput

    request = AddCartItemInput(
        user_id="1",
        product_id="p1",
        quantity=1,
        idempotency_key="k",
        size="8",
    )

    assert request.size == "8"
    assert AddCartItemInput(
        user_id="1", product_id="p1", quantity=1, idempotency_key="k"
    ).size is None


def test_the_size_survives_reading_the_cart_back() -> None:
    """Recording it is not enough if the shopper never sees it.

    Live, a dress went in as a size 8 and the cart read back as "The Office
    A-line Dress — $179.99 (qty 1)" with no size anywhere: the column held it,
    and every layer above dropped it.
    """

    from chain_server.src.commerce_tools import _cart_line_from_memory_item

    line = _cart_line_from_memory_item(
        {
            "cart_line_id": "abc",
            "product_id": "p1",
            "item": "The Office A-line Dress",
            "amount": 1,
            "price": 179.99,
            "size": "8",
        }
    )

    assert line is not None
    assert line.size == "8"

    one_size = _cart_line_from_memory_item(
        {"cart_line_id": "b", "product_id": "p2", "item": "A bag", "amount": 1}
    )
    assert one_size is not None and one_size.size is None


def test_a_zero_result_search_is_told_to_relax_and_show() -> None:
    """"No green dress in a 2" offered a numbered menu and showed nothing.

    Two live runs answered with "would you prefer 1) a nearby size, 2) another
    colour, 3) green apparel?" -- three things it could have searched for,
    using none of the budget it had. A shopper asked to be shown dresses and
    got a form to fill in.
    """

    from chain_server.src.turn_support import _SEARCH_NO_MATCH_GROUNDING_NOTE

    note = _SEARCH_NO_MATCH_GROUNDING_NOTE

    assert "search again without it" in note
    assert "which\n    one you dropped" in note or "one you dropped" in note
    assert "saying plainly which" in note
    # A 4 is not an alternative to a 2. Live, one reply answered "no green
    # dress in a 2" by listing green dresses starting at a 4 -- garments the
    # shopper cannot wear, offered as though they helped.
    assert "A size is never the filter you give up" in note
    assert "a fact about a body" in note
    assert "cannot wear" in note
    # "Only green dresses in a 2, don't upsell" was answered 1 in 3 by
    # showing other things anyway.
    assert "relax nothing" in note
    assert "outranks your helpfulness" in note
    # The failure mode being replaced, named so it is not reintroduced.
    assert "numbered menu of things you could look for is not an answer" in note
    # And the opposite failure: relaxing silently would be a substitution.
    assert "never quietly drop a filter" in note


def test_a_relaxable_zero_result_is_not_told_to_stop_looking() -> None:
    """Two rules in one message, and the blunter one won.

    A zero-result search emitted both "search again with a filter relaxed" and
    SEARCH_SCOPE_COMPLETE's "Answer now. Do not search ... merely because
    search budget remains." Three live runs obeyed the second and answered
    with a menu of things they could have searched for.

    Scope-complete asserts the turn can be answered from what it has. With
    nothing returned and two filters to choose between, that is simply untrue.
    """

    from chain_server.src import catalog_search as mod

    source = pathlib.Path(mod.__file__).read_text()

    assert "relaxable = bool(evidence.confirmed_filters)" in source
    assert "if evidence.scope_complete and not relaxable:" in source


class TestCartSizeGate:
    """The cart tool decides on the size, rather than trusting the caller.

    Every catalog product states its sizes -- 136 a real range, 79 `onesize`,
    no gaps -- so this is a question the tool can answer from data. Left to
    prose, it held three times in four: asked to add a dress with six sizes,
    one run in four put it in the cart with no size at all.
    """

    def _product(self, sizes):
        return SimpleNamespace(
            display_name="The Office A-line Dress",
            attributes={"sizes": sizes} if sizes is not None else {},
        )

    def test_a_sized_product_cannot_enter_the_cart_without_one(self) -> None:
        issue = _cart_size_issue(self._product(["2", "4", "6"]), None)

        assert "SIZE REQUIRED" in issue
        assert "2, 4, 6" in issue
        assert "Nothing was added" in issue

    def test_a_size_the_product_is_not_sold_in_is_refused(self) -> None:
        issue = _cart_size_issue(self._product(["2", "4", "6"]), "14")

        assert "not sold" in issue
        assert "2, 4, 6" in issue

    def test_a_size_the_product_is_sold_in_passes(self) -> None:
        assert _cart_size_issue(self._product(["2", "4", "6"]), "4") == ""

    def test_size_matching_ignores_case(self) -> None:
        assert _cart_size_issue(self._product(["S", "M", "L"]), "m") == ""

    def test_a_onesize_product_needs_no_size(self) -> None:
        """79 accessories are `onesize`; asking which size would be nonsense."""

        assert _cart_size_issue(self._product(["onesize"]), None) == ""

    def test_a_catalog_that_states_no_sizes_does_not_block_the_cart(self) -> None:
        """Refusing here would block on missing data, not on a disagreement."""

        assert _cart_size_issue(self._product(None), None) == ""
        assert _cart_size_issue(self._product([]), None) == ""

    def test_sizes_arriving_as_a_string_are_still_read(self) -> None:
        """Search evidence renders them as "sizes: 2, 4, 6"."""

        assert _cart_size_issue(self._product("2, 4, 6"), "4") == ""
        assert "SIZE REQUIRED" in _cart_size_issue(self._product("2, 4, 6"), None)



class TestSizeProvenance:
    """A size is a want, not a fact: only the shopper says which one."""

    def _issue(self, **kw):
        from types import SimpleNamespace

        from chain_server.src.turn_support import _cart_size_provenance_issue

        product = SimpleNamespace(
            display_name="A Dress",
            attributes={"sizes": kw.get("sizes", ["2", "4", "6", "8", "10"])},
        )
        return _cart_size_provenance_issue(
            product,
            kw.get("size", "8"),
            kw.get("stated_as"),
            kw.get("shopper_text", ""),
            kw.get("cart_line_size"),
        )

    def test_a_refusal_names_the_product_and_the_sizes_it_sells(self) -> None:
        """Told only that a size was not established, the model guessed twice.

        A shopper answered "Sweater: M and Boots: 6" -- a format the assistant
        itself had offered. The model turned M into 6 and the gate refused it,
        correctly. But the refusal named no product and listed no sizes, so the
        model attributed the failure to the boots and told the shopper size 6
        was not sold for a boot the catalog sells in 5, 6, 7 and 8.

        The refusal has to carry what the next sentence needs.
        """

        issue = self._issue(size="6", stated_as="Sweater: M", shopper_text="Sweater: M and Boots: 6")

        assert issue, "an inferred size must still be refused"
        assert "A Dress" in issue, "the refusal must say which product"
        assert "2, 4, 6, 8, 10" in issue, "and which sizes it is sold in"

    def test_a_size_nobody_asked_for_is_refused(self) -> None:
        """The invented 6, recorded from a real conversation.

        "add the Office A-line Dress to my cart" -- no size anywhere -- and the
        reply was "now in your cart, 1 x size 6". Every check passed: 6 is a
        real size for that dress, in range and in stock. The shopper left with
        three dresses and $360 they never asked for.
        """

        issue = self._issue(size="6", shopper_text="add the Office A-line Dress")

        assert "SIZE NOT ESTABLISHED" in issue
        assert "Ask which of those they want" in issue

    def test_the_shoppers_own_words_are_enough(self) -> None:
        assert self._issue(
            size="8", stated_as="size 8", shopper_text="size 8"
        ) == ""

    def test_the_words_may_be_part_of_a_longer_message(self) -> None:
        assert self._issue(
            size="10",
            stated_as="in a 10 as well",
            shopper_text="actually add it in a 10 as well",
        ) == ""

    def test_a_quotation_the_shopper_never_said_is_refused(self) -> None:
        """The only way past this is to invent a quote, and it is checked."""

        assert "SIZE NOT ESTABLISHED" in self._issue(
            size="6", stated_as="size 6", shopper_text="add the dress"
        )

    def test_the_size_already_on_their_line_needs_no_quote(self) -> None:
        """Adding another of what they already have is not a new choice."""

        assert self._issue(
            size="8", shopper_text="add another one", cart_line_size="8"
        ) == ""

    def test_a_different_size_from_the_one_in_the_cart_still_needs_words(self) -> None:
        assert "SIZE NOT ESTABLISHED" in self._issue(
            size="10", shopper_text="add another one", cart_line_size="8"
        )

    def test_a_product_with_no_size_is_untouched(self) -> None:
        """Bags, jewellery and sunglasses are onesize and ask nothing."""

        assert self._issue(size=None, shopper_text="add the tote") == ""


class TestQuantityProvenance:
    """The sibling of the size: a number the shopper did not choose."""

    def _issue(self, quantity, stated_as=None, shopper_text=""):
        from chain_server.src.turn_support import (
            _cart_quantity_provenance_issue,
        )

        from types import SimpleNamespace

        return _cart_quantity_provenance_issue(
            quantity,
            stated_as,
            shopper_text,
            SimpleNamespace(display_name="A Dress"),
        )

    def test_one_is_what_add_it_means(self) -> None:
        """The ordinary add asks nothing and must stay free."""

        assert self._issue(1, shopper_text="add it to my cart") == ""

    def test_a_quantity_refusal_names_the_product(self) -> None:
        """Same reason as the size: a batch refusal it cannot attribute is a guess."""

        assert "A Dress" in self._issue(3, shopper_text="add it to my cart")

    def test_a_number_nobody_asked_for_is_refused(self) -> None:
        """Quantity had no guard at all -- the size at least had to be sold."""

        issue = self._issue(3, shopper_text="add the flats to my cart")

        assert "QUANTITY NOT ESTABLISHED" in issue
        assert "ask how many" in issue

    def test_the_shoppers_own_words_are_enough(self) -> None:
        assert self._issue(
            2, stated_as="two of them", shopper_text="add two of them please"
        ) == ""

    def test_a_quantity_the_shopper_never_said_is_refused(self) -> None:
        assert "QUANTITY NOT ESTABLISHED" in self._issue(
            5, stated_as="five of them", shopper_text="add the flats"
        )


class TestProductProvenance:
    """Which product a description means is a judgement; something must confirm it."""

    def _product(self, ref="d0", name="Belle Noir Satin Gown"):
        from types import SimpleNamespace

        return SimpleNamespace(product_id=ref, display_name=name)

    def _dresses(self, n=4):
        names = [
            "Belle Noir Satin Gown",
            "Black Satin Lace-Up Dress",
            "Vivienne Lace Dress",
            "Black Polka-Dotted Slip Dress",
        ]
        return [self._product(f"d{i}", names[i]) for i in range(n)]

    def _evidence(self, *products, system_identified=()):
        class _Evidence:
            def values(self_inner):
                return products

            def identified_by_the_system(self_inner, ref):
                return ref in system_identified

        return _Evidence()

    def _issue(self, **kw):
        from chain_server.src.turn_support import (
            _cart_product_provenance_issue,
        )

        return _cart_product_provenance_issue(
            kw.get("product") or self._product(),
            kw.get("shopper_text", ""),
            kw["evidence"],
            kw.get("recently_shown", ()),
        )

    def test_a_partial_name_is_still_a_naming(self) -> None:
        """Shoppers shorten names, and a run of the name cannot be said by chance.

        "the A line dress" carries 'a line' from The Office A-line Dress and
        from no other candidate. Requiring the whole name refused the commonest
        correct add; matching scattered words fitted a navy dress to "the black
        one in a 2", because 'the' and 'a' belong to that name and to ordinary
        sentences alike.
        """

        for words in (
            "add the A line dress in a 2",
            "A line dress please",
            "add the Office dress",
        ):
            assert self._issue(
                product=self._product("d4", "The Office A-line Dress"),
                shopper_text=words,
                evidence=self._evidence(
                    self._product("d4", "The Office A-line Dress"),
                    *self._dresses(4),
                ),
            ) == "", words

    def test_ordinary_words_of_a_name_are_not_a_naming(self) -> None:
        """'the' and 'a' belong to the name and to the sentence around it."""

        assert "PRODUCT NOT ESTABLISHED" in self._issue(
            product=self._product("d4", "The Office A-line Dress"),
            shopper_text="add the black one in a 2",
            evidence=self._evidence(
                self._product("d4", "The Office A-line Dress"),
                *self._dresses(4),
            ),
        )

    def test_a_shortened_or_misspelt_name_still_names(self) -> None:
        """Shoppers shorten names and mistype them, and still mean one product.

        Whole-name matching refused "add the Southwest Bracelet" because the
        catalog calls it "Southwest Bracelet"; a word-run rule then refused
        "the noir one" because 'noir' is four letters. Likeness with a margin
        takes both, and takes a typo with them.
        """

        for words in (
            "add the A line dress in a 2",
            "add the Office dress",
            "add the Ofice dress",
            "A line dress please",
        ):
            assert self._issue(
                product=self._product("d4", "The Office A-line Dress"),
                shopper_text=words,
                evidence=self._evidence(
                    self._product("d4", "The Office A-line Dress"),
                    *self._dresses(4),
                ),
            ) == "", words

    def test_the_last_products_shown_are_candidates_too(self) -> None:
        """The check must not vanish because the model already narrowed.

        Four black dresses were on screen. The model resolved one of them, so
        this turn's evidence held a single product, nothing was ambiguous
        against it, and the add went through silently -- the whole failure,
        surviving the gate built to stop it.
        """

        issue = self._issue(
            product=self._product("d1", "Black Satin Lace-Up Dress"),
            shopper_text="add the black one in a size 8",
            evidence=self._evidence(self._product("d1", "Black Satin Lace-Up Dress")),
            recently_shown=[
                {"ref": "d1", "name": "Black Satin Lace-Up Dress"},
                {"ref": "d3", "name": "Black Polka-Dotted Slip Dress"},
            ],
        )

        assert "PRODUCT NOT ESTABLISHED" in issue

    def test_short_words_of_a_name_do_not_name_it(self) -> None:
        """'the' and 'a' are part of the name and of every other sentence.

        Counting them, "add the black one in a 2" fitted The Office A-line
        Dress on 'the' and 'a' alone -- a navy dress, from fourteen turns
        earlier, for a request about a black one.
        """

        assert "PRODUCT NOT ESTABLISHED" in self._issue(
            product=self._product("d4", "The Office A-line Dress"),
            shopper_text="add the black one in a 2",
            evidence=self._evidence(
                self._product("d4", "The Office A-line Dress"),
                *self._dresses(4),
            ),
        )

    def test_two_near_equal_candidates_are_asked_about(self) -> None:
        """The margin, not the score, is what protects the shopper.

        "the black one" fits two black dresses about equally. A threshold would
        still have a best of the two and would add it -- which is exactly what
        happened, fourteen turns from where the shopper was looking.
        """

        assert "PRODUCT NOT ESTABLISHED" in self._issue(
            product=self._product("d1", "Black Satin Lace-Up Dress"),
            shopper_text="add the black one in a 2",
            evidence=self._evidence(*self._dresses(4)),
        )

    def test_a_shared_word_does_not_dilute_a_distinct_one(self) -> None:
        """"the black polka dotted one" says black twice over and polka once.

        Without discarding runs that two names share, 'black' would fit two
        dresses, the count would be two, and a request that names one product
        plainly would be answered with a question.
        """

        assert self._issue(
            product=self._product("d3", "Black Polka-Dotted Slip Dress"),
            shopper_text="add the black polka dotted one",
            evidence=self._evidence(*self._dresses(4)),
        ) == ""

    def test_a_run_shared_by_two_candidates_is_refused(self) -> None:
        """"the satin one" is a run of two names, so it names neither."""

        assert "PRODUCT NOT ESTABLISHED" in self._issue(
            product=self._product("d0", "Belle Noir Satin Gown"),
            shopper_text="add the satin one in a 2",
            evidence=self._evidence(*self._dresses(4)),
        )

    def test_a_description_that_fits_several_is_refused(self) -> None:
        """The silent pick, recorded from the demo script.

        Four black dresses had been shown. "add the black one in a 2" resolved
        to a navy dress from fourteen turns earlier, and every check passed:
        the ref was established, the name matched the ref, the size was sold
        and the shopper really had said "in a 2".
        """

        issue = self._issue(
            shopper_text="add the black one in a 2",
            evidence=self._evidence(*self._dresses()),
        )

        assert "PRODUCT NOT ESTABLISHED" in issue
        assert "Ask which one" in issue

    def test_the_shopper_naming_it_is_enough(self) -> None:
        assert self._issue(
            shopper_text="add the Belle Noir Satin Gown in a 2",
            evidence=self._evidence(*self._dresses()),
        ) == ""

    def test_a_name_they_never_said_is_not_a_naming(self) -> None:
        """Otherwise the quotation is the model's word for its own choice."""

        assert "PRODUCT NOT ESTABLISHED" in self._issue(
            shopper_text="add the black one",
            evidence=self._evidence(*self._dresses()),
        )

    def test_the_record_picking_it_is_enough(self) -> None:
        """"that first one" is turn 2 of the 13-turn script.

        The words name nothing, but the shopper gave a position and the record
        resolved it. The system picked the product, not the model.
        """

        assert self._issue(
            shopper_text="do you have that first one in a size 6",
            evidence=self._evidence(*self._dresses(), system_identified={"d0"}),
        ) == ""

    def test_one_candidate_needs_no_words(self) -> None:
        """"add it" after a single product is unambiguous whatever they said."""

        assert self._issue(
            shopper_text="add it",
            evidence=self._evidence(self._product()),
        ) == ""


class TestAtomicRefusalSaysWhatWasReady:
    """The add is all or nothing; the refusal must still carry what was settled."""

    def _result(self, ready=None):
        from chain_server.src.response_format import _format_cart_add_result
        from types import SimpleNamespace

        cart = SimpleNamespace(contents=[], lines=[], total=None)
        failed = [
            "- PRODUCT_REF 'x': SIZE NOT ESTABLISHED for 'A Sweater': the "
            "shopper did not ask for a size 6. It is sold in 2, 4, 6, 8."
        ]
        return _format_cart_add_result([], failed, cart, ready)

    def test_an_established_item_is_reported_though_nothing_was_written(self) -> None:
        """A shopper answered for both items and was asked for both again.

        "Sweater: M and Boots: 6" -- the boots size was correct and sold. The
        sweater's was not, so nothing was added, which is the intended
        atomicity. But the boots vanished from the result, so the model had no
        way to know they were settled and asked for that size a second time.
        """

        result = self._result(ready=["- Yantra Leather Ankle Boots, size 6, qty 1"])

        assert "Yantra Leather Ankle Boots" in result
        assert "Do not ask for these again" in result
        # And it must not claim they went in.
        assert "Added:" not in result

    def test_an_item_refused_as_out_of_scope_is_not_listed_as_settled(self) -> None:
        """An item can pass every per-item gate and still be refused after them.

        The scope check runs once, over everything resolved, and can block an
        item that already passed size, quantity and product provenance. Listing
        it as settled would tell the shopper not to ask again about the very
        thing that failed -- a contradiction inside one message.
        """

        from chain_server.src.turn_support import _cart_add_scope_failures
        from types import SimpleNamespace

        dress = SimpleNamespace(
            product_id="ref_dress", display_name="Office A-line Dress"
        )
        bag = SimpleNamespace(
            product_id="ref_bag", display_name="Navy Leather Everyday Bag"
        )
        failures = _cart_add_scope_failures(
            "add the Navy Leather Everyday Bag",
            [("ref_dress", dress), ("ref_bag", bag)],
            [dress, bag],
        )

        # The ref travels with the message, so the caller never has to read it
        # back out of the prose.
        assert [ref for ref, _message in failures] == ["ref_dress"]
        assert all(isinstance(message, str) for _ref, message in failures)

    def test_nothing_established_says_nothing_extra(self) -> None:
        """One blocked item on its own must not grow a section about nobody."""

        result = self._result(ready=[])

        assert "Established, not added" not in result
