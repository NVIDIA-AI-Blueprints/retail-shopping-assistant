# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An ordinal add is checked against what was shown, not against a line count.

"Add the first one" is the only ordinal add in the suite, and all it asserted
was that the cart held one line. A cart line is not a resolution: the wrong
sandal passes, and so does a sandal from three turns earlier.

Nothing in the suite asserted what a reference resolved to. That is how a
broken lookup of the conversation record -- a bag returned by the record and
refused on arrival -- reached a journey run as an empty cart and nothing more
to go on.

The product cannot be named in the script, because which sandal ranks first
varies by run. It does not need to be: the turn that showed them recorded them
in the order the shopper saw, so the run holds the right answer and the check
reads it from there.
"""

from __future__ import annotations

from tests.evaluation.src.replay import TurnResult, check_turn


def _showing(*names: str) -> TurnResult:
    """One group, numbered as the turn streams it."""

    return _grouped(("", names))


def _grouped(*groups: tuple[str, tuple[str, ...]]) -> TurnResult:
    """A showing of several headed groups, each numbered from one.

    The number restarts under each heading, so the same position exists in
    every group and the group is half of the coordinate.
    """

    products = [
        {"display_name": name, "group": heading, "position": position}
        for heading, names in groups
        for position, name in enumerate(names, start=1)
    ]
    return TurnResult(
        index=5,
        said="ok, just show me sandals in a 7",
        reply="here are some sandals",
        products=products,
        cart=[],
        tools=["search_catalog_tool"],
        seconds=9.0,
    )


def _add(*cart_names: str) -> TurnResult:
    return TurnResult(
        index=6,
        said="add the first one",
        reply="added",
        products=[],
        cart=[{"item": name, "amount": 1} for name in cart_names],
        tools=["add_cart_items_tool"],
        seconds=9.0,
    )


def test_the_product_shown_first_is_the_one_the_cart_must_hold() -> None:
    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "position": 1}},
        _add("Strappy Tan Sandals"),
        [],
        [_showing("Strappy Tan Sandals", "Silver Satin Sandals")],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "pass")]


def test_a_different_product_in_the_cart_fails() -> None:
    """The failure a line count cannot see, and the reason this check exists."""

    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "position": 1}},
        _add("Silver Satin Sandals"),
        [],
        [_showing("Strappy Tan Sandals", "Silver Satin Sandals")],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "fail")]
    assert "position 1 was Strappy Tan Sandals" in checks[0].detail


def test_an_empty_cart_fails_rather_than_passing_vacuously() -> None:
    """The J02 failure, stated as a check: the add did not happen.

    Worth its own case because the empty answer is the one a check written
    loosely lets through -- nothing to compare, so nothing disagrees.
    """

    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "position": 1}},
        _add(),
        [],
        [_showing("Strappy Tan Sandals")],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "fail")]


def test_a_position_that_was_never_shown_fails() -> None:
    """A script pointing at a position the turn did not fill asserts nothing.

    Turn 5 showed one sandal; asking about its third is the scenario being
    wrong, and silently passing would leave the author believing a reference
    was checked.
    """

    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "position": 3}},
        _add("Strappy Tan Sandals"),
        [],
        [_showing("Strappy Tan Sandals")],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "fail")]
    assert "nothing shown there" in checks[0].detail


def test_the_group_picks_which_second_the_script_meant() -> None:
    """Two groups on screen, so a number alone names one product in each."""

    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "group": "shoes", "position": 2}},
        _add("Wine Red Pumps"),
        [],
        [
            _grouped(
                ("dresses", ("Coral Silk Maxi", "Vivienne Lace")),
                ("shoes", ("Buckled Heels", "Wine Red Pumps")),
            )
        ],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "pass")]


def test_the_same_number_in_another_group_fails() -> None:
    """The second dress is not the second shoes, and the check must say so."""

    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "group": "shoes", "position": 2}},
        _add("Vivienne Lace"),
        [],
        [
            _grouped(
                ("dresses", ("Coral Silk Maxi", "Vivienne Lace")),
                ("shoes", ("Buckled Heels", "Wine Red Pumps")),
            )
        ],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "fail")]
    assert "shoes position 2 was Wine Red Pumps" in checks[0].detail


def test_a_number_with_no_group_takes_the_first_group() -> None:
    """Which is what the resolver does with a bare ordinal, so the two agree."""

    checks = check_turn(
        {"cart_holds_shown": {"turn": 5, "position": 1}},
        _add("Coral Silk Maxi"),
        [],
        [
            _grouped(
                ("dresses", ("Coral Silk Maxi", "Vivienne Lace")),
                ("shoes", ("Buckled Heels", "Wine Red Pumps")),
            )
        ],
    )

    assert [(c.name, c.outcome) for c in checks] == [("cart_holds_shown", "pass")]
