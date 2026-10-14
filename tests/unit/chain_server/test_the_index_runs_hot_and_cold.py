# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The historical index numbers the newest showings and names the rest.

Numbering every showing put one product at four different positions at once
by turn 19 of J01, with a single line of counting rules to arbitrate. Only the
showings a shopper could still be counting from keep their numbers; everything
older keeps its name and loses its position, which is how a long-range
reference is made anyway -- "add the black one in a 2" describes a dress
rather than counting to it.
"""

from __future__ import annotations

import re
from typing import Any

from chain_server.src.conversation_products import format_historical_product_index


def _showing(turn: int, names: list[str], kind: str = "dresses") -> dict[str, Any]:
    return {
        "candidate_set_id": f"set-turn-{turn}",
        "turn_seq": turn,
        "products": [
            {
                "ref": f"ref-{name}",
                "name": name,
                "position": position,
                "category": kind,
                "group": kind,
            }
            for position, name in enumerate(names, start=1)
        ],
    }


def _roster_lines(rendered: str) -> list[str]:
    roster = rendered.partition("ALSO SHOWN EARLIER")[2]
    return [
        line.strip()
        for line in roster.splitlines()
        if line.startswith("  ") and "more shown earlier" not in line
    ]


def _kinds_listed(rendered: str) -> list[str]:
    """The roster's kind headings, in the order they are rendered."""

    return [line.split(":")[0] for line in _roster_lines(rendered)]


def _products_listed(rendered: str) -> int:
    """How many products the roster actually names."""

    return sum(len(line.split(": ")[1].split(", ")) for line in _roster_lines(rendered))


def test_the_newest_showings_keep_their_numbers() -> None:
    rendered = format_historical_product_index(
        [_showing(1, ["Ankle Boots"], "boots"), _showing(2, ["Coral Maxi"])]
    )

    assert "1:Coral Maxi" in rendered
    assert "1:Ankle Boots" in rendered
    assert "ALSO SHOWN EARLIER" not in rendered


def test_older_showings_keep_their_names_and_lose_their_numbers() -> None:
    older = [_showing(turn, [f"Sweater {turn}"], "sweaters") for turn in range(1, 6)]
    newest = _showing(6, [f"Dress {index}" for index in range(24)])

    rendered = format_historical_product_index([*older, newest])

    assert "1:Dress 0" in rendered
    assert "sweaters: " in rendered
    # The whole point of the split: no second run of ordinals for the model to
    # count from, and no ref to carry through every model call of the turn.
    roster = rendered.partition("ALSO SHOWN EARLIER")[2]
    assert "Sweater 3" in roster
    assert "1:Sweater" not in roster
    assert "ref-Sweater" not in roster


def test_the_widest_single_showing_is_never_split() -> None:
    """Whatever it costs. The turn after it has to answer "the second one"."""

    rendered = format_historical_product_index(
        [_showing(1, [f"Product {index}" for index in range(40)])]
    )

    assert "1:Product 0" in rendered
    assert "40:Product 39" in rendered
    assert "ALSO SHOWN EARLIER" not in rendered


def test_a_product_numbered_above_is_not_repeated_in_the_roster() -> None:
    rendered = format_historical_product_index(
        [
            _showing(1, ["Vivienne Lace Dress"]),
            _showing(2, ["Vivienne Lace Dress", *[f"Filler {i}" for i in range(24)]]),
        ]
    )

    assert "1:Vivienne Lace Dress" in rendered
    assert "ALSO SHOWN EARLIER" not in rendered


def test_the_roster_runs_newest_first_like_the_heading_promises() -> None:
    older = [_showing(turn, [f"Item {turn}"], f"kind{turn:02d}") for turn in range(1, 6)]
    newest = _showing(6, [f"Dress {index}" for index in range(24)])

    rendered = format_historical_product_index([*older, newest])

    assert _kinds_listed(rendered) == ["kind05", "kind04", "kind03", "kind02", "kind01"]


def test_a_roster_too_long_to_fit_is_cut_from_the_oldest_end() -> None:
    """And says how many it cut.

    The first version of this dropped the roster whole when it would not fit,
    which lost every earlier product on exactly the long conversations the
    roster exists for.
    """

    older = [_showing(turn, [f"Item {turn}"], f"kind{turn:02d}") for turn in range(1, 11)]
    newest = _showing(11, [f"Dress {index}" for index in range(24)])

    rendered = format_historical_product_index([*older, newest], max_chars=1_300)

    kinds = _kinds_listed(rendered)
    assert kinds == sorted(kinds, reverse=True)
    assert "kind10" in kinds
    assert "kind01" not in kinds
    counted = re.search(r"\(\+(\d+) more shown earlier\)", rendered)
    assert counted is not None
    assert _products_listed(rendered) + int(counted.group(1)) == 10


def test_nothing_is_ever_dropped_without_saying_so() -> None:
    """At every budget, down to one too small to name a single product."""

    older = [_showing(turn, [f"Item {turn}"], f"kind{turn:02d}") for turn in range(1, 11)]
    newest = _showing(11, [f"Dress {index}" for index in range(24)])
    reference_sets = [*older, newest]
    total = 10 + 24

    for max_chars in range(256, 1_400):
        rendered = format_historical_product_index(reference_sets, max_chars=max_chars)
        assert len(rendered) <= max_chars, max_chars
        if not rendered:
            continue
        numbered = rendered.count(":Dress ")
        named = _products_listed(rendered)
        counted = re.search(r"\(\+(\d+) more shown earlier\)", rendered)
        if counted is not None:
            assert numbered + named + int(counted.group(1)) == total, max_chars
        elif numbered + named < total:
            # Too tight to name them or even count them, so the bare marker
            # carries the one fact that is left: there was more than this.
            # Silence is allowed only where the marker itself would not fit.
            marker = "(earlier historical products omitted)"
            assert (
                marker in rendered or max_chars - len(rendered) <= len(marker)
            ), max_chars
        else:
            assert numbered + named == total, max_chars


def test_an_index_with_no_room_for_a_product_says_nothing_at_all() -> None:
    """Rather than a heading announcing an index that names nothing."""

    rendered = format_historical_product_index(
        [_showing(1, ["A Very Long Product Name " + "x" * 300])], max_chars=256
    )

    assert rendered == ""
