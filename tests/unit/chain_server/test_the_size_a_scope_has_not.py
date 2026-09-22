"""The size a scope does not come in, and the sentence that owes the shopper.

J01 t11, "do you have a tote bag in a size 8". The size reached the query, no
bag is made in a number, and the turn answered "the tote bags we carry come in
sizes 2, 4, 6 and 10" -- a dress run, true of nothing in the bags category --
then asked whether to show the bags it was already holding.

Dropping the size is what lets the scope answer at all. Saying so is what stops
the run being invented, so the two are tested together.
"""

from chain_server.src.grounding_evidence import (
    _customer_safe_search_evidence,
    _size_the_scope_has_not_line,
)
from chain_server.src.tool_evidence import SearchEvidence


def test_the_note_names_one_size_rather_than_a_number() -> None:
    line = _size_the_scope_has_not_line(
        {"size_the_scope_has_not": {"asked": "8", "comes_in": ["onesize"]}}
    )

    assert "size 8" in line
    assert "one size" in line
    # The failure this replaces was a fabricated run, so the note must not read
    # as licence to name any other number.
    assert "state no others" in line


def test_the_note_names_the_run_a_scope_really_has() -> None:
    """Footwear asked for a 12: the honest answer names 5-9, not "one size"."""

    line = _size_the_scope_has_not_line(
        {
            "size_the_scope_has_not": {
                "asked": "12",
                "comes_in": ["5", "6", "7", "8", "9"],
            }
        }
    )

    assert "size 12" in line
    assert "5, 6, 7, 8, 9" in line
    assert "one size" not in line


def test_no_note_without_a_dropped_size() -> None:
    assert _size_the_scope_has_not_line({}) == ""
    assert _size_the_scope_has_not_line({"size_the_scope_has_not": {}}) == ""
    # A size with no run to name says nothing, rather than saying nothing useful.
    assert (
        _size_the_scope_has_not_line(
            {"size_the_scope_has_not": {"asked": "8", "comes_in": []}}
        )
        == ""
    )


def test_the_dropped_size_travels_with_the_products() -> None:
    """The plumbing, which is the half a rendering test cannot see.

    The note is only worth having if it reaches the turn holding the products:
    a shopper shown four totes and told nothing is how the invented run got
    written in the first place.
    """

    evidence = SearchEvidence(
        outcome="results",
        products=[{"name": "Chic Canvas Tote Bag"}],
        size_the_scope_has_not={"asked": "8", "comes_in": ["onesize"]},
    )

    payload = next(iter(evidence.as_artifact().values()))
    assert payload["size_the_scope_has_not"] == {
        "asked": "8",
        "comes_in": ["onesize"],
    }

    rendered = _customer_safe_search_evidence(payload)
    assert "SIZE_THE_SCOPE_HAS_NOT" in rendered
    assert "one size" in rendered
