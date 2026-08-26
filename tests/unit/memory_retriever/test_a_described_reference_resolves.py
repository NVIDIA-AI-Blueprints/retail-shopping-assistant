"""A reference the shopper described resolves from what they were shown.

"Add the black one in a 2" could be asked of the reference index and never
answered from it. The index compares a PRODUCT_REF and a whole product name,
and "black one" is neither, so the reference came back NOT FOUND -- and a
catalog lookup then ranked the catalog by how much each name resembled the
string "black one", offering boots, a camisole, pumps and a purse, none of them
ever shown. The black dress in a size 2 had been on screen nine turns earlier,
with its colour and its sizes recorded alongside the showing.

The order the shopper expects is: what was shown, most recent first; resolve
when one fits; ask when several do; search only when none do.
"""

from __future__ import annotations

from memory_retriever.src.product_references import (
    ProductReferenceDescriptor,
    ProductReferenceMatch,
    _matched_occurrences,
)


def _shown(
    name: str,
    ref: str,
    colour: str,
    sizes: list[str],
    turn: int,
    position: int,
    candidate_set: str,
) -> ProductReferenceMatch:
    return ProductReferenceMatch(
        product={
            "product_id": ref,
            "display_name": name,
            "category": "dresses",
            "attributes": {"primary_color": colour, "sizes": sizes},
        },
        candidate_set_id=candidate_set,
        turn_sequence=turn,
        position=position,
        catalog_revision="catalog-v1",
    )


# J01's own shape: one dress on turn 10, four more on turn 17, two of them
# black. Occurrences arrive oldest first, as the store returns them.
_OCCURRENCES = [
    _shown("Black Satin Lace-Up Dress", "d1", "black", ["2", "4", "6"], 10, 1, "set10"),
    _shown("Coral Silk Maxi Dress", "d2", "pink", ["2", "4"], 17, 1, "set17"),
    _shown("Vivienne Lace Dress", "d3", "black", ["2", "4"], 17, 2, "set17"),
]


def _resolved(**selectors: object) -> list[str]:
    descriptor = ProductReferenceDescriptor(reference_id="ref", **selectors)
    return [
        match.product["display_name"]
        for match in _matched_occurrences(descriptor, _OCCURRENCES)
    ]


def test_one_shown_product_fits_the_description() -> None:
    assert _resolved(attributes={"primary_color": "pink"}) == [
        "Coral Silk Maxi Dress"
    ]
    # A size matches by membership: the record holds every size the product
    # offers, and only one of these black dresses is cut to a 6.
    assert _resolved(attributes={"primary_color": "black", "sizes": "6"}) == [
        "Black Satin Lace-Up Dress"
    ]


def test_several_fit_so_both_come_back_to_be_asked_about() -> None:
    """Two black dresses in a 2 were shown, so this is a question, not a pick.

    The caller turns more than one match into the clarification it already
    knows how to write. What matters here is that neither is silently dropped:
    picking the newest would have added the Vivienne to a cart that asked for
    the other one.
    """

    assert _resolved(attributes={"primary_color": "black", "sizes": "2"}) == [
        "Black Satin Lace-Up Dress",
        "Vivienne Lace Dress",
    ]


def test_nothing_shown_fits_so_it_is_a_search() -> None:
    assert _resolved(attributes={"primary_color": "green"}) == []
    assert _resolved(attributes={"primary_color": "black", "sizes": "14"}) == []
    # An attribute the record does not carry cannot be claimed to match.
    assert _resolved(attributes={"heel_height": "low"}) == []


def test_an_ordinal_alone_counts_within_the_most_recent_showing() -> None:
    """"The second one" means the second of what is on screen now.

    Counted across every showing at once it matched a second one in each, and
    a reference that could not be clearer became a clarification.
    """

    assert _resolved(ordinal=2) == ["Vivienne Lace Dress"]
    assert _resolved(ordinal=1) == ["Coral Silk Maxi Dress"]
    # Beyond what that showing held, rather than reaching into an older one.
    assert _resolved(ordinal=3) == []


def test_an_ordinal_given_a_scope_still_obeys_it() -> None:
    assert _resolved(ordinal=1, turn_sequence=10) == [
        "Black Satin Lace-Up Dress"
    ]
    assert _resolved(ordinal=1, candidate_set_id="set17") == [
        "Coral Silk Maxi Dress"
    ]


def test_a_description_narrows_an_exact_reference_rather_than_widening_it() -> None:
    """Selectors still compose: every one supplied has to agree."""

    assert _resolved(
        product_ref="d1", attributes={"primary_color": "black"}
    ) == ["Black Satin Lace-Up Dress"]
    assert _resolved(product_ref="d1", attributes={"primary_color": "pink"}) == []
