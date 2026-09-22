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
    _resolve_descriptor,
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


def _resolution(**selectors: object) -> tuple[str, list[str], list[str]]:
    """The whole answer: what it decided, which products, what it set aside."""

    result = _resolve_descriptor(
        ProductReferenceDescriptor(reference_id="ref", **selectors), _OCCURRENCES
    )
    return (
        result.status,
        [match.product["display_name"] for match in result.matches],
        result.corroboration_mismatch,
    )


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


def test_a_description_nothing_fits_asks_about_what_is_on_screen() -> None:
    """Asked about the green one with nothing green shown, ask which.

    A description is the model's reading of the shopper, not the shopper's own
    words, so it is worth choosing between candidates with and not worth
    losing every candidate over. Answering not_found sent the turn off to
    search the catalog for something the shopper had not asked to be searched
    for, with the products they were looking at still on the screen.

    What was set aside is named, so the reply can say the record holds no
    green one rather than inventing agreement.
    """

    status, names, ignored = _resolution(attributes={"primary_color": "green"})
    assert status == "ambiguous"
    assert ignored == ["attributes.primary_color"]
    assert names == ["Coral Silk Maxi Dress", "Vivienne Lace Dress"]

    # An attribute the record does not carry cannot narrow anything either.
    status, _names, ignored = _resolution(attributes={"heel_height": "low"})
    assert status == "ambiguous"
    assert ignored == ["attributes.heel_height"]

    # One description can fit while another does not: black narrows to the two
    # black dresses, the invented size is set aside, and the newer one wins.
    status, names, ignored = _resolution(
        attributes={"primary_color": "black", "sizes": "14"}
    )
    assert (status, names) == ("resolved", ["Vivienne Lace Dress"])
    assert ignored == ["attributes.sizes"]


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


def test_a_description_cannot_cancel_the_ref_that_identified_the_product() -> None:
    """The shape that lost J02 turn 2, from the other side.

    A ref is an identifier this system minted and printed into the prompt; a
    colour or a size run is the model's account of the request. Composing them
    let the second cancel the first, so a descriptor that named the product
    correctly resolved nothing, and the turn searched the catalog by name for
    the product whose ref it was already holding.
    """

    assert _resolution(product_ref="d1", attributes={"primary_color": "black"}) == (
        "resolved",
        ["Black Satin Lace-Up Dress"],
        [],
    )
    assert _resolution(product_ref="d1", attributes={"primary_color": "pink"}) == (
        "resolved",
        ["Black Satin Lace-Up Dress"],
        ["attributes.primary_color"],
    )
