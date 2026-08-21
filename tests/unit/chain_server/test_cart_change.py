"""What a turn did to the cart is computed, not inferred.

The grounding editor was already told not to claim a cart action absent from
CURRENT CART. It still passed "I've added the Ombre Canvas Tote Bag back to your
cart" on a turn where the add failed and the cart did not change -- J07 turn 9.
A prohibition left the editor comparing two lists; these tests cover the fact
that replaces it.
"""

from chain_server.src.agenttypes import Cart
from chain_server.src.response_format import format_cart_change


def _cart(*lines):
    return Cart(contents=[dict(line) for line in lines])


HEELS_7 = {"item": "Jade Suede Heels", "size": "7", "amount": 1}
HEELS_8 = {"item": "Jade Suede Heels", "size": "8", "amount": 1}
TOTE = {"item": "Ombre Canvas Tote Bag", "amount": 1}


def test_an_unchanged_cart_says_so_in_words_a_draft_cannot_talk_around() -> None:
    result = format_cart_change(_cart(HEELS_7), _cart(HEELS_7))
    assert "NOTHING CHANGED" in result
    assert "Do not tell the shopper otherwise" in result


def test_an_add_is_named_with_its_size() -> None:
    result = format_cart_change(_cart(HEELS_7), _cart(HEELS_7, TOTE))
    assert "added Ombre Canvas Tote Bag x1" in result
    assert "Jade Suede Heels" not in result, "an untouched line is not a change"


def test_a_removal_is_named() -> None:
    result = format_cart_change(_cart(HEELS_7, TOTE), _cart(HEELS_7))
    assert "removed Ombre Canvas Tote Bag x1" in result


def test_a_size_change_is_two_facts_not_one() -> None:
    """J07 turn 6: the shopper asked for an 8 and the cart kept the 7.

    A size change is a different line, so it must read as a removal and an
    add -- never as a silent no-op.
    """

    result = format_cart_change(_cart(HEELS_7), _cart(HEELS_8))
    assert "removed Jade Suede Heels (size 7) x1" in result
    assert "added Jade Suede Heels (size 8) x1" in result


def test_a_failed_size_change_reads_as_nothing_changed() -> None:
    """The exact J07 turn 6 failure: update ran, cart kept size 7."""

    assert "NOTHING CHANGED" in format_cart_change(_cart(HEELS_7), _cart(HEELS_7))


def test_a_quantity_increase_is_reported_as_the_difference() -> None:
    before = _cart({**TOTE, "amount": 1})
    after = _cart({**TOTE, "amount": 3})
    assert "added Ombre Canvas Tote Bag x2" in format_cart_change(before, after)


def test_an_unknown_snapshot_does_not_claim_nothing_changed() -> None:
    """Absent evidence is not evidence of absence: a missing snapshot must not
    be reported as a cart that did not change."""

    assert format_cart_change(None, _cart(TOTE)) == "not known for this turn"
    assert "NOTHING CHANGED" not in format_cart_change(None, _cart(TOTE))


def test_the_update_schema_accepts_a_size_so_it_can_be_refused() -> None:
    """J07 turn 6: the model sent quantity 1 with size 8.

    `size` was absent from the tool's args_schema, so pydantic dropped it
    before the function ran, the quantity was set 1 -> 1, and the assistant
    reported a size change that never happened. The field has to exist on the
    schema for the refusal to be reachable at all -- declaring it on the
    function signature alone leaves the wrapper silently discarding it.
    """

    from chain_server.src.deepagents_runtime import _UpdateCartItemsInput

    parsed = _UpdateCartItemsInput(cart_line_id="abc", quantity=1, size="8")
    assert parsed.size == "8", "a size the model sends must survive validation"

    schema = _UpdateCartItemsInput.model_json_schema()
    assert "size" in schema["properties"], (
        "the model cannot be refused for a field its schema never offered"
    )
    assert "Do not use" in schema["properties"]["size"]["description"]
