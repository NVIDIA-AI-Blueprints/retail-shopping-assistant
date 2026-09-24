"""A list the model encoded as a string is read, not rejected.

The model sometimes sends a list argument JSON-encoded inside a string:
`{"items": "[{\"product_ref\": ...}]"}`. `items` has forgiven that since the
turn it emptied a cart. `skill_names` did not, and it cost a cart the same way:

    add_cart_items_tool          -> rejected, skill_tool_not_granted
    activate_shopper_skills_tool {"skill_names": "[\"cart-management\"]"}
                                 -> error

The refusal was right -- a cart tool needs cart-management -- and activating it
and retrying was the right recovery. The punctuation killed the recovery, no
retry came, and the reply told the shopper the dress was in their cart. Two of
three J01 runs ended on that.

`references` did not forgive it either, and it cost a turn a third way. "do you
have that first one in a size 6" resolved correctly -- right product, right
ordinal, right turn -- and arrived as 301 characters of JSON with one `]`
absent. The tool errored, the error said nothing actionable, and the identical
301 characters were sent twenty-two times until the graph's recursion limit
ended the turn with a generic apology.

Decoding forgives the punctuation and nothing else: the contents still go
through the same model, so an unregistered skill or a malformed item fails
exactly as before. A missing closing bracket is forgiven; a missing closing
brace is not, because completing a half-written object invents fields, and a
descriptor that lost its `ordinal` would resolve quietly to the wrong product.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from chain_server.src.conversation_products import (
    ResolveConversationProductsRequest,
)
from chain_server.src.tools.cart import AddCartItemsToolInput
from chain_server.src.turn_support import _skill_activation_input_model
from pydantic import ValidationError

# The builder reads `role` and `exclusive_group` off each registered skill, so
# a bare tuple of names no longer describes a registry.
_REGISTERED = {
    "outfit-styling": SimpleNamespace(
        role="primary", exclusive_group="product_procedure"
    ),
    "product-discovery": SimpleNamespace(
        role="primary", exclusive_group="product_procedure"
    ),
    "cart-management": SimpleNamespace(role="standalone", exclusive_group=None),
}


def _activation_model() -> type:
    return _skill_activation_input_model(_REGISTERED)


def test_skill_names_may_arrive_as_json_text() -> None:
    model = _activation_model()
    assert model(skill_names='["cart-management"]').skill_names == [
        "cart-management"
    ]
    assert model(
        skill_names='["outfit-styling", "cart-management"]'
    ).skill_names == ["outfit-styling", "cart-management"]


def test_skill_names_still_arrive_as_a_list() -> None:
    model = _activation_model()
    assert model(skill_names=["cart-management"]).skill_names == [
        "cart-management"
    ]


def test_decoding_forgives_the_punctuation_and_nothing_else() -> None:
    """The contents face the same model, so every other rule still holds."""

    model = _activation_model()
    # Not a registered skill.
    with pytest.raises(ValidationError):
        model(skill_names='["knitting-advice"]')
    # Two primary procedures, which the model validator forbids.
    with pytest.raises(ValidationError):
        model(skill_names='["outfit-styling", "product-discovery"]')
    # Empty, which min_length forbids.
    with pytest.raises(ValidationError):
        model(skill_names="[]")
    # Not JSON at all, and not a list once decoded.
    with pytest.raises(ValidationError):
        model(skill_names="cart-management")
    with pytest.raises(ValidationError):
        model(skill_names='{"skill": "cart-management"}')


def test_cart_items_may_arrive_as_json_text() -> None:
    """The field this helper was written for, which had no test of its own."""

    parsed = AddCartItemsToolInput(
        items='[{"product_ref": "generated:3185c59c1cab8b83", "quantity": 2}]'
    )
    assert [(item.product_ref, item.quantity) for item in parsed.items] == [
        ("generated:3185c59c1cab8b83", 2)
    ]
    assert AddCartItemsToolInput(
        items=[{"product_ref": "generated:abc"}]
    ).items[0].quantity == 1
    with pytest.raises(ValidationError):
        AddCartItemsToolInput(items='[{"quantity": 2}]')
    with pytest.raises(ValidationError):
        AddCartItemsToolInput(items="generated:abc")


def test_references_may_arrive_as_json_text() -> None:
    parsed = ResolveConversationProductsRequest(
        references='[{"reference_id": "r1", "display_name": "Qute Cashmere Sweater"}]'
    )
    assert [r.reference_id for r in parsed.references] == ["r1"]


#: The argument from J02 turn 2, byte for byte, including the absent `]`.
_UNCLOSED_FROM_J02 = (
    '[{"reference_id": "first_sweater", '
    '"product_ref": "generated:8a785791aa953d49", '
    '"display_name": "Polished Peplum Pullover Sweater", '
    '"category": "sweaters", "turn_sequence": 1, '
    '"candidate_set_id": "7b7fe9f8be534c3a9b5510a9aff0fe5e", "ordinal": 1, '
    '"attributes": {"primary_color": "cream", "sizes": "6"}}'
)


def test_a_list_missing_only_its_closing_bracket_is_read() -> None:
    """The turn that spent 22 identical calls and died on the recursion limit."""

    descriptor = ResolveConversationProductsRequest(
        references=_UNCLOSED_FROM_J02
    ).references[0]
    assert descriptor.reference_id == "first_sweater"
    assert descriptor.display_name == "Polished Peplum Pullover Sweater"
    assert descriptor.ordinal == 1
    assert descriptor.turn_sequence == 1


def test_a_half_written_object_is_not_completed() -> None:
    """A brace is not supplied, because supplying one invents a descriptor.

    Cut mid-field, this reference has a category and no ordinal. Closing it
    would resolve to whatever the category matches first -- a different product
    than the shopper meant, returned without any sign it was a repair.
    """

    with pytest.raises(ValidationError):
        ResolveConversationProductsRequest(
            references='[{"reference_id": "r1", "category": "sweaters", "ordin'
        )
