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

Decoding forgives the punctuation and nothing else: the contents still go
through the same model, so an unregistered skill or a malformed item fails
exactly as before.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chain_server.src.deepagents_runtime import AddCartItemsToolInput
from chain_server.src.turn_support import _skill_activation_input_model

_REGISTERED = ("outfit-styling", "product-discovery", "cart-management")


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
