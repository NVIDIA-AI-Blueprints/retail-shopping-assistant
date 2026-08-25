"""A malformed shape must not cost the shopper a turn.

Two turns were lost to JSON punctuation, not to anything the shopper asked:
"add the Xenial Aviator Sunglasses" sent a scope's fields with no `scopes` list
around them. The call was rejected whole and the assistant told the shopper it
could not complete a valid catalog search for a plainly named product.
"""

from chain_server.src.turn_support import _one_scope_is_a_list_of_one


def _wrapped(data):
    seen = {}

    def handler(value):
        seen["value"] = value
        return value

    _one_scope_is_a_list_of_one(None, data, handler)
    return seen["value"]


def test_a_lone_scope_is_wrapped_into_a_list() -> None:
    result = _wrapped(
        {
            "category": "eyewear",
            "subcategory": "sunglasses",
            "semantic_query": "Xenial Aviator Sunglasses",
            "requested_product_type": "Xenial Aviator Sunglasses",
        }
    )
    assert isinstance(result["scopes"], list) and len(result["scopes"]) == 1
    assert result["scopes"][0]["category"] == "eyewear"
    assert "scopes" not in result["scopes"][0]


def test_a_properly_wrapped_call_is_untouched() -> None:
    original = {"scopes": [{"semantic_query": "heels"}], "not_covered": None}
    assert _wrapped(original) is original


def test_not_covered_stays_outside_the_scope() -> None:
    """It belongs to the call, not to one role -- wrapping must not bury it."""

    result = _wrapped({"semantic_query": "pan", "not_covered": ["pan"]})
    assert result["not_covered"] == ["pan"]
    assert "not_covered" not in result["scopes"][0]


def test_a_call_that_is_neither_shape_is_left_alone() -> None:
    original = {"something_else": 1}
    assert _wrapped(original) is original


def test_an_items_list_encoded_as_a_string_is_read() -> None:
    """J12 t6: the add found the product, read its details, then sent the list
    JSON-encoded inside a string and the call was rejected whole. The cart
    stayed empty on a turn where everything else had gone right."""

    from chain_server.src.deepagents_runtime import AddCartItemsToolInput

    parsed = AddCartItemsToolInput.model_validate(
        {"items": '[{"product_ref": "generated:abc", "quantity": 1}]'}
    )
    assert [item.product_ref for item in parsed.items] == ["generated:abc"]


def test_a_proper_list_is_untouched() -> None:
    from chain_server.src.deepagents_runtime import AddCartItemsToolInput

    parsed = AddCartItemsToolInput.model_validate(
        {"items": [{"product_ref": "generated:abc", "quantity": 2}]}
    )
    assert parsed.items[0].quantity == 2


def test_decoding_forgives_the_punctuation_and_nothing_else() -> None:
    """A malformed item must still fail: only the wrapper is forgiven."""

    import pytest as _pytest
    from pydantic import ValidationError

    from chain_server.src.deepagents_runtime import AddCartItemsToolInput

    with _pytest.raises(ValidationError):
        AddCartItemsToolInput.model_validate({"items": "not json at all"})
    with _pytest.raises(ValidationError):
        # decodes cleanly, but the item has no PRODUCT_REF
        AddCartItemsToolInput.model_validate({"items": '[{"quantity": 1}]'})
