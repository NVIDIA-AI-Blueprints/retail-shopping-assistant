"""An unbounded numeric filter asks for what no filter asks for.

"I have a wedding to go to and I need something to wear" sent every advertised
filter as null -- price as `{"min": null, "max": null}`. That is the model
saying it wants no price filter, written in the shape the schema handed it.

Refusing it cost the turn twice. The search was turned back, and the repair the
model reached for was to invent bounds spanning the whole catalog (39.90 to
269.99, which filters nothing) while dropping the subcategory to make room.
The changed scope then hit the repair lock, and turn one of J01 ended with "I
couldn't complete a valid catalog search for that request".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chain_server.src.turn_support import _required_constraints_input_model
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
    CatalogTaxonomyCapabilities,
)


def _model() -> type:
    return _required_constraints_input_model(
        CatalogCapabilities(
            filters={
                "price": CatalogFilterCapability(
                    type="number",
                    operators=["range"],
                    source_fields=["price"],
                    min_value=39.9,
                    max_value=269.99,
                ),
                "primary_color": CatalogFilterCapability(
                    type="enum",
                    operators=["in"],
                    source_fields=["primary_color"],
                    values=["black", "green"],
                ),
            },
            taxonomy=CatalogTaxonomyCapabilities(
                category_field="category",
                subcategory_field="subcategory",
                categories={},
            ),
        )
    )


def test_a_range_with_no_bounds_is_read_as_no_filter() -> None:
    model = _model()
    assert model(price={"min": None, "max": None}).price is None
    assert model(price=None).price is None
    # Alongside the other nulls the model sends in the same call.
    parsed = model(price={"min": None, "max": None}, primary_color=None)
    assert parsed.price is None
    assert parsed.primary_color is None


def test_a_bound_that_is_present_is_kept() -> None:
    model = _model()
    assert model(price={"min": None, "max": 60}).price.max == 60
    assert model(price={"min": 40, "max": None}).price.min == 40
    both = model(price={"min": 40, "max": 60}).price
    assert (both.min, both.max) == (40, 60)


def test_the_shape_of_a_filter_is_still_checked() -> None:
    """Emptying the range forgives nothing else about it."""

    model = _model()
    with pytest.raises(ValidationError):
        model(price={"min": 60, "max": 40})
    with pytest.raises(ValidationError):
        model(price={"minimum": None})
    with pytest.raises(ValidationError):
        model(primary_color="turquoise")
