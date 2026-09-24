# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that read and change the shopper's cart."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from ..agenttypes import State
from ..cart_format import _format_cart, _format_cart_total
from ..cart_operations import (
    add_items_to_the_cart,
    remove_a_cart_line,
    update_a_cart_line,
)
from ..cart_references import AddCartItemsToolItemInput
from ..runtime.control_signals import normalize_tool_result
from ..runtime.identity import RequestIdentity
from ..runtime.turn_scope import TurnScope
from .schemas import _a_list_written_as_json_text

if TYPE_CHECKING:
    from ..runtime.runtime import DeepAgentsRuntime


class AddCartItemsToolInput(BaseModel):
    _accept_items_as_text = field_validator("items", mode="before")(
        _a_list_written_as_json_text
    )

    items: list[AddCartItemsToolItemInput] = Field(
        ...,
        min_length=1,
        description=(
            "One or more products to add. Each must use a PRODUCT_REF "
            "established by current-turn search or historical-product resolution."
        ),
    )


class _UpdateCartItemsInput(BaseModel):
    cart_line_id: str = Field(
        description="CART_LINE_ID from get_cart_tool. Not the product name."
    )
    quantity: int = Field(
        ge=0,
        description=(
            "Total quantity to end up with; with `size`, in the new size. "
            "Removing a line is remove_cart_item_tool, not quantity 0."
        ),
    )
    size: str | None = Field(
        default=None,
        description=(
            "The size the shopper now wants for this line. Omit for a "
            "quantity change. A size this product is not sold in is refused, "
            "naming the ones it is."
        ),
    )

    @field_validator("size", mode="before")
    @classmethod
    def _a_number_is_a_size_too(cls, value: Any) -> Any:
        """Take a size written as a number, so the change can be reached.

        This field is how "change it to a 7" is asked for, and a model that
        sent `size: 7` rather than `size: "7"` never got that far -- pydantic
        refused the call for the type, three times running, with a validation
        error that says nothing about carts. It then gave up, sent the quantity
        alone, and told the shopper it had updated a dress it had never been
        asked about.

        Sizes are "2" and "onesize" in this catalog, so a bare number is the
        obvious slip. Coercing it costs nothing and delivers the guidance the
        field was declared for.
        """

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return str(int(value) if float(value).is_integer() else value)
        return value


def build_cart_tools(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
):
    """Cart read, change and total tools for this shopper's cart."""

    from langchain_core.tools import tool

    @tool(return_direct=False)
    def get_cart_tool() -> str:
        """Read the current cart. Use before cart mutations to get
        CART_LINE_ID values, or when the shopper asks what is in their cart.
        Do NOT call again if the cart was already read this turn and no
        mutation has occurred since.
        """

        cart = runtime._read_cart(identity.cart_user_id)
        state.cart = cart
        runtime._append_product_images(
            scope.retrieved,
            cart,
            scope.product_evidence.values(),
        )
        return _format_cart(cart)

    @tool(
        args_schema=AddCartItemsToolInput,
        return_direct=False,
        response_format="content_and_artifact",
    )
    def add_cart_items_tool(items: list[AddCartItemsToolItemInput]):
        """Add products to the cart. Use ONLY on explicit shopper intent to
        add, buy, or put items in the cart. Requires PRODUCT_REF values from
        current-turn search or historical-product resolution — not names.
        Call once with every item the shopper asked to add, not once
        per item. "All items" means the ones they asked for, not
        everything in play this turn: "add the black one in a 2 and
        show me a clutch to go with it" adds the dress and shows the
        clutch. A product the shopper asked to see is not an item.
        """

        return normalize_tool_result(add_items_to_the_cart(runtime, state, identity, scope, items))

    @tool(return_direct=False, response_format="content_and_artifact")
    def remove_cart_item_tool(cart_line_id: str, quantity: int = 1):
        """Remove a cart line. Use ONLY on explicit shopper intent to remove
        an item. Requires CART_LINE_ID from get_cart_tool — do not guess.
        Use update_cart_items_tool to change quantity instead of removing
        and re-adding.
        """

        return normalize_tool_result(
            remove_a_cart_line(runtime, state, identity, scope, cart_line_id, quantity)
        )

    @tool(
        args_schema=_UpdateCartItemsInput,
        return_direct=False,
        response_format="content_and_artifact",
    )
    def update_cart_items_tool(
        cart_line_id: str,
        quantity: int,
        size: str | None = None,
    ):
        """Change one cart line: its quantity, its size, or both. One call
        moves the line, so never add a size and remove a line to change
        one. Moving replaces: the line stops holding the size it held. A
        size the shopper wants as well as that one is a second line, so
        that is add_cart_items_tool and not this.
        Requires CART_LINE_ID from get_cart_tool.
        """

        return normalize_tool_result(
            update_a_cart_line(runtime, state, identity, scope, cart_line_id, quantity, size)
        )

    @tool(return_direct=False)
    def view_cart_total_tool() -> str:
        """Compute the cart subtotal. Use for budget checks or when the
        shopper asks for the total. Does not include tax or shipping. Use
        get_cart_tool for line contents.
        """

        cart = runtime._read_cart(identity.cart_user_id)
        state.cart = cart
        runtime._append_product_images(
            scope.retrieved,
            cart,
            scope.product_evidence.values(),
        )
        return _format_cart_total(cart)

    return get_cart_tool, add_cart_items_tool, remove_cart_item_tool, update_cart_items_tool, view_cart_total_tool
