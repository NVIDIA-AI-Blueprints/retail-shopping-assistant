# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that read and change the shopper's cart."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError, field_validator
from shared.commerce_contracts import (
    AddCartItemInput,
    GetProductDetailsInput,
    ProductSummary,
    RemoveCartItemInput,
    UpdateCartItemInput,
)

from ..agenttypes import Cart, State
from ..cart_format import (
    _format_cart,
    _format_cart_add_result,
    _format_cart_remove_result,
    _format_cart_total,
    _format_size_change_result,
    _format_update_cart_result,
)
from ..cart_references import (
    AddCartItemsToolItemInput,
    _cart_add_scope_failures,
    _cart_line_by_id,
    _cart_product_choice_note,
    _cart_resize_issue,
    _cart_size_issue,
    _identified_in_the_current_showing,
    _most_recently_shown,
    _normalize_cart_add_tool_items,
    _one_size_note,
    _shopper_words_this_conversation,
)
from ..commerce_tools import (
    add_cart_item,
    get_product_details,
    remove_cart_item,
    update_cart_item,
)
from ..conversation_products import ConversationProductsError, ProductReferenceDescriptor
from ..product_records import _product_detail_failure_message, _same_product_display_name
from ..runtime.control_signals import EFFECTS_KEY, normalize_tool_result
from ..runtime.identity import RequestIdentity
from ..runtime.turn_scope import TurnScope
from .schemas import _a_list_written_as_json_text

if TYPE_CHECKING:
    from ..runtime.runtime import DeepAgentsRuntime

logger = logging.getLogger(__name__)


def _refreshed_cart(
    runtime: DeepAgentsRuntime, state: State, identity: RequestIdentity, scope: TurnScope
) -> Cart:
    """Re-read the cart after a change, so the reply describes what it holds."""

    state.cart = runtime._read_cart(identity.cart_user_id)
    runtime._append_product_images(
        scope.retrieved, state.cart, scope.product_evidence.values()
    )
    return state.cart


def _effect(operation: str, idempotency_key: str, **fields: Any) -> dict[str, Any]:
    return {
        "operation": operation,
        "idempotency_key": idempotency_key,
        **{name: value for name, value in fields.items() if value is not None},
    }


def _result(rendered: str, committed: list[dict[str, Any]]):
    return (rendered, {EFFECTS_KEY: committed}) if committed else rendered


def _size_key(size: Any) -> str:
    value = str(size or "").strip().lower()
    return "" if value == "onesize" else value


def _lines_of(cart: Cart, product_id: str) -> list[dict[str, Any]]:
    return [
        line
        for line in cart.contents or []
        if str(line.get("product_id") or "") == product_id
    ]


def _already_held(product: ProductSummary, size: Any, line: dict[str, Any]) -> str:
    """An add of a line the cart already holds, answered without writing."""

    return (
        f"- {product.display_name}"
        + (f", size {size}" if size else "")
        + f": already in the cart (CART_LINE_ID {line.get('cart_line_id')}, "
        f"qty {line.get('amount')}), so it was not added again. If the "
        "shopper asked for more of it, set the new total with "
        "update_cart_items_tool on that line; otherwise tell them it is "
        "already in their cart."
    )


def _another_size_held(
    product: ProductSummary, size: Any, others: list[dict[str, Any]]
) -> str:
    """A second size added beside one the cart held, with the way back."""

    held = ", ".join(
        f"size {line.get('size')} (CART_LINE_ID {line.get('cart_line_id')})"
        for line in others
    )
    return (
        f"- {product.display_name}: the cart already held {held}, and now "
        f"also holds size {size}. If the shopper asked for size {size} as "
        "well, that is done. If they asked to change the size, remove the "
        "old line with remove_cart_item_tool now, so they do not pay for "
        "both."
    )


_LOOKUP_FAILED = (
    "this conversation's record could not be read, which is a fault on our "
    "side and not a missing product. Do not search for it and do not ask the "
    "shopper to identify it again. Tell them the cart could not be updated "
    "just now and that they can try again."
)
_NOT_ESTABLISHED = (
    "not established in this turn. If this product was shown earlier in the "
    "conversation, resolve it first and add the PRODUCT_REF that comes back -- "
    "evidence is per turn, so a reference from an earlier turn has to be "
    "resolved again. If it is a product name rather than a reference, or was "
    "never shown at all, search the catalog now and show the closest matches, "
    "then ask which to add. Never add a product the shopper has not been "
    "shown, and do not ask them for a catalogue name, a link, or a price."
)


def _product_for_ref(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    product_ref: str,
) -> tuple[ProductSummary | None, bool]:
    """This turn's evidence for a ref, else the conversation's durable index.

    The flag is True when the index could not be read, which is a fault on our
    side rather than a product nobody was shown.
    """

    product = scope.product_evidence.get(product_ref)
    if product is not None:
        return product, False
    descriptors = [ProductReferenceDescriptor(reference_id="cart-add", product_ref=product_ref)]
    try:
        result = runtime._conversation_products.resolve(identity.conversation_id, descriptors)
    except (ConversationProductsError, ValidationError) as exc:
        logger.error(
            "orchestrator | cart add | the conversation record could "
            "not be read for PRODUCT_REF %s: %s",
            product_ref,
            exc,
        )
        return None, True
    scope.product_evidence.add_resolutions(result.results, descriptors)
    state.system_identified_products = list(scope.product_evidence.system_identified())
    return scope.product_evidence.get(product_ref), False


def add_items_to_the_cart(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    items: list[AddCartItemsToolItemInput],
):
    """`add_cart_items_tool`: all or nothing, revalidated against the catalog."""

    try:
        requested_items = _normalize_cart_add_tool_items(items)
    except ValueError as exc:
        return f"Cart add failed: {exc}"
    if not requested_items:
        return "Cart add failed: provide at least one PRODUCT_REF to add."

    shopper_words = _shopper_words_this_conversation(state)
    resolved: list[tuple[str, ProductSummary, int, str | None]] = []
    failed: list[str] = []
    blocked: list[str] = []
    already_held: list[str] = []
    notes: list[str] = []
    cart_before = runtime._read_cart(identity.cart_user_id)
    for (product_ref, size), request in requested_items.items():
        product, lookup_failed = _product_for_ref(runtime, state, identity, scope, product_ref)
        if product is None:
            reason = _LOOKUP_FAILED if lookup_failed else _NOT_ESTABLISHED
            failed.append(f"- PRODUCT_REF '{product_ref}': {reason}")
            continue
        expected_name = request.get("expected_display_name") or ""
        if expected_name and not _same_product_display_name(expected_name, product.display_name):
            blocked.append(
                f"- PRODUCT_REF '{product_ref}': expected "
                f"'{expected_name}', but that ref resolves to "
                f"'{product.display_name}'. Use the matching PRODUCT_REF "
                "for the intended product before adding."
            )
            continue
        detail = get_product_details(
            GetProductDetailsInput(product_id=product.product_id),
            runtime.config.retriever_port,
            timeout_seconds=runtime.config.catalog_search_timeout_seconds,
        )
        if not detail.ok or detail.product is None:
            failed.append(
                f"- PRODUCT_REF '{product_ref}': "
                + _product_detail_failure_message(detail.error, cart_validation=True)
            )
            continue
        active = detail.product
        if not _same_product_display_name(active.display_name, product.display_name):
            blocked.append(
                f"- PRODUCT_REF '{product_ref}': That reference now "
                "resolves to a different product. Search again and use "
                "the new PRODUCT_REF before adding it."
            )
            continue
        # The catalog must sell the size and the shopper must have said it.
        size_issue = _cart_size_issue(active, size, shopper_words)
        if size_issue:
            blocked.append(f"- PRODUCT_REF '{product_ref}': {size_issue}")
            continue
        one_size_note = _one_size_note(active, size)
        if one_size_note:
            size = None
        same_line = next(
            (
                line
                for line in _lines_of(cart_before, active.product_id)
                if _size_key(line.get("size")) == _size_key(size)
            ),
            None,
        )
        if same_line is not None:
            already_held.append(_already_held(active, size, same_line))
            continue
        if one_size_note:
            notes.append(one_size_note)
        # Disclosed, not refused: a wrong line is one click away on screen.
        choice_note = _cart_product_choice_note(
            active,
            shopper_words,
            scope.product_evidence,
            _most_recently_shown(state),
            _identified_in_the_current_showing(state),
            size,
        )
        if choice_note:
            notes.append(f"- {active.display_name}: {choice_note}")
        resolved.append((product_ref, active, int(request["quantity"]), size))

    scope_failures = _cart_add_scope_failures(
        state.query,
        [(product_ref, product) for product_ref, product, _, _ in resolved],
        scope.product_evidence.values(),
    )
    blocked.extend(message for _ref, message in scope_failures)
    if blocked:
        out_of_scope = {ref for ref, _message in scope_failures}
        # Established items travel with the refusal so only the open question
        # is asked again. Out-of-scope items are not listed as settled.
        ready = [
            f"- {product.display_name}"
            + (f", size {size}" if size else "")
            + f", qty {quantity}"
            for ref, product, quantity, size in resolved
            if ref not in out_of_scope
        ]
        cart = _refreshed_cart(runtime, state, identity, scope)
        rendered = _format_cart_add_result([], failed + blocked, cart, ready)
        if already_held:
            rendered += "\n\n" + "\n".join(already_held)
        return rendered

    added: list[str] = []
    committed: list[dict[str, Any]] = []
    other_sizes: list[str] = []
    for product_ref, product, quantity, size in resolved:
        # Size is part of the key: a 6 and an 8 in one turn are two mutations.
        key = scope.cart_mutation_key(
            f"{identity.request_id}:add:{product.product_id}"
            f":{size or 'onesize'}:{quantity}"
        )
        result = add_cart_item(
            AddCartItemInput(
                user_id=str(identity.cart_user_id),
                product_id=product.product_id,
                display_name=product.display_name,
                quantity=quantity,
                size=size,
                unit_price=product.price,
                image_url=product.image_url,
                idempotency_key=key,
            ),
            runtime.config.memory_port,
        )
        if not result.ok:
            message = result.error.message if result.error else "Cart add failed."
            failed.append(f"- PRODUCT_REF '{product_ref}': {message}")
            continue
        committed.append(
            _effect("added to cart", key, product_id=product.display_name, quantity=quantity)
        )
        added.append(
            f"- {quantity} x {product.display_name}"
            + (f", size {size}" if size else "")
            + f" (PRODUCT_REF: {product.product_id})"
        )
        others = [
            line
            for line in _lines_of(cart_before, product.product_id)
            if _size_key(line.get("size")) != _size_key(size)
        ]
        if size and others:
            other_sizes.append(_another_size_held(product, size, others))

    rendered = _format_cart_add_result(
        added, failed, _refreshed_cart(runtime, state, identity, scope)
    )
    for extra in (notes, already_held, other_sizes):
        if extra:
            rendered += "\n\n" + "\n".join(extra)
    return _result(rendered, committed)


def remove_a_cart_line(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    cart_line_id: str,
    quantity: int = 1,
):
    """`remove_cart_item_tool`."""

    quantity = max(1, int(quantity or 1))
    line = _cart_line_by_id(cart_line_id, runtime._read_cart(identity.cart_user_id))
    if line is None:
        return f"No cart line with CART_LINE_ID '{cart_line_id}' could be found."
    key = scope.cart_mutation_key(
        f"{identity.request_id}:remove:{line['cart_line_id']}:{quantity}"
    )
    result = remove_cart_item(
        RemoveCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=line["cart_line_id"],
            product_id=line.get("product_id"),
            display_name=line["item"],
            quantity=quantity,
            idempotency_key=key,
        ),
        runtime.config.memory_port,
    )
    _refreshed_cart(runtime, state, identity, scope)
    rendered = _format_cart_remove_result(
        result, fallback=f"Removed {quantity} {line['item']} from cart."
    )
    if not result.ok:
        return rendered
    effect = _effect(
        "removed from cart",
        key,
        product_id=line["item"],
        cart_line_id=line["cart_line_id"],
        quantity=quantity,
    )
    return _result(rendered, [effect])


def change_a_line_size(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    cart_line_id: str,
    quantity: int,
    size: str,
):
    """Move a cart line to another size: add the new size, then remove the old.

    In that order, so a failure between the two leaves the shopper an extra
    line rather than nothing.
    """

    line = _cart_line_by_id(cart_line_id, state.cart)
    if line is None:
        return (
            "CART_UPDATE_REFUSED: no line with CART_LINE_ID "
            f"'{cart_line_id}' is in this cart. Call get_cart_tool and "
            "use a CART_LINE_ID it reports. Nothing was changed."
        )
    held = str(line.get("size") or "").strip()
    if held and size.casefold() == held.casefold():
        # The size it already has: a quantity change, not a second line.
        return update_a_cart_line(runtime, state, identity, scope, cart_line_id, quantity)
    if quantity == 0:
        return (
            "CART_UPDATE_REFUSED: quantity 0 and a new size contradict "
            "each other -- one deletes the line, the other replaces it. "
            "Send the quantity the shopper is keeping to change the "
            "size, or use remove_cart_item_tool to remove the line. "
            "Nothing was changed."
        )

    product_id = str(line.get("product_id") or "")
    detail = get_product_details(
        GetProductDetailsInput(product_id=product_id),
        runtime.config.retriever_port,
        timeout_seconds=runtime.config.catalog_search_timeout_seconds,
    )
    if not detail.ok or detail.product is None:
        return "CART_UPDATE_FAILED: " + _product_detail_failure_message(
            detail.error, cart_validation=True
        )
    # The same size check as an add, so a resize cannot seat a size an add refuses.
    size_issue = _cart_resize_issue(detail.product, size)
    if size_issue:
        return f"CART_UPDATE_REFUSED: {size_issue}"

    product = detail.product
    add_key = scope.cart_mutation_key(
        f"{identity.request_id}:resize:add:{product_id}:{size}:{quantity}"
    )
    add = add_cart_item(
        AddCartItemInput(
            user_id=str(identity.cart_user_id),
            product_id=product_id,
            display_name=product.display_name,
            quantity=quantity,
            size=size,
            unit_price=product.price,
            image_url=product.image_url,
            idempotency_key=add_key,
        ),
        runtime.config.memory_port,
    )
    if not add.ok:
        message = add.error.message if add.error else "Cart add failed."
        return (
            f"CART_UPDATE_FAILED: {message} The cart still holds "
            f"{product.display_name} in size {held or 'onesize'}, and "
            "nothing was removed."
        )

    remove_key = scope.cart_mutation_key(f"{identity.request_id}:resize:remove:{cart_line_id}")
    remove = remove_cart_item(
        RemoveCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=cart_line_id,
            quantity=int(line.get("amount") or 1),
            product_id=product_id,
            display_name=product.display_name,
            idempotency_key=remove_key,
        ),
        runtime.config.memory_port,
    )
    cart = _refreshed_cart(runtime, state, identity, scope)
    # Recorded as two mutations: on the unhappy path the cart holds both sizes.
    committed = [
        _effect("added to cart", add_key, product_id=product.display_name, quantity=quantity)
    ]
    if remove.ok:
        committed.append(
            _effect(
                "removed from cart",
                remove_key,
                cart_line_id=cart_line_id,
                product_id=product.display_name,
            )
        )
    rendered = _format_size_change_result(
        display_name=product.display_name,
        from_size=held,
        to_size=size,
        quantity=quantity,
        cart=cart,
        old_line_removed=remove.ok,
        old_line_id=cart_line_id,
    )
    return rendered, {EFFECTS_KEY: committed}


def update_a_cart_line(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    cart_line_id: str,
    quantity: int,
    size: str | None = None,
):
    """`update_cart_items_tool`: a size change, or a quantity change."""

    if size is not None and str(size).strip():
        return change_a_line_size(
            runtime, state, identity, scope, cart_line_id, quantity, str(size).strip()
        )
    if quantity == 0:
        # Deleting has its own tool, and a size change has its own route above.
        return (
            "CART_UPDATE_REFUSED: quantity 0 would delete this line, and "
            "this tool sets quantities. If the shopper wants the line "
            "gone, use remove_cart_item_tool. If they are changing a "
            "SIZE, send the new size in `size` with the quantity to "
            "keep, and this tool makes the change."
        )

    key = scope.cart_mutation_key(f"{identity.request_id}:update:{cart_line_id}:{quantity}")
    result = update_cart_item(
        UpdateCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=cart_line_id,
            quantity=quantity,
            idempotency_key=key,
        ),
        runtime.config.memory_port,
    )
    rendered = _format_update_cart_result(
        result, _refreshed_cart(runtime, state, identity, scope)
    )
    if not result.ok:
        return rendered
    effect = _effect(
        "cart quantity updated", key, cart_line_id=cart_line_id, quantity=quantity
    )
    return _result(rendered, [effect])


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
        """Take `size: 7` as "7"; refusing the type left the change unreachable."""

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

        return _format_cart(_refreshed_cart(runtime, state, identity, scope))

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

        return _format_cart_total(_refreshed_cart(runtime, state, identity, scope))

    return get_cart_tool, add_cart_items_tool, remove_cart_item_tool, update_cart_items_tool, view_cart_total_tool
