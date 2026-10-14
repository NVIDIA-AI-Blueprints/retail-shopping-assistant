# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the cart and each change to it are written for the model.

These turn already-decided data into the exact strings the model reads and
decide nothing themselves. Several are asserted byte-for-byte by the
evidence tests, so a wording change is a visible diff.
"""

from __future__ import annotations

from shared.commerce_contracts import Cart as CommerceCart
from shared.commerce_contracts import CartMutationResult

from .agenttypes import Cart


def _format_cart_add_result(
    added: list[str],
    failed: list[str],
    cart: Cart,
    ready: list[str] | None = None,
) -> str:
    """Report an add, including what was established but not written.

    The add is all or nothing, so one unanswered item holds back the rest. That
    is deliberate. What was not deliberate is that the held-back items vanished
    from the result: a shopper who gave a correct size for the boots and a
    letter size for the sweater was asked for both again, because nothing told
    the model the boots were already settled.
    """

    lines = ["CART_ADD_RESULT"]
    if added:
        lines.append("Added:")
        lines.extend(added)
    if failed:
        lines.append("Failed:")
        lines.extend(failed)
    if ready:
        lines.append(
            "Established, not added -- the add is all or nothing, so these are "
            "waiting on the item above. Do not ask for these again:"
        )
        lines.extend(ready)
    lines.append("Current cart:")
    lines.append(_format_cart_lines(cart))
    lines.append("Cart total:")
    lines.append(_format_cart_total(cart))
    return "\n".join(lines)


def _format_cart_lines(cart: Cart | CommerceCart) -> str:
    if isinstance(cart, CommerceCart):
        if not cart.lines:
            return "  (cart is empty)"
        lines = [
            f"  {line.cart_line_id} | {line.display_name} | qty {line.quantity}"
            + (f" | size {line.size}" if line.size else "")
            + (
                f" | {line.unit_price.currency} {line.unit_price.amount:.2f}"
                if line.unit_price
                else ""
            )
            for line in cart.lines
        ]
        if cart.subtotal:
            lines.append(
                f"  SUBTOTAL: {cart.subtotal.currency} {cart.subtotal.amount:.2f}"
            )
        return "\n".join(lines)

    if not cart.contents:
        return "(empty)"
    lines = []
    for item in cart.contents:
        price = item.get("price")
        suffix = ""
        if price is not None:
            try:
                suffix = f" @ ${float(price):.2f}"
            except (TypeError, ValueError):
                suffix = ""
        cart_line_id = item.get("cart_line_id") or item.get("item", "")
        # The size is what makes two lines of one dress make sense to a
        # shopper reading their own cart back.
        size = item.get("size")
        size_text = f" (size {size})" if size else ""
        lines.append(
            f"- CART_LINE_ID: {cart_line_id} | "
            f"{item.get('amount', 1)} x {item.get('item', '')}{size_text}{suffix}"
        )
    return "\n".join(lines)


def _format_cart(cart: Cart) -> str:
    return _format_cart_lines(cart)


def _format_cart_remove_result(
    result: CartMutationResult,
    *,
    fallback: str,
) -> str:
    if not result.ok:
        return result.error.message if result.error else "Cart remove failed."
    message = result.message or fallback
    if result.cart is not None:
        return "\n".join(
            [message, "Current cart:", _format_cart_lines(result.cart)]
        )
    return message


def _format_update_cart_result(
    result: CartMutationResult,
    cart: Cart | CommerceCart | None = None,
) -> str:
    if not result.ok:
        message = result.error.message if result.error else "unknown error"
        return f"CART UPDATE FAILED: {message}"
    lines = ["CART UPDATED"]
    if result.changed_line:
        lines.append(
            f"  {result.changed_line.display_name} → "
            f"qty {result.changed_line.quantity}"
        )
    active_cart = cart if cart is not None else result.cart
    if active_cart is not None:
        lines.append(_format_cart_lines(active_cart))
    return "\n".join(lines)


def _format_size_change_result(
    *,
    display_name: str,
    from_size: str,
    to_size: str,
    quantity: int,
    cart: Cart | CommerceCart | None,
    old_line_removed: bool,
    old_line_id: str,
) -> str:
    """Report a size change as the one change it is, or say what is left over.

    The add runs before the remove, so that a failure between them leaves the
    shopper an extra line rather than nothing. That is also why there are two
    reports: on the unhappy path the cart really does hold both sizes, and the
    turn has to say so and carry the id that finishes the job, rather than
    announce a replacement that only half happened.
    """

    held = from_size or "onesize"
    if old_line_removed:
        lines = [
            f"CART SIZE CHANGED: {display_name} is now qty {quantity}, size "
            f"{to_size}. The size {held} line was removed."
        ]
    else:
        lines = [
            f"CART SIZE PARTIALLY CHANGED: size {to_size} was added for "
            f"{display_name}, but the size {held} line could not be removed, "
            "so the cart holds both. Remove it with remove_cart_item_tool "
            f"using CART_LINE_ID {old_line_id}. Tell the shopper what the "
            "cart actually holds, not what was asked for."
        ]
    if cart is not None:
        lines.append(_format_cart_lines(cart))
    return "\n".join(lines)


def _format_cart_total(cart: Cart) -> str:
    if not cart.contents:
        return "Your cart is empty, so the total is $0.00."
    subtotal = 0.0
    missing = []
    lines = []
    for item in cart.contents:
        name = item.get("item", "")
        amount = int(item.get("amount") or 0)
        price = item.get("price")
        if price is None:
            missing.append(name)
            lines.append(f"- {amount} x {name}: price unavailable")
            continue
        line_total = float(price) * amount
        subtotal += line_total
        lines.append(f"- {amount} x {name} @ ${float(price):.2f} = ${line_total:.2f}")
    total = f"Cart total: ${subtotal:.2f}"
    if missing:
        total += f" excluding items without cached prices: {', '.join(missing)}"
    return "\n".join(lines + [total])


def _cart_line_key(line: dict) -> tuple:
    """Identity of a cart line for comparison: what a shopper would call
    'the same line' -- the product and the size, not the opaque line id."""
    return (
        str(line.get("item") or line.get("display_name") or ""),
        str(line.get("size") or ""),
    )


def format_cart_change(before: Cart | None, after: Cart | None) -> str:
    """State what this turn did to the cart, as a fact.

    The editor was already told not to claim a cart action absent from CURRENT
    CART, and it still passed "I've added the tote bag back" on a turn where the
    add failed and the cart was unchanged. A prohibition left it comparing two
    lists and judging; this hands it the answer. Computed from the two
    snapshots, so it cannot disagree with the cart.
    """

    if before is None or after is None:
        return "not known for this turn"
    b: dict[tuple, int] = {}
    for line in getattr(before, "contents", []) or []:
        k = _cart_line_key(line)
        b[k] = b.get(k, 0) + int(line.get("amount") or 0)
    a: dict[tuple, int] = {}
    for line in getattr(after, "contents", []) or []:
        k = _cart_line_key(line)
        a[k] = a.get(k, 0) + int(line.get("amount") or 0)
    changes: list[str] = []
    for k in sorted(set(a) | set(b), key=lambda x: (x[0], x[1])):
        name, size = k
        label = f"{name}" + (f" (size {size})" if size else "")
        delta = a.get(k, 0) - b.get(k, 0)
        if delta > 0:
            changes.append(f"- added {label} x{delta}")
        elif delta < 0:
            changes.append(f"- removed {label} x{-delta}")
    if not changes:
        return (
            "NOTHING CHANGED. No item was added, removed, or altered this "
            "turn. Do not tell the shopper otherwise, whatever the draft says."
        )
    return "\n".join(changes)
