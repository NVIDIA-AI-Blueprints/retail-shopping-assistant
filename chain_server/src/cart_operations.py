"""What the cart tools do, once the turn has resolved what they act on.

Adding, removing, resizing and re-quantifying cart lines. These ran as
closures inside `_create_agent`, which is how a single function reached 1,575
lines; the turn context they closed over -- the runtime, the turn's state, who
is asking, and the scope collecting what the turn did -- is now passed in.

Nothing about what they do has changed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from shared.commerce_contracts import (
    AddCartItemInput,
    GetProductDetailsInput,
    ProductSummary,
    RemoveCartItemInput,
    UpdateCartItemInput,
)

from .agenttypes import State
from .commerce_tools import (
    add_cart_item,
    get_product_details,
    remove_cart_item,
    update_cart_item,
)
from .control_signals import EFFECTS_KEY, committed_effect
from .conversation_products import ConversationProductsError, ProductReferenceDescriptor
from .response_format import (
    _format_cart_add_result,
    _format_cart_remove_result,
    _format_size_change_result,
    _format_update_cart_result,
)
from .turn_scope import TurnScope
from .turn_support import (
    AddCartItemsToolItemInput,
    RequestIdentity,
    _cart_add_scope_failures,
    _cart_line_by_id,
    _cart_product_choice_note,
    _cart_resize_issue,
    _cart_size_issue,
    _identified_in_the_current_showing,
    _most_recently_shown,
    _normalize_cart_add_tool_items,
    _one_size_note,
    _product_detail_failure_message,
    _same_product_display_name,
    _shopper_words_this_conversation,
)

if TYPE_CHECKING:
    from .deepagents_runtime import DeepAgentsRuntime

logger = logging.getLogger(__name__)


def add_items_to_the_cart(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    items: list[AddCartItemsToolItemInput],
):
    """Add products to the cart. Use ONLY on explicit shopper intent to
    add, buy, or put items in the cart. Requires PRODUCT_REF values from
    current-turn search or historical-product resolution — not names.
    Call once with every item the shopper asked to add, not once
    per item. "All items" means the ones they asked for, not
    everything in play this turn: "add the black one in a 2 and
    show me a clutch to go with it" adds the dress and shows the
    clutch. A product the shopper asked to see is not an item.
    """

    try:
        requested_items = _normalize_cart_add_tool_items(items)
        choices_from_a_description: list[str] = []
    except ValueError as exc:
        return f"Cart add failed: {exc}"
    if not requested_items:
        return "Cart add failed: provide at least one PRODUCT_REF to add."

    #: Refs whose lookup broke rather than came back empty. A reference
    #: that cannot be read is a fault in this service, and the answer it
    #: earns is not the one a reference nobody was ever shown earns.
    lookup_failures: list[str] = []

    def _resolve_from_conversation_index(product_ref: str):
        """Look one ref up in the conversation's durable product index."""

        descriptors = [
            ProductReferenceDescriptor(
                reference_id="cart-add",
                product_ref=product_ref,
            )
        ]
        try:
            result = runtime._conversation_products.resolve(
                identity.conversation_id,
                descriptors,
            )
        except (ConversationProductsError, ValidationError) as exc:
            # The lookup itself failed, which is not the same fact as
            # this conversation never having shown the product, and
            # returning the same `None` for both is how a working lookup
            # came out as a missing bag. The record held the product and
            # returned it; the response carried one field the product
            # contract does not admit, so it was refused here and the
            # refusal was read upstream as "no such reference". The
            # shopper was told the reference was not valid, and nothing
            # anywhere said why.
            logger.error(
                "chain-server | cart add | the conversation record could "
                "not be read for PRODUCT_REF %s: %s",
                product_ref,
                exc,
            )
            lookup_failures.append(product_ref)
            return None
        scope.product_evidence.add_resolutions(result.results, descriptors)
        state.system_identified_products = list(
            scope.product_evidence.system_identified()
        )
        return scope.product_evidence.get(product_ref)

    resolved: list[tuple[str, ProductSummary, int]] = []
    failed: list[str] = []
    blocked: list[str] = []
    for (product_ref, size), request in requested_items.items():
        product = scope.product_evidence.get(product_ref)
        if product is None:
            # The conversation's product index is the identity lane: it
            # is durable, scoped to this conversation, and printed into
            # the prompt every turn -- which is where the model read
            # this ref. Evidence is rebuilt per turn, so a ref shown two
            # turns ago is absent from it, and refusing on that basis
            # rejected a ref the shopper had genuinely been shown.
            # This is a lookup in the conversation's own record, not a
            # catalog search.
            product = _resolve_from_conversation_index(product_ref)
        if product is None and product_ref in lookup_failures:
            # Read, not missing. Sending this down the path below would
            # tell the model to search the catalog for a product the
            # record is holding, and to ask the shopper which of the
            # results they meant -- about the one they just named.
            # Nothing the model can do repairs a fault in this service,
            # so say that, and let the turn say so plainly rather than
            # inventing a reason the shopper is at fault.
            failed.append(
                f"- PRODUCT_REF '{product_ref}': this conversation's "
                "record could not be read, which is a fault on our "
                "side and not a missing product. Do not search for it "
                "and do not ask the shopper to identify it again. Tell "
                "them the cart could not be updated just now and that "
                "they can try again."
            )
            continue
        if product is None:
            # Two different situations reach here, and naming only one
            # of them stranded the other.
            #
            # A ref the shopper was never shown: the model passes the
            # product's name, and told only that a name is not a ref it
            # asked the shopper for the exact catalogue name -- the
            # assistant's own job, with a search budget unspent.
            #
            # A ref shown in an *earlier* turn: evidence is rebuilt per
            # turn, so a real ref from last turn is absent from this
            # one. "Add it in a 10 as well" carried the correct ref for
            # a dress added moments earlier, was told to go searching,
            # and gave up -- so a shopper asking for a second size got
            # "the add didn't go through".
            failed.append(
                f"- PRODUCT_REF '{product_ref}': not established in "
                "this turn. If this product was shown earlier in the "
                "conversation, resolve it first and add the PRODUCT_REF "
                "that comes back -- evidence is per turn, so a "
                "reference from an earlier turn has to be resolved "
                "again. If it is a product name rather than a "
                "reference, or was never shown at all, search the "
                "catalog now and show the closest matches, then ask "
                "which to add. Never add a product the shopper has not "
                "been shown, and do not ask them for a catalogue name, "
                "a link, or a price."
            )
            continue
        expected_name = request.get("expected_display_name") or ""
        if expected_name and not _same_product_display_name(
            expected_name,
            product.display_name,
        ):
            blocked.append(
                f"- PRODUCT_REF '{product_ref}': expected "
                f"'{expected_name}', but that ref resolves to "
                f"'{product.display_name}'. Use the matching PRODUCT_REF "
                "for the intended product before adding."
            )
            continue
        active_detail = get_product_details(
            GetProductDetailsInput(product_id=product.product_id),
            runtime.config.retriever_port,
            timeout_seconds=runtime.config.catalog_search_timeout_seconds,
        )
        if not active_detail.ok or active_detail.product is None:
            failed.append(
                f"- PRODUCT_REF '{product_ref}': "
                + _product_detail_failure_message(
                    active_detail.error,
                    cart_validation=True,
                )
            )
            continue
        if not _same_product_display_name(
            active_detail.product.display_name,
            product.display_name,
        ):
            blocked.append(
                f"- PRODUCT_REF '{product_ref}': That reference now "
                "resolves to a different product. Search again and use "
                "the new PRODUCT_REF before adding it."
            )
            continue
        # Whether the catalog sells this size is a fact. Whether the
        # shopper settled on it is answered from their own words, all
        # of them, this turn and every turn before -- so "add the Jade
        # Suede Heels in a 7" passes on the 7 they typed, and a 7 they
        # gave five turns ago for something else still counts as said.
        # What does not pass is a size that appears nowhere they spoke,
        # which is the only way one nobody picked reaches the cart.
        size_issue = _cart_size_issue(
            active_detail.product,
            size,
            _shopper_words_this_conversation(state),
        )
        if size_issue:
            blocked.append(f"- PRODUCT_REF '{product_ref}': {size_issue}")
            continue
        one_size_note = _one_size_note(active_detail.product, size)
        if one_size_note:
            size = None
            choices_from_a_description.append(one_size_note)
        # Disclosed, not refused. A description the model read one way
        # is added and said out loud, because the cart is on screen and
        # a wrong line is one click away -- where a refusal costs a
        # turn on every request it misjudges, and it misjudged plenty.
        choice_note = _cart_product_choice_note(
            active_detail.product,
            _shopper_words_this_conversation(state),
            scope.product_evidence,
            _most_recently_shown(state),
            _identified_in_the_current_showing(state),
            size,
        )
        if choice_note:
            choices_from_a_description.append(
                f"- {active_detail.product.display_name}: {choice_note}"
            )
        resolved.append(
            (
                product_ref,
                active_detail.product,
                int(request["quantity"]),
                size,
            )
        )

    scope_failures = _cart_add_scope_failures(
        state.query,
        [(product_ref, product) for product_ref, product, _, _ in resolved],
        scope.product_evidence.values(),
    )
    blocked.extend(message for _ref, message in scope_failures)
    out_of_scope = {ref for ref, _message in scope_failures}
    if blocked:
        state.cart = runtime._read_cart(identity.cart_user_id)
        runtime._append_product_images(
            scope.retrieved,
            state.cart,
            scope.product_evidence.values(),
        )
        # Nothing is written -- the add is all or nothing. But the items
        # that were established travel with the refusal, so the question
        # put to the shopper is only the one still open.
        # An item can pass every per-item gate and still be refused
        # below as outside this turn's request. Listing it as settled
        # would tell the shopper not to ask again about the very thing
        # that failed.
        ready = [
            f"- {product.display_name}"
            + (f", size {size}" if size else "")
            + f", qty {quantity}"
            for ref, product, quantity, size in resolved
            if ref not in out_of_scope
        ]
        return _format_cart_add_result(
            [], failed + blocked, state.cart, ready
        )

    added: list[str] = []
    committed: list[dict[str, Any]] = []
    for product_ref, product, quantity, size in resolved:
        result = add_cart_item(
            AddCartItemInput(
                user_id=str(identity.cart_user_id),
                product_id=product.product_id,
                display_name=product.display_name,
                quantity=quantity,
                size=size,
                unit_price=product.price,
                image_url=product.image_url,
                # Size is part of the key: adding a 6 and an 8 in one
                # turn are two mutations, not a retry of one.
                idempotency_key=(
                    f"{identity.request_id}:add:{product.product_id}"
                    f":{size or 'onesize'}:{quantity}"
                ),
            ),
            runtime.config.memory_port,
        )
        if result.ok:
            committed.append(
                {
                    "operation": "added to cart",
                    "idempotency_key": (
                        f"{identity.request_id}:add:"
                        f"{product.product_id}:{quantity}"
                    ),
                    "product_id": product.display_name,
                    "quantity": quantity,
                }
            )
            # The size travels with the line it went in as. Nothing
            # now refuses a size the shopper did not choose, so the
            # whole safety story is that it is visible -- to the model
            # writing the reply, and through it to the shopper, on the
            # turn it happens rather than at checkout.
            added.append(
                f"- {quantity} x {product.display_name}"
                + (f", size {size}" if size else "")
                + f" (PRODUCT_REF: {product.product_id})"
            )
        else:
            message = (
                result.error.message if result.error else "Cart add failed."
            )
            failed.append(f"- PRODUCT_REF '{product_ref}': {message}")

    state.cart = runtime._read_cart(identity.cart_user_id)
    runtime._append_product_images(
        scope.retrieved,
        state.cart,
        scope.product_evidence.values(),
    )
    rendered = _format_cart_add_result(added, failed, state.cart)
    if choices_from_a_description:
        rendered += "\n\n" + "\n".join(choices_from_a_description)
    if not committed:
        return rendered
    return rendered, {EFFECTS_KEY: committed}
# Not a tool. `remove_cart_item_tool` calls this directly, and a
# decorated function is a StructuredTool, which is not callable. Its
# sibling `_add_cart_items_impl` is undecorated for the same reason.
def remove_a_cart_line(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    cart_line_id: str,
    quantity: int=1,
):
    """Remove a cart line. Use ONLY on explicit shopper intent to remove
    an item. Requires CART_LINE_ID from get_cart_tool — do not guess.
    Use update_cart_items_tool to change quantity instead of removing
    and re-adding.
    """

    quantity = max(1, int(quantity or 1))
    cart = runtime._read_cart(identity.cart_user_id)
    line = _cart_line_by_id(cart_line_id, cart)
    if line is None:
        return f"No cart line with CART_LINE_ID '{cart_line_id}' could be found."
    result = remove_cart_item(
        RemoveCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=line["cart_line_id"],
            product_id=line.get("product_id"),
            display_name=line["item"],
            quantity=quantity,
            idempotency_key=f"{identity.request_id}:remove:{line['cart_line_id']}:{quantity}",
        ),
        runtime.config.memory_port,
    )
    state.cart = runtime._read_cart(identity.cart_user_id)
    runtime._append_product_images(
        scope.retrieved,
        state.cart,
        scope.product_evidence.values(),
    )
    rendered = _format_cart_remove_result(
        result,
        fallback=f"Removed {quantity} {line['item']} from cart.",
    )
    if not result.ok:
        return rendered
    return committed_effect(
        rendered,
        operation="removed from cart",
        idempotency_key=(
            f"{identity.request_id}:remove:{line['cart_line_id']}:{quantity}"
        ),
        cart_line_id=line["cart_line_id"],
        product_id=line["item"],
        quantity=quantity,
    )
def change_a_line_size(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    cart_line_id: str,
    quantity: int,
    size: str,
):
    """Move a cart line to another size, in the call that asked for it.

    A size is a separate cart line rather than a property of one, and
    the cart has no operation that changes it, so the change is an add
    followed by a remove -- in that order, because a failure between
    the two must leave the shopper an extra line rather than nothing.

    Nothing in that sequence needs the model. It has already said
    everything there is to say -- this line, that size, that many --
    and the rest is bookkeeping this code does without asking, then
    reports the cart it actually left behind.
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
        # The size asked for is the size they have, so this is a
        # quantity change that happens to name a size. Answering it as
        # one keeps the cart to a single line: routing it through the
        # add below would put a second line on the same size.
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
            detail.error,
            cart_validation=True,
        )
    # Whether the catalog sells the new size is a fact, and mostly the
    # same fact the add path checks. A resize that skipped it would
    # seat a size the shop does not sell by a route the add refuses --
    # and this is the route a shopper naming a size reaches.
    size_issue = _cart_resize_issue(detail.product, size)
    if size_issue:
        return f"CART_UPDATE_REFUSED: {size_issue}"

    product = detail.product
    add = add_cart_item(
        AddCartItemInput(
            user_id=str(identity.cart_user_id),
            product_id=product_id,
            display_name=product.display_name,
            quantity=quantity,
            size=size,
            unit_price=product.price,
            image_url=product.image_url,
            idempotency_key=(
                f"{identity.request_id}:resize:add:{product_id}"
                f":{size}:{quantity}"
            ),
        ),
        runtime.config.memory_port,
    )
    if not add.ok:
        # Adding first is what makes this failure safe: the line the
        # shopper has is untouched, so there is nothing half-done to
        # explain and nothing lost.
        message = add.error.message if add.error else "Cart add failed."
        return (
            f"CART_UPDATE_FAILED: {message} The cart still holds "
            f"{product.display_name} in size {held or 'onesize'}, and "
            "nothing was removed."
        )

    remove = remove_cart_item(
        RemoveCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=cart_line_id,
            quantity=int(line.get("amount") or 1),
            product_id=product_id,
            display_name=product.display_name,
            idempotency_key=(
                f"{identity.request_id}:resize:remove:{cart_line_id}"
            ),
        ),
        runtime.config.memory_port,
    )

    state.cart = runtime._read_cart(identity.cart_user_id)
    runtime._append_product_images(
        scope.retrieved,
        state.cart,
        scope.product_evidence.values(),
    )
    # Two mutations, recorded as two. A turn that dies after this still
    # has to be able to say what the cart holds, and on the unhappy
    # path what it holds is both sizes.
    committed: list[dict[str, Any]] = [
        {
            "operation": "added to cart",
            "idempotency_key": (
                f"{identity.request_id}:resize:add:{product_id}"
                f":{size}:{quantity}"
            ),
            "product_id": product.display_name,
            "quantity": quantity,
        }
    ]
    if remove.ok:
        committed.append(
            {
                "operation": "removed from cart",
                "idempotency_key": (
                    f"{identity.request_id}:resize:remove:{cart_line_id}"
                ),
                "cart_line_id": cart_line_id,
                "product_id": product.display_name,
            }
        )
    rendered = _format_size_change_result(
        display_name=product.display_name,
        from_size=held,
        to_size=size,
        quantity=quantity,
        cart=state.cart,
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
    size: str | None=None,
):
    """Change the quantity or the size of an item already in the cart.
    Use ONLY when the shopper explicitly asks for the change. Do NOT
    use for initial adds — use add_cart_items_tool. Do NOT guess the
    CART_LINE_ID; call get_cart_tool first if you do not have one.
    """

    if size is not None and str(size).strip():
        return change_a_line_size(runtime, state, identity, scope,
            cart_line_id,
            quantity,
            str(size).strip(),
        )

    if quantity == 0:
        # Deleting is a different intent from setting a quantity, and
        # it has its own tool. A size change has its own route, above,
        # so quantity 0 is never the way to correct a size.
        return (
            "CART_UPDATE_REFUSED: quantity 0 would delete this line, and "
            "this tool sets quantities. If the shopper wants the line "
            "gone, use remove_cart_item_tool. If they are changing a "
            "SIZE, send the new size in `size` with the quantity to "
            "keep, and this tool makes the change."
        )

    result = update_cart_item(
        UpdateCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=cart_line_id,
            quantity=quantity,
            idempotency_key=(
                f"{identity.request_id}:update:{cart_line_id}:{quantity}"
            ),
        ),
        runtime.config.memory_port,
    )
    state.cart = runtime._read_cart(identity.cart_user_id)
    runtime._append_product_images(
        scope.retrieved,
        state.cart,
        scope.product_evidence.values(),
    )
    rendered = _format_update_cart_result(result, state.cart)
    if not result.ok:
        return rendered
    return committed_effect(
        rendered,
        operation="cart quantity updated",
        idempotency_key=(
            f"{identity.request_id}:update:{cart_line_id}:{quantity}"
        ),
        cart_line_id=cart_line_id,
        quantity=quantity,
    )
