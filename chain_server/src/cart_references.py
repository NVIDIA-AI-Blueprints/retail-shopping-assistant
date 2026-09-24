# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which product, and which size, a cart request refers to."""

from __future__ import annotations

import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError
from shared.commerce_contracts import ProductSummary

from .agenttypes import Cart
from .catalog_format import _format_product_refs
from .product_records import (
    _normalize_product_name,
    _product_name_tokens,
    _product_name_tokens_match,
)
from .sizes import _ONE_SIZE, _advertised_sizes

if TYPE_CHECKING:
    from .conversation_products import ProductEvidence


class AddCartItemsToolItemInput(BaseModel):
    product_ref: str = Field(
        ...,
        min_length=1,
        description="PRODUCT_REF returned by search_catalog_tool in this conversation.",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="Quantity of this product to add.",
    )
    expected_display_name: str | None = Field(
        default=None,
        description=(
            "Shopper-facing product name the agent intends to add. When the "
            "shopper explicitly names the product, copy that exact product name."
        ),
    )
    size: str | None = Field(
        default=None,
        max_length=32,
        description=(
            "The size to add, exactly as the product lists it. Required when "
            "the product carries real sizes, and omitted when its only size "
            "is 'onesize' -- asking what size handbag someone wants is worse "
            "than not asking. Use only a size that product actually comes in; "
            "the sizes differ per product and are in its details."
        ),
    )


def _normalize_cart_add_tool_items(
    items: list[AddCartItemsToolItemInput] | list[dict[str, Any]],
) -> dict[tuple[str, str | None], dict[str, Any]]:
    normalized: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in items or []:
        try:
            parsed = (
                item
                if isinstance(item, AddCartItemsToolItemInput)
                else AddCartItemsToolItemInput.model_validate(item)
            )
            quantity = max(1, int(parsed.quantity or 1))
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError("each item must include a PRODUCT_REF and quantity") from exc
        # Keyed on size as well as reference: asking for a 6 and an 8 of one
        # dress is two lines, and merging them would quietly halve the order.
        size = (parsed.size or "").strip() or None
        entry = normalized.setdefault(
            (parsed.product_ref, size),
            {
                "quantity": 0,
                "size": size,
                "expected_display_name": (
                    parsed.expected_display_name.strip()
                    if parsed.expected_display_name
                    else ""
                ),
            },
        )
        entry["quantity"] += quantity
        if not entry["expected_display_name"] and parsed.expected_display_name:
            entry["expected_display_name"] = parsed.expected_display_name.strip()
    return normalized


def _cart_add_scope_failures(
    user_query: str,
    requested_products: list[tuple[str, ProductSummary]],
    available_products: Any,
) -> list[tuple[str, str]]:
    """Which requested products fall outside this turn's explicit request.

    Returns the ref beside its message. The ref is what a caller needs to know
    which item failed, and recovering it by reading the message back would be
    parsing prose for control state -- which is the thing this codebase refuses
    to do everywhere else.
    """

    explicitly_named = _explicitly_named_products(user_query, available_products)
    if not explicitly_named:
        return []

    explicit_names = {
        _normalize_product_name(product.display_name) for product in explicitly_named
    }
    failures: list[tuple[str, str]] = []
    for product_ref, product in requested_products:
        if _normalize_product_name(product.display_name) in explicit_names:
            continue
        failures.append(
            (
                product_ref,
                f"- PRODUCT_REF '{product_ref}': selected '{product.display_name}' "
                "is outside the current explicit add request. The current request "
                f"names: {_format_product_refs(explicitly_named)}. Retry with "
                "matching PRODUCT_REF values only, or ask a clarification.",
            )
        )
    return failures


def _identified_in_the_current_showing(state: Any) -> set[str]:
    """Products the record picked from the set now in front of the shopper.

    An identification is filed against the showing it was made from, so this is
    simply the newest showing's own list. When a newer set is presented it
    becomes the newest, carrying its own choices and none of the older set's --
    which is the lapse, expressed as a consequence of where the fact is kept
    rather than as a rule that has to be remembered.
    """

    sets = [
        entry
        for entry in (getattr(state, "historical_product_sets", None) or [])
        if isinstance(entry, dict) and isinstance(entry.get("products"), list)
    ]
    if not sets:
        return set()
    newest = max(sets, key=lambda entry: entry.get("turn_seq") or 0)
    return {
        str(ref) for ref in (newest.get("system_identified") or []) if ref
    }


def _reference_candidates(
    evidence: ProductEvidence,
    recently_shown: Sequence[Any] = (),
) -> list[Any]:
    """The products a reference in this turn could be pointing at."""

    candidates = list(evidence.values())
    seen = {candidate.product_id for candidate in candidates}
    for entry in recently_shown or ():
        ref = entry.get("ref") if isinstance(entry, dict) else None
        name = entry.get("name") if isinstance(entry, dict) else None
        if not ref or not name or ref in seen:
            continue
        seen.add(ref)
        candidates.append(SimpleNamespace(product_id=ref, display_name=name))
    return candidates


def _the_only_one_on_screen_in_that_size(
    product: Any,
    size: str | None,
    recently_shown: Sequence[Any] = (),
) -> bool:
    """Whether the size the shopper gave leaves one thing they could have meant.

    "Add the black one in a 2" was refused with ten products in play -- six of
    them clutches the same turn went and fetched because the sentence also
    asked for a clutch. Of what was actually on screen when the shopper spoke,
    the dress runs 2-12, the pumps 5-9 and the necklace is onesize. "In a 2"
    leaves exactly one.

    Both halves are facts. The shopper typed the size, and which products come
    in a 2 is published by the catalog and recorded with the showing. Nothing
    here reads what they meant; it counts what they could have meant.

    Only the showing in front of them counts. Products the turn fetched
    afterwards, for another role in the same sentence, were not on screen when
    the reference was spoken and cannot be what it pointed at.
    """

    from .conversation_products import _same_reference

    wanted = (size or "").strip().casefold()
    if not wanted:
        return False
    fits: list[str] = []
    for entry in recently_shown or ():
        if not isinstance(entry, dict):
            continue
        ref, sizes = entry.get("ref"), entry.get("sizes")
        if not ref or not isinstance(sizes, list) or not sizes:
            # A showing that never recorded its sizes cannot narrow anything,
            # and guessing from silence is how a wrong dress reaches a cart.
            return False
        values = {str(value).strip().casefold() for value in sizes}
        if values != {_ONE_SIZE} and wanted in values:
            fits.append(str(ref))
    return len(fits) == 1 and _same_reference(fits[0], str(product.product_id))


def _products_named_exactly(text: str, candidates: Any) -> list[Any]:
    """Candidates whose full catalog name the shopper actually wrote.

    Narrower than `_explicitly_named_products`, which also matches on token
    overlap so a shortened or misspelt name still lands. That second half is a
    reading; out-of-scope detection still wants it, a cart write does not.
    """

    normalized_text = _normalize_product_name(text)
    if not normalized_text:
        return []
    padded = f" {normalized_text} "
    named: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        name = _normalize_product_name(getattr(candidate, "display_name", ""))
        if not name or f" {name} " not in padded:
            continue
        key = getattr(candidate, "product_id", None) or name
        if key in seen:
            continue
        seen.add(key)
        named.append(candidate)
    return named


def _explicitly_named_products(
    text: str,
    available_products: Any,
) -> list[ProductSummary]:
    normalized_text = _normalize_product_name(text)
    if not normalized_text:
        return []

    padded_text = f" {normalized_text} "
    matches: list[ProductSummary] = []
    seen: set[str] = set()
    products = list(available_products)
    for product in products:
        normalized_name = _normalize_product_name(product.display_name)
        if not normalized_name:
            continue
        if f" {normalized_name} " not in padded_text:
            continue
        key = product.product_id or product.display_name
        if key in seen:
            continue
        seen.add(key)
        matches.append(product)

    query_tokens = set(_product_name_tokens(text))
    for product in products:
        key = product.product_id or product.display_name
        if key in seen:
            continue
        product_tokens = _product_name_tokens(product.display_name)
        required_overlap = 3 if len(product_tokens) > 3 and matches else 2
        if not _product_name_tokens_match(
            query_tokens,
            product_tokens,
            required_overlap=required_overlap,
        ):
            continue
        seen.add(key)
        matches.append(product)
    return matches


#: The value, written the other way. Numbers only: a quantity of two is the
#: same want whether the shopper typed it as a word or a digit.
_SPELLED_NUMBERS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve",
}


def _shopper_words_this_conversation(state: Any) -> str:
    """Everything the shopper has actually typed, this turn and before.

    A size settled one turn ago -- "do you have it in a 6?" answered, then "yes,
    add it" -- is established in the conversation and quotable from it. Reading
    only the current message refused adds for sizes the shopper had already
    given, which is the failure the cart reference had before it learned to look
    further back than this turn.
    """

    parts = [str(getattr(state, "query", "") or "")]
    for turn in getattr(state, "dialogue", None) or []:
        text = getattr(turn, "shopper_text", "")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _most_recently_shown(state: Any) -> list[dict]:
    """The last set of products put in front of the shopper."""

    sets = [
        entry
        for entry in (getattr(state, "historical_product_sets", None) or [])
        if isinstance(entry, dict) and isinstance(entry.get("products"), list)
    ]
    if not sets:
        return []
    newest = max(sets, key=lambda entry: entry.get("turn_seq") or 0)
    return [item for item in newest["products"] if isinstance(item, dict)]


def _cart_product_choice_note(
    product: Any,
    shopper_text: str,
    evidence: ProductEvidence,
    recently_shown: Sequence[Any] = (),
    already_identified: Sequence[str] = (),
    size: str | None = None,
) -> str:
    """Say when a product reached the cart from a description rather than a name.

    It discloses rather than refuses. The absence of confirmation is a gap in
    our bookkeeping, not a fact about the world, and refusing on it costs a
    turn every time it is wrong. A product nobody chose is caught by being
    visible: the cart is on screen, a wrong line is one click to remove, and
    the shopper is told which reading was taken.

    Silent when the choice is settled by something checkable:

    - only one product it could have been
    - the shopper wrote the catalog's own name for it
    - the record picked it, by a ref it minted or a position it wrote down
    - they chose it earlier and no newer showing has retired that
    - the size they gave leaves one thing on screen it could be
    """

    candidates = _reference_candidates(evidence, recently_shown)
    if len(candidates) <= 1:
        return ""
    if _the_only_one_on_screen_in_that_size(product, size, recently_shown):
        return ""
    if any(
        getattr(match, "product_id", None) == product.product_id
        for match in _products_named_exactly(shopper_text, candidates)
    ):
        return ""
    if evidence.identified_by_the_system(product.product_id):
        return ""
    if str(product.product_id) in {str(ref) for ref in (already_identified or ())}:
        return ""
    return (
        f"CHOSEN FROM A DESCRIPTION: the shopper did not name "
        f"'{product.display_name}', and {len(candidates)} products were in "
        "play. It has been added. Say which one you took them to mean and "
        "offer to change it."
    )


def _cart_size_issue(
    product: Any,
    size: str | None,
    catalog_vocabulary: str = "",
) -> str:
    """Say why this size cannot be added, or "" if it can.

    Every product in the catalog states its sizes -- 136 carry a real range and
    79 carry `onesize`, with no gaps -- so the tool has what it needs to decide
    rather than trusting the caller to have asked. Left to prose alone, "always
    confirm the size" held three times in four: a dress with six sizes reached
    the cart with no size at all.

    Sending no size is not the only way to add one nobody picked. Asked plainly
    to "add the Jade Suede Heels", the model read the range off the product
    detail, sent a 5, and passed a check that only asks whether the shop sells
    a 5 -- so the empty-size gate held and the shopper still got a size they
    had never mentioned, announced to them as "the smallest size they come in".
    A rule they never gave.

    So the size has to be one the conversation settles. Named is enough,
    whenever it was named: "dresses in a 2" five turns back still settles "add
    the lace one". A superlative is enough too, because it names a size by
    description -- but it has to be the shopper's superlative, not one the
    model supplies to fill the gap. Anything else is not a size to check, it
    is a size to ask for.
    """

    sizes = _advertised_sizes(product)
    chosen = (size or "").strip()
    if not sizes:
        # The catalog said nothing. Refusing here would block a cart on missing
        # data rather than on a real disagreement.
        return ""
    if sizes == [_ONE_SIZE]:
        return ""
    if not chosen:
        return (
            f"SIZE REQUIRED. '{product.display_name}' is sold in "
            f"{', '.join(sizes)}. Ask the shopper which size and add it then. "
            "Nothing was added."
        )
    if not any(chosen.casefold() == value.casefold() for value in sizes):
        return (
            f"SIZE '{chosen}' is not sold for '{product.display_name}'. "
            f"Available: {', '.join(sizes)}. Ask the shopper which of those "
            "they want. Nothing was added."
        )
    if catalog_vocabulary and not _size_the_conversation_settles(
        chosen, sizes, catalog_vocabulary
    ):
        return (
            f"SIZE NOT CHOSEN. The shopper has not said what size, so "
            f"'{chosen}' is yours rather than theirs. "
            f"'{product.display_name}' is sold in {', '.join(sizes)}. Ask "
            "which one. Nothing was added."
        )
    return ""


def _one_size_note(product: Any, size: str | None) -> str:
    """Say a size was dropped because this product comes in only one.

    A size cannot be wrong on a product that has one -- there is nothing else
    to have added -- so this is not a refusal. But it cannot be repeated back
    either. Asked to "add the black one in a size 8", the assistant added a
    one-size purse and told the shopper it was in size 8, a size that product
    has never had. Dropping it keeps the cart line honest; saying so here is
    what keeps the sentence honest too.
    """

    chosen = (size or "").strip()
    if not chosen or _advertised_sizes(product) != [_ONE_SIZE]:
        return ""
    if chosen.casefold() == _ONE_SIZE:
        return ""
    return (
        f"- {product.display_name}: added as one size. This product is sold "
        f"in one size only, so the '{chosen}' was not applied and must not be "
        "described to the shopper as its size."
    )


#: The shopper's own way of naming a size without saying the number.
_SMALLEST_WORDS = ("smallest", "littlest", "tiniest")


_LARGEST_WORDS = ("largest", "biggest")


def _size_the_conversation_settles(
    chosen: str,
    sizes: list[str],
    catalog_vocabulary: str,
) -> bool:
    """Whether the shopper's own words settle on this size.

    Word boundaries matter more than they look: a bare `in` match puts "5"
    inside "$159.99" and turns a price the assistant quoted into a size the
    shopper chose.
    """

    if re.search(rf"\b{re.escape(chosen)}\b", catalog_vocabulary, flags=re.IGNORECASE):
        return True
    spoken = catalog_vocabulary.casefold()
    if any(word in spoken for word in _SMALLEST_WORDS):
        return chosen.casefold() == sizes[0].casefold()
    if any(word in spoken for word in _LARGEST_WORDS):
        return chosen.casefold() == sizes[-1].casefold()
    return False


def _cart_resize_issue(product: Any, size: str) -> str:
    """Say why this line cannot move to this size, or "" if it can.

    The add path above is deliberately permissive about a product the catalog
    states one size for, or no sizes at all: refusing there would block a cart
    on missing data rather than on a real disagreement. A resize is the other
    way round. A product sold in one size has no second size to move to, so a
    size against it is a line the model has misread rather than a size the
    catalog is quiet about.

    J06 t9 is that turn. Asked to "make those a 7" with heels in a 6 and a tote
    bag in the cart, it sent the tote's CART_LINE_ID without reading the cart
    first -- and the tote, onesize, became "size 7" while the heels stayed a 6.
    Which line the shopper meant is the model's to read; that this one has no
    size to change is the catalog's to say.
    """

    sizes = _advertised_sizes(product)
    if not sizes or sizes == [_ONE_SIZE]:
        return (
            f"'{product.display_name}' is sold in one size, so this line has "
            "no size to change. If the shopper meant a different item, call "
            "get_cart_tool and use that line's CART_LINE_ID."
        )
    return _cart_size_issue(product, size)


def _cart_line_by_id(cart_line_id: str, cart: Cart) -> dict[str, Any] | None:
    if not cart.contents:
        return None
    target = (cart_line_id or "").strip()
    if not target:
        return None
    for item in cart.contents:
        if str(item.get("cart_line_id") or "").strip() == target:
            return item
    return None
