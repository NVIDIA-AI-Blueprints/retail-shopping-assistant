# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks and receipts applied to the turn's final reply."""

from __future__ import annotations

import json
from typing import Any

from ..agenttypes import Cart, State
from ..cart_format import _format_cart
from ..catalog_format import _format_filter_statement, _format_search_group
from ..tools.evidence import (
    evidence_of,
)
from ..tools.loop_control import (
    SEARCH_SCOPE_COMPLETE_PREFIX,
    UNSUPPORTED_CONSTRAINT_PREFIX,
)
from ..tools.skill_gate import SKILL_ACTIVATION_TOOL_NAME
from .message_shape import (
    _content_to_text,
    _current_turn_messages,
    _message_type,
    _result_messages,
    _value,
)

# Turning what a search found into what the shopper is told.

_UNSUPPORTED_REQUIREMENT_RESPONSE = (
    "I can't guarantee that requirement from the catalog information available "
    "to this assistant, so I won't present unverified matches. Would you like me "
    "to treat it as a preference and show candidates to verify on their product "
    "pages?"
)



_INTERNAL_SHOPPER_REPLACEMENTS = (
    ("The product detail tool doesn't return", "I don't have"),
    ("the product detail tool doesn't return", "I don't have"),
    ("The catalog detail tool doesn't return", "I don't have"),
    ("the catalog detail tool doesn't return", "I don't have"),
    ("The product detail tool does not return", "I don't have"),
    ("the product detail tool does not return", "I don't have"),
    ("The catalog detail tool does not return", "I don't have"),
    ("the catalog detail tool does not return", "I don't have"),
    ("the product detail tool", "the product details I can access"),
    ("the catalog detail tool", "the product details I can access"),
    ("because the tool requires", "because I need"),
    ("The tool requires", "I need"),
    ("the tool requires", "I need"),
)



def _has_unsupported_requirement_outcome(
    result: Any,
    *,
    request_id: str,
) -> bool:
    """Return whether the current turn contains an unenforceable requirement."""

    return any(
        _message_type(message) == "tool"
        and _content_to_text(_value(message, "content")).startswith(
            UNSUPPORTED_CONSTRAINT_PREFIX
        )
        for message in _current_turn_messages(
            _result_messages(result),
            request_id,
        )
    )



def _products_by_confirmed_filters(
    payload: dict[str, Any],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Split one search result into the filter sets actually applied to it.

    A call carrying several roles merges into one payload whose
    ``confirmed_filters`` is the union across those roles. Stating that union
    against every product is what let a $179.99 sweater be presented as
    confirmed under a $59.99 cap belonging to the shoes.
    """

    call_filters = payload.get("confirmed_filters") or {}
    grouped: list[dict[str, Any]] = []
    for product in payload.get("products") or []:
        scope = product.get("search_scope") if isinstance(product, dict) else None
        filters = (
            scope.get("confirmed_filters") or {}
            if isinstance(scope, dict)
            else call_filters
        )
        key = json.dumps(filters, sort_keys=True, default=str)
        for entry in grouped:
            if entry["key"] == key:
                entry["products"].append(product)
                break
        else:
            grouped.append({"key": key, "filters": filters, "products": [product]})
    if not grouped:
        return [(call_filters, [])]
    return [(entry["filters"], entry["products"]) for entry in grouped]



def _partial_product_results_response(state: State) -> str:
    products = [
        product for product in state.product_results if isinstance(product, dict)
    ]
    if not products:
        return ""

    lines = [
        "I found these grounded catalog options so far:",
        "",
    ]
    seen: set[str] = set()
    for product in products[:8]:
        name = str(product.get("display_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parts = [f"**{name}**"]
        category = str(product.get("category") or "").strip()
        if category:
            parts.append(category)
        price = _product_result_price(product)
        if price:
            parts.append(price)
        lines.append("- " + " — ".join(parts))

    if len(lines) <= 2:
        return ""
    lines.extend(
        [
            "",
            (
                "I do not want to overstate outdoor performance, material, care, "
                "or fit details without a completed detail check. I can continue "
                "from these options or narrow one piece at a time."
            ),
        ]
    )
    return "\n".join(lines)



def _format_search_only_response(
    state: State,
    result: Any,
    *,
    request_id: str,
    products: list[dict[str, Any]] | None = None,
    intro: str = "",
    heading: str = "Catalog candidates (verified name, price, and category):",
) -> str:
    """Render grounded search facts without interpreting display-name words."""

    search_groups = _search_result_groups(result, request_id=request_id)
    grouped_lines, grouped_names = _grouped_search_response_lines(search_groups)
    if len(grouped_lines) > 0:
        lines = grouped_lines
        displayed_names = grouped_names
    else:
        displayed_products = (
            products
            if products is not None
            else [
                product
                for product in state.product_results
                if isinstance(product, dict)
            ]
        )
        candidate_count = len(displayed_products)
        if intro.strip():
            lines = [
                "General guidance (not product-specific facts):",
                intro.strip(),
                "",
            ]
        else:
            noun = "candidate" if candidate_count == 1 else "candidates"
            lines = [f"I found {candidate_count} catalog {noun}.", ""]
        lines.extend((heading, ""))
        displayed_names = set()
        for product in displayed_products:
            name = str(product.get("display_name") or "").strip()
            if not name:
                continue
            displayed_names.add(name)
            parts = [f"**{name}**"]
            price = _product_result_price(product)
            if price:
                parts.append(price)
            category = str(product.get("category") or "").strip()
            if category:
                parts.append(category.replace("_", " "))
            lines.append("- " + " — ".join(parts))

    parent_relations = []
    for group in search_groups:
        relation = group.get("scope_relation") or {}
        requested_type = str(
            relation.get("requested_product_type") or ""
        ).strip()
        category = str(relation.get("advertised_category") or "").strip()
        normalized = {
            "requested_product_type": requested_type,
            "advertised_category": category,
        }
        if (
            relation.get("relation") == "model_selected_parent_category"
            and requested_type
            and category
            and normalized not in parent_relations
        ):
            parent_relations.append(normalized)
    composed_roles: list[str] = []
    for group in search_groups:
        relation = group.get("scope_relation") or {}
        role = str(relation.get("requested_product_type") or "").strip()
        if (
            relation.get("relation") == "model_composed_role"
            and role
            and role not in composed_roles
        ):
            composed_roles.append(role)
    if parent_relations:
        relation_lines = [
            (
                f"The catalog does not advertise **{relation['requested_product_type']}** "
                "as a separate product type. I searched the broader "
                f"**{relation['advertised_category']}** category for the closest "
                "options; each result keeps its actual catalog category."
            )
            for relation in parent_relations
        ]
        lines = relation_lines + [""] + lines
    if composed_roles:
        # One line however many roles were proposed: a four-role look would
        # otherwise open with four near-identical disclaimers.
        named = ", ".join(f"**{role}**" for role in composed_roles)
        lines = [
            f"You didn't name {named} — I suggested "
            "those pieces for this look."
        ] + [""] + lines

    filter_groups = _confirmed_search_filter_groups(
        result,
        request_id=request_id,
        displayed_names=displayed_names,
    )
    if filter_groups:
        lines.extend(("", "Catalog-confirmed filters by search:"))
        for group in filter_groups:
            product_names = group["product_names"]
            scope = (
                ", ".join(f"**{name}**" for name in product_names)
                if product_names
                else "Search candidates"
            )
            lines.append(f"- {scope}: {'; '.join(group['statements'])}.")
    if _has_unsupported_requirement_outcome(
        result,
        request_id=request_id,
    ):
        lines.extend(("", _UNSUPPORTED_REQUIREMENT_RESPONSE))
    if not _search_scope_is_complete(result, request_id=request_id):
        lines.extend(
            (
                "",
                (
                    "This is a partial result set. I can continue with the next "
                    "requested piece or search scope."
                ),
            )
        )
    lines.extend(
        (
            "",
            (
                "These candidates were ranked toward your requested direction. "
                "Product-specific material, construction, length, fit, comfort, "
                "care, or weather performance remains unverified unless listed "
                "above as a catalog-confirmed filter."
            ),
        )
    )
    return "\n".join(lines)



def _search_result_groups(result: Any, *, request_id: str) -> list[dict[str, Any]]:
    """Return successful search evidence grouped by the call that produced it."""

    groups: list[dict[str, Any]] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message)
        if not payload or payload.get("outcome") != "results":
            continue
        groups.append(
            {
                "guidance": _scrub_internal_shopper_language(
                    str(payload.get("shopper_guidance") or "")
                ).strip(),
                "products": payload.get("products") or [],
                "taxonomy": payload.get("taxonomy") or {},
                "scope_relation": _scope_relation_payload(payload),
            }
        )
    return groups



def _grouped_search_response_lines(
    groups: list[dict[str, Any]],
) -> tuple[list[str], set[str]]:
    """Render multi-search candidates without losing their guidance scope."""

    if len(groups) < 2 or not all(group.get("guidance") for group in groups):
        return [], set()
    lines: list[str] = []
    displayed_names: set[str] = set()
    displayed_product_refs: set[str] = set()
    for index, group in enumerate(groups, start=1):
        products = [
            product
            for product in group.get("products") or []
            if str(
                product.get("product_ref")
                or f"name:{product.get('name') or ''}"
            )
            not in displayed_product_refs
        ]
        if not products:
            continue
        lines.extend(_format_search_group(group, products, index=index))
        displayed_product_refs.update(
            str(
                product.get("product_ref")
                or f"name:{product.get('name') or ''}"
            )
            for product in products
        )
        displayed_names.update(str(product["name"]) for product in products)
    if lines and not lines[-1]:
        lines.pop()
    return lines, displayed_names



def _search_scope_is_complete(result: Any, *, request_id: str) -> bool:
    """Return whether a current-turn search declared its requested scope complete."""

    return any(
        _message_type(message) == "tool"
        and SEARCH_SCOPE_COMPLETE_PREFIX
        in _content_to_text(_value(message, "content"))
        for message in _current_turn_messages(_result_messages(result), request_id)
    )



def _search_guidance_evidence(result: Any, *, request_id: str) -> list[str]:
    """Return bounded pre-search shopper guidance from successful searches."""

    guidance: list[str] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message)
        if not payload or payload.get("outcome") != "results":
            continue
        text = _scrub_internal_shopper_language(
            str(payload.get("shopper_guidance") or "")
        ).strip()
        if text and text not in guidance:
            guidance.append(text)
    return guidance



def _confirmed_search_filter_groups(
    result: Any,
    *,
    request_id: str,
    displayed_names: set[str] | None = None,
) -> list[dict[str, list[str]]]:
    """Keep canonical filters scoped to products from the same search."""

    groups: list[dict[str, list[str]]] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message)
        if not payload or payload.get("outcome") != "results":
            continue
        for filters, products in _products_by_confirmed_filters(payload):
            statements: list[str] = []
            for name, value in filters.items():
                statement = _format_filter_statement(name, value)
                if statement:
                    statements.append(statement)
            if not statements:
                continue
            product_names = [
                product["name"]
                for product in products
                if product.get("name")
            ]
            if displayed_names is not None:
                product_names = [
                    name for name in product_names if name in displayed_names
                ]
                if not product_names:
                    continue
            groups.append(
                {"product_names": product_names, "statements": statements}
            )
    return groups



def _product_result_price(product: dict[str, Any]) -> str:
    price = product.get("price")
    if isinstance(price, dict):
        amount = price.get("amount")
        currency = str(price.get("currency") or "USD")
    else:
        amount = price
        currency = "USD"
    if not isinstance(amount, (int, float)):
        return ""
    return f"${float(amount):.2f} {currency}"



def _scope_relation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the scope-relation record from typed fields.

    The relation is a constant whenever an advertised parent was substituted,
    so presence of the parent is the whole condition.
    """

    category = payload.get("advertised_category")
    requested = payload.get("requested_product_type")
    if category and requested:
        return {
            "relation": "model_selected_parent_category",
            "requested_product_type": str(requested),
            "advertised_category": str(category),
        }
    if payload.get("composed_role") and requested:
        return {
            "relation": "model_composed_role",
            "requested_product_type": str(requested),
            "role_advertised_types": [
                str(value)
                for value in (payload.get("role_advertised_types") or [])
            ],
        }
    return {}



def _scrub_internal_shopper_language(text: str) -> str:
    scrubbed = text or ""
    for internal, replacement in _INTERNAL_SHOPPER_REPLACEMENTS:
        scrubbed = scrubbed.replace(internal, replacement)
    return scrubbed


def _products_found_receipt(state: Any) -> str:
    """Answer from the products this turn actually found, or "" if none.

    A turn that fetched a Cancun forecast, had one search refused and its retry
    succeed, then ran out of budget before writing anything, told the shopper
    "I could not complete that shopping request. Please try again." The work was
    done and thrown away, and the shopper was asked to pay for it twice.

    This is not a reply the assistant composed -- it names what was found and
    nothing more, because everything that would need judgement is exactly what
    there was no budget left to do.
    """

    products = [
        record
        for record in (getattr(state, "product_results", None) or [])
        if isinstance(record, dict) and record.get("display_name")
    ]
    if not products:
        return ""
    lines = ["Here is what I found before I ran out of time on this request:"]
    seen: set[str] = set()
    for record in products:
        name = str(record.get("display_name"))
        if name in seen:
            continue
        seen.add(name)
        price = record.get("price")
        amount = (
            f" -- {price.get('amount')} {price.get('currency')}"
            if isinstance(price, dict) and price.get("amount") is not None
            else ""
        )
        lines.append(f"- {name}{amount}")
        if len(seen) >= 6:
            break
    lines.append(
        "Ask me about any of these, or say what to change and I will search "
        "again."
    )
    return "\n".join(lines)


def _committed_effect_receipt(
    effects: list[dict[str, Any]],
    cart: Cart | None,
) -> str:
    """Tell the shopper exactly what was committed before the turn failed."""

    lines = [
        "Something went wrong finishing that request, but a cart change was "
        "already applied:",
        "",
    ]
    for effect in effects:
        operation = str(effect.get("operation") or "changed")
        target = str(
            effect.get("product_id") or effect.get("cart_line_id") or "an item"
        )
        quantity = effect.get("quantity")
        detail = f" (quantity {quantity})" if isinstance(quantity, int) else ""
        lines.append(f"- {operation}: {target}{detail}")
    lines.append("")
    if cart is not None:
        lines.append(_format_cart(cart))
        lines.append("")
    lines.append(
        "Please review your cart before retrying so the change is not applied "
        "twice."
    )
    return "\n".join(lines)


def _has_grounding_authority(state: State, current_evidence: str) -> bool:
    """Return whether this turn has any authority to check a draft against.

    Every turn hydrates memory lanes before the model runs. Gating the grounding
    editor on current-turn *tool* evidence alone discards that hydrated context:
    a follow-up or styling turn grounded in the historical product index or the
    authoritative cart would skip grounding entirely, leaving the draft free to
    assert product facts nothing supports.

    Dialogue is deliberately excluded. It establishes shopper intent, never
    product, policy, inventory, or cart fact, so it is not something a product
    claim can be checked against.
    """

    return bool(
        current_evidence
        or state.historical_product_sets
        or state.cart.contents
    )


def _has_search_only_tool_evidence(result: Any, *, request_id: str) -> bool:
    """Return whether current-turn commerce evidence contains only searches."""

    tool_names: list[str] = []
    has_search_result = False
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        returned_results = (evidence_of(message) or {}).get("outcome") == "results"
        if not name and returned_results:
            name = "search_catalog_tool"
        if name == SKILL_ACTIVATION_TOOL_NAME:
            continue
        tool_names.append(name)
        if name == "search_catalog_tool" and returned_results:
            has_search_result = True
    return (
        has_search_result
        and set(tool_names) == {"search_catalog_tool"}
    )


def _media_failure_response(media_analysis: str) -> str:
    detail = "video/image understanding is unavailable for this turn"
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
        if summary:
            detail = _clean_media_failure_detail(summary)

    return (
        f"I could not analyze the attached media because {detail}. "
        "Please describe the item in text, such as color, silhouette, material, "
        "and any visible details, and I can search the catalog from that description."
    )


def _clean_media_failure_detail(summary: str) -> str:
    detail = summary.strip()
    for prefix in (
        "Media was attached, but ",
        "Video was attached, but ",
        "Image was attached, but ",
    ):
        if detail.startswith(prefix):
            detail = detail[len(prefix):]
            break
    if detail:
        detail = detail[0].lower() + detail[1:]
    return detail.rstrip(". ") or "video/image understanding is unavailable for this turn"


def _in_presentation_order(
    products: list[dict[str, Any]],
    reply: str,
    groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The shown products, ordered as the reply presents them.

    The cards and the words are the same list to a shopper, so "the second one"
    has to mean one product. Left alone, the cards would follow the catalog's
    ranking and the sentences whatever the model wrote.

    The order is settled once, here, where the reply and the products are both
    in hand -- so every consumer downstream renders one order rather than
    each choosing its own.

    Within a group, never across them. A shopper asked for dresses and shoes
    sees two headed lists, and a reply that discusses a shoe before finishing
    with the dresses would otherwise lift that shoe into the dresses. The
    groups keep the order they were asked for; only the products inside one
    are sorted by where the reply names them.
    """

    if not groups:
        return _as_the_reply_names_them(products, reply)
    held: dict[str, dict[str, Any]] = {
        str(product.get("product_id") or ""): product
        for product in products
        if str(product.get("product_id") or "")
    }
    ordered: list[dict[str, Any]] = []
    placed: set[str] = set()
    for group in groups:
        members = []
        for product_id in group.get("product_ids") or []:
            product = held.get(str(product_id))
            if product is not None and str(product_id) not in placed:
                members.append(product)
                placed.add(str(product_id))
        ordered.extend(_as_the_reply_names_them(members, reply))
    # A product no group claimed still belongs on the screen. The name lookup
    # puts products in front of the shopper without going through a scope, so
    # this is not the empty case it looks like.
    ordered.extend(
        product
        for product in products
        if str(product.get("product_id") or "") not in placed
    )
    return ordered


def _as_the_reply_names_them(
    products: list[dict[str, Any]],
    reply: str,
) -> list[dict[str, Any]]:
    """One list, sorted by where the reply first names each product.

    This looks for the exact display names the service itself produced. It reads
    nothing else out of the reply, and decides nothing but sequence: a product
    the reply never names keeps its ranking, after the ones it does.
    """

    if not products or not reply:
        return products
    mentioned: list[tuple[int, int, dict[str, Any]]] = []
    unmentioned: list[tuple[int, dict[str, Any]]] = []
    for rank, product in enumerate(products):
        name = str(product.get("display_name") or "")
        at = reply.find(name) if name else -1
        if at >= 0:
            mentioned.append((at, rank, product))
        else:
            unmentioned.append((rank, product))
    # Rank breaks ties, so two products named in the same breath keep the
    # catalog's order between them.
    mentioned.sort(key=lambda item: (item[0], item[1]))
    return [product for _at, _rank, product in mentioned] + [
        product for _rank, product in unmentioned
    ]


def _images_in_product_order(
    images: dict[str, str],
    products: list[dict[str, Any]],
) -> dict[str, str]:
    """The image map, following the product order, keeping every entry.

    The cards render from this map, so it has to agree with the list beside it.
    Anything it holds that the products do not name is kept at the end rather
    than dropped: it was shown, and losing it would remove a card rather than
    move one.
    """

    if not images:
        return images
    named = [
        str(product.get("display_name") or "")
        for product in products
        if str(product.get("display_name") or "") in images
    ]
    seen = set(named)
    return {
        **{name: images[name] for name in named},
        **{name: url for name, url in images.items() if name not in seen},
    }


# Must not invite a retry. This path is reached after the turn's tools have
# already run, so a cart change may have completed; retrying duplicates it.
_GROUNDING_FAILURE_RESPONSE = (
    "I ran into a problem writing that reply. Ask me what's in your cart to see "
    "where things stand -- any change I made will show there."
)
