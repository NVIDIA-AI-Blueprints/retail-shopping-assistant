"""Turning what a search found into what the shopper is told.

Lifted out of `turn_support.py` unchanged.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import json
from typing import Any

from .agenttypes import State
from .message_shape import (
    _content_to_text,
    _current_turn_messages,
    _message_type,
    _result_messages,
    _value,
)
from .response_format import (
    _format_filter_statement,
    _format_search_group,
)
from .tool_evidence import (
    evidence_of,
)
from .tool_loop_control import (
    SEARCH_SCOPE_COMPLETE_PREFIX,
    UNSUPPORTED_CONSTRAINT_PREFIX,
)

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


