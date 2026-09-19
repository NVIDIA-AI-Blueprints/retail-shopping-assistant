"""What the turn may claim, and the evidence behind it.

Lifted out of `turn_support.py` unchanged.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import json
from typing import Any

from .message_shape import (
    _content_to_text,
    _current_turn_messages,
    _result_messages,
    _value,
)
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_TOOL_NOT_GRANTED,
)
from .tool_evidence import (
    detail_evidence_of,
    evidence_of,
)
from .tool_loop_control import (
    SEARCH_VALIDATION_ERROR_PREFIX,
)

_NO_DIRECT_CATALOG_MATCH_EVIDENCE = (
    "CUSTOMER_SAFE_NO_MATCH_EVIDENCE: The active catalog has no direct "
    "advertised taxonomy match for this requested product role. "
    "No retrieval ran and no alternative product type was selected. Say "
    "that plainly, preserve any successful evidence for other roles, and "
    "ask permission before searching a different advertised type. Do not "
    "name alternatives."
)



_PRODUCT_DETAIL_EVIDENCE_NOTE = (
    "Product details were read for these products, but the available "
    "detail data contains only the listed facts. Do "
    "not state material, care, dimensions, closures, fit, sizing, "
    "colorways, or outdoor performance unless the field appears in "
    "this evidence summary."
)



def _collect_tool_grounding_evidence(
    result: Any,
    *,
    max_chars: int,
    request_id: str | None = None,
) -> str:
    messages = _result_messages(result)
    if request_id is not None:
        messages = _current_turn_messages(messages, request_id)
    return _collect_message_grounding_evidence(messages, max_chars=max_chars)



def _collect_message_grounding_evidence(
    messages: list[Any],
    *,
    max_chars: int,
) -> str:
    """Collect customer-safe evidence from actual tool-role messages."""

    parts: list[str] = []
    for message in messages:
        content = _content_to_text(_value(message, "content"))
        if not content:
            continue
        if not _is_tool_evidence_message(message, content):
            continue
        parts.append(_customer_safe_tool_evidence(content, message))

    evidence = "\n\n---\n\n".join(parts).strip()
    if len(evidence) <= max_chars:
        return evidence
    return evidence[-max_chars:]



def _customer_safe_search_evidence(payload: dict[str, Any]) -> str:
    """Build the composer summary from typed evidence, without parsing prose."""

    taxonomy = payload.get("taxonomy") or {}
    confirmed_filters = payload.get("confirmed_filters") or {}
    if payload.get("outcome") == "no_direct_catalog_match":
        return _NO_DIRECT_CATALOG_MATCH_EVIDENCE
    if payload.get("outcome") == "zero_results":
        lines = [
            (
                "CUSTOMER_SAFE_SCOPED_NO_MATCH_EVIDENCE: Zero products matched "
                "only the exact advertised search scope below. This does not "
                "establish that a different, unsearched, or unadvertised product "
                "type is absent, and it does not support a catalog-wide "
                "availability claim."
            )
        ]
        if taxonomy:
            lines.append(
                "ADVERTISED_SEARCH_TAXONOMY: "
                + json.dumps(taxonomy, sort_keys=True)
            )
        if confirmed_filters:
            lines.append(
                "CONFIRMED_SEARCH_FILTERS: "
                + json.dumps(confirmed_filters, sort_keys=True)
            )
        relation = _scope_relation_line(payload, has_products=False)
        if relation:
            lines.append(relation)
        # Zero results told the model what was absent and nothing about what
        # was present, so it asked. "No green dress in a size 2 -- would you
        # like size 4 instead?" showed nothing, on a turn where the catalog
        # held plenty of size 2 dresses in other colours. A shopper asked to
        # choose between two things they cannot see has been given less than
        # nothing.
        #
        # This used to be answered by running the search again here, without
        # the optional filters, and handing the results over. That retry kept
        # only the size and dropped the product type with everything else, so
        # "a tote bag in a size 8" searched the whole catalog for size 8 --
        # which bags, being one size, are excluded from -- and four boots and
        # heels came back and were registered under a reply about tote bags.
        # Across every zero-result turn in the suite the model had already
        # issued the correct retry itself, keeping the garment and dropping the
        # colour, so the second search only ever added what the reply disowned.
        # What it knew that an instruction did not is said here instead.
        lines.append(
            "NEXT: nothing in the catalog matched all of these at once. Search "
            "again yourself, now, with one optional requirement dropped -- "
            "colour, pattern, style or price. Keep the product type and keep "
            "the size: a shopper asking for a dress is not answered with a "
            "skirt, and a size is a fact about a body, not a preference. Then "
            "show what that finds and say plainly which requirement could not "
            "be met. If the size is the only requirement there is, drop the "
            "size instead and search again: say first that nothing comes in "
            "the size they asked for, name the sizes these do come in -- one "
            "size, or a range that excludes theirs -- and never present them "
            "as the size they asked for. If the product type itself is one "
            "this shop does not carry, say that instead and search no "
            "further. Do not answer with a question alone, and never offer a "
            "choice between things the shopper cannot see: asking \"shall I "
            "show you the ones we do have\" is that refusal wearing a "
            "question mark. Show them."
        )
        return "\n".join(lines)

    lines = [_summarize_typed_product_evidence(payload)]
    unconfirmed = payload.get("unconfirmed_requirements") or []
    if unconfirmed:
        # Retrieval ranked on these; no filter enforced them. Saying so is what
        # makes running the search safe instead of a licence to claim a match.
        lines.append(
            "UNCONFIRMED_REQUIREMENTS: The catalog cannot filter on "
            + ", ".join(str(item) for item in unconfirmed)
            + ". These products were ranked for it but none is confirmed to "
            "meet it. Present them as candidates and say plainly that it is "
            "unconfirmed. Do not refuse the request."
        )
    size = _size_the_scope_has_not_line(payload)
    if size:
        lines.append(size)
    relation = _scope_relation_line(payload, has_products=True)
    if relation:
        lines.append(relation)
    audience = _assumed_audience_line(payload)
    if audience:
        lines.append(audience)
    return "\n".join(lines)



def _scope_relation_line(payload: dict[str, Any], *, has_products: bool) -> str:
    """Say plainly that a broader advertised parent was searched, if it was."""

    category = payload.get("advertised_category")
    requested = payload.get("requested_product_type")
    if not category or not requested:
        return _composed_role_line(payload, has_products=has_products)
    if not has_products:
        return (
            f"REQUESTED_SCOPE_RELATION: {requested} is not separately "
            f"advertised. The broader advertised category {category} returned "
            "zero products for this search, so do not claim that the requested "
            "type is absent from the whole catalog."
        )
    return (
        f"REQUESTED_SCOPE_RELATION: {requested} is not separately "
        f"advertised. The search used the broader advertised category {category}. "
        "Present these as closest options and keep every returned product's "
        "actual catalog category; do not relabel them as the requested type."
    )



def _composed_role_line(payload: dict[str, Any], *, has_products: bool) -> str:
    """Say that this role was the assistant's idea, not the shopper's.

    With products, naming the searched types adds nothing the shopper cannot
    read off the products themselves, and four such lines in a four-role look
    bury the answer. With none, naming them is the whole point: a miss inside
    two of five advertised types is not the role being unavailable, and that is
    exactly the claim a composer will otherwise make.
    """

    requested = payload.get("requested_product_type")
    if not payload.get("composed_role") or not requested:
        return ""
    types = [
        str(value) for value in (payload.get("role_advertised_types") or [])
    ]
    if not has_products:
        searched = ", ".join(sorted(types)) or "the pieces looked at"
        return (
            f"REQUESTED_SCOPE_RELATION: the shopper did not ask for {requested}; "
            "this role was proposed by the assistant. Nothing came back for "
            f"{searched}. Name those pieces the way a shopper would and do not "
            "claim the role is unavailable, because only those were looked at. "
            "Speak as someone standing in the shop: never say search, filter, "
            "scope, results, catalog, or a catalog's internal label."
        )
    return (
        f"REQUESTED_SCOPE_RELATION: the shopper did not ask for {requested}; "
        "this role was proposed by the assistant. Offer it as a suggestion "
        "rather than as something they asked for, and keep every returned "
        "product's actual catalog category."
    )



def _size_the_scope_has_not_line(payload: dict[str, Any]) -> str:
    """Say the asked size does not exist here, and name the run that does.

    The size was dropped before the search or there would be nothing to show.
    Left unsaid, that is an invitation to fill the gap: "do you have a tote bag
    in a size 8" was answered "the tote bags we carry come in sizes 2, 4, 6 and
    10", a size run belonging to dresses and to no bag in the catalog.

    So the catalog's own values travel with the products, and the sentence the
    reply owes the shopper is stated rather than left to be worked out.
    """

    record = payload.get("size_the_scope_has_not") or {}
    if not isinstance(record, dict):
        return ""
    asked = str(record.get("asked") or "").strip()
    comes_in = [str(value) for value in (record.get("comes_in") or []) if str(value).strip()]
    if not asked or not comes_in:
        return ""
    run = (
        "one size"
        if comes_in == ["onesize"]
        else ", ".join(value for value in comes_in if value != "onesize")
    )
    return (
        f"SIZE_THE_SCOPE_HAS_NOT: nothing here is made in size {asked}, so the "
        "size was dropped and these are what the scope holds. Say that first, "
        f"in a shopper's words, and name what these do come in: {run}. These "
        "sizes are the catalog's -- state no others, and never present a piece "
        f"as size {asked}. This is an answer, not a dead end: the products are "
        "below, so show them rather than asking whether to."
    )



def _assumed_audience_line(payload: dict[str, Any]) -> str:
    """Tell the shopper who these pieces are for, since nobody said.

    This does not belong to composed roles, though it was attached to them
    first. A shopper who asks in their own words for a "work casual outfit"
    names the role themselves, so nothing is composed -- and a catalog that is
    almost entirely womenswear hands them womenswear anyway, silently. That
    turn is exactly the one that needs the sentence.
    """

    audience = [str(value) for value in (payload.get("assumed_audience") or [])]
    if not audience:
        return ""
    return (
        "ASSUMED_AUDIENCE: nobody said who these pieces are for; every one "
        "that came back is for " + ", ".join(sorted(audience)) + ". Open by "
        "naming that as what you have assumed the shopper is looking for, in "
        "words that fit what they actually asked for -- \"assuming you're "
        "looking for women's dresses\", \"assuming these are for you\". Never "
        "describe a bag, a pair of sunglasses or a bracelet as something worn "
        "-- a reply about tote bags opened by calling them things to put on "
        "-- and invite them to "
        "correct it. One clause, then get on with the answer. It is an "
        "assumption about what they want, not a note about the shop's style "
        "or about what this reply happened to return. Never turn it into a "
        "question, or a guess, about who the shopper is or who they are "
        "buying for: a shopper who says \"I need something to wear\" has "
        "already said it is for them, and being asked whether they are "
        "shopping for someone else reads as not listening. Say nothing about "
        "which pieces suit a wider audience -- that is how the catalog tags "
        "its bags, not something anyone asked. Put it in a shopper's words, "
        "never a catalog label."
    )



#: Appended to every catalog evidence summary the composer reads. The labels
#: above it are internal -- taxonomy, confirmed filters, scope outcomes -- and a
#: model handed them will paraphrase them straight back. One live reply read "I
#: checked the broader apparel category with the adult all-genders filter, and
#: that search returned no matches under that scope", which is the evidence
#: block read aloud. A shopper is standing in a shop, not in front of a query
#: planner.
_SHOPPER_VOICE_NOTE = (
    "SPEAK AS A SHOP ASSISTANT: everything above is internal bookkeeping. Say "
    "what you looked at and what you found in the shopper's own words. Never "
    "say search, filter, scope, taxonomy, query, results, or catalog, and "
    "never repeat an internal label such as adult_all_genders -- say pieces "
    "anyone can wear. Name product types and prices plainly; those are the "
    "shopper's language already."
)



def _customer_safe_tool_evidence(content: str, message: Any = None) -> str:
    """Summarise one tool result for the composer.

    Catalog evidence is read from the typed payload on the message artifact.
    The text branches that follow handle results which carry no payload:
    framework-generated tool errors, and tools that emit no evidence yet.
    Nothing here parses catalog facts back out of prose.
    """

    if message is not None:
        payload = evidence_of(message)
        if payload is not None:
            return (
                _customer_safe_search_evidence(payload)
                + "\n\n"
                + _SHOPPER_VOICE_NOTE
            )
        detail = detail_evidence_of(message)
        if detail is not None:
            return _render_product_evidence_summary(
                detail.get("products") or [],
                heading="CUSTOMER_SAFE_PRODUCT_DETAIL_EVIDENCE",
                note=_PRODUCT_DETAIL_EVIDENCE_NOTE,
            )

    if content.startswith(SEARCH_VALIDATION_ERROR_PREFIX):
        return (
            "CUSTOMER_SAFE_INVALID_SEARCH_EVIDENCE: No valid catalog search "
            "scope was established and no retrieval ran. This does not support "
            "a product-availability or catalog-absence claim."
        )
    return _summarize_cart_evidence(content)



def _render_product_evidence_summary(
    products: list[dict[str, Any]],
    *,
    heading: str,
    note: str,
    confirmed_filters: dict[str, Any] | None = None,
    taxonomy_scope: dict[str, Any] | None = None,
    fallback_content: str | None = None,
) -> str:
    """Render the composer's product summary from records.

    Shared by the typed-payload path and the text path so the two cannot drift
    in wording; only where the records come from differs.
    """

    lines = [f"{heading}: {note}"]
    if confirmed_filters:
        lines.append(
            "CONFIRMED_SEARCH_FILTERS: Every product below passed each filter "
            "predicate. A one-value list confirms that value; a multi-value list "
            "confirms only membership in the set, not which value matched: "
            + json.dumps(confirmed_filters, sort_keys=True, default=str)
        )
    if taxonomy_scope:
        lines.append(
            "ADVERTISED_SEARCH_TAXONOMY: This search used only these advertised "
            "taxonomy values. Lists are inclusive scopes; they do not mean every "
            "product has every value. Do not describe an unlisted product type "
            "as advertised: "
            + json.dumps(taxonomy_scope, sort_keys=True, default=str)
        )
    if not products:
        # The text path keeps emitting its stripped line even when that line is
        # empty, so its output is unchanged. The typed path has no prose to fall
        # back to and must not append a blank line in its place.
        if fallback_content is None:
            return "\n".join(lines)
        return "\n".join(
            lines + [_strip_internal_ids_from_evidence_line(fallback_content)]
        )
    for product in products:
        summary_parts = [product["name"]]
        if product.get("category"):
            summary_parts.append(f"category: {product['category']}")
        if product.get("price"):
            summary_parts.append(f"price: {product['price']}")
        if product.get("image_url"):
            summary_parts.append("image: available")
        attributes = product.get("attributes")
        if isinstance(attributes, dict) and attributes:
            summary_parts.append(
                "confirmed: "
                + "; ".join(
                    f"{name.replace('_', ' ')}: {value}"
                    for name, value in attributes.items()
                )
            )
        if product.get("details"):
            summary_parts.append("details: " + "; ".join(product["details"]))
        lines.append("- " + " | ".join(summary_parts))
    return "\n".join(lines)



def _summarize_typed_product_evidence(payload: dict[str, Any]) -> str:
    """Summarise search results for the composer straight from typed evidence."""

    products = payload.get("products")
    return _render_product_evidence_summary(
        products if isinstance(products, list) else [],
        heading="CUSTOMER_SAFE_SEARCH_EVIDENCE",
            note=(
                "Search results support product names, prices, categories, "
                "image availability, confirmed search filters, any attribute "
                "listed as confirmed for that specific product, and a modest "
                "styling role. An attribute confirmed for one product is not "
                "evidence about another. They do not support care, "
                "construction, fit, comfort, weather, grass, gravel, heat, or "
                "best-in-category claims, nor any attribute not listed for that "
                "product. Treat names as display names, not attribute evidence; "
                "group claims require the attribute confirmed on every item."
            ),
        confirmed_filters=payload.get("confirmed_filters") or {},
        taxonomy_scope=payload.get("taxonomy") or {},
    )



def _summarize_cart_evidence(content: str) -> str:
    lines = ["CUSTOMER_SAFE_CART_EVIDENCE:"]
    for raw_line in content.splitlines():
        cleaned = _strip_internal_ids_from_evidence_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)



def _strip_internal_ids_from_evidence_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("PRODUCT_REF:") or stripped.startswith("CART_LINE_ID:"):
        return ""
    if stripped.startswith("- CART_LINE_ID:") and "|" in stripped:
        return "- " + stripped.split("|", 1)[1].strip()

    marker = "(PRODUCT_REF:"
    while marker in stripped:
        start = stripped.find(marker)
        end = stripped.find(")", start)
        if end == -1:
            stripped = stripped[:start].rstrip()
            break
        stripped = (stripped[:start] + stripped[end + 1 :]).strip()
    return stripped



def _is_tool_evidence_message(message: Any, content: str) -> bool:
    message_type = str(_value(message, "type") or "").lower()
    role = str(_value(message, "role") or "").lower()
    tool_name = str(_value(message, "name") or "")
    if tool_name == SKILL_ACTIVATION_TOOL_NAME:
        return False
    if content.startswith(
        (
            SKILL_ACTIVATION_COMPLETE,
            SKILL_ACTIVATION_REQUIRED,
            SKILL_TOOL_NOT_GRANTED,
            "SHOPPER_SKILL_ACTIVATION_FAILED:",
        )
    ):
        return False
    return message_type == "tool" or role == "tool"


