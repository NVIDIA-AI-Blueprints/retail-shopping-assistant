"""Lining a shopper's words up with the catalog's words.

Lifted out of `turn_support.py` unchanged.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from pydantic import (
    BaseModel,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    ProductSummary,
)

from .agenttypes import DialogueTurn


def _singularize_product_word(word: str) -> str:
    """Conservatively singularize one normalized product-type word."""

    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word



def _normalize_product_text(value: str) -> str:
    """Normalize product-type text for deterministic lexical comparison."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("&", " and ").replace("/", " or ")
    words = re.findall(r"[^\W_]+", normalized.replace("_", " "))
    return " ".join(_singularize_product_word(word) for word in words)



def _has_alternative_connector(value: str) -> bool:
    """Return whether a product phrase joins explicit alternatives."""

    normalized = _normalize_product_text(value)
    return bool(re.search(r"\b(?:and|or)\b", normalized))



def _text_mentions_product_type(text: str, product_type: str) -> bool:
    """Return whether text names a normalized product-type form."""

    normalized_text = _normalize_product_text(text)
    padded_text = f" {normalized_text} "
    normalized_product_type = _normalize_product_text(product_type)
    return bool(normalized_product_type) and (
        f" {normalized_product_type} " in padded_text
    )



def _requirement_word_stem(word: str) -> str:
    """Return a conservative stem for literal requirement provenance."""

    for suffix in ("ance", "ence", "ancy", "ency", "ant", "ent", "ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[: -len(suffix)]
    return _singularize_product_word(word)



def _shopper_stated_requirement(query: str, requirement: str) -> bool:
    """Return whether a proposed requirement is grounded in the current turn."""

    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    query_words = {
        _requirement_word_stem(word)
        for word in re.findall(r"[^\W_]+", normalized_query)
    }
    requirement_words = {
        _requirement_word_stem(word)
        for word in re.findall(
            r"[^\W_]+",
            unicodedata.normalize("NFKC", requirement).casefold(),
        )
    }
    return bool(requirement_words) and requirement_words.issubset(query_words)



def a_place_the_shopper_named(
    shopper_statements: Sequence[str],
    quoted: str,
) -> bool:
    """Whether the words offered as naming the place were ever actually said.

    The tool asks the model to quote the words that named the place, and the
    model quoted "Italy" on a turn reading "it's going to snow when we get
    back" -- and, in the next run, "Rome", which the shopper never said in any
    turn. A required field it can fill with anything is a field it will fill
    with anything.

    So the citation is checked against the record, which is the same thing
    `expected_display_name` does for a product name: not what the words mean,
    only whether they were said. Reusing the constraint-provenance reader so a
    quotation is judged the same way everywhere.

    The record is every turn the shopper has spoken, not only the current one.
    Checking the current turn alone was narrower than the defect and cost the
    behaviour it was meant to protect: nine turns into planning one trip to
    one city, "will I need a jacket in the evening" names no place, so the
    call was refused -- and the reply then said no forecast was available for
    Cancun and described a typical Cancun September anyway. Earlier runs had
    fetched that forecast and cited the provider.

    What made Rome wrong is not something this function can see. The shopper
    had stated the conditions, and "when we get back" is home rather than the
    city of the trip; both are judgments about meaning, and both are stated on
    the field the quotation comes from. What a substring check can establish is
    that the words were said by the shopper at all, which is what stopped the
    invented "Rome", and that is all it claims to establish.
    """

    if not quoted.strip():
        return False
    return any(
        _shopper_stated_requirement(statement, quoted)
        for statement in shopper_statements
    )



def _product_scope_key(value: str | None) -> str:
    """Return the full normalized product phrase preserved across repair."""

    return _normalize_product_text(value or "")



def _recent_shopper_statements(
    dialogue: Sequence[DialogueTurn],
    *,
    limit: int = 4,
) -> str:
    """Read prior shopper text from the typed lane, never from rendered prose.

    Assistant text is deliberately excluded: dialogue may carry shopper intent,
    but assistant prose is not product, policy, inventory, or cart evidence.
    """

    recent = list(dialogue)[-limit:]
    return "\n".join(turn.shopper_text for turn in recent if turn.shopper_text)



def _shopper_stated_product_scope(
    query: str,
    dialogue: Sequence[DialogueTurn],
    product_scope_key: str,
) -> bool:
    """Return whether current or recent shopper text states a product scope."""

    shopper_text = "\n".join(
        value for value in (query, _recent_shopper_statements(dialogue)) if value
    )
    return _text_mentions_product_type(shopper_text, product_scope_key)



def _resolved_agent_selected_product_type(
    *,
    query: str,
    dialogue: Sequence[DialogueTurn],
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Derive open-role provenance from the agent's single taxonomy choice."""

    if taxonomy_status != "agent_selected_type":
        return requested_product_type
    scope_key = _product_scope_key(requested_product_type)
    if scope_key and _shopper_stated_product_scope(query, dialogue, scope_key):
        return requested_product_type
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    subcategories = payload.get("subcategory") or []
    if len(subcategories) == 1:
        return str(subcategories[0])
    return requested_product_type



def _agent_selected_scope_is_advertised(
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
) -> bool:
    """Check that an open-role choice names its advertised taxonomy scope."""

    requested = _normalize_product_text(requested_product_type or "")
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    subcategories = {
        _normalize_product_text(value)
        for value in (payload.get("subcategory") or [])
    }
    return len(subcategories) == 1 and requested in subcategories



def _advertised_taxonomy_value(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> str | None:
    """Return the matching advertised taxonomy value, if one exists."""

    requested = _normalize_product_text(requested_product_type or "")
    for category_name, category in capabilities.taxonomy.categories.items():
        if requested == _normalize_product_text(category_name):
            return category_name
        for subcategory_name in category.subcategories:
            if requested == _normalize_product_text(subcategory_name):
                return subcategory_name
    return None



#: Garments shoppers name that a clothing catalog may simply not stock.
#:
def _advertised_scope_match(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> tuple[str, str, str, str] | None:
    """Return the longest advertised exact or suffix scope match."""

    raw_requested = requested_product_type or ""
    requested = _normalize_product_text(raw_requested)
    suffix_match_allowed = not _has_alternative_connector(raw_requested)
    matches: list[tuple[str, str, str, str]] = []
    for category_name, category in capabilities.taxonomy.categories.items():
        normalized_category = _normalize_product_text(category_name)
        if requested == normalized_category or (
            suffix_match_allowed
            and requested.endswith(f" {normalized_category}")
        ):
            matches.append(
                ("category", category_name, category_name, normalized_category)
            )
        for subcategory_name in category.subcategories:
            normalized_subcategory = _normalize_product_text(subcategory_name)
            if requested == normalized_subcategory or (
                suffix_match_allowed
                and requested.endswith(f" {normalized_subcategory}")
            ):
                matches.append(
                    (
                        "subcategory",
                        subcategory_name,
                        category_name,
                        normalized_subcategory,
                    )
                )
    return max(
        matches,
        key=lambda match: (len(match[3].split()), len(match[3])),
        default=None,
    )



def _duplicates_unavailable_product_type(
    requirements: Any,
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> bool:
    """Return whether requirements only repeat an unavailable product type."""

    requested = _normalize_product_text(requested_product_type or "")
    return bool(
        requested
        and isinstance(requirements, list)
        and len(requirements) == 1
        and not _has_alternative_connector(requested_product_type or "")
        and _advertised_scope_match(requested_product_type, capabilities) is None
        and all(
            isinstance(requirement, str)
            and _normalize_product_text(requirement) == requested
            for requirement in requirements
        )
    )



def _same_product_scope(
    first: str,
    second: str,
    capabilities: CatalogCapabilities,
) -> bool:
    """Compare full product scopes without conflating advertised siblings."""

    if first == second:
        return True
    if _has_alternative_connector(first):
        return False
    first_advertised = _advertised_taxonomy_value(first, capabilities)
    if first_advertised:
        return False
    advertised_match = _advertised_scope_match(first, capabilities)
    if advertised_match:
        return second == advertised_match[3]
    return first.endswith(f" {second}")



def _products_with_subcategory_coverage(
    products: list[ProductSummary],
    selection: tuple[str, list[str]] | None,
    limit: int,
) -> list[ProductSummary]:
    """Keep rank order while reserving one result per selected subcategory."""

    if selection is None or len(products) <= limit:
        return products
    _, subcategories = selection
    selected_indexes: set[int] = set()
    for subcategory in subcategories:
        normalized_subcategory = _normalize_product_text(subcategory)
        match = next(
            (
                index
                for index, product in enumerate(products)
                if _normalize_product_text(product.category or "")
                == normalized_subcategory
            ),
            None,
        )
        if match is not None:
            selected_indexes.add(match)

    for index in range(len(products)):
        if len(selected_indexes) >= max(limit, len(subcategories)):
            break
        selected_indexes.add(index)
    return [products[index] for index in sorted(selected_indexes)]



def _advertised_taxonomy_scope_issue(
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> str | None:
    """Enforce capability-owned relations for exact advertised scopes."""

    advertised_match = _advertised_scope_match(
        requested_product_type,
        capabilities,
    )
    if advertised_match is None:
        return None
    scope_kind, advertised_name, category_name, _ = advertised_match
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected_categories = payload.get("category") or []
    selected_subcategories = payload.get("subcategory") or []
    normalized_category = _normalize_product_text(category_name)
    normalized_selected_categories = {
        _normalize_product_text(value) for value in selected_categories
    }
    normalized_selected_subcategories = {
        _normalize_product_text(value) for value in selected_subcategories
    }
    if scope_kind == "category":
        category = capabilities.taxonomy.categories[category_name]
        owned_subcategories = {
            _normalize_product_text(value) for value in category.subcategories
        }
        selected_owned_children = (
            bool(normalized_selected_subcategories)
            and normalized_selected_subcategories.issubset(owned_subcategories)
            and normalized_selected_categories in (set(), {normalized_category})
        )
        if taxonomy_status == "exact_requested_type" and selected_owned_children:
            return (
                f"Requested product type '{requested_product_type}' binds to "
                f"advertised category '{advertised_name}', while the selected "
                "taxonomy contains only its advertised children. Keep those "
                "children together for the shopper's umbrella request."
            )
        # Naming the category the type binds to, and nothing else, substitutes
        # nothing -- the type is "jewelry" and the selected category is
        # jewelry. It was accepted only when the shopper had said the word,
        # so a turn deriving the category itself was refused for doing exactly
        # what this refusal instructs: "select that category directly".
        #
        # Asked for the most expensive thing in the shop, the assistant read
        # the published range, went to jewelry at $269.99 -- correctly, the
        # only department that reaches it -- and was turned back.
        exact_category = (
            taxonomy_status
            in {"exact_requested_type", "agent_selected_type"}
            and not normalized_selected_subcategories
            and normalized_selected_categories == {normalized_category}
        )
        owned_children = (
            taxonomy_status
            in {"member_of_requested_umbrella", "agent_selected_type"}
            and selected_owned_children
        )
        if exact_category or owned_children:
            return None
        return (
            f"Requested product type '{requested_product_type}' binds to advertised "
            f"category '{advertised_name}'. Select that category directly or only "
            "its advertised children; do not substitute another category."
        )
    selected_matches = (
        len(selected_subcategories) == 1
        and _normalize_product_text(selected_subcategories[0])
        == _normalize_product_text(advertised_name)
        and normalized_selected_categories in (set(), {normalized_category})
    )
    if selected_matches and taxonomy_status in {
        "exact_requested_type",
        "agent_selected_type",
    }:
        return None
    return (
        f"Requested product type '{requested_product_type}' binds to advertised "
        f"subcategory '{advertised_name}'. Select only that exact subcategory; "
        "do not substitute an advertised sibling."
    )



def _catalog_execution_taxonomy_status(
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
    semantic_query: str,
    capabilities: CatalogCapabilities,
    *,
    shopper_stated_scope: bool,
) -> str:
    """Derive the legacy execution mode without asking the model to label it."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    categories = payload.get("category") or []
    subcategories = payload.get("subcategory") or []
    if (
        not semantic_query.strip()
        and not requested_product_type
        and not categories
        and not subcategories
    ):
        return "image_only"
    if not shopper_stated_scope:
        return "agent_selected_type"

    advertised_match = _advertised_scope_match(
        requested_product_type,
        capabilities,
    )
    if advertised_match is not None:
        scope_kind = advertised_match[0]
        if scope_kind == "category" and subcategories:
            return "member_of_requested_umbrella"
        return "exact_requested_type"
    if len(categories) == 1 and not subcategories:
        return "parent_category_alternative"
    if subcategories:
        return "member_of_requested_umbrella"
    return "exact_requested_type"



def _exact_taxonomy_issue(
    requested_product_type: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Return a coherence issue for an exact taxonomy claim."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    categories = payload.get("category") or []
    subcategories = payload.get("subcategory") or []
    requested_matches_category = not subcategories and any(
        _normalize_product_text(requested_product_type)
        == _normalize_product_text(category)
        for category in categories
    )
    if requested_matches_category:
        return None

    selected_product_types = subcategories or categories
    if len(selected_product_types) != 1:
        return (
            "The selected taxonomy must faithfully represent one requested type "
            "or every child of a shopper-requested umbrella."
        )
    selected_product_type = selected_product_types[0]
    if _normalize_product_text(requested_product_type) == _normalize_product_text(
        selected_product_type
    ):
        return None
    return (
        "A single taxonomy value must match requested_product_type. If no "
        "advertised value faithfully represents it, ask a clarification instead"
    )


