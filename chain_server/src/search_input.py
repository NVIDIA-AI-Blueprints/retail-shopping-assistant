# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shaping a search request: its scope, taxonomy constraints and shopper guidance."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel
from shared.commerce_contracts import CatalogCapabilities

_UNSUPPORTED_SEARCH_MODE_MESSAGE = (
    "The requested search mode is not available for the active catalog. "
    "Ask the shopper to use an advertised mode."
)


#: Media-analysis fields that count as the shopper speaking.
#:
#: A photo is a statement. When a shopper attaches one and says "I like the
#: top", the garment, its colour and its fabric came from them via the camera --
#: refusing those as model-invented refuses the shopper's own words.
#:
#: Deliberately excludes `style_terms`, `occasion`, `search_queries`,
#: `uncertainties` and `safety_notes`. Those are the model's reading of the
#: image, not its content, and authorising them would let "boho-chic" become a
#: shopper-stated requirement.
_STATED_MEDIA_FIELDS = ("fashion_items", "colors", "materials_or_textures")


def stated_media_terms(media_analysis: str) -> str:
    """Return the media-analysis words that count as shopper-stated."""

    if not media_analysis:
        return ""
    try:
        parsed = json.loads(media_analysis)
    except (TypeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    words: list[str] = []
    for name in _STATED_MEDIA_FIELDS:
        value = parsed.get(name)
        # The VLM is not type-stable: the same key comes back as a string on one
        # turn and a list on the next, so both are accepted rather than trusted.
        if isinstance(value, str):
            words.append(value)
        elif isinstance(value, list):
            words.extend(str(item) for item in value)
    return " ".join(words)


def _generic_shopper_guidance(requested_product_type: str | None) -> str:
    """Return safe guidance after an inferred attribute is removed."""

    product_type = (requested_product_type or "products").replace("_", " ")
    return f"Finding {product_type} for the shopper's request."


_UNSUPPORTED_GUIDANCE_PATTERN = re.compile(
    r"\b(?:waterproof|water[ -]?resistant|weather[ -]?safe|bug[ -]?safe|"
    r"wet (?:surfaces?|grounds?|conditions?)|grass|gravel|all[ -]?day|best[ -]?in[ -]?category|"
    r"maximally|outdoor (?:surfaces?|walking)|"
    r"(?:handles?|suitable for|works? well (?:for|in)|secure for|stay secure for)"
    r"[^.]{0,32}(?:rain|wet weather|outdoor))\b",
    flags=re.IGNORECASE,
)


def _safe_shopper_guidance(
    shopper_guidance: str,
    requested_product_type: str | None,
) -> str:
    """Remove unsupported performance language from pre-search guidance."""

    if _UNSUPPORTED_GUIDANCE_PATTERN.search(shopper_guidance):
        return _generic_shopper_guidance(requested_product_type)
    return shopper_guidance


def _unsupported_requirement_message(requirements: list[str]) -> str:
    """Return the shopper-facing failure for unsupported hard requirements."""

    return (
        "The requested catalog requirement cannot be enforced: "
        + ", ".join(
            f"'{requirement}' is not an advertised hard filter"
            for requirement in requirements
        )
        + ". Decide from what the shopper has already told you: if they have "
        "said what to do when it cannot be confirmed, follow that; otherwise "
        "treat it as a ranking preference and say plainly it is unconfirmed, "
        "or ask them. Do not refuse the request."
    )


def _advertised_subcategories_for_selection(
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> list[str]:
    """Return advertised role choices within the selected category."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected_categories = payload.get("category") or []
    categories = capabilities.taxonomy.categories
    category_names = selected_categories or list(categories)
    return sorted(
        {
            subcategory
            for category_name in category_names
            if (category := categories.get(category_name)) is not None
            for subcategory in category.subcategories
        }
    )


def _selected_advertised_subcategories(
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[str, list[str]] | None:
    """Return one typed multi-subcategory selection owned by one category."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected = list(dict.fromkeys(payload.get("subcategory") or []))
    if len(selected) < 2:
        return None
    owners = [
        category_name
        for category_name, category in capabilities.taxonomy.categories.items()
        if all(value in category.subcategories for value in selected)
    ]
    selected_categories = set(payload.get("category") or [])
    if len(owners) != 1 or selected_categories not in (set(), {owners[0]}):
        return None
    return owners[0], selected


def _multi_subcategory_candidate_limit(
    selection: tuple[str, list[str]] | None,
    capabilities: CatalogCapabilities,
    default: int,
) -> int:
    """Fetch enough ranked candidates for a typed multi-subcategory selection."""

    if selection is None:
        return default
    category_name, subcategories = selection
    category = capabilities.taxonomy.categories[category_name]
    product_count = sum(
        category.subcategories[name].product_count for name in subcategories
    )
    return min(50, max(default, len(subcategories), product_count))


def _taxonomy_hard_constraints(
    selection: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[dict[str, list[str]], list[str]]:
    """Map generic taxonomy roles to catalog-owned hard-filter field names."""

    raw = selection.model_dump() if isinstance(selection, BaseModel) else selection
    categories = sorted(set(raw.get("category", [])), key=str.casefold)
    subcategories = sorted(set(raw.get("subcategory", [])), key=str.casefold)
    taxonomy = capabilities.taxonomy
    issues: list[str] = []

    for category in categories:
        if category not in taxonomy.categories:
            issues.append(f"category '{category}' is not advertised")

    owners: dict[str, list[str]] = {
        subcategory: sorted(
            [
                category_name
                for category_name, category in taxonomy.categories.items()
                if subcategory in category.subcategories
            ],
            key=str.casefold,
        )
        for subcategory in subcategories
    }
    for subcategory, subcategory_owners in owners.items():
        if not subcategory_owners:
            issues.append(f"subcategory '{subcategory}' is not advertised")
        elif categories and not set(categories).intersection(subcategory_owners):
            issues.append(
                f"subcategory '{subcategory}' is not available in selected categories"
            )

    if categories and subcategories:
        for category in categories:
            advertised = taxonomy.categories.get(category)
            if advertised is None:
                continue
            if not set(subcategories).intersection(advertised.subcategories):
                issues.append(
                    f"category '{category}' has no selected subcategory"
                )

    effective_categories = categories
    if subcategories and not categories:
        inferred_categories = sorted(
            {
                owner
                for subcategory_owners in owners.values()
                for owner in subcategory_owners
            },
            key=str.casefold,
        )
        if len(inferred_categories) > 1:
            issues.append(
                "subcategory selection has multiple owning categories; select "
                "exactly one advertised category"
            )
            effective_categories = []
        else:
            effective_categories = inferred_categories

    constraints: dict[str, list[str]] = {}
    if effective_categories:
        if taxonomy.category_field:
            constraints[taxonomy.category_field] = effective_categories
        else:
            issues.append("the catalog does not advertise a category filter field")
    if subcategories:
        if taxonomy.subcategory_field:
            constraints[taxonomy.subcategory_field] = subcategories
        else:
            issues.append("the catalog does not advertise a subcategory filter field")
    return constraints, issues


def _catalog_search_scope(
    taxonomy: dict[str, list[str]],
    required_constraints: dict[str, Any],
) -> dict[str, Any]:
    """Return the normalized hard-filter identity for one catalog search."""

    return {
        "taxonomy": _normalized_scope_value(taxonomy),
        "required_constraints": _normalized_scope_value(required_constraints),
    }


def _normalized_scope_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized_scope_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        normalized = [_normalized_scope_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None
