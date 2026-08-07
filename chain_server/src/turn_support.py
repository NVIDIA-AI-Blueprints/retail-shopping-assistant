# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn-support helpers: the stateless half of the deep-agents runtime.

`deepagents_runtime.py` held its orchestration class and every helper that class
calls in one 6,600-line file. This module is the helpers -- catalog scope
reasoning, evidence shaping, response assembly, diagnostics, and the pydantic
schemas for tool arguments. None of them touches `self`, the graph, or a live
service; each takes data and returns data, so they can be read and tested
without standing up a runtime.

They move as a single module rather than as themed ones because the call graph
between those themes is cyclic -- evidence calls scope, scope calls shared
helpers, and those call back into diagnostics and response assembly. Splitting
further means breaking those cycles, which is a design change rather than a
move, so it is left to a separate change instead of being smuggled into this one.

Moved verbatim from `deepagents_runtime.py`; no definition was edited.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from collections.abc import Sequence
from typing import Any, Literal
import unicodedata
import uuid
from langgraph.checkpoint.memory import MemorySaver
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from .agenttypes import Cart, DialogueTurn, State
from .response_format import (
    _format_cart,
    _format_detail_value,
    _format_filter_statement,
    _format_product_detail_record,
    _format_product_record,
    _format_product_refs,
    _format_search_group,
)
from .catalog_capabilities import (
    effective_filter_capabilities,
)
from .catalog_request import (
    CatalogSearchPlan,
)
from .conversation_memory import (
    ConversationEvent,
    FinalTurnStatus,
)
from .control_signals import (
    rejections_of,
)
from .tool_evidence import (
    detail_evidence_of,
    evidence_of,
)
from .message_shape import (
    _content_to_text,
    _current_turn_messages,
    _message_type,
    _result_messages,
    _tool_results_by_call_id,
    _value,
)
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
    SKILL_ACTIVATION_MULTIPLE_PRIMARY,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_TOOL_NOT_GRANTED,
)
from .tool_loop_control import (
    SEARCH_BUDGET_EXHAUSTED_PREFIX,
    CONSTRAINT_REVIEW_PREFIX,
    STOP_TOOL_USE_PREFIX,
    SEARCH_SCOPE_COMPLETE_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
    SERVER_CATALOG_CLARIFICATION,
    SERVER_RESTORED_TOOL_CALL_FIELDS,
    UNSUPPORTED_CONSTRAINT_PREFIX,
    UNSUPPORTED_TAXONOMY_PREFIX,
    _SERVER_REJECTED_TOOL_CALLS,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CommerceError,
    ProductDetail,
    ProductSummary,
)


logger = logging.getLogger(__name__)


_SHARED_CONFIG_ROOT_ENV = "SHARED_CONFIG_ROOT"


_STORE_POLICIES_RELATIVE_PATH = Path("chain_server/store_policies.yaml")


def _build_checkpointer():
    """Return the process-local LangGraph checkpointer."""

    store = os.environ.get("CHECKPOINT_STORE", "memory").strip().lower()
    if store != "memory":
        raise ValueError(
            "CHECKPOINT_STORE currently supports only 'memory'. "
            f"Received: {store!r}."
        )
    return MemorySaver()


def _store_policies_path() -> Path:
    """Resolve controlled policy content outside the agent-readable skill root."""

    configured_root = os.environ.get(_SHARED_CONFIG_ROOT_ENV, "").strip()
    if configured_root:
        return Path(configured_root) / _STORE_POLICIES_RELATIVE_PATH

    deployed_path = Path("/app/shared/configs") / _STORE_POLICIES_RELATIVE_PATH
    if deployed_path.is_file():
        return deployed_path

    return (
        Path(__file__).resolve().parents[2]
        / "shared"
        / "configs"
        / _STORE_POLICIES_RELATIVE_PATH
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


_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE = 24


_MAX_DIAGNOSTIC_PRODUCT_FACTS = 40


_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS = 500


_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE_CHARS = 32_000


_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS = 1.0


_UNSUPPORTED_SEARCH_MODE_MESSAGE = (
    "The requested search mode is not available for the active catalog. "
    "Ask the shopper to use an advertised mode."
)


_NO_DIRECT_TAXONOMY_RESPONSE = (
    "The catalog doesn't advertise a product type that directly matches this "
    "request. Would you like me to search a different advertised product type?"
)


_REJECTED_CATALOG_SEARCH_RESPONSE = (
    "I couldn't complete a valid catalog search for that request, so I don't "
    "have catalog results to show. Please try again or ask me to search a "
    "different advertised product type."
)


_CATALOG_REPAIR_CLARIFICATION_RESPONSE = (
    "Could you clarify the product type or requirement you want me to use?"
)


_UNSUPPORTED_REQUIREMENT_RESPONSE = (
    "I can't guarantee that requirement from the catalog information available "
    "to this assistant, so I won't present unverified matches. Would you like me "
    "to treat it as a preference and show candidates to verify on their product "
    "pages?"
)


_PRODUCT_NAME_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to", "with"})


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


class CatalogTaxonomyToolInput(BaseModel):
    """Catalog-derived taxonomy roles used by the agent search tool."""

    model_config = ConfigDict(extra="forbid")

    category: list[str] = Field(
        ...,
        description=(
            "Exact advertised category values required by the shopper. Use an "
            "empty list only when subcategory supplies the text-search scope or "
            "the search is image-only."
        ),
    )
    subcategory: list[str] = Field(
        ...,
        description=(
            "Exact advertised subcategory values required by the shopper. Use an "
            "empty list when category supplies the text-search scope or the search "
            "is image-only."
        ),
    )

    @field_validator("category", "subcategory", mode="before")
    @classmethod
    def deduplicate_values(cls, value: Any) -> Any:
        """Normalize repeated taxonomy values before cardinality validation."""

        if isinstance(value, str):
            return [value]
        return list(dict.fromkeys(value)) if isinstance(value, list) else value


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


def _product_scope_key(value: str | None) -> str:
    """Return the full normalized product phrase preserved across repair."""

    return _normalize_product_text(value or "")


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
        exact_category = (
            taxonomy_status == "exact_requested_type"
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


class SearchCatalogToolArguments(BaseModel):
    """Stable agent-facing catalog-search arguments."""

    model_config = ConfigDict(extra="forbid")

    semantic_query: str = Field(
        ...,
        description=(
            "One soft or descriptive product search string for semantic ranking. "
            "Product type may be repeated for relevance, but taxonomy and other "
            "must-haves are enforced only by the structured fields below. Use an "
            "empty string only for an image-only search."
        ),
    )
    shopper_guidance: str = Field(
        ...,
        max_length=400,
        description=(
            "One concise, product-agnostic shopper-facing sentence written under "
            "the active skill before search results are known. Connect this "
            "product role to the shopper's stated goal or direct antecedent. Do "
            "not name unselected or unavailable product types, name or describe "
            "candidate products, assert product attributes, or mention tools, "
            "schemas, filters, evidence, or identifiers. Use an empty string only "
            "for image-only search."
        ),
    )
    requested_product_type: str | None = Field(
        ...,
        description=(
            "Shortest product noun or umbrella phrase for this focused role, "
            "resolved from the shopper's current turn or direct antecedent. "
            "Exclude color, material, fit, occasion, weather, and style modifiers: "
            "'formal tops' and 'relaxed-fit tops' both use 'tops'. For a role the "
            "shopper did not name, use your own short role noun for it -- 'top', "
            "'shoes' -- and name every advertised subcategory that role covers in "
            "taxonomy. The reply will say the role was your suggestion. "
            "If this type is not separately advertised and you select one faithful "
            "advertised parent category, keep this shopper-named type unchanged. "
            "This is provenance, not catalog taxonomy or a ranking query. Use null "
            "only for image-only search."
        ),
    )
    taxonomy: CatalogTaxonomyToolInput = Field(
        ...,
        description=(
            "Required catalog-derived taxonomy selection. Allowed category and "
            "subcategory values come from the active catalog capabilities. Every "
            "selected value must be the requested product type or a child of an "
            "umbrella the shopper actually named. For a role the shopper did not "
            "name, select every advertised subcategory that role genuinely covers "
            "-- a proposed 'top' may select blouses and sweaters together. Do not "
            "widen a role to types it does not cover. Never select "
            "a parent or sibling as a substitute for an advertised type. If a "
            "shopper-named type is not separately advertised but one advertised "
            "category is its faithful broader parent, select only that category "
            "and leave subcategory empty; results remain alternatives under their "
            "actual catalog types. For example, skirts may satisfy bottoms; "
            "dresses may not."
        ),
    )
    required_constraints: dict[str, Any] = Field(
        ...,
        description=(
            "Every non-taxonomy must-have stated for the target products in the "
            "current shopper turn, as a structured field and value. Facts about "
            "an antecedent or anchor guide semantic styling judgment; do not copy "
            "them onto a complementary product unless the shopper explicitly asks "
            "for the same value. "
            "Use exact hard-filter properties and advertised enum values from "
            "current Catalog capabilities. Put a directly stated requirement "
            "those capabilities do not advertise in unadvertised_requirements. "
            "Season, weather, occasion, "
            "and subjective style/vibe "
            "context remain semantic direction unless the shopper directly "
            "requires an objective product attribute. A defining material is a "
            "must-have: for 'Any denim skirts available?', put 'denim' in "
            "unadvertised_requirements when composition is not a hard filter. "
            "Product types never belong in unadvertised_requirements. "
            "When the shopper names who an item is for, and the catalog "
            "advertises an audience filter, include it with every advertised "
            "value that suits that person -- a value covering all genders "
            "suits anyone. Without it, items that person cannot use stay in "
            "the results. Omit it entirely when the shopper has not said who "
            "the item is for."
        ),
    )
    scope_complete: bool = Field(
        ...,
        description=(
            "True only when this search plus existing turn evidence is enough to "
            "answer the shopper's complete current request. For a recommendation-"
            "only one-role request this is true. Set false when an explicitly "
            "requested product role, product-detail verification, availability "
            "check, or cart action still must run after this search. Do not set "
            "false merely to search alternatives or adjacent types, or because a "
            "broader multi-turn outfit project remains unfinished. 'Start with a "
            "beige top' and 'What bottoms go with that?' are each complete after "
            "their own one-role search."
        ),
    )
    search_mode: str | None = Field(
        default=None,
        description="Optional search mode from Catalog capabilities.",
    )


class SearchCatalogToolInput(SearchCatalogToolArguments):
    """Runtime-validated catalog search request."""

    taxonomy_status: Literal[
        "exact_requested_type",
        "member_of_requested_umbrella",
        "parent_category_alternative",
        "agent_selected_type",
        "no_direct_catalog_match",
        "image_only",
    ] = Field(..., description="Server-derived catalog execution mode.")

    @model_validator(mode="after")
    def text_search_has_taxonomy_scope(self) -> "SearchCatalogToolInput":
        if len(set(self.taxonomy.category)) > 1:
            raise ValueError("catalog search accepts at most one category")
        has_taxonomy = bool(
            self.taxonomy.category or self.taxonomy.subcategory
        )
        has_query = bool(self.semantic_query.strip())
        has_shopper_guidance = bool(self.shopper_guidance.strip())
        requested_product_type = (
            self.requested_product_type.strip()
            if isinstance(self.requested_product_type, str)
            else ""
        )
        if self.taxonomy_status in {
            "image_only",
            "no_direct_catalog_match",
        }:
            if has_shopper_guidance:
                raise ValueError(
                    "A non-text retrieval path requires empty shopper_guidance"
                )
        elif not has_shopper_guidance:
            raise ValueError(
                "catalog retrieval requires non-empty shopper_guidance"
            )
        if self.taxonomy_status != "image_only" and not requested_product_type:
            raise ValueError(
                "text catalog search requires requested_product_type"
            )
        if self.taxonomy_status == "image_only" and requested_product_type:
            raise ValueError(
                "image-only search requires requested_product_type=null"
            )
        if self.taxonomy_status == "no_direct_catalog_match":
            if has_taxonomy:
                raise ValueError(
                    "a non-retrieval result requires empty taxonomy arrays"
                )
            if not has_query:
                raise ValueError(
                    "a non-retrieval result requires a requested product type"
                )
            constraints = (
                self.required_constraints.model_dump(exclude_none=True)
                if isinstance(self.required_constraints, BaseModel)
                else self.required_constraints
            )
            has_constraint = any(
                value not in (None, "", [], {})
                for value in constraints.values()
            )
            if has_constraint:
                raise ValueError(
                    "a non-retrieval result cannot include required constraints"
                )
            return self
        if self.taxonomy_status == "image_only":
            if has_query or has_taxonomy:
                raise ValueError(
                    "image-only search requires an empty semantic query and taxonomy"
                )
            return self
        if self.taxonomy_status == "member_of_requested_umbrella" and not (
            self.taxonomy.subcategory
        ):
            raise ValueError(
                "an umbrella search requires an advertised subcategory"
            )
        if self.taxonomy_status == "agent_selected_type" and not (
            self.taxonomy.subcategory
        ):
            raise ValueError(
                "an open-role search requires an advertised subcategory"
            )
        if self.taxonomy_status == "parent_category_alternative" and not (
            len(self.taxonomy.category) == 1
            and not self.taxonomy.subcategory
        ):
            raise ValueError(
                "a parent-category alternative requires one advertised category "
                "and no subcategory"
            )
        if has_query and not has_taxonomy:
            raise ValueError(
                "text catalog search requires an advertised category or subcategory"
            )
        if not has_query:
            raise ValueError("text catalog search requires a semantic query")
        return self


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


class _CatalogNumberConstraint(BaseModel):
    """Inclusive numeric range accepted by catalog hard filters."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def has_a_bound(self) -> "_CatalogNumberConstraint":
        if self.min is None and self.max is None:
            raise ValueError("numeric constraint requires min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("numeric constraint min cannot exceed max")
        return self


def _search_catalog_tool_input_model(
    capabilities: CatalogCapabilities,
    *,
    validate_scope: bool = True,
    wearer_audience_field: str = "",
) -> type[SearchCatalogToolArguments]:
    """Create one tool schema whose enums come from the active catalog."""

    taxonomy = capabilities.taxonomy
    advertised_search_modes = tuple(dict.fromkeys(capabilities.retrieval_modes))
    search_mode_type = (
        Literal.__getitem__(advertised_search_modes)
        if advertised_search_modes
        else str
    )
    category_values = (
        sorted(taxonomy.categories, key=str.casefold)
        if taxonomy.category_field
        else []
    )
    subcategory_values = (
        sorted(
            {
                name
                for category in taxonomy.categories.values()
                for name in category.subcategories
            },
            key=str.casefold,
        )
        if taxonomy.subcategory_field
        else []
    )

    category_type, category_field = _taxonomy_list_field(
        category_values,
        role="category",
        advertised_field=taxonomy.category_field,
    )
    subcategory_type, subcategory_field = _taxonomy_list_field(
        subcategory_values,
        role="subcategory",
        advertised_field=taxonomy.subcategory_field,
    )
    taxonomy_model = create_model(
        "CatalogTaxonomySelection",
        __base__=CatalogTaxonomyToolInput,
        category=(category_type, category_field),
        subcategory=(subcategory_type, subcategory_field),
    )
    required_constraints_model = _required_constraints_input_model(
        capabilities,
        wearer_audience_field=wearer_audience_field,
    )
    return create_model(
        "CatalogSearchInput" if validate_scope else "CatalogSearchToolArguments",
        __base__=(
            SearchCatalogToolInput
            if validate_scope
            else SearchCatalogToolArguments
        ),
        taxonomy=(
            taxonomy_model,
            Field(
                ...,
                description=(
                    "Required taxonomy selection generated from the active catalog. "
                    "Use exact enum values. A text search requires at least one "
                    "category or subcategory; both arrays may be empty only for "
                    "an image-only search."
                    " Every value must be a kind of the requested scope: skirts "
                    "may satisfy bottoms; dresses may not. For a broad request "
                    "that names no product type, choose one exact advertised "
                    "subcategory as the focused starting role. If a shopper-named "
                    "type is not separately advertised but one faithful broader "
                    "advertised parent category exists, select only that category "
                    "and leave subcategory empty."
                ),
            ),
        ),
        required_constraints=(
            required_constraints_model,
            Field(
                ...,
                description=(
                    "Catalog hard filters and any defining requirement the active "
                    "catalog cannot enforce. Apply constraints only when the "
                    "current turn states them for the target products; an anchor's "
                    "attributes belong in semantic styling context unless the "
                    "shopper explicitly requests the same value. Use only the "
                    "advertised properties "
                    "in this object; put unsupported must-haves in "
                    "unadvertised_requirements instead of weakening them. For "
                    "'Do you have water-resistant bags?', use "
                    "{'unadvertised_requirements': ['water resistance']}. Do not "
                    "put broad season, weather, occasion, or subjective style/vibe "
                    "context here unless the shopper directly requires a product "
                    "attribute. 'Rainy day outfit' does not require water "
                    "resistance; 'water-resistant bags' does. Recommendation "
                    "adjectives such as comfortable, relaxed, soft, breathable, "
                    "lightweight, casual, dressy, bold, bright, vibrant, or "
                    "sporty are always semantic ranking preferences, not "
                    "objective hard filters. Before calling the tool, compare "
                    "every target-product modifier with this advertised schema "
                    "and include every exact matching filter value. A product type "
                    "never belongs in unadvertised_requirements. Use an empty "
                    "object for image-only search."
                ),
            ),
        ),
        search_mode=(
            search_mode_type | None,
            Field(
                default=None,
                description="Optional search mode advertised by the active catalog.",
            ),
        ),
    )


def _search_catalog_scopes_input_model(
    capabilities: CatalogCapabilities,
    *,
    max_scopes: int = 1,
    wearer_audience_field: str = "",
) -> type[BaseModel]:
    """Wrap the catalog search arguments in a list of scopes.

    Stage one of the scoped-search contract, and deliberately nothing more: the
    scope object is exactly today's argument object, so the only thing that
    changes for the model is one level of nesting.

    That isolation is the point. Argument malformation scales sharply with
    nesting -- measured across the run archive, a flat `product_ref` argument was
    wrapped in stray punctuation 1-10% of the time and the same value inside a
    nested list of objects 43% of the time. If the model fumbles a nested list of
    fields it already emits correctly, the cause is depth itself and the rest of
    the contract has to be shaped around that. If it does not, the 43% was about
    mixing authored and transcribed fields in one object, which is a different
    and more tractable problem.

    `max_scopes` stays at one until that question is answered.
    """

    scope_model = _search_catalog_tool_input_model(
        capabilities,
        validate_scope=False,
        wearer_audience_field=wearer_audience_field,
    )
    return create_model(
        "CatalogSearchScopes",
        scopes=(
            list[scope_model],
            Field(
                ...,
                min_length=1,
                max_length=max_scopes,
                description=(
                    "One search scope per advertised category. Each scope owns "
                    "its own taxonomy and constraints, so a filter for one role "
                    "can never exclude another role's products."
                ),
            ),
        ),
        not_covered=(
            list[str] | None,
            Field(
                default=None,
                max_length=10,
                description=(
                    "Product types the shopper asked for that no advertised "
                    "category covers, in the shopper's own words. Do not search "
                    "for these -- naming them here is what records the request so "
                    "it can be answered. A shopper asking for 'a pan, a shoe and "
                    "a bag' gets scopes for the shoe and the bag, and 'pan' here."
                ),
            ),
        ),
    )


def _taxonomy_list_field(
    values: list[str],
    *,
    role: str,
    advertised_field: str | None,
) -> tuple[Any, Any]:
    field_name = advertised_field or "not advertised"
    description = (
        f"Exact {role} values advertised through catalog field '{field_name}'. "
        "Use an empty list when the other taxonomy role supplies the text scope or "
        "the search is image-only."
    )
    if role == "category":
        description += " Select at most one category per catalog search."
    if not values:
        return list[str], Field(..., max_length=0, description=description)

    literal_type = Literal.__getitem__(tuple(values))
    max_length = 1 if role == "category" else None
    return list[literal_type], Field(
        ...,
        max_length=max_length,
        description=description,
    )


#: Every advertised filter otherwise gets "Advertised hard filter '<name>'."
#: For the audience field that generic line is the whole bug: the model was
#: reading the enum value names and nothing else, so "sunglasses for men" and
#: "my husband" scoped correctly while "shades for hubby" sent no filter and
#: returned women's sunglasses. The rule below is not new -- it is the one
#: already written in the system prompt, moved to the channel the model reads.
_WEARER_AUDIENCE_FILTER_DESCRIPTION = (
    "Who the products are for. When the shopper names the person this is for, "
    "send every advertised value that suits them, and a value covering all "
    "genders always suits anyone, so include it alongside the value for their "
    "gender. Shoppers name people casually and every one of these counts the "
    "same as a formal word: hubby, my guy, my man, my other half, my brother, "
    "my dad, my son, my wife, my girlfriend, my mum, my sister, my daughter, "
    "the kids. Buying for a woman means the women's value and the all-genders "
    "value together, because both suit her; buying for a man in a catalog "
    "that advertises no men's value means the all-genders value alone, "
    "because the women's pieces genuinely do not suit him. When nobody is "
    "named, omit this filter entirely -- do not send a covers-everyone value "
    "to mean unspecified, which would discard everything stocked for one "
    "audience. If nothing advertised suits the named person, such as a child "
    "in an adult-only catalog, search for them anyway and let it return "
    "nothing; never substitute what does not suit them. The person-words "
    "listed above are for reading what the shopper said. They are not "
    "audiences to offer back: a reply may name only audiences this catalog "
    "advertises, and must never suggest looking for one it does not stock. "
    "Only this turn's "
    "words count. An audience established earlier in the conversation never "
    "carries into a request that does not name that person again, however "
    "obviously they are still around: send no filter and let the shopper "
    "redirect you. Enumerating the ways a shopper moves on is hopeless, so "
    "the rule is the reverse -- naming someone is what turns the filter on."
)


def _required_constraints_input_model(
    capabilities: CatalogCapabilities,
    *,
    wearer_audience_field: str = "",
) -> type[BaseModel]:
    """Create a hard-constraint schema from advertised catalog filters."""

    taxonomy_fields = {
        field_name
        for field_name in (
            capabilities.taxonomy.category_field,
            capabilities.taxonomy.subcategory_field,
        )
        if field_name
    }
    fields: dict[str, tuple[Any, Any]] = {}
    for name, capability in sorted(effective_filter_capabilities(capabilities).items()):
        if name in taxonomy_fields:
            continue
        if capability.type in {"enum", "enum_list"} and capability.values:
            value_type = Literal.__getitem__(tuple(capability.values))
            field_type = value_type | list[value_type] | None
        elif capability.type == "number":
            field_type = _CatalogNumberConstraint | None
        else:
            field_type = str | list[str] | None
        fields[name] = (
            field_type,
            Field(
                default=None,
                description=(
                    _WEARER_AUDIENCE_FILTER_DESCRIPTION
                    if name and name == wearer_audience_field
                    else f"Advertised hard filter '{name}'."
                ),
            ),
        )
    fields["unadvertised_requirements"] = (
        list[str],
        Field(
            default_factory=list,
            description=(
                "Only objective product requirements directly stated in the "
                "current shopper turn and absent from this schema. Never infer a "
                "requirement from season, weather, occasion, or subjective "
                "style/vibe context: 'rainy day outfit' produces an empty list. "
                "Use ['water resistance'] for 'water-resistant bags' because that "
                "turn directly states the product requirement. Never omit or "
                "soften a directly stated objective requirement. Subjective "
                "recommendation adjectives such as comfortable, relaxed, soft, "
                "breathable, lightweight, casual, dressy, bold, bright, vibrant, "
                "or sporty always stay semantic; never put them in this field. "
                "Before calling the tool, include every exact advertised filter "
                "value that modifies the target product; do not leave this object "
                "empty when one applies. "
                "In an 'A or B' request, do not put the "
                "supported advertised branch here. A product type never belongs "
                "here."
            ),
        ),
    )
    return create_model(
        "CatalogRequiredConstraints",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class _ShopperSkillActivationInput(BaseModel):
    """Shared composition rules for dynamic shopper-skill activation."""

    model_config = ConfigDict(extra="forbid")

    skill_names: list[str]

    @model_validator(mode="after")
    def primary_procedures_are_exclusive(self) -> "_ShopperSkillActivationInput":
        selected = set(self.skill_names)
        primary = selected.intersection(
            {"outfit-styling", "product-discovery"}
        )
        if len(primary) > 1:
            raise PydanticCustomError(
                SKILL_ACTIVATION_MULTIPLE_PRIMARY,
                "select exactly one primary procedure: outfit-styling or "
                "product-discovery, never both",
            )
        if "budget-shopping" in selected and len(primary) != 1:
            raise PydanticCustomError(
                SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
                "budget-shopping requires exactly one primary procedure: "
                "outfit-styling or product-discovery",
            )
        return self


def _skill_activation_input_model(
    skill_names: tuple[str, ...],
) -> type[BaseModel]:
    """Create the semantic skill-selection schema from the active registry."""

    skill_name_type = Literal.__getitem__(skill_names)
    return create_model(
        "ShopperSkillActivationInput",
        __base__=_ShopperSkillActivationInput,
        skill_names=(
            list[skill_name_type],
            Field(
                ...,
                min_length=1,
                max_length=len(skill_names),
                description=(
                    "Smallest set of registered shopper skills whose descriptions "
                    "cover the current turn's complete intent. For product search "
                    "or styling, select exactly one primary procedure: outfit-"
                    "styling or product-discovery, never both. Cart and policy "
                    "intents may select their standalone skill without a primary."
                ),
            ),
        ),
    )


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


@dataclass(frozen=True)
class RequestIdentity:
    """Server-owned identity used to scope one assistant turn."""

    session_id: str
    conversation_id: str
    cart_id: str
    context_user_id: int
    cart_user_id: int
    request_id: str
    shopper_profile_id: str | None = None

    @property
    def legacy_user_id(self) -> int:
        return self.context_user_id

    @property
    def checkpoint_thread_id(self) -> str:
        return json.dumps(
            [self.conversation_id, self.request_id],
            separators=(",", ":"),
        )


def _conversation_turn_status(termination_reason: str) -> FinalTurnStatus:
    if termination_reason in {
        "input_guardrail_blocked",
        "output_guardrail_blocked",
    }:
        return "blocked"
    if termination_reason == "completed":
        return "completed"
    return "failed"


def create_request_identity(
    *,
    legacy_user_id: int,
    session_id: str | None = None,
    conversation_id: str | None = None,
    cart_id: str | None = None,
    request_id: str | None = None,
    shopper_profile_id: str | None = None,
) -> RequestIdentity:
    """Create scoped request identity while preserving legacy user_id behavior."""

    session = session_id or f"legacy-session-{legacy_user_id}"
    conversation = conversation_id or f"legacy-conversation-{legacy_user_id}"
    cart = cart_id or f"legacy-cart-{legacy_user_id}"
    return RequestIdentity(
        session_id=session,
        conversation_id=conversation,
        cart_id=cart,
        context_user_id=(
            _stable_numeric_id("conversation", conversation_id)
            if conversation_id
            else legacy_user_id
        ),
        cart_user_id=_stable_numeric_id("cart", cart_id) if cart_id else legacy_user_id,
        request_id=request_id or str(uuid.uuid4()),
        shopper_profile_id=shopper_profile_id,
    )


def _stable_numeric_id(namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _empty_agent_diagnostics(final_termination_reason: str) -> dict[str, Any]:
    return {
        "skill_files_read": [],
        "tool_calls": [],
        "rejected_tool_calls": [],
        "duplicate_tool_calls": [],
        "product_evidence": [],
        "product_evidence_truncated": False,
        "catalog_scope_outcomes": [],
        "final_termination_reason": final_termination_reason,
        "partial_graph_messages": [],
    }


async def _partial_graph_messages(
    agent: Any,
    invoke_config: dict[str, Any],
) -> tuple[list[Any], str | None]:
    """Read the last graph state before its failed checkpoint is deleted."""

    get_state = getattr(agent, "aget_state", None)
    if get_state is None:
        return [], "state_snapshot_unavailable"
    try:
        snapshot = await asyncio.wait_for(
            get_state(invoke_config),
            timeout=_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("Timed out snapshotting Deep Agents state before cleanup")
        return [], "state_snapshot_timeout"
    except Exception as exc:  # noqa: BLE001 - diagnostics cannot block cleanup.
        error_type = type(exc).__name__
        logger.warning(
            "Could not snapshot Deep Agents state before cleanup: %s",
            error_type,
        )
        return [], error_type

    values = _value(snapshot, "values")
    messages = _value(values, "messages")
    return (messages if isinstance(messages, list) else []), None


def _collect_agent_diagnostics(
    messages: list[Any],
    *,
    request_id: str,
    final_termination_reason: str,
    preserve_partial_messages: bool = False,
) -> dict[str, Any]:
    """Collect current-turn skill, tool, and termination diagnostics."""

    diagnostics = _empty_agent_diagnostics(final_termination_reason)
    turn_messages = _current_turn_messages(messages, request_id)
    tool_results = _tool_results_by_call_id(turn_messages)
    skill_files_read: list[str] = []
    successful_product_tool_calls: dict[str, str] = {}

    for message in turn_messages:
        if _message_type(message) != "ai":
            continue
        calls = [
            (raw_call, None)
            for raw_call in (_value(message, "tool_calls") or [])
        ]
        additional_kwargs = _value(message, "additional_kwargs") or {}
        restored_fields_by_call: dict[str, list[str]] = {}
        if isinstance(additional_kwargs, dict):
            restored_calls = additional_kwargs.get(
                SERVER_RESTORED_TOOL_CALL_FIELDS
            )
            if isinstance(restored_calls, list):
                for restored_call in restored_calls[:8]:
                    tool_call_id = str(
                        _value(restored_call, "tool_call_id") or ""
                    )
                    fields = _value(restored_call, "fields")
                    if not tool_call_id or not isinstance(fields, list):
                        continue
                    restored_fields_by_call[tool_call_id] = [
                        str(field)[:64] for field in fields[:8]
                    ]
            calls.extend(
                (raw_call, str(_value(raw_call, "rejection_reason") or "rejected"))
                for raw_call in (
                    additional_kwargs.get(_SERVER_REJECTED_TOOL_CALLS) or []
                )
            )
        for raw_call, forced_rejection_reason in calls:
            call = _normalized_tool_call(raw_call)
            sequence = len(diagnostics["tool_calls"]) + 1
            result_message = tool_results.get(call["tool_call_id"])
            if forced_rejection_reason:
                status, rejection_reason = "rejected", forced_rejection_reason
            else:
                status, rejection_reason = _tool_call_status(
                    call["tool_name"],
                    result_message,
                )
            entry = {
                "sequence": sequence,
                "tool_name": call["tool_name"],
                "arguments": call["arguments"],
                "status": status,
            }
            if rejection_reason:
                entry["rejection_reason"] = rejection_reason
            scope_rejections = rejections_of(result_message)
            if len(scope_rejections) > 1 and any(scope_rejections):
                # A call that searched several roles and was refused only some
                # of them is not a rejected call, so its refusals would
                # otherwise be counted nowhere at all.
                entry["scope_rejections"] = scope_rejections
            restored_fields = restored_fields_by_call.get(call["tool_call_id"])
            if restored_fields:
                entry["restored_fields"] = restored_fields
            if rejection_reason == "duplicate_catalog_scope":
                entry["duplicate"] = True
                diagnostics["duplicate_tool_calls"].append(sequence)
            if status == "rejected":
                diagnostics["rejected_tool_calls"].append(sequence)
            diagnostics["tool_calls"].append(entry)

            if (
                status == "completed"
                and call["tool_call_id"]
                and call["tool_name"]
                in {"search_catalog_tool", "get_product_details_tool"}
            ):
                successful_product_tool_calls[call["tool_call_id"]] = call[
                    "tool_name"
                ]

            for skill_path in _skill_file_paths(call, status):
                if skill_path not in skill_files_read:
                    skill_files_read.append(skill_path)

    diagnostics["skill_files_read"] = skill_files_read
    product_evidence, product_evidence_truncated = _diagnostic_product_evidence(
        turn_messages,
        successful_product_tool_calls,
    )
    diagnostics["product_evidence"] = product_evidence
    diagnostics["product_evidence_truncated"] = product_evidence_truncated
    diagnostics["catalog_scope_outcomes"] = _diagnostic_catalog_scope_outcomes(
        turn_messages
    )
    if preserve_partial_messages:
        partial, truncated = _serialize_partial_graph_messages(turn_messages)
        diagnostics["partial_graph_messages"] = partial
        if truncated:
            diagnostics["partial_graph_messages_truncated"] = True
    return diagnostics


def _safe_collect_agent_diagnostics(
    messages: list[Any],
    *,
    request_id: str,
    final_termination_reason: str,
    preserve_partial_messages: bool = False,
) -> dict[str, Any]:
    """Collect diagnostics without allowing tracing to change turn behavior."""

    try:
        return _collect_agent_diagnostics(
            messages,
            request_id=request_id,
            final_termination_reason=final_termination_reason,
            preserve_partial_messages=preserve_partial_messages,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must fail independently.
        error_type = type(exc).__name__
        logger.warning("Could not collect Deep Agents diagnostics: %s", error_type)
        diagnostics = _empty_agent_diagnostics(final_termination_reason)
        diagnostics["diagnostic_collection_error"] = error_type
        return diagnostics


def _no_direct_taxonomy_response(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Return the fixed shopper response for a current-turn no-match result."""

    outcomes = _business_tool_result_contents(
        _current_turn_messages(_result_messages(result), request_id)
    )
    no_direct_outcomes = [
        content
        for content in outcomes
        if content.startswith(
            f"{STOP_TOOL_USE_PREFIX} No faithful advertised catalog taxonomy"
        )
    ]
    repair_or_no_direct = all(
        content.startswith(
            (
                SEARCH_VALIDATION_ERROR_PREFIX,
                f"{STOP_TOOL_USE_PREFIX} No faithful advertised catalog taxonomy",
            )
        )
        for content in outcomes
    )
    if no_direct_outcomes and repair_or_no_direct:
        return _NO_DIRECT_TAXONOMY_RESPONSE
    return None


def _rejected_catalog_search_response(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Fail closed when every current-turn business call is a rejected search."""

    messages = _current_turn_messages(_result_messages(result), request_id)
    tool_results = _tool_results_by_call_id(messages)
    business_calls: list[tuple[str, str]] = []
    for message in messages:
        if _message_type(message) != "ai":
            continue
        calls = [
            (raw_call, False)
            for raw_call in (_value(message, "tool_calls") or [])
        ]
        additional_kwargs = _value(message, "additional_kwargs") or {}
        if isinstance(additional_kwargs, dict):
            calls.extend(
                (raw_call, True)
                for raw_call in (
                    additional_kwargs.get(_SERVER_REJECTED_TOOL_CALLS) or []
                )
            )
        for raw_call, server_rejected in calls:
            call = _normalized_tool_call(raw_call)
            tool_name = call["tool_name"]
            if tool_name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}:
                continue
            status = (
                "rejected"
                if server_rejected
                else _tool_call_status(
                    tool_name,
                    tool_results.get(call["tool_call_id"]),
                )[0]
            )
            business_calls.append((tool_name, status))

    if not business_calls:
        return None
    if _catalog_repair_clarification_response(
        result,
        request_id=request_id,
    ):
        return None
    if all(
        tool_name == "search_catalog_tool" and status == "rejected"
        for tool_name, status in business_calls
    ):
        return _REJECTED_CATALOG_SEARCH_RESPONSE
    return None


def _catalog_repair_clarification_response(
    result: Any,
    *,
    request_id: str,
) -> str:
    """Return a fixed response for a server-marked no-tool repair branch."""

    messages = _current_turn_messages(_result_messages(result), request_id)
    for message in reversed(messages):
        if _message_type(message) != "ai" or _value(message, "tool_calls"):
            continue
        additional_kwargs = _value(message, "additional_kwargs") or {}
        if not isinstance(additional_kwargs, dict) or not additional_kwargs.get(
            SERVER_CATALOG_CLARIFICATION
        ):
            continue
        return _CATALOG_REPAIR_CLARIFICATION_RESPONSE
    return ""


def _business_tool_result_contents(messages: list[Any]) -> list[str]:
    """Return non-activation tool outcomes in graph order."""

    outcomes: list[str] = []
    for message in messages:
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"} or content.startswith(
            (
                SKILL_ACTIVATION_COMPLETE,
                SKILL_ACTIVATION_REQUIRED,
                SKILL_TOOL_NOT_GRANTED,
                "SHOPPER_SKILL_ACTIVATION_FAILED:",
            )
        ):
            continue
        outcomes.append(content)
    return outcomes


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


def _normalized_tool_call(raw_call: Any) -> dict[str, Any]:
    arguments = _value(raw_call, "args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"value": arguments}
    return {
        "tool_call_id": str(_value(raw_call, "id") or ""),
        "tool_name": str(_value(raw_call, "name") or "unknown"),
        "arguments": _diagnostic_json_value(arguments),
    }


def _tool_call_status(
    tool_name: str,
    result_message: Any | None,
) -> tuple[str, str | None]:
    if result_message is None:
        return "pending", None
    content = _content_to_text(_value(result_message, "content"))
    rejection_reason = _tool_rejection_reason(
        content,
        rejections_of(result_message),
    )
    if rejection_reason:
        return "rejected", rejection_reason
    if _value(result_message, "status") == "error":
        return "error", None
    if tool_name == "read_file" and content.lower().startswith(
        ("error", "file not found")
    ):
        return "error", None
    return "completed", None


def _tool_rejection_reason(
    content: str,
    scope_rejections: list[Any] | None = None,
) -> str | None:
    """Name what refused this call, preferring what the gate itself recorded.

    Nine catalog-search gates render one model-visible prefix, so the text can
    only ever say ``invalid_catalog_request``; a gate that recorded its own code
    names itself instead. The code is believed only when every searched scope
    carries one, because a call that refused one role and answered another is
    not a refused call. A result carrying no codes -- an older checkpoint, or
    one of the paths that deliberately records none -- falls back to matching
    the text, so a missed path degrades to today's answer rather than losing
    its reason.
    """

    codes = list(scope_rejections or ())
    if codes and all(codes):
        return str(codes[0])
    markers = (
        (SKILL_ACTIVATION_REQUIRED, "skill_activation_required"),
        (SKILL_TOOL_NOT_GRANTED, "skill_tool_not_granted"),
        ("SHOPPER_SKILL_ACTIVATION_FAILED:", "skill_activation_failed"),
        (
            f"{STOP_TOOL_USE_PREFIX} This catalog taxonomy and constraint scope was already searched",
            "duplicate_catalog_scope",
        ),
        (f"{STOP_TOOL_USE_PREFIX} Catalog search limit reached", "catalog_search_limit"),
        (
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy",
            "no_advertised_taxonomy_match",
        ),
        (
            f"{STOP_TOOL_USE_PREFIX} Product-detail read limit reached",
            "product_detail_read_limit",
        ),
        (
            "The catalog search request does not match current capabilities:",
            "invalid_catalog_request",
        ),
        (SEARCH_VALIDATION_ERROR_PREFIX, "invalid_catalog_request"),
        (CONSTRAINT_REVIEW_PREFIX, "constraint_review_required"),
        (
            UNSUPPORTED_TAXONOMY_PREFIX,
            "unsupported_catalog_taxonomy",
        ),
        (
            UNSUPPORTED_CONSTRAINT_PREFIX,
            "unsupported_catalog_constraint",
        ),
        (_UNSUPPORTED_SEARCH_MODE_MESSAGE, "unsupported_search_mode"),
    )
    for marker, reason in markers:
        if content.startswith(marker):
            return reason
    if content.startswith(STOP_TOOL_USE_PREFIX):
        return "stop_tool_use"
    return None


def _skill_file_paths(call: dict[str, Any], status: str) -> list[str]:
    if status != "completed":
        return []
    arguments = call["arguments"]
    if call["tool_name"] == SKILL_ACTIVATION_TOOL_NAME:
        names = arguments.get("skill_names") or []
        if not isinstance(names, list):
            return []
        return [
            f"/shopper/{name}/SKILL.md"
            for name in names
            if isinstance(name, str) and name.strip()
        ]
    if call["tool_name"] != "read_file":
        return []
    path = str(arguments.get("file_path") or arguments.get("path") or "")
    normalized = path.replace("\\", "/")
    if normalized.startswith("/shopper/") and normalized.endswith("/SKILL.md"):
        return [normalized]
    return []


def _serialize_partial_graph_messages(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], bool]:
    relevant = [
        message for message in messages if _message_type(message) in {"ai", "tool"}
    ]
    truncated = len(relevant) > 24
    serialized: list[dict[str, Any]] = []
    for message in relevant[-24:]:
        content = _content_to_text(_value(message, "content"))
        content_truncated = len(content) > 2000
        payload: dict[str, Any] = {
            "type": _message_type(message),
            "content": content[:2000],
        }
        for field in ("name", "tool_call_id"):
            value = _value(message, field)
            if value:
                payload[field] = str(value)
        tool_calls = _value(message, "tool_calls")
        if tool_calls:
            payload["tool_calls"] = [
                _normalized_tool_call(call) for call in tool_calls
            ]
        if content_truncated:
            payload["truncated"] = True
            truncated = True
        serialized.append(payload)
    return serialized, truncated


def _diagnostic_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): _diagnostic_json_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _diagnostic_product_evidence(
    messages: list[Any],
    successful_tool_calls: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    """Return bounded product facts from successful current-turn tool results."""

    evidence: list[dict[str, Any]] = []
    truncated = False
    aggregate_limit_reached = False
    for message in messages:
        if _message_type(message) != "tool":
            continue

        tool_call_id = str(_value(message, "tool_call_id") or "")
        source_tool = successful_tool_calls.get(tool_call_id)
        payload = evidence_of(message)
        if source_tool == "search_catalog_tool":
            if not payload or payload.get("outcome") != "results":
                continue
            evidence_type = "search_result"
            search_scope = {
                "taxonomy": _bounded_product_evidence_value(
                    payload.get("taxonomy") or {}
                ),
                "confirmed_filters": _bounded_product_evidence_value(
                    payload.get("confirmed_filters") or {}
                ),
                "composed_role": bool(payload.get("composed_role")),
            }
            records = payload.get("products") or []
        elif source_tool == "get_product_details_tool":
            detail = detail_evidence_of(message)
            if not detail:
                continue
            evidence_type = "product_detail"
            search_scope = None
            records = detail.get("products") or []
        else:
            continue

        for product in records:
            product_ref = product.get("product_ref")
            product_name = product.get("name")
            if not product_ref or not product_name:
                continue
            record = {
                "product_ref": _bounded_product_evidence_value(product_ref),
                "product_name": _bounded_product_evidence_value(product_name),
                "source_tool": source_tool,
                "evidence_type": evidence_type,
                "facts": _diagnostic_product_facts(product),
            }
            # A multi-role call's payload carries the union of every role's
            # taxonomy and filters. The product knows which role retrieved it;
            # prefer that over the union, or the trace records a product as
            # confirmed under a filter it was never searched with.
            product_scope = _product_search_scope(product)
            if product_scope is not None:
                record["search_scope"] = product_scope
            elif search_scope is not None:
                record["search_scope"] = search_scope
            if (
                len(evidence) >= _MAX_DIAGNOSTIC_PRODUCT_EVIDENCE
                or aggregate_limit_reached
            ):
                truncated = True
                continue
            candidate = [*evidence, record]
            if len(json.dumps(candidate, sort_keys=True, default=str)) > (
                _MAX_DIAGNOSTIC_PRODUCT_EVIDENCE_CHARS
            ):
                truncated = True
                aggregate_limit_reached = True
                continue
            evidence.append(record)
    return evidence, truncated


def _diagnostic_catalog_scope_outcomes(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Return bounded server-authored outcomes that contain no products."""

    outcomes: list[dict[str, Any]] = []
    allowed_fields = {
        "outcome",
        "requested_product_type",
        "taxonomy",
        "confirmed_filters",
        "composed_role",
    }
    for message in messages:
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message)
        outcome = (payload or {}).get("scope_outcome") or {}
        if (
            not outcome
            or not set(outcome).issubset(allowed_fields)
            or outcome.get("outcome")
            not in {"no_direct_catalog_match", "zero_results"}
        ):
            continue
        bounded = _bounded_product_evidence_value(outcome)
        if isinstance(bounded, dict) and bounded not in outcomes:
            outcomes.append(bounded)
        if len(outcomes) >= 8:
            break
    return outcomes


def _wearer_audience_events(
    state: Any,
    identity: Any,
    *,
    field_name: str,
) -> list[ConversationEvent]:
    """Record an audience this turn declared, so the next turn inherits it.

    Read from the turn's own diagnostics rather than the graph messages: the
    per-product scope stamp already carries the filters each role was searched
    with, and zero-result scopes carry theirs too, so nothing new has to be
    threaded through finalize.

    The server records what the model declared and never reads the shopper's
    prose to work out who an item is for, which it is forbidden to do. A turn
    that filtered on nothing declares nothing and leaves an earlier declaration
    standing -- silence is how the model forgets, not how a shopper changes
    their mind.
    """

    if not field_name:
        return []
    diagnostics = getattr(state, "agent_diagnostics", None) or {}
    scopes: list[Any] = [
        (record or {}).get("search_scope")
        for record in (diagnostics.get("product_evidence") or [])
    ]
    scopes.extend(diagnostics.get("catalog_scope_outcomes") or [])
    declared: list[str] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        values = (scope.get("confirmed_filters") or {}).get(field_name)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value)
            if text and text not in declared:
                declared.append(text)
    if not declared:
        return []
    return [
        ConversationEvent(
            event_key=f"wearer-audience:{identity.request_id}",
            event_type="wearer_audience_declared",
            source_kind="runtime",
            payload={"audience": declared[:8]},
        )
    ]


def _audience_assumption_events(
    state: Any,
    identity: Any,
) -> list[ConversationEvent]:
    """Record that this conversation has now been told what was assumed.

    Catalog search wrote it while the turn ran, so this reads the turn's own
    state rather than re-deriving anything. Nothing is recorded on a turn that
    disclosed nothing, which leaves an earlier disclosure standing: the shopper
    is owed the sentence once, not once per search.

    Deliberately not a `wearer_audience_declared`. That event is a fact the
    model established and may narrow later searches with; this one is only the
    shop admitting a guess, and must never scope anything.
    """

    disclosed = [str(value) for value in (getattr(state, "disclosed_audience", None) or [])]
    if not disclosed:
        return []
    return [
        ConversationEvent(
            event_key=f"audience-assumption:{identity.request_id}",
            event_type="audience_assumption_disclosed",
            source_kind="runtime",
            payload={"audience": disclosed[:8]},
        )
    ]


def _turn_audience_events(
    state: Any,
    identity: Any,
    *,
    field_name: str,
) -> list[ConversationEvent]:
    """Record what this turn settled about who is being shopped for.

    A declaration outranks an assumption made earlier in the same turn. That
    is the self-correcting case: an unscoped search returns womenswear, the
    evidence says so, the model recognises the shopper named a husband and
    searches again with the values that suit him. Recording both would leave
    the conversation carrying a guess the turn had already overturned.
    """

    declared = _wearer_audience_events(state, identity, field_name=field_name)
    if declared:
        return declared
    return _audience_assumption_events(state, identity)


def _product_search_scope(product: Any) -> dict[str, Any] | None:
    """Return the scope that actually retrieved one product, if it carries one.

    Only a multi-role call stamps this, because only then is the call-level
    scope a union of several roles rather than a description of this product.
    """

    if not isinstance(product, dict):
        return None
    scope = product.get("search_scope")
    if not isinstance(scope, dict):
        return None
    return {
        "taxonomy": _bounded_product_evidence_value(scope.get("taxonomy") or {}),
        "confirmed_filters": _bounded_product_evidence_value(
            scope.get("confirmed_filters") or {}
        ),
        "composed_role": bool(scope.get("composed_role")),
    }


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


def _diagnostic_product_facts(product: dict[str, Any]) -> dict[str, Any]:
    """Extract bounded, structured facts from one parsed product record."""

    facts: dict[str, Any] = {}
    for key in ("category", "brand", "price"):
        value = product.get(key)
        if value:
            facts[key] = _bounded_product_evidence_value(value)
    facts["image_available"] = bool(product.get("image_url"))

    # Attributes the catalog confirmed on a search result. The composer is
    # allowed to state these, so the evidence trace has to carry them: a fact an
    # observer cannot see the support for is indistinguishable from an invented
    # one, and gets judged as invention however accurate it is.
    attributes = product.get("attributes")
    if isinstance(attributes, dict):
        for name, value in attributes.items():
            if len(facts) >= _MAX_DIAGNOSTIC_PRODUCT_FACTS:
                break
            bounded_name = str(_bounded_product_evidence_value(name))
            if bounded_name and bounded_name not in facts:
                facts[bounded_name] = _bounded_product_evidence_value(value)

    for raw_detail in product.get("details") or []:
        if len(facts) >= _MAX_DIAGNOSTIC_PRODUCT_FACTS:
            break
        if not isinstance(raw_detail, str) or ":" not in raw_detail:
            continue
        name, value = raw_detail.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        bounded_name = str(_bounded_product_evidence_value(name))
        if bounded_name not in facts:
            facts[bounded_name] = _bounded_product_evidence_value(value)
    return facts


def _bounded_product_evidence_value(value: Any, *, depth: int = 0) -> Any:
    """Bound strings and collections copied into product evidence diagnostics."""

    if isinstance(value, str):
        return value[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if depth >= 4:
        return str(value)[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]
    if isinstance(value, dict):
        return {
            str(key)[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]: (
                _bounded_product_evidence_value(item, depth=depth + 1)
            )
            for key, item in list(value.items())[:_MAX_DIAGNOSTIC_PRODUCT_FACTS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_product_evidence_value(item, depth=depth + 1)
            for item in value[:_MAX_DIAGNOSTIC_PRODUCT_FACTS]
        ]
    return str(value)[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]


def _append_product_results(state: State, products: list[ProductSummary]) -> None:
    existing_ids = {
        str(product.get("product_id") or "")
        for product in state.product_results
        if isinstance(product, dict)
    }
    for product in products:
        payload = product.model_dump(mode="json")
        product_id = str(payload.get("product_id") or "")
        if product_id and product_id in existing_ids:
            continue
        state.product_results.append(payload)
        if product_id:
            existing_ids.add(product_id)


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
        if name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}:
            continue
        tool_names.append(name)
        if name == "search_catalog_tool" and returned_results:
            has_search_result = True
    return (
        has_search_result
        and set(tool_names) == {"search_catalog_tool"}
    )


def _has_successful_non_search_tool_evidence(
    result: Any,
    *,
    request_id: str,
) -> bool:
    """Return whether another current-turn shopping tool completed."""

    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        tool_name = str(_value(message, "name") or "")
        if tool_name in {
            "",
            SKILL_ACTIVATION_TOOL_NAME,
            "read_file",
            "search_catalog_tool",
        }:
            continue
        if _tool_call_status(tool_name, message)[0] == "completed":
            return True
    return False


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
    unavailable_types = _no_direct_search_types(
        result,
        request_id=request_id,
    )
    if unavailable_types:
        lines.extend(("", "Unavailable requested catalog types:"))
        lines.extend(
            f"- **{product_type}** is not advertised; I did not substitute "
            "an adjacent product type."
            for product_type in unavailable_types
        )
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


def _no_direct_search_types(result: Any, *, request_id: str) -> list[str]:
    """Return current-turn product types with a server-authored no-direct outcome."""

    product_types: list[str] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message)
        if not payload or payload.get("outcome") != "no_direct_catalog_match":
            continue
        outcome = payload
        product_type = str(outcome.get("requested_product_type") or "").strip()
        if product_type and product_type not in product_types:
            product_types.append(product_type)
    return product_types


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


def _should_short_circuit_media_failure(state: State) -> bool:
    if not state.media or not state.media_analysis:
        return False
    if not any(str(item.get("type") or "").lower() == "video" for item in state.media):
        return False
    if not _media_analysis_unavailable(state.media_analysis):
        return False
    return _query_depends_on_media(state.query)


def _record_media_model_usage(state: State, config: Any) -> None:
    if not getattr(config, "vlm_enabled", False) or not getattr(config, "vlm_name", None):
        _add_model_usage(state, "vlm", status="disabled", calls=0)
        return

    status = "failed" if _media_analysis_unavailable(state.media_analysis) else "used"
    _add_model_usage(
        state,
        "vlm",
        status=status,
        calls=1,
        detail="Attached media understanding",
    )


def _record_catalog_model_usage(
    state: State,
    plan: CatalogSearchPlan,
    ok: bool,
    *,
    fallback_attempted: bool = False,
) -> None:
    status = "used" if ok else "failed"
    uses_image_endpoint = bool(state.image) and plan.search_mode in {"image", "hybrid"}
    text_embedding_calls = 1 if plan.semantic_queries else 0
    if uses_image_endpoint and text_embedding_calls == 0:
        # Image retrieval currently includes one deterministic text-side query
        # alongside the image embedding, even when the shopper supplied no text.
        text_embedding_calls = 1
    if fallback_attempted:
        text_embedding_calls += 1 if plan.semantic_queries else 0

    if text_embedding_calls:
        _add_model_usage(
            state,
            "text_embedding",
            status=status,
            calls=text_embedding_calls,
            detail="Catalog text/vector retrieval",
        )
    if uses_image_endpoint:
        _add_model_usage(
            state,
            "image_embedding",
            status=status,
            calls=1,
            detail="Catalog image similarity retrieval",
        )


def _record_language_model_failure(state: State) -> None:
    _add_model_usage(
        state,
        "app_llm",
        status="failed",
        calls=1,
        detail="Planning, tool use, and response generation failed",
    )


def _record_safety_model_usage(state: State, mode: str, *, ok: bool = True) -> None:
    status = "used" if ok else "failed"
    detail = "Input and output safety checks" if ok else "Guardrails check failed open"
    if mode == "input":
        _add_model_usage(
            state,
            "content_safety",
            status=status,
            calls=1,
            detail=detail,
        )
        _add_model_usage(
            state,
            "topic_control",
            status=status,
            calls=1,
            detail="Input topic check" if ok else "Guardrails topic check failed open",
        )
        return

    _add_model_usage(
        state,
        "content_safety",
        status=status,
        calls=1,
        detail=detail,
    )


def _add_model_usage(
    state: State,
    role: str,
    *,
    status: str,
    calls: int,
    detail: str = "",
) -> None:
    existing = state.model_usage.get(role, {})
    existing_calls = _safe_int(existing.get("calls"))
    existing_status = str(existing.get("status") or "")
    merged_status = _merged_model_usage_status(existing_status, status)
    existing_detail = str(existing.get("detail") or "")
    next_detail = (
        existing_detail
        if existing_status == merged_status and existing_detail
        else detail or existing_detail
    )
    state.model_usage[role] = {
        "status": merged_status,
        "calls": existing_calls + max(0, calls),
        "detail": next_detail,
    }


def _merged_model_usage_status(existing: str, current: str) -> str:
    priority = {"failed": 4, "used": 3, "disabled": 2, "not_used": 1, "": 0}
    return existing if priority.get(existing, 0) > priority.get(current, 0) else current


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _media_analysis_unavailable(media_analysis: str) -> bool:
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        text = media_analysis
    else:
        if not isinstance(parsed, dict):
            text = str(parsed)
        else:
            parts = [str(parsed.get("summary") or "")]
            uncertainties = parsed.get("uncertainties")
            if isinstance(uncertainties, list):
                parts.extend(str(item) for item in uncertainties)
            text = " ".join(parts)

    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "not configured",
            "unavailable",
            "authentication failed",
            "could not authenticate",
            "not allowed to access",
            "media analysis failed",
            "vlm returned no analysis",
        )
    )


def _query_depends_on_media(query: str) -> bool:
    lowered = (query or "").strip().lower()
    if not lowered or lowered == "the user submitted visual media without additional text.":
        return True
    return any(
        phrase in lowered
        for phrase in (
            "this video",
            "the video",
            "in video",
            "from video",
            "attached video",
            "she is wearing",
            "he is wearing",
            "they are wearing",
            "person is wearing",
            "like that",
            "like this",
            "similar to that",
            "similar to this",
            "what is in this",
            "what's in this",
            "what am i wearing",
            "what are they wearing",
            "describe this",
        )
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


def _collect_token_usage(result: Any) -> dict[str, int]:
    """Collect normalized token usage from LangChain/OpenAI message metadata."""

    usage = _empty_token_usage()
    for message in _result_messages(result):
        record = _message_token_usage_record(message)
        if not record:
            continue

        input_tokens = _token_int(record, ("input_tokens", "prompt_tokens", "input"))
        output_tokens = _token_int(
            record,
            ("output_tokens", "completion_tokens", "output"),
        )
        total_tokens = _token_int(record, ("total_tokens", "total"))
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue

        usage["input_tokens"] += int(input_tokens or 0)
        usage["output_tokens"] += int(output_tokens or 0)
        usage["total_tokens"] += int(total_tokens or 0)
        usage["model_calls"] += 1
    return usage


def _merge_token_usage(
    base: dict[str, Any] | None,
    additional: dict[str, Any] | None,
) -> dict[str, int]:
    merged = _normalized_token_usage(base)
    extra = _normalized_token_usage(additional)
    for key in merged:
        merged[key] += extra[key]
    return merged


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
    relation = _scope_relation_line(payload, has_products=True)
    if relation:
        lines.append(relation)
    audience = _assumed_audience_line(payload)
    if audience:
        lines.append(audience)
    return "\n".join(lines)


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
        "naming that as what you have assumed the shopper is looking for -- "
        "\"assuming you're looking for ... clothes\" -- and invite them to "
        "correct it. It is an assumption about what they want, not a note "
        "about the shop's style or about what this reply happened to return, "
        "and not a question about who they are. Say which of these suit a "
        "wider audience if any do. Put it in a shopper's words, never a "
        "catalog label: a value such as adult_all_genders is said as pieces "
        "anyone can wear."
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
    if tool_name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}:
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


def _normalized_token_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    usage = _empty_token_usage()
    if not isinstance(raw, dict):
        return usage
    for key in usage:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            usage[key] = max(0, int(value))
    return usage


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
    }


def _message_token_usage_record(message: Any) -> Any:
    usage_metadata = _value(message, "usage_metadata")
    if usage_metadata:
        return usage_metadata

    response_metadata = _value(message, "response_metadata")
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage:
            return token_usage

    for key in ("token_usage", "usage"):
        record = _value(message, key)
        if record:
            return record
    return None


def _token_int(record: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _value(record, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None


_NON_ATTRIBUTE_SEARCH_KEYS = frozenset({"catalog_text", "similarity", "taxonomy"})


def _search_attribute_facts(product: Any) -> dict[str, str]:
    """Structured attributes the catalog confirmed for one search hit.

    The catalog declares which fields are product detail and returns them with
    every search result. They were dropped here, so the model was told to spend
    one of its two product-detail reads to fetch what the response already
    carried -- and when that budget ran out it reported a confirmed attribute as
    unknown.
    """

    attributes = getattr(product, "attributes", None)
    if not isinstance(attributes, dict):
        return {}
    facts: dict[str, str] = {}
    for name, value in sorted(attributes.items()):
        if name in _NON_ATTRIBUTE_SEARCH_KEYS:
            continue
        text = _format_detail_value(value).strip()
        if text:
            facts[str(name)] = text
    return facts


def _search_product_record(product: Any) -> dict[str, Any]:
    """Project one search hit into the record both the model text and the
    composer summary are rendered from.

    Previously the model-visible text was the only rendering and the composer
    parsed it back into this same shape. Building the record once removes the
    round trip, and keeps the two renderings unable to disagree.
    """

    return {
        "product_ref": str(product.product_id),
        "name": str(product.display_name),
        "category": str(getattr(product, "category", "") or ""),
        "price": (
            f"${product.price.amount:.2f} {product.price.currency}"
            if product.price
            else ""
        ),
        "image_url": str(product.image_url or ""),
        "attributes": _search_attribute_facts(product),
    }


def _format_product(product: Any) -> str:
    return _format_product_record(_search_product_record(product))


def _product_detail_record(product: ProductDetail) -> dict[str, Any]:
    """Project one product-detail read into the record the text renders from."""

    return {
        "product_ref": str(product.product_id),
        "name": str(product.display_name),
        "category": str(product.category or ""),
        "brand": str(product.brand or ""),
        "price": (
            f"${product.price.amount:.2f} {product.price.currency}"
            if product.price
            else ""
        ),
        "image_url": str(product.image_url or ""),
        "details": [
            f"{name.replace('_', ' ')}: {_format_detail_value(value)}"
            for name, value in sorted((product.attributes or {}).items())
        ],
    }


def _format_product_details(product: ProductDetail) -> str:
    return _format_product_detail_record(_product_detail_record(product))


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
) -> list[str]:
    explicitly_named = _explicitly_named_products(user_query, available_products)
    if not explicitly_named:
        return []

    explicit_names = {
        _normalize_product_name(product.display_name) for product in explicitly_named
    }
    failures = []
    for product_ref, product in requested_products:
        if _normalize_product_name(product.display_name) in explicit_names:
            continue
        failures.append(
            f"- PRODUCT_REF '{product_ref}': selected '{product.display_name}' is "
            "outside the current explicit add request. The current request names: "
            f"{_format_product_refs(explicitly_named)}. Retry with matching "
            "PRODUCT_REF values only, or ask a clarification."
        )
    return failures


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


def _same_product_display_name(expected: str, actual: str) -> bool:
    return _normalize_product_name(expected) == _normalize_product_name(actual)


def _product_detail_failure_message(
    error: CommerceError | None,
    *,
    cart_validation: bool,
) -> str:
    if error is not None and error.code == "product_not_found":
        return (
            "The product is no longer present in the active catalog. "
            "Search again before adding it."
            if cart_validation
            else (
                "That product is no longer available in the active catalog. "
                "Search the catalog again before using its details."
            )
        )
    if error is not None and error.retryable:
        return (
            "The catalog is temporarily unavailable, so the cart was not changed. "
            "Please try again."
            if cart_validation
            else "Product details are temporarily unavailable. Please try again."
        )
    return (
        "The product could not be verified, so the cart was not changed. "
        "Search again before adding it."
        if cart_validation
        else "Product details could not be verified. Search the catalog again."
    )


def _normalize_product_name(value: str) -> str:
    chars = []
    for char in str(value or "").casefold():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def _product_name_tokens(value: str) -> list[str]:
    return [
        token
        for token in _normalize_product_name(value).split()
        if token not in _PRODUCT_NAME_STOPWORDS
    ]


def _product_name_tokens_match(
    query_tokens: set[str],
    product_tokens: list[str],
    *,
    required_overlap: int,
) -> bool:
    if len(product_tokens) < 2:
        return False
    overlap = query_tokens.intersection(product_tokens)
    required = min(required_overlap, len(set(product_tokens)))
    return len(overlap) >= required


def _scrub_internal_shopper_language(text: str) -> str:
    scrubbed = text or ""
    for internal, replacement in _INTERNAL_SHOPPER_REPLACEMENTS:
        scrubbed = scrubbed.replace(internal, replacement)
    return scrubbed


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


_SEARCH_RESULT_GROUNDING_NOTE = (
    "SEARCH_RESULT_GROUNDING_NOTE: Use search results for candidate names, prices, "
    "categories, image availability, confirmed filters listed in "
    "SEARCH_FILTER_EVIDENCE, advertised taxonomy listed in "
    "SEARCH_TAXONOMY_EVIDENCE, and modest styling fit only. Treat product names as "
    "display names, not attribute evidence. Do not infer or group-claim "
    "length, color, print, material, care, construction, fit, comfort, weather, "
    "grass, gravel, or best-in-category performance from names or search snippets. "
    "Do not override a confirmed filter based on words in a display name."
)

_SEARCH_NO_MATCH_GROUNDING_NOTE = (
    "SEARCH_NO_MATCH_GROUNDING_NOTE: Zero products matched this exact "
    "advertised taxonomy and filter scope. This result does not establish "
    "whether products exist in a different, unsearched, or unadvertised "
    "product type."
)

_SEARCH_SCOPE_COMPLETE_NOTE = (
    "SEARCH_SCOPE_COMPLETE: The shopper's current request can now be answered "
    "from this search and existing turn evidence. Answer now. Do not search an "
    "adjacent category or substitute merely because search budget remains. Use "
    "the direct antecedent from recent discussion as the styling anchor; an item "
    "does not need to be in the cart to receive styling advice."
)

_SEARCH_BUDGET_EXHAUSTED_NOTE = (
    f"{SEARCH_BUDGET_EXHAUSTED_PREFIX} No additional catalog searches are "
    "available this turn. Continue with any requested non-search action, or "
    "answer honestly from the grounded products already returned."
)


def _detail_fields_already_held(
    product: Any,
    capabilities: CatalogCapabilities,
) -> bool:
    """Return whether evidence already holds every advertised detail field.

    A search returns the same attributes a detail read does -- measured across
    all five advertised categories and 20 products, the detail-only set was
    empty -- so re-reading spends a model round trip to learn what is already in
    hand.

    But "empty on this catalog" is not "empty on every catalog", and silently
    dropping a field is worse than a redundant call. So this asks the capability
    contract rather than assuming: only when evidence covers every field the
    product's own category advertises as a detail field is the read redundant.
    Any gap, any unknown category, and the fetch goes ahead.
    """

    held = getattr(product, "attributes", None)
    if not held:
        return False
    category = getattr(product, "category", None)
    taxonomy = capabilities.taxonomy
    advertised: set[str] = set()
    for name, entry in taxonomy.categories.items():
        subcategories = getattr(entry, "subcategories", {}) or {}
        if category not in (name, *subcategories):
            continue
        advertised |= {
            field
            for field, capability in entry.filters.items()
            if getattr(capability, "detail", False)
        }
    if not advertised:
        return False
    return advertised <= set(held)
