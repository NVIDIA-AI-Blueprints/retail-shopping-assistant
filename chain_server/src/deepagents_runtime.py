# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, AsyncIterator, Literal
import unicodedata
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)
import requests

from .agenttypes import Cart, State
from .catalog_capabilities import (
    CatalogCapabilitiesClient,
    effective_filter_capabilities,
    format_catalog_capabilities_for_prompt,
)
from .catalog_execution import execute_catalog_search
from .catalog_request import (
    CatalogSearchIntent,
    CatalogSearchPlan,
    build_catalog_search_plan,
)
from .commerce_tools import (
    add_cart_item,
    check_product_availability,
    get_cart,
    get_product_details,
    get_store_policy,
    remove_cart_item,
    update_cart_item,
)
from .conversation_memory import (
    ConversationMemoryClient,
    ConversationMemoryError,
    FinalTurnStatus,
    TurnReplayOutput,
    TurnStartResult,
    format_conversation_context,
)
from .media_perception import MediaPerceptionClient
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_TOOL_NOT_GRANTED,
    ShopperSkillActivationError,
    ShopperSkillActivationMiddleware,
)
from .tool_policy import (
    load_shopper_skill_registry as _shopper_skill_registry,
    validate_registered_tool_names,
)
from .tool_loop_control import (
    CONSTRAINT_REVIEW_PREFIX,
    EXPLICIT_ALTERNATIVE_CORRECTION_PREFIX,
    SEARCH_BUDGET_EXHAUSTED_PREFIX,
    SEARCH_SCOPE_COMPLETE_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
    SERVER_RESTORED_TOOL_CALL_FIELDS,
    ToolLoopControlMiddleware,
    _SERVER_REJECTED_TOOL_CALLS,
)
from shared.commerce_contracts import (
    AddCartItemInput,
    CatalogCapabilities,
    Cart as CommerceCart,
    CartMutationResult,
    CheckProductAvailabilityInput,
    CheckProductAvailabilityResult,
    CommerceError,
    GetCartInput,
    GetProductDetailsInput,
    GetStorePolicyInput,
    GetStorePolicyResult,
    ProductDetail,
    ProductSummary,
    RemoveCartItemInput,
    UpdateCartItemInput,
)


logger = logging.getLogger(__name__)
_SHOPPER_SKILLS_ENV = "SHOPPER_SKILLS_ROOT"
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


try:
    from deepagents.backends import FilesystemBackend as _FilesystemBackend
except Exception:  # pragma: no cover - dependency import is validated at runtime.
    _FilesystemBackend = None

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
_SEARCH_FILTER_EVIDENCE_PREFIX = "SEARCH_FILTER_EVIDENCE:"
_SEARCH_TAXONOMY_EVIDENCE_PREFIX = "SEARCH_TAXONOMY_EVIDENCE:"
_SEARCH_DIRECTION_EVIDENCE_PREFIX = "SEARCH_DIRECTION_EVIDENCE:"
_SEARCH_GUIDANCE_EVIDENCE_PREFIX = "SEARCH_GUIDANCE_EVIDENCE:"
_CATALOG_SCOPE_OUTCOME_PREFIX = "CATALOG_SCOPE_OUTCOME:"
_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE = 24
_MAX_DIAGNOSTIC_PRODUCT_FACTS = 40
_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS = 500
_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE_CHARS = 32_000
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
_PRODUCT_DETAIL_GROUNDING_NOTE = (
    "PRODUCT_DETAIL_GROUNDING_NOTE: This detail result exposes only "
    "the fields shown below. Material, care, dimensions, closures, fit, "
    "sizing, colorways, and outdoor performance are unavailable unless explicitly "
    "listed. Do not infer them from product names or prior marketing text."
)
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
_UNSUPPORTED_REQUIREMENT_RESPONSE = (
    "I can't guarantee that requirement from the catalog information available "
    "to this assistant, so I won't present unverified matches. Would you like me "
    "to treat it as a preference and show candidates to verify on their product "
    "pages?"
)
_GROUNDING_EDITOR_SYSTEM_PROMPT = """You are a final response editor for a retail shopping assistant.

Rewrite the draft response only as needed so it is grounded in TOOL EVIDENCE
and CURRENT CART. Keep the shopper's requested task and any successful cart
action intact.

Rules:
- Return only the final shopper-facing response text.
- For a styling request, answer the styling question rather than returning a raw
  product list. Connect candidates to the shopper's goal or direct antecedent
  using category/role, exact confirmed filters, and general styling judgment.
  Keep styling judgment visibly separate from catalog facts and never derive it
  from words parsed out of a display name.
- Labeling text as styling judgment does not permit display-name inference. If
  the evidence does not distinguish candidates, give a useful group-level
  rationale and offer a detail check instead of inventing item-level differences.
- Do not add products, prices, cart actions, or product facts absent from TOOL
  EVIDENCE or CURRENT CART.
- CURRENT-TURN TOOL EVIDENCE is the only evidence for a search or mutation in
  this turn. PRIOR-TURN TOOL EVIDENCE may support direct references to products
  previously shown, but it does not prove that a new search or mutation ran.
- If TOOL EVIDENCE says there is no direct advertised taxonomy match for one
  requested role, do not claim a search ran for that role. Report that role's
  gap, preserve any other successful current-turn role, and ask whether to
  search a different advertised type. Do not name alternatives unless their
  exact taxonomy values appear in TOOL EVIDENCE.
- A scoped zero-result search proves only that its exact advertised taxonomy
  and filter scope returned no products. It does not prove that a different,
  unsearched, or unadvertised product type is absent, and it never supports a
  catalog-wide availability claim.
- Use RECENT DISCUSSION to resolve direct references such as "that" and
  "those." A discussed product or styling anchor does not need to be in CURRENT
  CART. RECENT DISCUSSION cannot establish whether the current search succeeded
  or supply the current turn's candidates. Do not introduce an absent-cart
  caveat unless the shopper asks about the cart or requests a cart mutation.
- Remove PRODUCT_REF, CART_LINE_ID, tool names, and internal IDs.
- Remove internal skill, mode, evaluator, judge, cache, backend, tool-evidence,
  structured-field, and data-layer language. Use shopper-safe phrasing such as
  "I don't have fabric or care details available for that item."
- If the draft says "product detail tool", "catalog detail tool", "the tool
  requires", or similar internal mechanics, rewrite it into shopper-safe
  language without the word "tool".
- If a product appears only in search results, you may state only its name,
  price, category/role, image availability, exact values in confirmed search-
  filter evidence, and a modest styling reason. Every other word in its display
  name is non-evidence.
- Confirmed search-filter evidence applies to every product returned by that
  search. Preserve it and do not contradict it. One allowed value confirms that
  value; multiple allowed values prove only membership in the set, not which
  value each product has. Do not infer adjacent attributes that the evidence
  does not name.
- For styling, preserve a concise candidate set and the draft's grounded styling
  rationale. Do not omit or override a confirmed filter merely because words in
  a display name appear to conflict. If that visible conflict matters
  to the request, flag it as catalog information worth verifying rather than
  resolving it from the name.
- For each search-only candidate, delete descriptive sentences that merely
  restate or interpret words in the display name. Keep the name, price,
  category/role, confirmed filters, and a modest reason tied to the shopper's
  stated goal.
- In a search-only response, copy a candidate's display name only as its exact
  title. Do not shorten it into an attribute, classify or group candidates by
  words appearing in their names, or use those words as the reason one
  candidate differs from another. Without product details, give one concise
  group-level styling rationale based on the shopper's goal, advertised role,
  and confirmed filters instead of item-specific attribute rationales.
- Advertised search-taxonomy evidence lists the valid product types used by that
  search. Do not call an unlisted product type advertised or offer it as an
  advertised alternative.
- Treat product names as display names, not proof of length, color, print,
  material, construction, fit, care, or vibe. Do not say a product is solid,
  floral, gingham, maxi, knee-length, woven, structured, neutral, lightweight,
  polished, or dressier unless that attribute appears in product-detail evidence.
- Material, care, dimensions, pockets, closures, fit, comfort, and outdoor
  practicality claims require matching product-detail evidence and a direct
  shopper need for that fact.
- Group claims such as "all are maxi length", "both are cotton", "the lightest",
  "most polished", or "best for heat" require product-detail evidence for every
  item included in that claim. Remove the claim if any item lacks that support.
- Do not say an item is stable on grass or gravel, water-resistant, bug-safe,
  all-day comfortable, maximally breathable, or best-in-category unless the
  evidence explicitly says that exact claim.
- Do not convert indirect evidence into outdoor surface performance. If the
  evidence says flat sole, ankle strap, linen, cotton, or elastic waistband,
  state only that fact when needed; do not add grass, gravel, outdoor-surface,
  heat, or all-evening performance claims. Avoid phrases such as "works well
  for outdoor surfaces"; use "a flat shoe option" or "fits the practical
  direction" instead.
- The shopper and Judge see only the final answer, not hidden tool output. Avoid
  long product-spec dumps; keep catalog facts item-specific and visibly modest.
- If image evidence is available, do not say images are unavailable or that you
  cannot show them. Say the product image should appear with the result, or
  simply answer the comparison.
- Preserve exact cart totals and cart contents when they are present in CURRENT
  CART or tool evidence.
- If the draft is already compliant, return it unchanged.
"""
_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {"write_todos", "ls", "write_file", "edit_file", "glob", "grep", "execute"}
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
            "taxonomy_status is 'image_only' or 'no_direct_catalog_match'."
        ),
    )
    subcategory: list[str] = Field(
        ...,
        description=(
            "Exact advertised subcategory values required by the shopper. Use an "
            "empty list when category supplies the text-search scope or the search "
            "status is 'image_only' or 'no_direct_catalog_match'."
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
        + ". Ask the shopper whether to treat it as a preference."
    )


def _recent_shopper_statements(context: str, *, limit: int = 4) -> str:
    """Extract prior shopper text without exposing assistant responses."""

    statements = re.findall(
        r"(?:^|\n)User:\s*(.*?)(?=\nAssistant:|\nUser:|\Z)",
        context or "",
        flags=re.DOTALL,
    )
    compact = [" ".join(statement.split()) for statement in statements]
    return "\n".join(statement for statement in compact[-limit:] if statement)


def _shopper_stated_product_scope(
    query: str,
    context: str,
    product_scope_key: str,
) -> bool:
    """Return whether current or recent shopper text states a product scope."""

    shopper_text = "\n".join(
        value for value in (query, _recent_shopper_statements(context)) if value
    )
    return _text_mentions_product_type(shopper_text, product_scope_key)


def _resolved_agent_selected_product_type(
    *,
    query: str,
    context: str,
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Derive open-role provenance from the agent's single taxonomy choice."""

    if taxonomy_status != "agent_selected_type":
        return requested_product_type
    scope_key = _product_scope_key(requested_product_type)
    if scope_key and _shopper_stated_product_scope(query, context, scope_key):
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


def _explicit_advertised_alternatives(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> tuple[str, list[str]] | None:
    """Resolve literal advertised alternatives owned by one category."""

    normalized = _normalize_product_text(requested_product_type or "")
    parts = re.split(r"\b(?:and|or)\b", normalized)
    if len(parts) < 2 or any(not part.strip() for part in parts):
        return None

    matches: list[tuple[str, str]] = []
    for part in parts:
        normalized_part = part.strip()
        part_matches = [
            (category_name, subcategory_name)
            for category_name, category in capabilities.taxonomy.categories.items()
            for subcategory_name in category.subcategories
            if normalized_part == _normalize_product_text(subcategory_name)
        ]
        if len(part_matches) != 1:
            return None
        matches.append(part_matches[0])

    category_names = {category_name for category_name, _ in matches}
    subcategories = list(dict.fromkeys(name for _, name in matches))
    if len(category_names) != 1 or len(subcategories) != len(matches):
        return None
    return matches[0][0], subcategories


def _explicit_advertised_alternatives_in_text(
    text: str,
    capabilities: CatalogCapabilities,
) -> tuple[str, list[str]] | None:
    """Find one literal advertised alternative pair in shopper text."""

    normalized = _normalize_product_text(text)
    matches: list[tuple[str, list[str]]] = []
    for category_name, category in capabilities.taxonomy.categories.items():
        subcategories = list(category.subcategories)
        for first in subcategories:
            normalized_first = _normalize_product_text(first)
            for second in subcategories:
                if first == second:
                    continue
                normalized_second = _normalize_product_text(second)
                pattern = (
                    rf"(?<!\w){re.escape(normalized_first)}\s+"
                    rf"(?:and|or)\s+{re.escape(normalized_second)}(?!\w)"
                )
                if re.search(pattern, normalized):
                    matches.append((category_name, [first, second]))
    return matches[0] if len(matches) == 1 else None


def _advertised_alternative_selection_issue(
    requested_product_type: str | None,
    alternatives: tuple[str, list[str]],
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Require an exact taxonomy projection for literal alternatives."""

    category_name, advertised_subcategories = alternatives
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected_categories = {
        _normalize_product_text(value)
        for value in (payload.get("category") or [])
    }
    selected_subcategories = {
        _normalize_product_text(value)
        for value in (payload.get("subcategory") or [])
    }
    expected_subcategories = {
        _normalize_product_text(value) for value in advertised_subcategories
    }
    requested_parts = {
        part.strip()
        for part in re.split(
            r"\b(?:and|or)\b",
            _normalize_product_text(requested_product_type or ""),
        )
        if part.strip()
    }
    category_matches = selected_categories in (
        set(),
        {_normalize_product_text(category_name)},
    )
    if (
        requested_parts == expected_subcategories
        and taxonomy_status == "member_of_requested_umbrella"
        and category_matches
        and selected_subcategories == expected_subcategories
    ):
        return None
    return (
        "The current shopper request names exact advertised alternatives "
        f"{advertised_subcategories!r} in category '{category_name}'. "
        "Set requested_product_type to include every named alternative. Select "
        "every named alternative exactly once and no other subcategory, using "
        "member_of_requested_umbrella."
    )


def _alternative_candidate_limit(
    alternatives: tuple[str, list[str]] | None,
    capabilities: CatalogCapabilities,
    default: int,
) -> int:
    """Fetch enough ranked candidates to cover literal alternatives."""

    if alternatives is None:
        return default
    category_name, subcategories = alternatives
    category = capabilities.taxonomy.categories[category_name]
    product_count = sum(
        category.subcategories[name].product_count for name in subcategories
    )
    return min(50, max(default, len(subcategories), product_count))


def _products_with_alternative_coverage(
    products: list[ProductSummary],
    alternatives: tuple[str, list[str]] | None,
    limit: int,
) -> list[ProductSummary]:
    """Keep rank order while reserving one result per returned alternative."""

    if alternatives is None or len(products) <= limit:
        return products
    _, subcategories = alternatives
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

    alternatives = _explicit_advertised_alternatives(
        requested_product_type,
        capabilities,
    )
    if alternatives is not None:
        return _advertised_alternative_selection_issue(
            requested_product_type,
            alternatives,
            taxonomy_status,
            taxonomy,
        )

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
                "taxonomy contains only its advertised children. Preserve that "
                "taxonomy and change only taxonomy_status to "
                "member_of_requested_umbrella."
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


class SearchCatalogToolArguments(BaseModel):
    """Agent-facing search arguments narrowed to the active catalog schema."""

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
            "schemas, filters, evidence, or identifiers. Use an empty string for "
            "image-only or no-direct requests."
        ),
    )
    requested_product_type: str | None = Field(
        ...,
        description=(
            "Shortest product noun or umbrella phrase for this focused role, "
            "resolved from the shopper's current turn or direct antecedent. "
            "Exclude color, material, fit, occasion, weather, and style modifiers: "
            "'formal tops' and 'relaxed-fit tops' both use 'tops'. For a genuinely "
            "open role selected by the agent, use the chosen advertised role noun. "
            "This is provenance, not catalog taxonomy or a ranking query. Use null "
            "only for image-only search."
        ),
    )
    taxonomy_status: Literal[
        "exact_requested_type",
        "member_of_requested_umbrella",
        "agent_selected_type",
        "no_direct_catalog_match",
        "image_only",
    ] = Field(
        ...,
        description=(
            "Direction-aware semantic relation to the shopper's requested product "
            "type. Use 'exact_requested_type' when the selected taxonomy directly "
            "represents the requested focused role. Use "
            "'member_of_requested_umbrella' when the shopper named a true umbrella "
            "or explicit alternatives and every selected value is a faithful child "
            "or named alternative; do not "
            "reverse this relation merely because the requested type is a child of "
            "a selected parent category. Apply this direction test: each selected "
            "catalog value must answer yes to 'is this a kind of the product scope "
            "the shopper asked for?' A skirt is a kind of bottom; a dress is not a "
            "kind of bottom; a flat or sandal is not a kind of sneaker even though "
            "all are footwear. Use 'agent_selected_type' only when a broad "
            "styling or discovery request names no concrete product type; choose "
            "one exact advertised subcategory as the focused starting role. "
            "Evaluate every product role named "
            "in the current turn independently. If the shopper names a product "
            "type for a role, including as an alternative, confirmation, "
            "comparison, or follow-up, 'agent_selected_type' is forbidden for "
            "that role. Never use it to replace an explicitly requested "
            "unavailable type. Use "
            "'no_direct_catalog_match' only for an explicitly requested concrete "
            "product type when only parent, adjacent, or substitute types exist. "
            "Decide that from the product type alone and send no required "
            "constraints on this no-retrieval path. "
            "A modifier does not erase an advertised product type: for example, "
            "if skirts are advertised but denim is not enforceable, keep skirts "
            "as taxonomy and preserve denim as an unadvertised requirement. "
            "Never use it for an outfit, occasion, season, weather need, style/vibe, "
            "or product attribute. Use 'image_only' only "
            "when no text query or taxonomy scope is needed."
        ),
    )
    taxonomy: CatalogTaxonomyToolInput = Field(
        ...,
        description=(
            "Required catalog-derived taxonomy selection. Allowed category and "
            "subcategory values come from the active catalog capabilities. Every "
            "selected value must be the requested product type or a child of an "
            "umbrella the shopper actually named, except for advertised values "
            "serving one focused role chosen under agent_selected_type. Never "
            "select a parent "
            "or sibling as a substitute. For example, skirts may satisfy bottoms; "
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
            "The active catalog replaces this base field with its advertised hard "
            "filters and an explicit lane for directly stated requirements it "
            "cannot enforce. Season, weather, occasion, and subjective style/vibe "
            "context remain semantic direction unless the shopper directly "
            "requires an objective product attribute. A defining material is a "
            "must-have: for 'Any denim skirts available?', put 'denim' in "
            "unadvertised_requirements when composition is not a hard filter."
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
                    f"{self.taxonomy_status} requires empty shopper_guidance"
                )
        elif not has_shopper_guidance:
            raise ValueError(
                "catalog retrieval requires non-empty shopper_guidance"
            )
        if self.taxonomy_status != "image_only" and not requested_product_type:
            raise ValueError(
                f"{self.taxonomy_status} requires a requested product type"
            )
        if self.taxonomy_status == "image_only" and requested_product_type:
            raise ValueError(
                "image_only requires requested_product_type=null"
            )
        if self.taxonomy_status == "no_direct_catalog_match":
            if has_taxonomy:
                raise ValueError(
                    "no_direct_catalog_match requires empty taxonomy arrays"
                )
            if not has_query:
                raise ValueError(
                    "no_direct_catalog_match requires a requested product type"
                )
            constraints = (
                self.required_constraints.model_dump(exclude_none=True)
                if isinstance(self.required_constraints, BaseModel)
                else self.required_constraints
            )
            repeated_product_types = constraints.get(
                "unadvertised_requirements",
                [],
            )
            requested_product_words = _normalize_product_text(
                requested_product_type
            ).split()
            has_other_constraint = any(
                value not in (None, "", [], {})
                for name, value in constraints.items()
                if name != "unadvertised_requirements"
            )
            has_distinct_requirement = any(
                not requested_product_words
                or requested_product_words[-1]
                not in _normalize_product_text(str(value)).split()
                for value in repeated_product_types
            )
            if has_other_constraint or has_distinct_requirement:
                raise ValueError(
                    "no_direct_catalog_match cannot include required constraints"
                )
            return self
        if self.taxonomy_status == "image_only":
            if has_query or has_taxonomy:
                raise ValueError(
                    "image_only requires an empty semantic query and taxonomy"
                )
            return self
        if self.taxonomy_status == "member_of_requested_umbrella" and not (
            self.taxonomy.subcategory
        ):
            raise ValueError(
                "member_of_requested_umbrella requires an advertised subcategory"
            )
        if self.taxonomy_status == "agent_selected_type" and not (
            self.taxonomy.subcategory
        ):
            raise ValueError(
                "agent_selected_type requires an advertised subcategory"
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
            "exact_requested_type requires one matching taxonomy value unless "
            "the shopper requested the selected category. Use "
            "member_of_requested_umbrella only when every selected value is a "
            "true child; otherwise use no_direct_catalog_match"
        )
    selected_product_type = selected_product_types[0]
    if _normalize_product_text(requested_product_type) == _normalize_product_text(
        selected_product_type
    ):
        return None
    return (
        "a single exact_requested_type taxonomy value requires "
        "requested_product_type to match that product type. Use "
        "member_of_requested_umbrella only when every selected value is a true "
        "child of the requested type; otherwise use no_direct_catalog_match"
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
                    "category or subcategory unless taxonomy_status is "
                    "'no_direct_catalog_match'; both arrays may be empty for that "
                    "no-retrieval result or when taxonomy_status is 'image_only'."
                    " Every value must be a kind of the requested scope: skirts "
                    "may satisfy bottoms; dresses may not. For a broad request "
                    "that names no product type, agent_selected_type chooses one "
                    "exact advertised subcategory as the focused starting role."
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
                    "and include every exact matching filter value."
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
        "taxonomy_status is 'image_only' or 'no_direct_catalog_match'."
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


def _required_constraints_input_model(
    capabilities: CatalogCapabilities,
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
                description=f"Advertised hard filter '{name}'.",
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
                "supported advertised branch here."
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
            raise ValueError(
                "select exactly one primary procedure: outfit-styling or "
                "product-discovery, never both"
            )
        if "budget-shopping" in selected and len(primary) != 1:
            raise ValueError(
                "budget-shopping requires exactly one primary procedure: "
                "outfit-styling or product-discovery"
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


class AddCartItemsToolInput(BaseModel):
    items: list[AddCartItemsToolItemInput] = Field(
        ...,
        min_length=1,
        description=(
            "One or more catalog products to add. Each product must use a "
            "PRODUCT_REF returned by search_catalog_tool."
        ),
    )


class _UpdateCartItemsInput(BaseModel):
    cart_line_id: str = Field(
        description="CART_LINE_ID from get_cart_tool. Not the product name."
    )
    quantity: int = Field(
        ge=0,
        description="New total quantity. Set to 0 to remove the line.",
    )


class _GetStorePolicyInput(BaseModel):
    topic: Literal[
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    ] = Field(description="Policy topic to look up.")


class _CheckAvailabilityInput(BaseModel):
    product_ref: str = Field(
        description="PRODUCT_REF from a prior catalog search in this conversation."
    )
    variant_hint: str | None = Field(
        default=None,
        description="Requested size wording, such as 'size 8'.",
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

    @property
    def legacy_user_id(self) -> int:
        return self.context_user_id


def _conversation_turn_status(termination_reason: str) -> FinalTurnStatus:
    if termination_reason in {
        "input_guardrail_blocked",
        "output_guardrail_blocked",
    }:
        return "blocked"
    if termination_reason == "completed":
        return "completed"
    return "failed"


class DeepAgentsRuntime:
    """Small adapter around the Deep Agents SDK.

    This class intentionally keeps commerce truth in existing services. Deep
    Agents gets scoped context and deterministic tools; it does not own carts,
    profiles, prices, inventory, or session identity.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._checkpointer = _build_checkpointer()
        self._profile_registered = False
        self._product_refs: dict[str, dict[str, ProductSummary]] = {}
        self._media_perception = MediaPerceptionClient(config)
        self._catalog_capabilities = CatalogCapabilitiesClient(
            config.retriever_port,
            timeout_seconds=config.catalog_search_timeout_seconds,
        )
        self._conversation_memory = ConversationMemoryClient(config.memory_port)

    def catalog_capabilities(self) -> CatalogCapabilities:
        """Return the process-lifecycle catalog capability contract."""

        return self._catalog_capabilities.get()

    async def astream(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> AsyncIterator[str]:
        output = await self._run_turn(state, identity)
        products = output.product_results or []
        if products:
            yield json.dumps(
                {"type": "products", "payload": products, "timestamp": time.time()}
            )
        images = output.retrieved or {}
        yield json.dumps({"type": "images", "payload": images, "timestamp": time.time()})
        if output.response:
            yield json.dumps(
                {"type": "content", "payload": output.response, "timestamp": time.time()}
            )
        yield json.dumps(
            {
                "type": "metrics",
                "payload": {
                    "timings": output.timings,
                    "total_seconds": sum(output.timings.values()),
                    "token_usage": _normalized_token_usage(output.token_usage),
                    "model_usage": output.model_usage,
                    "agent_diagnostics": output.agent_diagnostics,
                },
                "timestamp": time.time(),
            }
        )

    async def ainvoke(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> dict[str, Any]:
        output = await self._run_turn(state, identity)
        return {
            "response": output.response,
            "images": output.retrieved or {},
            "cart": output.cart.model_dump(mode="json"),
            "timings": output.timings,
            "token_usage": _normalized_token_usage(output.token_usage),
            "model_usage": output.model_usage,
            "agent_diagnostics": output.agent_diagnostics,
        }

    async def _run_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> State:
        state.user_id = identity.context_user_id
        state.agent_diagnostics = _empty_agent_diagnostics("not_started")
        turn = self._start_conversation_turn(state, identity)
        if turn is not None and turn.replayed:
            return self._restore_replayed_turn(state, turn)
        if turn is None and state.response:
            return state

        try:
            output = await self._execute_turn(state, identity)
        except asyncio.CancelledError:
            if turn is not None:
                if not state.response:
                    state.response = (
                        "This request was interrupted. Please check your cart "
                        "before retrying."
                    )
                self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="request_cancelled",
                )
            raise
        except Exception:
            if turn is not None:
                self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="unexpected_runtime_error",
                )
            raise

        if turn is not None:
            self._finalize_conversation_turn(state, identity, turn)
        return output

    async def _execute_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> State:
        start = time.monotonic()

        if state.guardrails:
            safety_start = time.monotonic()
            input_safe, input_check_ok = self._check_safety(
                "input",
                identity.context_user_id,
                state.query,
            )
            state.timings["safety_input"] = time.monotonic() - safety_start
            _record_safety_model_usage(state, "input", ok=input_check_ok)
            if not input_safe:
                state.response = self.config.unsafe_message
                state.timings["deepagents"] = time.monotonic() - start
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "input_guardrail_blocked"
                )
                return state

        media_start = time.monotonic()
        state.media_analysis = await self._media_perception.analyze(state)
        if state.media:
            state.timings["media_perception"] = time.monotonic() - media_start
            _record_media_model_usage(state, self.config)
        if _should_short_circuit_media_failure(state):
            state.response = _media_failure_response(state.media_analysis)
            state.timings["deepagents"] = time.monotonic() - start
            state.agent_diagnostics = _empty_agent_diagnostics("media_failure")
            return state

        turn_capabilities = await asyncio.to_thread(self._catalog_capabilities.get)
        invoke_config = {
            "configurable": {"thread_id": identity.conversation_id},
            "recursion_limit": self.config.deepagents_recursion_limit,
        }
        agent = None
        try:
            agent = self._create_agent(
                state,
                identity,
                turn_capabilities,
            )
            input_message = self._build_user_message(state, identity)
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": input_message}]},
                config=invoke_config,
            )
            draft_response = _extract_final_text(result)
            state.token_usage = _collect_token_usage(result)
            state.agent_diagnostics = _safe_collect_agent_diagnostics(
                _result_messages(result),
                request_id=identity.request_id,
                final_termination_reason="completed",
            )
            state.response = (
                _no_direct_taxonomy_response(
                    result,
                    request_id=identity.request_id,
                )
                or _unsupported_requirement_response(
                    result,
                    request_id=identity.request_id,
                )
                or self._rewrite_response_for_grounding(
                    state,
                    result,
                    draft_response,
                    request_id=identity.request_id,
                )
            )
            if not state.response:
                state.response = (
                    "I could not complete that shopping request. Please try again."
                )
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = "incomplete_agent_response"
        except Exception as exc:  # noqa: BLE001 - keep endpoint resilient.
            partial_messages, capture_error = await _partial_graph_messages(
                agent,
                invoke_config,
            )
            if isinstance(exc, GraphRecursionError):
                termination_reason = "recursion_limit"
            elif isinstance(exc, ShopperSkillActivationError):
                termination_reason = "skill_activation_failed"
            else:
                termination_reason = "agent_error"
            state.agent_diagnostics = _safe_collect_agent_diagnostics(
                partial_messages,
                request_id=identity.request_id,
                final_termination_reason=termination_reason,
                preserve_partial_messages=True,
            )
            if capture_error:
                state.agent_diagnostics["partial_graph_capture_error"] = capture_error
            logger.exception("DeepAgentsRuntime failed")
            await self._reset_agent_thread(identity)
            fallback_response = _partial_product_results_response(state)
            state.response = fallback_response or (
                "I encountered an error while helping with your shopping request. "
                "Please try again."
            )
            _record_language_model_failure(state)
            state.timings["deepagents_error"] = time.monotonic() - start
            return state

        if state.guardrails:
            safety_start = time.monotonic()
            output_safe, output_check_ok = self._check_safety(
                "output",
                identity.context_user_id,
                state.response,
            )
            state.timings["safety_output"] = time.monotonic() - safety_start
            _record_safety_model_usage(state, "output", ok=output_check_ok)
            if not output_safe:
                state.response = self.config.unsafe_message
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = "output_guardrail_blocked"

        state.timings["deepagents"] = time.monotonic() - start
        return state

    def _create_agent(
        self,
        state: State,
        identity: RequestIdentity,
        turn_capabilities: CatalogCapabilities | None = None,
    ):
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from langchain_core.tools import tool

        # One cached lifecycle contract is authoritative for prompt construction
        # and deterministic validation. Catalog requests are revalidated by the
        # active catalog service before execution.
        if turn_capabilities is None:
            turn_capabilities = self._catalog_capabilities.get()

        if not self._profile_registered:
            register_harness_profile(
                f"openai:{self.config.llm_name}",
                HarnessProfile(
                    excluded_tools=_EXCLUDED_DEEP_AGENT_TOOLS,
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )
            self._profile_registered = True

        skills_root = self._shopper_skills_root()
        skills_backend = self._create_skills_backend()
        if skills_backend is None:
            raise RuntimeError("Shopper skill backend is unavailable.")
        skill_registry = _shopper_skill_registry(skills_root)
        skill_activation_input = _skill_activation_input_model(
            tuple(skill_registry)
        )
        retrieved: dict[str, str] = {}
        state.retrieved = retrieved
        self._append_cached_cart_images(retrieved, state.cart, identity)
        search_input_model = _search_catalog_tool_input_model(turn_capabilities)
        search_tool_arguments_model = _search_catalog_tool_input_model(
            turn_capabilities,
            validate_scope=False,
        )
        constraint_input_model = search_input_model.model_fields[
            "required_constraints"
        ].annotation
        catalog_searches_this_turn = 0
        searched_catalog_scopes: list[dict[str, Any]] = []
        searched_shopper_scopes: set[tuple[str, str]] = set()
        catalog_tool_lock = Lock()
        product_detail_reads_this_turn = 0
        failed_repair_scope_key: str | None = None
        failed_agent_selected_scope = False
        failed_constraint_scope_key: str | None = None
        constraint_reviewed_scopes: set[str] = set()
        pending_constraint_reviews: dict[str, dict[str, Any]] = {}
        pending_taxonomy_constraints: dict[str, Any] | None = None
        pending_no_direct_constraint_clear = False
        pending_schema_requirements: list[str] = []

        def _lock_taxonomy_constraint_values(
            scope_key: str | None,
            constraints: dict[str, Any],
            *,
            allow_no_direct_clear: bool = False,
        ) -> str:
            """Store canonical hard constraints for one taxonomy repair."""

            nonlocal pending_taxonomy_constraints
            nonlocal pending_no_direct_constraint_clear
            constraints = _normalized_scope_value(constraints)
            constraints.pop("unadvertised_requirements", None)
            pending_taxonomy_constraints = constraints
            pending_no_direct_constraint_clear = allow_no_direct_clear
            if allow_no_direct_clear:
                return (
                    " If the repaired taxonomy_status remains "
                    "no_direct_catalog_match, clear advertised "
                    "required_constraints as that status requires. If the "
                    "repair changes to a retrieving taxonomy_status, preserve "
                    "all validated advertised required_constraints exactly."
                )
            if not constraints:
                return (
                    " The rejected call had no advertised required_constraints. "
                    "Keep advertised required_constraints empty on repair. "
                    "Change only taxonomy_status, taxonomy, or an explicitly "
                    "identified ungrounded product scope."
                )
            return (
                " Preserve all already validated advertised "
                "required_constraints exactly on repair. Change only "
                "taxonomy_status, taxonomy, or an explicitly identified "
                "ungrounded product scope."
            )

        def _lock_taxonomy_constraints(
            scope_key: str | None,
            request: SearchCatalogToolArguments,
        ) -> str:
            """Preserve validated hard constraints across one taxonomy repair."""

            return _lock_taxonomy_constraint_values(
                scope_key,
                request.required_constraints.model_dump(exclude_none=True),
            )

        @tool(args_schema=search_tool_arguments_model, return_direct=False)
        def search_catalog_tool(
            semantic_query: str,
            requested_product_type: str | None,
            taxonomy: BaseModel | dict[str, Any],
            required_constraints: BaseModel | dict[str, Any],
            shopper_guidance: str = "",
            scope_complete: bool = True,
            taxonomy_status: str = "exact_requested_type",
            search_mode: str | None = None,
        ) -> str:
            """Find products by description, category, or price. Use for browse,
            search, and recommendation requests. Set taxonomy_status to
            'exact_requested_type' when the selected taxonomy directly represents
            the requested focused role. For a single selected value, the requested
            type must name that value; the semantic query may focus on soft
            ranking direction. Use
            'member_of_requested_umbrella' only when
            the shopper named an umbrella or explicit alternatives and every
            selected value is its faithful child or named alternative;
            never reverse that relationship because the requested type belongs to
            a selected parent. Ask whether each selected value is a kind of the
            shopper's requested scope: a skirt is a kind of bottom; a flat or
            sandal is not a kind of sneaker. For a broad styling or discovery
            request that names no concrete product type, use 'agent_selected_type'
            and choose one exact advertised subcategory as the focused starting
            role. Never use
            it for a role whose type the shopper named, including an alternative,
            confirmation, comparison, or follow-up. Use 'no_direct_catalog_match' only
            for an explicitly requested concrete product type when only parent,
            adjacent, or substitute types exist. Never use it for an outfit,
            occasion, season, weather need, style/vibe, or product attribute. Set
            empty taxonomy arrays on that path so it reports the gap without
            retrieval. Use 'image_only' only with an empty text query and taxonomy. A turn
            asking for one recommendation-only product role permits one inclusive
            matched search, then an answer; set scope_complete=true. Set it false
            only when an explicitly requested product role, product-detail check,
            availability check, or cart action still must run after this search.
            Judge completeness against this user turn, not an unfinished multi-turn
            outfit project. "Start with a beige top" and "What bottoms go with
            that?" are complete after their one-role searches.
            An unavailable type inside a one-role umbrella does not make the scope
            partial. Do NOT search an adjacent category or substitute
            after a result. Do NOT use if the product was already found in this conversation
            (use get_product_details_tool).
            Each call covers at most one catalog category. Include all faithful
            subtypes for one requested role, but never mix categories in one call.
            Copy every attribute that defines the requested products into
            required_constraints. Use only advertised filter properties directly;
            put any defining requirement not in the schema into
            unadvertised_requirements. For "Any denim skirts available?", use
            {"unadvertised_requirements": ["denim"]}. For "Do you have water-resistant bags?",
            use {"unadvertised_requirements": ["water resistance"]}, not a soft
            semantic preference. Broad season, weather, occasion, and subjective
            style/vibe context stays in semantic_query unless the shopper directly
            requires a product attribute. Apply hard constraints only when the
            current turn states them for the target products; do not copy an
            anchor's color or material onto a complementary role. "Rainy day
            outfit" does not imply a
            water-resistance requirement. Treat comfortable, relaxed, soft,
            breathable, lightweight, casual, dressy, bold, bright, vibrant, and
            sporty as semantic recommendation preferences, never objective hard
            filters. Before calling this tool, compare every target-product
            modifier with the advertised required_constraints schema and include
            every exact matching filter value; do not leave the object empty when
            one applies.
            For alternatives joined by "or", search
            a faithful advertised branch and explain any unavailable branch; do not
            put the supported branch in unadvertised_requirements.
            When the shopper names an umbrella or explicit alternatives, use
            member_of_requested_umbrella with every faithful advertised child in
            one call; do not narrow it with agent_selected_type.
            Do NOT repeat a search with the same taxonomy and hard constraints;
            semantic paraphrases do not create a new scope.

            Every call also supplies shopper_guidance: one concise,
            product-agnostic sentence connecting the searched role to the
            shopper's stated goal or direct antecedent. Write it before results
            are known. Do not name or describe candidate products or expose
            internal search mechanics, and do not name product types outside the
            selected advertised scope. Use an empty string for image-only and
            no-direct requests.

            Always set requested_product_type to the shortest product-type phrase
            for this role. Keep only the product noun or umbrella and exclude color,
            material, fit, occasion, weather, and style modifiers. For example,
            "formal tops" and "relaxed-fit tops" both use "tops". For
            agent_selected_type, use the advertised role noun you selected. Set it
            to null only for image-only search. It is not a catalog enum or semantic
            query.
            """

            nonlocal catalog_searches_this_turn
            nonlocal failed_agent_selected_scope
            nonlocal failed_constraint_scope_key
            nonlocal failed_repair_scope_key
            nonlocal pending_no_direct_constraint_clear
            nonlocal pending_taxonomy_constraints
            nonlocal pending_schema_requirements
            capabilities = turn_capabilities
            if capabilities.catalog_id == "unavailable" and not capabilities.filters:
                return "Catalog search is unavailable. Please try again."

            shopper_explicit_alternatives = (
                _explicit_advertised_alternatives_in_text(
                    state.query,
                    capabilities,
                )
            )
            requested_product_type = _resolved_agent_selected_product_type(
                query=state.query,
                context=state.context,
                requested_product_type=requested_product_type,
                taxonomy_status=taxonomy_status,
                taxonomy=taxonomy,
            )
            candidate_scope_key = _product_scope_key(requested_product_type)
            locked_repair_scope = (
                failed_constraint_scope_key or failed_repair_scope_key
            )
            repairing_same_scope = bool(
                locked_repair_scope
                and (
                    candidate_scope_key == failed_constraint_scope_key
                    if failed_constraint_scope_key
                    else _same_product_scope(
                        locked_repair_scope,
                        candidate_scope_key,
                        capabilities,
                    )
                )
            )
            if (
                locked_repair_scope
                and not repairing_same_scope
            ):
                expected_scope_key = locked_repair_scope
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "A catalog search repair cannot replace product scope "
                    f"'{expected_scope_key}' "
                    f"with '{candidate_scope_key or 'none'}'. Preserve the "
                    "requested_product_type and repair taxonomy_status or "
                    "taxonomy instead."
                )
            if (
                failed_agent_selected_scope
                and taxonomy_status != "agent_selected_type"
            ):
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "This open-role repair must preserve "
                    "taxonomy_status='agent_selected_type' and choose exactly "
                    "one advertised subcategory for the role."
                )

            taxonomy_payload = (
                taxonomy.model_dump()
                if isinstance(taxonomy, BaseModel)
                else taxonomy
            )
            constraint_payload = (
                required_constraints.model_dump()
                if isinstance(required_constraints, BaseModel)
                else required_constraints
            )
            shopper_stated_scope = bool(
                candidate_scope_key
                and _shopper_stated_product_scope(
                    state.query,
                    state.context,
                    candidate_scope_key,
                )
            )
            raw_unadvertised_requirements = (
                constraint_payload.get("unadvertised_requirements", [])
                if isinstance(constraint_payload, dict)
                else []
            )
            stated_unadvertised_requirements = (
                [
                    requirement
                    for requirement in raw_unadvertised_requirements
                    if isinstance(requirement, str)
                    and _shopper_stated_requirement(state.query, requirement)
                ]
                if isinstance(raw_unadvertised_requirements, list)
                else []
            )
            if (
                isinstance(raw_unadvertised_requirements, list)
                and raw_unadvertised_requirements
                and (shopper_stated_scope or stated_unadvertised_requirements)
            ):
                return _unsupported_requirement_message(
                    raw_unadvertised_requirements
                )
            try:
                request = search_input_model.model_validate(
                    {
                        "semantic_query": semantic_query,
                        "shopper_guidance": shopper_guidance,
                        "requested_product_type": requested_product_type,
                        "taxonomy_status": taxonomy_status,
                        "taxonomy": taxonomy_payload,
                        "required_constraints": constraint_payload,
                        "scope_complete": scope_complete,
                        "search_mode": search_mode,
                    }
                )
            except ValidationError as exc:
                validation_errors = [
                    {
                        "loc": list(error.get("loc") or ()),
                        "type": str(error.get("type") or "validation_error"),
                        "msg": str(error.get("msg") or "Invalid value"),
                    }
                    for error in exc.errors(include_url=False)
                ]
                taxonomy_error = any(
                    any(
                        marker in (
                            str(error.get("loc", ""))
                            + " "
                            + str(error.get("msg", ""))
                        )
                        .replace("_", " ")
                        .casefold()
                        for marker in (
                            "taxonomy",
                            "category",
                            "subcategory",
                            "requested product type",
                        )
                    )
                    for error in validation_errors
                )
                repair_guidance = ""
                canonical_agent_selected_scope = bool(
                    candidate_scope_key
                    and taxonomy_status == "agent_selected_type"
                    and _agent_selected_scope_is_advertised(
                        requested_product_type,
                        taxonomy,
                    )
                )
                if shopper_stated_scope or canonical_agent_selected_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                elif taxonomy_error and taxonomy_status == "agent_selected_type":
                    failed_agent_selected_scope = True
                if candidate_scope_key and taxonomy_error:
                    if (
                        taxonomy_status == "agent_selected_type"
                        and not shopper_stated_scope
                    ):
                        advertised_choices = _advertised_subcategories_for_selection(
                            taxonomy,
                            capabilities,
                        )
                        repair_guidance = (
                            " For an open-role agent_selected_type search, choose "
                            "exactly one advertised subcategory and copy it into "
                            "requested_product_type. Choose exactly one of these "
                            "currently advertised subcategories: "
                            + json.dumps(advertised_choices, ensure_ascii=False)
                            + "."
                        )
                        constraints = (
                            required_constraints.model_dump(exclude_none=True)
                            if isinstance(required_constraints, BaseModel)
                            else required_constraints
                        )
                        proposed_requirements = constraints.get(
                            "unadvertised_requirements",
                            [],
                        )
                        if proposed_requirements:
                            pending_schema_requirements = list(
                                proposed_requirements
                            )
                            repair_guidance += (
                                " The rejected call proposed "
                                "unadvertised_requirements "
                                + json.dumps(
                                    proposed_requirements,
                                    ensure_ascii=False,
                                )
                                + ". Preserve only an objective product attribute "
                                "directly stated for the selected role; remove one "
                                "inferred from season, weather, occasion, or style."
                            )
                constraint_lock = ""
                try:
                    validated_constraints = constraint_input_model.model_validate(
                        constraint_payload
                    )
                except ValidationError:
                    pass
                else:
                    constraint_lock = _lock_taxonomy_constraint_values(
                        candidate_scope_key,
                        validated_constraints.model_dump(exclude_none=True),
                        allow_no_direct_clear=(
                            taxonomy_status == "no_direct_catalog_match"
                        ),
                    )
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "The catalog search request does not match current "
                    f"capabilities: {validation_errors}"
                    + repair_guidance
                    + constraint_lock
                )

            all_constraints = request.required_constraints.model_dump(
                exclude_none=True,
            )
            normalized_advertised_constraints = _normalized_scope_value(
                all_constraints
            )
            normalized_advertised_constraints.pop(
                "unadvertised_requirements",
                None,
            )
            if (
                pending_taxonomy_constraints is not None
                and not (
                    pending_no_direct_constraint_clear
                    and request.taxonomy_status == "no_direct_catalog_match"
                )
                and normalized_advertised_constraints
                != pending_taxonomy_constraints
            ):
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "A taxonomy repair must preserve previously validated "
                    "advertised required_constraints exactly. Change only "
                    "taxonomy_status, taxonomy, or an explicitly identified "
                    "ungrounded product scope."
                )
            if pending_taxonomy_constraints is not None:
                pending_taxonomy_constraints = None
                pending_no_direct_constraint_clear = False
            normalized_constraints = dict(all_constraints)
            unadvertised_requirements = normalized_constraints.pop(
                "unadvertised_requirements",
                [],
            )
            shopper_stated_scope = bool(
                candidate_scope_key
                and _shopper_stated_product_scope(
                    state.query,
                    state.context,
                    candidate_scope_key,
                )
            )
            if unadvertised_requirements and shopper_stated_scope:
                return _unsupported_requirement_message(
                    unadvertised_requirements
                )

            if shopper_explicit_alternatives is not None:
                shopper_alternative_issue = (
                    _advertised_alternative_selection_issue(
                        request.requested_product_type,
                        shopper_explicit_alternatives,
                        request.taxonomy_status,
                        request.taxonomy,
                    )
                )
                if shopper_alternative_issue:
                    _, advertised_subcategories = shopper_explicit_alternatives
                    alternative_scope_key = _product_scope_key(
                        " or ".join(advertised_subcategories)
                    )
                    failed_repair_scope_key = alternative_scope_key
                    failed_agent_selected_scope = False
                    return (
                        SEARCH_VALIDATION_ERROR_PREFIX
                        + EXPLICIT_ALTERNATIVE_CORRECTION_PREFIX
                        + " "
                        + shopper_alternative_issue
                        + _lock_taxonomy_constraints(
                            alternative_scope_key,
                            request,
                        )
                    )

            advertised_taxonomy_issue = _advertised_taxonomy_scope_issue(
                request.requested_product_type,
                request.taxonomy_status,
                request.taxonomy,
                capabilities,
            )
            if advertised_taxonomy_issue:
                if shopper_stated_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + advertised_taxonomy_issue
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                    + (
                        " Preserve the shopper-stated requested_product_type."
                        if shopper_stated_scope
                        else " The rejected requested_product_type was not "
                        "shopper-stated. Re-read the current shopper request "
                        "and correct it rather than preserving this scope."
                    )
                )

            if pending_schema_requirements and not unadvertised_requirements:
                request = request.model_copy(
                    update={
                        "shopper_guidance": _generic_shopper_guidance(
                            request.requested_product_type
                        )
                    }
                )
                pending_schema_requirements = []
            pending_constraint_review = pending_constraint_reviews.get(
                candidate_scope_key
            )
            if pending_constraint_review and (
                request.taxonomy_status
                != pending_constraint_review["taxonomy_status"]
                or request.taxonomy.model_dump()
                != pending_constraint_review["taxonomy"]
                or request.scope_complete
                != pending_constraint_review["scope_complete"]
                or request.search_mode != pending_constraint_review["search_mode"]
                or normalized_constraints
                != pending_constraint_review["required_constraints"]
            ):
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "A constraint-provenance repair must preserve "
                    "requested_product_type, taxonomy_status, taxonomy, "
                    "scope_complete, search_mode, and all "
                    "advertised required constraints exactly. Change only the "
                    "reviewed unadvertised requirement wording or remove an "
                    "inferred requirement; the soft semantic query may be "
                    "corrected within the preserved product scope."
                )
            agent_selected_issue: str | None = None
            agent_selected_shopper_scope = False
            if (
                request.taxonomy_status == "agent_selected_type"
                and (
                    _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                    or not _agent_selected_scope_is_advertised(
                        request.requested_product_type,
                        request.taxonomy,
                    )
                )
            ):
                agent_selected_shopper_scope = bool(
                    candidate_scope_key
                    and _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                )
                if agent_selected_shopper_scope:
                    payload = request.taxonomy.model_dump()
                    selected_values = (
                        payload.get("subcategory")
                        or payload.get("category")
                        or []
                    )
                    advertised_match = _advertised_scope_match(
                        request.requested_product_type,
                        capabilities,
                    )
                    exact_selected_scope = bool(
                        advertised_match
                        and len(selected_values) == 1
                        and _normalize_product_text(selected_values[0])
                        == _normalize_product_text(advertised_match[1])
                    )
                    repair_status = (
                        "Change only taxonomy_status to exact_requested_type."
                        if exact_selected_scope
                        else (
                            "If every selected value is a kind of the named "
                            "scope, change only taxonomy_status to "
                            "member_of_requested_umbrella; do not narrow to one "
                            "child or report no direct match."
                        )
                    )
                    agent_selected_issue = (
                        "The shopper named requested_product_type "
                        f"'{request.requested_product_type}', so "
                        "agent_selected_type is forbidden. Preserve that "
                        "requested_product_type and these selected advertised "
                        "values on repair: "
                        + json.dumps(selected_values, sort_keys=True)
                        + ". "
                        + repair_status
                    )
                else:
                    advertised_choices = _advertised_subcategories_for_selection(
                        request.taxonomy,
                        capabilities,
                    )
                    agent_selected_issue = (
                        "An open-role agent_selected_type search must select "
                        "exactly one advertised subcategory and copy that exact "
                        "subcategory into requested_product_type. Do not use a "
                        "parent category or an unadvertised role. Choose exactly "
                        "one of these currently advertised subcategories: "
                        + json.dumps(advertised_choices, ensure_ascii=False)
                        + ". Rewrite "
                        "shopper_guidance for that selected role without "
                        "asserting an inferred product attribute."
                    )
            if agent_selected_issue:
                if agent_selected_shopper_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                elif candidate_scope_key:
                    failed_agent_selected_scope = True
                constraint_issue = ""
                if unadvertised_requirements:
                    constraint_issue = (
                        " The same rejected call proposed "
                        "unadvertised_requirements "
                        + json.dumps(
                            unadvertised_requirements,
                            ensure_ascii=False,
                        )
                        + ". Preserve only an objective product attribute "
                        "directly stated for the selected role. If it was "
                        "inferred from season, weather, occasion, or style, "
                        "send an empty list and remove its promise from "
                        "shopper_guidance."
                    )
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + agent_selected_issue
                    + constraint_issue
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                )
            failed_agent_selected_scope = False

            if (
                request.taxonomy_status != "no_direct_catalog_match"
                and unadvertised_requirements
            ):
                stated_requirements = [
                    requirement
                    for requirement in unadvertised_requirements
                    if _shopper_stated_requirement(state.query, requirement)
                ]
                shopper_stated_scope = bool(
                    candidate_scope_key
                    and _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                )
                if stated_requirements or shopper_stated_scope:
                    return _unsupported_requirement_message(
                        unadvertised_requirements
                    )
                review_scope = candidate_scope_key or "__unknown__"
                if review_scope in constraint_reviewed_scopes:
                    return (
                        "The requested catalog requirement cannot be enforced: "
                        "its current-turn provenance could not be established. "
                        "Ask the shopper to state the exact required attribute "
                        "or allow it to be treated as a preference."
                    )
                constraint_reviewed_scopes.add(review_scope)
                pending_constraint_reviews[review_scope] = {
                    "requirements": list(unadvertised_requirements),
                    "taxonomy_status": request.taxonomy_status,
                    "taxonomy": request.taxonomy.model_dump(),
                    "scope_complete": request.scope_complete,
                    "search_mode": request.search_mode,
                    "required_constraints": dict(normalized_constraints),
                }
                failed_constraint_scope_key = review_scope
                return (
                    CONSTRAINT_REVIEW_PREFIX
                    + "These proposed unadvertised requirements do not match "
                    "the current shopper turn's normalized wording: "
                    + json.dumps(unadvertised_requirements, ensure_ascii=False)
                    + ". Preserve requested_product_type "
                    + json.dumps(request.requested_product_type)
                    + ", taxonomy_status "
                    + json.dumps(request.taxonomy_status)
                    + ", taxonomy "
                    + json.dumps(request.taxonomy.model_dump(), sort_keys=True)
                    + ", and scope_complete "
                    + json.dumps(request.scope_complete)
                    + ", search_mode "
                    + json.dumps(request.search_mode)
                    + ", and advertised required constraints "
                    + json.dumps(normalized_constraints, sort_keys=True)
                    + ". Keep semantic_query within that same product scope; "
                    "you may remove inferred attribute wording from it"
                    + ". If the shopper explicitly stated the same objective "
                    "requirement using different words, replace each value with "
                    "the shopper's shortest exact wording. Otherwise the model "
                    "inferred it: remove it from required_constraints and remove "
                    "the attribute claim from shopper_guidance. Implied weather, "
                    "occasion, or style goals are not explicit requirements."
                )

            reviewed_constraint = pending_constraint_reviews.pop(
                candidate_scope_key,
                None,
            )
            if reviewed_constraint:
                request = request.model_copy(
                    update={
                        "shopper_guidance": _generic_shopper_guidance(
                            request.requested_product_type
                        )
                    }
                )

            exact_taxonomy_issue = (
                _exact_taxonomy_issue(
                    request.requested_product_type or "",
                    request.taxonomy,
                )
                if (
                    request.taxonomy_status == "exact_requested_type"
                    and _advertised_scope_match(
                        request.requested_product_type,
                        capabilities,
                    )
                    is None
                )
                else None
            )
            if exact_taxonomy_issue:
                shopper_stated_scope = bool(
                    candidate_scope_key
                    and _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                )
                if shopper_stated_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + exact_taxonomy_issue
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                    + (
                        ". Preserve the shopper-stated requested_product_type "
                        f"{json.dumps(request.requested_product_type)}."
                        if shopper_stated_scope
                        else ". Re-read the current shopper request and correct "
                        "requested_product_type if the rejected value was not "
                        "shopper-stated."
                    )
                    + " Choose only advertised taxonomy values that faithfully "
                    "represent that scope. Do not repeat exact_requested_type "
                    "for this repair. If each selected value is a kind of "
                    "a shopper-stated umbrella, use "
                    "member_of_requested_umbrella; otherwise use "
                    "no_direct_catalog_match rather than an adjacent type."
                )

            advertised_match = (
                _advertised_taxonomy_value(
                    request.requested_product_type,
                    capabilities,
                )
                if request.taxonomy_status == "no_direct_catalog_match"
                else None
            )
            if advertised_match:
                failed_repair_scope_key = candidate_scope_key
                failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + f"The requested product type '{request.requested_product_type}' "
                    f"matches advertised taxonomy value '{advertised_match}'. "
                    "Do not report no direct match; select that advertised value "
                    "with exact_requested_type."
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                )

            failed_repair_scope_key = None
            failed_constraint_scope_key = None
            failed_agent_selected_scope = False

            shopper_scope_key = (
                (_normalize_product_text(state.query), candidate_scope_key)
                if candidate_scope_key
                and _text_mentions_product_type(
                    state.query,
                    candidate_scope_key,
                )
                else None
            )

            if request.taxonomy_status == "no_direct_catalog_match":
                with catalog_tool_lock:
                    if (
                        shopper_scope_key is not None
                        and shopper_scope_key in searched_shopper_scopes
                    ):
                        return (
                            "STOP_TOOL_USE: This shopper-requested product scope "
                            "was already searched in this turn. Do not search an "
                            "adjacent taxonomy or report the requested scope as "
                            "unavailable. Use the result already returned.\n\n"
                            + _SEARCH_SCOPE_COMPLETE_NOTE
                        )
                lines = [
                    "STOP_TOOL_USE: No faithful advertised catalog taxonomy "
                    "matches the requested product type "
                    f"'{request.requested_product_type}'. "
                    "Do not search adjacent product types. Tell the shopper the "
                    "requested type is not advertised and ask before offering an "
                    "alternative.",
                    _format_catalog_scope_outcome(
                        {
                            "outcome": "no_direct_catalog_match",
                            "requested_product_type": request.requested_product_type,
                        }
                    ),
                ]
                if request.scope_complete:
                    lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
                return "\n\n".join(lines)

            taxonomy_constraints, taxonomy_issues = _taxonomy_hard_constraints(
                request.taxonomy,
                capabilities,
            )
            taxonomy_fields = {
                field_name
                for field_name in (
                    capabilities.taxonomy.category_field,
                    capabilities.taxonomy.subcategory_field,
                )
                if field_name
            }
            overlapping_fields = sorted(
                taxonomy_fields.intersection(normalized_constraints)
            )
            if overlapping_fields:
                taxonomy_issues.append(
                    "taxonomy fields must use the taxonomy selection, not "
                    "required_constraints: " + ", ".join(overlapping_fields)
                )
            if taxonomy_issues:
                return (
                    "The requested catalog taxonomy cannot be enforced: "
                    + "; ".join(taxonomy_issues)
                    + ". Ask the shopper to choose an advertised product type."
                )

            normalized_search_mode = _tool_search_mode(request.search_mode)
            if request.search_mode is not None and (
                normalized_search_mode is None
                or request.search_mode not in capabilities.retrieval_modes
            ):
                return _UNSUPPORTED_SEARCH_MODE_MESSAGE

            intent = CatalogSearchIntent(
                semantic_query=request.semantic_query,
                required_constraints={
                    **normalized_constraints,
                    **taxonomy_constraints,
                },
                search_mode=normalized_search_mode,
            )
            explicit_alternatives = _explicit_advertised_alternatives(
                request.requested_product_type,
                capabilities,
            )
            plan = build_catalog_search_plan(
                intent,
                capabilities,
                has_image=bool(state.image),
                top_k=_alternative_candidate_limit(
                    explicit_alternatives,
                    capabilities,
                    self.config.top_k_retrieve,
                ),
            )
            if not plan.should_search:
                if plan.constraint_issues:
                    return (
                        "The requested catalog requirement cannot be enforced: "
                        + "; ".join(plan.constraint_issues)
                        + ". Ask the shopper to relax it or use an advertised filter."
                    )
                if plan.no_search_reason == "image_search_unavailable":
                    return (
                        "Image search is not available for the active catalog. "
                        "Ask the shopper to describe what they want to find."
                    )
                if plan.no_search_reason == "unsupported_search_mode":
                    return _UNSUPPORTED_SEARCH_MODE_MESSAGE
                if plan.no_search_reason == "missing_image_for_search_mode":
                    return (
                        "That search mode requires an attached image. Ask the shopper "
                        "to attach one or use text search."
                    )
                return "Catalog search requires a query or image."

            with catalog_tool_lock:
                search_scope = _catalog_search_scope(
                    taxonomy_constraints,
                    normalized_constraints,
                )
                if (
                    shopper_scope_key is not None
                    and shopper_scope_key in searched_shopper_scopes
                ):
                    return (
                        "STOP_TOOL_USE: This shopper-requested product scope "
                        "was already searched in this turn. Do not search an "
                        "adjacent taxonomy. Use the result already returned.\n\n"
                        + _SEARCH_SCOPE_COMPLETE_NOTE
                    )
                if search_scope in searched_catalog_scopes:
                    return (
                        "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already "
                        "searched in this turn. Do not retry it "
                        "with a paraphrase or query expansion. Use the products "
                        "already returned, or ask one concise clarifying question."
                    )
                if catalog_searches_this_turn >= self.config.max_catalog_searches_per_turn:
                    return (
                        "STOP_TOOL_USE: Catalog search limit reached for this turn. "
                        "Do not call more tools this turn. Use the products already "
                        "returned in this turn to answer concisely, or ask one concise "
                        "clarifying question if the available products are not enough."
                    )
                searched_catalog_scopes.append(search_scope)
                if shopper_scope_key is not None:
                    searched_shopper_scopes.add(shopper_scope_key)
                catalog_searches_this_turn += 1
                search_budget_exhausted = (
                    catalog_searches_this_turn
                    >= self.config.max_catalog_searches_per_turn
                )

            search_start = time.monotonic()
            execution = execute_catalog_search(
                plan,
                self.config.retriever_port,
                image_base64=state.image,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            catalog_elapsed = time.monotonic() - search_start
            result = execution.result
            if result.ok:
                result = result.model_copy(
                    update={
                        "products": _products_with_alternative_coverage(
                            result.products,
                            explicit_alternatives,
                            self.config.top_k_retrieve,
                        )
                    }
                )
            with catalog_tool_lock:
                state.timings["catalog_search"] = max(
                    state.timings.get("catalog_search", 0.0),
                    catalog_elapsed,
                )
                _record_catalog_model_usage(
                    state,
                    plan,
                    result.ok,
                    fallback_attempted=execution.fallback_attempted,
                )
                if result.ok and result.products:
                    self._remember_products(identity, result.products)
                    _append_product_results(state, result.products)
                    for product in result.products:
                        if product.image_url:
                            retrieved[product.display_name] = product.image_url
            if not result.ok:
                return result.error.message if result.error else "Catalog search failed."

            confirmed_filters = {
                name: value
                for name, value in plan.hard_filters.items()
                if name not in taxonomy_fields
            }
            if not result.products:
                lines = [
                    _SEARCH_NO_MATCH_GROUNDING_NOTE,
                    _format_search_taxonomy_evidence(taxonomy_constraints),
                ]
                if confirmed_filters:
                    lines.append(_format_search_filter_evidence(confirmed_filters))
                lines.append(
                    _format_catalog_scope_outcome(
                        {
                            "outcome": "zero_results",
                            "requested_product_type": request.requested_product_type,
                            "taxonomy": taxonomy_constraints,
                            "confirmed_filters": confirmed_filters,
                        }
                    )
                )
                if request.scope_complete:
                    lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
                elif search_budget_exhausted:
                    lines.append(_SEARCH_BUDGET_EXHAUSTED_NOTE)
                return "\n\n".join(lines)

            lines = [
                _SEARCH_RESULT_GROUNDING_NOTE,
                _format_search_direction_evidence(request.semantic_query),
                _format_search_guidance_evidence(
                    _safe_shopper_guidance(
                        request.shopper_guidance,
                        request.requested_product_type,
                    )
                ),
                _format_search_taxonomy_evidence(taxonomy_constraints),
            ]
            if confirmed_filters:
                lines.append(_format_search_filter_evidence(confirmed_filters))
            if request.scope_complete:
                lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
            elif search_budget_exhausted:
                lines.append(_SEARCH_BUDGET_EXHAUSTED_NOTE)
            for product in result.products:
                lines.append(_format_product(product))
            prefix = (
                "Image similarity returned no matches; text fallback results:\n\n"
                if execution.fallback_used
                else ""
            )
            return prefix + "\n\n".join(lines)

        @tool(return_direct=False)
        def get_cart_tool() -> str:
            """Read the current cart. Use before cart mutations to get
            CART_LINE_ID values, or when the shopper asks what is in their cart.
            Do NOT call again if the cart was already read this turn and no
            mutation has occurred since.
            """

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_cached_cart_images(retrieved, cart, identity)
            return _format_cart(cart)

        @tool(return_direct=False)
        def get_product_details_tool(product_ref: str) -> str:
            """Get detailed facts (material, care, dimensions, closures) for a
            product already found in this conversation. Requires a PRODUCT_REF
            from search — not a product name. Do NOT call for initial
            recommendations. Stop immediately if STOP_TOOL_USE is returned.
            """

            nonlocal product_detail_reads_this_turn
            if (
                product_detail_reads_this_turn
                >= self.config.max_product_detail_reads_per_turn
            ):
                return (
                    "STOP_TOOL_USE: Product-detail read limit reached for this "
                    "turn. Do not call more tools this turn. Answer now from the "
                    "details already read and keep any other products to names, "
                    "prices, categories, image availability, and styling role."
                )
            product_detail_reads_this_turn += 1

            cached_product = self._product_from_ref(identity, product_ref)
            if cached_product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search the catalog first and use the PRODUCT_REF from the result."
                )
            detail_result = get_product_details(
                GetProductDetailsInput(product_id=cached_product.product_id),
                self.config.retriever_port,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            if not detail_result.ok or detail_result.product is None:
                return _product_detail_failure_message(
                    detail_result.error,
                    cart_validation=False,
                )
            product = detail_result.product
            if not _same_product_display_name(
                product.display_name,
                cached_product.display_name,
            ):
                return (
                    "That product reference now resolves to a different item. "
                    "Search the catalog again before using its details."
                )
            if product.image_url:
                retrieved[product.display_name] = product.image_url
            return _format_product_details(product)

        @tool(args_schema=AddCartItemsToolInput, return_direct=False)
        def add_cart_items_tool(items: list[AddCartItemsToolItemInput]) -> str:
            """Add products to the cart. Use ONLY on explicit shopper intent to
            add, buy, or put items in the cart. Requires PRODUCT_REF values from
            search — not names. Call once with all items, not once per item.
            """

            try:
                requested_items = _normalize_cart_add_tool_items(items)
            except ValueError as exc:
                return f"Cart add failed: {exc}"
            if not requested_items:
                return "Cart add failed: provide at least one PRODUCT_REF to add."

            refs = self._product_refs.get(identity.conversation_id, {})
            resolved: list[tuple[str, ProductSummary, int]] = []
            failed: list[str] = []
            blocked: list[str] = []
            for product_ref, request in requested_items.items():
                product = self._product_from_ref(identity, product_ref)
                if product is None:
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': Search the catalog first "
                        "and use the PRODUCT_REF from the result."
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
                    self.config.retriever_port,
                    timeout_seconds=self.config.catalog_search_timeout_seconds,
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
                resolved.append(
                    (product_ref, active_detail.product, int(request["quantity"]))
                )

            blocked.extend(
                _cart_add_scope_failures(
                    state.query,
                    [(product_ref, product) for product_ref, product, _ in resolved],
                    refs.values(),
                )
            )
            if blocked:
                state.cart = self._read_cart(identity.cart_user_id)
                self._append_cached_cart_images(retrieved, state.cart, identity)
                return _format_cart_add_result([], failed + blocked, state.cart)

            added: list[str] = []
            for product_ref, product, quantity in resolved:
                result = add_cart_item(
                    AddCartItemInput(
                        user_id=str(identity.cart_user_id),
                        product_id=product.product_id,
                        display_name=product.display_name,
                        quantity=quantity,
                        unit_price=product.price,
                        image_url=product.image_url,
                        idempotency_key=(
                            f"{identity.request_id}:add:{product.product_id}:{quantity}"
                        ),
                    ),
                    self.config.memory_port,
                )
                if result.ok:
                    added.append(
                        f"- {quantity} x {product.display_name} "
                        f"(PRODUCT_REF: {product.product_id})"
                    )
                else:
                    message = (
                        result.error.message if result.error else "Cart add failed."
                    )
                    failed.append(f"- PRODUCT_REF '{product_ref}': {message}")

            state.cart = self._read_cart(identity.cart_user_id)
            self._append_cached_cart_images(retrieved, state.cart, identity)
            return _format_cart_add_result(added, failed, state.cart)

        @tool(return_direct=False)
        def remove_cart_item_tool(cart_line_id: str, quantity: int = 1) -> str:
            """Remove a cart line. Use ONLY on explicit shopper intent to remove
            an item. Requires CART_LINE_ID from get_cart_tool — do not guess.
            Use update_cart_items_tool to change quantity instead of removing
            and re-adding.
            """

            quantity = max(1, int(quantity or 1))
            cart = self._read_cart(identity.cart_user_id)
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
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            self._append_cached_cart_images(retrieved, state.cart, identity)
            return _format_cart_remove_result(
                result,
                fallback=f"Removed {quantity} {line['item']} from cart.",
            )

        @tool(args_schema=_UpdateCartItemsInput, return_direct=False)
        def update_cart_items_tool(cart_line_id: str, quantity: int) -> str:
            """Change the quantity of an item already in the cart, or remove it
            by setting quantity to 0. Use ONLY when the shopper explicitly asks
            to change a quantity or remove by quantity. Do NOT use for initial
            adds — use add_cart_items_tool. Do NOT guess the CART_LINE_ID; call
            get_cart_tool first if you do not have one.
            """

            result = update_cart_item(
                UpdateCartItemInput(
                    user_id=str(identity.cart_user_id),
                    cart_line_id=cart_line_id,
                    quantity=quantity,
                    idempotency_key=(
                        f"{identity.request_id}:update:{cart_line_id}:{quantity}"
                    ),
                ),
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            self._append_cached_cart_images(retrieved, state.cart, identity)
            return _format_update_cart_result(result, state.cart)

        @tool(args_schema=_GetStorePolicyInput, return_direct=False)
        def get_store_policy_tool(
            topic: Literal[
                "returns",
                "shipping",
                "sizing",
                "payment",
                "price_match",
                "gift_cards",
            ],
        ) -> str:
            """Look up store policy for: returns, shipping, sizing, payment,
            price_match, or gift_cards. Use ONLY for these policy topics. Do
            NOT use for product facts, prices, or availability. If this tool
            returns a not-found error, relay the message to the shopper and
            direct them to the retailer's help center. Do NOT substitute model
            knowledge for a missing policy.
            """

            result = get_store_policy(
                GetStorePolicyInput(topic=topic),
                _store_policies_path(),
            )
            return _format_policy_result(result)

        @tool(args_schema=_CheckAvailabilityInput, return_direct=False)
        def check_product_availability_tool(
            product_ref: str,
            variant_hint: str | None = None,
        ) -> str:
            """Check whether a specific product is available or in stock. Use
            ONLY when the shopper explicitly asks about availability, stock,
            or a specific size. Requires a PRODUCT_REF from search. Do
            NOT use for browsing. The deterministic stub reports general
            availability, sized availability for apparel and footwear, and
            one-size availability for other product categories.
            """

            product = self._product_from_ref(identity, product_ref)
            if product is None:
                return (
                    f"PRODUCT_REF '{product_ref}' is unknown in this conversation. "
                    "Search the catalog first and use the PRODUCT_REF from the result."
                )
            result = check_product_availability(
                CheckProductAvailabilityInput(
                    product_ref=product_ref,
                    variant_hint=variant_hint,
                ),
                product,
            )
            return _format_availability_result(result)

        @tool(return_direct=False)
        def view_cart_total_tool() -> str:
            """Compute the cart subtotal. Use for budget checks or when the
            shopper asks for the total. Does not include tax or shipping. Use
            get_cart_tool for line contents.
            """

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_cached_cart_images(retrieved, cart, identity)
            return _format_cart_total(cart)

        shopping_tools = [
            search_catalog_tool,
            get_product_details_tool,
            get_cart_tool,
            add_cart_items_tool,
            remove_cart_item_tool,
            update_cart_items_tool,
            view_cart_total_tool,
            get_store_policy_tool,
            check_product_availability_tool,
        ]
        validate_registered_tool_names(
            {
                str(
                    getattr(candidate, "name", None)
                    or getattr(candidate, "__name__", "")
                )
                for candidate in shopping_tools
            }
        )
        skill_gate = ShopperSkillActivationMiddleware(
            request_id=identity.request_id,
            skill_descriptions={
                name: skill.description
                for name, skill in skill_registry.items()
            },
            skill_tool_grants={
                name: skill.tools_granted
                for name, skill in skill_registry.items()
            },
        )
        tool_loop_control = ToolLoopControlMiddleware()

        @tool(args_schema=skill_activation_input, return_direct=False)
        def activate_shopper_skills_tool(
            skill_names: list[str],
        ) -> str:
            """Select and load shopper behavior skills for this turn. This is
            the required first step before answering or calling shopping tools.
            Select the smallest set whose registered descriptions cover the
            complete current intent. Use outfit-styling for outfit building,
            completion, refinement, or mid-browse styling questions; style-led
            single-piece selection, including a statement or balancing piece,
            also uses outfit-styling. Use product-discovery for search and browse
            without styling intent. These are alternative primary procedures,
            never a pair. Add budget-shopping only as a modifier when the shopper
            states a budget. Keep outfit-styling as the primary skill throughout
            an active outfit-building thread, including its single-piece follow-up
            searches.

            """

            selected_names = list(dict.fromkeys(skill_names))
            try:
                selected_files = {
                    skill_registry[name].path: skill_registry[name].content
                    for name in selected_names
                }
                activated = skill_gate.activate(selected_files, selected_names)
            except (KeyError, ValueError):
                skill_gate.fail()
                return (
                    "SHOPPER_SKILL_ACTIVATION_FAILED: Registered skill "
                    "instructions could not be loaded."
                )
            if not activated:
                return "SHOPPER_SKILL_ACTIVATION_ALREADY_COMPLETE"
            return (
                f"{SKILL_ACTIVATION_COMPLETE} "
                + ", ".join(selected_files)
            )

        return create_deep_agent(
            model=self._create_chat_model(),
            tools=[activate_shopper_skills_tool, *shopping_tools],
            system_prompt=self._system_prompt(turn_capabilities),
            middleware=[tool_loop_control, skill_gate],
            backend=skills_backend,
            checkpointer=self._checkpointer,
        )

    async def _reset_agent_thread(self, identity: RequestIdentity) -> None:
        try:
            delete_thread = getattr(self._checkpointer, "adelete_thread", None)
            if delete_thread is not None:
                await delete_thread(identity.conversation_id)
            else:
                self._checkpointer.delete_thread(identity.conversation_id)
        except Exception as exc:  # pragma: no cover - cleanup is best effort.
            logger.warning("Could not reset Deep Agents thread after failure: %s", exc)

    def _create_chat_model(self):
        from langchain_openai import ChatOpenAI

        api_key_env = getattr(self.config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        return ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def _rewrite_response_for_grounding(
        self,
        state: State,
        result: Any,
        draft_response: str,
        *,
        request_id: str,
    ) -> str:
        rejected_search_response = _rejected_catalog_search_response(
            result,
            request_id=request_id,
        )
        if rejected_search_response:
            return rejected_search_response

        max_evidence_chars = getattr(
            self.config,
            "grounding_rewrite_max_evidence_chars",
            12000,
        )
        current_evidence = _collect_tool_grounding_evidence(
            result,
            max_chars=max_evidence_chars,
            request_id=request_id,
        )
        prior_evidence = _collect_message_grounding_evidence(
            _prior_turn_messages(_result_messages(result), request_id),
            max_chars=max_evidence_chars,
        )
        search_only = bool(current_evidence) and _has_search_only_tool_evidence(
            result,
            request_id=request_id,
        )
        if not getattr(self.config, "grounding_rewrite_enabled", True):
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            return _scrub_internal_shopper_language(draft_response)
        if not draft_response:
            if not search_only:
                return draft_response
            return self._rewrite_search_only_response(
                state,
                result,
                request_id=request_id,
            )
        if not current_evidence and not prior_evidence:
            return _scrub_internal_shopper_language(draft_response)

        start = time.monotonic()
        prompt = (
            f"USER QUERY:\n{state.query}\n\n"
            f"RECENT DISCUSSION:\n{state.context or '(none)'}\n\n"
            f"CURRENT CART:\n{_format_cart(state.cart)}\n\n"
            f"AVAILABLE IMAGES:\n{_format_retrieved_images(state.retrieved)}\n\n"
            "CURRENT-TURN TOOL EVIDENCE:\n"
            f"{current_evidence or '(none)'}\n\n"
            "PRIOR-TURN TOOL EVIDENCE:\n"
            f"{prior_evidence or '(none)'}\n\n"
            f"DRAFT RESPONSE:\n{draft_response}"
        )
        try:
            rewrite_result = self._create_chat_model().invoke(
                [
                    {"role": "system", "content": _GROUNDING_EDITOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:  # noqa: BLE001 - response editor has a safe fallback.
            logger.exception("Grounding response editor failed")
            state.timings["grounding_rewrite"] = time.monotonic() - start
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail=(
                    "Final response grounding rewrite failed closed"
                    if search_only
                    else "Final response grounding rewrite failed open"
                ),
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            return _scrub_internal_shopper_language(draft_response)

        state.timings["grounding_rewrite"] = time.monotonic() - start
        _add_model_usage(
            state,
            "app_llm_grounding_editor",
            status="used",
            calls=1,
            detail="Final response grounding rewrite",
        )
        state.token_usage = _merge_token_usage(
            state.token_usage,
            _collect_token_usage(rewrite_result),
        )

        rewritten = _content_to_text(_value(rewrite_result, "content"))
        if not rewritten:
            rewritten = _content_to_text(rewrite_result)
        if not rewritten and search_only:
            return self._rewrite_search_only_response(
                state,
                result,
                request_id=request_id,
            )
        return _scrub_internal_shopper_language(rewritten or draft_response)

    def _rewrite_search_only_response(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
    ) -> str:
        """Return the deterministic fallback for a search-only turn."""

        shopper_guidance = _search_guidance_evidence(
            result,
            request_id=request_id,
        )
        return self._build_search_only_response(
            state,
            result,
            request_id=request_id,
            intro=(
                "\n".join(shopper_guidance)
                or self._active_skill_response_guidance(state)
            ),
        )

    def _active_skill_response_guidance(self, state: State) -> str:
        active_paths = set(
            state.agent_diagnostics.get("skill_files_read", [])
        )
        registry = _shopper_skill_registry(self._shopper_skills_root())
        guidance = [
            skill.response_guidance
            for skill in registry.values()
            if skill.path in active_paths
        ]
        return "\n".join(guidance)

    def _build_search_only_response(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
        intro: str = "",
    ) -> str:
        """Return a deterministic facts-only fallback for search evidence."""

        return _format_search_only_response(
            state,
            result,
            request_id=request_id,
            intro=intro,
        )

    def _create_skills_backend(self):
        if _FilesystemBackend is None:
            logger.warning("Deep Agents filesystem backend unavailable; shopper skills disabled.")
            return None
        skills_root = self._shopper_skills_root()
        if skills_root is None:
            logger.warning("Shopper skills root not found; Deep Agents skills disabled.")
            return None
        return _FilesystemBackend(root_dir=skills_root, virtual_mode=True)

    def _shopper_skills_root(self) -> Path | None:
        configured_root = os.environ.get(_SHOPPER_SKILLS_ENV)
        candidates = [Path(configured_root)] if configured_root else []
        candidates.append(Path(__file__).resolve().parents[1] / "skills")

        for candidate in candidates:
            if (candidate / "shopper").is_dir():
                return candidate
        return None

    def _system_prompt(
        self,
        capabilities: CatalogCapabilities,
    ) -> str:
        catalog_context = format_catalog_capabilities_for_prompt(capabilities)
        prompt = f"""You are a retail shopping assistant for the products advertised by the active catalog.

Use tools for catalog facts and cart actions. Do not invent product names,
prices, availability, materials, care instructions, tax, shipping, stock
status, delivery dates, or cart changes.
PRODUCT_REF and CART_LINE_ID are internal identifiers for tool calls. Do not
show them to shoppers in normal responses.
Do not expose skill names, tool names, entry-mode names, evaluator/judge names,
cache/backend details, structured-field labels, or internal data-layer language
to shoppers. If a detail is unavailable, say it plainly, for example: "I don't
have fabric or care details available for that item."
Do not upgrade shopper assumptions, preference language, or earlier styling
inferences into catalog facts. Separate confirmed product facts from styling
judgment, and keep outfit-wide material or comfort claims item-specific unless
every included piece is supported by tool evidence.
Do not group leather, rubber, metal, or generic canvas under "natural fibers";
attribute materials item by item.
Outdoor-practicality claims require exact support: do not say products are
stable on grass or gravel, water-resistant, all-day comfortable, weather-safe,
secure for a full event, or good for outdoor surfaces unless the catalog says
so. With indirect evidence, state only the confirmed construction detail and
keep the styling judgment separate.
Do not convert sole or strap facts into surface guarantees. Rubber sole means
rubber sole; ankle strap means ankle strap. Do not add grass, gravel, weather,
outdoor-surface performance, all-day comfort, maximum breathability, or
best-in-category performance claims unless those claims are directly supported
by product details.

Catalog capabilities:
{catalog_context}

Rules:
- Every turn begins with shopper-skill activation. Select the smallest set of
  registered skills that covers the complete current intent, then follow the
  injected full instructions before constructing shopping-tool arguments or a
  final response. Never combine activation and shopping calls in one response.
- Outfit styling and product discovery are alternative primary procedures;
  never activate both. Budget shopping may accompany either only as a modifier
  when the shopper states a budget.
- Style-led fashion selection belongs to outfit styling even when the shopper
  asks for one statement or balancing piece. Product discovery is for browse
  and filter intent without styling judgment.
- Keep the primary skill aligned with the active conversation task. An outfit-
  building or styling thread continues to use outfit styling for piece-by-piece
  searches until the shopper changes tasks; do not reclassify a follow-up as
  product discovery merely because it asks for one product type.
- The model owns the semantic choice of taxonomy values from the exact values
  advertised in Catalog capabilities. Runtime validation only enforces that
  contract; it does not infer shopper meaning or substitute product types.
- Product discovery, product recommendations, budget filters, and image-similar
  shopping use search_catalog_tool. Every call declares `taxonomy_status` as
  `exact_requested_type`, `member_of_requested_umbrella`,
  `agent_selected_type`, `no_direct_catalog_match`, or `image_only`. Exact means
  the selected taxonomy directly represents the requested focused role. For a
  single selected value, the requested type must name that value; the semantic
  query may focus on soft ranking direction. Umbrella-member means the shopper
  named a true umbrella and every selected value is its child. Never reverse
  that relation merely because the requested type belongs to a broader selected
  category. Apply one direction test to every selected value: "is this a kind of
  the product scope the shopper asked for?" A skirt is a kind of bottom; a flat
  or sandal is not a kind of sneaker even though all are footwear. If no value
  qualifies because a broad styling or discovery request did not name a product
  type, `agent_selected_type` may choose exactly one advertised subcategory as
  the focused starting role. It must not replace
  an explicitly requested unavailable type. Evaluate every product role named
  in the current turn independently. If the shopper names a product type for a
  role, including as an alternative, confirmation, comparison, or follow-up,
  `agent_selected_type` is forbidden for that role. If no value
  qualifies for an explicitly requested concrete product type, use
  `no_direct_catalog_match` with empty taxonomy arrays; that path reports the
  gap without retrieval and has no required constraints. Decide it from the
  requested product type alone: an unavailable attribute or subjective modifier
  does not erase an advertised type. Never use that status for an outfit, occasion, season,
  weather need, style/vibe, or product attribute. Never search an adjacent type
  as a substitute.
- Every search call must include exactly one `semantic_query`, one required
  nullable `requested_product_type`, one product-agnostic `shopper_guidance`,
  one `taxonomy_status`, one `taxonomy` object, and one `required_constraints`
  object, plus `scope_complete`.
  `requested_product_type` is the shortest product noun or true umbrella from
  the current turn or direct antecedent. Exclude color, material, fit,
  occasion, weather, and style modifiers; for `agent_selected_type`, use the
  chosen advertised role noun. It is null only for image-only search. Set
  `scope_complete` true only when this search
  plus existing turn evidence is enough to answer the shopper's complete current
  request. A recommendation-only one-role request is complete after its one
  inclusive search. Set it false while an explicitly requested product role,
  product-detail verification, availability check, or cart action still must
  run after the search. Never set it false merely to search alternatives. A
  requested type with no faithful advertised taxonomy match does not make a
  one-role scope partial; report that gap without another search.
  Judge completeness against the current user turn, not an unfinished broader
  outfit project. "Start with a beige top" and "What bottoms go with that?" are
  each complete after their one-role search. Put
  product meaning and soft preferences in `semantic_query`. Use an
  empty string only for image-only search; do not create synonyms, paraphrases,
  or query expansions as additional searches.
- Write `shopper_guidance` before retrieval as one concise sentence connecting
  the selected role to the shopper's stated goal or direct antecedent. It may
  express product-agnostic styling judgment, but it must not name or describe
  candidate products, name product types outside the selected advertised scope,
  or mention internal search mechanics. Use an empty string for image-only and
  no-direct requests.
- The `taxonomy.category` and `taxonomy.subcategory` arrays contain only the
  exact values allowed by the current tool schema, which is generated from the
  active Catalog capabilities. List each value once. Use all applicable values
  for an inclusive request containing alternatives. When both arrays are used,
  every subcategory must belong to a selected category and every selected
  category must own at least one selected subcategory. Every text search needs
  at least one taxonomy value when `taxonomy_status` is `exact_requested_type`
  or `member_of_requested_umbrella`; if a broad request has no clear scope, ask
  one concise clarifying question. Both arrays may be empty only for an
  `image_only` search or a `no_direct_catalog_match` no-retrieval result. Taxonomy is mapped
  deterministically to the Catalog's actual filter field names.
- Each catalog call covers at most one category and one focused product role.
  Include every faithful advertised subtype for that role in the same call, but
  never mix apparel, footwear, bags, or other categories in one retrieval. A
  multi-role request uses separate focused calls up to the search cap. An outfit
  or style/vibe is not a taxonomy umbrella; use `agent_selected_type` to choose
  one focused starting role when the shopper did not name one.
- If an explicitly requested product type has no faithful advertised taxonomy
  value, do not broaden to its parent category, omit it, or silently substitute
  an adjacent type. Tell the shopper that type is unavailable and ask before
  searching an alternative.
- Never describe an unadvertised or unselected product type as advertised. When
  offering a next search direction, name only exact taxonomy values shown in
  Catalog capabilities or the current search's taxonomy evidence.
- Put every non-taxonomy shopper must-have in `required_constraints`, including
  requirements whose fields are semantic/detail-only or absent from Catalog
  capabilities. Do not omit an unsupported must-have or rely on semantic search
  to enforce it; deterministic validation must refuse that search instead of
  weakening it. Preserve performance requirements such as water resistance or
  machine washability here as well. An attribute that defines the requested
  products is required even without the words "must have." Semantic relevance cannot guarantee
  a must-have requirement. Recommendation adjectives such as comfortable,
  relaxed, soft, breathable, lightweight, casual, dressy, bold, bright,
  vibrant, or sporty always remain semantic preferences, never objective hard
  filters. Before every search, compare each modifier on the target product
  with the advertised filter schema and copy every exact matching value into
  `required_constraints`; never leave the object empty when one applies. Use only advertised
  filter properties directly. For "Do you have
  water-resistant bags?", include
  `{{"unadvertised_requirements": ["water resistance"]}}`; do not send an
  empty object.
- Apply hard constraints only to the target products named in the current turn.
  An anchor's confirmed color, material, or other attribute is styling context,
  not a hard filter for a complementary role, unless the shopper explicitly asks
  for the same value or palette.
- For required constraints advertised as hard filters, enum values must exactly
  match listed values and numeric values use an object with `min` and/or `max`.
  When one shopper constraint includes multiple applicable advertised enum
  values, include all of them in one list and one search rather than trying one
  value at a time.
- Media-only or descriptive media requests such as "what's in this look",
  "describe this outfit", "what am I wearing", or "what colors are here" must
  be answered from MEDIA ANALYSIS. Do not call search_catalog_tool and do not
  show catalog products unless the shopper explicitly asks to find, shop,
  recommend, compare, price-check, check availability, or add an item.
- Use at most {self.config.max_catalog_searches_per_turn} catalog search calls per
  user turn. One normalized taxonomy-and-required-constraint scope can execute
  only once in a turn; do not retry the same hard-filter scope with different
  semantic wording. For outfit requests
  with multiple required item types, run one focused search per distinct
  taxonomy scope, then stop and synthesize from those results.
- An outfit request with a season, weather need, occasion, or style/vibe already
  has enough direction to begin with a grounded partial outfit. Do not answer
  only with a questionnaire; search the most useful core role first and ask at
  most one concise follow-up while presenting the grounded result.
- An unspecified request for one style-led piece, such as a statement piece,
  does not identify a product role. Ask one concise category or occasion question
  before searching. This does not apply to an outfit or complete-look request,
  where the named vibe, occasion, season, or weather need is enough to begin.
- A whole or complete outfit remains incomplete until current or directly
  referenced prior evidence covers multiple complementary roles, or the search
  cap is reached and the missing role is disclosed. A one-piece dress may be the
  clothing core, but does not by itself complete an outfit request.
- For a broad style/vibe request, select a useful core role from exact taxonomy
  values currently advertised by the catalog. Do not invent an unadvertised
  product type from the vibe or copy a generic styling example into taxonomy.
- Treat broad weather or occasion context as styling direction, not automatically
  as a product-attribute guarantee. A "rainy day outfit" or "wet-weather outfit"
  should search practical
  advertised roles without adding an unadvertised requirement; add water
  resistance to `unadvertised_requirements` only when the shopper directly
  requires it for a product. "Rainy day outfit" does not imply water resistance;
  "water-resistant bags" directly requires it.
- Subjective style/vibe language is semantic direction unless the shopper makes
  it an explicit hard requirement. Objective product attributes such as material,
  weather performance, sale status, or a specific shade remain must-haves when
  they define the requested product.
- For alternatives joined by "or", search the faithful advertised branch and
  preserve every named branch. Search a supported branch with
  `scope_complete=false` when another branch is unavailable, then report the
  unavailable branch with one `no_direct_catalog_match` call. Do not reject a
  supported branch merely because another alternative is unavailable, and never
  list the supported taxonomy value in `unadvertised_requirements`.
- When every named alternative is advertised, use
  `member_of_requested_umbrella` and include all of them in one call. Do not use
  `agent_selected_type` to narrow an explicit umbrella or alternatives.
- Use at most {self.config.max_product_detail_reads_per_turn} product-detail
  reads per user turn. Product details are for direct product fact questions,
  cart or comparison follow-ups, or already-shortlisted items; they are not
  required for an initial no-anchor outfit recommendation.
- If a tool returns STOP_TOOL_USE, stop tool calling immediately and produce the
  best concise shopper-facing answer from the evidence already available.
- A tool result is enough to produce a final answer. Once you have at least one
  plausible product for each required item type, answer from those results. Do
  not keep searching for alternatives unless the shopper explicitly rejects the
  current result.
- A request for one product role gets one inclusive search scope containing all
  faithful advertised types for that role. Do not use remaining search budget
  on adjacent categories, one-piece substitutes, or unrelated product types.
  A dress is not a bottom and does not satisfy a request for separates.
- In an active styling thread, a group of recent candidates can be the direct
  antecedent. If they share a confirmed constraint that is sufficient for the
  next request, use it as the provisional anchor and search the requested role.
  Do not require one exact product selection or an occasion when the shopper is
  asking for generally compatible options and the shared anchor is sufficient.
- In the final response to a follow-up search, explicitly connect the new
  candidates to the named antecedent or to that candidate group's shared
  confirmed constraint. Do not return an unexplained product list.
- Product-detail or research questions about a product already returned by
  search_catalog_tool should use get_product_details_tool with that
  PRODUCT_REF. Do not run another broad catalog search for known-product facts.
- Initial recommendations should use product name, price, category or role,
  and one styling reason. Do not enumerate materials, dimensions, pockets,
  closures, care, or construction details unless the shopper asks for those
  details and you have called get_product_details_tool.
- For search-only recommendations, keep every product line to name, price,
  category/role, image availability when useful, and exact confirmed filters.
  Put the styling reason in a separate sentence based on role and shopper
  context; never derive it by interpreting words in the display name.
- A successful search may report confirmed filters. Every returned product
  passed each reported predicate. A single allowed value confirms that value;
  multiple allowed values confirm membership in the set, not which value each
  product has. Do not infer an adjacent attribute from a confirmed filter.
- Search-only product names are display names, not confirmed attributes. Do not
  parse length, color, print, material, construction, fit, care, or formality
  from descriptive names unless product details confirm the attribute. You may
  say "candidate" or "could be worth checking" and offer to pull details.
- Do not make group-level claims such as "all are maxi length", "both are
  cotton", "the lightest", "most polished", or "best for heat" unless every
  item in the group has product-detail evidence supporting that exact claim.
- For no-anchor outfit building, do not call product details just to make the
  outfit sound richer. Search by the needed item roles, choose a coherent set,
  and keep the rationale to color, proportion, formality, silhouette, and
  shopper goal.
- If the shopper mentions outdoor practicality in a broad outfit request,
  prefer searched categories that naturally fit the situation, such as flat
  shoes or a light layer, but do not state product material, breathability,
  ground stability, outdoor-surface performance, heat performance, or
  all-evening comfort unless the shopper asks a direct product-specific
  question and details support it.
- Product comparison tables, material claims, dimensions, pocket/closure
  details, care/washability answers, comfort claims, and outdoor-practicality
  claims require get_product_details_tool for each relevant PRODUCT_REF before
  finalizing the answer. If you have only search results, keep the answer to
  names, prices, and brief candidate fit.
- Even after product details, compare only confirmed construction facts for
  surface or weather concerns: lower heel versus higher heel, strap versus no
  strap, rubber sole versus unspecified sole, zip closure versus open top. Do
  not state the resulting performance on grass, gravel, rain, bugs, spills, or
  outdoor ground unless product details explicitly state it.
- Shopper wording is not product evidence. If the shopper mentions an
  unverified attribute such as heel shape, material, colorway, fit, or care,
  verify it with tools or refer to it as the shopper's preference, not as a
  catalog fact.
- When the shopper asks to add an item that has not already been searched in
  this conversation, call search_catalog_tool first, then call
  add_cart_items_tool with a one-item list containing the selected PRODUCT_REF.
- If an image is attached, the current image is already available to
  search_catalog_tool. Use that tool for "this", "similar", and image-price
  refinement requests.
- If MEDIA ANALYSIS is present, use it as the visual/video understanding of
  the attached media. It can guide search_catalog_tool queries and follow-up
  pronoun resolution, but catalog results remain the source of truth for
  product names and prices. Catalog results are not inventory evidence.
- If MEDIA ANALYSIS says media analysis failed, VLM authentication failed, the
  VLM is unavailable, or video understanding is not configured, say so plainly.
  Do not infer video-similar products from the media; ask the shopper for a
  text description or search only from explicit text in the shopper request.
  If an image is attached, image embedding search through search_catalog_tool is
  still available even when MEDIA ANALYSIS is unavailable.
- Cart reads require get_cart_tool. Cart totals require view_cart_total_tool.
- Use recent discussion, not CURRENT CART, to resolve ordinary product and
  styling references such as "that" and "those." A discussed anchor does not
  need to be in the cart for styling advice. Mention that an item is absent from
  the cart only when the shopper asks about cart contents or a cart mutation.
- Cart mutations require explicit shopper intent and must use
  add_cart_items_tool, remove_cart_item_tool, or update_cart_items_tool. Never
  claim a cart mutation unless the tool reports success.
- Cart mutation scope must match the shopper's explicit add or remove request.
  Selection, approval, or styling preference is not cart intent by itself.
  If the shopper asks to "add those", add only the items named in that add
  request or its direct antecedent. Do not add earlier anchor, core outfit, or
  optional pieces unless the shopper explicitly includes them in the cart
  request.
- For an explicit cart swap, finish the whole swap before the final response:
  remove the rejected cart line, add the selected replacement when a valid
  PRODUCT_REF is already available, then summarize the updated cart. If the
  replacement has not been searched in this conversation, search first.
- If cart mutation scope is ambiguous, ask one concise clarification before
  calling any cart mutation tool. Example: "Do you want me to add just the bag,
  layer, and earrings, or the full outfit including the dress and sandals?"
- For cart styling requests, inspect CURRENT CART or call get_cart_tool first.
  Do not search for products already named as cart contents just to verify them.
  If the cart is empty but the shopper names items, say you do not see those
  items in the cart yet, then give provisional styling advice from the named
  items without claiming cart truth. Search at most once for a missing piece
  only after identifying the gap.
- Use PRODUCT_REF from search_catalog_tool when adding items. Do not pass
  display names as product_ref values to add_cart_items_tool. Include
  expected_display_name for each item so the tool can verify that the selected
  PRODUCT_REF resolves to the shopper-facing product name you intend to add.
- When the shopper asks to add multiple selected products, call
  add_cart_items_tool once with an item list. The tool may report partial
  success; the final answer must clearly distinguish added items from failures.
- Use PRODUCT_REF from search_catalog_tool when requesting product details. Do
  not pass display names to get_product_details_tool.
- Use internal identifiers only in tool calls. Do not expose PRODUCT_REF or
  CART_LINE_ID in customer-facing responses.
- Use CART_LINE_ID from CURRENT CART or get_cart_tool when removing an item. Do
  not guess cart line IDs from product names.
- Use update_cart_items_tool for quantity changes. Set quantity to zero only
  when the shopper explicitly asks to remove that line.
- Store policy questions about returns, shipping, sizing, payment, price
  matching, or gift cards require get_store_policy_tool. Never substitute model
  knowledge for policy content that the tool does not return.
- Explicit stock, inventory, or size availability questions
  require check_product_availability_tool with a PRODUCT_REF from a prior
  search. Relay its deterministic result rather than guessing from catalog
  presence.
- If the shopper asks for anything under a budget without a product type,
  category, occasion, style, outfit goal, or image, ask one concise clarifying
  question instead of guessing.
- Tax and delivery dates are not available through the current tools. Do not
  treat a catalog result alone as proof that an item is in stock or ready to
  ship; availability claims require check_product_availability_tool.
- Previous preference context is guidance only. The current shopper request
  wins when it conflicts with previous preferences.
- Keep final answers concise and grounded in tool results. Attribute materials,
  comfort, construction, and outdoor-practicality claims to the specific items
  that support them instead of making unsupported whole-outfit claims. Avoid
  guarantee language such as "will stay comfortable all evening" unless the
  catalog evidence supports the guarantee. Before finalizing, remove or soften
  unsupported phrases about grass, gravel, water resistance, all-day comfort,
  maximum breathability, or best-in-category performance.
"""
        return prompt

    def _build_user_message(self, state: State, identity: RequestIdentity) -> str:
        return (
            f"REQUEST ID: {identity.request_id}\n"
            f"SESSION ID: {identity.session_id}\n"
            f"CONVERSATION ID: {identity.conversation_id}\n"
            f"CART ID: {identity.cart_id}\n\n"
            f"USER QUERY: {state.query}\n"
            f"IMAGE ATTACHED: {'yes' if state.image else 'no'}\n\n"
            f"MEDIA ATTACHED:\n{_format_media_summary(state.media)}\n\n"
            f"MEDIA ANALYSIS:\n{state.media_analysis or '(none)'}\n\n"
            f"CURRENT CART:\n{_format_cart(state.cart)}\n\n"
            f"RECENT DISCUSSION:\n{state.context or '(none)'}"
        )

    def _remember_products(
        self, identity: RequestIdentity, products: list[ProductSummary]
    ) -> None:
        refs = self._product_refs.setdefault(identity.conversation_id, {})
        for product in products:
            refs[product.product_id] = product
        while len(refs) > 50:
            refs.pop(next(iter(refs)))

    def _product_from_ref(
        self, identity: RequestIdentity, product_ref: str
    ) -> ProductSummary | None:
        return self._product_refs.get(identity.conversation_id, {}).get(
            (product_ref or "").strip()
        )

    def _append_cached_cart_images(
        self,
        retrieved: dict[str, str],
        cart: Cart,
        identity: RequestIdentity,
    ) -> None:
        if not cart.contents:
            return
        refs = self._product_refs.get(identity.conversation_id, {})
        if not refs:
            return

        products_by_key: dict[str, ProductSummary] = {}
        for product in refs.values():
            products_by_key[product.product_id] = product
            products_by_key[product.display_name] = product

        for item in cart.contents:
            product = products_by_key.get(str(item.get("product_id") or ""))
            if product is None:
                product = products_by_key.get(str(item.get("item") or ""))
            if product is not None and product.image_url:
                retrieved[product.display_name] = product.image_url

    def _read_cart(self, user_id: int) -> Cart:
        result = get_cart(GetCartInput(user_id=str(user_id)), self.config.memory_port)
        if not result.ok or result.cart is None:
            return Cart()
        return Cart(
            contents=[
                {
                    "cart_line_id": line.cart_line_id,
                    "product_id": line.product_id,
                    "item": line.display_name,
                    "amount": line.quantity,
                    "price": line.unit_price.amount if line.unit_price else None,
                }
                for line in result.cart.lines
            ]
        )

    def _start_conversation_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> TurnStartResult | None:
        """Start one durable turn and apply its context/cart snapshot."""

        start = time.monotonic()
        try:
            turn = self._conversation_memory.start_turn(
                identity.conversation_id,
                request_id=identity.request_id,
                shopper_text=state.query,
                media=state.media,
                cart_user_id=identity.cart_user_id,
            )
        except (ConversationMemoryError, ValidationError) as exc:
            logger.error("Failed to start durable conversation turn: %s", exc)
            state.context = ""
            state.cart = Cart()
            if getattr(exc, "status_code", None) == 409:
                error_code = getattr(exc, "code", "memory_turn_conflict")
                if error_code in {"turn_in_progress", "conversation_turn_in_progress"}:
                    state.response = (
                        "This conversation is still processing another request. "
                        "Please retry shortly."
                    )
                elif error_code == "turn_abandoned":
                    state.response = (
                        "That earlier request was interrupted. Please retry with "
                        "a new request."
                    )
                elif error_code == "turn_superseded":
                    state.response = (
                        "That interrupted request was superseded by a newer turn. "
                        "Please continue from the latest conversation state."
                    )
                else:
                    state.response = (
                        "That request identifier was already used for different "
                        "input. Please retry with a new request."
                    )
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            else:
                state.response = (
                    "I cannot safely load this conversation right now. "
                    "Please retry shortly."
                )
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "memory_start_failed"
                )
                state.agent_diagnostics["memory_start_error"] = getattr(
                    exc,
                    "code",
                    "memory_start_payload_invalid",
                )
            return None
        finally:
            state.timings["memory"] = time.monotonic() - start

        state.context = format_conversation_context(
            turn.recent_turns,
            max_chars=max(1000, int(self.config.memory_length)),
        )
        state.cart = Cart(
            contents=[
                item.model_dump(mode="json", exclude_none=True) for item in turn.cart
            ]
        )
        return turn

    def _restore_replayed_turn(
        self,
        state: State,
        turn: TurnStartResult,
    ) -> State:
        """Restore a finalized turn without repeating model or tool work."""

        state.response = turn.assistant_text or (
            "That earlier request did not complete. Please retry with a new request."
        )
        if turn.output is not None:
            state.product_results = [
                product.model_dump(mode="json")
                for product in turn.output.product_results
            ]
            state.retrieved = dict(turn.output.retrieved)
            state.agent_diagnostics = dict(turn.output.agent_diagnostics)
        else:
            state.agent_diagnostics = _empty_agent_diagnostics(
                turn.termination_reason or "durable_turn_replayed"
            )
        return state

    def _finalize_conversation_turn(
        self,
        state: State,
        identity: RequestIdentity,
        turn: TurnStartResult,
        *,
        status: FinalTurnStatus | None = None,
        termination_reason: str | None = None,
    ) -> None:
        """Persist one terminal turn without changing its shopper response."""

        reason = termination_reason or str(
            state.agent_diagnostics.get("final_termination_reason") or "completed"
        )
        final_status = status or _conversation_turn_status(reason)
        state.agent_diagnostics["final_termination_reason"] = reason
        start = time.monotonic()
        try:
            output = TurnReplayOutput(
                product_results=state.product_results,
                retrieved=state.retrieved,
                agent_diagnostics=state.agent_diagnostics,
            )
            self._conversation_memory.finalize_turn(
                identity.conversation_id,
                turn.turn_id,
                request_id=identity.request_id,
                attempt_id=turn.attempt_id,
                assistant_text=state.response,
                status=final_status,
                termination_reason=reason,
                output=output,
            )
        except (ConversationMemoryError, ValidationError) as exc:
            logger.error("Failed to finalize durable conversation turn: %s", exc)
            error_code = getattr(
                exc,
                "code",
                "memory_finalize_payload_invalid",
            )
            state.agent_diagnostics["memory_finalize_error"] = error_code
            if error_code == "turn_attempt_superseded":
                state.response = (
                    "This request was superseded by a newer attempt. "
                    "Please use the latest response."
                )
                state.product_results = []
                state.retrieved = {}
                state.agent_diagnostics["final_termination_reason"] = error_code
        finally:
            state.timings["memory"] = state.timings.get("memory", 0.0) + (
                time.monotonic() - start
            )

    def _check_safety(self, mode: str, user_id: int, text: str) -> tuple[bool, bool]:
        endpoint = "input" if mode == "input" else "output"
        try:
            response = requests.post(
                f"{self.config.rails_port}/rail/{endpoint}/check",
                json={"user_id": user_id, "query": text},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Guardrails %s check failed: %s", mode, exc)
            return True, False

        responses = payload.get("response") or []
        if not responses:
            return True, True
        return responses[0].get("content") == text, True


def create_request_identity(
    *,
    legacy_user_id: int,
    session_id: str | None = None,
    conversation_id: str | None = None,
    cart_id: str | None = None,
    request_id: str | None = None,
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
        snapshot = await get_state(invoke_config)
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


def _current_turn_messages(messages: list[Any], request_id: str) -> list[Any]:
    marker = f"REQUEST ID: {request_id}"
    start: int | None = None
    for index, message in enumerate(messages):
        if _message_type(message) != "human":
            continue
        if marker in _content_to_text(_value(message, "content")):
            start = index + 1
    return [] if start is None else messages[start:]


def _prior_turn_messages(messages: list[Any], request_id: str) -> list[Any]:
    """Return messages before the current server-owned request marker."""

    marker = f"REQUEST ID: {request_id}"
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if _message_type(message) != "human":
            continue
        if marker in _content_to_text(_value(message, "content")):
            return messages[:index]
    return []


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
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy"
        )
    ]
    repair_or_no_direct = all(
        content.startswith(
            (
                SEARCH_VALIDATION_ERROR_PREFIX,
                "STOP_TOOL_USE: No faithful advertised catalog taxonomy",
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
    if all(
        tool_name == "search_catalog_tool" and status == "rejected"
        for tool_name, status in business_calls
    ):
        return _REJECTED_CATALOG_SEARCH_RESPONSE
    return None


def _unsupported_requirement_response(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Return the fixed shopper response for an unenforceable must-have."""

    outcomes = _business_tool_result_contents(
        _current_turn_messages(_result_messages(result), request_id)
    )
    unsupported_outcomes = [
        content
        for content in outcomes
        if content.startswith(
            "The requested catalog requirement cannot be enforced:"
        )
    ]
    if unsupported_outcomes and len(unsupported_outcomes) == len(outcomes):
        return _UNSUPPORTED_REQUIREMENT_RESPONSE
    return None


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
            "The requested catalog requirement cannot be enforced:"
        )
        for message in _current_turn_messages(
            _result_messages(result),
            request_id,
        )
    )


def _tool_results_by_call_id(messages: list[Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in messages:
        if _message_type(message) != "tool":
            continue
        tool_call_id = str(_value(message, "tool_call_id") or "")
        if tool_call_id:
            results[tool_call_id] = message
    return results


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
    rejection_reason = _tool_rejection_reason(content)
    if rejection_reason:
        return "rejected", rejection_reason
    if _value(result_message, "status") == "error":
        return "error", None
    if tool_name == "read_file" and content.lower().startswith(
        ("error", "file not found")
    ):
        return "error", None
    return "completed", None


def _tool_rejection_reason(content: str) -> str | None:
    markers = (
        (SKILL_ACTIVATION_REQUIRED, "skill_activation_required"),
        (SKILL_TOOL_NOT_GRANTED, "skill_tool_not_granted"),
        ("SHOPPER_SKILL_ACTIVATION_FAILED:", "skill_activation_failed"),
        (
            "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already searched",
            "duplicate_catalog_scope",
        ),
        ("STOP_TOOL_USE: Catalog search limit reached", "catalog_search_limit"),
        (
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy",
            "no_advertised_taxonomy_match",
        ),
        (
            "STOP_TOOL_USE: Product-detail read limit reached",
            "product_detail_read_limit",
        ),
        (
            "The catalog search request does not match current capabilities:",
            "invalid_catalog_request",
        ),
        (SEARCH_VALIDATION_ERROR_PREFIX, "invalid_catalog_request"),
        (CONSTRAINT_REVIEW_PREFIX, "constraint_review_required"),
        (
            "The requested catalog taxonomy cannot be enforced:",
            "unsupported_catalog_taxonomy",
        ),
        (
            "The requested catalog requirement cannot be enforced:",
            "unsupported_catalog_constraint",
        ),
        (_UNSUPPORTED_SEARCH_MODE_MESSAGE, "unsupported_search_mode"),
    )
    for marker, reason in markers:
        if content.startswith(marker):
            return reason
    if content.startswith("STOP_TOOL_USE:"):
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


def _message_type(message: Any) -> str:
    message_type = str(_value(message, "type") or _value(message, "role") or "")
    return {"assistant": "ai", "user": "human"}.get(message_type, message_type)


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
        content = _content_to_text(_value(message, "content"))
        if source_tool == "search_catalog_tool":
            if "SEARCH_RESULT_GROUNDING_NOTE" not in content:
                continue
            evidence_type = "search_result"
            search_scope = {
                "taxonomy": _bounded_product_evidence_value(
                    _search_taxonomy_evidence(content)
                ),
                "confirmed_filters": _bounded_product_evidence_value(
                    _search_filter_evidence(content)
                ),
            }
        elif source_tool == "get_product_details_tool":
            if "PRODUCT_DETAIL_GROUNDING_NOTE" not in content:
                continue
            evidence_type = "product_detail"
            search_scope = None
        else:
            continue

        for product in _product_evidence_records(content):
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
            if search_scope is not None:
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
    }
    for message in messages:
        if _message_type(message) != "tool":
            continue
        outcome = _search_json_evidence(
            _content_to_text(_value(message, "content")),
            _CATALOG_SCOPE_OUTCOME_PREFIX,
        )
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


def _diagnostic_product_facts(product: dict[str, Any]) -> dict[str, Any]:
    """Extract bounded, structured facts from one parsed product record."""

    facts: dict[str, Any] = {}
    for key in ("category", "brand", "price"):
        value = product.get(key)
        if value:
            facts[key] = _bounded_product_evidence_value(value)
    facts["image_available"] = bool(product.get("image_url"))

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


def _extract_final_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if _message_type(message) in {"human", "system", "tool"}:
                    continue
                if _value(message, "tool_calls"):
                    continue
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                text = _content_to_text(content)
                if text and not text.startswith(
                    (
                        SKILL_ACTIVATION_COMPLETE,
                        SKILL_ACTIVATION_REQUIRED,
                        SKILL_TOOL_NOT_GRANTED,
                        "SHOPPER_SKILL_ACTIVATION_FAILED:",
                    )
                ):
                    return text
        return _content_to_text(result.get("response")) or ""
    return _content_to_text(getattr(result, "content", result)) or ""


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
        content = _content_to_text(_value(message, "content"))
        if not name and "SEARCH_RESULT_GROUNDING_NOTE" in content:
            name = "search_catalog_tool"
        if name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}:
            continue
        tool_names.append(name)
        if name == "search_catalog_tool" and "SEARCH_RESULT_GROUNDING_NOTE" in (
            content
        ):
            has_search_result = True
    return (
        has_search_result
        and set(tool_names) == {"search_catalog_tool"}
    )


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
                "Want a closer comparison? I can verify material, construction, "
                "length, fit, care, or weather performance for a specific option."
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
        content = _content_to_text(_value(message, "content"))
        if "SEARCH_RESULT_GROUNDING_NOTE" not in content:
            continue
        guidance = _search_json_evidence(
            content,
            _SEARCH_GUIDANCE_EVIDENCE_PREFIX,
        )
        groups.append(
            {
                "guidance": _scrub_internal_shopper_language(
                    str(guidance.get("text") or "")
                ).strip(),
                "products": _product_evidence_records(content),
                "taxonomy": _search_taxonomy_evidence(content),
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


def _format_search_group(
    group: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    index: int,
) -> list[str]:
    """Format one search group's bounded guidance and verified products."""

    taxonomy = group.get("taxonomy") or {}
    values = taxonomy.get("subcategory") or taxonomy.get("category") or []
    label = " / ".join(str(value).replace("_", " ") for value in values)
    title = label.title() if label else f"Product group {index}"
    lines = [f"**{title}**", "", "General guidance (not product-specific facts):"]
    lines.extend((str(group["guidance"]), "", "Catalog candidates:", ""))
    for product in products:
        parts = [f"**{product['name']}**"]
        if product.get("price"):
            parts.append(str(product["price"]))
        if product.get("category"):
            parts.append(str(product["category"]).replace("_", " "))
        lines.append("- " + " — ".join(parts))
    lines.append("")
    return lines


def _no_direct_search_types(result: Any, *, request_id: str) -> list[str]:
    """Return current-turn product types with a server-authored no-direct outcome."""

    product_types: list[str] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        outcome = _search_json_evidence(
            _content_to_text(_value(message, "content")),
            _CATALOG_SCOPE_OUTCOME_PREFIX,
        )
        if not outcome or outcome.get("outcome") != "no_direct_catalog_match":
            continue
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
        content = _content_to_text(_value(message, "content"))
        if "SEARCH_RESULT_GROUNDING_NOTE" not in content:
            continue
        payload = _search_json_evidence(
            content,
            _SEARCH_GUIDANCE_EVIDENCE_PREFIX,
        )
        text = _scrub_internal_shopper_language(
            str((payload or {}).get("text") or "")
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
        content = _content_to_text(_value(message, "content"))
        filters = _search_filter_evidence(content)
        statements: list[str] = []
        for name, value in filters.items():
            statement = _format_filter_statement(name, value)
            if statement:
                statements.append(statement)
        if not statements:
            continue
        product_names = [
            product["name"]
            for product in _product_evidence_records(content)
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


def _format_filter_statement(name: str, value: Any) -> str:
    label = name.replace("_", " ")
    if isinstance(value, list):
        values = [str(item).replace("_", " ") for item in value]
        if len(values) == 1:
            return f"{label} is {values[0]}"
        if values:
            return f"{label} is one of {', '.join(values)}"
        return ""
    if isinstance(value, dict):
        bounds = []
        if value.get("min") is not None:
            bounds.append(f"minimum {value['min']}")
        if value.get("max") is not None:
            bounds.append(f"maximum {value['max']}")
        return f"{label} {' and '.join(bounds)}" if bounds else ""
    return f"{label} is {value}"


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
        parts.append(_customer_safe_tool_evidence(content))

    evidence = "\n\n---\n\n".join(parts).strip()
    if len(evidence) <= max_chars:
        return evidence
    return evidence[-max_chars:]


def _customer_safe_tool_evidence(content: str) -> str:
    if content.startswith(SEARCH_VALIDATION_ERROR_PREFIX):
        return (
            "CUSTOMER_SAFE_INVALID_SEARCH_EVIDENCE: No valid catalog search "
            "scope was established and no retrieval ran. This does not support "
            "a product-availability or catalog-absence claim."
        )
    if content.startswith(
        "STOP_TOOL_USE: No faithful advertised catalog taxonomy"
    ):
        return (
            "CUSTOMER_SAFE_NO_MATCH_EVIDENCE: The active catalog has no direct "
            "advertised taxonomy match for this requested product role. "
            "No retrieval ran and no alternative product type was selected. Say "
            "that plainly, preserve any successful evidence for other roles, and "
            "ask permission before searching a different advertised type. Do not "
            "name alternatives."
        )
    if "SEARCH_NO_MATCH_GROUNDING_NOTE" in content:
        taxonomy = _search_taxonomy_evidence(content)
        confirmed_filters = _search_filter_evidence(content)
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
        return "\n".join(lines)
    if "SEARCH_RESULT_GROUNDING_NOTE" in content:
        return _summarize_product_evidence(
            content,
            heading="CUSTOMER_SAFE_SEARCH_EVIDENCE",
            note=(
                "Search results support only product names, prices, categories, "
                "image availability, confirmed search filters, and a modest "
                "styling role. Beyond an exact confirmed filter, they do not "
                "support length, color, print, materials, care, construction, "
                "fit, comfort, weather, grass, gravel, heat, or best-in-category "
                "claims. Treat names as display names, not attribute evidence; "
                "group claims require product-detail evidence for every item."
            ),
            confirmed_filters=_search_filter_evidence(content),
            taxonomy_scope=_search_taxonomy_evidence(content),
        )
    if "PRODUCT_DETAIL_GROUNDING_NOTE" in content:
        return _summarize_product_evidence(
            content,
            heading="CUSTOMER_SAFE_PRODUCT_DETAIL_EVIDENCE",
            note=(
                "Product details were read for these products, but the available "
                "detail data contains only the listed facts. Do "
                "not state material, care, dimensions, closures, fit, sizing, "
                "colorways, or outdoor performance unless the field appears in "
                "this evidence summary."
            ),
        )
    return _summarize_cart_evidence(content)


def _summarize_product_evidence(
    content: str,
    *,
    heading: str,
    note: str,
    confirmed_filters: dict[str, Any] | None = None,
    taxonomy_scope: dict[str, Any] | None = None,
) -> str:
    products = _product_evidence_records(content)
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
        return "\n".join(lines + [_strip_internal_ids_from_evidence_line(content)])
    for product in products:
        summary_parts = [product["name"]]
        if product.get("category"):
            summary_parts.append(f"category: {product['category']}")
        if product.get("price"):
            summary_parts.append(f"price: {product['price']}")
        if product.get("image_url"):
            summary_parts.append("image: available")
        if product.get("details"):
            summary_parts.append("details: " + "; ".join(product["details"]))
        lines.append("- " + " | ".join(summary_parts))
    return "\n".join(lines)


def _search_filter_evidence(content: str) -> dict[str, Any]:
    """Read the canonical hard-filter marker from one search result."""

    return _search_json_evidence(content, _SEARCH_FILTER_EVIDENCE_PREFIX)


def _search_taxonomy_evidence(content: str) -> dict[str, Any]:
    """Read the advertised taxonomy marker from one search result."""

    return _search_json_evidence(content, _SEARCH_TAXONOMY_EVIDENCE_PREFIX)


def _search_json_evidence(content: str, prefix: str) -> dict[str, Any]:
    """Read one JSON evidence marker from a tool result."""

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        payload = line.removeprefix(prefix).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _product_evidence_records(content: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    reading_details = False
    key_map = {
        "NAME:": "name",
        "CATEGORY:": "category",
        "BRAND:": "brand",
        "PRICE:": "price",
        "IMAGE_URL:": "image_url",
    }
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("PRODUCT_REF:"):
            if current.get("name"):
                records.append(current)
            current = {"product_ref": line.removeprefix("PRODUCT_REF:").strip()}
            reading_details = False
            continue
        if line == "DETAILS:":
            reading_details = True
            continue
        if reading_details and line.startswith("- ") and ":" in line:
            label, value = line[2:].split(":", 1)
            if label.strip() and value.strip():
                current.setdefault("details", []).append(
                    f"{label.strip()}: {value.strip()}"
                )
            continue
        for prefix, key in key_map.items():
            if line.startswith(prefix):
                current[key] = line[len(prefix) :].strip()
                break
    if current.get("name"):
        records.append(current)
    return records


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


def _result_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            return messages
        return [result]

    messages = getattr(result, "messages", None)
    if isinstance(messages, list):
        return messages
    return [result]


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


def _value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _token_int(record: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _value(record, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None


def _format_search_filter_evidence(filters: dict[str, Any]) -> str:
    """Format canonical hard filters proven by a successful search."""

    return (
        f"{_SEARCH_FILTER_EVIDENCE_PREFIX} "
        + json.dumps(filters, sort_keys=True, default=str)
    )


def _format_search_direction_evidence(semantic_query: str) -> str:
    """Record the model-authored preference used for successful ranking."""

    return (
        f"{_SEARCH_DIRECTION_EVIDENCE_PREFIX} "
        + json.dumps(semantic_query, ensure_ascii=False)
    )


def _format_search_guidance_evidence(shopper_guidance: str) -> str:
    """Record bounded product-agnostic guidance authored before retrieval."""

    return (
        f"{_SEARCH_GUIDANCE_EVIDENCE_PREFIX} "
        + json.dumps({"text": shopper_guidance.strip()}, ensure_ascii=False)
    )


def _format_search_taxonomy_evidence(taxonomy: dict[str, Any]) -> str:
    """Format the advertised taxonomy scope used by a successful search."""

    return (
        f"{_SEARCH_TAXONOMY_EVIDENCE_PREFIX} "
        + json.dumps(taxonomy, sort_keys=True, default=str)
    )


def _format_catalog_scope_outcome(outcome: dict[str, Any]) -> str:
    """Format one bounded non-product catalog outcome for diagnostics."""

    return (
        f"{_CATALOG_SCOPE_OUTCOME_PREFIX} "
        + json.dumps(outcome, sort_keys=True, default=str)
    )


def _format_product(product: Any) -> str:
    lines = [
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if getattr(product, "category", None):
        lines.append(f"CATEGORY: {product.category}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    lines.append(
        "DETAILS: Call get_product_details_tool with this PRODUCT_REF before "
        "stating materials, dimensions, pockets, closures, care, comfort, or "
        "outdoor-practicality claims."
    )
    return "\n".join(lines)


def _format_product_details(product: ProductDetail) -> str:
    lines = [
        _PRODUCT_DETAIL_GROUNDING_NOTE,
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if product.category:
        lines.append(f"CATEGORY: {product.category}")
    if product.brand:
        lines.append(f"BRAND: {product.brand}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    if product.attributes:
        lines.append("DETAILS:")
        for name, value in sorted(product.attributes.items()):
            lines.append(
                f"- {name.replace('_', ' ')}: {_format_detail_value(value)}"
            )
    else:
        lines.append("NO_ADDITIONAL_STRUCTURED_DETAILS")
    return "\n".join(lines)


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(
            f"{key}={value[key]}" for key in sorted(value)
        )
    return str(value)


def _normalize_cart_add_tool_items(
    items: list[AddCartItemsToolItemInput] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
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
        entry = normalized.setdefault(
            parsed.product_ref,
            {
                "quantity": 0,
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
    cached_products: Any,
) -> list[str]:
    explicitly_named = _explicitly_named_products(user_query, cached_products)
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
            f"{_format_cached_product_refs(explicitly_named)}. Retry with matching "
            "PRODUCT_REF values only, or ask a clarification."
        )
    return failures


def _explicitly_named_products(text: str, cached_products: Any) -> list[ProductSummary]:
    normalized_text = _normalize_product_name(text)
    if not normalized_text:
        return []

    padded_text = f" {normalized_text} "
    matches: list[ProductSummary] = []
    seen: set[str] = set()
    products = list(cached_products)
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


def _format_cached_product_refs(products: list[ProductSummary]) -> str:
    return ", ".join(
        f"{product.display_name} (PRODUCT_REF: {product.product_id})"
        for product in products
    )


def _scrub_internal_shopper_language(text: str) -> str:
    scrubbed = text or ""
    for internal, replacement in _INTERNAL_SHOPPER_REPLACEMENTS:
        scrubbed = scrubbed.replace(internal, replacement)
    return scrubbed


def _format_cart_add_result(added: list[str], failed: list[str], cart: Cart) -> str:
    lines = ["CART_ADD_RESULT"]
    if added:
        lines.append("Added:")
        lines.extend(added)
    if failed:
        lines.append("Failed:")
        lines.extend(failed)
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
        lines.append(
            f"- CART_LINE_ID: {cart_line_id} | "
            f"{item.get('amount', 1)} x {item.get('item', '')}{suffix}"
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


def _format_policy_result(result: GetStorePolicyResult) -> str:
    if not result.ok or result.policy is None:
        message = result.error.message if result.error else "unknown error"
        return f"POLICY NOT AVAILABLE: {message}"
    policy = result.policy
    return f"STORE POLICY — {policy.title}\n{policy.body}"


def _format_availability_result(result: CheckProductAvailabilityResult) -> str:
    return f"AVAILABILITY ({result.product_ref}): {result.message}"


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


def _format_retrieved_images(retrieved: dict[str, str] | None) -> str:
    if not retrieved:
        return "(none)"
    lines = []
    for name, image_url in retrieved.items():
        lines.append(f"- {name}: image available")
    return "\n".join(lines)


def _format_media_summary(media: list[dict[str, Any]]) -> str:
    if not media:
        return "(none)"
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown")
        counts[media_type] = counts.get(media_type, 0) + 1
    return ", ".join(f"{count} {media_type}(s)" for media_type, count in sorted(counts.items()))


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
