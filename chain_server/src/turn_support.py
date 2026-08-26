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
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from types import SimpleNamespace
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
    not_carried_of,
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
        default_factory=list,
        description=(
            "Exact advertised subcategory values required by the shopper. Use an "
            "empty list when category supplies the text-search scope or the search "
            "is image-only. Omitting it means the whole category, which is what "
            "'show me some jewellery' asks for."
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


WEATHER_PLACE_NOT_IN_THIS_TURN = (
    "WEATHER_PLACE_NOT_STATED: no forecast for this turn -- the words you "
    "quoted are not in it, and a place named on an earlier turn is not where "
    "the shopper is asking about now.\n"
    "Carry on and answer them. A forecast was never the request: they asked "
    "what to wear. If they said what the conditions will be -- \"it's going "
    "to snow when we get back\" -- that is the answer to the weather "
    "question, they are the authority on their own trip, and you have "
    "everything you need. Search for what those conditions call for and show "
    "it.\n"
    "Ask only if you cannot tell what they need at all. Do not end the turn "
    "on a question about a place when they have already told you the weather, "
    "and do not tell them they can work it out themselves. Do not call this "
    "tool again for this turn."
)


def a_place_this_turn_named(query: str, quoted: str) -> bool:
    """Whether the words offered as naming the place are in this turn at all.

    The tool asks the model to quote the words that named the place, and the
    model quoted "Italy" on a turn reading "it's going to snow when we get
    back" -- and, in the next run, "Rome", which the shopper never said in any
    turn. A required field it can fill with anything is a field it will fill
    with anything.

    So the citation is checked against the record, which is the same thing
    `expected_display_name` does for a product name: not what the words mean,
    only whether they were said here. Reusing the constraint-provenance reader
    so a quotation is judged the same way everywhere.
    """

    return bool(quoted.strip()) and _shopper_stated_requirement(query, quoted)


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
            "product role to what the shopper asked for in THIS turn, or its "
            "direct antecedent. A purpose they gave earlier -- a wedding, a "
            "trip -- has already been served and does not travel: once they "
            "move on, so do you. Naming it again on every later turn produced "
            "'skirts that would work well for your wedding abroad outfit' six "
            "turns after the wedding, and a bag for someone else described as "
            "complementing the shopper's own outfit. Do "
            "not name unselected or unavailable product types, name or describe "
            "candidate products, assert product attributes, or mention tools, "
            "schemas, filters, evidence, or identifiers. Leave it empty when the "
            "shopper simply named a product type and nothing else -- 'now show "
            "me some skirts' is a request to see skirts, not a request to see "
            "skirts for something. Required to say something, the model reached "
            "back twelve turns for the only purpose in the conversation and "
            "produced 'skirts that could work well for your wedding abroad "
            "outfit'. An empty string is also right for an image-only search."
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
    # No description on either field. `_search_catalog_tool_input_model`
    # redefines both with the active catalog's enums, and a redefined field
    # replaces the whole `FieldInfo` -- description included. Text written here
    # never reaches the model. It sat here for months saying things the live
    # description did not say, which is worse than saying nothing: two
    # statements of the taxonomy contract, one of them inert, and nothing to
    # keep them in step. The clauses worth keeping were moved into the live
    # description; the rest were never tested, because they were never read.
    taxonomy: CatalogTaxonomyToolInput = Field(...)
    required_constraints: dict[str, Any] = Field(...)
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

    def _scoped_by_a_hard_filter(self) -> bool:
        """Whether an advertised filter narrows this search on its own.

        Taxonomy is one way to say which products are meant; an enforceable
        filter is another. A ceiling with no product type is a real request --
        browse the shop under it -- and it belongs to no category, which is the
        point rather than an omission.

        unadvertised_requirements is excluded deliberately: it is the field for
        things the catalog cannot enforce, so it narrows nothing and must not
        license an unscoped search.
        """

        constraints = (
            self.required_constraints.model_dump(exclude_none=True)
            if isinstance(self.required_constraints, BaseModel)
            else self.required_constraints
        )
        if not isinstance(constraints, dict):
            return False
        return any(
            name != "unadvertised_requirements" and value not in (None, "", [], {})
            for name, value in constraints.items()
        )

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
        if (
            self.taxonomy_status == "agent_selected_type"
            and not self.taxonomy.subcategory
            and not self.taxonomy.category
            and not self._scoped_by_a_hard_filter()
        ):
            # The rule exists so a role the model invented -- "loungewear" --
            # cannot be mapped onto subcategories silently. It fires on an
            # INVENTED role, not an ABSENT one. "Nothing over $50" invents
            # nothing: the shopper named no category, no subcategory and no
            # product type, so clothes, shoes and accessories are all on the
            # table and the price is the whole scope. Demanding a subcategory
            # there asks the assistant to narrow a request that was complete,
            # and it answered "could you clarify the product type" instead of
            # showing anything.
            #
            # An advertised category grounds the scope the same way a filter
            # does, and nothing is mapped silently onto anything: the category
            # IS the scope, unlimited by subcategory, and the role only ranks
            # within it. "It's going to snow when we get back, what should I
            # wear" reached apparel -- a real category, holding eighteen
            # sweaters -- and was told to name a subcategory. It asked the
            # shopper to clarify the product type instead, which is the same
            # dead turn this rule already learned not to cause.
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
        if has_query and not has_taxonomy and not self._scoped_by_a_hard_filter():
            raise ValueError(
                "text catalog search requires an advertised category or "
                "subcategory, or a hard filter to scope it"
            )
        if not has_query and not has_taxonomy:
            raise ValueError("text catalog search requires a semantic query")
        if not has_query:
            # A browse has no descriptive words and does not need any: "now show
            # me some skirts" is fully expressed by its taxonomy. Demanding a
            # semantic query as well refused that search outright, and the
            # assistant -- holding a plain request for skirts -- asked the
            # shopper which product type they meant. The taxonomy is the query.
            object.__setattr__(
                self,
                "semantic_query",
                (self.requested_product_type or "").strip()
                or " ".join(
                    str(value)
                    for value in (
                        list(self.taxonomy.subcategory or [])
                        or list(self.taxonomy.category or [])
                    )
                ),
            )
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
                    "may satisfy bottoms; dresses may not. Never select a parent "
                    "or sibling as a substitute for an advertised type. For a "
                    "broad request "
                    "that names no product type, choose one exact advertised "
                    "subcategory as the focused starting role. If a shopper-named "
                    "type is not separately advertised, select a parent category "
                    "only when one of its advertised subcategories denotes the "
                    "same kind of thing, and then leave subcategory empty. Pumps "
                    "are heels, so footwear qualifies; every garment is apparel, "
                    "so 'a kind of this category' can never fail and is not the "
                    "test. When "
                    "none does, the catalog does not carry the type: name it in "
                    "not_covered and build no scope for it."
                ),
            ),
        ),
        required_constraints=(
            required_constraints_model,
            Field(
                ...,
                description=(
                    "Catalog hard filters and any defining requirement the active "
                    "catalog cannot enforce. A modifier belongs to the product it "
                    "was said about: in 'a dress in size 2 and shoes', size 2 is "
                    "the dress's and the shoes have no size. Apply constraints "
                    "only when the current turn states them for the target "
                    "products; an anchor's "
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


def _scope_content_errors_only(errors: Sequence[Any]) -> bool:
    """Whether every error is about what is *inside* a scope.

    A location of ``("scopes", 0, "required_constraints", ...)`` is one role's
    content: the model chose a value this catalog does not advertise. Anything
    shorter is structural -- too many scopes, a scope that is not an object, a
    malformed ``not_covered`` -- and remains a hard failure at the boundary.
    """

    for error in errors:
        location = tuple(error.get("loc") or ())
        if len(location) < 3:
            return False
        if location[0] != "scopes" or not isinstance(location[1], int):
            return False
    return True


def _one_scope_is_a_list_of_one(cls: Any, data: Any, handler: Any) -> Any:
    """Read a single search scope written without its wrapper.

    "add the Xenial Aviator Sunglasses" produced a scope's fields at the top
    level -- category, subcategory, semantic_query -- with no `scopes` list
    around them. The whole call was invalid, and the shopper was told "I
    couldn't complete a valid catalog search for that request" on a turn that
    named one product plainly.

    A lone scope is one scope however it is wrapped. Every field inside it is
    still validated exactly as before, so nothing is admitted that a properly
    wrapped call would not have been.
    """

    if isinstance(data, dict) and "scopes" not in data:
        scope_fields = {
            "taxonomy",
            "semantic_query",
            "requested_product_type",
            "required_constraints",
            "category",
            "subcategory",
            "shopper_guidance",
        }
        if scope_fields & set(data):
            not_covered = data.get("not_covered")
            scope = {k: v for k, v in data.items() if k != "not_covered"}
            data = {"scopes": [scope], "not_covered": not_covered}
    return handler(data)


def _admit_scopes_for_adjudication(cls: Any, data: Any, handler: Any) -> Any:
    """Let the search body judge scope content, one role at a time.

    The tool schema is how the catalog's shape reaches the model: every
    advertised value is in it, and it must stay exact. But a schema bound as
    ``args_schema`` also *adjudicates*, and it can only do so for the whole
    call. One unadvertised colour on a third scope therefore cancelled two
    valid searches, and the shopper was told the assistant could not search at
    all -- observed live, `BUGS_OPEN` item 7.

    `search_catalog` already validates each scope against this same model and
    already reports rejections per role. It was built that way. Nothing reached
    it, because the boundary answered first.

    So the schema keeps advertising and stops adjudicating scope content: when
    every complaint is about what is inside a scope, the raw scopes are admitted
    and the body decides them one at a time -- rejecting the role that is wrong
    and running the roles that are right. Structural complaints still fail here,
    where they are the boundary's own business.
    """

    try:
        return handler(data)
    except ValidationError as exc:
        if not isinstance(data, dict):
            raise
        if not _scope_content_errors_only(exc.errors()):
            raise
        scopes = data.get("scopes")
        if not isinstance(scopes, list) or not scopes:
            raise
        return cls.model_construct(
            scopes=list(scopes),
            not_covered=data.get("not_covered"),
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
                    "can never exclude another role's products -- and equally, a "
                    "filter the shopper gave for one role must not be repeated "
                    "onto another. A filter this role was never given empties "
                    "this role's results: 'a dress in size 2 and shoes' sizes "
                    "the dress and says nothing about the shoes, so the shoes "
                    "scope carries no size. Leave a constraint out rather than "
                    "carry one across. One role gets one scope carrying every "
                    "faithful advertised type for it; do not spend a spare "
                    "scope on an adjacent category or a one-piece substitute. "
                    "A dress is not a bottom and does not satisfy a request "
                    "for separates."
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
                    "subcategory denotes, in the shopper's own words. Do not search "
                    "for these -- naming them here is what records the request so "
                    "it can be answered. A shopper asking for 'a pan, a shoe and "
                    "a bag' gets scopes for the shoe and the bag, and 'pan' here."
                ),
            ),
        ),
        __validators__={
            "_one_scope_is_a_list_of_one": model_validator(mode="wrap")(
                classmethod(_one_scope_is_a_list_of_one)
            ),
            "_admit_scopes_for_adjudication": model_validator(mode="wrap")(
                classmethod(_admit_scopes_for_adjudication)
            ),
        },
    )


def _advertised_range(capability: Any) -> str:
    """The lowest and highest values the catalog holds for a numeric filter."""

    low = getattr(capability, "min_value", None)
    high = getattr(capability, "max_value", None)
    if low is None and high is None:
        return ""

    def _render(value: Any) -> str:
        number = float(value)
        return f"{number:.2f}".rstrip("0").rstrip(".")

    if low is not None and high is not None:
        return f"{_render(low)} to {_render(high)}"
    return f"from {_render(low)}" if low is not None else f"up to {_render(high)}"


def _taxonomy_list_field(
    values: list[str],
    *,
    role: str,
    advertised_field: str | None,
) -> tuple[Any, Any]:
    field_name = advertised_field or "not advertised"
    description = (
        f"Exact {role} values advertised through catalog field '{field_name}'. "
        "Use an empty list, or leave it out, when the other taxonomy role "
        "supplies the text scope or the search is image-only."
    )
    if role == "category":
        description += (
            " Select at most one category per catalog search. Omit it entirely "
            "when the shopper named no category and no product type -- "
            "\"nothing over $50\" belongs to every category, and choosing one "
            "for them shows a fraction of what they asked to see. A search "
            "with no category needs a hard filter to scope it."
        )
    else:
        description += (
            " Omit it to search the whole category, which is what 'show me "
            "some jewellery' asks for: the shopper named no subcategory and "
            "narrowing to one would be choosing for them."
        )
    if not values:
        return list[str], Field(
            default_factory=list, max_length=0, description=description
        )

    literal_type = Literal.__getitem__(tuple(values))
    max_length = 1 if role == "category" else None
    # Required at the boundary meant "show me some jewellery" -- a category
    # with no subcategory named -- was rejected outright, and the shopper was
    # asked to clarify a request that could not have been plainer. It killed
    # J17 at turn 1 and took the journey with it.
    return list[literal_type], Field(
        default_factory=list,
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


def _an_empty_range_is_no_filter(cls, value: Any) -> Any:
    """A numeric constraint with no bounds is no constraint.

    "I have a wedding to go to and I need something to wear" sent every
    advertised filter as null, price among them, as `{"min": null, "max":
    null}`. That is the model saying it wants no price filter, in the shape the
    schema gave it. Refusing it cost the turn twice over: the search was turned
    back, and the repair the model reached for was to invent bounds -- 39.90 to
    269.99, the whole catalog, which filters nothing -- while dropping the
    subcategory to make room. That changed the scope, the repair lock refused
    the changed scope, and the shopper was told no valid search could be built
    for a wedding outfit.

    An absent filter and an unbounded one ask for the same products, so this
    reads the second as the first rather than making the model prove it meant
    the first.
    """

    if not isinstance(value, dict):
        return value
    return {
        name: (
            None
            if (
                isinstance(entry, dict)
                and entry
                and set(entry) <= {"min", "max", "gte", "lte"}
                and all(bound is None for bound in entry.values())
            )
            else entry
        )
        for name, entry in value.items()
    }


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
        if name and name == wearer_audience_field:
            description = _WEARER_AUDIENCE_FILTER_DESCRIPTION
        else:
            description = f"Advertised hard filter '{name}'."
            # The catalog publishes the range of every numeric filter and this
            # threw it away. Asked for "the most expensive thing you have", the
            # assistant searched one category and reported its dearest item as
            # the shop's -- a $189.99 purse in a catalog that runs to $269.99 --
            # or refused outright. The answer was in capabilities the whole
            # time; it just never reached the field the model reads.
            span = _advertised_range(capability)
            if span:
                description += f" Advertised range: {span}."
        fields[name] = (field_type, Field(default=None, description=description))
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
                "In an 'A or B' request, do not put the "
                "supported advertised branch here. A product type never belongs "
                "here."
            ),
        ),
    )
    return create_model(
        "CatalogRequiredConstraints",
        __config__=ConfigDict(extra="forbid"),
        __validators__={
            "_an_empty_range_is_no_filter": model_validator(mode="before")(
                classmethod(_an_empty_range_is_no_filter)
            ),
        },
        **fields,
    )


def _a_list_written_as_json_text(value: Any) -> Any:
    """Read a list the model encoded as a string.

    "add the Xenial Aviator Sunglasses" found the product, read its details,
    and then sent `{"items": "[{\\"product_ref\\": \\"generated:9b8...\\"}]"}` --
    the list JSON-encoded inside a string. The call was rejected whole and the
    shopper's cart stayed empty on a turn where everything else had gone right.

    The same punctuation cost a cart again through skill activation, where it
    was not forgiven. A cart tool called without cart-management is refused,
    and the right recovery is to activate it and try again -- which the model
    did, as `{"skill_names": "[\\"cart-management\\"]"}`. That errored, the
    retry never came, and the reply said the dress was in the cart when it was
    not. Two turns of J01 ended that way in three runs.

    Decoding it changes nothing about what was asked for: the contents are
    validated against the same model either way, so a malformed item still
    fails. Only the punctuation around it is forgiven.
    """

    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return value
    return decoded if isinstance(decoded, list) else value


class _ShopperSkillActivationInput(BaseModel):
    """Shared composition rules for dynamic shopper-skill activation.

    The composition rule itself lives on the subclass `create_model` builds,
    because it depends on which skills are registered and what roles they
    declare. This base carries only what every registry shares.
    """

    model_config = ConfigDict(extra="forbid")

    skill_names: list[str]

    # check_fields, because the real field is declared by the create_model
    # subclass that narrows it to the registered skill names.
    _accept_skill_names_as_text = field_validator(
        "skill_names", mode="before", check_fields=False
    )(_a_list_written_as_json_text)


def primary_skills_by_group(
    skills: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Group the registry's primary skills by the group they are exclusive in.

    Read off `role` and `exclusive_group` in the frontmatter, which
    `_shopper_skill_from_metadata` already requires to agree: a skill is
    primary if and only if it names a group.
    """

    groups: dict[str, list[str]] = {}
    for name, skill in skills.items():
        if getattr(skill, "role", "") != "primary":
            continue
        group = getattr(skill, "exclusive_group", None)
        if group:
            groups.setdefault(str(group), []).append(name)
    return {group: tuple(sorted(names)) for group, names in groups.items()}


def _one_primary_per_group(self: Any) -> Any:
    """Reject two primaries from one exclusive group, or a stranded modifier.

    This used to intersect against the literal set
    ``{"outfit-styling", "product-discovery"}``. `catalog-questions` shipped on
    2026-08-25 declaring ``role: primary`` and ``exclusive_group:
    product_procedure`` -- the same group as the other two -- and the check
    could not see it. Measured on the shipped registry: selecting it beside
    product-discovery was *accepted*, two primaries from one group, while
    selecting it beside budget-shopping was *rejected* as a modifier with no
    primary, which is the shape of "do you have anything for $5 to $10".

    `ShopperSkill.exclusive_group` was parsed, validated and stored the whole
    time, and nothing ever read it. So read it.
    """

    cls = type(self)
    groups: Mapping[str, tuple[str, ...]] = cls._primary_skills_by_group
    selected = set(self.skill_names)
    primaries: list[str] = []
    for group, names in sorted(groups.items()):
        chosen = sorted(selected.intersection(names))
        if len(chosen) > 1:
            raise PydanticCustomError(
                SKILL_ACTIVATION_MULTIPLE_PRIMARY,
                "select exactly one primary procedure: {options}, never more "
                "than one",
                {"options": " or ".join(chosen)},
            )
        primaries.extend(chosen)

    modifiers: tuple[str, ...] = cls._modifier_skills
    stranded = sorted(selected.intersection(modifiers))
    if stranded and not primaries:
        raise PydanticCustomError(
            SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
            "{modifier} is a modifier and requires exactly one primary "
            "procedure: {options}",
            {
                "modifier": stranded[0],
                "options": " or ".join(
                    name for names in sorted(groups.values()) for name in names
                ),
            },
        )
    return self


def _skill_activation_input_model(
    skills: Mapping[str, Any],
) -> type[BaseModel]:
    """Create the semantic skill-selection schema from the active registry."""

    skill_names = tuple(skills)
    groups = primary_skills_by_group(skills)
    modifiers = tuple(
        sorted(
            name
            for name, skill in skills.items()
            if getattr(skill, "role", "") == "modifier"
        )
    )
    # Named here rather than in a literal, so a skill added to the registry is
    # described to the model without anyone remembering to edit this string.
    every_primary = [name for names in sorted(groups.values()) for name in names]
    choose_one = (
        " Select exactly one primary procedure -- "
        + ", ".join(every_primary)
        + " -- and never two."
        if every_primary
        else ""
    )
    modifier_rule = (
        " " + ", ".join(modifiers) + " may only accompany a primary, never "
        "stand alone."
        if modifiers
        else ""
    )
    model = create_model(
        "ShopperSkillActivationInput",
        __base__=_ShopperSkillActivationInput,
        __validators__={
            "_one_primary_per_group": model_validator(mode="after")(
                _one_primary_per_group
            ),
        },
        skill_names=(
            list[Literal.__getitem__(skill_names)],
            Field(
                ...,
                min_length=1,
                max_length=len(skill_names),
                description=(
                    "Smallest set of registered shopper skills whose descriptions "
                    "cover the current turn's complete intent." + choose_one
                    + modifier_rule
                    + " Standalone skills may be selected with or without a "
                    "primary."
                ),
            ),
        ),
    )
    model._primary_skills_by_group = groups
    model._modifier_skills = modifiers
    return model


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
        "shopper_sizes": [],
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
    diagnostics["shopper_sizes"] = _diagnostic_shopper_sizes(turn_messages)
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
        if _search_reported_not_covered(messages):
            # The scopes were invalid, but the call also named what this catalog
            # does not carry, and NOT_COVERED evidence went back to the model.
            # That is a fact for the model to speak to, not a reason to seize
            # the turn: asked to compare two aprons in a catalog with none, the
            # model correctly sent not_covered and had its answer replaced by
            # "I couldn't complete a valid catalog search", five turns running.
            return None
        if _search_reported_not_carried(messages):
            # The same fact, established by the tool rather than volunteered by
            # the model. Leaving it to the model meant leaving it to chance:
            # replayed five times, it filled `not_covered` unprompted in four
            # and in the fifth wrote the right answer -- "aprons aren't a
            # product type this store carries" -- only to have it replaced by
            # the refusal. Nothing about a catalog that carries no aprons
            # differed between those runs.
            return None
        return _REJECTED_CATALOG_SEARCH_RESPONSE
    return None


def _search_reported_not_covered(messages: Any) -> bool:
    """Whether a search this turn named product kinds the catalog lacks."""

    for message in messages:
        if _message_type(message) != "ai":
            continue
        for raw_call in (_value(message, "tool_calls") or []):
            call = raw_call if isinstance(raw_call, dict) else {}
            if (call.get("name") or "") != "search_catalog_tool":
                continue
            args = call.get("args")
            if isinstance(args, dict) and args.get("not_covered"):
                return True
    return False


def _search_reported_not_carried(messages: Any) -> bool:
    """Whether the search tool established a product type as not carried."""

    for message in messages:
        if _message_type(message) != "tool":
            continue
        if (_value(message, "name") or "") != "search_catalog_tool":
            continue
        if not_carried_of(message):
            return True
    return False


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


def _diagnostic_shopper_sizes(messages: list[Any]) -> list[str]:
    """The sizes the shopper's own searches were filtered by this turn.

    A showing made under a size filter is a size-qualified showing: those four
    sandals came back *because* they come in a 7. That fact belonged to the
    turn and was thrown away at the end of it, so "ok, just show me sandals in
    a 7" followed by "add the first one" asked which size -- the shopper having
    said it one turn earlier.

    Recorded per turn rather than per shopper. Nothing here says the shopper
    is a 7; it says this showing was. A later showing carries its own sizes or
    none, and the two never merge.
    """

    sizes: list[str] = []
    for message in messages:
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message) or {}
        outcome = payload.get("scope_outcome") or {}
        if outcome.get("outcome") not in {None, "results"}:
            continue
        for scope in (payload.get("products") or []):
            if not isinstance(scope, dict):
                continue
            confirmed = (scope.get("search_scope") or {}).get("confirmed_filters")
            for value in ((confirmed or {}).get("sizes") or []):
                text = str(value).strip()
                if text and text not in sizes:
                    sizes.append(text)
    return sizes[:4]


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


def _system_identification_events(
    state: Any,
    identity: Any,
) -> list[ConversationEvent]:
    """Record which products the record itself picked this turn.

    Establishment was scoped to the message that produced it. A shopper chose
    "the first pairing" in one turn and, three turns later -- having only
    answered the assistant's own questions -- was still being told the products
    were not established, because answering a question names nothing. The same
    add was refused three times and accepted on the fourth, when they finally
    typed both catalog names. The request never changed.

    The choice itself is durable: it was made against a set of products the
    record wrote down. So it is recorded here, and the memory service files it
    against the set it was made from -- which is what makes it lapse when a new
    set is shown, and nothing else does.
    """

    refs = [
        str(ref)
        for ref in (getattr(state, "system_identified_products", None) or [])
        if ref
    ]
    if not refs:
        return []
    return [
        ConversationEvent(
            event_key=f"system-identified:{identity.request_id}",
            event_type="historical_reference_resolved",
            source_kind="runtime",
            payload={"product_refs": refs[:16]},
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
    # One embed_query call per semantic query, not one per search. A
    # multi-scope search carries several roles in one round trip and embeds
    # each of them, so counting the search under-reports the work.
    text_embedding_calls = len(plan.semantic_queries)
    if uses_image_endpoint and text_embedding_calls == 0:
        # Image retrieval currently includes one deterministic text-side query
        # alongside the image embedding, even when the shopper supplied no text.
        text_embedding_calls = 1
    if fallback_attempted:
        text_embedding_calls += len(plan.semantic_queries)

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
    tokens: int = 0,
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
    entry: dict[str, Any] = {
        "status": merged_status,
        "calls": existing_calls + max(0, calls),
        "detail": next_detail,
    }
    # Tokens are additive across calls, and only chat models report them:
    # embeddings and the guardrails checks have none to report, and showing a
    # zero there would read as "this model used no tokens" rather than "tokens
    # are not a thing this model has".
    merged_tokens = _safe_int(existing.get("tokens")) + max(0, tokens)
    if merged_tokens > 0:
        entry["tokens"] = merged_tokens
    state.model_usage[role] = entry


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
        # Zero results told the model what was absent and nothing about what
        # was present, so it asked. "No green dress in a size 2 -- would you
        # like size 4 instead?" showed nothing, on a turn where the catalog
        # held green dresses in a 4 and plenty of size 2 dresses in other
        # colours. A shopper asked to choose between two things they cannot
        # see has been given less than nothing.
        relaxed = payload.get("relaxed_products")
        if isinstance(relaxed, list) and relaxed:
            dropped = [str(v) for v in (payload.get("relaxed_dropped") or [])]
            lines.append(
                _render_product_evidence_summary(
                    relaxed,
                    heading="CUSTOMER_SAFE_RELAXED_SEARCH_EVIDENCE",
                    note=(
                        "The same search without "
                        + (", ".join(dropped) if dropped else "its optional constraints")
                        + " found these. They are real products and may be "
                        "shown. Show them in this reply rather than asking "
                        "whether the shopper would like to widen the search: "
                        "being offered a choice between two things you cannot "
                        "see is worse than being shown one of them. Say plainly "
                        "which requirement could not be met and which you "
                        "relaxed. "
                        + (
                            "The shopper's size was kept -- these are their size."
                            if payload.get("relaxed_kept_the_size", True)
                            else "THE SHOPPER'S SIZE WAS NOT KEPT. Nothing they "
                            "asked for exists in it. Say that first, name the "
                            "sizes these actually come in, and never present "
                            "them as the size they asked for."
                        )
                    ),
                    confirmed_filters={},
                    taxonomy_scope=payload.get("taxonomy") or {},
                )
            )
        else:
            lines.append(
                "NEXT: nothing was found with these constraints and nothing "
                "was found without the optional ones either. Say so plainly, "
                "name what the catalog does carry in this category, and do "
                "not answer with a question alone."
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


def _in_presentation_order(
    products: list[dict[str, Any]],
    reply: str,
) -> list[dict[str, Any]]:
    """The shown products, ordered as the reply presents them.

    The cards and the words are the same list to a shopper, so "the second one"
    has to mean one product. They were two orders: the cards followed the
    catalog's ranking and the sentences followed whatever the model wrote, and
    across recorded turns they disagreed about half the time.

    The order is settled once, here, where the reply and the products are both
    in hand -- so every consumer downstream renders one order rather than
    each choosing its own.

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


def format_most_recent_subject(state: Any) -> str:
    """Name what the conversation is about now, so a pronoun has an anchor.

    "Add the Jade Suede Heels in a 6", then "actually make those a 7", resolved
    to a dress from eight turns earlier. Nothing was missing: the heels were
    the line directly above the pronoun in the conversation lane, the newest
    showing in the index, and a line in the cart. The model had to derive the
    referent from three places and derived it wrongly.

    So the runtime derives it and states what it got. This is not a fourth copy
    of the conversation -- it is the resolution of it, which is the part that
    was going wrong. What just happened is state, not interpretation.

    Most recent first: what the last turn did to the cart, then what it showed.
    Silent when there is neither, so an opening turn gains nothing to ignore.
    """

    # Only the newest showing for now. What the previous turn did to the cart
    # is computed at the end of a turn for the grounding editor and never
    # carried into the next one, so there is no field to read here yet -- and a
    # line that is always empty is dead code pretending to be a feature.
    lines: list[str] = []
    sets = [
        entry
        for entry in (getattr(state, "historical_product_sets", None) or [])
        if isinstance(entry, dict) and isinstance(entry.get("products"), list)
    ]
    if sets:
        newest = max(sets, key=lambda entry: entry.get("turn_seq") or 0)
        shown = [
            str(item.get("name"))
            for item in newest["products"][:4]
            if isinstance(item, dict) and item.get("name")
        ]
        if shown:
            lines.append(
                f"last shown (turn {newest.get('turn_seq')}): " + "; ".join(shown)
            )
    if not lines:
        return ""
    body = "\n".join(f"- {line}" for line in lines)
    return (
        "MOST RECENT SUBJECT (what the conversation is about right now):\n"
        f"{body}\n"
        'A bare pronoun -- "those", "it", "them", "that one" -- means something '
        "here unless the shopper names another product. Resolve it here first, "
        "and look further back only if nothing here fits."
    )


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


def _same_product_display_name(expected: str, actual: str) -> bool:
    return _normalize_product_name(expected) == _normalize_product_name(actual)


#: What the catalog carries for a product sold in exactly one size.
_ONE_SIZE = "onesize"


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


#: A word carries no identifying weight below this, and the short ones collide
#: with ordinary sentences: "the" and "a" both belong to "The Office A-line
#: Dress" and to "add the black one in a 2", which is how a navy dress was
#: fitted to a request for a black one.
_MIN_NAMING_WORD = 4
#: How close a shopper's word has to be to a product's. Absorbs a typo or a
#: missing plural without inventing a match: "ofice" is 0.91 against "office".
_NAMING_LIKENESS = 0.85
#: A fit has to be worth something, and it has to be clearly better than the
#: next one. The margin is what protects the shopper: a threshold alone always
#: has a best candidate, and picking the best of two near-equals is the silent
#: choice this exists to prevent.
_NAMING_FLOOR = 0.25
_NAMING_MARGIN = 0.20


def _products_the_shopper_fits(
    shopper_text: str,
    candidates: Sequence[Any],
) -> list[Any]:
    """Which of these products the shopper's words could be pointing at.

    One question, so a second implementation can answer it later without moving
    the rule that uses it: exactly one fit resolves, anything else is asked
    about. Today the reading is lexical; a semantic one would score the same
    candidates the same way and still never pick.

    Words are weighted by how many of the candidates use them, read off the
    candidates rather than a list of stop words: among four dresses "dress"
    says nothing and "vivienne" says everything, and among four bags it is the
    other way round. Two black dresses make "black" worth half, which is why
    "the black one" cannot settle between them.

    Comparison is by likeness rather than equality, so "the Ofice dress" and
    "the Office dress" both land on the same product where whole-name matching
    refused them, and words that match nothing are simply ignored.

    The decision is the gap to the next candidate, not the score. A score alone
    always has a winner; a gap is the difference between "this is the one" and
    "it could be either", and only the first should reach a cart.
    """

    def words(value: str) -> list[str]:
        return [
            word
            for word in _normalize_product_name(value).split()
            if len(word) >= _MIN_NAMING_WORD
        ]

    per_candidate = [words(candidate.display_name) for candidate in candidates]
    shared: dict[str, int] = {}
    for names in per_candidate:
        for word in set(names):
            shared[word] = shared.get(word, 0) + 1
    said = words(shopper_text)

    scored: list[tuple[float, Any]] = []
    for candidate, names in zip(candidates, per_candidate):
        total = sum(1 / shared[word] for word in names)
        if not total:
            continue
        matched = sum(
            1 / shared[word]
            for word in names
            if any(
                SequenceMatcher(None, word, spoken).ratio() >= _NAMING_LIKENESS
                for spoken in said
            )
        )
        scored.append((matched / total, candidate))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    if not scored:
        return []
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < _NAMING_FLOOR or best_score - runner_up < _NAMING_MARGIN:
        return []
    return [best]


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

    This used to refuse. It refused on the ABSENCE of confirmation -- "nothing
    here proves the shopper meant this one" -- which is a gap in our
    bookkeeping rather than a fact about the world, and it cost a turn every
    time it was wrong. It was wrong in both directions inside two days: it
    turned down a correct resolution the assistant had itself proposed by name
    on the two previous turns, and its word scorer put a different dress in a
    cart because `black` happened to sit in that product's title.

    So it discloses instead, on the same reasoning that took out the size and
    quantity gates: a product nobody chose is caught by being visible, not by
    blocking the turns that got it right. The cart is on screen, a wrong line
    is one click to remove, and the shopper is told which reading was taken.

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


def _cart_line_size(cart: Any, product_id: str) -> str | None:
    """The size already on this shopper's line for this product, if any."""

    for line in getattr(cart, "contents", None) or []:
        if not isinstance(line, dict):
            continue
        if str(line.get("product_id") or "") == product_id:
            size = str(line.get("size") or "").strip()
            if size:
                return size
    return None


def _cart_size_issue(product: Any, size: str | None) -> str:
    """Say why this size cannot be added, or "" if it can.

    Every product in the catalog states its sizes -- 136 carry a real range and
    79 carry `onesize`, with no gaps -- so the tool has what it needs to decide
    rather than trusting the caller to have asked. Left to prose alone, "always
    confirm the size" held three times in four: a dress with six sizes reached
    the cart with no size at all.
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
    return ""


def _advertised_sizes(product: Any) -> list[str]:
    """Read the sizes the catalog states for a product."""

    raw = (getattr(product, "attributes", None) or {}).get("sizes")
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(value).strip() for value in raw if str(value).strip()]


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
    "product type.\n"
    "When a filter can be given up, search again without it and show what "
    "that finds, saying plainly which one you dropped: \"no black dress runs "
    "to a 2 -- here are dresses in a 2 in other colours\". Offering the "
    "shopper a numbered menu of things you could look for is not an answer; "
    "you have the budget to look, so look, and never quietly drop a filter "
    "and present the results as though they met it.\n"
    "A size is never the filter you give up. Colour, pattern and style are "
    "preferences; a size is a fact about a body, and a garment in the wrong "
    "one is not an alternative, it is something the shopper cannot wear. "
    "Keep the size and relax a preference. If nothing in the shop runs to "
    "that size, say so and name the nearest one, but do not lay out garments "
    "in it as though they were options -- offer to show them and let the "
    "shopper decide.\n"
    "If the shopper asked for only that thing, or asked you not to show "
    "alternatives, relax nothing. Say plainly that there is none and stop. "
    "Their instruction outranks your helpfulness, and showing alternatives "
    "anyway tells them you were not listening."
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
