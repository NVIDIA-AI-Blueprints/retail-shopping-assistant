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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_products import ProductEvidence

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

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
from shared.commerce_contracts import (
    CatalogCapabilities,
    CommerceError,
    ProductDetail,
    ProductSummary,
)

from .agenttypes import Cart, State
from .catalog_capabilities import (
    effective_filter_capabilities,
)
from .conversation_memory import (
    ConversationEvent,
    FinalTurnStatus,
)
from .message_shape import (
    _content_to_text,
    _current_turn_messages,
    _message_type,
    _result_messages,
    _value,
)
from .response_format import (
    _format_cart,
    _format_detail_value,
    _format_product_detail_record,
    _format_product_record,
    _format_product_refs,
)
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
    SKILL_ACTIVATION_MULTIPLE_PRIMARY,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_TOOL_NOT_GRANTED,
)
from .tool_evidence import (
    evidence_of,
)
from .tool_loop_control import (
    SEARCH_BUDGET_EXHAUSTED_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
    STOP_TOOL_USE_PREFIX,
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


_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS = 1.0


_UNSUPPORTED_SEARCH_MODE_MESSAGE = (
    "The requested search mode is not available for the active catalog. "
    "Ask the shopper to use an advertised mode."
)


_NO_DIRECT_TAXONOMY_RESPONSE = (
    "The catalog doesn't advertise a product type that directly matches this "
    "request. Would you like me to search a different advertised product type?"
)


_PRODUCT_NAME_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to", "with"})


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


WEATHER_PLACE_NOT_STATED = (
    "WEATHER_PLACE_NOT_STATED: no forecast -- the words you quoted as naming "
    "the place are not in anything the shopper has said, in this turn or any "
    "earlier one.\n"
    "If they did name a place, here or on an earlier turn of this same trip, "
    "quote their actual words and call again.\n"
    "Carry on and answer them either way -- a forecast was not the whole "
    "request. If they said what the conditions will be -- \"it's going to "
    "snow when we get back\" -- that is the answer to the weather question, "
    "they are the authority on their own trip, and you have everything you "
    "need. Search for what those conditions call for and show it.\n"
    "Ask only if you cannot tell what they need at all, and then ask for the "
    "one thing you are missing. Do not end the turn on a question about a "
    "place when they have already told you the weather, and do not tell them "
    "they can work it out themselves. Above all, do not describe conditions "
    "you did not fetch: typical, seasonal, usually and this time of year are "
    "not forecasts, and a reply that says the weather is unavailable and then "
    "supplies some is worse than either half alone."
)


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
            "the active skill before search results are known. It is read out "
            "on a clarification or a degraded turn, so the shopper's own words "
            "copied in here are read back to the person who just said them. "
            "Connect this "
            "product role to what the shopper asked for in THIS turn, or its "
            "direct antecedent. A purpose they gave earlier -- a wedding, a "
            "trip -- has already been served and does not travel: once they "
            "move on, so do you. Naming it again on every later turn produced "
            "'skirts that would work well for your wedding abroad outfit' six "
            "turns after the wedding, and a bag for someone else described as "
            "complementing the shopper's own outfit. It is about this role, "
            "never a running tally of everything shown so far -- 'skirts to "
            "complete your look with bracelets, sunglasses, tote bags and "
            "heels' invents an outfit out of a browse. Do "
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
    def text_search_has_taxonomy_scope(self) -> SearchCatalogToolInput:
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


class _CatalogNumberConstraint(BaseModel):
    """Inclusive numeric range accepted by catalog hard filters."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def has_a_bound(self) -> _CatalogNumberConstraint:
        if self.min is None and self.max is None:
            raise ValueError("numeric constraint requires min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("numeric constraint min cannot exceed max")
        return self


#: Built schemas, kept for the life of the process.
#:
#: Every input to these builders is fixed once the service is running: the
#: capability contract is cached by CatalogCapabilitiesClient on its first
#: success, and the rest are configuration. So the schema is identical on every
#: turn -- verified byte-for-byte -- and rebuilding it was 15ms per turn of
#: producing the same object, on the event loop that also serves the turn.
#:
#: Keyed on the contract's *content*, not its catalog_id. Two contracts can
#: carry the same id and differ -- which is not hypothetical: thirty-two tests
#: failed on an id-based key, each having built a catalog named like the shipped
#: one with different fields, and each being served the other's schema. In
#: production one process sees one catalog and the distinction never arises;
#: the point of hashing is that a cache should not be correct only because of
#: how it happens to be used.
#:
#: The hash costs 1.9ms against 7ms to rebuild, so it pays for itself, and it
#: needs no assumption about when a catalog may change.
_SCHEMA_CACHE: dict[tuple[Any, ...], Any] = {}


def _capabilities_identity(capabilities: CatalogCapabilities) -> str:
    """A stable fingerprint of everything a schema is built from."""

    return hashlib.blake2b(
        capabilities.model_dump_json().encode("utf-8"), digest_size=16
    ).hexdigest()


def _cached_schema(key: tuple[Any, ...], build: Any) -> Any:
    """Return a built schema, building it once per distinct key."""

    if key not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[key] = build()
    return _SCHEMA_CACHE[key]


def clear_schema_cache() -> None:
    """Drop every built schema. For tests that construct several catalogs."""

    _SCHEMA_CACHE.clear()


def _search_catalog_tool_input_model(
    capabilities: CatalogCapabilities,
    *,
    validate_scope: bool = True,
    wearer_audience_field: str = "",
) -> type[SearchCatalogToolArguments]:
    """Return the tool schema for this catalog, building it at most once.

    The enums come from the catalog, and the catalog does not change under a
    running process -- CatalogCapabilitiesClient caches its first successful
    contract and never refetches, so a schema built from it can be kept for as
    long as that contract is. Rebuilding it was 8ms of producing an identical
    object on every turn, on the event loop that also has to serve the turn.
    """

    return _cached_schema(
        ("search_tool", _capabilities_identity(capabilities), validate_scope, wearer_audience_field),
        lambda: _build_search_catalog_tool_input_model(
            capabilities,
            validate_scope=validate_scope,
            wearer_audience_field=wearer_audience_field,
        ),
    )


def _build_search_catalog_tool_input_model(
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
                    # The size-2-and-shoes example lives on `scopes`, which is
                    # where cross-role ownership belongs, and was stated here
                    # too. Both arrived on 2026-08-20 (#191). Measured across
                    # the run archive on J01 turn 12 -- "my husband is coming
                    # too, he needs sunglasses" -- the audience filter was
                    # correct 3 times in 5 on the build before that day, and
                    # once in the ~105 runs after it. Nothing was removed in
                    # between and the audience rule itself has not changed
                    # since 2026-08-06; 140 characters of size-and-role worked
                    # example were added to the front of the object where the
                    # audience value gets chosen. Stated once, on the field
                    # that owns it.
                    #
                    # The same lesson, applied again to what was still here.
                    # Four rules about what belongs in
                    # `unadvertised_requirements` -- the water-resistant bags
                    # example, the eleven-adjective list, the product-type ban,
                    # and the season-and-weather rule -- were stated on that
                    # field and restated here. Only the destination differed,
                    # and two of them genuinely have two destinations, so those
                    # two keep one compact statement each for this object and
                    # the worked example and the list stay on the field they
                    # are about.
                    "Catalog hard filters and any defining requirement the active "
                    "catalog cannot enforce. When the shopper names a specific "
                    "product and asks whether it meets some condition -- within a "
                    "budget, in stock, in a colour, on sale -- do not send that "
                    "condition as a filter. You are checking one product, not "
                    "narrowing a category, and a filter can only hide the answer. "
                    "Asked whether a named bracelet came within a $150 ceiling, a "
                    "search filtered to $150 could not return the $169.99 bracelet "
                    "and the shopper was told the shop did not have it. Find the "
                    "product first, then compare. Apply constraints "
                    "only when the current turn states them for the target "
                    "products; an anchor's "
                    "attributes belong in semantic styling context unless the "
                    "shopper explicitly requests the same value. Use only the "
                    "advertised properties "
                    "in this object; put unsupported must-haves in "
                    "unadvertised_requirements instead of weakening them, under "
                    "the rules stated on that field. Broad season, weather, "
                    "occasion or subjective style context is not a hard filter "
                    "and does not belong in this object unless the shopper "
                    "directly requires a product attribute; neither do "
                    "recommendation adjectives, which are always semantic "
                    "ranking preferences. Before calling the tool, "
                    "compare "
                    "every target-product modifier with this advertised schema "
                    "and include every exact matching filter value. Use an empty "
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
    """Return the scoped-search schema for this catalog, built at most once."""

    return _cached_schema(
        ("search_scopes", _capabilities_identity(capabilities), max_scopes, wearer_audience_field),
        lambda: _build_search_catalog_scopes_input_model(
            capabilities,
            max_scopes=max_scopes,
            wearer_audience_field=wearer_audience_field,
        ),
    )


def _build_search_catalog_scopes_input_model(
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
    # Two halves, learned separately and measured apart.
    #
    # The MEANING of the values comes from Catalog capabilities, not from a
    # table of cases written here. Enumerating them scored a man 0/10 to 11/18
    # depending on wording, and "shades for hubby" never once: the model had
    # to infer from the bare string `womens` whether it was a wearability
    # constraint or a department, and men do shop womenswear. Published as a
    # fact, hubby went to 4/4 and dad to 4/4.
    #
    # The TRIGGER and the ORDER have to stay here. A version that carried only
    # the meanings sent no filter at all for a wife or a sister, 0/10 -- it
    # knew what the values meant and never reached for them. So: say plainly
    # when to send it, and keep the covers-everyone value unconditional so
    # there is no judgment to lose.
    #
    # The trigger was a list of thirteen person-words, and it failed on the
    # commonest word of all. J01 turn 12 said "my husband is coming too" and
    # sent no filter, because the list held "hubby" and "my man" but not
    # "husband" -- it had been measured on "hubby" and the ordinary word was
    # never tried. Turn 13 then said "he also wants a bag" and sent none
    # either, because a pronoun names nobody by that rule, and a womens floral
    # clutch came back for a man. Turn 12 hid the same failure only because
    # every sunglass this catalog stocks happens to suit all genders.
    #
    # So the trigger is a rule now and not an enumeration: the person appears
    # in this turn's words, in any form, including a pronoun pointing back.
    # The reverse rule stays -- silence means no filter -- because that is the
    # part that cannot be enumerated, and "now show me some skirts" must not
    # inherit a husband.
    "Who the products are for. Send this whenever this turn's words say who "
    "the shopper is buying for, however casually and in whatever form: a "
    "formal word, an affectionate one, or a pronoun pointing back to someone "
    "already mentioned. \"My husband is coming too\" says it, \"something for "
    "hubby\" says it, and so does \"he also wants a bag\" one turn later -- a "
    "pronoun referring to a person is that person, named. Build the list "
    "in two steps, in this order. First: the value covering all genders is "
    "always in the list, because it suits everyone -- it is never the thing "
    "you leave out. Second: add any other value only when its published "
    "meaning fits the person named. Buying for a woman, the women's value fits "
    "and is added, so the list holds both. Buying for a man or a child, "
    "nothing is added and the covers-everyone value goes alone. Being the "
    "closest value available is not the test and never justifies adding one. "
    "If nothing published suits the named person, such as a child in a "
    "catalog whose values are all adult, send what covers everyone, say so "
    "in the reply, and never substitute what does not suit them. "
    "How the shopper referred to the person is for reading what they said. "
    "It is not an audience to offer back: a reply may name only audiences "
    "this catalog advertises, and must never suggest looking for one it "
    "does not stock. "
    "When nobody is named, omit this filter entirely -- do not send a "
    "covers-everyone value to mean unspecified, which "
    "discards everything stocked for one audience. Only this turn's words "
    "count, and an audience established earlier never carries into a request "
    "that says nothing about who it is for, however obviously that person is "
    "still around: send no filter and let the shopper redirect you. "
    "Enumerating the ways a shopper moves on is hopeless, so the rule is the "
    "reverse -- the person appearing in what they just said is what turns the "
    "filter on. \"Now show me some skirts\" names nobody and gets no filter, "
    "even one turn after a husband was mentioned."
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
            # The catalog publishes what its values mean; render them onto the
            # field the model fills, not only into the capabilities block.
            # Measured: with the meanings in the system prompt alone, "shades
            # for hubby" went 0/4 to 3/4 -- the fact was doing real work -- but
            # a wife or sister collapsed to 1/14, because the binding channel
            # had lost the instruction to the weaker one. The fact belongs
            # where the decision is made.
            meanings = getattr(capability, "value_meanings", None) or {}
            if meanings:
                description += " What each value means, published by this "
                description += "catalog: " + "; ".join(
                    f"{value} -- {text}"
                    for value, text in sorted(meanings.items())
                ) + "."
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
    text = value.strip()
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        decoded = _a_list_whose_last_bracket_never_arrived(text)
        if decoded is None:
            return value
    return decoded if isinstance(decoded, list) else value


def _a_list_whose_last_bracket_never_arrived(text: str) -> Any | None:
    """Read a list missing only its closing bracket, or None if that is not it.

    "do you have that first one in a size 6" resolved correctly -- the right
    product_ref, the right ordinal, the right turn -- and arrived as 301
    characters of JSON with one `]` absent. The call errored, the error said
    nothing the model could act on, and it sent the identical 301 characters
    twenty-two times until the graph's recursion limit ended the turn.

    Only the bracket is supplied, never a brace. A missing `]` means every
    object in the list closed, so nothing is being guessed at. A missing `}`
    would mean an object was cut mid-field, and completing that invents a
    descriptor: a reference that lost its `ordinal` but kept its `category`
    would resolve, quietly, to a different product than the shopper meant.
    Those still fail, which is the outcome they should have.
    """

    if not text.startswith("[") or text.endswith("]"):
        return None
    try:
        return json.loads(text + "]")
    except (TypeError, ValueError):
        return None


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
    for _group, names in sorted(groups.items()):
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
    digest = hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()
    return int(digest[:15], 16)


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


def _business_tool_result_contents(messages: list[Any]) -> list[str]:
    """Return non-activation tool outcomes in graph order."""

    outcomes: list[str] = []
    for message in messages:
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if name == SKILL_ACTIVATION_TOOL_NAME or content.startswith(
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


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None


_NON_ATTRIBUTE_SEARCH_KEYS = frozenset({"catalog_text", "similarity", "taxonomy"})


def _in_presentation_order(
    products: list[dict[str, Any]],
    reply: str,
    groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The shown products, ordered as the reply presents them.

    The cards and the words are the same list to a shopper, so "the second one"
    has to mean one product. They were two orders: the cards followed the
    catalog's ranking and the sentences followed whatever the model wrote, and
    across recorded turns they disagreed about half the time.

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
