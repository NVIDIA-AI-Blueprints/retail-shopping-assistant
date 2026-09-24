"""Building the tool schema the model is shown, from the catalog's capabilities.

Lifted out of `turn_support.py` unchanged.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import hashlib
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
)

from ..catalog_capabilities import (
    effective_filter_capabilities,
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
            "The kind of product this focused role is, worked out from the "
            "shopper's latest words and the last few turns before considering "
            "what this catalog carries. Name the product itself, not the part "
            "it plays or the shopper's phrasing: 'something to layer over a "
            "sleeveless dress' is 'cardigan', never 'layer'. Never replace what "
            "they asked for with something the catalog has instead: 'jeans' "
            "stays 'jeans' whether or not it is sold here. "
            "Exclude color, material, fit, occasion, weather, and style modifiers: "
            "'formal tops' and 'relaxed-fit tops' both use 'tops'. In taxonomy, "
            "name only the advertised subcategories this kind of product falls "
            "under, not every one the role could touch. When the shopper did "
            "not name it, the reply will say the choice was yours. "
            "If this type is not separately advertised and you select one faithful "
            "advertised parent category, keep this type unchanged. "
            "This is not catalog taxonomy or a ranking query. Use null "
            "for image-only search, and for a request that names no product type "
            "at all -- 'nothing over $50' is a complete request scoped by its "
            "price, and inventing a noun for it makes this field say the shopper "
            "asked for something they did not. Null here is not an unscoped "
            "search: taxonomy or a hard filter still has to say which products "
            "are meant."
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
        if self.taxonomy_status == "image_only":
            if has_shopper_guidance:
                raise ValueError(
                    "A non-text retrieval path requires empty shopper_guidance"
                )
        elif not has_shopper_guidance:
            raise ValueError(
                "catalog retrieval requires non-empty shopper_guidance"
            )
        if (
            self.taxonomy_status != "image_only"
            and not requested_product_type
            and not has_taxonomy
            and not self._scoped_by_a_hard_filter()
        ):
            # Required unconditionally, this field had to be invented on the one
            # request that has no product type to give it. "Nothing over $50"
            # named none, so the model filled it with "items" -- and the
            # vocabulary judge, whose question is whether a named product type
            # is stocked, answered that this catalog carries no "items". The
            # turn told the shopper the shop had nothing, with four bags under
            # fifty dollars sitting behind a search that was never run.
            #
            # A scope still has to say which products are meant. Taxonomy says
            # it, and so does an enforceable filter; what is not allowed is an
            # unscoped search, and the rules below still refuse one.
            raise ValueError(
                "text catalog search requires requested_product_type, an "
                "advertised category or subcategory, or a hard filter"
            )
        if self.taxonomy_status == "image_only" and requested_product_type:
            raise ValueError(
                "image-only search requires requested_product_type=null"
            )
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


