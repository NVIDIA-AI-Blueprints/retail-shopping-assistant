# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One catalog search, from tool arguments to the evidence the model reads.

This was a 943-line closure inside `DeepAgentsRuntime._create_agent`, which made
it unreachable from a test and impossible to read without also reading the agent
that built it. It captured six things from that scope; `SearchContext` names them
explicitly so the search can be called, and read, on its own.

The order of what follows is the order a search actually goes through: admit the
call, validate its arguments against current catalog capabilities, establish that
its taxonomy and its stated requirements came from the shopper rather than from
the model, then plan, execute, and render what was found. Most of the length is
the third step -- each gate that turns a call back has to tell the model exactly
what to preserve and what to change, or the repair loops.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pydantic import (
    BaseModel,
    ValidationError,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
)

from .agenttypes import State
from .catalog_execution import execute_catalog_search
from .catalog_request import (
    CatalogSearchIntent,
    _filter_values,
    build_catalog_search_plan,
)
from .control_signals import (
    NOT_CARRIED_KEY,
    REJECTIONS_KEY,
    ControlSignal,
    SearchRejection,
    control,
)
from .response_format import (
    SEARCH_RESULT_ATTRIBUTE_LIMIT_NOTE,
    _format_catalog_scope_outcome,
    _format_colour_words_read_as_advertised_ones,
    _format_product_record,
    _format_search_composed_role_evidence,
    _format_search_direction_evidence,
    _format_search_filter_evidence,
    _format_search_guidance_evidence,
    _format_search_scope_relation_evidence,
    _format_search_taxonomy_evidence,
    _format_search_unadvertised_type_evidence,
    _format_words_this_catalog_cannot_filter_on,
)
from .tool_evidence import (
    EVIDENCE_KEY,
    SearchEvidence,
)
from .tool_loop_control import (
    CONSTRAINT_REVIEW_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
)
from .turn_scope import CatalogRepairState, TurnScope
from .turn_support import (
    _ONE_SIZE,
    _SEARCH_BUDGET_EXHAUSTED_NOTE,
    _SEARCH_NO_MATCH_GROUNDING_NOTE,
    _SEARCH_RESULT_GROUNDING_NOTE,
    _SEARCH_SCOPE_COMPLETE_NOTE,
    _UNSUPPORTED_SEARCH_MODE_MESSAGE,
    SearchCatalogToolArguments,
    _advertised_scope_match,
    _advertised_subcategories_for_selection,
    _advertised_taxonomy_scope_issue,
    _advertised_taxonomy_value,
    _agent_selected_scope_is_advertised,
    _append_product_results,
    _catalog_execution_taxonomy_status,
    _catalog_search_scope,
    _duplicates_unavailable_product_type,
    _exact_taxonomy_issue,
    _generic_shopper_guidance,
    _multi_subcategory_candidate_limit,
    _normalize_product_text,
    _normalized_scope_value,
    _product_scope_key,
    _products_with_subcategory_coverage,
    _record_catalog_model_usage,
    _resolved_agent_selected_product_type,
    _safe_shopper_guidance,
    _same_product_scope,
    _search_product_record,
    _selected_advertised_subcategories,
    _shopper_stated_product_scope,
    _shopper_stated_requirement,
    _taxonomy_hard_constraints,
    _tool_search_mode,
    _unsupported_requirement_message,
    stated_media_terms,
)
from .vocabulary_judge import ScopeQuestion

#: What a search step hands back: nothing, meaning the search continues, or the
#: text the model reads -- paired with the evidence artifact behind it when the
#: step produced one.
StepResult = str | tuple[str, dict[str, Any]] | None

#: Said wherever a role is reported uncoverable.
#
#: Refusing the search was only ever half of it. Told this shop has no jeans,
#: the assistant said so plainly and then offered a navy skirt, two dresses and
#: a blouse as "the closest dark-blue bottoms I found" -- disclosure and
#: substitution in the same breath, under a heading that still read "Bottoms --
#: dark blue jeans". Nothing had forbidden the second half.
_NO_STAND_IN = (
    "Show nothing for it and offer no other garment as the closest version of "
    "it. Offering to look for a different kind of piece is fine if the shopper "
    "is asked first."
)


@dataclass(frozen=True)
class SearchContext:
    """What one catalog search needs from the turn that created it.

    These six were closure captures. Naming them keeps the search callable
    outside `_create_agent` and makes its dependencies visible: two are live
    turn state that the search mutates (`state`, `scope`), two are read-only
    facts about this turn (`config`, `capabilities`), and two are the argument
    schemas built from those capabilities.
    """

    config: Any
    state: State
    scope: TurnScope
    capabilities: CatalogCapabilities
    search_input_model: type[BaseModel]
    constraint_input_model: type[BaseModel]
    #: Optional, and absent in tests that build a context by hand. A scope the
    #: judge never saw is decided by the gates that decided it before.
    vocabulary_judge: Any = None


def _lock_taxonomy_constraint_values(
    repair: CatalogRepairState,
    scope_key: str | None,
    constraints: dict[str, Any],
    *,
    allow_no_direct_clear: bool = False,
) -> str:
    """Store canonical hard constraints for one taxonomy repair."""

    constraints = _normalized_scope_value(constraints)
    constraints.pop("unadvertised_requirements", None)
    repair.pending_taxonomy_constraints = constraints
    repair.pending_no_direct_constraint_clear = allow_no_direct_clear
    serialized_constraints = json.dumps(
        constraints,
        ensure_ascii=False,
        sort_keys=True,
    )
    if allow_no_direct_clear:
        return (
            " Clear advertised required_constraints if the corrected "
            "request does not retrieve. Otherwise preserve these "
            "capability-validated advertised required_constraints "
            f"exactly: {serialized_constraints}."
        )
    if not constraints:
        return (
            " The rejected call had no advertised required_constraints. "
            "Keep advertised required_constraints empty on repair. "
            "Change only taxonomy or an explicitly identified ungrounded "
            "product scope."
        )
    return (
        " Preserve these capability-validated advertised "
        "required_constraints exactly on repair: "
        f"{serialized_constraints}. Change only taxonomy or an "
        "explicitly identified ungrounded product scope."
    )


def _lock_taxonomy_constraints(
    repair: CatalogRepairState,
    scope_key: str | None,
    request: SearchCatalogToolArguments,
) -> str:
    """Preserve validated hard constraints across one taxonomy repair."""

    return _lock_taxonomy_constraint_values(
        repair,
        scope_key,
        request.required_constraints.model_dump(exclude_none=True),
    )


def _stated_shopper_text(state: Any) -> str:
    """What the shopper said this turn, including what they showed.

    Provenance was computed from the typed query alone, so every attribute a
    shopper conveyed by attaching a photo or video read as model-invented and
    was refused. An image is a statement; this makes the gate able to hear it.

    The garment is a statement on the same terms as the colour, which is why
    `fashion_items` sits in `_STATED_MEDIA_FIELDS` alongside `colors`. "Shop
    the jeans in this video" names jeans exactly as typing the word would, and
    reading the roles from the query alone made the video's own garments count
    as roles the model had composed -- so the model was free to rename one, and
    it did: told the shop has no jeans it asked for "bottoms", where a skirt
    genuinely is a kind of bottoms, and three skirts came back for a jeans
    request. Every product-scope gate reads this, so the video's words bind the
    scope the same way the typed ones do.

    Only the media fields that describe the media's *content* are included --
    see `_STATED_MEDIA_FIELDS`. The model's reading of the image is still not
    the shopper speaking.
    """

    return " ".join(
        part for part in (
            getattr(state, "query", "") or "",
            stated_media_terms(getattr(state, "media_analysis", "") or ""),
        ) if part
    )


@dataclass
class _Attempt:
    """What one catalog search accumulates as it moves through the steps below.

    Each step reads what earlier steps worked out and leaves its own findings
    here. Keeping it in one object is what lets the steps stay separate
    functions without threading a dozen arguments through each of them.
    """

    semantic_query: str
    requested_product_type: str | None
    taxonomy: BaseModel | dict[str, Any]
    required_constraints: BaseModel | dict[str, Any]
    shopper_guidance: str
    #: Repair bookkeeping for this scope alone. Sharing one across scopes made a
    #: rejection of the shoes lock out the bag for the rest of the turn.
    repair: Any = None
    scope_complete: bool = True
    search_mode: str | None = None
    advertised_choices: Any = None
    candidate_scope_key: Any = None
    #: The taxonomy and constraints this scope searched on, kept so a reworded
    #: repeat of it can be recognised and answered from what it already found.
    catalog_scope: Any = None
    #: Whether the shopper named this scope, in words or by showing it.
    #: Decided once in the first step; every later step reads it.
    shopper_named_scope: bool = False
    capabilities: Any = None
    #: True when the shopper never named this role and the model composed it.
    #: Recorded rather than refused: the reply has to present it as proposed,
    #: and must not read a miss inside the searched types as the role being
    #: unavailable.
    composed_role: bool = False
    #: Where the judge says this requested word lives in the catalogue, and the
    #: scope this role will search. Four flags used to stand here -- not-a-kind
    #: members, ruled-or-not, umbrella members, umbrella-ruled-or-not -- because
    #: the judge graded the model's guess and the result had to be reassembled
    #: from a verdict and the guess it was about. It names the values now, so
    #: there is one value to carry.
    #:
    #: Empty list and None are different and the difference is load bearing.
    #: Empty is a verdict: this catalogue sells no such thing, and the shopper
    #: is told so. None means unjudged, and reporting that as "not carried"
    #: would turn an endpoint timeout into a claim about the shop's stock.
    judged_subcategories: tuple[str, ...] | None = None
    constraint_payload: Any = None
    evidence: Any = None
    #: Shopper scopes searched by *earlier calls*, snapshotted at the start
    #: of this one. A role must not collide with its own siblings.
    prior_shopper_scopes: Any = None
    execution: Any = None
    lines: Any = None
    normalized_constraints: Any = field(default_factory=dict)
    plan: Any = None
    #: Which gate turned this scope back, recorded where the decision is made.
    #: The nine gates below all render one prefix, so the text they hand the
    #: model cannot say which one refused; this can.
    rejection_code: str | None = None
    #: The shopper's own word for a product type this catalog advertises
    #: nothing for. A rejection says the arguments were wrong; this says the
    #: thing does not exist here, which is an answer rather than an error.
    not_carried: str | None = None
    #: Words this catalog cannot filter on, set aside before validation rather
    #: than refused. Field name to the values dropped from it.
    set_aside: dict[str, list[str]] = field(default_factory=dict)
    #: Unadvertised colour words this scope offered, to the advertised colours
    #: the judge says they may mean. Keyed casefolded on the word as sent.
    colour_map: dict[str, list[str]] = field(default_factory=dict)
    #: The subset of those that became a filter, to what they filtered on.
    #: Read by the disclosure, since "ranked not filtered" stops being true
    #: for them.
    colours_mapped: dict[str, list[str]] = field(default_factory=dict)
    request: Any = None
    result: Any = None
    search_budget_exhausted: Any = None
    selected_subcategories: Any = None
    shopper_scope_key: Any = None
    shopper_stated_scope: Any = None
    suppress_requirement_disclosure: Any = None
    taxonomy_constraints: Any = field(default_factory=dict)
    taxonomy_fields: Any = field(default_factory=set)
    taxonomy_payload: Any = None
    taxonomy_status: Any = None
    unconfirmable_requirements: Any = field(default_factory=list)
    size_the_scope_has_not: Any = field(default_factory=dict)


def _rejected(
    attempt: _Attempt,
    code: SearchRejection,
    result: StepResult,
) -> StepResult:
    """Name the gate that is turning this scope back, and return its own text.

    Wrapping the return rather than assigning on the line above is deliberate:
    a code recorded anywhere other than the return it belongs to can drift away
    from it, which is how the single shared prefix stopped meaning anything.

    Not every early return is a rejection. A catalog that is unavailable, an
    image search the catalog does not offer, a request carrying neither query
    nor image, and a retrieval that actually failed all hand the model an
    instruction to keep the conversation going, and are reported as completed
    calls today. Giving them a code would silently reclassify them as refusals
    and replace the model's answer with the fixed refusal response, so they
    stay uncoded until that is a decision someone makes on purpose.
    """

    attempt.rejection_code = str(code)
    if attempt.repair is not None:
        attempt.repair.last_rejected_scope = _scope_as_sent(attempt)
    return result


def _scope_as_sent(attempt: _Attempt) -> str:
    """Fingerprint this scope, so an unchanged retry can be recognised."""

    return json.dumps(
        {
            "semantic_query": attempt.semantic_query,
            "requested_product_type": attempt.requested_product_type,
            "taxonomy": attempt.taxonomy,
            "required_constraints": attempt.required_constraints,
        },
        sort_keys=True,
        default=str,
    )


def _reconciled_with_what_is_advertised(
    ctx: SearchContext,
    attempt: _Attempt,
) -> StepResult:
    """Set aside a filter value this catalog cannot honour, and search anyway.

    The advertised vocabulary is loaded, finite, and right here. A value
    outside it failed schema validation and the scope was handed back to the
    model to repair -- a round trip through a full prompt to resolve what a
    set membership test answers. "cream" is not one of the colours this shop
    advertises, so it cannot be a filter; it does not have to be, because it
    is already in `semantic_query` where the index can rank on it.

    Filter values only, and product types deliberately not. Set aside a
    colour and what remains is still a search for the right garment, ranked
    rather than filtered. Set aside a type and what remains is the
    department -- and a department is not a family. Every member of footwear
    is a shoe, but apparel here is 39 skirts, 33 dresses, 18 sweaters, 9
    blouses, 2 camisoles and 1 jumpsuit, so ranking "dark blue straight leg
    jeans" across it returns the skirts. A type this shop does not sell stays
    a schema error, which is the earliest and plainest place to say so.

    Nothing here translates. Cream is not mapped to beige. The words stay in
    the query, the index ranks on them, and the disclosure says the value was
    ranked rather than filtered so the shopper can judge it themselves.
    """

    capabilities = ctx.capabilities
    set_aside: dict[str, list[str]] = {}

    constraints = attempt.required_constraints
    constraints = (
        constraints.model_dump(exclude_none=True)
        if isinstance(constraints, BaseModel)
        else dict(constraints or {})
    )
    for name, value in list(constraints.items()):
        capability = capabilities.filters.get(name)
        advertised_values = list(getattr(capability, "values", None) or ())
        if capability is None or not advertised_values:
            continue
        advertised = {
            str(advertised_value).casefold(): str(advertised_value)
            for advertised_value in advertised_values
        }
        offered = value if isinstance(value, (list, tuple)) else [value]
        dropped = [
            str(item)
            for item in offered
            if str(item).casefold() not in advertised
        ]
        if not dropped:
            continue
        set_aside[name] = dropped
        # What this catalog can honour stays. Only the word it cannot goes.
        #
        # Dropping the whole field was tried first and is worse, because the
        # field *is* the filter: losing it leaves no colour constraint at all.
        # Asked for a cream sweater as `["cream", "beige"]`, the search ranked
        # on "cable-knit" alone and returned sweaters in any colour, red among
        # them. Keeping `["beige"]` returns beige ones.
        #
        # Keeping half can still be narrower than the model meant -- `["cream",
        # "white"]` filters to white in a shop whose cream is beige. That is
        # why the scope prompt asks for every advertised value the shopper's
        # word could be rather than the single nearest, and why the disclosure
        # below names what was set aside either way.
        kept = [
            advertised[str(item).casefold()]
            for item in offered
            if str(item).casefold() in advertised
        ]

        # A colour word this shop does not list is still a colour. The judge
        # says which advertised ones it could mean, and those filter in its
        # place, so the constraint survives rather than the field being deleted
        # and sweaters coming back in any colour.
        #
        # Unioned rather than consulted only when nothing else survived: cream
        # should mean the same thing whether or not the model happened to send
        # a valid value beside it. `["beige", "cream"]` filtered on beige alone
        # is half the shade the shopper described, and that shape is the common
        # one -- 118 of 3,040 colour scopes, against 15 that named nothing
        # advertised at all.
        if name == _colour_field(ctx):
            for word in dropped:
                mapped = attempt.colour_map.get(word.casefold()) or []
                if not mapped:
                    continue
                attempt.colours_mapped[word] = mapped
                kept.extend(colour for colour in mapped if colour not in kept)
            # Only the words that mapped to nothing were truly set aside. The
            # rest were filtered on, and the set-aside disclosure says results
            # are ranked and guarantee nothing -- untrue of these, and the
            # model repeats it to the shopper.
            remaining = [
                word for word in dropped if word not in attempt.colours_mapped
            ]
            if remaining:
                set_aside[name] = remaining
            else:
                set_aside.pop(name, None)

        if kept:
            constraints[name] = kept
        else:
            del constraints[name]

    attempt.required_constraints = constraints
    attempt.set_aside = set_aside
    return None


def _admit_search(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Decide whether this call may run at all, before anything is parsed.

    Two of these are repair locks rather than validation. Once a call has been
    turned back, the model may retry -- but a retry that quietly moves to a
    different product scope is not a repair, it is a second search wearing the
    first one's budget. Both gates below refuse that.
    """

    requested_product_type = attempt.requested_product_type
    required_constraints = attempt.required_constraints
    semantic_query = attempt.semantic_query
    taxonomy = attempt.taxonomy

    taxonomy = taxonomy or {"category": [], "subcategory": []}
    required_constraints = required_constraints or {}
    capabilities = ctx.capabilities
    if capabilities.catalog_id == "unavailable" and not capabilities.filters:
        return "Catalog search is unavailable. Please try again."
    if (
        attempt.repair is not None
        and attempt.repair.last_rejected_scope is not None
        and attempt.repair.last_rejected_scope == _scope_as_sent(attempt)
    ):
        # Byte for byte what was just turned back. Whatever the rejection
        # asked for, this is not it, and the gates below will reach the same
        # verdict for the same reason -- so the only thing a third attempt
        # buys is another full prompt. The turn is told to answer with what
        # it has instead, which it can: the roles that searched, searched.
        attempt.repair.last_rejected_scope = None
        return (
            "SEARCH_NOT_REPAIRED: this is the same request that was just "
            "turned back, unchanged, so it was not run again. Do not send it "
            "a third time. Answer the shopper with the results you already "
            "have, and tell them plainly which part of what they asked for "
            "you could not look up."
        )
    initial_scope_key = _product_scope_key(requested_product_type)
    shopper_stated_requested_scope = bool(
        initial_scope_key
        and _shopper_stated_product_scope(
            _stated_shopper_text(ctx.state),
            ctx.state.dialogue,
            initial_scope_key,
        )
    )
    taxonomy_status = _catalog_execution_taxonomy_status(
        requested_product_type,
        taxonomy,
        semantic_query,
        capabilities,
        shopper_stated_scope=shopper_stated_requested_scope,
    )

    requested_product_type = _resolved_agent_selected_product_type(
        query=_stated_shopper_text(ctx.state),
        dialogue=ctx.state.dialogue,
        requested_product_type=requested_product_type,
        taxonomy_status=taxonomy_status,
        taxonomy=taxonomy,
    )
    candidate_scope_key = _product_scope_key(requested_product_type)
    locked_repair_scope = (
        attempt.repair.failed_constraint_scope_key or attempt.repair.failed_repair_scope_key
    )
    repairing_same_scope = bool(
        locked_repair_scope
        and (
            candidate_scope_key == attempt.repair.failed_constraint_scope_key
            if attempt.repair.failed_constraint_scope_key
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
        return _rejected(
            attempt,
            SearchRejection.REPAIR_CHANGED_PRODUCT_SCOPE,
            SEARCH_VALIDATION_ERROR_PREFIX
            + "A catalog search repair cannot replace product scope "
            f"'{expected_scope_key}' "
            f"with '{candidate_scope_key or 'none'}'. Preserve the "
            "requested_product_type and repair taxonomy instead.",
        )
    attempt.candidate_scope_key = candidate_scope_key
    # Decided here and read everywhere else. The same three inputs -- this
    # scope's key, what the shopper said and showed, and the recent dialogue --
    # were re-derived at five later points, four of them inside one function,
    # and a pure text match on identical arguments cannot give five different
    # answers. What it can do is drift: while one site read the typed query and
    # another read the query plus the media, "cream" counted as the shopper
    # speaking and "jeans" from the same video counted as the model's own
    # invention, three lines apart. One derivation cannot disagree with itself.
    attempt.shopper_named_scope = bool(
        candidate_scope_key
        and _shopper_stated_product_scope(
            _stated_shopper_text(ctx.state),
            ctx.state.dialogue,
            candidate_scope_key,
        )
    )
    attempt.capabilities = capabilities
    attempt.requested_product_type = requested_product_type
    attempt.required_constraints = required_constraints
    attempt.taxonomy = taxonomy
    attempt.taxonomy_status = taxonomy_status
    return None


def _classify_requirements(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Work out where each stated requirement came from.

    A requirement the shopper actually said is ranked on and disclosed. One the
    model inferred earns a single review. One that merely repeats the product
    type is neither -- there is no invented attribute in it to account for. This
    step only classifies; the gates that act on the classification come later.
    """

    capabilities = attempt.capabilities
    requested_product_type = attempt.requested_product_type
    required_constraints = attempt.required_constraints
    taxonomy = attempt.taxonomy

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
    shopper_stated_scope = attempt.shopper_named_scope
    raw_unadvertised_requirements = (
        constraint_payload.get("unadvertised_requirements", [])
        if isinstance(constraint_payload, dict)
        else []
    )
    suppress_requirement_disclosure = False
    duplicated_product_type = bool(
        shopper_stated_scope
        and _duplicates_unavailable_product_type(
            raw_unadvertised_requirements,
            requested_product_type,
            capabilities,
        )
    )
    if duplicated_product_type:
        # The product type is already carried by requested_product_type
        # and by the semantic query, and unadvertised_requirements never
        # becomes a filter. Rejecting here discarded a search whose
        # outbound catalog payload was identical to one that succeeds,
        # so correct the annotation and let retrieval run.
        # Deliberately no rewrite: deterministic code does not repair
        # model arguments. The field is popped before hard filters are
        # built and never reaches the catalog, so leaving it untouched
        # changes nothing except that the model keeps ownership of what
        # it declared. Only the veto is removed.
        #
        # It is still not disclosed as an unconfirmable *attribute*: the
        # value is the product type, and telling a shopper that
        # "sneakers is not an advertised hard filter" describes the
        # schema rather than their request. Choosing what to say is not
        # rewriting what the model sent.
        suppress_requirement_disclosure = True
    stated_unadvertised_requirements = (
        [
            requirement
            for requirement in raw_unadvertised_requirements
            if isinstance(requirement, str)
            and _shopper_stated_requirement(
                _stated_shopper_text(ctx.state), requirement
            )
        ]
        if isinstance(raw_unadvertised_requirements, list)
        else []
    )
    # An unenforceable requirement is a ranking preference, not a veto.
    # It is already carried by the semantic query and is stripped before
    # hard filters are built, so the search that would have run here is
    # the same search either way. Abandoning it left the composer with
    # nothing to show and turned a valid request into a refusal.
    unconfirmable_requirements = (
        list(raw_unadvertised_requirements)
        if not suppress_requirement_disclosure
        and isinstance(raw_unadvertised_requirements, list)
        and raw_unadvertised_requirements
        and (shopper_stated_scope or stated_unadvertised_requirements)
        else []
    )

    attempt.constraint_payload = constraint_payload
    attempt.shopper_stated_scope = shopper_stated_scope
    attempt.suppress_requirement_disclosure = suppress_requirement_disclosure
    attempt.taxonomy_payload = taxonomy_payload
    attempt.unconfirmable_requirements = unconfirmable_requirements
    return None


def _validated_request(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Validate the arguments against what this catalog currently advertises.

    Most of the length is the rejection path. A bare schema error tells the model
    nothing it can act on, so the failure carries the advertised values it may
    choose from and the exact constraints it must preserve -- otherwise the
    repair drifts and each attempt spends another search from the turn's budget.
    """

    advertised_choices = attempt.advertised_choices
    candidate_scope_key = attempt.candidate_scope_key
    capabilities = attempt.capabilities
    constraint_payload = attempt.constraint_payload
    request = attempt.request
    requested_product_type = attempt.requested_product_type
    required_constraints = attempt.required_constraints
    scope_complete = attempt.scope_complete
    search_mode = attempt.search_mode
    semantic_query = attempt.semantic_query
    shopper_guidance = attempt.shopper_guidance
    shopper_stated_scope = attempt.shopper_stated_scope
    taxonomy = attempt.taxonomy
    taxonomy_payload = attempt.taxonomy_payload
    taxonomy_status = attempt.taxonomy_status

    try:
        request = ctx.search_input_model.model_validate(
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
        # A composed role that has already settled on an advertised type is a
        # committed scope and may not drift on repair. One that has not is
        # still the model's own wording, with nothing of the shopper's in it to
        # preserve, so it stays free to be re-composed.
        canonical_agent_selected_scope = bool(
            candidate_scope_key
            and taxonomy_status == "agent_selected_type"
            and _agent_selected_scope_is_advertised(
                requested_product_type,
                taxonomy,
            )
        )
        if shopper_stated_scope or canonical_agent_selected_scope:
            attempt.repair.failed_repair_scope_key = candidate_scope_key
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
                    " For a role the shopper did not name, keep your role "
                    "noun in requested_product_type and select every "
                    "advertised subcategory that role covers. Choose from "
                    "these currently advertised subcategories: "
                    + json.dumps(advertised_choices, ensure_ascii=False)
                    + ". If the role is not a kind of any of them, do not "
                    "choose one: name it in not_covered and tell the shopper "
                    "this catalog does not carry it. Offering only the list "
                    "read as an instruction to pick from it -- asked to "
                    "compare two aprons, a catalog with no aprons produced an "
                    "empty taxonomy five turns running rather than saying so."
                )
                # The role may not exist at all. "Nothing over $50" names no
                # product type, so every category the shop has is in scope and
                # any single one of them is the wrong answer -- but this
                # refusal only ever described how to narrow, so the model kept
                # narrowing. It picked apparel four runs in five and the
                # shopper was asked to clarify a request that was complete.
                #
                # Both wider shapes are already legal; neither was ever said
                # out loud at the point they were needed.
                if _hard_filter_scopes_this(required_constraints):
                    repair_guidance += (
                        " If the shopper named no product type at all -- a "
                        "budget, a colour, nothing else -- then no category is "
                        "the right one and choosing one shows a fraction of "
                        "what they asked for. Either leave the category out "
                        "entirely and let the filter scope the search, or name "
                        "every advertised category and it will be split into "
                        "one search each."
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
                    attempt.repair.pending_schema_requirements = list(
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
            elif shopper_stated_scope and _advertised_scope_match(
                requested_product_type,
                capabilities,
            ) is None:
                # The same dead end, reached by the other door: a type the
                # shopper names is classified exact_requested_type, never
                # agent_selected_type, so it took none of the guidance above and
                # heard only that its arguments failed validation.
                #
                # Whether it is carried is not in doubt here -- nothing this
                # catalog advertises matches the word -- so this is recorded as
                # a fact rather than left to the model to volunteer. The
                # guidance stops it substituting a different product for the one
                # the shopper asked about.
                attempt.not_carried = str(requested_product_type)
                repair_guidance = (
                    " This catalog advertises nothing of that kind, so there is "
                    "no taxonomy that would make this search valid. Do not "
                    "substitute a product type the shopper did not ask for: "
                    "tell them plainly it is not carried, and offer the "
                    "advertised kinds only if they ask what there is instead."
                )
        constraint_lock = ""
        try:
            validated_constraints = ctx.constraint_input_model.model_validate(
                constraint_payload
            )
        except ValidationError:
            pass
        else:
            constraint_lock = _lock_taxonomy_constraint_values(
                attempt.repair,
                candidate_scope_key,
                validated_constraints.model_dump(exclude_none=True),
                allow_no_direct_clear=(
                    taxonomy_status == "no_direct_catalog_match"
                ),
            )
        return _rejected(
            attempt,
            SearchRejection.CAPABILITIES_SCHEMA_MISMATCH,
            SEARCH_VALIDATION_ERROR_PREFIX
            + "The catalog search request does not match current "
            f"capabilities: {validation_errors}"
            + repair_guidance
            + constraint_lock,
        )

    attempt.advertised_choices = advertised_choices
    attempt.request = request
    return None


def _reviewed_provenance(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Hold the repair to the request it is repairing.

    This is the longest step because every way out of it has to tell the model
    precisely what to keep and what to change. Each gate turns the call back for
    one reason: a repair that altered constraints it was supposed to preserve, a
    taxonomy the catalog does not advertise, an open-role search that never chose
    a role, or a requirement whose provenance in this turn cannot be established.
    """

    candidate_scope_key = attempt.candidate_scope_key
    capabilities = attempt.capabilities
    request = attempt.request
    suppress_requirement_disclosure = attempt.suppress_requirement_disclosure
    unconfirmable_requirements = attempt.unconfirmable_requirements

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
        attempt.repair.pending_taxonomy_constraints is not None
        and not (
            attempt.repair.pending_no_direct_constraint_clear
            and request.taxonomy_status == "no_direct_catalog_match"
        )
        and normalized_advertised_constraints
        != attempt.repair.pending_taxonomy_constraints
    ):
        return _rejected(
            attempt,
            SearchRejection.REPAIR_CHANGED_CONSTRAINTS,
            SEARCH_VALIDATION_ERROR_PREFIX
            + "A taxonomy repair must preserve previously validated "
            "advertised required_constraints exactly. Change only "
            "taxonomy or an explicitly identified "
            "ungrounded product scope.",
        )
    if attempt.repair.pending_taxonomy_constraints is not None:
        attempt.repair.pending_taxonomy_constraints = None
        attempt.repair.pending_no_direct_constraint_clear = False
    normalized_constraints = dict(all_constraints)
    unadvertised_requirements = normalized_constraints.pop(
        "unadvertised_requirements",
        [],
    )
    shopper_stated_scope = attempt.shopper_named_scope
    if (
        unadvertised_requirements
        and shopper_stated_scope
        and not suppress_requirement_disclosure
    ):
        # Rank on it, disclose it, do not abandon the search.
        unconfirmable_requirements = list(unadvertised_requirements)

    advertised_taxonomy_issue = _advertised_taxonomy_scope_issue(
        request.requested_product_type,
        request.taxonomy_status,
        request.taxonomy,
        capabilities,
    )
    if advertised_taxonomy_issue:
        if shopper_stated_scope:
            attempt.repair.failed_repair_scope_key = candidate_scope_key
        return _rejected(
            attempt,
            SearchRejection.TAXONOMY_NOT_ADVERTISED_FOR_SCOPE,
            SEARCH_VALIDATION_ERROR_PREFIX
            + advertised_taxonomy_issue
            + _lock_taxonomy_constraints(attempt.repair, candidate_scope_key, request)
            + (
                " Preserve the shopper-stated requested_product_type."
                if shopper_stated_scope
                else " The rejected requested_product_type was not "
                "shopper-stated. Re-read the current shopper request "
                "and correct it rather than preserving this scope."
            ),
        )

    if attempt.repair.pending_schema_requirements and not unadvertised_requirements:
        request = request.model_copy(
            update={
                "shopper_guidance": _generic_shopper_guidance(
                    request.requested_product_type
                )
            }
        )
        attempt.repair.pending_schema_requirements = []
    pending_constraint_review = attempt.repair.pending_constraint_reviews.get(
        candidate_scope_key
    )
    if pending_constraint_review and (
        request.taxonomy.model_dump()
        != pending_constraint_review["taxonomy"]
        or request.scope_complete
        != pending_constraint_review["scope_complete"]
        or request.search_mode != pending_constraint_review["search_mode"]
        or normalized_constraints
        != pending_constraint_review["required_constraints"]
    ):
        return _rejected(
            attempt,
            SearchRejection.CONSTRAINT_REPAIR_CHANGED_REQUEST,
            SEARCH_VALIDATION_ERROR_PREFIX
            + "A constraint-provenance repair must preserve "
            "requested_product_type, taxonomy, scope_complete, "
            "search_mode, and all "
            "advertised required constraints exactly. Change only the "
            "reviewed unadvertised requirement wording or remove an "
            "inferred requirement; the soft semantic query may be "
            "corrected within the preserved product scope.",
        )
    # A role the shopper never named is the model's own composition -- "a top"
    # for someone who asked for an outfit, covering blouses and sweaters
    # because the catalog has no "tops". That is not a fault to turn back. It
    # is a fact to record: the reply must present the role as proposed rather
    # than as something the shopper asked for, and must not read a miss within
    # the searched types as the role being unavailable.
    #
    # Refusing it cost a round trip and a worse answer every time it fired. In
    # one whole-look turn the shoes role was forced from [flats, sandals] down
    # to flats, and the shopper had to ask for sandals back two turns later.
    agent_selected_issue: str | None = None
    agent_selected_shopper_scope = attempt.shopper_named_scope
    attempt.composed_role = (
        request.taxonomy_status == "agent_selected_type"
        and not agent_selected_shopper_scope
    )
    # Recorded, not policed. A composed role is the model covering a garment
    # this shop has no single word for -- "a top" across blouses and sweaters,
    # "shoes" across flats and heels and boots -- and a gate used to stand here
    # deciding whether such a role was honest or a substitution, from a list of
    # eighteen garment words and a rule about how many subcategories a scope
    # may name. Asked to shop a look whose jeans this shop does not carry, the
    # model sent `subcategory: ["skirts"]` with `semantic_query: "dark wash
    # straight-leg jeans"`; every declared field was advertised and
    # self-consistent, so no amount of reading the declaration could catch it.
    # The list caught jeans and missed belts filed under blouses 39 times,
    # because a list only knows the words on it.
    #
    # `_resolved_against_the_catalogue` settles it instead, by asking the
    # catalogue's own vocabulary where the word lives. Substitution is not
    # detected there, it is impossible: the model's taxonomy is overwritten
    # rather than inspected, so "jeans" cannot arrive as skirts whether or not
    # jeans are on anybody's list.
    if (
        request.taxonomy_status == "agent_selected_type"
        and agent_selected_shopper_scope
    ):
        # The shopper did name this scope, so the model may not quietly answer
        # it as an open role: that narrows what they asked for.
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
            "Keep that exact advertised taxonomy selection."
            if exact_selected_scope
            else (
                "Select only advertised values that are kinds of the "
                "named scope; do not narrow to one convenient child."
            )
        )
        agent_selected_issue = (
            "The shopper named requested_product_type "
            f"'{request.requested_product_type}', so "
            "preserve that requested_product_type and these advertised "
            "values on repair: "
            + json.dumps(selected_values, sort_keys=True)
            + ". "
            + repair_status
        )
    if agent_selected_issue:
        attempt.repair.failed_repair_scope_key = candidate_scope_key
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
        return _rejected(
            attempt,
            SearchRejection.SHOPPER_SCOPE_TAXONOMY_MISMATCH,
            SEARCH_VALIDATION_ERROR_PREFIX
            + agent_selected_issue
            + constraint_issue
            + _lock_taxonomy_constraints(attempt.repair, candidate_scope_key, request),
        )

    if (
        request.taxonomy_status != "no_direct_catalog_match"
        and unadvertised_requirements
    ):
        stated_requirements = [
            requirement
            for requirement in unadvertised_requirements
            if _shopper_stated_requirement(
                _stated_shopper_text(ctx.state), requirement
            )
        ]
        shopper_stated_scope = attempt.shopper_named_scope
        if (
            stated_requirements or shopper_stated_scope
        ) and not suppress_requirement_disclosure:
            # Rank on it, disclose it, do not abandon the search.
            unconfirmable_requirements = list(unadvertised_requirements)
        if (
            not unconfirmable_requirements
            and not suppress_requirement_disclosure
        ):
            # Provenance could not be established from this turn, so the
            # model may have inferred the requirement. That still earns
            # one review. A requirement the shopper actually stated does
            # not: it ranks the search and is disclosed instead. Nor does
            # a value that is simply the product type -- there is no
            # invented attribute there to establish provenance for.
            review_scope = candidate_scope_key or "__unknown__"
            if review_scope in attempt.repair.constraint_reviewed_scopes:
                return _rejected(
                    attempt,
                    SearchRejection.REQUIREMENT_PROVENANCE_UNESTABLISHED,
                    "The requested catalog requirement cannot be enforced: "
                    "its current-turn provenance could not be established. "
                    "Ask the shopper to state the exact required attribute "
                    "or allow it to be treated as a preference.",
                )
            attempt.repair.constraint_reviewed_scopes.add(review_scope)
            attempt.repair.pending_constraint_reviews[review_scope] = {
                "requirements": list(unadvertised_requirements),
                "taxonomy": request.taxonomy.model_dump(),
                "scope_complete": request.scope_complete,
                "search_mode": request.search_mode,
                "required_constraints": dict(normalized_constraints),
            }
            attempt.repair.failed_constraint_scope_key = review_scope
            return _rejected(
                attempt,
                SearchRejection.CONSTRAINT_REVIEW_REQUIRED,
                CONSTRAINT_REVIEW_PREFIX
                + "These proposed unadvertised requirements do not match "
                "the current shopper turn's normalized wording: "
                + json.dumps(unadvertised_requirements, ensure_ascii=False)
                + ". Preserve requested_product_type "
                + json.dumps(request.requested_product_type)
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
                "occasion, or style goals are not explicit requirements.",
            )

    reviewed_constraint = attempt.repair.pending_constraint_reviews.pop(
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
        shopper_stated_scope = attempt.shopper_named_scope
        if shopper_stated_scope:
            attempt.repair.failed_repair_scope_key = candidate_scope_key
        return _rejected(
            attempt,
            SearchRejection.EXACT_TAXONOMY_NOT_ADVERTISED,
            SEARCH_VALIDATION_ERROR_PREFIX
            + exact_taxonomy_issue
            + _lock_taxonomy_constraints(attempt.repair, candidate_scope_key, request)
            + (
                ". Preserve the shopper-stated requested_product_type "
                f"{json.dumps(request.requested_product_type)}."
                if shopper_stated_scope
                else ". Re-read the current shopper request and correct "
                "requested_product_type if the rejected value was not "
                "shopper-stated."
            )
            + " Choose only advertised taxonomy values that faithfully "
            "represent that scope. If none does, ask one concise "
            "clarifying question instead of searching an adjacent type.",
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
        attempt.repair.failed_repair_scope_key = candidate_scope_key
        return _rejected(
            attempt,
            SearchRejection.ADVERTISED_MATCH_REPORTED_AS_GAP,
            SEARCH_VALIDATION_ERROR_PREFIX
            + f"The requested product type '{request.requested_product_type}' "
            f"matches advertised taxonomy value '{advertised_match}'. "
            "Select that advertised value instead of reporting a gap."
            + _lock_taxonomy_constraints(attempt.repair, candidate_scope_key, request),
        )

    attempt.repair.failed_repair_scope_key = None
    attempt.repair.failed_constraint_scope_key = None

    attempt.normalized_constraints = normalized_constraints
    attempt.request = request
    attempt.unconfirmable_requirements = unconfirmable_requirements
    return None


def _no_direct_match_outcome(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Report an unadvertised product type as a gap rather than searching around it.

    Substituting an adjacent taxonomy here is what makes an assistant appear to
    answer while showing something the shopper did not ask for, so this returns
    the gap and stops the tool loop instead.
    """

    candidate_scope_key = attempt.candidate_scope_key
    evidence = attempt.evidence
    lines = attempt.lines
    request = attempt.request

    # A role is a role whoever named it. This key is what stops the same role
    # being searched twice in a turn, and it used to be set only when the
    # shopper's typed words contained the product type -- so a look lifted
    # from a video had no key at all, and no retry of it was ever a duplicate.
    #
    # That is how "I love this look" cost nine model calls and 122k tokens of
    # prompt. The video's jeans are not carried here, so the model filed them
    # under jumpsuits, and every search succeeded: jumpsuits came back, then
    # skirts, then blouses, then camisoles, then dresses, each a correct
    # hard-filtered slice of a catalog that has no jeans, each told to answer
    # now and none of them a duplicate of the last. Five searches and four
    # round trips to learn what the first one had already shown.
    #
    # Keyed on the role alone, the second of those is a duplicate and says so.
    # What this does not catch is the first -- one search is the price of
    # finding out -- and what it does not block is a retry after an empty
    # result, because the key is withdrawn below when nothing came back.
    shopper_scope_key = (
        (_normalize_product_text(ctx.state.query), candidate_scope_key)
        if candidate_scope_key
        else None
    )

    if request.taxonomy_status == "no_direct_catalog_match":
        with ctx.scope.catalog_lock:
            if (
                shopper_scope_key is not None
                and shopper_scope_key in ctx.scope.searched_shopper_scopes
            ):
                # Answered from the earlier search, as above. This arm is the
                # model declaring no direct match for a role it already
                # searched, which is the same repeat wearing a different
                # status.
                answer = _already_answered(ctx, attempt)
                if answer is not None:
                    return answer
        evidence = SearchEvidence(
            outcome="no_direct_catalog_match",
            requested_product_type=request.requested_product_type,
            scope_complete=bool(request.scope_complete),
            scope_outcome={
                "outcome": "no_direct_catalog_match",
                "requested_product_type": request.requested_product_type,
            },
        )
        lines = [
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy "
            "matches the requested product type "
            f"'{request.requested_product_type}'. "
            "Do not search adjacent product types. Tell the shopper the "
            "requested type is not advertised and ask before offering an "
            "alternative.",
            _format_catalog_scope_outcome(evidence.scope_outcome),
        ]
        if evidence.scope_complete:
            lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
        return _rejected(
            attempt,
            SearchRejection.NO_ADVERTISED_TAXONOMY_MATCH,
            ("\n\n".join(lines), evidence.as_artifact()),
        )

    attempt.evidence = evidence
    attempt.lines = lines
    attempt.shopper_scope_key = shopper_scope_key
    return None


def _size_order(value: str) -> tuple[int, float, str]:
    """Sort sizes as a shopper reads a size run, with the wordy ones last."""

    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value.casefold())


def _sizes_this_scope_comes_in(taxonomy: Any, capabilities: Any) -> list[str]:
    """Every size the searched subcategories advertise, in reading order.

    Empty where a searched subcategory publishes no sizes at all: that makes
    the scope's vocabulary unknown rather than narrow, and a size must not be
    called inapplicable on missing data.
    """

    subcategories = list(getattr(taxonomy, "subcategory", None) or [])
    if not subcategories:
        return []
    categories = getattr(getattr(capabilities, "taxonomy", None), "categories", None) or {}
    advertised: list[str] = []
    seen_any = False
    for category in categories.values():
        for name, published in (getattr(category, "subcategories", None) or {}).items():
            if name not in subcategories:
                continue
            sizes = (getattr(published, "filters", None) or {}).get("sizes")
            values = [
                text
                for text in (
                    str(value.value if hasattr(value, "value") else value).strip()
                    for value in (getattr(sizes, "values", None) or ())
                )
                if text
            ]
            if not values:
                return []
            seen_any = True
            advertised.extend(values)
    if not seen_any:
        return []
    return sorted(dict.fromkeys(advertised), key=_size_order)


def _size_that_cannot_apply(
    taxonomy: Any,
    constraints: Any,
    capabilities: Any,
) -> str:
    """The size asked for, when the searched scope comes in no such size.

    "Do you have a tote bag in a size 8" spent a turn on: "there aren't any in
    that size... would you like me to show you tote bags in their standard one
    size?" Four tote bags were sitting in the result the whole time.

    The assistant was obeying us. The zero-result guidance says a size is never
    the filter you give up -- right for a garment, where the wrong size is
    something the shopper cannot wear -- and it says to offer rather than show.
    A tote bag has no sizes to be wrong about: every bags subcategory
    advertises `onesize` and nothing else. A number there is not an unmet
    requirement, it is one that cannot apply.

    Returns the offending size so the caller can drop it and say why. Silent
    once any asked size is one the scope advertises, so a garment search keeps
    the size it was given and filters on it before anything is presented.

    The vocabulary decides this, not the kind of thing being searched. Bags are
    the loud case -- every bags subcategory advertises `onesize`, so a number
    there is not an unmet requirement but one that cannot apply -- and asking
    footwear for a 12 is the same fact more quietly: these run 5-9, and
    filtering on 12 empties the result while reading like a stock-out.

    The size is read through the same coercion the query is built with, because
    a filter is declared `value | list[value]` and both shapes are legal calls.
    This guard once read the list shape only, so the turn it was written for
    came back a second time sending `"8"` where the test sent `["8"]`.
    """

    asked = (constraints or {}).get("sizes") if isinstance(constraints, dict) else None
    if asked is None:
        return ""
    wanted = _filter_values(asked)
    if not wanted or {value.casefold() for value in wanted} == {_ONE_SIZE}:
        return ""

    advertised = {
        value.casefold() for value in _sizes_this_scope_comes_in(taxonomy, capabilities)
    }
    if not advertised:
        return ""
    if any(value.casefold() in advertised for value in wanted):
        return ""
    return ", ".join(wanted)


def _planned_search(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Turn the validated request into a retrieval plan.

    Taxonomy is a selection, not a filter: if the same field arrives in both
    places the request is ambiguous and is refused rather than resolved by
    precedence, which would silently drop one of the two.
    """

    capabilities = attempt.capabilities
    normalized_constraints = attempt.normalized_constraints
    request = attempt.request
    taxonomy_constraints = attempt.taxonomy_constraints

    taxonomy_constraints, taxonomy_issues = _taxonomy_hard_constraints(
        request.taxonomy,
        capabilities,
    )
    # A number asked of a scope that has no sizes is not a requirement that
    # failed; it is one that never applied. Drop it and say so, rather than
    # returning nothing and offering to look again.
    inapplicable_size = _size_that_cannot_apply(
        request.taxonomy, normalized_constraints, capabilities
    )
    if inapplicable_size:
        normalized_constraints = {
            name: value
            for name, value in normalized_constraints.items()
            if name != "sizes"
        }
        attempt.normalized_constraints = normalized_constraints
        # Dropping it silently is how "do you have a tote bag in a size 8" got
        # answered with invented sizes: the products came back with the size
        # gone from the question and nothing saying so, leaving the size run to
        # be guessed. The catalog's own values travel with the result instead.
        attempt.size_the_scope_has_not = {
            "asked": inapplicable_size,
            "comes_in": _sizes_this_scope_comes_in(request.taxonomy, capabilities),
        }
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
        return _rejected(
            attempt,
            SearchRejection.UNSUPPORTED_CATALOG_TAXONOMY,
            "The requested catalog taxonomy cannot be enforced: "
            + "; ".join(taxonomy_issues)
            + ". Ask the shopper to choose an advertised product type.",
        )

    normalized_search_mode = _tool_search_mode(request.search_mode)
    if request.search_mode is not None and (
        normalized_search_mode is None
        or request.search_mode not in capabilities.retrieval_modes
    ):
        return _rejected(
            attempt,
            SearchRejection.UNSUPPORTED_SEARCH_MODE,
            _UNSUPPORTED_SEARCH_MODE_MESSAGE,
        )

    intent = CatalogSearchIntent(
        semantic_query=request.semantic_query,
        required_constraints={
            **normalized_constraints,
            **taxonomy_constraints,
        },
        search_mode=normalized_search_mode,
    )
    selected_subcategories = _selected_advertised_subcategories(
        request.taxonomy,
        capabilities,
    )
    plan = build_catalog_search_plan(
        intent,
        capabilities,
        has_image=bool(ctx.state.image),
        top_k=_multi_subcategory_candidate_limit(
            selected_subcategories,
            capabilities,
            ctx.config.top_k_retrieve,
        ),
    )
    # A partly-honoured enum joins the requirements the model already declared
    # unconfirmable, rather than getting a disclosure channel of its own: it is
    # exactly that -- something the catalog cannot confirm, ranked and disclosed
    # instead of vetoing the search.
    if plan.partial_constraints:
        existing = list(attempt.unconfirmable_requirements or [])
        attempt.unconfirmable_requirements = existing + [
            item for item in plan.partial_constraints if item not in existing
        ]
    if not plan.should_search:
        if plan.constraint_issues:
            return _rejected(
                attempt,
                SearchRejection.UNSUPPORTED_CATALOG_CONSTRAINT,
                "The requested catalog requirement cannot be enforced: "
                + "; ".join(plan.constraint_issues)
                + ". Ask the shopper to relax it or use an advertised filter.",
            )
        if plan.no_search_reason == "image_search_unavailable":
            return (
                "Image search is not available for the active catalog. "
                "Ask the shopper to describe what they want to find."
            )
        if plan.no_search_reason == "unsupported_search_mode":
            return _rejected(
                attempt,
                SearchRejection.UNSUPPORTED_SEARCH_MODE,
                _UNSUPPORTED_SEARCH_MODE_MESSAGE,
            )
        if plan.no_search_reason == "missing_image_for_search_mode":
            return (
                "That search mode requires an attached image. Ask the shopper "
                "to attach one or use text search."
            )
        return "Catalog search requires a query or image."

    attempt.plan = plan
    attempt.selected_subcategories = selected_subcategories
    attempt.taxonomy_constraints = taxonomy_constraints
    attempt.taxonomy_fields = taxonomy_fields
    return None


def _scope_answer_keys(attempt: _Attempt) -> list[str]:
    """The keys a later repeat of this scope could arrive under.

    Two, because a repeat comes in two shapes and they are not the same key.
    The shopper key is the role as the shopper asked for it, which catches the
    same role being asked twice. The catalog key is the taxonomy and
    constraints actually sent, which catches one query reworded into another.
    """

    keys: list[str] = []
    if attempt.shopper_scope_key is not None:
        keys.append("shopper:" + json.dumps(attempt.shopper_scope_key))
    if attempt.catalog_scope is not None:
        keys.append(
            "catalog:" + json.dumps(attempt.catalog_scope, sort_keys=True, default=str)
        )
    return keys


def _remember_what_this_scope_answered(
    ctx: SearchContext,
    attempt: _Attempt,
    outcome: StepResult,
) -> None:
    """Keep this scope's answer, so a repeat of it is answered from here.

    Every scope that finished retrieval is recorded, including one that found
    nothing, and the empty case earns its place by telling two situations
    apart. A repeat can arrive after the first search returned zero, or while
    the first search is still running -- two threads reaching the same scope
    together, which the turn lock serialises but does not prevent. Without a
    record for the empty case both look identical, and the honest answer for
    one ("nothing matched") is a false statement about the other, whose
    products are on their way.

    A relaxed retry is unaffected. It carries different constraints, so it is
    a different catalog scope with a different key, and it searches.
    """

    result = attempt.result
    if outcome is None or result is None or not getattr(result, "ok", False):
        return
    with ctx.scope.catalog_lock:
        for key in _scope_answer_keys(attempt):
            ctx.scope.answered_scopes.setdefault(key, outcome)


def _already_answered(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """This scope's earlier answer, if it has one."""

    for key in _scope_answer_keys(attempt):
        answer = ctx.scope.answered_scopes.get(key)
        if answer is not None:
            return answer
    return None


def _reserved_search_slot(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Claim this turn's budget for the search, under the turn lock.

    Reserving before executing is what makes the per-turn cap hold when tool
    calls overlap; checking and incrementing separately would let two searches
    both observe the last remaining slot.
    """

    normalized_constraints = attempt.normalized_constraints
    search_budget_exhausted = attempt.search_budget_exhausted
    shopper_scope_key = attempt.shopper_scope_key
    taxonomy_constraints = attempt.taxonomy_constraints

    with ctx.scope.catalog_lock:
        search_scope = _catalog_search_scope(
            taxonomy_constraints,
            normalized_constraints,
        )
        attempt.catalog_scope = search_scope
        # Judged against the scopes earlier calls searched, never against a
        # sibling in this one. The rule exists to stop a retry paraphrasing an
        # answered search; two roles of one call are not retries of each other,
        # and refusing the second killed half a request that was correctly
        # formed -- "black crew neck, or any black one under $60" lost its
        # fallback to its own first half. An identical sibling is still caught
        # below, by taxonomy and constraints, where the comparison is exact.
        already_searched = (
            attempt.prior_shopper_scopes
            if attempt.prior_shopper_scopes is not None
            else ctx.scope.searched_shopper_scopes
        )
        # A scope asked for twice is answered twice, from what it found the
        # first time. Both of these used to be refusals reading "use the
        # result already returned" -- advice about data, in place of the data,
        # and the model cannot act on advice about products it was not given.
        # So it asked again, was told again, and J02 turn 4 spent 23 identical
        # searches and the graph's whole recursion budget on the word "shoes".
        #
        # Serving the answer costs one dictionary lookup and no retrieval, and
        # leaves the model nothing to retry: it has the products.
        repeated = (
            shopper_scope_key is not None
            and shopper_scope_key in already_searched
        ) or search_scope in ctx.scope.searched_catalog_scopes
        if repeated:
            answer = _already_answered(ctx, attempt)
            if answer is not None:
                return answer
            # Claimed and not yet finished, which is two threads arriving at
            # the same scope together. Its products land in this turn's
            # evidence either way, so this says only what is known to be true
            # and leaves nothing to retry.
            return (
                "SEARCH_SCOPE_ALREADY_RUNNING: this exact scope is already "
                "being searched in this turn, and whatever it finds is part "
                "of this turn's evidence. Do not send it again. Answer from "
                "this turn's evidence once it is complete."
            )
        if ctx.scope.catalog_searches >= ctx.config.max_catalog_searches_per_turn:
            return _rejected(
                attempt,
                SearchRejection.CATALOG_SEARCH_LIMIT,
                control(
                    "STOP_TOOL_USE: Catalog search limit reached for this turn. "
                    "Do not call more tools this turn. Use the products already "
                    "returned in this turn to answer concisely, or ask one concise "
                    "clarifying question if the available products are not enough.",
                    ControlSignal.STOP_TOOL_USE,
                ),
            )
        ctx.scope.searched_catalog_scopes.append(search_scope)
        if shopper_scope_key is not None:
            # Recorded now so siblings and retries see it, and withdrawn later
            # if the search found nothing. The duplicate rule exists to stop a
            # retry paraphrasing an *answered* search; a scope that returned
            # zero has not been answered, and refusing the relaxed retry left
            # "no green dress in a 2" with nothing to show but a menu.
            ctx.scope.searched_shopper_scopes.add(shopper_scope_key)
        ctx.scope.catalog_searches += 1
        search_budget_exhausted = (
            ctx.scope.catalog_searches
            >= ctx.config.max_catalog_searches_per_turn
        )

    attempt.search_budget_exhausted = search_budget_exhausted
    return None



def _executed_search(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Run the retrieval and record what it cost and returned."""

    plan = attempt.plan
    selected_subcategories = attempt.selected_subcategories

    search_start = time.monotonic()
    execution = execute_catalog_search(
        plan,
        ctx.config.retriever_port,
        image_base64=ctx.state.image,
        timeout_seconds=ctx.config.catalog_search_timeout_seconds,
    )
    catalog_elapsed = time.monotonic() - search_start
    result = execution.result
    if result.ok:
        result = result.model_copy(
            update={
                "products": _products_with_subcategory_coverage(
                    result.products,
                    selected_subcategories,
                    ctx.config.top_k_retrieve,
                )
            }
        )
    with ctx.scope.catalog_lock:
        ctx.state.timings["catalog_search"] = max(
            ctx.state.timings.get("catalog_search", 0.0),
            catalog_elapsed,
        )
        _record_catalog_model_usage(
            ctx.state,
            plan,
            result.ok,
            fallback_attempted=execution.fallback_attempted,
        )
    if not result.ok:
        return result.error.message if result.error else "Catalog search failed."

    attempt.execution = execution
    attempt.result = result
    return None


def _published_in_plan_order(
    ctx: SearchContext, attempts: list[_Attempt]
) -> None:
    """Record what was found in the order the roles were asked for.

    This used to run inside each retrieval, so the products reached the shopper
    in whatever order the scopes happened to finish. Retrieval fans out across
    a thread pool, so that order is not stable: the same request for a sweater
    and boots arrived grouped on three runs and interleaved on a fourth.

    Only the shopper's screen was affected -- the evidence the model reads has
    always been rendered from `attempts`, in plan order, as `SCOPE 1`,
    `SCOPE 2`. So the model described products in one order while the pictures
    beside its words sat in another, and "the first one" meant two different
    garments depending on which the shopper counted. Publishing here, from the
    same list the renderer uses, is what makes the two agree.
    """

    with ctx.scope.catalog_lock:
        for attempt in attempts:
            result = attempt.result
            if result is None or not result.ok or not result.products:
                continue
            ctx.scope.product_evidence.add(result.products)
            _append_product_results(ctx.state, result.products)
            for product in result.products:
                if product.image_url:
                    ctx.scope.retrieved[product.display_name] = product.image_url


def _rendered_evidence(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Render what was found as the evidence the model is allowed to speak from.

    The payload is built first and every line is rendered from it, so the text
    the model reads and the artifact a later turn recovers cannot disagree.
    """

    evidence = attempt.evidence
    execution = attempt.execution
    lines = attempt.lines
    plan = attempt.plan
    request = attempt.request
    result = attempt.result
    search_budget_exhausted = attempt.search_budget_exhausted
    taxonomy_constraints = attempt.taxonomy_constraints
    taxonomy_fields = attempt.taxonomy_fields
    unconfirmable_requirements = attempt.unconfirmable_requirements

    confirmed_filters = {
        name: value
        for name, value in plan.hard_filters.items()
        if name not in taxonomy_fields
    }
    # One derivation of the parent-scope fact: the payload carries it and
    # the model-visible line is rendered from the same value.
    advertised_category = (
        request.taxonomy.category[0]
        if request.taxonomy_status == "parent_category_alternative"
        and request.taxonomy.category
        else None
    )
    # What that parent actually holds. "Apparel" is true of every garment, so
    # the category alone cannot tell the shopper whether their kind is here;
    # its advertised subcategories can, and they are already published.
    advertised_subcategories = (
        sorted(

                getattr(
                    (attempt.capabilities.taxonomy.categories or {}).get(
                        advertised_category
                    ),
                    "subcategories",
                    None,
                )
                or {}

        )
        if advertised_category and attempt.capabilities.taxonomy
        else []
    )
    role_advertised_types = (
        list(request.taxonomy.subcategory or [])
        if attempt.composed_role
        else []
    )
    # The shopper named a type, it is not one this catalog lists, and the model
    # answered with advertised types of its own choosing. A scope naming real
    # subcategories was indistinguishable from a direct search for the thing
    # asked for, so "jeans" came back as skirts with nothing saying so.
    substituted_types = (
        [str(value) for value in (request.taxonomy.subcategory or [])]
        if (
            not advertised_category
            and not attempt.composed_role
            and attempt.shopper_stated_scope
            and request.requested_product_type
            and request.taxonomy.subcategory
            and _advertised_scope_match(
                request.requested_product_type,
                attempt.capabilities,
            )
            is None
        )
        else []
    )
    substituted_within = (
        sorted(

                getattr(
                    (attempt.capabilities.taxonomy.categories or {}).get(
                        (request.taxonomy.category or [None])[0]
                    ),
                    "subcategories",
                    None,
                )
                or {}

        )
        if substituted_types and attempt.capabilities.taxonomy
        else []
    )
    scope_relation_evidence = (
        _format_search_scope_relation_evidence(
            requested_product_type=request.requested_product_type or "",
            advertised_category=advertised_category,
            advertised_subcategories=advertised_subcategories,
        )
        if advertised_category
        else _format_search_unadvertised_type_evidence(
            requested_product_type=request.requested_product_type or "",
            searched_types=substituted_types,
            advertised_subcategories=substituted_within,
        )
        if substituted_types
        else (
            _format_search_composed_role_evidence(
                requested_product_type=request.requested_product_type or "",
                role_advertised_types=role_advertised_types,
            )
            if attempt.composed_role
            else ""
        )
    )
    if not result.products:
        # An empty scope is not an answered one: let a relaxed retry through,
        # while the exact taxonomy-and-filters check below still refuses a
        # literal repeat.
        if attempt.shopper_scope_key is not None:
            with ctx.scope.catalog_lock:
                ctx.scope.searched_shopper_scopes.discard(
                    attempt.shopper_scope_key
                )
        evidence = SearchEvidence(
            outcome="zero_results",
            taxonomy=taxonomy_constraints,
            confirmed_filters=confirmed_filters,
            requested_product_type=request.requested_product_type,
            advertised_category=advertised_category,
            composed_role=attempt.composed_role,
            role_advertised_types=role_advertised_types,
            scope_complete=bool(request.scope_complete),
            budget_exhausted=bool(search_budget_exhausted),
            unconfirmed_requirements=unconfirmable_requirements,
            size_the_scope_has_not=dict(attempt.size_the_scope_has_not or {}),
            scope_outcome={
                "outcome": "zero_results",
                "requested_product_type": request.requested_product_type,
                "taxonomy": taxonomy_constraints,
                "confirmed_filters": confirmed_filters,
                # A role nobody named that found nothing is the case the
                # disclosure exists for, and the one an operator most needs to
                # see: without it a zero-result composed role is indistinguishable
                # from a shopper asking for something the catalog lacks.
                "composed_role": attempt.composed_role,
            },
        )
        lines = [
            _SEARCH_NO_MATCH_GROUNDING_NOTE,
            _format_search_taxonomy_evidence(evidence.taxonomy),
        ]
        # A scope that found nothing and had a word set aside is the case this
        # disclosure exists for: "we have no cream sweaters" and "cream is not
        # a colour here" are different answers, and only one of them is true.
        set_aside_note = _format_words_this_catalog_cannot_filter_on(
            attempt.set_aside
        )
        if set_aside_note:
            lines.append(set_aside_note)
        colour_note = _format_colour_words_read_as_advertised_ones(
            attempt.colours_mapped
        )
        if colour_note:
            lines.append(colour_note)
        if scope_relation_evidence:
            lines.append(scope_relation_evidence)
        if evidence.confirmed_filters:
            lines.append(
                _format_search_filter_evidence(evidence.confirmed_filters)
            )
        lines.append(_format_catalog_scope_outcome(evidence.scope_outcome))
        if evidence.unconfirmed_requirements:
            lines.append(
                _unsupported_requirement_message(
                    evidence.unconfirmed_requirements
                )
            )
        # Scope-complete says "you have what you need, answer now". That is
        # false when nothing came back and two filters were combined: the
        # honest next move is to relax one and look again. Emitting both left
        # the model with "search again" and "do not search again" in one
        # message, and the older, blunter rule won -- three runs answered with
        # a numbered menu and showed nothing.
        # Any filter at all can be relaxed and looked at again. Requiring two
        # left the single-filter cases stranded: "a tote bag in a size 8" is
        # unanswerable as asked -- bags are one size -- and returned a
        # clarifying question with nothing to look at. Dropping a filter
        # silently is still forbidden; the evidence requires saying which one
        # went.
        relaxable = bool(evidence.confirmed_filters)
        if evidence.scope_complete and not relaxable:
            lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
        elif evidence.budget_exhausted:
            lines.append(_SEARCH_BUDGET_EXHAUSTED_NOTE)
        return "\n\n".join(lines), evidence.as_artifact()

    evidence = SearchEvidence(
        outcome="results",
        taxonomy=taxonomy_constraints,
        confirmed_filters=confirmed_filters,
        semantic_query=request.semantic_query,
        shopper_guidance=_safe_shopper_guidance(
            request.shopper_guidance,
            request.requested_product_type,
        ),
        requested_product_type=request.requested_product_type,
        advertised_category=advertised_category,
        composed_role=attempt.composed_role,
        role_advertised_types=role_advertised_types,
        scope_complete=bool(request.scope_complete),
        budget_exhausted=bool(search_budget_exhausted),
        unconfirmed_requirements=unconfirmable_requirements,
        size_the_scope_has_not=dict(attempt.size_the_scope_has_not or {}),
        products=[
            _search_product_record(product) for product in result.products
        ],
    )
    evidence.assumed_audience = _assumed_audience(
        str(getattr(ctx.config, "wearer_audience_field", "") or ""),
        confirmed_filters,
        evidence.products,
        already_disclosed=list(getattr(ctx.state, "assumed_audience", None) or []),
    )
    for value in evidence.assumed_audience:
        if value not in ctx.state.disclosed_audience:
            ctx.state.disclosed_audience.append(value)
    lines = [
        _SEARCH_RESULT_GROUNDING_NOTE,
        _format_search_direction_evidence(evidence.semantic_query),
        _format_search_guidance_evidence(evidence.shopper_guidance),
        _format_search_taxonomy_evidence(evidence.taxonomy),
    ]
    # Results that came back ranked on a word rather than filtered by it are
    # not the promise a filter makes, and the difference travels with them.
    set_aside_note = _format_words_this_catalog_cannot_filter_on(
        attempt.set_aside
    )
    if set_aside_note:
        lines.append(set_aside_note)
    colour_note = _format_colour_words_read_as_advertised_ones(
        attempt.colours_mapped
    )
    if colour_note:
        lines.append(colour_note)
    if scope_relation_evidence:
        lines.append(scope_relation_evidence)
    if evidence.confirmed_filters:
        lines.append(
            _format_search_filter_evidence(evidence.confirmed_filters)
        )
        # A category the shopper never named, reached because a filter scoped
        # the search. It is shown rather than refused -- a partial answer beats
        # "could you clarify", which is what three runs in five used to get --
        # and it is said out loud, because the shopper asked for everything
        # under their ceiling and this is one department of it.
        chosen = _a_category_the_shopper_did_not_name(evidence, attempt)
        if chosen:
            lines.append(
                f"CATEGORY CHOSEN FOR THEM: the shopper named no product type, "
                f"so these are {chosen} only. Say so, and offer the other "
                "departments."
            )
    if evidence.unconfirmed_requirements:
        lines.append(
            _unsupported_requirement_message(
                evidence.unconfirmed_requirements
            )
        )
    if evidence.scope_complete:
        lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
    elif evidence.budget_exhausted:
        lines.append(_SEARCH_BUDGET_EXHAUSTED_NOTE)
    for record in evidence.products:
        lines.append(_format_product_record(record))
    if evidence.products:
        lines.append(SEARCH_RESULT_ATTRIBUTE_LIMIT_NOTE)
    prefix = (
        "Image similarity returned no matches; text fallback results:\n\n"
        if execution.fallback_used
        else ""
    )
    return prefix + "\n\n".join(lines), evidence.as_artifact()


def _assumed_audience(
    field_name: str,
    confirmed_filters: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    already_disclosed: list[str] | None = None,
) -> list[str]:
    """Who the returned pieces are for, on a search that never asked.

    An unfiltered search still comes back with an audience -- a catalog that is
    mostly womenswear returns womenswear whatever the shopper said -- and the
    shopper is the one party who cannot see that nobody chose it. Once the
    audience is a confirmed filter it is the shopper's own constraint, so there
    is nothing assumed and this returns nothing.

    Read off the products rather than the catalog's capabilities: the values
    that came back are the ones the reply is about, and a catalog with a
    different range therefore discloses a different audience without a code
    change.

    A conversation already told stays told. The trigger is true on nearly every
    turn, so without this three consecutive replies opened "assuming you're
    looking for women's clothes" -- which stops being a disclosure and becomes
    a tic the shopper has to read past.
    """

    if already_disclosed:
        return []
    if not field_name or field_name in (confirmed_filters or {}):
        return []
    values: list[str] = []
    for record in products:
        attributes = record.get("attributes")
        if not isinstance(attributes, dict):
            continue
        text = str(attributes.get(field_name) or "").strip()
        if text and text not in values:
            values.append(text)
    return values[:8]


#: Everything a scope decides before it touches the network. Each step either
#: ends that scope -- returning the text the model reads -- or leaves what it
#: worked out on the attempt for the next one.
def _resolved_against_the_catalogue(
    ctx: SearchContext,
    attempt: _Attempt,
) -> StepResult:
    """Put this role where the judge says it lives. One pass, then search.

    A second pass over the model's request, and the last word on it. The model
    says what the shopper asked for; the judge, which holds the catalogue's
    vocabulary, says which advertised subcategories that is. "Shoes" becomes
    flats, heels, sandals and boots. "Pumps" becomes heels. "Jeans" becomes
    nothing, and nothing is the answer -- this shop does not sell them.

    What the model scoped the role to is overwritten rather than checked, and
    that is the point. Checking produces a verdict about a guess, a verdict has
    to be reported, and a report the model can read is an invitation to guess
    again: told jeans are not skirts it tried jumpsuits, then dresses, blouses,
    camisoles and sweaters, and one turn spent twelve searches walking the enum.
    Overwriting leaves nothing to report and nothing to resubmit, so the twelve
    searches have no shape to take.

    Three outcomes, all of them final:

        named        search those subcategories
        named empty  answer the shopper: not carried
        unjudged     search what the model sent, unchanged

    The third is the degraded path and it trusts the model, because the
    alternative is worse in both directions. An unreachable judge is not
    evidence about what the shop stocks, so "we do not carry jeans" cannot be
    said on it -- that would turn a timeout into a fact. Nor can the role be
    narrowed to an exact name match, which reads as caution and is not: it
    refuses "a top", "shoes" and every composed role the moment the endpoint
    hiccups, and silently returns nothing for searches that worked all day.

    So an outage costs accuracy on the cases the judge was added for -- a
    substitution can reach the shopper again while it lasts -- and costs
    nothing on the ordinary ones. That is the trade the old gates made too,
    with more code and worse results.
    """

    named = attempt.judged_subcategories
    if named is None:
        return None
    if named:
        attempt.taxonomy = _scoped_to(attempt.taxonomy, named, ctx.capabilities)
        return None
    return _role_this_shop_does_not_carry(ctx, attempt)


def _scoped_to(
    taxonomy: BaseModel | dict[str, Any],
    subcategories: Sequence[str],
    capabilities: CatalogCapabilities,
) -> dict[str, Any]:
    """This role's scope, as the judge named it.

    The categories are derived from the subcategories rather than kept from
    what the model sent, because the judge can move a role across a category
    boundary -- "shoes" sent as apparel resolves into footwear -- and a scope
    carrying the old category alongside the new subcategories filters to their
    intersection, which is empty.
    """

    payload = (
        taxonomy.model_dump()
        if isinstance(taxonomy, BaseModel)
        else dict(taxonomy or {})
    )
    payload["subcategory"] = list(dict.fromkeys(subcategories))
    payload["category"] = [
        name
        for name, category in capabilities.taxonomy.categories.items()
        if any(value in category.subcategories for value in payload["subcategory"])
    ]
    return payload


def _role_this_shop_does_not_carry(
    ctx: SearchContext,
    attempt: _Attempt,
) -> StepResult:
    """Report a garment this shop does not sell, as this role's answer.

    Returned as text rather than through `_rejected`, and the distinction is
    the whole redesign in one line. A rejection arms the turn's repair locks,
    whose single slot then belongs to this role, so every *other* role in the
    call comes back `repair_changed_product_scope`: refusing the hat refused
    the boots and the sweaters beside it, no legal move remained, and the turn
    died on the graph's recursion limit with the shopper told to retry.

    There is also nothing to repair. "This shop does not sell jeans" is not a
    malformed argument, has no corrected form, and must leave the roles around
    it alone. So it travels with them, as the finding for its own scope.
    """

    word = str(attempt.requested_product_type or "").strip()
    # Collected into the call's NOT_CARRIED notice and delivered beside the
    # roles that did find something, so a whole-look turn answers "sweaters
    # found, boots found, jeans not carried" -- one complete answer rather than
    # a partial failure.
    attempt.not_carried = word
    return (
        f"SCOPE_CLOSED: '{word}' is not sold here. No advertised subcategory "
        "is that garment, so no taxonomy would make this scope true and there "
        "is no correction to make. Tell the shopper plainly, answer with what "
        "the other roles found, and do not offer another garment in its place "
        "or present one as the closest version of it."
    )


def _advertised_subcategories(capabilities: CatalogCapabilities) -> frozenset[str]:
    """Every subcategory this catalog advertises, across all categories."""

    return frozenset(
        name
        for category in capabilities.taxonomy.categories.values()
        for name in category.subcategories
    )


def _scope_members(
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[list[str], bool]:
    """What this scope will actually search, and whether it named a whole category.

    A scope may name its subcategories or it may name only a category, and the
    second is a legal text search across everything in it. Judging the first
    while ignoring the second left the parent category open as an escape: told
    jeans are not skirts, the model sent `category: [apparel], subcategory: []`
    and ranked "dark blue jeans" against all of apparel, and two blouses and two
    dresses came back as the shopper's dark bottom.

    So a category-only scope is judged on the members it is about to search.
    That also mends the opposite case rather than only blocking this one: asked
    for pumps across all of footwear, the members that are pumps are the heels,
    and the scope narrows to them instead of returning boots and sandals too.
    """

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else dict(taxonomy or {})
    selected = list(dict.fromkeys(payload.get("subcategory") or []))
    if selected:
        return selected, False

    members: list[str] = []
    for name in dict.fromkeys(payload.get("category") or []):
        category = capabilities.taxonomy.categories.get(name)
        if category:
            members.extend(category.subcategories)
    return list(dict.fromkeys(members)), True


_PLAN_STEPS = (
    # First, because every step after it is entitled to a request this catalog
    # can actually answer. A word it cannot filter on is set aside here rather
    # than refused six steps later.
    _reconciled_with_what_is_advertised,
    _admit_search,
    # The catalogue's own answer to where this role goes, and the last word on
    # it. Before the validation below, which now only ever sees a scope this
    # step wrote.
    _resolved_against_the_catalogue,
    _classify_requirements,
    _validated_request,
    _reviewed_provenance,
    _no_direct_match_outcome,
    _planned_search,
)

#: Claiming the turn's budget, then the one step that does I/O.
_RESERVE_STEP = _reserved_search_slot
_EXECUTE_STEP = _executed_search
_RENDER_STEP = _rendered_evidence


def _planned_scope(ctx: SearchContext, attempt: _Attempt) -> StepResult:
    """Run one scope up to the point of retrieval, touching no network.

    Returning early here costs nothing: no budget is spent and no request is
    sent, so a scope that cannot run never takes anything from the scopes that
    can.
    """

    for step in _PLAN_STEPS:
        outcome = step(ctx, attempt)
        if outcome is not None:
            return outcome
    return None


def _a_category_the_shopper_did_not_name(evidence: Any, attempt: Any) -> str:
    """The department this browse narrowed to without being asked.

    "Nothing over $50" is a request for everything under a ceiling. A search
    that answers it with one category is answering a fraction, and the shopper
    cannot tell from the products alone that a choice was made for them.
    """

    if getattr(attempt, "taxonomy_status", "") != "agent_selected_type":
        return ""
    taxonomy = getattr(evidence, "taxonomy", None) or {}
    if not isinstance(taxonomy, dict):
        return ""
    categories = [
        str(value)
        for values in taxonomy.values()
        if isinstance(values, list)
        for value in values
    ]
    return categories[0] if len(categories) == 1 else ""


def _hard_filter_scopes_this(required_constraints: Any) -> bool:
    """Whether an advertised filter narrows the search on its own.

    unadvertised_requirements is excluded: it is the field for what the catalog
    cannot enforce, so it scopes nothing.
    """

    constraints = (
        required_constraints.model_dump(exclude_none=True)
        if isinstance(required_constraints, BaseModel)
        else required_constraints
    )
    if not isinstance(constraints, dict):
        return False
    return any(
        name != "unadvertised_requirements" and value not in (None, "", [], {})
        for name, value in constraints.items()
    )


def _one_scope_per_category(ctx: SearchContext, scopes: list[Any]) -> list[Any]:
    """Fan a several-category browse into one scope each, subcategories filled.

    "I'm shopping on a tight budget, nothing over $50" names no category and no
    product type, so every category is in scope. The model asks for exactly
    that -- all five categories with a price ceiling -- and the schema allows
    one category per scope, so the call is turned back and the shopper is asked
    to clarify a request that was already complete.

    Each split scope carries the category's own advertised subcategories. That
    is not a workaround: it is what the refusal asks for in as many words --
    "for a role the shopper did not name, select every advertised subcategory
    that role covers". Filling them from the catalog satisfies the open-role
    rule rather than relaxing it, so a model that invents a role ("loungewear")
    and names one category is still turned back as it should be.

    Left alone when the split will not fit the call's scope budget, and when
    only one category was asked for.
    """

    limit = max(1, int(getattr(ctx.config, "max_search_scopes_per_call", 1) or 1))
    categories = getattr(getattr(ctx.capabilities, "taxonomy", None), "categories", None) or {}

    fanned: list[Any] = []
    for raw in scopes:
        fields = raw if isinstance(raw, dict) else raw.model_dump()
        taxonomy = fields.get("taxonomy") or {}
        asked = taxonomy.get("category") if isinstance(taxonomy, dict) else None
        named = list(taxonomy.get("subcategory") or []) if isinstance(taxonomy, dict) else []
        # Two or more categories only. One category with no subcategory is
        # indistinguishable from an invented role: "rainy outfit under $60"
        # arrives as apparel with a price and nothing else, and filling in
        # every apparel subcategory would answer it with the whole department.
        # The open-role rule exists for exactly that and is left alone.
        #
        # Naming five categories is not a role. It is "everything", and that is
        # the shape this fans out.
        if named or not isinstance(asked, list) or len(asked) < 2:
            fanned.append(raw)
            continue
        if len(scopes) - 1 + len(asked) > limit:
            fanned.append(raw)
            continue
        expanded: list[Any] = []
        for category in asked:
            advertised = categories.get(str(category))
            subcategories = sorted(getattr(advertised, "subcategories", None) or {})
            if not subcategories:
                expanded = []
                break
            expanded.append(
                {
                    **fields,
                    "taxonomy": {
                        **taxonomy,
                        "category": [category],
                        "subcategory": subcategories,
                    },
                }
            )
        fanned.extend(expanded or [raw])
    return fanned


def _umbrella_candidates(
    payload: dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[str, ...]:
    """The advertised subcategories a word this catalog lacks might cover.

    Narrowed to the categories the scope named, so "shoes" under `footwear` is
    asked about footwear rather than about every subcategory in the shop. A
    scope naming no category has nothing to narrow by and is asked about all
    of them, which is the honest reading of a request that named none.
    """

    named = [
        name
        for name in dict.fromkeys(payload.get("category") or [])
        if name in capabilities.taxonomy.categories
    ]
    if named:
        members: list[str] = []
        for name in named:
            members.extend(capabilities.taxonomy.categories[name].subcategories)
        return tuple(dict.fromkeys(members))
    return tuple(sorted(_advertised_subcategories(capabilities)))


def _colour_field(ctx: SearchContext) -> str:
    """The catalog's colour filter, by name from config rather than baked in.

    Named the same way as the audience field and for the same reason: the field
    name is this deployment's, the values are the catalog's. A shop calling it
    `colour` sets one line.
    """

    return str(getattr(ctx.config, "colour_field", "primary_color") or "")


def _colour_words_not_advertised(
    ctx: SearchContext,
    attempt: _Attempt,
    advertised: list[str],
) -> list[str]:
    """The colour words this scope asked to filter on and this catalog lacks.

    Read here because here is the only place it can be read: the step that
    sets an unhonourable value aside is the first of the plan steps, and by the
    time it has run the word is gone from the constraints.

    Gated on membership rather than sending every colour the scope offered.
    Measured over 3,040 scopes carrying a colour filter, 95.6% named advertised
    values throughout -- the scope prompt asks the model to do this mapping and
    it mostly does. Asking about those would bolt dead words onto almost every
    search call to serve the 4% that need it.
    """

    if not advertised:
        return []
    constraints = attempt.required_constraints
    constraints = (
        constraints.model_dump(exclude_none=True)
        if isinstance(constraints, BaseModel)
        else dict(constraints or {})
    )
    offered = constraints.get(_colour_field(ctx))
    if offered is None:
        return []
    if not isinstance(offered, (list, tuple)):
        offered = [offered]
    known = {value.casefold() for value in advertised}
    return [
        str(value)
        for value in offered
        if str(value).strip() and str(value).casefold() not in known
    ]


def _judge_this_call(ctx: SearchContext, attempts: list[_Attempt]) -> None:
    """Ask the vocabulary question once for every role in this call.

    Batched deliberately. A shopper saying "I want to shop this look" sends
    three roles in one call, and asking per role would triple both the latency
    and the token cost of a question that fits in one request -- thirteen scopes
    and four colour words together measured 3.7s and roughly 400 tokens, against
    a 47,753-token median turn.

    Nothing is raised out of here. A judge that cannot be reached leaves every
    scope unruled, and an unruled scope is decided by the gates that decided it
    before this existed.
    """

    judge = getattr(ctx, "vocabulary_judge", None)
    if judge is None:
        return

    # One question per distinct word the shopper asked for. What the model
    # scoped it to is not sent and not needed: the judge names the
    # subcategories itself, so there is no guess to grade and no shape to read
    # the answer back in. Deduplicated because two roles of one call often
    # share a word, and asking twice paid twice for the same lookup.
    questions = [
        ScopeQuestion(word)
        for word in dict.fromkeys(
            str(attempt.requested_product_type).strip()
            for attempt in attempts
            if attempt.requested_product_type
        )
        if word
    ]

    capability = ctx.capabilities.filters.get(_colour_field(ctx))
    advertised_colours = [
        str(value) for value in (getattr(capability, "values", None) or ())
    ]
    colour_words = sorted(
        {
            word
            for attempt in attempts
            for word in _colour_words_not_advertised(ctx, attempt, advertised_colours)
        }
    )

    # A colour word alone is worth the call. Requiring a taxonomy question too
    # would let the one case the mapping exists for decide whether the mapping
    # runs: "show me it in cream" against a scope carried over from the
    # previous turn asks nothing about taxonomy.
    if not questions and not colour_words:
        return

    subcategory_names = sorted(_advertised_subcategories(ctx.capabilities))
    verdict = judge.judge(
        questions,
        colour_words,
        subcategories=subcategory_names,
        colours=advertised_colours,
    )
    if verdict.unavailable:
        return

    for attempt in attempts:
        # Before the taxonomy reading below, whose `continue`s would otherwise
        # skip it. Keyed on the word as this scope sent it: the reply is the
        # turn's, but a scope only ever looks up a word it dropped itself, so a
        # call carrying "black boots" beside "cream sweater" cannot pick up the
        # other's answer.
        attempt.colour_map = {
            word.casefold(): verdict.colours_for(word)
            for word in _colour_words_not_advertised(ctx, attempt, advertised_colours)
        }
        if attempt.requested_product_type:
            attempt.judged_subcategories = verdict.subcategories_for(
                str(attempt.requested_product_type)
            )


def search_catalog(
    ctx: SearchContext,
    scopes: list[dict[str, Any]],
    scope_complete: bool = True,
    search_mode: str | None = None,
    not_covered: list[str] | None = None,
):
    """Execute one catalog search per product role, concurrently.

    A shopper asking for "a dress, shoes and a bag" is asking three questions.
    Answering them one call at a time cost three model round trips at roughly
    8.7s each while the retrievals themselves take under a second -- measured
    across one conversation, retrieval was 3.1% of the elapsed time and round
    trips were the rest.

    So the scopes are planned first, with no I/O, and only the ones that survive
    planning retrieve. Those go out together: ten concurrent retrievals measured
    1.58s against 0.61s for one.

    Each scope keeps its own filters, so a `heel_type` chosen for the shoes
    cannot delete the bags -- which is what a shared filter did, returning eight
    heels and no clutches for a two-category search.
    """

    scopes = _one_scope_per_category(ctx, list(scopes))
    attempts: list[_Attempt] = []
    for raw in scopes:
        fields = raw if isinstance(raw, dict) else raw.model_dump()
        attempt = _Attempt(
            semantic_query=fields.get("semantic_query", ""),
            requested_product_type=fields.get("requested_product_type"),
            taxonomy=fields.get("taxonomy") or {},
            required_constraints=fields.get("required_constraints") or {},
            shopper_guidance=fields.get("shopper_guidance", ""),
            scope_complete=bool(fields.get("scope_complete", scope_complete)),
            search_mode=fields.get("search_mode", search_mode),
        )
        # Repair bookkeeping belongs to the product scope, not to the call, so a
        # repair still spans tool calls while one rejected role cannot lock out
        # another.
        # One scope sees the turn's repair state exactly as it always has, so a
        # single-scope call behaves identically to before. Several scopes each
        # plan against a snapshot of it, so one rejected role cannot lock out
        # another, and their mutations are merged once planning is done.
        attempt.repair = ctx.scope.repair
        attempts.append(attempt)

    _judge_this_call(ctx, attempts)

    if len(attempts) > 1:
        # Per-scope purity: each scope is judged against the repair state as it
        # stood at the start of the call, never against what a sibling scope
        # just wrote. Otherwise the same call would give different answers
        # depending on the order the model happened to list the roles.
        baseline = deepcopy(ctx.scope.repair)
        for attempt in attempts:
            attempt.repair = deepcopy(baseline)

    outcomes: list[StepResult] = [_planned_scope(ctx, a) for a in attempts]
    if len(attempts) > 1:
        for attempt in attempts:
            _merge_repair(ctx.scope.repair, attempt.repair)
            attempt.repair = ctx.scope.repair

    runnable = [i for i, outcome in enumerate(outcomes) if outcome is None]

    # The product budget is shared, so more roles mean fewer products each
    # rather than a larger reply.
    if runnable:
        share = max(
            3,
            min(
                int(getattr(ctx.config, "top_k_retrieve_broad", 12) or 12),
                int(getattr(ctx.config, "search_products_per_call", 36) or 36)
                // len(runnable),
            ),
        )
        for index in runnable:
            plan = attempts[index].plan
            if plan is not None and getattr(plan, "top_k", None):
                attempts[index].plan = plan.model_copy(
                    update={"top_k": min(plan.top_k, share)}
                )

    # Reserving is sequential and lock-guarded; only retrieval fans out.
    prior_shopper_scopes = frozenset(ctx.scope.searched_shopper_scopes)
    for index in list(runnable):
        attempts[index].prior_shopper_scopes = prior_shopper_scopes
        outcome = _RESERVE_STEP(ctx, attempts[index])
        if outcome is not None:
            outcomes[index] = outcome
            runnable.remove(index)

    if len(runnable) == 1:
        outcomes[runnable[0]] = _EXECUTE_STEP(ctx, attempts[runnable[0]])
    elif runnable:
        with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
            for index, outcome in zip(
                runnable,
                pool.map(lambda i: _EXECUTE_STEP(ctx, attempts[i]), runnable),
                strict=True,
            ):
                outcomes[index] = outcome

    # After the fan-out and before anything is rendered, so what the shopper
    # sees is ordered by the plan rather than by which scope won the race.
    _published_in_plan_order(ctx, attempts)

    notices: list[str] = []
    # What the tool established itself, before anything the model volunteered:
    # a scope whose product type this catalog advertises nothing for. Its call
    # was rejected on its arguments, but "we do not carry aprons" is a current
    # fact about the catalog and the shopper is owed it either way.
    not_carried = [
        attempt.not_carried for attempt in attempts if attempt.not_carried
    ]
    if not_carried:
        notices.append(
            "NOT_CARRIED: this catalog advertises nothing of these kinds, so "
            "no search of it can succeed. Tell the shopper plainly that it is "
            "not carried: "
            + ", ".join(dict.fromkeys(not_carried))
            + ". " + _NO_STAND_IN
        )
    rendered: list[str] = list(notices)
    if not_covered:
        # The shopper asked for something no advertised category covers. It
        # costs no retrieval, but recording it is what stops the request being
        # silently dropped: without this the tool sees two scopes and cannot
        # know a third thing was asked for.
        rendered.append(
            "NOT_COVERED: this catalog carries nothing of these kinds, so they "
            "were not searched. Tell the shopper plainly rather than omitting "
            "them: "
            + ", ".join(str(item) for item in not_covered)
            + ". " + _NO_STAND_IN
        )
    artifacts: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        outcome = outcomes[index]
        if outcome is None:
            outcome = _RENDER_STEP(ctx, attempt)
        text, artifact = outcome if isinstance(outcome, tuple) else (outcome, None)
        role = attempt.requested_product_type or f"scope {index + 1}"
        rendered.append(f"SCOPE {index + 1} ({role}):\n{text}")
        if artifact:
            artifacts.append(artifact)
        _remember_what_this_scope_answered(ctx, attempt, outcome)

    codes = [attempt.rejection_code for attempt in attempts]
    if len(attempts) == 1:
        single = outcomes[0] if outcomes[0] is not None else _RENDER_STEP(ctx, attempts[0])
        if single is None:
            single = "Catalog search returned nothing."
        # A one-scope call took its own text and left `rendered` behind, so a
        # notice raised for that scope never reached the model.
        if notices:
            text, artifact = (
                single if isinstance(single, tuple) else (single, None)
            )
            single = ("\n\n".join([*notices, text]), artifact) if artifact else (
                "\n\n".join([*notices, text])
            )
        return _with_not_carried(
            _with_scope_rejections(single, codes),
            not_carried,
        )
    merged = _merged_artifacts(artifacts)
    text = "\n\n".join(rendered)
    return _with_not_carried(
        _with_scope_rejections((text, merged) if merged else text, codes),
        not_carried,
    )


def _with_not_carried(
    result: StepResult,
    not_carried: list[str],
) -> StepResult:
    """Carry the not-carried product types out on the artifact.

    A reader that has to tell "the arguments were wrong" from "the thing does
    not exist here" cannot do it from the rejection codes: both are the same
    schema mismatch. This is the distinction, recorded rather than inferred.
    """

    if not not_carried:
        return result
    text, artifact = result if isinstance(result, tuple) else (result, None)
    return text, {
        **(artifact or {}),
        NOT_CARRIED_KEY: list(dict.fromkeys(not_carried)),
    }


def _with_scope_rejections(
    result: StepResult,
    codes: list[str | None],
) -> StepResult:
    """Carry each scope's gate code out on the artifact, beside its own text.

    One tool call can now search several roles, so the codes are a list in
    scope order with ``None`` where a scope was not turned back. That is what
    lets a reader tell a call that refused every role -- and is therefore a
    refused call -- from one that refused a role and answered the rest.
    """

    if not any(codes):
        return result
    text, artifact = result if isinstance(result, tuple) else (result, None)
    return text, {**(artifact or {}), REJECTIONS_KEY: codes}


def _merged_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine several scopes' evidence into one payload of the same shape.

    Every consumer -- turn diagnostics, the grounding editor, and the durable
    presented-product record a later turn resolves against -- reads one evidence
    dict and checks `outcome`. An earlier version merged by key and produced a
    list of dicts, so those readers silently skipped it: a four-scope search
    completed, returned products, and recorded none of them. The shape is the
    contract, so merging must preserve it.
    """

    payloads = [a[EVIDENCE_KEY] for a in artifacts if a and EVIDENCE_KEY in a]
    if not payloads:
        return artifacts[0] if artifacts else None
    if len(payloads) == 1:
        return {EVIDENCE_KEY: payloads[0]}

    with_results = [p for p in payloads if p.get("outcome") == "results"]
    base = dict((with_results or payloads)[0])
    products: list[Any] = []
    taxonomy: dict[str, Any] = {}
    filters: dict[str, Any] = {}
    unconfirmed: list[Any] = []
    audience: list[Any] = []
    for payload in payloads:
        # The call-level taxonomy and filters below are the union across roles,
        # which is right for "what did this call cover" and wrong for any claim
        # about one product. Reading the union as though it applied to every
        # product reported a $179.99 sweater as confirmed under a $59.99 cap
        # that belonged to the shoes. Each product carries the scope that
        # actually retrieved it, so a reader never has to guess.
        scope_stamp = {
            "taxonomy": payload.get("taxonomy") or {},
            "confirmed_filters": payload.get("confirmed_filters") or {},
            "composed_role": bool(payload.get("composed_role")),
        }
        for product in payload.get("products") or []:
            products.append(
                {**product, "search_scope": scope_stamp}
                if isinstance(product, dict)
                else product
            )
        for name, value in (payload.get("taxonomy") or {}).items():
            existing = taxonomy.get(name)
            if isinstance(existing, list) and isinstance(value, list):
                taxonomy[name] = existing + [v for v in value if v not in existing]
            elif existing is None:
                taxonomy[name] = value
        for name, value in (payload.get("confirmed_filters") or {}).items():
            filters.setdefault(name, value)
        for item in payload.get("unconfirmed_requirements") or []:
            if item not in unconfirmed:
                unconfirmed.append(item)
        # A look is disclosed once, for the whole look. One role naming an
        # audience the shopper did not is enough to owe them the sentence,
        # and a role searched under a stated audience contributes nothing.
        for item in payload.get("assumed_audience") or []:
            if item not in audience:
                audience.append(item)
    base["outcome"] = "results" if with_results else base.get("outcome")
    base["products"] = products
    base["taxonomy"] = taxonomy
    base["confirmed_filters"] = filters
    base["unconfirmed_requirements"] = unconfirmed
    base["assumed_audience"] = audience[:8]
    base["result_set_complete"] = all(
        p.get("result_set_complete") for p in payloads
    )
    merged: dict[str, Any] = {EVIDENCE_KEY: base}
    for artifact in artifacts:
        for key, value in (artifact or {}).items():
            if key != EVIDENCE_KEY:
                merged.setdefault(key, value)
    return merged


def _merge_repair(target: Any, source: Any) -> None:
    """Fold one scope's repair bookkeeping back onto the turn's.

    Sets and dicts union, because they are already keyed by scope. The
    single-slot fields take the first scope that claimed them: a repair is
    answered by the scope it belongs to, and a later scope must not overwrite
    what an earlier rejection recorded.
    """

    target.constraint_reviewed_scopes |= source.constraint_reviewed_scopes
    for key, value in source.pending_constraint_reviews.items():
        target.pending_constraint_reviews.setdefault(key, value)
    for name in (
        "failed_repair_scope_key",
        "failed_constraint_scope_key",
        "pending_taxonomy_constraints",
    ):
        if getattr(target, name) is None and getattr(source, name) is not None:
            setattr(target, name, getattr(source, name))
    target.pending_no_direct_constraint_clear = (
        target.pending_no_direct_constraint_clear
        or source.pending_no_direct_constraint_clear
    )
    if not target.pending_schema_requirements:
        target.pending_schema_requirements = list(source.pending_schema_requirements)
