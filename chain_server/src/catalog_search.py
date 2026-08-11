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

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

import json
import time
from typing import Any
from pydantic import (
    BaseModel,
    ValidationError,
)
from .agenttypes import State
from .catalog_execution import execute_catalog_search
from .catalog_request import (
    CatalogSearchIntent,
    build_catalog_search_plan,
)
from .control_signals import (
    ControlSignal,
    REJECTIONS_KEY,
    SearchRejection,
    control,
)
from .tool_evidence import (
    EVIDENCE_KEY,
    SearchEvidence,
)
from .turn_scope import CatalogRepairState, TurnScope
from .tool_loop_control import (
    CONSTRAINT_REVIEW_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
)
from .turn_support import (
    _SEARCH_BUDGET_EXHAUSTED_NOTE,
    _SEARCH_NO_MATCH_GROUNDING_NOTE,
    _SEARCH_RESULT_GROUNDING_NOTE,
    _SEARCH_SCOPE_COMPLETE_NOTE,
    SearchCatalogToolArguments,
    _UNSUPPORTED_SEARCH_MODE_MESSAGE,
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
    stated_media_terms,
    _taxonomy_hard_constraints,
    _text_mentions_product_type,
    _tool_search_mode,
    _unsupported_requirement_message,
)
from .response_format import (
    _format_catalog_scope_outcome,
    _format_product_record,
    _format_search_direction_evidence,
    _format_search_filter_evidence,
    _format_search_guidance_evidence,
    _format_search_composed_role_evidence,
    _format_search_scope_relation_evidence,
    _format_search_taxonomy_evidence,
)


#: What a search step hands back: nothing, meaning the search continues, or the
#: text the model reads -- paired with the evidence artifact behind it when the
#: step produced one.
StepResult = str | tuple[str, dict[str, Any]] | None


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
    capabilities: Any = None
    #: True when the shopper never named this role and the model composed it.
    #: Recorded rather than refused: the reply has to present it as proposed,
    #: and must not read a miss inside the searched types as the role being
    #: unavailable.
    composed_role: bool = False
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
    return result


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
    initial_scope_key = _product_scope_key(requested_product_type)
    shopper_stated_requested_scope = bool(
        initial_scope_key
        and _shopper_stated_product_scope(
            ctx.state.query,
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
        query=ctx.state.query,
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

    candidate_scope_key = attempt.candidate_scope_key
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
    shopper_stated_scope = bool(
        candidate_scope_key
        and _shopper_stated_product_scope(
            ctx.state.query,
            ctx.state.dialogue,
            candidate_scope_key,
        )
    )
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
    # Provenance for these is deliberately not established by matching the
    # model's phrasing against the shopper's words. That gate refused
    # "cable-knit texture" because the camera had said "cable knit" and the
    # extra noun was not typed anywhere -- dropping the sweater at the centre
    # of the shopper's own video and answering about boots instead. There is
    # no principled stopping point for a word list, and nothing here can
    # exclude a product: these are stripped before hard filters are built and
    # ride the semantic query, so the search is the same either way.
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

    advertised_choices = attempt.advertised_choices
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
    shopper_stated_scope = bool(
        candidate_scope_key
        and _shopper_stated_product_scope(
            ctx.state.query,
            ctx.state.dialogue,
            candidate_scope_key,
        )
    )
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
    agent_selected_shopper_scope = bool(
        candidate_scope_key
        and _shopper_stated_product_scope(
            ctx.state.query,
            ctx.state.dialogue,
            candidate_scope_key,
        )
    )
    attempt.composed_role = (
        request.taxonomy_status == "agent_selected_type"
        and not agent_selected_shopper_scope
    )
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
        if not suppress_requirement_disclosure:
            # Rank on it, disclose it, do not abandon the search.
            unconfirmable_requirements = list(unadvertised_requirements)
            # The review-and-refuse path that stood here is gone. It sent the
            # model back to rewrite a requirement whenever the shopper's typed
            # words did not contain it, and refused the scope outright on the
            # second attempt. Nothing it guarded can change a result: these are
            # stripped before hard filters are built and only ride the semantic
            # query. It cost a shopper the sweater in their own video, because
            # the camera said "cable knit" and the model wrote "cable-knit
            # texture". Provenance that matters lives on required_constraints.

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
        shopper_stated_scope = bool(
            candidate_scope_key
            and _shopper_stated_product_scope(
                ctx.state.query,
                ctx.state.dialogue,
                candidate_scope_key,
            )
        )
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

    shopper_scope_key = (
        (_normalize_product_text(ctx.state.query), candidate_scope_key)
        if candidate_scope_key
        and _text_mentions_product_type(
            ctx.state.query,
            candidate_scope_key,
        )
        else None
    )

    if request.taxonomy_status == "no_direct_catalog_match":
        with ctx.scope.catalog_lock:
            if (
                shopper_scope_key is not None
                and shopper_scope_key in ctx.scope.searched_shopper_scopes
            ):
                return _rejected(
                    attempt,
                    SearchRejection.DUPLICATE_SHOPPER_SCOPE,
                    control(
                        "STOP_TOOL_USE: This shopper-requested product scope "
                        "was already searched in this turn. Do not search an "
                        "adjacent taxonomy or report the requested scope as "
                        "unavailable. Use the result already returned.\n\n"
                        + _SEARCH_SCOPE_COMPLETE_NOTE,
                        ControlSignal.STOP_TOOL_USE,
                    ),
                )
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
        if (
            shopper_scope_key is not None
            and shopper_scope_key in already_searched
        ):
            return _rejected(
                attempt,
                SearchRejection.DUPLICATE_SHOPPER_SCOPE,
                control(
                    "STOP_TOOL_USE: This shopper-requested product scope "
                    "was already searched in this turn. Do not search an "
                    "adjacent taxonomy. Use the result already returned.\n\n"
                    + _SEARCH_SCOPE_COMPLETE_NOTE,
                    ControlSignal.STOP_TOOL_USE,
                ),
            )
        if search_scope in ctx.scope.searched_catalog_scopes:
            return _rejected(
                attempt,
                SearchRejection.DUPLICATE_CATALOG_SCOPE,
                control(
                    "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already "
                    "searched in this turn. Do not retry it "
                    "with a paraphrase or query expansion. Use the products "
                    "already returned, or ask one concise clarifying question.",
                    ControlSignal.STOP_TOOL_USE,
                ),
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
        if result.ok and result.products:
            ctx.scope.product_evidence.add(result.products)
            _append_product_results(ctx.state, result.products)
            for product in result.products:
                if product.image_url:
                    ctx.scope.retrieved[product.display_name] = product.image_url
    if not result.ok:
        return result.error.message if result.error else "Catalog search failed."

    attempt.execution = execution
    attempt.result = result
    return None


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
    role_advertised_types = (
        list(request.taxonomy.subcategory or [])
        if attempt.composed_role
        else []
    )
    scope_relation_evidence = (
        _format_search_scope_relation_evidence(
            requested_product_type=request.requested_product_type or "",
            advertised_category=advertised_category,
        )
        if advertised_category
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
        # Build the payload first; every line below renders from it.
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
    if scope_relation_evidence:
        lines.append(scope_relation_evidence)
    if evidence.confirmed_filters:
        lines.append(
            _format_search_filter_evidence(evidence.confirmed_filters)
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
_PLAN_STEPS = (
    _admit_search,
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

    attempts: list[_Attempt] = []
    for index, raw in enumerate(scopes):
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
            ):
                outcomes[index] = outcome

    rendered: list[str] = []
    if not_covered:
        # The shopper asked for something no advertised category covers. It
        # costs no retrieval, but recording it is what stops the request being
        # silently dropped: without this the tool sees two scopes and cannot
        # know a third thing was asked for.
        rendered.append(
            "NOT_COVERED: this catalog carries nothing of these kinds, so they "
            "were not searched. Tell the shopper plainly rather than omitting "
            "them: " + ", ".join(str(item) for item in not_covered)
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

    codes = [attempt.rejection_code for attempt in attempts]
    if len(attempts) == 1:
        single = outcomes[0] if outcomes[0] is not None else _RENDER_STEP(ctx, attempts[0])
        if single is None:
            single = "Catalog search returned nothing."
        return _with_scope_rejections(single, codes)
    merged = _merged_artifacts(artifacts)
    text = "\n\n".join(rendered)
    return _with_scope_rejections((text, merged) if merged else text, codes)


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
