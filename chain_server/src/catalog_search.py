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

from dataclasses import dataclass

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
    control,
)
from .tool_evidence import (
    SearchEvidence,
)
from .turn_scope import TurnScope
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
    _shopper_stated_requirement,
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
    _format_search_scope_relation_evidence,
    _format_search_taxonomy_evidence,
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


def _lock_taxonomy_constraint_values(
    ctx: SearchContext,
    scope_key: str | None,
    constraints: dict[str, Any],
    *,
    allow_no_direct_clear: bool = False,
) -> str:
    """Store canonical hard constraints for one taxonomy repair."""

    constraints = _normalized_scope_value(constraints)
    constraints.pop("unadvertised_requirements", None)
    ctx.scope.repair.pending_taxonomy_constraints = constraints
    ctx.scope.repair.pending_no_direct_constraint_clear = allow_no_direct_clear
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
    ctx: SearchContext,
    scope_key: str | None,
    request: SearchCatalogToolArguments,
) -> str:
    """Preserve validated hard constraints across one taxonomy repair."""

    return _lock_taxonomy_constraint_values(
        ctx,
        scope_key,
        request.required_constraints.model_dump(exclude_none=True),
    )


def search_catalog(
    ctx: SearchContext,
    semantic_query: str,
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
    required_constraints: BaseModel | dict[str, Any],
    shopper_guidance: str,
    scope_complete: bool = True,
    search_mode: str | None = None,
):
    """Execute one catalog search; may return control signals."""

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
        ctx.scope.repair.failed_constraint_scope_key or ctx.scope.repair.failed_repair_scope_key
    )
    repairing_same_scope = bool(
        locked_repair_scope
        and (
            candidate_scope_key == ctx.scope.repair.failed_constraint_scope_key
            if ctx.scope.repair.failed_constraint_scope_key
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
            "requested_product_type and repair taxonomy instead."
        )
    if (
        ctx.scope.repair.failed_agent_selected_scope
        and taxonomy_status != "agent_selected_type"
    ):
        return (
            SEARCH_VALIDATION_ERROR_PREFIX
            + "This open-role repair must choose exactly one advertised "
            "subcategory for the role."
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
    stated_unadvertised_requirements = (
        [
            requirement
            for requirement in raw_unadvertised_requirements
            if isinstance(requirement, str)
            and _shopper_stated_requirement(ctx.state.query, requirement)
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
        canonical_agent_selected_scope = bool(
            candidate_scope_key
            and taxonomy_status == "agent_selected_type"
            and _agent_selected_scope_is_advertised(
                requested_product_type,
                taxonomy,
            )
        )
        if shopper_stated_scope or canonical_agent_selected_scope:
            ctx.scope.repair.failed_repair_scope_key = candidate_scope_key
            ctx.scope.repair.failed_agent_selected_scope = False
        elif taxonomy_error and taxonomy_status == "agent_selected_type":
            ctx.scope.repair.failed_agent_selected_scope = True
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
                    " For an open-role search, choose "
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
                    ctx.scope.repair.pending_schema_requirements = list(
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
                ctx,
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
        ctx.scope.repair.pending_taxonomy_constraints is not None
        and not (
            ctx.scope.repair.pending_no_direct_constraint_clear
            and request.taxonomy_status == "no_direct_catalog_match"
        )
        and normalized_advertised_constraints
        != ctx.scope.repair.pending_taxonomy_constraints
    ):
        return (
            SEARCH_VALIDATION_ERROR_PREFIX
            + "A taxonomy repair must preserve previously validated "
            "advertised required_constraints exactly. Change only "
            "taxonomy or an explicitly identified "
            "ungrounded product scope."
        )
    if ctx.scope.repair.pending_taxonomy_constraints is not None:
        ctx.scope.repair.pending_taxonomy_constraints = None
        ctx.scope.repair.pending_no_direct_constraint_clear = False
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
            ctx.scope.repair.failed_repair_scope_key = candidate_scope_key
            ctx.scope.repair.failed_agent_selected_scope = False
        return (
            SEARCH_VALIDATION_ERROR_PREFIX
            + advertised_taxonomy_issue
            + _lock_taxonomy_constraints(ctx, candidate_scope_key, request)
            + (
                " Preserve the shopper-stated requested_product_type."
                if shopper_stated_scope
                else " The rejected requested_product_type was not "
                "shopper-stated. Re-read the current shopper request "
                "and correct it rather than preserving this scope."
            )
        )

    if ctx.scope.repair.pending_schema_requirements and not unadvertised_requirements:
        request = request.model_copy(
            update={
                "shopper_guidance": _generic_shopper_guidance(
                    request.requested_product_type
                )
            }
        )
        ctx.scope.repair.pending_schema_requirements = []
    pending_constraint_review = ctx.scope.repair.pending_constraint_reviews.get(
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
        return (
            SEARCH_VALIDATION_ERROR_PREFIX
            + "A constraint-provenance repair must preserve "
            "requested_product_type, taxonomy, scope_complete, "
            "search_mode, and all "
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
                ctx.state.query,
                ctx.state.dialogue,
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
                ctx.state.query,
                ctx.state.dialogue,
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
        else:
            advertised_choices = _advertised_subcategories_for_selection(
                request.taxonomy,
                capabilities,
            )
            agent_selected_issue = (
                "An open-role search must select exactly one advertised "
                "subcategory and copy that exact "
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
            ctx.scope.repair.failed_repair_scope_key = candidate_scope_key
            ctx.scope.repair.failed_agent_selected_scope = False
        elif candidate_scope_key:
            ctx.scope.repair.failed_agent_selected_scope = True
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
            + _lock_taxonomy_constraints(ctx, candidate_scope_key, request)
        )
    ctx.scope.repair.failed_agent_selected_scope = False

    if (
        request.taxonomy_status != "no_direct_catalog_match"
        and unadvertised_requirements
    ):
        stated_requirements = [
            requirement
            for requirement in unadvertised_requirements
            if _shopper_stated_requirement(ctx.state.query, requirement)
        ]
        shopper_stated_scope = bool(
            candidate_scope_key
            and _shopper_stated_product_scope(
                ctx.state.query,
                ctx.state.dialogue,
                candidate_scope_key,
            )
        )
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
            if review_scope in ctx.scope.repair.constraint_reviewed_scopes:
                return (
                    "The requested catalog requirement cannot be enforced: "
                    "its current-turn provenance could not be established. "
                    "Ask the shopper to state the exact required attribute "
                    "or allow it to be treated as a preference."
                )
            ctx.scope.repair.constraint_reviewed_scopes.add(review_scope)
            ctx.scope.repair.pending_constraint_reviews[review_scope] = {
                "requirements": list(unadvertised_requirements),
                "taxonomy": request.taxonomy.model_dump(),
                "scope_complete": request.scope_complete,
                "search_mode": request.search_mode,
                "required_constraints": dict(normalized_constraints),
            }
            ctx.scope.repair.failed_constraint_scope_key = review_scope
            return (
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
                "occasion, or style goals are not explicit requirements."
            )

    reviewed_constraint = ctx.scope.repair.pending_constraint_reviews.pop(
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
            ctx.scope.repair.failed_repair_scope_key = candidate_scope_key
            ctx.scope.repair.failed_agent_selected_scope = False
        return (
            SEARCH_VALIDATION_ERROR_PREFIX
            + exact_taxonomy_issue
            + _lock_taxonomy_constraints(ctx, candidate_scope_key, request)
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
            "clarifying question instead of searching an adjacent type."
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
        ctx.scope.repair.failed_repair_scope_key = candidate_scope_key
        ctx.scope.repair.failed_agent_selected_scope = False
        return (
            SEARCH_VALIDATION_ERROR_PREFIX
            + f"The requested product type '{request.requested_product_type}' "
            f"matches advertised taxonomy value '{advertised_match}'. "
            "Select that advertised value instead of reporting a gap."
            + _lock_taxonomy_constraints(ctx, candidate_scope_key, request)
        )

    ctx.scope.repair.failed_repair_scope_key = None
    ctx.scope.repair.failed_constraint_scope_key = None
    ctx.scope.repair.failed_agent_selected_scope = False

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
                return control(
                    "STOP_TOOL_USE: This shopper-requested product scope "
                    "was already searched in this turn. Do not search an "
                    "adjacent taxonomy or report the requested scope as "
                    "unavailable. Use the result already returned.\n\n"
                    + _SEARCH_SCOPE_COMPLETE_NOTE,
                    ControlSignal.STOP_TOOL_USE,
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
        return "\n\n".join(lines), evidence.as_artifact()

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

    with ctx.scope.catalog_lock:
        search_scope = _catalog_search_scope(
            taxonomy_constraints,
            normalized_constraints,
        )
        if (
            shopper_scope_key is not None
            and shopper_scope_key in ctx.scope.searched_shopper_scopes
        ):
            return control(
                "STOP_TOOL_USE: This shopper-requested product scope "
                "was already searched in this turn. Do not search an "
                "adjacent taxonomy. Use the result already returned.\n\n"
                + _SEARCH_SCOPE_COMPLETE_NOTE,
                ControlSignal.STOP_TOOL_USE,
            )
        if search_scope in ctx.scope.searched_catalog_scopes:
            return control(
                "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already "
                "searched in this turn. Do not retry it "
                "with a paraphrase or query expansion. Use the products "
                "already returned, or ask one concise clarifying question.",
                ControlSignal.STOP_TOOL_USE,
            )
        if ctx.scope.catalog_searches >= ctx.config.max_catalog_searches_per_turn:
            return control(
                "STOP_TOOL_USE: Catalog search limit reached for this turn. "
                "Do not call more tools this turn. Use the products already "
                "returned in this turn to answer concisely, or ask one concise "
                "clarifying question if the available products are not enough.",
                ControlSignal.STOP_TOOL_USE,
            )
        ctx.scope.searched_catalog_scopes.append(search_scope)
        if shopper_scope_key is not None:
            ctx.scope.searched_shopper_scopes.add(shopper_scope_key)
        ctx.scope.catalog_searches += 1
        search_budget_exhausted = (
            ctx.scope.catalog_searches
            >= ctx.config.max_catalog_searches_per_turn
        )

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
    scope_relation_evidence = (
        _format_search_scope_relation_evidence(
            requested_product_type=request.requested_product_type or "",
            advertised_category=advertised_category,
        )
        if advertised_category
        else ""
    )
    if not result.products:
        # Build the payload first; every line below renders from it.
        evidence = SearchEvidence(
            outcome="zero_results",
            taxonomy=taxonomy_constraints,
            confirmed_filters=confirmed_filters,
            requested_product_type=request.requested_product_type,
            advertised_category=advertised_category,
            scope_complete=bool(request.scope_complete),
            budget_exhausted=bool(search_budget_exhausted),
            unconfirmed_requirements=unconfirmable_requirements,
            scope_outcome={
                "outcome": "zero_results",
                "requested_product_type": request.requested_product_type,
                "taxonomy": taxonomy_constraints,
                "confirmed_filters": confirmed_filters,
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
        if evidence.scope_complete:
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
        scope_complete=bool(request.scope_complete),
        budget_exhausted=bool(search_budget_exhausted),
        unconfirmed_requirements=unconfirmable_requirements,
        products=[
            _search_product_record(product) for product in result.products
        ],
    )
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
