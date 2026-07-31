# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure authority compilation for one turn's typed weather scope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shared.weather_receipts import WeatherLocationScope, WeatherReceiptWindow
from shared.weather_scope import (
    CurrentWeatherScope,
    CurrentWeatherScopeResolution,
    effective_resolved_weather_scope_values,
)

from .weather_scope_resolver import WeatherScopeResolverDecision


EventContextNextQuestion = Literal[
    "event_location",
    "event_venue",
    "event_date",
    "none",
]


@dataclass(frozen=True)
class WeatherScopeAuthorityOutcome:
    """One validated outcome at the semantic/prior-authority boundary."""

    resolver_decision: WeatherScopeResolverDecision | None
    resolver_applies: bool
    resolution: CurrentWeatherScopeResolution | None
    next_question: EventContextNextQuestion
    weather_refresh: bool
    weather_receipt_id: str | None
    blocks_weather: bool
    current_turn_replacement: bool
    effective_location: WeatherLocationScope | None
    effective_window: WeatherReceiptWindow | None


def resolver_pending_binding_is_valid(
    current_scope: CurrentWeatherScope,
    decision: WeatherScopeResolverDecision,
) -> bool:
    """Accept pending-question control only for the exact durable handle."""

    if decision.pending_disposition == "not_addressed":
        return True
    return bool(
        current_scope.pending_question is not None
        and current_scope.pending_source_turn_id is not None
        and current_scope.pending_source_sequence is not None
        and decision.pending_source_turn_id
        == current_scope.pending_source_turn_id
    )


def compile_weather_scope_authority(
    *,
    current_scope: CurrentWeatherScope,
    proposed_resolution: CurrentWeatherScopeResolution | None,
    resolver_decision: WeatherScopeResolverDecision | None,
    resolver_required: bool,
    atomic_scope_supported: bool,
    expected_projection_version: int,
    next_question: EventContextNextQuestion,
    weather_refresh: bool,
    weather_receipt_id: str | None,
    scope_v6_supported: bool = True,
) -> WeatherScopeAuthorityOutcome:
    """Compile typed current facts with separately resolved prior authority.

    The semantic resolver may authorize imports from the durable scope. It can
    never veto a current-turn component, which was independently validated
    before this function is called.
    """

    decision = resolver_decision
    resolver_applies = bool(resolver_required and decision is not None)
    unsupported_decline = bool(
        resolver_applies
        and decision is not None
        and decision.pending_disposition == "declined"
        and resolver_pending_binding_is_valid(current_scope, decision)
        and not scope_v6_supported
    )
    pending_control_rejected = False
    if (
        resolver_applies
        and decision is not None
        and (
            not resolver_pending_binding_is_valid(current_scope, decision)
            or (
                decision.pending_disposition == "declined"
                and not scope_v6_supported
            )
        )
    ):
        pending_control_rejected = True
        decision = WeatherScopeResolverDecision(
            subject_relation=(
                decision.subject_relation
                if current_scope.pending_question is None
                else "unclear"
            ),
            pending_disposition="not_addressed",
        )

    resolver_fails_closed = bool(
        resolver_required
        and (
            decision is None
            or decision.subject_relation == "unclear"
        )
    )
    resolver_new_subject = bool(
        resolver_applies
        and decision is not None
        and decision.subject_relation == "new_subject"
    )
    resolver_unchanged = bool(
        resolver_applies
        and decision is not None
        and decision.subject_relation == "unchanged"
    )
    resolver_answers_pending = bool(
        resolver_applies
        and decision is not None
        and decision.pending_disposition == "answered"
    )
    resolver_resumes_pending = bool(
        resolver_applies
        and decision is not None
        and decision.pending_disposition == "resume_requested"
    )
    resolver_declines_pending = bool(
        resolver_applies
        and decision is not None
        and decision.pending_disposition == "declined"
    )

    resolution = proposed_resolution
    if unsupported_decline:
        resolution = _legacy_decline_fallback_resolution(resolution)
    elif resolution is not None and not scope_v6_supported:
        resolution = _without_unavailable_actions(resolution)
    refresh = weather_refresh
    receipt_id = weather_receipt_id
    current_turn_replacement = _is_complete_current_turn_replacement(
        resolution
    )

    if resolution is not None:
        refresh = False
        receipt_id = None
    if pending_control_rejected:
        refresh = False
        receipt_id = None
    if (
        resolver_fails_closed
        or resolver_new_subject
        or resolver_answers_pending
        or resolver_declines_pending
        or resolver_resumes_pending
    ):
        refresh = False
        receipt_id = None
    if resolver_unchanged and resolution is not None:
        refresh = False
        receipt_id = None
        resolution = _without_prior_retains(resolution)

    pending_completion_retains_prior_counterpart = (
        _pending_completion_retains_prior_counterpart(
            current_scope,
            resolution,
        )
    )
    if resolver_new_subject:
        if resolution is None:
            if not atomic_scope_supported:
                raise ValueError("atomic weather scope is unavailable")
            resolution = _cleared_scope_resolution(
                current_scope,
                expected_projection_version=expected_projection_version,
            )
        else:
            resolution = _without_prior_retains(resolution)
    elif resolver_fails_closed and resolution is not None:
        resolution = _without_prior_retains(resolution)

    if (
        resolver_new_subject
        and resolution is not None
        and current_scope.pending_question is not None
        and scope_v6_supported
    ):
        resolution = _attach_exact_pending_supersession(
            current_scope,
            resolution,
        )

    if resolver_declines_pending:
        if resolution is None:
            if not atomic_scope_supported:
                raise ValueError("atomic weather scope is unavailable")
            resolution = _pending_decline_resolution(
                current_scope,
                expected_projection_version=expected_projection_version,
            )
        resolution = _attach_exact_pending_decline(
            current_scope,
            resolution,
        )
    elif resolution is not None and not resolver_new_subject:
        resolution = _without_unbound_pending_unavailability(
            current_scope,
            resolution,
        )

    pending_component_is_set = _pending_component_is_set(
        current_scope,
        resolution,
    )
    pending_retain_is_authorized = bool(
        resolver_answers_pending
        and pending_component_is_set
        and pending_completion_retains_prior_counterpart
    )
    if pending_retain_is_authorized and resolution is not None:
        resolution = _attach_exact_pending_retain_completion(
            current_scope,
            resolution,
        )

    pending_counterpart_authority_bypassed = bool(
        pending_completion_retains_prior_counterpart
        and not pending_retain_is_authorized
        and not resolver_declines_pending
    )
    pending_authority_failed = bool(
        (resolver_answers_pending and not pending_component_is_set)
        or pending_counterpart_authority_bypassed
    )
    if pending_authority_failed:
        refresh = False
        receipt_id = None
        if resolution is not None:
            resolution = _without_prior_retains(resolution)

    current_turn_replacement = _is_complete_current_turn_replacement(
        resolution
    )
    depends_on_prior_authority = bool(
        resolution is None
        or resolution.location_action == "retain"
        or resolution.window_action == "retain"
    )
    blocks_weather = bool(
        resolver_declines_pending
        or unsupported_decline
        or (
            (resolver_fails_closed or pending_authority_failed)
            and depends_on_prior_authority
        )
    )

    effective_location, effective_window = (
        effective_resolved_weather_scope_values(current_scope, resolution)
    )
    effective_location_unavailable = _component_is_unavailable(
        current_scope,
        resolution,
        component="location",
    )
    effective_window_unavailable = _component_is_unavailable(
        current_scope,
        resolution,
        component="window",
    )
    if refresh and (
        effective_location is None or effective_window is None
    ):
        raise ValueError("weather refresh requires a complete effective scope")

    resume_without_scope_write = bool(
        resolver_resumes_pending
        and resolution is None
        and current_scope.pending_question is not None
    )
    accepted_question = (
        "none"
        if resolver_declines_pending
        else (
            current_scope.pending_question
            if resume_without_scope_write
            else next_question
        )
    )
    if (
        resolver_unchanged
        and not resolver_resumes_pending
        and current_scope.pending_question is not None
        and accepted_question == current_scope.pending_question
    ):
        accepted_question = "none"

    preserve_same_subject_pending = _same_subject_preserves_pending_binding(
        current_scope,
        decision,
        resolution,
        accepted_question,
    )
    preserve_failed_pending_answer = bool(
        resolver_answers_pending
        and not pending_component_is_set
        and not current_turn_replacement
        and current_scope.pending_question is not None
    )
    preserve_rejected_pending_control = bool(
        pending_control_rejected
        and resolution is not None
        and not pending_component_is_set
        and current_scope.pending_question is not None
    )
    if resolver_declines_pending:
        accepted_question = "none"
    elif (
        effective_location_unavailable or effective_window_unavailable
    ) and accepted_question in {"event_location", "event_date"}:
        accepted_question = "none"
    elif pending_counterpart_authority_bypassed:
        accepted_question = _opposite_pending_question(current_scope)
    elif (
        resolution is not None
        and effective_location is None
        and not effective_location_unavailable
        and not preserve_same_subject_pending
        and not preserve_rejected_pending_control
    ):
        accepted_question = "event_location"
    elif (
        accepted_question == "event_location"
        and (
            effective_location is not None
            or effective_location_unavailable
        )
    ) or (
        accepted_question == "event_date"
        and (
            effective_window is not None
            or effective_window_unavailable
        )
    ):
        accepted_question = "none"

    if (
        not resume_without_scope_write
        and resolution is None
        and accepted_question in {
            "event_location",
            "event_date",
        }
    ):
        if not atomic_scope_supported:
            raise ValueError("atomic weather scope is unavailable")
        resolution = _pending_scope_resolution(
            current_scope,
            expected_projection_version=expected_projection_version,
            pending_question=accepted_question,
        )
        if resolver_fails_closed:
            resolution = _without_prior_retains(resolution)

    if resolution is not None and not pending_retain_is_authorized:
        accepted_pending_question = (
            accepted_question
            if accepted_question in {"event_location", "event_date"}
            else (
                current_scope.pending_question
                if (
                    preserve_same_subject_pending
                    or preserve_failed_pending_answer
                    or preserve_rejected_pending_control
                )
                else None
            )
        )
        resolution = resolution.model_copy(
            update={
                "pending_question": accepted_pending_question,
                "preserve_pending_source_turn_id": (
                    current_scope.pending_source_turn_id
                    if (
                        preserve_same_subject_pending
                        or preserve_rejected_pending_control
                    )
                    else None
                ),
            }
        )

    effective_location, effective_window = (
        effective_resolved_weather_scope_values(current_scope, resolution)
    )
    return WeatherScopeAuthorityOutcome(
        resolver_decision=decision,
        resolver_applies=resolver_applies,
        resolution=resolution,
        next_question=accepted_question,
        weather_refresh=refresh,
        weather_receipt_id=receipt_id,
        blocks_weather=blocks_weather,
        current_turn_replacement=current_turn_replacement,
        effective_location=effective_location,
        effective_window=effective_window,
    )


def _without_prior_retains(
    resolution: CurrentWeatherScopeResolution,
) -> CurrentWeatherScopeResolution:
    """Clear every retain when semantic prior authority is unavailable."""

    return CurrentWeatherScopeResolution(
        expected_projection_version=resolution.expected_projection_version,
        expected_scope_revision=resolution.expected_scope_revision,
        location_action=(
            "clear"
            if resolution.location_action == "retain"
            else resolution.location_action
        ),
        window_action=(
            "clear"
            if resolution.window_action == "retain"
            else resolution.window_action
        ),
        location_scope=resolution.location_scope,
        requested_window=resolution.requested_window,
    )


def _without_unavailable_actions(
    resolution: CurrentWeatherScopeResolution,
) -> CurrentWeatherScopeResolution:
    """Down-project v6-only component state for an older memory contract."""

    return resolution.model_copy(
        update={
            "location_action": (
                "clear"
                if resolution.location_action == "unavailable"
                else resolution.location_action
            ),
            "window_action": (
                "clear"
                if resolution.window_action == "unavailable"
                else resolution.window_action
            ),
        }
    )


def _legacy_decline_fallback_resolution(
    resolution: CurrentWeatherScopeResolution | None,
) -> CurrentWeatherScopeResolution | None:
    """Keep only independent current-turn facts when memory lacks v6 state."""

    if resolution is None:
        return None
    location_is_set = resolution.location_action == "set"
    window_is_set = resolution.window_action == "set"
    if not location_is_set and not window_is_set:
        return None
    return CurrentWeatherScopeResolution(
        expected_projection_version=resolution.expected_projection_version,
        expected_scope_revision=resolution.expected_scope_revision,
        location_action="set" if location_is_set else "clear",
        window_action="set" if window_is_set else "clear",
        location_scope=(
            resolution.location_scope if location_is_set else None
        ),
        requested_window=(
            resolution.requested_window if window_is_set else None
        ),
    )


def _is_complete_current_turn_replacement(
    resolution: CurrentWeatherScopeResolution | None,
) -> bool:
    return bool(
        resolution is not None
        and resolution.location_action == "set"
        and resolution.window_action == "set"
    )


def _cleared_scope_resolution(
    current_scope: CurrentWeatherScope,
    *,
    expected_projection_version: int,
) -> CurrentWeatherScopeResolution:
    return CurrentWeatherScopeResolution(
        expected_projection_version=expected_projection_version,
        expected_scope_revision=current_scope.revision,
        location_action="clear",
        window_action="clear",
    )


def _pending_component_is_set(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution | None,
) -> bool:
    if resolution is None or current_scope.pending_question is None:
        return False
    if current_scope.pending_question == "event_location":
        return resolution.location_action == "set"
    return resolution.window_action == "set"


def _attach_exact_pending_retain_completion(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution,
) -> CurrentWeatherScopeResolution:
    """Attach the exact handle without rewriting the proposed actions."""

    completion_handle = current_scope.pending_source_turn_id
    if completion_handle is None:
        raise ValueError("pending completion requires an exact source handle")
    if not _pending_completion_retains_prior_counterpart(
        current_scope,
        resolution,
    ):
        raise ValueError(
            "pending completion handle authorizes only an exact counterpart retain"
        )
    return resolution.model_copy(
        update={
            "pending_question": None,
            "preserve_pending_source_turn_id": None,
            "complete_pending_source_turn_id": completion_handle,
        }
    )


def _attach_exact_pending_decline(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution,
) -> CurrentWeatherScopeResolution:
    """Consume only the exact durable pending question without rewriting facts."""

    decline_handle = current_scope.pending_source_turn_id
    if (
        current_scope.pending_question is None
        or decline_handle is None
        or current_scope.pending_source_sequence is None
    ):
        raise ValueError("pending decline requires an exact source binding")
    target_action = (
        resolution.location_action
        if current_scope.pending_question == "event_location"
        else resolution.window_action
    )
    if target_action != "unavailable":
        raise ValueError(
            "pending decline requires the target component to be unavailable"
        )
    return resolution.model_copy(
        update={
            "pending_question": None,
            "preserve_pending_source_turn_id": None,
            "complete_pending_source_turn_id": None,
            "decline_pending_source_turn_id": decline_handle,
        }
    )


def _attach_exact_pending_supersession(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution,
) -> CurrentWeatherScopeResolution:
    """Bind whole-subject replacement to the exact pending question it removes."""

    supersede_handle = current_scope.pending_source_turn_id
    if (
        current_scope.pending_question is None
        or supersede_handle is None
        or current_scope.pending_source_sequence is None
    ):
        raise ValueError("pending supersession requires an exact source binding")
    return resolution.model_copy(
        update={
            "preserve_pending_source_turn_id": None,
            "complete_pending_source_turn_id": None,
            "decline_pending_source_turn_id": None,
            "supersede_pending_source_turn_id": supersede_handle,
        }
    )


def _pending_decline_resolution(
    current_scope: CurrentWeatherScope,
    *,
    expected_projection_version: int,
) -> CurrentWeatherScopeResolution:
    """Create the smallest write that removes one exact pending binding."""

    if current_scope.pending_question is None:
        raise ValueError("pending decline requires a pending question")
    return CurrentWeatherScopeResolution(
        expected_projection_version=expected_projection_version,
        expected_scope_revision=current_scope.revision,
        location_action=(
            "unavailable"
            if current_scope.pending_question == "event_location"
            else (
                "retain"
                if (
                    current_scope.location is not None
                    or current_scope.location_unavailable is not None
                )
                else "clear"
            )
        ),
        window_action=(
            "unavailable"
            if current_scope.pending_question == "event_date"
            else (
                "retain"
                if (
                    current_scope.window is not None
                    or current_scope.window_unavailable is not None
                )
                else "clear"
            )
        ),
    )


def _pending_completion_retains_prior_counterpart(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution | None,
) -> bool:
    if resolution is None or current_scope.pending_question is None:
        return False
    if current_scope.pending_question == "event_location":
        return (
            resolution.location_action == "set"
            and resolution.window_action == "retain"
        )
    return (
        resolution.window_action == "set"
        and resolution.location_action == "retain"
    )


def _same_subject_preserves_pending_binding(
    current_scope: CurrentWeatherScope,
    decision: WeatherScopeResolverDecision | None,
    resolution: CurrentWeatherScopeResolution | None,
    next_question: EventContextNextQuestion,
) -> bool:
    if (
        decision is None
        or decision.subject_relation != "same_subject"
        or decision.pending_disposition != "not_addressed"
        or resolution is None
        or next_question != "none"
        or current_scope.pending_question is None
    ):
        return False
    if current_scope.pending_question == "event_location":
        return resolution.location_action == "clear"
    return resolution.window_action == "clear"


def _opposite_pending_question(
    current_scope: CurrentWeatherScope,
) -> Literal["event_location", "event_date"]:
    if current_scope.pending_question == "event_location":
        return "event_date"
    return "event_location"


def _pending_scope_resolution(
    current_scope: CurrentWeatherScope,
    *,
    expected_projection_version: int,
    pending_question: Literal["event_location", "event_date"],
) -> CurrentWeatherScopeResolution:
    if (
        pending_question == "event_location"
        and (
            current_scope.location is not None
            or current_scope.location_unavailable is not None
        )
    ) or (
        pending_question == "event_date"
        and (
            current_scope.window is not None
            or current_scope.window_unavailable is not None
        )
    ):
        raise ValueError(
            "pending weather question targets established or unavailable context"
        )
    return CurrentWeatherScopeResolution(
        expected_projection_version=expected_projection_version,
        expected_scope_revision=current_scope.revision,
        location_action=(
            "retain"
            if (
                current_scope.location is not None
                or current_scope.location_unavailable is not None
            )
            else "clear"
        ),
        window_action=(
            "retain"
            if (
                current_scope.window is not None
                or current_scope.window_unavailable is not None
            )
            else "clear"
        ),
        pending_question=pending_question,
    )


def _without_unbound_pending_unavailability(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution,
) -> CurrentWeatherScopeResolution:
    """Keep exact-handle decline as the only pending cancellation authority."""

    if current_scope.pending_question == "event_location":
        if resolution.location_action != "unavailable":
            return resolution
        return resolution.model_copy(update={"location_action": "clear"})
    if current_scope.pending_question == "event_date":
        if resolution.window_action != "unavailable":
            return resolution
        return resolution.model_copy(update={"window_action": "clear"})
    return resolution


def _component_is_unavailable(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution | None,
    *,
    component: Literal["location", "window"],
) -> bool:
    """Resolve only typed scope state; never interpret shopper prose."""

    current_marker = getattr(current_scope, f"{component}_unavailable")
    if resolution is None:
        return current_marker is not None
    action = getattr(resolution, f"{component}_action")
    if action == "retain":
        return current_marker is not None
    return action == "unavailable"
