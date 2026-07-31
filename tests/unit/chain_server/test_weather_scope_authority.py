# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date

import pytest

from chain_server.src.weather_scope_authority import (
    compile_weather_scope_authority,
)
from chain_server.src.weather_scope_resolver import (
    WeatherScopeResolverDecision,
)
from shared.weather_receipts import (
    ShopperLocationWeatherScope,
    WeatherReceiptWindow,
)
from shared.weather_scope import (
    CurrentWeatherScope,
    CurrentWeatherScopeResolution,
    apply_current_weather_scope_resolution,
)


def _current_scope() -> CurrentWeatherScope:
    return CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "location": {
                "value": {
                    "kind": "shopper_provided_location",
                    "location": "NYC",
                    "location_query": "NYC, NY",
                },
                "source_turn_id": "turn-nyc",
                "source_sequence": 1,
            },
            "window": {
                "value": {
                    "start_date": "2026-08-07",
                    "end_date": "2026-08-07",
                },
                "source_turn_id": "turn-nyc",
                "source_sequence": 1,
            },
        }
    )


def _resolution(
    location_action: str,
    window_action: str,
) -> CurrentWeatherScopeResolution:
    return CurrentWeatherScopeResolution(
        expected_projection_version=5,
        expected_scope_revision=2,
        location_action=location_action,
        window_action=window_action,
        location_scope=(
            ShopperLocationWeatherScope(location="Seattle")
            if location_action == "set"
            else None
        ),
        requested_window=(
            WeatherReceiptWindow(
                start_date=date(2026, 8, 16),
                end_date=date(2026, 8, 16),
            )
            if window_action == "set"
            else None
        ),
    )


def _pending_location_scope() -> CurrentWeatherScope:
    return CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_location",
            "pending_source_turn_id": "turn-conference",
            "pending_source_sequence": 2,
            "window": {
                "value": {
                    "start_date": "2026-08-16",
                    "end_date": "2026-08-16",
                },
                "source_turn_id": "turn-conference",
                "source_sequence": 2,
            },
        }
    )


def _pending_date_scope() -> CurrentWeatherScope:
    return CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_date",
            "pending_source_turn_id": "turn-conference",
            "pending_source_sequence": 2,
            "location": {
                "value": {
                    "kind": "shopper_provided_location",
                    "location": "NYC",
                    "location_query": "NYC, NY",
                },
                "source_turn_id": "turn-conference",
                "source_sequence": 2,
            },
        }
    )


def _resolver_decision(
    subject_relation: str,
    pending_disposition: str = "not_addressed",
    pending_source_turn_id: str = "turn-conference",
) -> WeatherScopeResolverDecision:
    return WeatherScopeResolverDecision.model_validate(
        {
            "subject_relation": subject_relation,
            "pending_disposition": pending_disposition,
            **(
                {"pending_source_turn_id": pending_source_turn_id}
                if pending_disposition != "not_addressed"
                else {}
            ),
        }
    )


@pytest.mark.parametrize(
    ("proposed_actions", "effective_actions"),
    [
        (("set", "set"), ("set", "set")),
        (("set", "clear"), ("set", "clear")),
        (("clear", "set"), ("clear", "set")),
        (("clear", "clear"), ("clear", "clear")),
        (("set", "retain"), ("set", "clear")),
        (("retain", "set"), ("clear", "set")),
    ],
)
def test_unchanged_preserves_current_facts_and_clears_only_prior_retains(
    proposed_actions: tuple[str, str],
    effective_actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_current_scope(),
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision("unchanged"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=True,
        weather_receipt_id="receipt-from-old-scope",
    )

    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == effective_actions
    assert outcome.weather_refresh is False
    assert outcome.weather_receipt_id is None


@pytest.mark.parametrize(
    ("subject_relation", "pending_disposition"),
    [
        ("same_subject", "not_addressed"),
        ("same_subject", "answered"),
        ("new_subject", "not_addressed"),
        ("unchanged", "not_addressed"),
        ("unchanged", "resume_requested"),
        ("unclear", "not_addressed"),
    ],
)
def test_complete_current_turn_replacement_is_independent_authority(
    subject_relation: str,
    pending_disposition: str,
) -> None:
    current_scope = (
        _pending_location_scope()
        if pending_disposition != "not_addressed"
        else _current_scope()
    )
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=_resolution("set", "set"),
        resolver_decision=_resolver_decision(
            subject_relation,
            pending_disposition,
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert outcome.resolution.location_action == "set"
    assert outcome.resolution.window_action == "set"
    assert outcome.current_turn_replacement is True
    assert outcome.blocks_weather is False
    if pending_disposition == "answered":
        assert outcome.resolution.complete_pending_source_turn_id is None
    assert outcome.effective_location == ShopperLocationWeatherScope(
        location="Seattle"
    )
    assert outcome.effective_window == WeatherReceiptWindow(
        start_date=date(2026, 8, 16),
        end_date=date(2026, 8, 16),
    )


@pytest.mark.parametrize(
    "pending_scope",
    [_pending_location_scope(), _pending_date_scope()],
)
def test_answered_pending_current_turn_replacement_has_no_completion_handle(
    pending_scope: CurrentWeatherScope,
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution("set", "set"),
        resolver_decision=_resolver_decision("same_subject", "answered"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == ("set", "set")
    assert outcome.resolution.complete_pending_source_turn_id is None
    assert outcome.resolution.pending_question is None
    assert outcome.current_turn_replacement is True
    assert outcome.blocks_weather is False


@pytest.mark.parametrize(
    "actions",
    [
        ("set", "clear"),
        ("clear", "set"),
        ("clear", "clear"),
    ],
)
def test_failed_resolver_does_not_mark_self_contained_partial_scope_unclear(
    actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_current_scope(),
        proposed_resolution=_resolution(*actions),
        resolver_decision=_resolver_decision("unclear"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.blocks_weather is False
    assert outcome.resolution is not None
    assert "retain" not in {
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    }


def test_failed_resolver_blocks_an_implicit_reuse_of_the_existing_scope() -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_current_scope(),
        proposed_resolution=None,
        resolver_decision=_resolver_decision("unclear"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=True,
        weather_receipt_id="receipt-from-old-scope",
    )

    assert outcome.resolution is None
    assert outcome.weather_refresh is False
    assert outcome.weather_receipt_id is None
    assert outcome.blocks_weather is True


def test_unchanged_not_addressed_suppresses_an_already_pending_question() -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_pending_location_scope(),
        proposed_resolution=None,
        resolver_decision=_resolver_decision("unchanged"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="event_location",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.next_question == "none"
    assert outcome.resolution is None


def test_resume_requested_reasks_pending_question_without_a_scope_write() -> None:
    current_scope = _pending_location_scope()
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=None,
        resolver_decision=_resolver_decision(
            "unchanged",
            "resume_requested",
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=True,
        weather_receipt_id="old-receipt",
    )

    assert outcome.next_question == "event_location"
    assert outcome.resolution is None
    assert outcome.weather_refresh is False
    assert outcome.weather_receipt_id is None
    assert current_scope.pending_source_turn_id == "turn-conference"


@pytest.mark.parametrize(
    ("pending_scope", "expected_actions"),
    [
        (_pending_location_scope(), ("unavailable", "retain")),
        (_pending_date_scope(), ("retain", "unavailable")),
    ],
)
def test_declined_pending_consumes_exact_binding_with_minimal_scope_write(
    pending_scope: CurrentWeatherScope,
    expected_actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=None,
        resolver_decision=_resolver_decision("same_subject", "declined"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="event_location",
        weather_refresh=True,
        weather_receipt_id="old-receipt",
    )

    assert outcome.next_question == "none"
    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == expected_actions
    assert outcome.resolution.pending_question is None
    assert outcome.resolution.decline_pending_source_turn_id == "turn-conference"
    assert outcome.resolution.preserve_pending_source_turn_id is None
    assert outcome.resolution.complete_pending_source_turn_id is None
    assert outcome.weather_refresh is False
    assert outcome.weather_receipt_id is None
    assert outcome.blocks_weather is True

    persisted = apply_current_weather_scope_resolution(
        pending_scope,
        outcome.resolution,
        source_turn_id="decline-turn",
        source_sequence=3,
    )
    assert persisted.pending_question is None
    assert persisted.pending_source_turn_id is None
    assert persisted.pending_source_sequence is None


@pytest.mark.parametrize(
    ("pending_scope", "proposed_actions"),
    [
        (_pending_location_scope(), ("unavailable", "clear")),
        (_pending_date_scope(), ("clear", "unavailable")),
        (_pending_location_scope(), ("unavailable", "set")),
    ],
)
def test_declined_pending_preserves_current_turn_component_actions(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision("same_subject", "declined"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="event_date",
        weather_refresh=True,
        weather_receipt_id="old-receipt",
    )

    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == proposed_actions
    assert outcome.resolution.decline_pending_source_turn_id == "turn-conference"
    assert outcome.next_question == "none"
    assert outcome.blocks_weather is True


@pytest.mark.parametrize(
    ("pending_scope", "proposed_actions"),
    [
        (_pending_location_scope(), ("clear", "retain")),
        (_pending_date_scope(), ("retain", "clear")),
    ],
)
def test_declined_pending_requires_unavailable_target_action(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="target component to be unavailable"):
        compile_weather_scope_authority(
            current_scope=pending_scope,
            proposed_resolution=_resolution(*proposed_actions),
            resolver_decision=_resolver_decision("same_subject", "declined"),
            resolver_required=True,
            atomic_scope_supported=True,
            expected_projection_version=5,
            next_question="none",
            weather_refresh=False,
            weather_receipt_id=None,
        )


def test_typed_unavailability_suppresses_question_and_persists_source() -> None:
    current_scope = _pending_location_scope()
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=_resolution("set", "unavailable"),
        resolver_decision=_resolver_decision("same_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="event_date",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert outcome.resolution.location_action == "set"
    assert outcome.resolution.window_action == "unavailable"
    assert outcome.next_question == "none"
    assert outcome.resolution.pending_question is None
    persisted = apply_current_weather_scope_resolution(
        current_scope,
        outcome.resolution,
        source_turn_id="current-turn",
        source_sequence=3,
    )
    assert persisted.location is not None
    assert persisted.window is None
    assert persisted.window_unavailable is not None
    assert persisted.window_unavailable.source_turn_id == "current-turn"


@pytest.mark.parametrize(
    ("pending_scope", "proposed_actions", "unavailable_field"),
    [
        pytest.param(
            _pending_location_scope(),
            ("clear", "unavailable"),
            "window_unavailable",
            id="pending-location-date-withdrawn",
        ),
        pytest.param(
            _pending_date_scope(),
            ("unavailable", "clear"),
            "location_unavailable",
            id="pending-date-location-withdrawn",
        ),
    ],
)
def test_opposite_component_unavailability_retires_pending_binding_atomically(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
    unavailable_field: str,
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision("same_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == proposed_actions
    assert outcome.next_question == "none"
    assert outcome.resolution.pending_question is None
    assert outcome.resolution.preserve_pending_source_turn_id is None
    assert outcome.resolution.complete_pending_source_turn_id is None
    assert outcome.resolution.decline_pending_source_turn_id is None
    assert outcome.resolution.supersede_pending_source_turn_id is None

    persisted = apply_current_weather_scope_resolution(
        pending_scope,
        outcome.resolution,
        source_turn_id="withdrawal-turn",
        source_sequence=3,
    )
    assert persisted.pending_question is None
    assert persisted.pending_source_turn_id is None
    assert persisted.pending_source_sequence is None
    unavailable = getattr(persisted, unavailable_field)
    assert unavailable is not None
    assert unavailable.source_turn_id == "withdrawal-turn"
    assert unavailable.source_sequence == 3


@pytest.mark.parametrize(
    (
        "pending_scope",
        "withdrawal_actions",
        "later_actions",
        "fresh_question",
    ),
    [
        pytest.param(
            _pending_location_scope(),
            ("clear", "unavailable"),
            ("clear", "set"),
            "event_location",
            id="date-set-reopens-location",
        ),
        pytest.param(
            _pending_date_scope(),
            ("unavailable", "clear"),
            ("set", "clear"),
            "event_date",
            id="location-set-reopens-date",
        ),
    ],
)
def test_later_set_creates_a_fresh_binding_after_opposite_withdrawal(
    pending_scope: CurrentWeatherScope,
    withdrawal_actions: tuple[str, str],
    later_actions: tuple[str, str],
    fresh_question: str,
) -> None:
    retired = apply_current_weather_scope_resolution(
        pending_scope,
        _resolution(*withdrawal_actions),
        source_turn_id="withdrawal-turn",
        source_sequence=3,
    )
    later_resolution = _resolution(*later_actions).model_copy(
        update={
            "expected_projection_version": 6,
            "expected_scope_revision": retired.revision,
        }
    )

    outcome = compile_weather_scope_authority(
        current_scope=retired,
        proposed_resolution=later_resolution,
        resolver_decision=_resolver_decision("same_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=6,
        next_question=fresh_question,
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert outcome.next_question == fresh_question
    assert outcome.resolution.pending_question == fresh_question
    assert outcome.resolution.preserve_pending_source_turn_id is None
    persisted = apply_current_weather_scope_resolution(
        retired,
        outcome.resolution,
        source_turn_id="later-turn",
        source_sequence=4,
    )
    assert persisted.location_unavailable is None
    assert persisted.window_unavailable is None
    assert persisted.pending_question == fresh_question
    assert persisted.pending_source_turn_id == "later-turn"
    assert persisted.pending_source_turn_id != pending_scope.pending_source_turn_id
    assert persisted.pending_source_sequence == 4


@pytest.mark.parametrize(
    ("scope_payload", "later_actions", "fresh_question"),
    [
        pytest.param(
            {
                "revision": 2,
                "location": {
                    "value": {
                        "kind": "shopper_provided_location",
                        "location": "Seattle",
                    },
                    "source_turn_id": "location-turn",
                    "source_sequence": 1,
                },
                "window_unavailable": {
                    "source_turn_id": "date-withdrawal-turn",
                    "source_sequence": 2,
                },
            },
            ("retain", "clear"),
            "event_date",
            id="date-clear",
        ),
        pytest.param(
            {
                "revision": 2,
                "window": {
                    "value": {
                        "start_date": "2026-08-16",
                        "end_date": "2026-08-16",
                    },
                    "source_turn_id": "date-turn",
                    "source_sequence": 1,
                },
                "location_unavailable": {
                    "source_turn_id": "location-withdrawal-turn",
                    "source_sequence": 2,
                },
            },
            ("clear", "retain"),
            "event_location",
            id="location-clear",
        ),
    ],
)
def test_later_clear_makes_an_unavailable_component_freshly_askable(
    scope_payload: dict[str, object],
    later_actions: tuple[str, str],
    fresh_question: str,
) -> None:
    current_scope = CurrentWeatherScope.model_validate(scope_payload)
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=_resolution(*later_actions),
        resolver_decision=_resolver_decision("same_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question=fresh_question,
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert outcome.next_question == fresh_question
    assert outcome.resolution.pending_question == fresh_question
    persisted = apply_current_weather_scope_resolution(
        current_scope,
        outcome.resolution,
        source_turn_id="later-clear-turn",
        source_sequence=3,
    )
    assert persisted.location_unavailable is None
    assert persisted.window_unavailable is None
    assert persisted.pending_question == fresh_question
    assert persisted.pending_source_turn_id == "later-clear-turn"
    assert persisted.pending_source_sequence == 3


@pytest.mark.parametrize(
    ("scope_payload", "proposed_actions", "next_question"),
    [
        (
            {
                "revision": 2,
                "location_unavailable": {
                    "source_turn_id": "location-unavailable-turn",
                    "source_sequence": 2,
                },
            },
            ("retain", "clear"),
            "event_date",
        ),
        (
            {
                "revision": 2,
                "window_unavailable": {
                    "source_turn_id": "date-unavailable-turn",
                    "source_sequence": 2,
                },
            },
            ("clear", "retain"),
            "event_location",
        ),
    ],
)
def test_any_retained_unavailability_suppresses_all_weather_questions(
    scope_payload: dict[str, object],
    proposed_actions: tuple[str, str],
    next_question: str,
) -> None:
    current_scope = CurrentWeatherScope.model_validate(scope_payload)
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision("same_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question=next_question,
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.next_question == "none"
    assert outcome.resolution is not None
    assert outcome.resolution.pending_question is None


def test_unavailable_weather_component_does_not_suppress_venue_question() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "location_unavailable": {
                "source_turn_id": "location-unavailable-turn",
                "source_sequence": 2,
            },
        }
    )
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=_resolution("retain", "clear"),
        resolver_decision=_resolver_decision("same_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="event_venue",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.next_question == "event_venue"
    assert outcome.resolution is not None
    assert outcome.resolution.pending_question is None


@pytest.mark.parametrize(
    ("pending_scope", "proposed_actions"),
    [
        (_pending_location_scope(), ("unavailable", "set")),
        (_pending_date_scope(), ("set", "unavailable")),
    ],
)
def test_new_subject_supersedes_exact_old_pending_binding(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision("new_subject"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == proposed_actions
    assert (
        outcome.resolution.supersede_pending_source_turn_id
        == "turn-conference"
    )
    assert outcome.resolution.decline_pending_source_turn_id is None
    assert outcome.next_question == "none"


def test_declined_pending_is_inert_without_memory_v6_capability() -> None:
    current_scope = _pending_location_scope()
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=None,
        resolver_decision=_resolver_decision("same_subject", "declined"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
        scope_v6_supported=False,
    )

    assert outcome.resolver_decision is not None
    assert outcome.resolver_decision.subject_relation == "unclear"
    assert outcome.resolver_decision.pending_disposition == "not_addressed"
    assert outcome.resolution is None
    assert outcome.next_question == "none"
    assert outcome.blocks_weather is True
    assert current_scope.pending_source_turn_id == "turn-conference"


def test_v5_decline_without_current_facts_preserves_scope_without_write() -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_pending_location_scope(),
        proposed_resolution=_resolution("unavailable", "retain"),
        resolver_decision=_resolver_decision("same_subject", "declined"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
        scope_v6_supported=False,
    )

    assert outcome.resolution is None
    assert outcome.next_question == "none"
    assert outcome.blocks_weather is True


def test_v5_decline_keeps_independent_set_and_preserves_pending() -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_pending_location_scope(),
        proposed_resolution=_resolution("unavailable", "set"),
        resolver_decision=_resolver_decision("same_subject", "declined"),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
        scope_v6_supported=False,
    )

    assert outcome.resolution is not None
    assert outcome.resolution.location_action == "clear"
    assert outcome.resolution.window_action == "set"
    assert outcome.resolution.pending_question == "event_location"
    assert (
        outcome.resolution.preserve_pending_source_turn_id
        == "turn-conference"
    )
    assert outcome.resolution.decline_pending_source_turn_id is None
    assert outcome.resolution.supersede_pending_source_turn_id is None


@pytest.mark.parametrize(
    ("pending_scope", "proposed_actions", "expected_actions"),
    [
        (_pending_location_scope(), ("set", "retain"), ("set", "retain")),
        (_pending_date_scope(), ("retain", "set"), ("retain", "set")),
    ],
)
def test_answered_pending_retain_completes_the_exact_binding(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
    expected_actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision(
            "same_subject",
            "answered",
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.next_question == "none"
    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == expected_actions
    assert (
        outcome.resolution.complete_pending_source_turn_id
        == "turn-conference"
    )


@pytest.mark.parametrize(
    (
        "pending_scope",
        "proposed_actions",
        "next_question",
        "expected_actions",
    ),
    [
        (
            _pending_location_scope(),
            ("set", "clear"),
            "event_date",
            ("set", "clear"),
        ),
        (
            _pending_date_scope(),
            ("clear", "set"),
            "event_location",
            ("clear", "set"),
        ),
        (
            _pending_location_scope(),
            ("clear", "set"),
            "event_location",
            ("clear", "set"),
        ),
        (
            _pending_date_scope(),
            ("set", "clear"),
            "event_date",
            ("set", "clear"),
        ),
    ],
)
def test_answered_pending_never_rewrites_explicit_set_or_clear_actions(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
    next_question: str,
    expected_actions: tuple[str, str],
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision(
            "same_subject",
            "answered",
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question=next_question,
        weather_refresh=True,
        weather_receipt_id="old-scope-receipt",
    )

    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == expected_actions
    assert outcome.resolution.complete_pending_source_turn_id is None
    assert outcome.resolution.pending_question == next_question
    assert outcome.next_question == next_question
    assert outcome.weather_refresh is False
    assert outcome.weather_receipt_id is None
    assert outcome.blocks_weather is False
    if next_question == "event_date":
        assert outcome.effective_location == ShopperLocationWeatherScope(
            location="Seattle"
        )
        assert outcome.effective_window is None
    else:
        assert outcome.effective_location is None
        assert outcome.effective_window == WeatherReceiptWindow(
            start_date=date(2026, 8, 16),
            end_date=date(2026, 8, 16),
        )
    persisted = apply_current_weather_scope_resolution(
        pending_scope,
        outcome.resolution,
        source_turn_id="current-turn",
        source_sequence=3,
    )
    assert persisted.pending_question == next_question
    assert persisted.pending_source_turn_id == "current-turn"
    assert persisted.pending_source_sequence == 3
    if next_question == "event_date":
        assert persisted.location is not None
        assert persisted.location.value == ShopperLocationWeatherScope(
            location="Seattle"
        )
        assert persisted.window is None
    else:
        assert persisted.location is None
        assert persisted.window is not None
        assert persisted.window.value == WeatherReceiptWindow(
            start_date=date(2026, 8, 16),
            end_date=date(2026, 8, 16),
        )


@pytest.mark.parametrize(
    ("pending_scope", "proposed_actions", "next_question"),
    [
        (_pending_location_scope(), ("set", "clear"), "event_date"),
        (_pending_date_scope(), ("clear", "set"), "event_location"),
    ],
)
def test_wrong_pending_handle_cannot_erase_current_turn_set_or_clear(
    pending_scope: CurrentWeatherScope,
    proposed_actions: tuple[str, str],
    next_question: str,
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=pending_scope,
        proposed_resolution=_resolution(*proposed_actions),
        resolver_decision=_resolver_decision(
            "same_subject",
            "answered",
            pending_source_turn_id="another-turn",
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question=next_question,
        weather_refresh=True,
        weather_receipt_id="old-scope-receipt",
    )

    assert outcome.resolver_decision is not None
    assert outcome.resolver_decision.subject_relation == "unclear"
    assert outcome.resolution is not None
    assert (
        outcome.resolution.location_action,
        outcome.resolution.window_action,
    ) == proposed_actions
    assert outcome.resolution.complete_pending_source_turn_id is None
    assert outcome.resolution.pending_question == next_question
    assert outcome.weather_refresh is False
    assert outcome.weather_receipt_id is None
    assert outcome.blocks_weather is False


@pytest.mark.parametrize(
    "pending_disposition",
    ["answered", "declined", "resume_requested"],
)
def test_pending_control_with_the_wrong_handle_fails_closed(
    pending_disposition: str,
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_pending_location_scope(),
        proposed_resolution=None,
        resolver_decision=_resolver_decision(
            (
                "unchanged"
                if pending_disposition == "resume_requested"
                else "same_subject"
            ),
            pending_disposition,
            pending_source_turn_id="another-turn",
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolver_decision is not None
    assert outcome.resolver_decision.subject_relation == "unclear"
    assert outcome.resolver_decision.pending_disposition == "not_addressed"
    assert outcome.next_question == "none"
    assert outcome.resolution is None
    assert outcome.blocks_weather is True


def test_stale_pending_axis_does_not_erase_valid_subject_continuity() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "location": {
                "value": {
                    "kind": "shopper_provided_location",
                    "location": "Seattle",
                },
                "source_turn_id": "turn-seattle",
                "source_sequence": 2,
            },
            "window_unavailable": {
                "source_turn_id": "turn-seattle",
                "source_sequence": 2,
            },
        }
    )
    outcome = compile_weather_scope_authority(
        current_scope=current_scope,
        proposed_resolution=_resolution("retain", "set"),
        resolver_decision=_resolver_decision(
            "same_subject",
            "answered",
            pending_source_turn_id="stale-question",
        ),
        resolver_required=True,
        atomic_scope_supported=True,
        expected_projection_version=5,
        next_question="none",
        weather_refresh=False,
        weather_receipt_id=None,
    )

    assert outcome.resolver_decision is not None
    assert outcome.resolver_decision.subject_relation == "same_subject"
    assert outcome.resolver_decision.pending_disposition == "not_addressed"
    assert outcome.resolution is not None
    assert outcome.resolution.location_action == "retain"
    assert outcome.resolution.window_action == "set"
    assert outcome.effective_location == ShopperLocationWeatherScope(
        location="Seattle"
    )
    assert outcome.effective_window == WeatherReceiptWindow(
        start_date=date(2026, 8, 16),
        end_date=date(2026, 8, 16),
    )
    assert outcome.blocks_weather is False
