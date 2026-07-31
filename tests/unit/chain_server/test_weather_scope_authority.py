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
    assert outcome.effective_location == ShopperLocationWeatherScope(
        location="Seattle"
    )
    assert outcome.effective_window == WeatherReceiptWindow(
        start_date=date(2026, 8, 16),
        end_date=date(2026, 8, 16),
    )


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


def test_answered_pending_question_completes_the_exact_binding() -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_pending_location_scope(),
        proposed_resolution=_resolution("set", "retain"),
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
    assert outcome.resolution.location_action == "set"
    assert outcome.resolution.window_action == "retain"
    assert (
        outcome.resolution.complete_pending_source_turn_id
        == "turn-conference"
    )


@pytest.mark.parametrize("pending_disposition", ["answered", "resume_requested"])
def test_pending_control_with_the_wrong_handle_fails_closed(
    pending_disposition: str,
) -> None:
    outcome = compile_weather_scope_authority(
        current_scope=_pending_location_scope(),
        proposed_resolution=None,
        resolver_decision=_resolver_decision(
            (
                "same_subject"
                if pending_disposition == "answered"
                else "unchanged"
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
