# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date

import pytest
from pydantic import ValidationError

from shared.weather_receipts import (
    ShopperLocationWeatherScope,
    WeatherReceiptWindow,
)
from shared.weather_scope import (
    CurrentWeatherScope,
    CurrentWeatherScopeResolution,
    WeatherScopeLocationAuthority,
    WeatherScopeUnavailableAuthority,
    WeatherScopeWindowAuthority,
    apply_current_weather_scope_resolution,
    current_weather_scope_source_references,
    effective_resolved_weather_scope_unavailability,
    effective_resolved_weather_scope_values,
)


def _location_authority() -> WeatherScopeLocationAuthority:
    return WeatherScopeLocationAuthority(
        value=ShopperLocationWeatherScope(location="Seattle"),
        source_turn_id="location-turn",
        source_sequence=1,
    )


def _window_authority() -> WeatherScopeWindowAuthority:
    return WeatherScopeWindowAuthority(
        value=WeatherReceiptWindow(
            start_date=date(2026, 8, 16),
            end_date=date(2026, 8, 16),
        ),
        source_turn_id="window-turn",
        source_sequence=2,
    )


def _unavailable(
    component: str,
    sequence: int,
) -> WeatherScopeUnavailableAuthority:
    return WeatherScopeUnavailableAuthority(
        source_turn_id=f"{component}-unavailable-turn",
        source_sequence=sequence,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "revision": 1,
            "location": _location_authority(),
            "location_unavailable": _unavailable("location", 1),
        },
        {
            "revision": 1,
            "window": _window_authority(),
            "window_unavailable": _unavailable("window", 2),
        },
    ],
)
def test_value_and_unavailability_are_mutually_exclusive(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        CurrentWeatherScope.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "revision": 1,
            "location_unavailable": _unavailable("location", 1),
            "pending_question": "event_location",
            "pending_source_turn_id": "pending-turn",
            "pending_source_sequence": 1,
        },
        {
            "revision": 1,
            "window_unavailable": _unavailable("window", 1),
            "pending_question": "event_date",
            "pending_source_turn_id": "pending-turn",
            "pending_source_sequence": 1,
        },
    ],
)
def test_pending_question_requires_an_askable_missing_component(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="askable"):
        CurrentWeatherScope.model_validate(payload)


def test_unavailable_markers_are_source_bound_scope_references() -> None:
    scope = CurrentWeatherScope(
        revision=1,
        location_unavailable=_unavailable("location", 3),
        window_unavailable=_unavailable("window", 4),
    )

    assert current_weather_scope_source_references(scope) == (
        ("location-unavailable-turn", 3),
        ("window-unavailable-turn", 4),
    )


def test_unavailable_action_is_stamped_and_retain_preserves_the_marker() -> None:
    current_scope = CurrentWeatherScope(
        revision=1,
        location=_location_authority(),
        window=_window_authority(),
    )
    unavailable_resolution = CurrentWeatherScopeResolution(
        expected_projection_version=4,
        expected_scope_revision=1,
        location_action="retain",
        window_action="unavailable",
    )

    persisted = apply_current_weather_scope_resolution(
        current_scope,
        unavailable_resolution,
        source_turn_id="date-unavailable-turn",
        source_sequence=3,
    )

    assert persisted.location == current_scope.location
    assert persisted.window is None
    assert persisted.window_unavailable == WeatherScopeUnavailableAuthority(
        source_turn_id="date-unavailable-turn",
        source_sequence=3,
    )
    assert effective_resolved_weather_scope_values(persisted, None) == (
        ShopperLocationWeatherScope(location="Seattle"),
        None,
    )
    assert effective_resolved_weather_scope_unavailability(
        persisted,
        None,
    ) == (False, True)

    retain_resolution = CurrentWeatherScopeResolution(
        expected_projection_version=5,
        expected_scope_revision=2,
        location_action="set",
        window_action="retain",
        location_scope=ShopperLocationWeatherScope(location="Portland"),
    )
    retained = apply_current_weather_scope_resolution(
        persisted,
        retain_resolution,
        source_turn_id="location-update-turn",
        source_sequence=4,
    )

    assert retained.window is None
    assert retained.window_unavailable == persisted.window_unavailable


def test_later_set_and_clear_both_remove_unavailability() -> None:
    current_scope = CurrentWeatherScope(
        revision=2,
        location=_location_authority(),
        window_unavailable=_unavailable("window", 2),
    )
    set_resolution = CurrentWeatherScopeResolution(
        expected_projection_version=5,
        expected_scope_revision=2,
        location_action="retain",
        window_action="set",
        requested_window=WeatherReceiptWindow(
            start_date=date(2026, 8, 23),
            end_date=date(2026, 8, 23),
        ),
    )
    set_scope = apply_current_weather_scope_resolution(
        current_scope,
        set_resolution,
        source_turn_id="date-set-turn",
        source_sequence=3,
    )

    assert set_scope.window is not None
    assert set_scope.window_unavailable is None

    unavailable_again = apply_current_weather_scope_resolution(
        set_scope,
        CurrentWeatherScopeResolution(
            expected_projection_version=6,
            expected_scope_revision=3,
            location_action="retain",
            window_action="unavailable",
        ),
        source_turn_id="date-unavailable-again",
        source_sequence=4,
    )
    cleared = apply_current_weather_scope_resolution(
        unavailable_again,
        CurrentWeatherScopeResolution(
            expected_projection_version=7,
            expected_scope_revision=4,
            location_action="retain",
            window_action="clear",
        ),
        source_turn_id="new-subject-turn",
        source_sequence=5,
    )

    assert cleared.window is None
    assert cleared.window_unavailable is None
    assert effective_resolved_weather_scope_unavailability(cleared, None) == (
        False,
        False,
    )


@pytest.mark.parametrize(
    ("pending_question", "location_action", "window_action"),
    [
        ("event_location", "clear", "retain"),
        ("event_date", "retain", "clear"),
    ],
)
def test_exact_pending_decline_requires_target_unavailable(
    pending_question: str,
    location_action: str,
    window_action: str,
) -> None:
    scope_payload: dict[str, object] = {
        "revision": 2,
        "pending_question": pending_question,
        "pending_source_turn_id": "pending-turn",
        "pending_source_sequence": 2,
    }
    if pending_question == "event_location":
        scope_payload["window"] = _window_authority()
    else:
        scope_payload["location"] = _location_authority()
    current_scope = CurrentWeatherScope.model_validate(scope_payload)
    resolution = CurrentWeatherScopeResolution.model_validate(
        {
            "expected_projection_version": 5,
            "expected_scope_revision": 2,
            "location_action": location_action,
            "window_action": window_action,
            "decline_pending_source_turn_id": "pending-turn",
        }
    )

    with pytest.raises(ValueError, match="pending decline conflict"):
        apply_current_weather_scope_resolution(
            current_scope,
            resolution,
            source_turn_id="decline-turn",
            source_sequence=3,
        )


@pytest.mark.parametrize(
    ("pending_question", "location_action", "window_action"),
    [
        ("event_location", "unavailable", "retain"),
        ("event_date", "retain", "unavailable"),
    ],
)
def test_exact_pending_decline_stamps_unavailable_target(
    pending_question: str,
    location_action: str,
    window_action: str,
) -> None:
    scope_payload: dict[str, object] = {
        "revision": 2,
        "pending_question": pending_question,
        "pending_source_turn_id": "pending-turn",
        "pending_source_sequence": 2,
    }
    if pending_question == "event_location":
        scope_payload["window"] = _window_authority()
    else:
        scope_payload["location"] = _location_authority()
    current_scope = CurrentWeatherScope.model_validate(scope_payload)
    resolution = CurrentWeatherScopeResolution.model_validate(
        {
            "expected_projection_version": 5,
            "expected_scope_revision": 2,
            "location_action": location_action,
            "window_action": window_action,
            "decline_pending_source_turn_id": "pending-turn",
        }
    )

    persisted = apply_current_weather_scope_resolution(
        current_scope,
        resolution,
        source_turn_id="decline-turn",
        source_sequence=3,
    )

    assert persisted.pending_question is None
    if pending_question == "event_location":
        assert persisted.location_unavailable is not None
        assert persisted.window == current_scope.window
    else:
        assert persisted.window_unavailable is not None
        assert persisted.location == current_scope.location


@pytest.mark.parametrize(
    ("pending_question", "location_action", "window_action"),
    [
        ("event_location", "unavailable", "retain"),
        ("event_date", "retain", "unavailable"),
    ],
)
def test_pending_unavailability_without_decline_handle_is_rejected(
    pending_question: str,
    location_action: str,
    window_action: str,
) -> None:
    scope_payload: dict[str, object] = {
        "revision": 2,
        "pending_question": pending_question,
        "pending_source_turn_id": "pending-turn",
        "pending_source_sequence": 2,
    }
    if pending_question == "event_location":
        scope_payload["window"] = _window_authority()
    else:
        scope_payload["location"] = _location_authority()
    current_scope = CurrentWeatherScope.model_validate(scope_payload)
    resolution = CurrentWeatherScopeResolution(
        expected_projection_version=5,
        expected_scope_revision=2,
        location_action=location_action,
        window_action=window_action,
    )

    with pytest.raises(ValueError, match="requires exact handle"):
        apply_current_weather_scope_resolution(
            current_scope,
            resolution,
            source_turn_id="unbound-unavailable-turn",
            source_sequence=3,
        )


@pytest.mark.parametrize(
    ("marker_field", "pending_question"),
    [
        ("location_unavailable", "event_date"),
        ("window_unavailable", "event_location"),
    ],
)
def test_any_unavailable_component_forbids_every_pending_weather_question(
    marker_field: str,
    pending_question: str,
) -> None:
    with pytest.raises(ValueError, match="every component to be askable"):
        CurrentWeatherScope.model_validate(
            {
                "revision": 2,
                marker_field: {
                    "source_turn_id": "unavailable-turn",
                    "source_sequence": 1,
                },
                "pending_question": pending_question,
                "pending_source_turn_id": "pending-turn",
                "pending_source_sequence": 2,
            }
        )


@pytest.mark.parametrize(
    ("pending_question", "location_action", "window_action"),
    [
        ("event_location", "unavailable", "set"),
        ("event_date", "set", "unavailable"),
    ],
)
def test_exact_pending_supersession_allows_new_subject_unavailability(
    pending_question: str,
    location_action: str,
    window_action: str,
) -> None:
    scope_payload: dict[str, object] = {
        "revision": 2,
        "pending_question": pending_question,
        "pending_source_turn_id": "old-subject-turn",
        "pending_source_sequence": 2,
    }
    if pending_question == "event_location":
        scope_payload["window"] = _window_authority()
    else:
        scope_payload["location"] = _location_authority()
    current_scope = CurrentWeatherScope.model_validate(scope_payload)
    resolution_payload: dict[str, object] = {
        "expected_projection_version": 5,
        "expected_scope_revision": 2,
        "location_action": location_action,
        "window_action": window_action,
        "supersede_pending_source_turn_id": "old-subject-turn",
    }
    if location_action == "set":
        resolution_payload["location_scope"] = {
            "kind": "shopper_provided_location",
            "location": "Seattle",
        }
    if window_action == "set":
        resolution_payload["requested_window"] = {
            "start_date": "2026-08-16",
            "end_date": "2026-08-16",
        }
    resolution = CurrentWeatherScopeResolution.model_validate(
        resolution_payload
    )

    persisted = apply_current_weather_scope_resolution(
        current_scope,
        resolution,
        source_turn_id="new-subject-turn",
        source_sequence=3,
    )

    assert persisted.pending_question is None
    if pending_question == "event_location":
        assert persisted.location_unavailable is not None
        assert persisted.window is not None
    else:
        assert persisted.location is not None
        assert persisted.window_unavailable is not None


def test_pending_supersession_rejects_wrong_handle() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_location",
            "pending_source_turn_id": "old-subject-turn",
            "pending_source_sequence": 2,
            "window": _window_authority(),
        }
    )
    resolution = CurrentWeatherScopeResolution(
        expected_projection_version=5,
        expected_scope_revision=2,
        location_action="unavailable",
        window_action="clear",
        supersede_pending_source_turn_id="wrong-turn",
    )

    with pytest.raises(ValueError, match="supersession conflict"):
        apply_current_weather_scope_resolution(
            current_scope,
            resolution,
            source_turn_id="new-subject-turn",
            source_sequence=3,
        )


def test_pending_supersession_cannot_retain_old_subject_components() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_location",
            "pending_source_turn_id": "old-subject-turn",
            "pending_source_sequence": 2,
            "window": _window_authority(),
        }
    )
    resolution = CurrentWeatherScopeResolution(
        expected_projection_version=5,
        expected_scope_revision=2,
        location_action="clear",
        window_action="retain",
        supersede_pending_source_turn_id="old-subject-turn",
    )

    with pytest.raises(ValueError, match="supersession conflict"):
        apply_current_weather_scope_resolution(
            current_scope,
            resolution,
            source_turn_id="new-subject-turn",
            source_sequence=3,
        )
