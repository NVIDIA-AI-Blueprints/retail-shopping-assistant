# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a forecast is allowed to do to a reply.

A forecast is the one piece of evidence in this assistant that comes from
outside the shop, costs money, and sounds authoritative. It establishes what
the weather will be and nothing whatsoever about a product: no item becomes
warm, waterproof or suitable because it is cold outside.

Failure is the ordinary case rather than the edge. The horizon is fifteen days
and most event shopping happens further out than that, so "no forecast" has to
degrade into styling the occasion, never into failing the turn.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from chain_server.src.response_format import (
    WEATHER_BUDGET_EXHAUSTED,
    WEATHER_CALLS_PER_TURN,
    _format_weather_result,
)
from chain_server.src.weather import (
    WeatherAttribution,
    WeatherDay,
    WeatherRequestedWindow,
    WeatherResult,
    weather_failure,
)


def _result() -> WeatherResult:
    return WeatherResult(
        fetched_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
        requested_window=WeatherRequestedWindow(
            start_date=date(2026, 8, 15), end_date=date(2026, 8, 15)
        ),
        resolved_location="Cancun, Quintana Roo, Mexico",
        timezone="America/Cancun",
        days=[
            WeatherDay(
                date=date(2026, 8, 15),
                condition="rain",
                precipitation_probability_pct=70.0,
                precipitation_types=["rain"],
                temperature_low_f=78.0,
                temperature_high_f=89.0,
            )
        ],
        attribution=WeatherAttribution(
            label="Weather data provided by Visual Crossing",
            url="https://www.visualcrossing.com/",
        ),
    )


def test_a_forecast_says_what_the_weather_is_and_nothing_about_a_product() -> None:
    """The whole risk of this feature in one assertion.

    Knowing it will rain supplies a plausible-sounding reason to call a
    cashmere sweater warm enough or a leather bag water-resistant. The catalog
    confirms neither, and the forecast confirms nothing about any item.
    """

    evidence = _format_weather_result(_result())

    assert "nothing about any product" in evidence
    assert "never makes an item warm, waterproof or suitable" in evidence
    assert "styling judgement" in evidence


def test_the_forecast_itself_is_rendered() -> None:
    evidence = _format_weather_result(_result())

    assert "2026-08-15: rain" in evidence
    assert "78-89F" in evidence
    assert "precipitation 70%" in evidence


def test_the_resolved_place_is_named_so_it_can_be_corrected() -> None:
    """"Springfield" is dozens of places and the provider picks one silently.

    Same shape as the audience disclosure: state what was assumed, invite the
    correction.
    """

    evidence = _format_weather_result(_result())

    assert "Cancun, Quintana Roo, Mexico" in evidence
    assert "correct you" in evidence


def test_attribution_travels_with_the_data() -> None:
    """Required by the provider's terms wherever weather, or anything derived
    from it, is shown. Carried on the evidence so a reply cannot show a
    forecast without it."""

    evidence = _format_weather_result(_result())

    assert "Weather data provided by Visual Crossing" in evidence
    assert "https://www.visualcrossing.com/" in evidence
    assert "an estimate, not a guarantee" in evidence
    assert "never a safety warning" in evidence


def test_every_failure_degrades_to_styling_the_occasion() -> None:
    """Beyond the horizon, place not found, provider down, weather disabled --
    all four end the same way, because a turn that cannot see the weather is
    still a turn that can dress someone."""

    for code in (
        "weather_outside_forecast_horizon",
        "weather_location_not_found",
        "weather_unavailable",
        "weather_disabled",
    ):
        evidence = _format_weather_result(weather_failure(code))

        assert evidence.startswith("WEATHER_UNAVAILABLE")
        assert "Style the occasion instead" in evidence
        assert "do not guess or infer the weather" in evidence
        # The shopper hears a sentence, not an error code.
        assert "do not repeat this code" in evidence
        # And the gap is still named: losing the forecast must not also lose
        # the honest "you will want a coat we don't stock".
        assert "does not stock" in evidence


def test_the_turn_has_a_forecast_budget() -> None:
    """A paid external call, and one shopper is at one event on one date."""

    assert WEATHER_CALLS_PER_TURN == 2
    assert "style the occasion" in WEATHER_BUDGET_EXHAUSTED
    assert "do not guess the weather" in WEATHER_BUDGET_EXHAUSTED


def test_the_budget_is_actually_enforced() -> None:
    """Not just declared. Deleting the check from the tool left every test
    passing, because the constant was asserted and the enforcement was buried
    in a closure nothing could reach."""

    from chain_server.src.response_format import claim_weather_call
    from chain_server.src.turn_scope import TurnScope

    scope = TurnScope()
    granted = [claim_weather_call(scope) for _ in range(WEATHER_CALLS_PER_TURN + 3)]

    assert granted[:WEATHER_CALLS_PER_TURN] == [True] * WEATHER_CALLS_PER_TURN
    assert not any(granted[WEATHER_CALLS_PER_TURN:])
    assert scope.weather_calls == WEATHER_CALLS_PER_TURN


def test_the_rewriter_may_not_strip_the_attribution() -> None:
    """The grounding editor removes unsupported text, and attribution reads
    like clutter. The provider's terms make it mandatory wherever weather or
    anything derived from it is shown, so the stage that can delete it is told
    it may not."""

    from chain_server.src import deepagents_runtime as runtime_mod

    prompt = runtime_mod._GROUNDING_EDITOR_SYSTEM_PROMPT

    assert "Keep the\n  provider attribution and its link" in prompt
    assert "removing it as clutter is not an option" in prompt
    assert "A\n  forecast never confirms a product property." in prompt
