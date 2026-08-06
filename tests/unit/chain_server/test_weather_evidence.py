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

import pathlib

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


def test_every_failure_says_one_sentence_and_gets_out_of_the_way() -> None:
    """Beyond the horizon, place not found, provider down, weather disabled --
    all four end the same way, and briefly.

    No forecast is the ordinary state, not an event. How to behave without one
    lives in the agent prompt, where it applies whether or not this tool exists
    at all; repeating it here is what produced three differently-worded copies
    of one idea.
    """

    for code in (
        "weather_outside_forecast_horizon",
        "weather_location_not_found",
        "weather_unavailable",
        "weather_disabled",
    ):
        evidence = _format_weather_result(weather_failure(code))

        assert evidence.startswith("WEATHER_UNAVAILABLE")
        assert "Style the occasion." in evidence
        # The shopper hears a sentence, not an error code.
        assert "Do not repeat this code" in evidence
        # Short enough that it cannot take over the reply.
        assert len(evidence.splitlines()) == 1


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


def test_a_disabled_forecast_is_not_registered_at_all() -> None:
    """Off means absent, not present-and-failing.

    While the tool merely returned a failure, the model still called it every
    time, and 2 of 5 replies opened by apologising for a forecast the shopper
    had never asked about. An unregistered tool cannot be called, so there is
    nothing to explain away and no latency spent. Weather ships off, so this
    is the shipped behaviour.

    The policy registry still describes the capability -- only registration is
    conditional -- which is what keeps policy and skill frontmatter honest.
    """

    from chain_server.src.tool_policy import (
        SHOPPING_TOOL_POLICIES,
        validate_registered_tool_names,
    )

    assert "get_weather_forecast_tool" in SHOPPING_TOOL_POLICIES

    registered = set(SHOPPING_TOOL_POLICIES) - {"get_weather_forecast_tool"}
    validate_registered_tool_names(
        registered, disabled=("get_weather_forecast_tool",)
    )


def test_switching_one_tool_off_does_not_switch_the_guard_off() -> None:
    """The exact-match guard is what forces policy, runtime and frontmatter to
    be activated together. Making one tool optional must not weaken it."""

    import pytest

    from chain_server.src.tool_policy import (
        SHOPPING_TOOL_POLICIES,
        validate_registered_tool_names,
    )

    with pytest.raises(ValueError, match="missing="):
        validate_registered_tool_names(
            set(), disabled=("get_weather_forecast_tool",)
        )

    with pytest.raises(ValueError, match="unexpected="):
        validate_registered_tool_names(
            set(SHOPPING_TOOL_POLICIES) | {"invented_tool"},
        )


def test_a_plainly_out_of_range_date_costs_no_provider_call() -> None:
    """Out of horizon is the common case, not the edge.

    Most event shopping happens more than fifteen days ahead, so "a wedding in
    Cancun in November" was the single most-wasted call in the feature: it
    fetched, was billed, and only then failed the horizon check.

    The authoritative check stays after the response, because only the
    provider knows the location's local today. This one uses UTC with a day of
    slack either side, so it can only reject windows no timezone could bring
    into range.
    """

    from datetime import timedelta

    from chain_server.src.weather import (
        VisualCrossingWeatherClient,
        WeatherConfig,
        WeatherRequest,
    )

    calls: list[str] = []

    class _Session:
        def get(self, url, **kwargs):  # pragma: no cover - must not run
            calls.append(url)
            raise AssertionError("a provider call was made for an impossible date")

    now = datetime(2026, 8, 6, 12, tzinfo=timezone.utc)
    client = VisualCrossingWeatherClient(
        WeatherConfig(enabled=True),
        "key",
        session=_Session(),
        clock=lambda: now,
    )

    far_future = (now.date() + timedelta(days=90)).isoformat()
    outcome = client.get_forecast(
        WeatherRequest(location="Cancun", date=far_future)
    )

    assert not outcome.ok
    assert outcome.code == "weather_outside_forecast_horizon"
    assert calls == []


def test_a_date_inside_the_horizon_still_reaches_the_provider() -> None:
    """The cheap pre-check must not start rejecting real work."""

    from datetime import timedelta

    from chain_server.src.weather import (
        VisualCrossingWeatherClient,
        WeatherConfig,
        WeatherRequest,
    )

    reached: list[str] = []

    class _Session:
        def get(self, url, **kwargs):
            reached.append(url)
            raise RuntimeError("stop here; reaching the provider is the assertion")

    now = datetime(2026, 8, 6, 12, tzinfo=timezone.utc)
    client = VisualCrossingWeatherClient(
        WeatherConfig(enabled=True),
        "key",
        session=_Session(),
        clock=lambda: now,
    )

    soon = (now.date() + timedelta(days=5)).isoformat()
    try:
        client.get_forecast(WeatherRequest(location="Cancun", date=soon))
    except RuntimeError:
        pass

    assert len(reached) == 1


def test_no_date_asks_rather_than_forecasting_today() -> None:
    """"A wedding in Cancun" must not become today's weather.

    The library treats a missing date as local today, which is right for "what
    is it like there now" and wrong for the only thing a shopper asks. Enforced
    at the tool boundary rather than in the request model, because the library
    has other callers and its today-mode is a documented contract.
    """

    from chain_server.src.response_format import WEATHER_NO_DATE

    assert "no date was given" in WEATHER_NO_DATE
    assert "today is not what the shopper is dressing for" in WEATHER_NO_DATE
    # A styling question, not a request for a parameter, and never on its own.
    assert "rather than as a request for a parameter" in WEATHER_NO_DATE
    assert "grounded starting point in the same reply" in WEATHER_NO_DATE


def test_the_tool_says_when_not_to_call_it() -> None:
    """A description that says only what a tool does gets called whenever it
    might vaguely apply. "Wedding in Cancun in November" called it, because
    nothing said a forecast horizon exists.

    This lives on the tool because the decision point is the call itself.
    """

    from chain_server.src import deepagents_runtime as runtime_mod

    source = pathlib.Path(runtime_mod.__file__).read_text()

    assert "Do not call it otherwise" in source
    # The four refusals, each measured or reasoned in the contract.
    assert "Today is not what they are dressing for" in source
    assert "further out than about 15 days" in source
    assert "a city yes, a country no" in source
    assert "No place, or nothing they are dressing for" in source
    # Travel date is not the dressing date.
    assert "not the one they travel on" in source



def test_the_no_date_guard_is_actually_enforced() -> None:
    """Not just declared. Deleting the guard from the tool left every test
    passing, because the message was asserted and the check sat in a closure
    nothing could reach."""

    from chain_server.src.response_format import weather_call_needs_a_date

    assert weather_call_needs_a_date(None, None, None)
    assert not weather_call_needs_a_date(date(2026, 8, 15), None, None)
    assert not weather_call_needs_a_date(
        None, date(2026, 8, 15), date(2026, 8, 16)
    )
