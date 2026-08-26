"""A forecast call has to quote the words that named the place.

"It's going to snow when we get back, what should I wear" fetched the forecast
for Rome -- the wedding two turns earlier -- and offered warm-weather clothes
to a shopper describing snow. It did that in two runs of three.

The prose rule was already there and already explicit: call it when "the
shopper named a CITY... they named a date or window". It says nothing about
*when* they named it, and a city named two turns ago satisfies it as written.

Four attempts to fix this by rewriting rules measured 0/3, 0/3, 1/3, and one
that fixed the turn while breaking the journey. The file itself records why:
a country was forecast as "Italia" at 74-101F until `location` became `city`,
because a rule the model reads is weaker than a parameter it has to fill.

So this is a parameter. The current turn either names a place or it does not,
and when it does not there is nothing to quote.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chain_server.src.response_format import WeatherForecastInput


def test_the_words_that_named_the_place_are_required() -> None:
    with pytest.raises(ValidationError):
        WeatherForecastInput(city="Rome", date="2026-08-29")
    with pytest.raises(ValidationError):
        WeatherForecastInput(
            city="Rome", date="2026-08-29", shopper_words_naming_the_place=""
        )


def test_a_call_that_quotes_the_turn_is_accepted() -> None:
    request = WeatherForecastInput(
        city="Cancun",
        date="2026-08-29",
        shopper_words_naming_the_place="it's in Cancun around mid next week",
    )

    assert request.city == "Cancun"
    assert "Cancun" in request.shopper_words_naming_the_place


def test_the_field_says_it_wants_this_turn_and_not_an_earlier_one() -> None:
    """The description is the whole mechanism; a vague one is the old rule."""

    described = WeatherForecastInput.model_fields[
        "shopper_words_naming_the_place"
    ].description

    assert "THIS turn" in described
    assert "Not an earlier turn" in described
    assert "ask which place they mean" in described
    # And that a shopper who states the weather has already answered it.
    assert "no forecast is needed" in described


def test_a_quotation_is_checked_against_the_turn_it_claims_to_be_from() -> None:
    """A required field the model can fill with anything, it will.

    Given the field, one run quoted "Italy" on a turn reading "it's going to
    snow when we get back", and the next quoted "Rome" -- a word the shopper
    had not said in any turn of the conversation. So the citation is checked:
    not what the words mean, only whether they were said here.
    """

    from chain_server.src.turn_support import a_place_this_turn_named

    snow = "it's going to snow when we get back, what should I wear"
    assert not a_place_this_turn_named(snow, "Italy")
    assert not a_place_this_turn_named(snow, "Rome")
    assert not a_place_this_turn_named(snow, "")

    cancun = "it's in Cancun around mid next week"
    assert a_place_this_turn_named(cancun, "Cancun")
    assert a_place_this_turn_named(cancun, "in Cancun around mid next week")
    # Case and punctuation are not what is being judged.
    assert a_place_this_turn_named(cancun, "cancun")


def _weather_tool(base_config):
    """The runtime's forecast tool, reached the way a turn reaches it."""

    from unittest.mock import patch

    from chain_server.src.agenttypes import State
    from chain_server.src.deepagents_runtime import DeepAgentsRuntime
    from chain_server.src.turn_support import RequestIdentity
    from chain_server.src.weather import WeatherConfig

    setattr(base_config, "weather", WeatherConfig(enabled=True))
    runtime = DeepAgentsRuntime(base_config)
    captured: dict = {}

    identity = RequestIdentity(
        request_id="r1",
        session_id="s1",
        conversation_id="c1",
        cart_id="cart1",
        context_user_id=1,
        cart_user_id=1,
    )
    with patch("deepagents.create_deep_agent", lambda **kw: captured.update(kw)):
        runtime._create_agent(
            State(
                user_id=1,
                query="it's going to snow when we get back, what should I wear",
            ),
            identity,
        )
    return {tool.name: tool for tool in captured["tools"]}[
        "get_weather_forecast_tool"
    ]


def test_the_tool_turns_back_a_citation_from_another_turn(base_config) -> None:
    """Wired to the tool, not only to the reader.

    Deleting the call and keeping the helper left every test passing, which is
    how a check gets written and never runs. This turn is about snow and names
    no place; "Italy" comes from two turns earlier and "Rome" from nowhere at
    all, and both were sent in real runs.
    """

    tool = _weather_tool(base_config)

    for quoted in ("Italy", "Rome"):
        refused = str(
            tool.invoke(
                {
                    "city": "Rome",
                    "date": "2026-08-29",
                    "shopper_words_naming_the_place": quoted,
                }
            )
        )
        assert "WEATHER_PLACE_NOT_STATED" in refused
        assert "ask which place they mean" in refused


def test_a_citation_from_this_turn_is_not_turned_back(base_config) -> None:
    """The refusal is about provenance and nothing else."""

    tool = _weather_tool(base_config)

    answered = str(
        tool.invoke(
            {
                "city": "Boston",
                "date": "2026-08-29",
                # Words that really are in the query above.
                "shopper_words_naming_the_place": "when we get back",
            }
        )
    )
    assert "WEATHER_PLACE_NOT_STATED" not in answered
