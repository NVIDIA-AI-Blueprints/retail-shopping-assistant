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

So this is a parameter: the model has to quote the words, and the quotation is
checked against what the shopper actually said.

The check was first written against the current turn alone, which is one turn
narrower than the defect. Nine turns into planning one trip to Cancun, "will I
need a jacket in the evening" names no city -- and the refusal produced "I
don't have a live forecast for Cancun, so I can't tell you for certain what
the evenings will be like next week", followed by a description of them. Both
halves of the bug this file exists to stop, in the reply the fix caused.

What made the forecast for Rome wrong was not the age of the citation. The
shopper had said it was going to snow, and "when we get back" is home rather
than the city of the trip. Those are judgments about meaning and they are
stated on the field. The record the code checks is now every turn the shopper
has spoken, which is what stops an invented place and nothing more.
"""

from __future__ import annotations

import pytest
from chain_server.src.response_format import WeatherForecastInput
from pydantic import ValidationError


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


def test_the_field_says_which_place_and_names_what_disqualifies_one() -> None:
    """The description is the whole mechanism; a vague one is the old rule.

    It used to say "THIS turn" and "Not an earlier turn", which is one turn
    narrower than the defect and refused the case below.
    """

    described = WeatherForecastInput.model_fields[
        "shopper_words_naming_the_place"
    ].description

    assert "asking about NOW" in described
    assert "Usually from THIS turn" in described
    assert "carries that same trip forward" in described
    # The two things that actually made the forecast for Rome wrong.
    assert "when we get back" in described
    assert "no forecast is needed" in described
    assert "ask which place they mean" in described
    # And the failure a refusal is supposed to prevent, not cause.
    assert "do not describe conditions you have not fetched" in described


def test_a_quotation_is_checked_against_what_the_shopper_actually_said() -> None:
    """A required field the model can fill with anything, it will.

    Given the field, one run quoted "Italy" on a turn reading "it's going to
    snow when we get back", and the next quoted "Rome" -- a word the shopper
    had not said in any turn of the conversation. So the citation is checked:
    not what the words mean, only whether they were said.

    The record is every turn the shopper has spoken. Checking the current turn
    alone also refused "will I need a jacket in the evening" nine turns into
    planning one trip to Cancun, and the reply then said there was no live
    forecast for Cancun and described the evenings there anyway. Whether a
    carried-over place is the right one is a judgment about meaning, stated on
    the field; what this function establishes is only that the shopper said
    the words.
    """

    from chain_server.src.lexical_provenance import a_place_the_shopper_named

    snow = "it's going to snow when we get back, what should I wear"
    assert not a_place_the_shopper_named([snow], "Italy")
    assert not a_place_the_shopper_named([snow], "Rome")
    assert not a_place_the_shopper_named([snow], "")
    # Nothing said at all is nothing to quote.
    assert not a_place_the_shopper_named([], "Cancun")

    cancun = "it's in Cancun around mid next week"
    assert a_place_the_shopper_named([cancun], "Cancun")
    assert a_place_the_shopper_named([cancun], "in Cancun around mid next week")
    # Case and punctuation are not what is being judged.
    assert a_place_the_shopper_named([cancun], "cancun")

    # The J12 turn: the city is nine turns back, the current turn has none.
    jacket = "will I need a jacket in the evening"
    assert a_place_the_shopper_named([jacket, "I'm going to Cancun"], "Cancun")
    assert not a_place_the_shopper_named([jacket, "I'm going to Cancun"], "Rome")


def _weather_tool(
    base_config,
    query: str = "it's going to snow when we get back, what should I wear",
    said_earlier: tuple[str, ...] = (),
):
    """The runtime's forecast tool, reached the way a turn reaches it.

    `said_earlier` is what the shopper said on prior turns, because that is
    now part of the record a citation is checked against.
    """

    from unittest.mock import patch

    from chain_server.src.agenttypes import DialogueTurn, State
    from chain_server.src.deepagents_runtime import DeepAgentsRuntime
    from chain_server.src.turn_support import RequestIdentity
    from chain_server.src.weather import WeatherConfig

    base_config.weather = WeatherConfig(enabled=True)
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
                query=query,
                dialogue=[
                    DialogueTurn(
                        sequence=index + 1,
                        shopper_text=text,
                        assistant_text="",
                    )
                    for index, text in enumerate(said_earlier)
                ],
            ),
            identity,
        )
    return {tool.name: tool for tool in captured["tools"]}[
        "get_weather_forecast_tool"
    ]


def test_the_tool_turns_back_a_citation_the_shopper_never_said(
    base_config,
) -> None:
    """Wired to the tool, not only to the reader.

    Deleting the call and keeping the helper left every test passing, which is
    how a check gets written and never runs. This turn is about snow and names
    no place, and nothing was said before it: "Italy" and "Rome" are both
    unsourced here, and both were sent in real runs.
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
        # And it says to carry on rather than to stop and ask. Leading with
        # the question, the assistant answered "I don't have a place to check
        # the weather for... otherwise you already know the weather at your
        # destination and can decide what to wear based on that" -- to a
        # shopper who had said it was going to snow and asked what to wear.
        assert "Carry on and answer them" in refused
        assert "Search for what those conditions call for and show" in refused
        assert "Ask only if you cannot tell what they need" in refused
        # The refusal must not become the thing it prevents. Refused on the
        # J12 jacket turn, the reply said there was no live forecast for
        # Cancun and then described a typical Cancun September.
        assert "do not describe conditions you did not fetch" in refused


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


def test_the_place_may_come_from_an_earlier_turn_of_the_same_trip(
    base_config,
) -> None:
    """The case the current-turn-only check refused.

    Nine turns into planning one trip, "will I need a jacket in the evening"
    names no city. Refused, the reply said there was no live forecast for
    Cancun and described the evenings there anyway -- naming the city it had
    just said it could not look up. The shopper's objection is the right one:
    the city and the dates are both known, so there is no honest sense in
    which a forecast is unavailable.

    Whether a carried-over place is the one they mean stays a judgment, and
    the field carrying the quotation is where it is stated. What changes here
    is only that the record includes what they said before.
    """

    tool = _weather_tool(
        base_config,
        query="will I need a jacket in the evening?",
        said_earlier=(
            "I'm going to Cancun next week, what's the weather like?",
            "show me some dresses for it",
        ),
    )

    answered = str(
        tool.invoke(
            {
                "city": "Cancun",
                "date": "2026-08-29",
                "shopper_words_naming_the_place": "going to Cancun next week",
            }
        )
    )
    assert "WEATHER_PLACE_NOT_STATED" not in answered

    # Invention is still invention, however many turns there are to hide in.
    refused = str(
        tool.invoke(
            {
                "city": "Rome",
                "date": "2026-08-29",
                "shopper_words_naming_the_place": "the wedding in Rome",
            }
        )
    )
    assert "WEATHER_PLACE_NOT_STATED" in refused


def test_the_request_is_the_last_thing_the_prompt_says(base_config) -> None:
    """It was fifth of nine, with eleven thousand characters behind it.

    The last words before the model chose a tool were a product showing from
    turn one. Asked what to wear for snow, it searched for a warm-weather
    wedding dress in black, high-neck, size 2 -- the constraints of a turn six
    earlier, whose reply is quoted in the history that follows the request.
    """

    from chain_server.src.agenttypes import State
    from chain_server.src.deepagents_runtime import DeepAgentsRuntime
    from chain_server.src.turn_support import RequestIdentity

    runtime = DeepAgentsRuntime(base_config)
    state = State(
        user_id=1,
        query="it's going to snow when we get back, what should I wear",
    )
    state.context = "RECENT CONVERSATION:\n" + ("older turns\n" * 400)

    message = runtime._build_user_message(
        state,
        RequestIdentity(
            request_id="r1",
            session_id="s1",
            conversation_id="c1",
            cart_id="cart1",
            context_user_id=1,
            cart_user_id=1,
        ),
    )

    assert message.rstrip().endswith(state.query)
    # And the history is still there, in front of it rather than after it.
    assert "older turns" in message
    assert message.index("older turns") < message.rindex(state.query)
