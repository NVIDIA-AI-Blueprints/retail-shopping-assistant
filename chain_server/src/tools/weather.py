# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The destination-weather tool the agent calls for a forecast."""

from __future__ import annotations

from datetime import date as CalendarDate
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ..agenttypes import State
from ..lexical_provenance import a_place_the_shopper_named
from ..turn_scope import TurnScope
from ..weather import WeatherRequest

if TYPE_CHECKING:
    from ..deepagents_runtime import DeepAgentsRuntime


WEATHER_PLACE_NOT_STATED = (
    "WEATHER_PLACE_NOT_STATED: no forecast -- the words you quoted as naming "
    "the place are not in anything the shopper has said, in this turn or any "
    "earlier one.\n"
    "If they did name a place, here or on an earlier turn of this same trip, "
    "quote their actual words and call again.\n"
    "Carry on and answer them either way -- a forecast was not the whole "
    "request. If they said what the conditions will be -- \"it's going to "
    "snow when we get back\" -- that is the answer to the weather question, "
    "they are the authority on their own trip, and you have everything you "
    "need. Search for what those conditions call for and show it.\n"
    "Ask only if you cannot tell what they need at all, and then ask for the "
    "one thing you are missing. Do not end the turn on a question about a "
    "place when they have already told you the weather, and do not tell them "
    "they can work it out themselves. Above all, do not describe conditions "
    "you did not fetch: typical, seasonal, usually and this time of year are "
    "not forecasts, and a reply that says the weather is unavailable and then "
    "supplies some is worse than either half alone."
)


#: One event, one date: a turn never needs many forecasts, and each is a paid
#: external call.
WEATHER_CALLS_PER_TURN = 2


WEATHER_NO_DATE = (
    "WEATHER_NEEDS_A_DATE: no forecast was fetched, because no date was given "
    "and today is not what the shopper is dressing for. Ask them, as part of a "
    "styling question rather than as a request for a parameter, and show a "
    "grounded starting point in the same reply."
)


class WeatherForecastInput(BaseModel):
    """The agent-facing shape of a forecast request.

    Separate from `WeatherRequest` for one reason: the field is called `city`.
    Twice the prose form of this rule was ignored -- "going to Italy tomorrow"
    called the tool 5/5 and forecast "Italia" at 74-101F as though a country
    had one temperature -- and a rule the model reads is weaker than a
    parameter it has to fill. `location` invites any place; `city` does not.
    """

    model_config = ConfigDict(extra="forbid")

    city: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description=(
            "One city, town or postal code -- Cancun, Napa CA, 94558. Never a "
            "country, region or coastline: they have no single weather, so ask "
            "the shopper which city instead of calling."
        ),
    )
    #: Where the place came from, as a parameter rather than a rule, for the
    #: reason above. A place carried over from an earlier turn can be right
    #: ("will I need a jacket in the evening", mid-way through planning one
    #: trip) or wrong ("it's going to snow when we get back" -- the shopper has
    #: given the conditions, and "back" is home, not the trip). So the field
    #: asks which place they are asking about now, and names the two things
    #: that disqualify a carried-over one.
    #:
    #: Nothing to quote is the signal. Ask which place they mean.
    shopper_words_naming_the_place: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Quote the shopper's own words naming the place they are asking "
            "about NOW. If they have already said what the weather will be, "
            "they are the authority on their own trip, no forecast is needed "
            "at all, and this is not a call to make -- dress what they told "
            "you. Usually from THIS turn. Words from an earlier turn count "
            "only when this turn carries that same trip forward and puts no "
            "other place in play: nine turns into planning one trip, \"will "
            "I need a jacket in the evening\" is asking about that city. What "
            "never counts is a place they have moved off -- \"when we get "
            "back\" is home, not the city of the trip, and the trip's "
            "forecast contradicts them; served exactly that, the assistant "
            "recommended a satin dress and ballet flats for snow. If you "
            "cannot tell which place they mean, there is nothing to quote: "
            "ask which place they mean instead, and do not describe "
            "conditions you have not fetched."
        ),
    )
    date: CalendarDate | None = Field(
        default=None, description="One exact ISO date, resolved against TODAY."
    )
    start_date: CalendarDate | None = Field(
        default=None, description="Inclusive ISO range start; use with end_date."
    )
    end_date: CalendarDate | None = Field(
        default=None, description="Inclusive ISO range end; use with start_date."
    )


def weather_call_needs_a_date(
    date: Any, start_date: Any, end_date: Any
) -> bool:
    """Whether this call would silently forecast today instead of the event.

    Extracted from the tool closure so it can be tested. Left inside, deleting
    it entirely kept all 1089 tests passing -- the third time today that a
    constant was asserted while the enforcement sat somewhere nothing could
    reach.
    """

    return date is None and start_date is None and end_date is None


WEATHER_BUDGET_EXHAUSTED = (
    "WEATHER_UNAVAILABLE: this turn has already looked up the forecast it is "
    "allowed to. Use what you have and style the occasion; do not guess the "
    "weather."
)


def claim_weather_call(scope: Any) -> bool:
    """Take one of this turn's forecast calls, or report that none are left.

    Extracted from the tool closure so the budget can be tested. It could not
    be before, and deleting the check entirely left all 1081 tests passing --
    a paid external call with no enforced ceiling and nothing to notice.
    """

    with scope.weather_lock:
        if scope.weather_calls >= WEATHER_CALLS_PER_TURN:
            return False
        scope.weather_calls += 1
        return True


def _format_weather_result(result: Any) -> str:
    """Render a forecast, or an honest failure, as evidence for the reply.

    Failures are the common case, not the edge: most event shopping happens
    more than fifteen days ahead, and the horizon is fifteen days. So every
    failure says the same thing -- say plainly that the forecast is not
    available, then style the occasion -- because a turn that cannot see the
    weather is still a turn that can dress someone.
    """

    if not getattr(result, "ok", False):
        # One sentence, because no forecast is the ordinary state rather than
        # an event. Everything about how to behave without one already lives
        # in the agent prompt, where it applies whether or not this tool
        # exists at all.
        return (
            "WEATHER_UNAVAILABLE: "
            + str(getattr(result, "message", "No forecast is available."))
            + " Do not repeat this code. Style the occasion."
        )
    lines = [
        "WEATHER_EVIDENCE: a live forecast for the place and dates below. It "
        "supports what the conditions will be, and nothing about any product. "
        "It never makes an item warm, waterproof or suitable -- say what the "
        "weather is, then reason about the outfit as styling judgement.",
        "STILL_SHOW_THE_CLOTHES: a forecast is not an answer on its own. "
        "Search and show real pieces in the same reply. Live, a forecast turn "
        "returned weather and generic advice with nothing to buy, three times "
        "out of three -- a shop that talked about the weather and forgot to "
        "sell anything.",
        f"PLACE_RESOLVED_BY_PROVIDER: {getattr(result, 'resolved_location', '')}",
    ]
    for day in getattr(result, "days", []) or []:
        parts = [f"{day.date.isoformat()}: {day.condition}"]
        if day.temperature_low_f is not None and day.temperature_high_f is not None:
            parts.append(
                f"{day.temperature_low_f:.0f}-{day.temperature_high_f:.0f}F"
            )
        if day.precipitation_probability_pct is not None:
            parts.append(
                f"precipitation {day.precipitation_probability_pct:.0f}%"
            )
        if day.precipitation_types:
            parts.append("as " + ", ".join(day.precipitation_types))
        lines.append("  " + "; ".join(parts))
    lines.append(
        "SAY_WHICH_PLACE: open by naming the place these numbers are for, as "
        "the place you chose to look up rather than as settled fact, and "
        "invite the correction in the same breath -- \"using Rome for the "
        "forecast; say if you meant somewhere else\". Naming it is not enough "
        "on its own: a shopper who said only \"Italy\" never chose the city "
        "you picked, and live, one who meant Florence was given Rome's "
        "forecast at 73-101F as though it were where they would be. If what "
        "came back is broader than a town -- a country or a region, like "
        "Italia or Toscana -- then these numbers describe that whole area and "
        "nowhere in particular: say so plainly, and ask which city they will "
        "be in before dressing them for the weather."
    )
    # The provider's own label and link travel with the data, so the terms are
    # met by whatever provider answered rather than by a constant here.
    attribution = getattr(result, "attribution", None)
    if attribution is not None:
        lines.append(
            f"REQUIRED_ATTRIBUTION: include \"{attribution.label}\" and the "
            f"link {attribution.url} wherever you use this. A forecast is an "
            "estimate, not a guarantee, and never a safety warning."
        )
    return "\n".join(lines)


def forecast_prompt_section() -> str:
    """When to fetch a forecast, for a request that was granted the tool.

    Its own section rather than part of the search fan-out rule, because a
    turn with nothing to search -- "going to Cancun next week, what's the
    weather like" -- still needs it.

    Held out of the static prompt for the same reason the catalog rules
    are: ordering instructions for a tool the request was not granted are
    unreadable cost.
    """

    return """Forecast ordering:
- A shopper who has told you the conditions has already answered the weather
  question. "It's going to snow when we get back" needs no lookup at all:
  they are the authority on their own trip, and a forecast fetched for
  anywhere else contradicts them. Dress what they said. Measured: that
  sentence produced a forecast for the wedding city two turns earlier, rain
  at 65-82F, and a satin dress with ballet flats for a shopper heading into
  snow. Every rule below is about a shopper asking what the conditions are.
- For a shopper who is asking, whether to look the weather up is answered on
  the tool's own schema, by the place, the date and the window -- not by
  whether the turn also asks for products. A question about the conditions
  somewhere is answered by fetching them, with or without an outfit attached.
- When the turn does fan out to product roles, look the weather up BEFORE that
  fan-out, not after: once the roles are out you are told to stop and
  synthesize, and the forecast never gets asked for. Conditions change which
  pieces you would even search for, so they belong first. Measured: the same
  sentence about a trip fetched a forecast on its own and skipped it entirely
  once it arrived mid-conversation and read as an outfit request."""


def build_weather_tool(
    runtime: DeepAgentsRuntime,
    state: State,
    scope: TurnScope,
):
    """The live forecast tool, spending this turn's forecast budget in `scope`."""

    from langchain_core.tools import tool

    @tool(args_schema=WeatherForecastInput, return_direct=False)
    def get_weather_forecast_tool(
        city: str,
        shopper_words_naming_the_place: str,
        date: CalendarDate | None = None,
        start_date: CalendarDate | None = None,
        end_date: CalendarDate | None = None,
    ) -> str:
        """Live daily forecast for one place, for the dates in question.

        FIRST, AND OVER EVERYTHING BELOW: the shopper outranks this tool.
        If they have told you what the conditions will be -- "it's going
        to snow when we get back" -- the weather question is answered and
        there is nothing to look up. They are the authority on their own
        trip. Dress what they told you and do not call this tool at all.

        Not because the call would be unnecessary, but because it is
        actively worse than no forecast. Asked that, the assistant looked
        up Rome -- the wedding two turns before -- reported rain at 65-82F
        and recommended a satin sheath dress and blush ballet flats to a
        shopper heading into snow. "When we get back" is home, and home is
        not a place the assistant knows. Every clause after this one is
        about a shopper ASKING what the conditions are, never about one
        telling you.

        Call it, without being asked, when all three hold of the question
        you are answering. The shopper named a CITY, town or postal code.
        They named a date or window. That window is within about 15 days
        of TODAY. A destination wedding, a trip, an outdoor event.
        Conditions change what to wear more than anything else about a
        destination.

        Those three are the whole test, and they are a test on the
        question rather than on one turn's wording. Two things follow.

        It does not also have to be an outfit request. "I'm going to
        Cancun next week, what's the weather like" names the place, names
        the window, and is inside it -- so it is a call, and the forecast
        is the entire answer. Asked exactly that, the assistant instead
        replied that it had no live forecast and then described what
        September in Cancun is typically like, which is both a refusal and
        the thing a refusal is supposed to prevent.

        And when they are asking, the place and the dates may have been
        established earlier in the same conversation. Nine turns into
        planning one trip to Cancun, "will I need a jacket in the evening"
        is a question about Cancun on those dates. It was refused for
        naming no city, and the reply then said there was no live forecast
        for Cancun and described the evenings there anyway -- naming the
        city it claimed not to be able to look up. If they are asking, and
        you have the place and the date from the trip under discussion,
        you are not missing a forecast; you have not asked for one yet.
        Say which city and dates the numbers are for, so they can correct
        you.

        That is licence to carry a place forward for a shopper who wants
        conditions they do not have. It is not licence to look one up for
        a shopper who already gave you theirs, and it is not licence for a
        place they have moved off.

        The `city` argument takes a city, town or postal code. A country
        or region has no single weather, so prefer asking which city over
        calling with one. If you do call with something broad, the reply
        must say the numbers cover that whole area and ask which city --
        never present them as the weather where the shopper will be.

        Do not call it otherwise. Specifically:
        - The shopper already said what the weather will be. "It's going to
          snow when we get back" is the answer, and they are the authority
          on their own trip. A forecast cannot improve on it and a forecast
          for somewhere else contradicts it. This is the rule at the top:
          no call, and no lookup of any kind.
        - No date, here or anywhere in the conversation.
          Today is not what they are dressing for; ask instead.
        - A date further out than about 15 days. There is no forecast that
          far ahead, so a call cannot produce anything true.
        - No place at all. A place is the one thing that cannot be
          supplied from anywhere else.

        In each of those cases, name the one thing you are missing and ask
        for it. Do not answer the question anyway from what you know about
        the place: typical, seasonal, this time of year and tends to be are
        not forecasts, and a reply that opens by saying the weather is
        unavailable and then supplies some is the failure above.

        A country or region does not stop you. "We're going to Italy at the
        weekend" was answered with no forecast at all and a flat assertion
        that the weather would be warm -- worse than either asking or
        calling. Call it for the place they named, using its capital or
        largest city when they named a country, then say which city the
        numbers are for and ask whether that is where they will be. What
        you may never do is describe weather you did not fetch.

        Dress the date they are dressing for, not the one they travel on:
        "flying to Rome tomorrow, what do I wear at the weekend" is a
        forecast for the weekend. Resolve relative dates against TODAY
        first -- one exact ISO date, or a complete inclusive ISO start/end
        range -- and never send a relative date or invent a place.
        """

        if not a_place_the_shopper_named(
            (state.query, *(turn.shopper_text for turn in state.dialogue)),
            shopper_words_naming_the_place,
        ):
            return WEATHER_PLACE_NOT_STATED
        if weather_call_needs_a_date(date, start_date, end_date):
            # The library treats a missing date as local today, which is
            # right for "what is it like there now" and wrong for the only
            # thing a shopper asks: "a wedding in Cancun" would silently
            # get today's weather for an event months away. Ask instead.
            return WEATHER_NO_DATE
        if not claim_weather_call(scope):
            return WEATHER_BUDGET_EXHAUSTED
        return _format_weather_result(
            runtime._weather_client.get_forecast(
                WeatherRequest(
                    location=city,
                    date=date,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        )

    return get_weather_forecast_tool
