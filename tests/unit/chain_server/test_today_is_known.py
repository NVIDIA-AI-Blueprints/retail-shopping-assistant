"""The assistant must know what day it is.

The weather tool may only be called for a window "within about 15 days of
TODAY", and every shopper phrases dates their own way -- "this weekend", "next
week". The prompt never said what today was, so none of that could be resolved:
"we're going to Italy first, what do I wear at the weekend" fetched no forecast
and the reply asserted warm weather anyway.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from chain_server.src.deepagents_runtime import _today_for_the_shopper

# Resolved from this file, not the working directory: CI runs pytest with
# `working-directory: tests`, where a path relative to the repo root does not
# exist. The rest of the suite already does this.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_today_is_the_real_date_not_the_build_date() -> None:
    """Read at request time: an image running a week would otherwise date
    every conversation to the day it was built."""

    rendered = _today_for_the_shopper()
    now = datetime.now(UTC)
    assert now.strftime("%d") in rendered
    assert now.strftime("%B") in rendered
    assert now.strftime("%Y") in rendered


def test_it_reads_like_a_person_wrote_it() -> None:
    rendered = _today_for_the_shopper()
    assert re.match(r"^[A-Z][a-z]+day \d{2} [A-Z][a-z]+ \d{4}$", rendered), rendered


def test_the_prompt_states_the_date_and_what_it_is_for() -> None:
    source = (_REPO_ROOT / "chain_server/src/deepagents_runtime.py").read_text()
    assert "TODAY IS {_today_for_the_shopper()}" in source
    block = source[source.index("TODAY IS") :][:600]
    assert "only date you know" in block
    # The date itself is unconditional: "the wedding is next weekend" needs
    # resolving whether or not this deployment has a forecast tool. The
    # fifteen-day window is not -- it describes a tool, and it is asserted
    # against the assembled prompt below rather than the source, because the
    # source now holds both branches.
    assert "fifteen days" not in block


def _prompts_either_way(base_config) -> tuple[str, str]:
    """Build once, toggle the flag, restore it.

    `runtime.config` is the same object the fixture handed in, so mutating its
    weather section leaked into the next construction and blew up building the
    weather client. Toggle in place and put it back.
    """

    from types import SimpleNamespace

    from chain_server.src import deepagents_runtime as runtime_mod

    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    original = getattr(runtime.config, "weather", None)
    try:
        runtime.config.weather = SimpleNamespace(enabled=True)
        on = runtime._system_prompt()
        runtime.config.weather = SimpleNamespace(enabled=False)
        off = runtime._system_prompt()
    finally:
        runtime.config.weather = original
    return on, off


def test_the_forecast_window_is_stated_only_when_the_tool_exists(
    base_config,
) -> None:
    """Weather ships off, and the tool is then not registered at all.

    The prompt kept describing it regardless: about a thousand characters
    telling the model when it may call a forecast, and to fetch one before
    fanning out searches, on a deployment where no such tool is offered. That
    is the defect this file's own comments record fixing once, for the
    framework's base prompt.
    """

    on, off = _prompts_either_way(base_config)

    assert "TODAY IS" in on and "TODAY IS" in off
    assert "only date you know" in on and "only date you know" in off

    assert "fifteen days" in on
    assert "fifteen days" not in off


def test_the_ordering_rule_ships_with_the_grant_rather_than_the_prompt(
    base_config,
) -> None:
    """A flag that approximates "was this request granted the tool" is worse
    than the grant, which answers it exactly.

    The ordering rule was a sub-bullet of the search fan-out rule, gated on
    the deployment flag. That kept it off a deployment with no weather tool,
    and still sent it to every request on a weather deployment that was not
    granted one -- the activation step, a cart read, a policy question. Worse,
    a turn that fans out to no product roles has no fan-out to go before, so
    the only statement of when to call read as inapplicable to the shape that
    most needs it: "going to Cancun next week, what's the weather like".
    """

    from chain_server.src.deepagents_runtime import DeepAgentsRuntime

    on, off = _prompts_either_way(base_config)
    section = DeepAgentsRuntime._forecast_prompt_section()

    assert "look the weather" not in on
    assert "look the weather" not in off
    assert "look the weather" in section
    assert "the forecast never gets asked for" in section
    # And the half the fan-out bullet could not say: a turn with nothing to
    # search is still a turn that should fetch.
    assert "whether the turn also asks for products" in section
    # And the rule that outranks all of it: a shopper who states the
    # conditions has answered the question, so there is nothing to look up.
    assert "told you the conditions has already answered" in section


def test_a_country_is_forecast_and_disclosed_rather_than_refused() -> None:
    """Refusing to call for a country left the model asserting the weather
    instead, which is worse than either asking or calling."""

    source = (_REPO_ROOT / "chain_server/src/deepagents_runtime.py").read_text()
    # The docstring grew when a bare conditions question became a call, when a
    # carried-over place stopped being disqualified by its age, and again when
    # the shopper's own statement of the conditions moved to the top as the
    # rule that outranks the rest. The window has to reach past all of that to
    # the country paragraph it is actually about.
    weather = source[source.index("def get_weather_forecast_tool") :][:6500]
    assert "capital or\n            largest city" in weather
    assert "never do is describe weather you did not fetch" in weather
    assert "Anything broader than a city, per above. Ask which city." not in weather
