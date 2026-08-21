"""The assistant must know what day it is.

The weather tool may only be called for a window "within about 15 days of
TODAY", and every shopper phrases dates their own way -- "this weekend", "next
week". The prompt never said what today was, so none of that could be resolved:
"we're going to Italy first, what do I wear at the weekend" fetched no forecast
and the reply asserted warm weather anyway.
"""

import re
from datetime import datetime, timezone

from chain_server.src.deepagents_runtime import _today_for_the_shopper


def test_today_is_the_real_date_not_the_build_date() -> None:
    """Read at request time: an image running a week would otherwise date
    every conversation to the day it was built."""

    rendered = _today_for_the_shopper()
    now = datetime.now(timezone.utc)
    assert now.strftime("%d") in rendered
    assert now.strftime("%B") in rendered
    assert now.strftime("%Y") in rendered


def test_it_reads_like_a_person_wrote_it() -> None:
    rendered = _today_for_the_shopper()
    assert re.match(r"^[A-Z][a-z]+day \d{2} [A-Z][a-z]+ \d{4}$", rendered), rendered


def test_the_prompt_states_the_date_and_what_it_is_for() -> None:
    source = open("chain_server/src/deepagents_runtime.py").read()
    assert "TODAY IS {_today_for_the_shopper()}" in source
    block = source[source.index("TODAY IS") :][:600]
    assert "fifteen days" in block, "the forecast window is counted from today"
    assert "only date you know" in block


def test_a_country_is_forecast_and_disclosed_rather_than_refused() -> None:
    """Refusing to call for a country left the model asserting the weather
    instead, which is worse than either asking or calling."""

    source = open("chain_server/src/deepagents_runtime.py").read()
    weather = source[source.index("def get_weather_forecast_tool") :][:3000]
    assert "capital or\n            largest city" in weather
    assert "never do is describe weather you did not fetch" in weather
    assert "Anything broader than a city, per above. Ask which city." not in weather
