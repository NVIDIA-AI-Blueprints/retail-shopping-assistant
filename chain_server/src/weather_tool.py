# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dormant LangChain wrapper for the typed weather client."""

from __future__ import annotations

from datetime import date as CalendarDate
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from .weather import WeatherClient, WeatherRequest, weather_failure


def get_weather_forecast_tool(client: WeatherClient) -> BaseTool:
    """Build the directly testable tool without registering it with an agent."""

    def get_weather_forecast(
        location: str,
        date: CalendarDate | None = None,
        start_date: CalendarDate | None = None,
        end_date: CalendarDate | None = None,
    ) -> dict[str, Any]:
        request = WeatherRequest(
            location=location,
            date=date,
            start_date=start_date,
            end_date=end_date,
        )
        return client.get_forecast(request).model_dump(mode="json")

    return StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast_tool",
        description=(
            "Live daily forecast for one place. Call it only when the "
            "shopper named a place, or a saved location was disclosed to "
            "them. Pass the place in their own words -- \"Cancun\", \"Napa, "
            "CA\", a postal code -- and resolve any relative date against "
            "TODAY before calling: supply no date for today, one exact ISO "
            "date, or a complete inclusive ISO start/end range. Never invent "
            "a place, never send coordinates or shopper data, and never send "
            "a relative date. Forecasts reach about 15 days; beyond that this "
            "returns a failure rather than a guess, and you should style the "
            "occasion instead."
        ),
        args_schema=WeatherRequest,
        return_direct=False,
        handle_validation_error=lambda _error: weather_failure(
            "weather_request_invalid"
        ).model_dump_json(),
    )
