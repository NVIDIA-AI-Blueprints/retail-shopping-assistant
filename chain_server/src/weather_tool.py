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
        zipcode: str,
        date: CalendarDate | None = None,
        start_date: CalendarDate | None = None,
        end_date: CalendarDate | None = None,
    ) -> dict[str, Any]:
        request = WeatherRequest(
            zipcode=zipcode,
            date=date,
            start_date=start_date,
            end_date=end_date,
        )
        return client.get_forecast(request).model_dump(mode="json")

    return StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast_tool",
        description=(
            "Get normalized daily live-forecast evidence for exactly one "
            "five-digit US ZIP. Supply no date for local today, one exact ISO "
            "date, or a complete inclusive ISO start/end range. Never supply "
            "relative dates, prose locations, coordinates, or shopper data."
        ),
        args_schema=WeatherRequest,
        return_direct=False,
        handle_validation_error=lambda _error: weather_failure(
            "weather_request_invalid"
        ).model_dump_json(),
    )
