#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one explicit, redacted direct weather-client smoke.

The command reads its closed request from ``WEATHER_SMOKE_ZIP`` and optionally
``WEATHER_SMOKE_DATE`` or the complete pair ``WEATHER_SMOKE_START_DATE`` /
``WEATHER_SMOKE_END_DATE``. It never prints the ZIP, dates, resolved location,
forecast values, provider body, key, prepared URL, or raw exception.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from pydantic import ValidationError
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chain_server.src.weather import (  # noqa: E402
    WeatherConfig,
    WeatherFailure,
    WeatherRequest,
    build_weather_client,
    weather_failure,
)


def _weather_config() -> WeatherConfig:
    config_path = REPO_ROOT / "shared/configs/chain_server/config.yaml"
    raw = yaml.safe_load(config_path.read_text())
    weather_data = dict(raw["weather"])
    enabled_override = os.environ.get("WEATHER_ENABLED")
    if enabled_override not in {None, ""}:
        normalized = enabled_override.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError("invalid weather enabled flag")
        weather_data["enabled"] = normalized == "true"
    return WeatherConfig(**weather_data)


def _weather_request() -> WeatherRequest:
    return WeatherRequest(
        zipcode=os.environ.get("WEATHER_SMOKE_ZIP", ""),
        date=os.environ.get("WEATHER_SMOKE_DATE") or None,
        start_date=os.environ.get("WEATHER_SMOKE_START_DATE") or None,
        end_date=os.environ.get("WEATHER_SMOKE_END_DATE") or None,
    )


def _request_shape(request: WeatherRequest) -> tuple[str, int]:
    explicit_window = request.explicit_window()
    if explicit_window is None:
        return "today", 1
    days = (explicit_window[1] - explicit_window[0]).days + 1
    return ("single_date" if days == 1 else "date_range"), days


def _summary(
    *,
    outcome: Any,
    mode: str,
    window_days: int,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "case": "operator_weather_smoke",
        "provider": "visual_crossing",
        "mode": mode,
        "window_days": window_days,
        "ok": outcome.ok,
        "outcome": "success" if outcome.ok else outcome.code,
        "retryable": False if outcome.ok else outcome.retryable,
        "schema_valid": True,
        "latency_ms": round(latency_ms, 1),
    }


def main() -> int:
    started = time.monotonic()
    mode = "invalid"
    window_days = 0
    try:
        config = _weather_config()
    except (KeyError, OSError, ValueError, yaml.YAMLError):
        outcome = weather_failure("weather_config_invalid")
    except Exception:
        outcome = weather_failure("weather_unavailable")
    else:
        try:
            request = _weather_request()
            mode, window_days = _request_shape(request)
        except ValidationError:
            outcome = weather_failure("weather_request_invalid")
        else:
            try:
                outcome = build_weather_client(config).get_forecast(request)
            except Exception:
                outcome = weather_failure("weather_unavailable")

    elapsed_ms = (time.monotonic() - started) * 1000
    print(
        json.dumps(
            _summary(
                outcome=outcome,
                mode=mode,
                window_days=window_days,
                latency_ms=elapsed_ms,
            ),
            sort_keys=True,
        )
    )
    return 0 if not isinstance(outcome, WeatherFailure) else 1


if __name__ == "__main__":
    raise SystemExit(main())
