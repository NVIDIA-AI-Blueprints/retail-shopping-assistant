# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import requests

from chain_server.src.shopper_profiles import (
    ShopperProfilesClient,
    ShopperProfilesError,
)


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload


class _Session:
    def __init__(
        self,
        response: _Response | None = None,
        error: requests.RequestException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, timeout: float) -> _Response:
        self.calls.append({"url": url, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _profile() -> dict[str, str]:
    return {
        "shopper_profile_id": "shopper_alex",
        "display_name": "Alex",
        "shopper_type": "occasion_driven_explorer",
        "behavior": (
            "Gives occasion and vibe first, answers concise clarification, "
            "then asks for a complete look."
        ),
        "zipcode": "98101",
    }


def _profiles() -> list[dict[str, str]]:
    return [
        _profile(),
        {
            "shopper_profile_id": "shopper_casey",
            "display_name": "Casey",
            "shopper_type": "strict_budget_style_mixer",
            "behavior": "Keeps both the complete look and its total budget in view.",
            "zipcode": "85004",
        },
        {
            "shopper_profile_id": "shopper_jordan",
            "display_name": "Jordan",
            "shopper_type": "impatient_decisive",
            "behavior": "Uses concise references and changes cart decisions.",
            "zipcode": "10001",
        },
        {
            "shopper_profile_id": "shopper_morgan",
            "display_name": "Morgan",
            "shopper_type": "skeptical_researcher",
            "behavior": "Checks product evidence before deciding.",
            "zipcode": "60601",
        },
        {
            "shopper_profile_id": "shopper_riley",
            "display_name": "Riley",
            "shopper_type": "iterative_refiner",
            "behavior": "Refines one part of an existing direction.",
            "zipcode": "33130",
        },
    ]


def test_client_lists_typed_profiles() -> None:
    list_session = _Session(_Response(_profiles()))
    list_client = ShopperProfilesClient(
        "http://memory:8011/",
        timeout_seconds=4,
        session=list_session,
    )

    result = list_client.list_profiles()

    assert result[0].shopper_profile_id == "shopper_alex"
    assert result[0].zipcode == "98101"
    assert list_session.calls == [
        {"url": "http://memory:8011/shopper-profiles", "timeout": 4}
    ]


@pytest.mark.parametrize(
    "response,error_code,status_code,retryable",
    [
        (
            _Response({}, status_code=404),
            "shopper_profiles_unavailable",
            404,
            False,
        ),
        (
            _Response({}, status_code=503),
            "shopper_profiles_unavailable",
            503,
            True,
        ),
        (
            _Response([]),
            "shopper_profiles_response_invalid",
            200,
            False,
        ),
    ],
)
def test_client_errors_are_stable(
    response: _Response,
    error_code: str,
    status_code: int,
    retryable: bool,
) -> None:
    client = ShopperProfilesClient("http://memory", session=_Session(response))

    with pytest.raises(ShopperProfilesError) as caught:
        client.list_profiles()

    assert caught.value.code == error_code
    assert caught.value.status_code == status_code
    assert caught.value.retryable is retryable


def test_client_maps_transport_failure_without_exposing_request_details() -> None:
    client = ShopperProfilesClient(
        "http://memory",
        session=_Session(error=requests.ConnectionError("private host")),
    )

    with pytest.raises(
        ShopperProfilesError,
        match="temporarily unavailable",
    ) as caught:
        client.list_profiles()

    assert caught.value.code == "shopper_profiles_request_failed"
    assert caught.value.retryable is True
