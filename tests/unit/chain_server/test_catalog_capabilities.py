# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import requests

from chain_server.src.catalog_capabilities import CatalogCapabilitiesClient


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls = []

    def get(self, url, timeout):
        self.calls.append({"url": url, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.response


def test_fetches_catalog_capabilities() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "catalog_id": "fashion",
                "retrieval_modes": ["text", "image"],
                "image_search_enabled": True,
                "filters": {
                    "category": {
                        "type": "enum",
                        "operators": ["in"],
                        "source_fields": ["subcategory"],
                        "values": ["bag"],
                    }
                },
                "soft_facets": {},
            }
        )
    )
    client = CatalogCapabilitiesClient(
        "http://catalog-retriever:8010/",
        timeout_seconds=3,
        session=session,
    )

    capabilities = client.get()

    assert capabilities.catalog_id == "fashion"
    assert capabilities.filters["category"].values == ["bag"]
    assert session.calls == [
        {"url": "http://catalog-retriever:8010/capabilities", "timeout": 3}
    ]


def test_caches_capabilities_until_forced_refresh() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "catalog_id": "fashion",
                "retrieval_modes": ["text"],
                "image_search_enabled": False,
                "filters": {},
                "soft_facets": {},
            }
        )
    )
    client = CatalogCapabilitiesClient("http://catalog", session=session)

    assert client.get() is client.get()
    assert len(session.calls) == 1

    client.get(force_refresh=True)

    assert len(session.calls) == 2


def test_unavailable_capabilities_return_empty_contract() -> None:
    client = CatalogCapabilitiesClient(
        "http://catalog",
        session=FakeSession(exc=requests.Timeout("down")),
    )

    capabilities = client.get()

    assert capabilities.catalog_id == "unavailable"
    assert capabilities.filters == {}
