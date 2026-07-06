# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import requests

from chain_server.src.catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
)


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


def test_unavailable_capabilities_are_not_cached() -> None:
    session = FakeSession(exc=requests.Timeout("down"))
    client = CatalogCapabilitiesClient("http://catalog", session=session)

    assert client.get().catalog_id == "unavailable"
    assert client.get().catalog_id == "unavailable"

    assert len(session.calls) == 2


def test_formats_catalog_owned_filters_for_prompt() -> None:
    capabilities = CatalogCapabilities(
        catalog_id="custom_catalog",
        retrieval_modes=["text", "hybrid"],
        image_search_enabled=True,
        filters={
            "category": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["department"],
                values=["dress", "watch"],
            ),
            "price": CatalogFilterCapability(
                type="number",
                operators=["gte", "lte"],
                source_fields=["price"],
                min_value=10.0,
                max_value=250.0,
            ),
            "color": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["color"],
                values=["black", "green"],
            ),
        },
    )

    prompt_text = format_catalog_capabilities_for_prompt(capabilities)

    assert "Catalog ID: custom_catalog" in prompt_text
    assert "- category: enum; operators in; values dress, watch" in prompt_text
    assert "- price: number; operators gte, lte; range 10.0 to 250.0" in prompt_text
    assert "- color: enum; operators in; values black, green" in prompt_text
