# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import requests

from chain_server.src.catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogCoverage,
    CatalogFieldCapability,
    CatalogFilterCapability,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogTaxonomySubcategory,
    CatalogValueCapability,
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


class BlockingSession(FakeSession):
    def __init__(self, response) -> None:
        super().__init__(response=response)
        self.started = Event()
        self.release = Event()

    def get(self, url, timeout):
        self.calls.append({"url": url, "timeout": timeout})
        self.started.set()
        if not self.release.wait(timeout=1):
            raise requests.Timeout("test release timed out")
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


def test_caches_first_success_for_process_lifecycle() -> None:
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

    first = client.get()
    session.exc = requests.Timeout("catalog restarted")

    assert client.get() is first
    assert client.get() is first
    assert len(session.calls) == 1


def test_concurrent_first_reads_share_one_service_request() -> None:
    session = BlockingSession(
        FakeResponse(
            {
                "catalog_id": "fashion",
                "retrieval_modes": ["text"],
                "filters": {},
            }
        )
    )
    client = CatalogCapabilitiesClient("http://catalog", session=session)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(client.get)
        assert session.started.wait(timeout=1)
        second_started = Event()

        def second_read():
            second_started.set()
            return client.get()

        second_future = pool.submit(second_read)
        assert second_started.wait(timeout=1)
        session.release.set()
        first = first_future.result(timeout=1)
        second = second_future.result(timeout=1)

    assert second is first
    assert len(session.calls) == 1


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


def test_retries_until_first_success_then_caches() -> None:
    session = FakeSession(exc=requests.Timeout("down"))
    client = CatalogCapabilitiesClient("http://catalog", session=session)

    unavailable = client.get()
    session.exc = None
    session.response = FakeResponse(
        {
            "catalog_id": "fashion",
            "retrieval_modes": ["text"],
            "filters": {},
        }
    )
    available = client.get()

    assert unavailable.catalog_id == "unavailable"
    assert available.catalog_id == "fashion"
    assert client.get() is available
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

    assert "Retrieval modes: text, hybrid" in prompt_text
    assert "- category: enum; values dress, watch" in prompt_text
    assert "- price: number; range 10.0 to 250.0" in prompt_text
    assert "- color: enum; values black, green" in prompt_text


def test_formats_nested_taxonomy_and_semantic_fields_compactly() -> None:
    neckline = CatalogFieldCapability(
        type="enum",
        filterable=True,
        searchable=True,
        detail=True,
        operators=["in"],
        source_fields=["neckline"],
        coverage=CatalogCoverage(present=31, total=32),
        values=[
            CatalogValueCapability(value="crew", count=1),
            CatalogValueCapability(value="v_neck", count=19),
        ],
    )
    care = CatalogFieldCapability(
        type="text",
        searchable=True,
        detail=True,
        source_fields=["care"],
        coverage=CatalogCoverage(present=5, total=32),
    )
    capabilities = CatalogCapabilities(
        catalog_id="fashion",
        product_count=32,
        retrieval_modes=["text"],
        fields={"neckline": neckline, "care": care},
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category",
            subcategory_field="subcategory",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=32,
                    filters={"neckline": neckline},
                    semantic_fields={"neckline": neckline, "care": care},
                    subcategories={
                        "dresses": CatalogTaxonomySubcategory(
                            product_count=32,
                            filters={"neckline": neckline},
                            semantic_fields={"neckline": neckline, "care": care},
                        )
                    },
                )
            },
        ),
    )

    prompt_text = format_catalog_capabilities_for_prompt(capabilities)

    assert "- neckline: enum; values crew, v_neck" in prompt_text
    assert "- care: text; detail yes" in prompt_text
    assert "- neckline: enum; values crew, v_neck; semantic yes" in prompt_text
    assert "- category=apparel" in prompt_text
    assert "  - subcategory=dresses" in prompt_text
    assert "filters: neckline" in prompt_text
    assert "semantic/detail: care" in prompt_text
    assert "neckline=[crew, v_neck]" not in prompt_text
    assert "31/32" not in prompt_text
    assert "5/32" not in prompt_text


def test_formats_custom_taxonomy_keys_and_all_enum_values() -> None:
    values = [
        CatalogValueCapability(value=f"tag_{index}", count=1) for index in range(81)
    ]
    tags = CatalogFieldCapability(
        type="enum_list",
        filterable=True,
        searchable=False,
        operators=["in"],
        source_fields=["tags"],
        coverage=CatalogCoverage(present=1, total=1),
        values=values,
    )
    capabilities = CatalogCapabilities(
        catalog_id="custom",
        retrieval_modes=["text"],
        fields={"tags": tags},
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="department",
            subcategory_field="product_type",
            categories={
                "home": CatalogTaxonomyCategory(
                    product_count=1,
                    filters={"tags": tags},
                    subcategories={
                        "lamp": CatalogTaxonomySubcategory(
                            product_count=1,
                            filters={"tags": tags},
                        )
                    },
                )
            },
        ),
    )

    prompt_text = format_catalog_capabilities_for_prompt(capabilities)

    assert "- tags: enum_list; values tag_0" in prompt_text
    assert "tag_80; semantic no" in prompt_text
    assert "department > product_type" in prompt_text
    assert "- department=home" in prompt_text
    assert "  - product_type=lamp" in prompt_text
    assert "+1 more" not in prompt_text
