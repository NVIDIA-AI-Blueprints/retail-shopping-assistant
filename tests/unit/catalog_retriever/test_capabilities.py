# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from catalog_retriever.src.capabilities import build_catalog_capabilities


def test_capabilities_are_declared_and_filled_from_registry_fields(tmp_path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "category,subcategory,name,price,color\n"
        "apparel,dress,Blue Dress,49.99,blue\n"
        "accessories,bag,Work Bag,89.50,black\n"
        "ignored,ignored,No Price,,red\n",
        encoding="utf-8",
    )
    config = {
        "catalog_id": "test_catalog",
        "data_source": str(csv_path),
        "image_enabled": True,
        "filter_registry": {
            "category": {
                "type": "enum",
                "source_fields": ["subcategory"],
                "operators": ["in"],
            },
            "price": {
                "type": "number",
                "source_fields": ["price"],
                "operators": ["gte", "lte"],
                "request_aliases": {"min": "min_price", "max": "max_price"},
            },
        },
        "soft_facets": {
            "color": {"type": "enum", "source_fields": ["color"]},
            "style": {"type": "text"},
        },
    }

    capabilities = build_catalog_capabilities(config)

    assert capabilities.catalog_id == "test_catalog"
    assert capabilities.retrieval_modes == ["text", "image", "hybrid"]
    assert capabilities.filters["category"].values == ["bag", "dress", "ignored"]
    assert capabilities.filters["price"].min_value == 49.99
    assert capabilities.filters["price"].max_value == 89.5
    assert capabilities.filters["price"].request_aliases == {
        "min": "min_price",
        "max": "max_price",
    }
    assert capabilities.soft_facets["color"].values == ["black", "blue", "red"]
    assert "name" not in capabilities.filters


def test_missing_catalog_data_still_returns_declared_capabilities() -> None:
    capabilities = build_catalog_capabilities(
        {
            "catalog_id": "empty",
            "data_source": "/missing/products.csv",
            "image_enabled": False,
            "filter_registry": {
                "category": {
                    "type": "enum",
                    "source_fields": ["subcategory"],
                    "operators": ["in"],
                }
            },
        }
    )

    assert capabilities.catalog_id == "empty"
    assert capabilities.retrieval_modes == ["text"]
    assert capabilities.image_search_enabled is False
    assert capabilities.filters["category"].values == []
