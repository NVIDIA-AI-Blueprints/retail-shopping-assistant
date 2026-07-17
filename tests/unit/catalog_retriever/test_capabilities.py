# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from catalog_retriever.src.catalog import load_catalog


REPO_ROOT = Path(__file__).resolve().parents[3]


def _current_snapshot():
    return load_catalog(
        str(REPO_ROOT / "shared/data/enriched_products.jsonl"),
        str(REPO_ROOT / "shared/data/enriched_products.schema.yaml"),
        catalog_id="fashion_products",
        image_enabled=False,
        text_model_name="text-model",
    )


def _current_products() -> list[dict]:
    return [
        json.loads(line)
        for line in (
            REPO_ROOT / "shared/data/enriched_products.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def test_current_catalog_capabilities_are_generated_from_jsonl() -> None:
    snapshot = _current_snapshot()
    capabilities = snapshot.capabilities
    products = _current_products()

    assert capabilities.catalog_id == "fashion_products"
    assert capabilities.product_count == len(products)
    assert capabilities.retrieval_modes == ["text"]

    category_field = snapshot.schema.taxonomy.fields[0]
    expected_categories = sorted({product[category_field] for product in products})
    assert list(capabilities.taxonomy.categories) == expected_categories
    for category in expected_categories:
        expected_count = sum(
            product[category_field] == category for product in products
        )
        assert capabilities.taxonomy.categories[category].product_count == expected_count

    for field_name, spec in snapshot.schema.fields.items():
        field = capabilities.fields[field_name]
        assert field.filterable is ("filter" in spec.uses)
        assert field.searchable is ("semantic" in spec.uses)
        assert field.detail is ("detail" in spec.uses)
        assert field.coverage.present == sum(
            _has_value(product.get(field_name)) for product in products
        )
        assert (field_name in capabilities.filters) is (
            field.filterable and field.coverage.present > 0
        )


def test_scoped_values_and_coverage_are_derived_without_category_rules() -> None:
    snapshot = _current_snapshot()
    capabilities = snapshot.capabilities
    products = _current_products()
    category_field, subcategory_field = snapshot.schema.taxonomy.fields

    for category_name, category in capabilities.taxonomy.categories.items():
        category_products = [
            product
            for product in products
            if product[category_field] == category_name
        ]
        expected_subcategories = sorted(
            {product[subcategory_field] for product in category_products}
        )
        assert list(category.subcategories) == expected_subcategories
        for subcategory_name, subcategory in category.subcategories.items():
            scoped_products = [
                product
                for product in category_products
                if product[subcategory_field] == subcategory_name
            ]
            assert subcategory.product_count == len(scoped_products)
            for field_name, field in subcategory.filters.items():
                assert field.coverage.present == sum(
                    _has_value(product.get(field_name))
                    for product in scoped_products
                )
                expected_values = sorted(
                    {
                        str(value)
                        for product in scoped_products
                        for value in (
                            product.get(field_name)
                            if isinstance(product.get(field_name), list)
                            else [product.get(field_name)]
                        )
                        if _has_value(value)
                    },
                    key=str.casefold,
                )
                if field.values:
                    assert [item.value for item in field.values] == expected_values


def test_new_catalog_values_and_categories_require_no_code_change(tmp_path) -> None:
    data_path = tmp_path / "products.jsonl"
    schema_path = tmp_path / "products.schema.yaml"
    data_path.write_text(
        '{"id":"1","name":"Blue Pant","summary":"Soft","image":"a.jpg",'
        '"price":"20","category":"apparel","subcategory":"pants",'
        '"leg_shape":"wide","tags":["travel","soft"]}\n'
        '{"id":"2","name":"Red Gadget","summary":"Useful","image":"b.jpg",'
        '"price":"40","category":"electronics","subcategory":"gadgets",'
        '"tags":["travel"]}\n',
        encoding="utf-8",
    )
    schema_path.write_text(
        """
record:
  product_id: id
  name: name
  description: summary
  image: image
  price: price
taxonomy:
  fields: [category, subcategory]
fields:
  category: {type: enum, uses: [filter, semantic, detail]}
  subcategory: {type: enum, uses: [filter, semantic, detail]}
  price: {type: number, uses: [filter, detail]}
  leg_shape: {type: enum, uses: [filter, semantic, detail]}
  tags: {type: enum_list, uses: [filter, semantic, detail]}
""".strip(),
        encoding="utf-8",
    )

    capabilities = load_catalog(
        str(data_path),
        str(schema_path),
        image_enabled=False,
    ).capabilities

    assert list(capabilities.taxonomy.categories) == ["apparel", "electronics"]
    assert "pants" in capabilities.taxonomy.categories["apparel"].subcategories
    assert [item.value for item in capabilities.fields["tags"].values] == [
        "soft",
        "travel",
    ]
    assert capabilities.filters["tags"].type == "enum_list"


def test_unknown_fields_are_unclassified_not_filters(tmp_path) -> None:
    data_path = tmp_path / "products.jsonl"
    schema_path = tmp_path / "products.schema.yaml"
    data_path.write_text(
        '{"id":"1","name":"Item","summary":"Text","image":"a.jpg",'
        '"price":10,"category":"new","subcategory":"kind",'
        '"future_signal":"value"}\n',
        encoding="utf-8",
    )
    schema_path.write_text(
        """
record: {product_id: id, name: name, description: summary, image: image, price: price}
taxonomy: {fields: [category, subcategory]}
fields:
  category: {type: enum, uses: [filter, semantic]}
  subcategory: {type: enum, uses: [filter, semantic]}
  price: {type: number, uses: [filter]}
""".strip(),
        encoding="utf-8",
    )

    capabilities = load_catalog(
        str(data_path), str(schema_path), image_enabled=False
    ).capabilities

    assert capabilities.fields["future_signal"].type == "unclassified"
    assert capabilities.fields["future_signal"].observed_type == "string"
    assert "future_signal" not in capabilities.filters


def test_declared_but_absent_filter_field_is_not_advertised(tmp_path) -> None:
    data_path = tmp_path / "products.jsonl"
    schema_path = tmp_path / "products.schema.yaml"
    data_path.write_text(
        '{"id":"1","name":"Item","summary":"Text","image":"a.jpg",'
        '"price":10,"category":"new","subcategory":"kind"}\n',
        encoding="utf-8",
    )
    schema_path.write_text(
        """
record: {product_id: id, name: name, description: summary, image: image, price: price}
taxonomy: {fields: [category, subcategory]}
fields:
  category: {type: enum, uses: [filter, semantic]}
  subcategory: {type: enum, uses: [filter, semantic]}
  future_tag: {type: enum, uses: [filter, semantic]}
""".strip(),
        encoding="utf-8",
    )

    capabilities = load_catalog(
        str(data_path), str(schema_path), image_enabled=False
    ).capabilities

    assert capabilities.fields["future_tag"].filterable is True
    assert capabilities.fields["future_tag"].coverage.present == 0
    assert "future_tag" not in capabilities.filters
