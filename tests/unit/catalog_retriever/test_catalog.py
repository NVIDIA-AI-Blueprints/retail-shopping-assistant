# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from catalog_retriever.src.catalog import build_search_document, load_catalog


SCHEMA = """
record:
  product_id: id
  name: name
  description: enriched
  fallback_description: description
  image: image
  price: price
taxonomy:
  fields: [category, subcategory]
fields:
  category: {type: enum, uses: [filter, semantic, detail]}
  subcategory: {type: enum, uses: [filter, semantic, detail]}
  price: {type: number, uses: [filter, detail]}
  color: {type: enum, uses: [filter, semantic, detail]}
  care: {type: text, uses: [semantic, detail]}
  tags: {type: enum_list, uses: [filter, semantic, detail]}
  url: {type: text, uses: []}
  source_row: {type: number, uses: []}
""".strip()


def _write_source(tmp_path, products, schema: str = SCHEMA):
    data_path = tmp_path / "products.jsonl"
    schema_path = tmp_path / "products.schema.yaml"
    data_path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    schema_path.write_text(schema, encoding="utf-8")
    return data_path, schema_path


def _product(**updates):
    product = {
        "id": "p1",
        "name": "Blue Travel Pant",
        "enriched": "A soft travel pant.",
        "description": "Long original marketing copy.",
        "image": "/images/p1.jpg",
        "price": "49.90",
        "category": "apparel",
        "subcategory": "pants",
        "color": "navy_blue",
        "care": "Machine wash cold.",
        "tags": ["soft", "travel", "soft"],
        "url": "/images/p1.jpg",
        "source_row": 3,
    }
    product.update(updates)
    return product


def test_loads_snapshot_and_preserves_source_record(tmp_path) -> None:
    data_path, schema_path = _write_source(tmp_path, [_product()])

    snapshot = load_catalog(
        str(data_path),
        str(schema_path),
        catalog_id="test",
        image_enabled=False,
        text_model_name="text-a",
    )

    assert snapshot.product_count == 1
    assert snapshot.products_by_id["p1"]["source_row"] == 3
    assert snapshot.capabilities.product_count == 1
    assert len(snapshot.fingerprint) == 64


def test_product_id_round_trips_exactly_across_snapshot_and_detail(tmp_path) -> None:
    product_id = "generated:abc123"
    data_path, schema_path = _write_source(
        tmp_path, [_product(id=product_id)]
    )

    snapshot = load_catalog(
        str(data_path), str(schema_path), image_enabled=False
    )
    detail = snapshot.product_detail(product_id)

    assert snapshot.products[0][snapshot.schema.record.product_id] == product_id
    assert list(snapshot.products_by_id) == [product_id]
    assert detail is not None
    assert detail.product_id == product_id


def test_search_document_is_deterministic_and_excludes_raw_fields(tmp_path) -> None:
    data_path, schema_path = _write_source(tmp_path, [_product()])
    snapshot = load_catalog(str(data_path), str(schema_path), image_enabled=False)

    document = snapshot.search_documents[0]

    assert document == "\n".join(
        [
            "name: Blue Travel Pant",
            "taxonomy: apparel > pants",
            "attributes:",
            "- care: Machine wash cold.",
            "- color: navy blue",
            "- tags: soft, travel",
            "summary: A soft travel pant.",
        ]
    )
    assert "p1" not in document
    assert "49.90" not in document
    assert "/images" not in document
    assert "source row" not in document
    assert "Long original marketing copy" not in document


def test_description_is_only_a_fallback(tmp_path) -> None:
    product = _product(enriched="", description="Fallback description.")
    data_path, schema_path = _write_source(tmp_path, [product])
    snapshot = load_catalog(str(data_path), str(schema_path), image_enabled=False)

    assert snapshot.search_documents[0].endswith("summary: Fallback description.")


def test_product_detail_contains_only_declared_detail_fields(tmp_path) -> None:
    data_path, schema_path = _write_source(tmp_path, [_product()])
    detail = load_catalog(
        str(data_path), str(schema_path), image_enabled=False
    ).product_detail("p1")

    assert detail is not None
    assert detail.product_id == "p1"
    assert detail.price.amount == 49.9
    assert detail.attributes["care"] == "Machine wash cold."
    assert detail.attributes["category"] == "apparel"
    assert "url" not in detail.attributes
    assert detail.source_uri is None


def test_malformed_json_reports_line_number(tmp_path) -> None:
    data_path, schema_path = _write_source(tmp_path, [_product()])
    data_path.write_text(json.dumps(_product()) + "\n{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


@pytest.mark.parametrize(
    "products,match",
    [
        ([_product(id="")], "Missing required field 'id'"),
        ([_product(), _product()], "Duplicate product ID 'p1'"),
        ([_product(price="not-a-price")], "Invalid numeric field 'price'"),
        ([_product(price="-1")], "Invalid numeric field 'price'"),
        ([_product(price="NaN")], "Invalid numeric field 'price'"),
        ([_product(price="Infinity")], "Invalid numeric field 'price'"),
        ([_product(color=3)], "Invalid enum field 'color'"),
        ([_product(care={"wash": "cold"})], "Invalid text field 'care'"),
        ([_product(tags=["travel", 2])], "Invalid enum_list field 'tags'"),
        ([_product(category="")], "Missing required field 'category'"),
    ],
)
def test_invalid_required_values_fail_the_entire_load(tmp_path, products, match) -> None:
    data_path, schema_path = _write_source(tmp_path, products)

    with pytest.raises(ValueError, match=match):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


@pytest.mark.parametrize("product_id", [0, False, True, ["p1"], {"id": "p1"}])
def test_product_id_must_be_a_string(tmp_path, product_id) -> None:
    data_path, schema_path = _write_source(
        tmp_path, [_product(id=product_id)]
    )

    with pytest.raises(ValueError, match="Invalid product ID.*expected a string"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


@pytest.mark.parametrize(
    "product_id",
    [" p1", "p1 ", "p1  suffix", "p1\tsuffix", "p1\nsuffix"],
)
def test_product_id_must_already_be_canonical(tmp_path, product_id: str) -> None:
    data_path, schema_path = _write_source(
        tmp_path, [_product(id=product_id)]
    )

    with pytest.raises(ValueError, match="Invalid product ID.*already be canonical"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


@pytest.mark.parametrize("product_id", ["parent/child", ".", ".."])
def test_product_id_must_be_one_safe_url_path_segment(
    tmp_path, product_id: str
) -> None:
    data_path, schema_path = _write_source(
        tmp_path, [_product(id=product_id)]
    )

    with pytest.raises(ValueError, match="Invalid product ID.*one URL path segment"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


def test_fingerprint_changes_with_data_schema_or_embedding_model(tmp_path) -> None:
    data_path, schema_path = _write_source(tmp_path, [_product()])
    first = load_catalog(
        str(data_path),
        str(schema_path),
        image_enabled=False,
        text_model_name="text-a",
    )
    second = load_catalog(
        str(data_path),
        str(schema_path),
        image_enabled=False,
        text_model_name="text-b",
    )

    assert first.fingerprint != second.fingerprint

    changed_product = _product(enriched="Changed summary.")
    data_path.write_text(json.dumps(changed_product) + "\n", encoding="utf-8")
    third = load_catalog(
        str(data_path),
        str(schema_path),
        image_enabled=False,
        text_model_name="text-a",
    )
    assert first.fingerprint != third.fingerprint


def test_fingerprint_changes_when_local_image_bytes_change(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "p1.jpg"
    image_path.write_bytes(b"first-image")
    data_path, schema_path = _write_source(
        tmp_path, [_product(image="/images/p1.jpg")]
    )

    first = load_catalog(
        str(data_path),
        str(schema_path),
        image_enabled=True,
        image_model_name="image-model",
        shared_root=str(tmp_path),
    )
    image_path.write_bytes(b"second-image")
    second = load_catalog(
        str(data_path),
        str(schema_path),
        image_enabled=True,
        image_model_name="image-model",
        shared_root=str(tmp_path),
    )

    assert first.fingerprint != second.fingerprint


def test_enabled_image_index_requires_each_local_image(tmp_path) -> None:
    data_path, schema_path = _write_source(
        tmp_path, [_product(image="/images/missing.jpg")]
    )

    with pytest.raises(FileNotFoundError, match="Image for product 'p1'"):
        load_catalog(
            str(data_path),
            str(schema_path),
            image_enabled=True,
            shared_root=str(tmp_path),
        )


def test_build_search_document_has_no_category_specific_branches(tmp_path) -> None:
    data_path, schema_path = _write_source(tmp_path, [_product()])
    snapshot = load_catalog(str(data_path), str(schema_path), image_enabled=False)

    future = _product(
        id="future",
        category="electronics",
        subcategory="gadgets",
        name="Travel Gadget",
    )
    assert "taxonomy: electronics > gadgets" in build_search_document(
        future, snapshot.schema
    )


def test_text_field_cannot_be_declared_as_a_hard_filter(tmp_path) -> None:
    invalid_schema = SCHEMA.replace(
        "care: {type: text, uses: [semantic, detail]}",
        "care: {type: text, uses: [filter, semantic, detail]}",
    )
    data_path, schema_path = _write_source(
        tmp_path, [_product()], schema=invalid_schema
    )

    with pytest.raises(ValueError, match="text fields cannot be hard filters"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


@pytest.mark.parametrize(
    "reserved_field",
    ["pk", "text", "vector", "catalog_fingerprint"],
)
def test_record_roles_cannot_use_reserved_index_fields(
    tmp_path, reserved_field: str
) -> None:
    invalid_schema = SCHEMA.replace("product_id: id", f"product_id: {reserved_field}")
    data_path, schema_path = _write_source(
        tmp_path, [_product()], schema=invalid_schema
    )

    with pytest.raises(ValueError, match="reserved index field"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)


@pytest.mark.parametrize(
    "reserved_field",
    ["pk", "text", "vector", "catalog_fingerprint"],
)
def test_declared_fields_cannot_use_reserved_index_fields(
    tmp_path, reserved_field: str
) -> None:
    invalid_schema = SCHEMA + (
        f"\n  {reserved_field}: {{type: enum, uses: [filter, semantic]}}"
    )
    data_path, schema_path = _write_source(
        tmp_path, [_product()], schema=invalid_schema
    )

    with pytest.raises(ValueError, match="reserved index field"):
        load_catalog(str(data_path), str(schema_path), image_enabled=False)
