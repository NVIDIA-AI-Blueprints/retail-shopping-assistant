# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic catalog loading and semantic-document construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from shared.commerce_contracts import CatalogCapabilities, Money, ProductDetail


SEARCH_DOCUMENT_TEMPLATE_VERSION = "1"

CatalogSourceType = Literal["enum", "enum_list", "number", "text"]
CatalogFieldUse = Literal["filter", "semantic", "detail"]
_RESERVED_INDEX_FIELDS = frozenset(
    {"pk", "text", "vector", "catalog_fingerprint"}
)


class CatalogRecordSchema(BaseModel):
    """Map catalog-independent record roles to source field names."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    fallback_description: str | None = None
    image: str = Field(..., min_length=1)
    price: str = Field(..., min_length=1)
    source_uri: str | None = None


class CatalogTaxonomySchema(BaseModel):
    """Ordered taxonomy fields, currently one or two levels."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fields: list[str] = Field(..., min_length=1, max_length=2)

    @model_validator(mode="after")
    def unique_fields(self) -> "CatalogTaxonomySchema":
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("taxonomy fields must be unique")
        return self


class CatalogFieldSchema(BaseModel):
    """Declare a field's meaning without declaring any catalog values."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: CatalogSourceType
    uses: list[CatalogFieldUse] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_uses(self) -> "CatalogFieldSchema":
        self.uses = list(dict.fromkeys(self.uses))
        if "filter" in self.uses and self.type == "text":
            raise ValueError(
                "text fields cannot be hard filters; use enum, enum_list, or number"
            )
        return self


class CatalogSchema(BaseModel):
    """Complete catalog-owned interpretation of a JSONL source."""

    model_config = ConfigDict(extra="forbid")

    record: CatalogRecordSchema
    taxonomy: CatalogTaxonomySchema
    fields: dict[str, CatalogFieldSchema] = Field(default_factory=dict)

    @model_validator(mode="after")
    def declared_taxonomy_fields(self) -> "CatalogSchema":
        missing = [name for name in self.taxonomy.fields if name not in self.fields]
        if missing:
            raise ValueError(
                "taxonomy fields must be declared under fields: " + ", ".join(missing)
            )
        reserved = sorted(
            (self.core_fields | set(self.fields) | set(self.taxonomy.fields))
            & _RESERVED_INDEX_FIELDS
        )
        if reserved:
            raise ValueError(
                "catalog schema field mappings use reserved index field(s): "
                + ", ".join(reserved)
            )
        return self

    @property
    def core_fields(self) -> set[str]:
        mapped = {
            self.record.product_id,
            self.record.name,
            self.record.description,
            self.record.image,
            self.record.price,
        }
        for optional in (
            self.record.fallback_description,
            self.record.source_uri,
        ):
            if optional:
                mapped.add(optional)
        return mapped

    @property
    def searchable_attribute_fields(self) -> list[str]:
        taxonomy = set(self.taxonomy.fields)
        return sorted(
            name
            for name, spec in self.fields.items()
            if "semantic" in spec.uses and name not in taxonomy
        )

    @property
    def detail_fields(self) -> list[str]:
        return sorted(
            name for name, spec in self.fields.items() if "detail" in spec.uses
        )


@dataclass(frozen=True)
class CatalogSnapshot:
    """One validated catalog used by every catalog-service read path."""

    catalog_id: str
    fingerprint: str
    schema: CatalogSchema
    products: tuple[dict[str, Any], ...]
    products_by_id: Mapping[str, dict[str, Any]]
    search_documents: tuple[str, ...]
    capabilities: CatalogCapabilities

    @property
    def product_count(self) -> int:
        return len(self.products)

    def product_detail(self, product_id: str) -> ProductDetail | None:
        product = self.products_by_id.get(product_id)
        if product is None:
            return None
        return build_product_detail(product, self.schema)


def load_catalog(
    data_path: str,
    schema_path: str,
    *,
    catalog_id: str = "default",
    image_enabled: bool = True,
    text_model_name: str = "",
    image_model_name: str | None = None,
    shared_root: str | None = None,
) -> CatalogSnapshot:
    """Load and fully validate a JSONL catalog and its role sidecar."""

    source_path = Path(data_path)
    sidecar_path = Path(schema_path)
    data_bytes = _read_required_bytes(source_path, "catalog data")
    schema_bytes = _read_required_bytes(sidecar_path, "catalog schema")
    schema = _load_schema(schema_bytes, sidecar_path)
    products = _load_products(data_bytes, source_path, schema, image_enabled=image_enabled)
    image_assets_digest = _image_assets_digest(
        products,
        schema,
        image_enabled=image_enabled,
        shared_root=shared_root,
    )
    fingerprint = _catalog_fingerprint(
        data_bytes=data_bytes,
        schema_bytes=schema_bytes,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        image_enabled=image_enabled,
        image_assets_digest=image_assets_digest,
    )
    search_documents = tuple(
        build_search_document(product, schema) for product in products
    )

    from .capabilities import build_catalog_capabilities

    capabilities = build_catalog_capabilities(
        catalog_id=catalog_id,
        products=products,
        schema=schema,
        image_enabled=image_enabled,
    )
    products_by_id = {
        product[schema.record.product_id]: product for product in products
    }
    return CatalogSnapshot(
        catalog_id=catalog_id,
        fingerprint=fingerprint,
        schema=schema,
        products=tuple(products),
        products_by_id=MappingProxyType(products_by_id),
        search_documents=search_documents,
        capabilities=capabilities,
    )


def build_search_document(product: Mapping[str, Any], schema: CatalogSchema) -> str:
    """Build one stable passage for text embedding without category branches."""

    name = _clean_text(product.get(schema.record.name))
    taxonomy_values = [
        _humanize_value(product.get(field), schema.fields[field].type)
        for field in schema.taxonomy.fields
        if _has_value(product.get(field))
    ]
    lines = [
        f"name: {name}",
        f"taxonomy: {' > '.join(taxonomy_values)}",
        "attributes:",
    ]
    for field_name in schema.searchable_attribute_fields:
        value = product.get(field_name)
        if not _has_value(value):
            continue
        spec = schema.fields[field_name]
        lines.append(
            f"- {_humanize_label(field_name)}: {_humanize_value(value, spec.type)}"
        )

    summary = product.get(schema.record.description)
    if not _has_value(summary) and schema.record.fallback_description:
        summary = product.get(schema.record.fallback_description)
    lines.append(f"summary: {_clean_text(summary)}")
    return "\n".join(lines)


def build_product_detail(
    product: Mapping[str, Any], schema: CatalogSchema
) -> ProductDetail:
    """Map one source record to the shared deterministic product contract."""

    description = product.get(schema.record.description)
    if not _has_value(description) and schema.record.fallback_description:
        description = product.get(schema.record.fallback_description)

    taxonomy_values = [
        _clean_text(product.get(field))
        for field in schema.taxonomy.fields
        if _has_value(product.get(field))
    ]
    attributes = {
        field: _detail_value(product[field])
        for field in schema.detail_fields
        if field != schema.record.price and _has_value(product.get(field))
    }
    price = _coerce_number(product.get(schema.record.price))
    source_uri = None
    if schema.record.source_uri and _has_value(product.get(schema.record.source_uri)):
        source_uri = _clean_text(product.get(schema.record.source_uri))

    return ProductDetail(
        product_id=product[schema.record.product_id],
        display_name=_clean_text(product.get(schema.record.name)),
        description=_clean_text(description),
        category=taxonomy_values[-1] if taxonomy_values else None,
        price=Money(amount=price) if price is not None else None,
        image_url=_clean_text(product.get(schema.record.image)) or None,
        attributes=attributes,
        source_uri=source_uri,
    )


def _read_required_bytes(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path.read_bytes()


def _load_schema(schema_bytes: bytes, path: Path) -> CatalogSchema:
    try:
        raw = yaml.safe_load(schema_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid catalog schema {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid catalog schema {path}: expected a mapping")
    return CatalogSchema.model_validate(raw)


def _load_products(
    data_bytes: bytes,
    path: Path,
    schema: CatalogSchema,
    *,
    image_enabled: bool,
) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(data_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            product = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
        if not isinstance(product, dict):
            raise ValueError(f"Invalid catalog record at line {line_number}: expected an object")

        product_id = _required_product_id(
            product, schema.record.product_id, line_number
        )
        product[schema.record.product_id] = product_id
        if product_id in seen_ids:
            raise ValueError(
                f"Duplicate product ID '{product_id}' in {path} at line {line_number}"
            )
        seen_ids.add(product_id)
        _required_text(product, schema.record.name, line_number)
        for taxonomy_field in schema.taxonomy.fields:
            _required_text(product, taxonomy_field, line_number)
        if image_enabled:
            _required_text(product, schema.record.image, line_number)
        price = _coerce_number(product.get(schema.record.price))
        if price is None or price < 0:
            raise ValueError(
                f"Invalid numeric field '{schema.record.price}' at line {line_number}"
            )
        description = product.get(schema.record.description)
        fallback = (
            product.get(schema.record.fallback_description)
            if schema.record.fallback_description
            else None
        )
        if not _has_value(description) and not _has_value(fallback):
            raise ValueError(
                f"Missing description fields at line {line_number}: "
                f"'{schema.record.description}'"
            )
        _validate_declared_field_types(product, schema, line_number)
        products.append(product)

    if not products:
        raise ValueError(f"Catalog data file contains no products: {path}")
    return products


def _validate_declared_field_types(
    product: Mapping[str, Any], schema: CatalogSchema, line_number: int
) -> None:
    for field_name, spec in schema.fields.items():
        value = product.get(field_name)
        if not _has_value(value):
            continue
        valid = True
        if spec.type == "number":
            valid = _coerce_number(value) is not None
        elif spec.type == "enum_list":
            valid = isinstance(value, list) and all(
                isinstance(item, str) and _has_value(item) for item in value
            )
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(
                f"Invalid {spec.type} field '{field_name}' at line {line_number}"
            )


def _required_text(product: Mapping[str, Any], field: str, line_number: int) -> str:
    value = _clean_text(product.get(field))
    if not value:
        raise ValueError(f"Missing required field '{field}' at line {line_number}")
    return value


def _required_product_id(
    product: Mapping[str, Any], field: str, line_number: int
) -> str:
    """Validate an opaque ID that must round-trip through one URL path segment."""

    raw_value = product.get(field)
    if not isinstance(raw_value, str):
        raise ValueError(
            f"Invalid product ID field '{field}' at line {line_number}: "
            "expected a string"
        )

    canonical = _clean_text(raw_value)
    if not canonical:
        raise ValueError(f"Missing required field '{field}' at line {line_number}")
    if raw_value != canonical:
        raise ValueError(
            f"Invalid product ID field '{field}' at line {line_number}: "
            "value must already be canonical"
        )
    if "/" in raw_value or raw_value in {".", ".."}:
        raise ValueError(
            f"Invalid product ID field '{field}' at line {line_number}: "
            "value must be safe as one URL path segment"
        )
    return raw_value


def _catalog_fingerprint(
    *,
    data_bytes: bytes,
    schema_bytes: bytes,
    text_model_name: str,
    image_model_name: str | None,
    image_enabled: bool,
    image_assets_digest: bytes,
) -> str:
    digest = sha256()
    for label, payload in (
        (b"data", data_bytes),
        (b"schema", schema_bytes),
        (b"text_model", text_model_name.encode("utf-8")),
        (
            b"image_model",
            ((image_model_name or "") if image_enabled else "").encode("utf-8"),
        ),
        (b"image_enabled", str(image_enabled).encode("ascii")),
        (b"image_assets", image_assets_digest),
        (b"template", SEARCH_DOCUMENT_TEMPLATE_VERSION.encode("ascii")),
    ):
        digest.update(label + b"\0" + payload + b"\0")
    return digest.hexdigest()


def _image_assets_digest(
    products: list[dict[str, Any]],
    schema: CatalogSchema,
    *,
    image_enabled: bool,
    shared_root: str | None,
) -> bytes:
    """Hash local image bytes so changed pixels trigger image reindexing."""

    if not image_enabled:
        return b""

    root = Path(shared_root or os.environ.get("SHARED_ROOT", "/app/shared")).resolve()
    digest = sha256()
    for product in products:
        product_id = _clean_text(product.get(schema.record.product_id))
        reference = _clean_text(product.get(schema.record.image))
        digest.update(product_id.encode("utf-8") + b"\0")
        digest.update(reference.encode("utf-8") + b"\0")
        if reference.startswith(("http://", "https://", "data:")):
            continue

        image_path = (root / reference.lstrip("/")).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(
                f"Image path for product '{product_id}' escapes SHARED_ROOT: {reference}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image for product '{product_id}' was not found: {image_path}"
            )
        digest.update(image_path.read_bytes())
        digest.update(b"\0")
    return digest.digest()


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip().replace("$", "").replace(",", ""))
            return number if isfinite(number) else None
        except ValueError:
            return None
    return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _humanize_label(value: str) -> str:
    return _clean_text(value).replace("_", " ")


def _humanize_value(value: Any, field_type: CatalogSourceType) -> str:
    if isinstance(value, list):
        cleaned = sorted(
            {
                _clean_text(item).replace("_", " ")
                for item in value
                if _has_value(item)
            }
        )
        return ", ".join(cleaned)
    cleaned = _clean_text(value)
    if field_type == "enum":
        return cleaned.replace("_", " ")
    return cleaned


def _detail_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_detail_value(item) for item in value]
    return value
