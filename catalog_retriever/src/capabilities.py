# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate catalog capabilities from one validated in-memory snapshot."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

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

from .catalog import CatalogFieldSchema, CatalogSchema


def build_catalog_capabilities(
    *,
    catalog_id: str,
    products: Sequence[Mapping[str, Any]],
    schema: CatalogSchema,
    image_enabled: bool,
) -> CatalogCapabilities:
    """Describe field meaning, observed values, and taxonomy-scoped coverage."""

    fields = _global_fields(products, schema)
    filters = {
        name: _compatibility_filter(name, capability)
        for name, capability in fields.items()
        if capability.filterable and capability.coverage.present > 0
    }
    retrieval_modes = ["text"]
    if image_enabled:
        retrieval_modes.extend(["image", "hybrid"])

    return CatalogCapabilities(
        catalog_id=catalog_id,
        product_count=len(products),
        retrieval_modes=retrieval_modes,
        image_search_enabled=image_enabled,
        filters=filters,
        fields=fields,
        taxonomy=_taxonomy_capabilities(products, schema),
    )


def _global_fields(
    products: Sequence[Mapping[str, Any]], schema: CatalogSchema
) -> dict[str, CatalogFieldCapability]:
    source_fields = set().union(*(product.keys() for product in products))
    ignored_core_fields = schema.core_fields - set(schema.fields)
    names = sorted((source_fields | set(schema.fields)) - ignored_core_fields)
    return {
        name: _field_capability(name, products, schema)
        for name in names
    }


def _field_capability(
    name: str,
    products: Sequence[Mapping[str, Any]],
    schema: CatalogSchema,
) -> CatalogFieldCapability:
    spec = schema.fields.get(name)
    present_products = [product for product in products if _has_value(product.get(name))]
    coverage = CatalogCoverage(present=len(present_products), total=len(products))
    if spec is None:
        return CatalogFieldCapability(
            type="unclassified",
            observed_type=_observed_type(product.get(name) for product in present_products),
            source_fields=[name],
            coverage=coverage,
        )

    filterable = "filter" in spec.uses
    values: list[CatalogValueCapability] = []
    min_value: float | None = None
    max_value: float | None = None
    if spec.type in {"enum", "enum_list"}:
        counts = Counter(_iter_field_values(present_products, name, spec))
        values = [
            CatalogValueCapability(value=value, count=counts[value])
            for value in sorted(counts, key=lambda item: (item.casefold(), item))
        ]
    elif spec.type == "number":
        numbers = [
            number
            for product in present_products
            if (number := _coerce_number(product.get(name))) is not None
        ]
        if numbers:
            min_value, max_value = min(numbers), max(numbers)

    return CatalogFieldCapability(
        type=spec.type,
        filterable=filterable,
        searchable="semantic" in spec.uses,
        detail="detail" in spec.uses,
        taxonomy=name in schema.taxonomy.fields,
        operators=_operators(spec) if filterable else [],
        source_fields=[name],
        coverage=coverage,
        values=values,
        min_value=min_value,
        max_value=max_value,
        # Only for values this catalog actually carries: a meaning for a value
        # no product has would describe a filter that returns nothing.
        value_meanings={
            key: text
            for key, text in spec.value_meanings.items()
            if any(value.value == key for value in values)
        },
    )


def _compatibility_filter(
    name: str, capability: CatalogFieldCapability
) -> CatalogFilterCapability:
    aliases = {}
    if capability.type == "number":
        aliases = {"min": f"min_{name}", "max": f"max_{name}"}
    return CatalogFilterCapability(
        type=capability.type,  # type: ignore[arg-type]
        operators=capability.operators,
        source_fields=capability.source_fields,
        values=[value.value for value in capability.values],
        min_value=capability.min_value,
        max_value=capability.max_value,
        request_aliases=aliases,
        value_meanings=dict(capability.value_meanings),
    )


def _taxonomy_capabilities(
    products: Sequence[Mapping[str, Any]], schema: CatalogSchema
) -> CatalogTaxonomyCapabilities:
    category_field = schema.taxonomy.fields[0]
    subcategory_field = (
        schema.taxonomy.fields[1] if len(schema.taxonomy.fields) > 1 else None
    )
    categories: dict[str, CatalogTaxonomyCategory] = {}
    for category, category_products in _group_products(products, category_field).items():
        subcategories: dict[str, CatalogTaxonomySubcategory] = {}
        if subcategory_field:
            for subcategory, scoped_products in _group_products(
                category_products, subcategory_field
            ).items():
                filters, semantic_fields = _scoped_fields(scoped_products, schema)
                subcategories[subcategory] = CatalogTaxonomySubcategory(
                    product_count=len(scoped_products),
                    filters=filters,
                    semantic_fields=semantic_fields,
                )
        filters, semantic_fields = _scoped_fields(category_products, schema)
        categories[category] = CatalogTaxonomyCategory(
            product_count=len(category_products),
            filters=filters,
            semantic_fields=semantic_fields,
            subcategories=subcategories,
        )

    return CatalogTaxonomyCapabilities(
        category_field=category_field,
        subcategory_field=subcategory_field,
        categories=categories,
    )


def _scoped_fields(
    products: Sequence[Mapping[str, Any]], schema: CatalogSchema
) -> tuple[dict[str, CatalogFieldCapability], dict[str, CatalogFieldCapability]]:
    capabilities = {
        name: _field_capability(name, products, schema)
        for name in sorted(schema.fields)
    }
    filters = {
        name: capability
        for name, capability in capabilities.items()
        if capability.filterable and capability.coverage.present > 0
    }
    semantic_fields = {
        name: capability
        for name, capability in capabilities.items()
        if capability.searchable and capability.coverage.present > 0
    }
    return filters, semantic_fields


def _group_products(
    products: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for product in products:
        value = _clean_text(product.get(field))
        if value:
            groups.setdefault(value, []).append(product)
    return {name: groups[name] for name in sorted(groups, key=str.casefold)}


def _operators(spec: CatalogFieldSchema) -> list[str]:
    if spec.type in {"enum", "enum_list"}:
        return ["in"]
    if spec.type == "number":
        return ["gte", "lte"]
    return []


def _iter_field_values(
    products: Iterable[Mapping[str, Any]],
    name: str,
    spec: CatalogFieldSchema,
) -> Iterable[str]:
    for product in products:
        raw = product.get(name)
        values = raw if spec.type == "enum_list" and isinstance(raw, list) else [raw]
        normalized_values = {
            normalized
            for value in values
            if (normalized := _clean_text(value))
        }
        yield from normalized_values


def _observed_type(values: Iterable[Any]) -> str | None:
    observed = {_json_type(value) for value in values if _has_value(value)}
    if not observed:
        return None
    return next(iter(observed)) if len(observed) == 1 else "mixed"


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace("$", "").replace(",", ""))
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
