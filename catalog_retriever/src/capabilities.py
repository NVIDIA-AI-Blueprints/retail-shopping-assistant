# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog-owned capability discovery.

The catalog service declares which fields are real filters. Runtime data is
used only to fill enum values and numeric ranges for those declared fields.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
)


def build_catalog_capabilities(config: dict[str, Any]) -> CatalogCapabilities:
    rows = _read_rows(config.get("data_source", ""))
    filter_registry = config.get("filter_registry") or {}

    filters = {
        name: _build_filter_capability(spec, rows)
        for name, spec in filter_registry.items()
        if isinstance(spec, dict)
    }

    retrieval_modes = ["text"]
    if bool(config.get("image_enabled")):
        retrieval_modes.extend(["image", "hybrid"])

    return CatalogCapabilities(
        catalog_id=str(config.get("catalog_id") or "default"),
        retrieval_modes=retrieval_modes,
        image_search_enabled=bool(config.get("image_enabled")),
        filters=filters,
    )


def _read_rows(csv_path: str) -> list[dict[str, str]]:
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _build_filter_capability(
    spec: dict[str, Any], rows: list[dict[str, str]]
) -> CatalogFilterCapability:
    source_fields = _source_fields(spec)
    capability_type = str(spec.get("type") or "text")
    values: list[str] = []
    min_value: float | None = None
    max_value: float | None = None

    if capability_type == "enum":
        values = _enum_values(rows, source_fields)
    elif capability_type == "number":
        min_value, max_value = _number_range(rows, source_fields)

    return CatalogFilterCapability(
        type=capability_type,  # type: ignore[arg-type]
        operators=[str(value) for value in spec.get("operators", [])],
        source_fields=source_fields,
        values=values,
        min_value=min_value,
        max_value=max_value,
        request_aliases={
            str(key): str(value)
            for key, value in (spec.get("request_aliases") or {}).items()
        },
    )


def _source_fields(spec: dict[str, Any]) -> list[str]:
    fields = spec.get("source_fields")
    if isinstance(fields, list):
        return [str(field) for field in fields if str(field).strip()]
    field = spec.get("source_field")
    return [str(field)] if field else []


def _enum_values(rows: list[dict[str, str]], source_fields: list[str]) -> list[str]:
    values: set[str] = set()
    for row in rows:
        for field in source_fields:
            value = str(row.get(field) or "").strip()
            if value:
                values.add(value)
    return sorted(values)


def _number_range(
    rows: list[dict[str, str]], source_fields: list[str]
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for row in rows:
        for field in source_fields:
            value = _coerce_float(row.get(field))
            if value is not None:
                values.append(value)
    if not values:
        return None, None
    return min(values), max(values)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
