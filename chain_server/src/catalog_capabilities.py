# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain-server client for catalog-owned capability metadata."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import requests

from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFieldCapability,
    CatalogFilterCapability,
)


logger = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 2.0


class CatalogCapabilitiesClient:
    """Fetch and cache catalog capabilities from the catalog service."""

    def __init__(
        self,
        catalog_retriever_url: str,
        *,
        timeout_seconds: float | None = None,
        session: Any | None = None,
    ) -> None:
        self.catalog_retriever_url = catalog_retriever_url.rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else _DEFAULT_TIMEOUT_SECONDS
        )
        self.session = session or requests
        self._cached: CatalogCapabilities | None = None
        self._cache_lock = Lock()

    def get(self) -> CatalogCapabilities:
        """Return the first successful capability contract for this process."""

        if self._cached is not None:
            return self._cached

        # UI capability reads and shopper turns can arrive together at startup.
        # Only one of them should perform the first service request.
        with self._cache_lock:
            if self._cached is not None:
                return self._cached

            try:
                response = self.session.get(
                    f"{self.catalog_retriever_url}/capabilities",
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                capabilities = CatalogCapabilities.model_validate(response.json())
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Catalog capabilities unavailable: %s", exc)
                # Do not cache failure: the next request retries until the
                # catalog service supplies one valid lifecycle contract.
                return CatalogCapabilities(catalog_id="unavailable")

            self._cached = capabilities
            return capabilities


def format_catalog_capabilities_for_prompt(
    capabilities: CatalogCapabilities,
) -> str:
    """Render catalog-owned filter metadata for the agent prompt."""

    if capabilities.catalog_id == "unavailable":
        return (
            "Catalog capability metadata is currently unavailable. Use "
            "search_catalog_tool for product discovery and ask a concise "
            "clarifying question when the shopper request is underspecified."
        )

    lines: list[str] = []
    if capabilities.retrieval_modes:
        lines.append(f"Retrieval modes: {', '.join(capabilities.retrieval_modes)}")

    filters = effective_filter_capabilities(capabilities)
    if filters:
        lines.append("Hard filters (enum values are exact; numbers use min/max):")
        for name, capability in sorted(filters.items()):
            field = capabilities.fields.get(name)
            lines.append(
                f"- {name}: "
                + _format_filter_capability(
                    capability,
                    searchable=field.searchable if field is not None else None,
                )
            )

    semantic_only = {
        name: field
        for name, field in capabilities.fields.items()
        if field.searchable and not field.filterable and field.coverage.present > 0
    }
    if semantic_only:
        lines.append("Semantic/detail fields (not hard filters):")
        for name, field in sorted(semantic_only.items()):
            lines.append(f"- {name}: {_format_semantic_field(field)}")

    if capabilities.taxonomy.categories:
        category_field = capabilities.taxonomy.category_field or "taxonomy_level_1"
        subcategory_field = (
            capabilities.taxonomy.subcategory_field or "taxonomy_level_2"
        )
        lines.append(
            "Taxonomy-specific field availability "
            f"({category_field} > {subcategory_field}; "
            "use exact values from Hard filters above):"
        )
        taxonomy_fields = {
            field
            for field in (
                capabilities.taxonomy.category_field,
                capabilities.taxonomy.subcategory_field,
            )
            if field
        }
        for category_name in sorted(
            capabilities.taxonomy.categories,
            key=str.casefold,
        ):
            category = capabilities.taxonomy.categories[category_name]
            lines.append(f"- {category_field}={category_name}")
            lines.extend(
                _format_scope(
                    category.filters,
                    category.semantic_fields,
                    "  ",
                    excluded_fields=taxonomy_fields,
                )
            )
            for subcategory_name in sorted(
                category.subcategories,
                key=str.casefold,
            ):
                subcategory = category.subcategories[subcategory_name]
                lines.append(f"  - {subcategory_field}={subcategory_name}")
                lines.extend(
                    _format_scope(
                        subcategory.filters,
                        subcategory.semantic_fields,
                        "    ",
                        excluded_fields=taxonomy_fields,
                    )
                )

    return "\n".join(lines)


def effective_filter_capabilities(
    capabilities: CatalogCapabilities,
) -> dict[str, CatalogFilterCapability]:
    """Use authoritative field roles, with legacy flat capabilities as fallback."""

    if not capabilities.fields:
        return capabilities.filters
    return {
        name: CatalogFilterCapability(
            type=field.type,  # type: ignore[arg-type]
            operators=field.operators,
            source_fields=field.source_fields,
            values=[value.value for value in field.values],
            min_value=field.min_value,
            max_value=field.max_value,
            request_aliases=(
                {"min": f"min_{name}", "max": f"max_{name}"}
                if field.type == "number"
                else {}
            ),
        )
        for name, field in capabilities.fields.items()
        if field.filterable and field.coverage.present > 0
    }


def _format_filter_capability(
    capability: CatalogFilterCapability,
    *,
    searchable: bool | None,
) -> str:
    parts = [capability.type]
    if capability.values:
        parts.append(f"values {_format_values(capability.values)}")
    elif capability.min_value is not None or capability.max_value is not None:
        parts.append(
            f"range {_format_range(capability.min_value, capability.max_value)}"
        )
    if searchable is not None:
        parts.append(f"semantic {'yes' if searchable else 'no'}")
    return "; ".join(parts)


def _format_semantic_field(field: CatalogFieldCapability) -> str:
    return f"{field.type}; detail {'yes' if field.detail else 'no'}"


def _format_scope(
    filters: dict[str, CatalogFieldCapability],
    semantic_fields: dict[str, CatalogFieldCapability],
    indent: str,
    *,
    excluded_fields: set[str],
) -> list[str]:
    lines: list[str] = []
    filter_names = sorted(set(filters) - excluded_fields)
    if filter_names:
        lines.append(indent + "filters: " + ", ".join(filter_names))
    semantic_only = set(semantic_fields) - set(filters) - excluded_fields
    if semantic_only:
        lines.append(indent + "semantic/detail: " + ", ".join(sorted(semantic_only)))
    return lines


def _format_values(values: list[str]) -> str:
    return ", ".join(values)


def _format_range(min_value: float | None, max_value: float | None) -> str:
    minimum = "*" if min_value is None else str(min_value)
    maximum = "*" if max_value is None else str(max_value)
    return f"{minimum} to {maximum}"
