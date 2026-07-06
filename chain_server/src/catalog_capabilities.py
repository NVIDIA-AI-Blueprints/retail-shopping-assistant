# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain-server client for catalog-owned capability metadata."""

from __future__ import annotations

import logging
from typing import Any

import requests

from shared.commerce_contracts import (
    CatalogCapabilities,
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

    def get(self, *, force_refresh: bool = False) -> CatalogCapabilities:
        if self._cached is not None and not force_refresh:
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
            return CatalogCapabilities(catalog_id="unavailable")

        self._cached = capabilities
        return capabilities

    def clear(self) -> None:
        self._cached = None


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

    lines = [f"Catalog ID: {capabilities.catalog_id}"]
    if capabilities.retrieval_modes:
        lines.append(f"Retrieval modes: {', '.join(capabilities.retrieval_modes)}")

    if capabilities.filters:
        lines.append("Hard filters:")
        for name, capability in sorted(capabilities.filters.items()):
            lines.append(f"- {name}: {_format_filter_capability(capability)}")

    return "\n".join(lines)


def _format_filter_capability(capability: CatalogFilterCapability) -> str:
    parts = [capability.type]
    if capability.operators:
        parts.append(f"operators {', '.join(capability.operators)}")
    if capability.values:
        parts.append(f"values {_format_values(capability.values)}")
    elif capability.min_value is not None or capability.max_value is not None:
        parts.append(f"range {_format_range(capability.min_value, capability.max_value)}")
    return "; ".join(parts)


def _format_values(values: list[str], *, limit: int = 80) -> str:
    displayed = values[:limit]
    suffix = f", ... +{len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(displayed) + suffix


def _format_range(min_value: float | None, max_value: float | None) -> str:
    minimum = "*" if min_value is None else str(min_value)
    maximum = "*" if max_value is None else str(max_value)
    return f"{minimum} to {maximum}"
