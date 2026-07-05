# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain-server client for catalog-owned capability metadata."""

from __future__ import annotations

import logging
from typing import Any

import requests

from shared.commerce_contracts import CatalogCapabilities


logger = logging.getLogger(__name__)


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
        self.timeout_seconds = timeout_seconds
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
            capabilities = CatalogCapabilities(catalog_id="unavailable")

        self._cached = capabilities
        return capabilities

    def clear(self) -> None:
        self._cached = None
