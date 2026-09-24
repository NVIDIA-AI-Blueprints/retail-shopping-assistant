# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which sizes a product advertises, and the marker for one-size products."""

from __future__ import annotations

from typing import Any

#: What the catalog carries for a product sold in exactly one size.
_ONE_SIZE = "onesize"


def _advertised_sizes(product: Any) -> list[str]:
    """Read the sizes the catalog states for a product."""

    raw = (getattr(product, "attributes", None) or {}).get("sizes")
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(value).strip() for value in raw if str(value).strip()]
