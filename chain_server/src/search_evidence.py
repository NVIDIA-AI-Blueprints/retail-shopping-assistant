# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed catalog-search evidence, and the text rendered from it.

A search produces typed facts. Those facts were previously rendered into prose
for the model and then parsed back out of that prose to rebuild the composer's
evidence -- the runtime reading its own output to learn what it already knew.

Here the payload is built once and the model-visible text is rendered *from*
it. That ordering matters: the text is a projection of the payload, not a
parallel copy, so a consumer still reading the text cannot disagree with one
reading the payload. Emitting both independently would recreate the drift that
duplicated control prefixes had.

Rendering deliberately stays with the existing ``_format_search_*`` functions
rather than being reimplemented here. They do not share one JSON convention --
filter and taxonomy use ``sort_keys=True, default=str`` while direction and
guidance use ``ensure_ascii=False`` -- so a unified renderer would silently
change what the model sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Artifact key carrying the typed evidence for one search call.
EVIDENCE_KEY = "search_evidence"


@dataclass
class SearchEvidence:
    """Everything a search established, as data rather than prose."""

    outcome: str  # "results" | "zero_results"
    taxonomy: dict[str, Any] = field(default_factory=dict)
    confirmed_filters: dict[str, Any] = field(default_factory=dict)
    semantic_query: str = ""
    shopper_guidance: str = ""
    requested_product_type: str | None = None
    advertised_category: str | None = None
    scope_complete: bool = False
    budget_exhausted: bool = False
    products: list[dict[str, Any]] = field(default_factory=list)

    def as_artifact(self) -> dict[str, Any]:
        return {
            EVIDENCE_KEY: {
                "outcome": self.outcome,
                "taxonomy": self.taxonomy,
                "confirmed_filters": self.confirmed_filters,
                "semantic_query": self.semantic_query,
                "shopper_guidance": self.shopper_guidance,
                "requested_product_type": self.requested_product_type,
                "advertised_category": self.advertised_category,
                "scope_complete": self.scope_complete,
                "budget_exhausted": self.budget_exhausted,
                "products": self.products,
            }
        }


def evidence_of(message: Any) -> dict[str, Any] | None:
    """Read typed search evidence from a tool message, without parsing text."""

    artifact = (
        message.get("artifact")
        if isinstance(message, dict)
        else getattr(message, "artifact", None)
    )
    if not isinstance(artifact, dict):
        return None
    payload = artifact.get(EVIDENCE_KEY)
    return payload if isinstance(payload, dict) else None
