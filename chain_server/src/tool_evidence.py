# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed tool evidence, and the text rendered from it.

A catalog tool produces typed facts. Those facts were previously rendered into prose
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

#: Artifact key carrying the typed evidence for one product-detail read.
DETAIL_EVIDENCE_KEY = "product_detail_evidence"


@dataclass
class SearchEvidence:
    """Everything a search established, as data rather than prose."""

    outcome: str  # "results" | "zero_results" | "no_direct_catalog_match"
    #: Products the same search finds with its optional constraints dropped,
    #: never its size. Present only on a zero-result scope, so the reply can
    #: show what the shop does have instead of asking which absence to explore.
    relaxed_products: list[Any] = field(default_factory=list)
    relaxed_dropped: list[str] = field(default_factory=list)
    #: False when the only way to find anything was to drop the shopper's size.
    #: The reply must then name the size it is showing instead of theirs.
    relaxed_kept_the_size: bool = True
    taxonomy: dict[str, Any] = field(default_factory=dict)
    confirmed_filters: dict[str, Any] = field(default_factory=dict)
    semantic_query: str = ""
    shopper_guidance: str = ""
    requested_product_type: str | None = None
    advertised_category: str | None = None
    #: True when the shopper named no product for this role and the model
    #: composed it -- "a top" for someone who asked for an outfit. The role is
    #: a suggestion, so the reply may not present it as something the shopper
    #: asked for, and a miss inside ``role_advertised_types`` is not the role
    #: being unavailable.
    composed_role: bool = False
    #: The advertised subcategories that composed role actually covered.
    role_advertised_types: list[str] = field(default_factory=list)
    scope_complete: bool = False
    budget_exhausted: bool = False
    products: list[dict[str, Any]] = field(default_factory=list)
    #: The bounded, product-free outcome diagnostics report, when there is one.
    scope_outcome: dict[str, Any] | None = None
    #: Shopper requirements the catalog cannot enforce as filters. These ranked
    #: the search but were never applied, so no product below is confirmed to
    #: meet them.
    unconfirmed_requirements: list[str] = field(default_factory=list)
    #: Who every returned piece turns out to be for, when nobody said. A search
    #: that does not filter on the audience field still comes back with an
    #: audience, and the shopper is the only party who cannot see that it was
    #: assumed. Empty once the audience is a stated constraint, because then
    #: there is no assumption left to disclose.
    assumed_audience: list[str] = field(default_factory=list)

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
                "composed_role": self.composed_role,
                "role_advertised_types": self.role_advertised_types,
                "scope_complete": self.scope_complete,
                "budget_exhausted": self.budget_exhausted,
                "products": self.products,
                "scope_outcome": self.scope_outcome,
                "unconfirmed_requirements": self.unconfirmed_requirements,
                "assumed_audience": self.assumed_audience,
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


@dataclass
class ProductDetailEvidence:
    """The facts one product-detail read established."""

    products: list[dict[str, Any]] = field(default_factory=list)

    def as_artifact(self) -> dict[str, Any]:
        return {DETAIL_EVIDENCE_KEY: {"products": self.products}}


def detail_evidence_of(message: Any) -> dict[str, Any] | None:
    """Read typed product-detail evidence from a tool message."""

    artifact = (
        message.get("artifact")
        if isinstance(message, dict)
        else getattr(message, "artifact", None)
    )
    if not isinstance(artifact, dict):
        return None
    payload = artifact.get(DETAIL_EVIDENCE_KEY)
    return payload if isinstance(payload, dict) else None
