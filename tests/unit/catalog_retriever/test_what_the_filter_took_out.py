# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A filtered search keeps the nearest thing the filter removed.

Filtering answers "what fits" and can never answer "does it exist". Those are
different questions, and a shopper asking the second one got the first one's
answer: told a bracelet this catalog sells was not stocked, because it cost
more than the budget the search was filtered by.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from catalog_retriever.src.retriever import Retriever


class _Stub(Retriever):
    """A retriever with only the state these two methods read."""

    def __init__(self) -> None:
        self.product_id_field = "record_id"
        self.name_field = "name"
        self.price_field = "price"
        self.description_field = "description"
        self.fallback_description_field = ""
        self.image_field = "image"
        self.taxonomy_fields = ["category", "subcategory"]
        self.detail_fields = ["price"]


def _hit(record_id: str, name: str, price: str, similarity: float) -> Any:
    document = SimpleNamespace(
        page_content=f"{name} description",
        metadata={
            "record_id": record_id,
            "name": name,
            "price": price,
            "description": f"{name} description",
            "image": "",
            "category": "jewelry",
            "subcategory": "bracelets",
        },
    )
    return (document, similarity)


def test_the_nearest_product_a_filter_removed_comes_back() -> None:
    """The one the shopper asked about, priced out of its own search.

    J10 turn 6. "Is the Southwest Bracelet within that" was searched as
    bracelets under the $110.01 left of a $150 budget. The bracelet is
    $169.99, so the filter removed it, and the reply said this shop does not
    have one. It does.
    """

    southwest = _hit("a", "Southwest Bracelet", "169.99", 0.81)
    pearl = _hit("b", "Pearl Bracelet", "49.99", 0.42)

    near_miss = _Stub()._best_candidate_the_filters_removed(
        [southwest, pearl], [pearl]
    )

    assert near_miss is not None
    assert near_miss["display_name"] == "Southwest Bracelet"
    assert near_miss["price"]["amount"] == 169.99


def test_the_nearest_one_is_the_best_match_not_the_first_seen() -> None:
    """Candidates arrive in retrieval order, which is not similarity order."""

    weak = _hit("a", "Beaded Bracelet Set", "89.99", 0.11)
    strong = _hit("b", "Southwest Bracelet", "169.99", 0.81)

    near_miss = _Stub()._best_candidate_the_filters_removed([weak, strong], [])

    assert near_miss is not None
    assert near_miss["display_name"] == "Southwest Bracelet"


def test_a_filter_that_removed_nothing_reports_nothing() -> None:
    """No exclusion, no evidence about one. The common case pays nothing."""

    kept = [_hit("a", "Pearl Bracelet", "49.99", 0.42)]

    assert _Stub()._best_candidate_the_filters_removed(kept, kept) is None
