# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An empty search is a fact about the query, not about the shop."""

from __future__ import annotations

from types import SimpleNamespace

from chain_server.src.catalog_search import _advertised_counts
from chain_server.src.turn_support import _customer_safe_search_evidence


def _capabilities():
    """The shape the catalog service already publishes."""

    return SimpleNamespace(
        taxonomy=SimpleNamespace(
            categories={
                "apparel": SimpleNamespace(
                    product_count=102,
                    subcategories={
                        "skirts": SimpleNamespace(product_count=39),
                        "sweaters": SimpleNamespace(product_count=18),
                        "blouses": SimpleNamespace(product_count=9),
                    },
                )
            }
        )
    )


def test_the_counts_come_from_what_the_catalog_advertises() -> None:
    """A search matching nothing must still know what the shop holds.

    A live session said "I am not seeing any blouses, camisoles, jumpsuits,
    skirts, or sweaters" about a catalog holding sixty-nine of them.
    """

    counts = _advertised_counts(
        _capabilities(),
        {"category": ["apparel"], "subcategory": ["skirts", "sweaters"]},
    )

    assert counts == {"skirts": 39, "sweaters": 18}


def test_a_category_search_reports_the_category() -> None:
    """No subcategory named, so the category total is the honest figure."""

    counts = _advertised_counts(_capabilities(), {"category": ["apparel"]})

    assert counts == {"apparel": 102}


def test_a_category_and_its_parts_are_not_reported_together() -> None:
    """Reporting 102 beside 39 and 18 reads as a contradiction."""

    counts = _advertised_counts(
        _capabilities(),
        {"category": ["apparel"], "subcategory": ["skirts"]},
    )

    assert counts == {"skirts": 39}


def test_the_zero_result_evidence_carries_the_number() -> None:
    """The rule was already stated in prose and still produced the falsehood.

    Forbidding the wrong conclusion is not the same as supplying the right
    fact. This asserts the fact reaches the model.
    """

    summary = _customer_safe_search_evidence(
        {
            "outcome": "zero_results",
            "taxonomy": {"subcategory": ["skirts"]},
            "advertised_counts": {"skirts": 39},
        }
    )

    assert "CATALOG_STOCKS" in summary
    assert "skirts 39" in summary


def test_nothing_advertised_says_nothing() -> None:
    """An unknown taxonomy must not grow an empty, confusing line."""

    summary = _customer_safe_search_evidence(
        {"outcome": "zero_results", "taxonomy": {}, "advertised_counts": {}}
    )

    assert "CATALOG_STOCKS" not in summary
