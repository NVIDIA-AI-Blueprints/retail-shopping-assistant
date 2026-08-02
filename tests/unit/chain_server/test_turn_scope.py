# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from chain_server.src.turn_scope import CatalogRepairState, TurnScope


def test_each_turn_gets_independent_mutable_state() -> None:
    """Two turns must not share budgets, evidence, or repair bookkeeping.

    These fields were closure locals, so isolation came free. As dataclass
    fields it has to be asserted: a shared mutable default would leak one
    shopper's search budget and product evidence into another's turn.
    """

    first = TurnScope()
    second = TurnScope()

    first.catalog_searches += 1
    first.product_detail_reads += 1
    first.searched_catalog_scopes.append({"category": ["bags"]})
    first.searched_shopper_scopes.add(("bags", "text"))
    first.retrieved["Cobalt Bag"] = "http://example/bag.png"
    first.product_resolution_used = True
    first.repair.constraint_reviewed_scopes.add("bags")
    first.repair.pending_schema_requirements.append("denim")
    first.repair.pending_constraint_reviews["bags"] = {"reviewed": True}

    assert second.catalog_searches == 0
    assert second.product_detail_reads == 0
    assert second.searched_catalog_scopes == []
    assert second.searched_shopper_scopes == set()
    assert second.retrieved == {}
    assert second.product_resolution_used is False
    assert second.repair.constraint_reviewed_scopes == set()
    assert second.repair.pending_schema_requirements == []
    assert second.repair.pending_constraint_reviews == {}
    assert second.repair is not first.repair
    assert second.product_evidence is not first.product_evidence
    assert second.catalog_lock is not first.catalog_lock
    assert second.resolution_lock is not first.resolution_lock


def test_repair_state_starts_with_nothing_in_flight() -> None:
    repair = CatalogRepairState()

    assert repair.failed_repair_scope_key is None
    assert repair.failed_constraint_scope_key is None
    assert repair.failed_agent_selected_scope is False
    assert repair.pending_taxonomy_constraints is None
    assert repair.pending_no_direct_constraint_clear is False


def test_retrieved_is_shared_by_reference_for_in_place_mutation() -> None:
    """Tools mutate ``retrieved`` in place and the runtime reads the same dict."""

    scope = TurnScope()
    exposed = scope.retrieved

    scope.retrieved["Navy Blazer"] = "http://example/blazer.png"

    assert exposed is scope.retrieved
    assert exposed == {"Navy Blazer": "http://example/blazer.png"}
