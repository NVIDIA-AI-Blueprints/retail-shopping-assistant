# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalog's schemas are built once, not once a turn.

Their enums come from the catalog, and the catalog does not change under a
running process: CatalogCapabilitiesClient caches its first successful contract
and never refetches. So the built schema was identical on every turn -- verified
byte-for-byte before this change -- and rebuilding it cost 14ms per turn on the
same event loop that has to serve the turn.

The risk a cache introduces is serving one catalog's schema for another's, so
that is what most of these tests are about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catalog_retriever.src.catalog import load_catalog
from chain_server.src.catalog_capabilities import CatalogCapabilities
from chain_server.src.turn_support import (
    _search_catalog_scopes_input_model,
    _search_catalog_tool_input_model,
    clear_schema_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Each test starts empty, and leaves nothing for the next one."""

    clear_schema_cache()
    yield
    clear_schema_cache()


@pytest.fixture(scope="module")
def capabilities() -> CatalogCapabilities:
    """The catalog the repository ships, read from disk.

    Built from the same two files the service builds from, rather than a
    captured blob: the fixture then cannot drift from the catalog, and nothing
    needs a running service to run the suite.
    """

    snapshot = load_catalog(
        str(REPO_ROOT / "shared/data/enriched_products.jsonl"),
        str(REPO_ROOT / "shared/data/enriched_products.schema.yaml"),
        catalog_id="fashion_products",
        image_enabled=False,
        text_model_name="text-model",
    )
    raw = snapshot.capabilities
    if not isinstance(raw, dict):
        raw = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    return CatalogCapabilities.model_validate(raw)


def test_the_same_catalog_yields_the_same_object(capabilities) -> None:
    first = _search_catalog_tool_input_model(capabilities)
    second = _search_catalog_tool_input_model(capabilities)

    assert first is second


def test_a_different_catalog_gets_its_own_schema(capabilities) -> None:
    """The failure a cache invites: one catalog's enums served for another."""

    other = capabilities.model_copy(update={"catalog_id": "another_shop"})

    assert _search_catalog_tool_input_model(capabilities) is not (
        _search_catalog_tool_input_model(other)
    )


def test_two_catalogs_sharing_a_name_still_get_their_own_schemas(
    capabilities,
) -> None:
    """Why the key is the contract's content and not its catalog_id.

    An id-based key failed thirty-two existing tests, each of which builds a
    catalog named like the shipped one with different fields, and each of which
    was served the other's schema. In production one process sees one catalog
    and this never arises -- which is exactly why it would not have been caught
    without them.
    """

    same_name_different_fields = capabilities.model_copy(
        update={"retrieval_modes": [*capabilities.retrieval_modes, "invented_mode"]}
    )

    assert same_name_different_fields.catalog_id == capabilities.catalog_id
    assert _search_catalog_tool_input_model(capabilities) is not (
        _search_catalog_tool_input_model(same_name_different_fields)
    )


def test_the_audience_field_is_part_of_the_identity(capabilities) -> None:
    """It changes the schema, so it cannot be dropped from the key."""

    plain = _search_catalog_tool_input_model(capabilities)
    with_audience = _search_catalog_tool_input_model(
        capabilities, wearer_audience_field="target_audience"
    )

    assert plain is not with_audience


def test_scope_validation_is_part_of_the_identity(capabilities) -> None:
    strict = _search_catalog_tool_input_model(capabilities, validate_scope=True)
    loose = _search_catalog_tool_input_model(capabilities, validate_scope=False)

    assert strict is not loose


def test_the_scope_count_is_part_of_the_identity(capabilities) -> None:
    one = _search_catalog_scopes_input_model(capabilities, max_scopes=1)
    three = _search_catalog_scopes_input_model(capabilities, max_scopes=3)

    assert one is not three


def test_the_cached_schema_is_the_one_the_builder_produces(capabilities) -> None:
    """A cache that returned something subtly different would be worse than none.

    Compared by rendered schema rather than by identity, since the point is
    that what reaches the model is unchanged.
    """

    from chain_server.src.turn_support import _build_search_catalog_tool_input_model

    built = _build_search_catalog_tool_input_model(
        capabilities, wearer_audience_field="target_audience"
    )
    cached = _search_catalog_tool_input_model(
        capabilities, wearer_audience_field="target_audience"
    )

    assert cached.model_json_schema() == built.model_json_schema()


def test_clearing_the_cache_rebuilds(capabilities) -> None:
    """So a test that does construct two catalogs is not poisoned by an earlier one."""

    first = _search_catalog_tool_input_model(capabilities)
    clear_schema_cache()

    assert _search_catalog_tool_input_model(capabilities) is not first
