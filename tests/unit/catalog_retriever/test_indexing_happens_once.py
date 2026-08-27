# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexing is a deployment step, not something each pod does on the way up.

Rebuilding drops the collections and then refills them. One process doing that
is fine. Two are not, and not merely wastefully: the second can drop what the
first is halfway through filling, and nothing afterwards notices, because the
fingerprint is written row by row -- a half-filled collection carries the right
fingerprint on every row it has, so the check that is supposed to catch a stale
index passes on a partial one.

Nothing in the process can prevent that; there is no lock and a lock across
replicas would need somewhere to live. The deployment prevents it, by building
once as a Job and telling the serving pods not to. These tests are about the
pieces that makes possible: a pod that can describe a catalog without building
it, a check it can answer readiness with, and a build that is reachable on its
own.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from catalog_retriever.src import retriever as retriever_mod
from catalog_retriever.src.retriever import Retriever, RetrieverConfig
from shared.commerce_contracts import CatalogFilterCapability


@pytest.fixture
def retriever_config() -> RetrieverConfig:
    return RetrieverConfig(
        text_embed_port="http://embed",
        image_embed_port=None,
        text_model_name="text-model",
        image_model_name=None,
        text_api_key_env="EMBED_API_KEY",
        image_api_key_env=None,
        image_enabled=False,
        db_port="19530",
        db_name="catalog",
        sim_threshold=0.2,
        text_collection="text",
        image_collection="image",
        filter_capabilities={
            "category": CatalogFilterCapability(
                type="enum", operators=["eq"], source_fields=["category"],
                values=["dress"],
            )
        },
        catalog_size=0,
        product_id_field="product_id",
        name_field="name",
        description_field="description",
        fallback_description_field="description",
        image_field="image",
        price_field="price",
        taxonomy_fields=["category"],
        detail_fields=[],
    )


@pytest.fixture
def retriever(retriever_config, monkeypatch: pytest.MonkeyPatch) -> Retriever:
    class _FakeOpenAI:
        def __init__(self, *_, **__) -> None:
            self.embeddings = SimpleNamespace(create=lambda **_: None)

    class _FakeMilvus:
        def __init__(self, *_, **__) -> None:
            self.col = None
            self.matches_catalog = MagicMock(return_value=False)
            self.reset = MagicMock()
            self.add_embeddings = MagicMock()

    monkeypatch.setattr(retriever_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(retriever_mod, "Milvus", _FakeMilvus)
    return Retriever(config=retriever_config)


def _snapshot(fingerprint: str = "fp1", count: int = 3) -> Any:
    return SimpleNamespace(
        fingerprint=fingerprint,
        product_count=count,
        products=[],
        search_documents=[],
        capabilities=SimpleNamespace(filters={}),
        schema=SimpleNamespace(
            record=SimpleNamespace(
                product_id="product_id", name="name", description="description",
                fallback_description="description", image="image", price="price",
            ),
            taxonomy=SimpleNamespace(fields=["category"]),
            detail_fields=[],
        ),
    )


def test_a_pod_can_learn_the_catalog_without_building_anything(retriever) -> None:
    """The reason a non-indexing pod can still answer questions.

    It has to know which fields it is serving. Before this was separable, the
    only way to find that out was to run the build.
    """

    retriever.describe_snapshot(_snapshot(count=7))

    assert retriever.catalog_size == 7
    retriever.text_db.reset.assert_not_called()
    retriever.text_db.add_embeddings.assert_not_called()


def test_describing_a_catalog_does_not_even_look_at_the_collection(retriever) -> None:
    """Readiness probes call the check; startup calls this. Only one may be slow.

    `matches_catalog` flushes the collection, so describing must not do it.
    """

    retriever.describe_snapshot(_snapshot())

    retriever.text_db.matches_catalog.assert_not_called()


def test_the_check_is_true_only_when_the_index_is_this_catalog(retriever) -> None:
    retriever.text_db.matches_catalog = MagicMock(return_value=True)

    assert retriever.matches_snapshot(_snapshot()) is True

    retriever.text_db.matches_catalog = MagicMock(return_value=False)

    assert retriever.matches_snapshot(_snapshot()) is False


def test_a_current_index_is_never_rebuilt(retriever) -> None:
    """The steady state, and why N pods booting is normally harmless.

    Each checks the fingerprint, sees a match, and stops. The danger is only
    when the catalog has changed and every one of them decides to rebuild.
    """

    retriever.text_db.matches_catalog = MagicMock(return_value=True)

    retriever.sync_snapshot(_snapshot())

    retriever.text_db.reset.assert_not_called()


def test_a_stale_index_is_dropped_before_it_is_rebuilt(retriever) -> None:
    """The dangerous path, stated plainly so the risk is not lost.

    `reset` is what makes concurrent indexing destructive rather than wasteful:
    it removes the collection that another pod may be filling.
    """

    retriever.text_db.matches_catalog = MagicMock(return_value=False)
    retriever.text_embeddings = MagicMock(return_value=[])

    with pytest.raises(RuntimeError):
        # Zero embeddings for three products, so it refuses rather than serving
        # a partial snapshot -- but reset has already happened by then.
        retriever.sync_snapshot(_snapshot(count=3))

    retriever.text_db.reset.assert_called_once()


def test_the_indexer_is_reachable_as_its_own_entry_point() -> None:
    """It has to be runnable as a Job without starting a server.

    Checked by import rather than by running it, because running it would build
    a real retriever against a real Milvus.
    """

    from catalog_retriever.src import index_catalog

    assert callable(index_catalog.main)


def test_a_serving_pod_has_no_way_to_index() -> None:
    """One path, not a switch that could be set wrong.

    A flag permitting a pod to index would only ever be wrong in the deployment
    that has more than one pod -- which is the deployment that matters. So the
    serving module must not call the build at all, and must not read a flag that
    would let it.

    Asserted against the source because importing main builds a retriever and
    needs a live Milvus. That makes this the weakest test here, so it pins exact
    statements rather than names appearing anywhere.
    """

    source = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "catalog_retriever"
        / "src"
        / "main.py"
    ).read_text()

    # The build, in any form, must not be reachable from the serving module.
    assert "sync_snapshot" not in source
    assert "CATALOG_INDEX_ON_BOOT" not in source
    # Without this a pod would not know which fields it serves.
    assert "retriever.describe_snapshot(snapshot)" in source
    # Without this it would serve searches against a missing index and answer
    # them with no products rather than refusing traffic.
    assert "if not index_is_ready():" in source
