# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``catalog_retriever.src.retriever``.

The real ``Retriever.__init__`` opens Milvus connections and OpenAI clients,
neither of which we want to stand up in unit tests. We patch those out so
tests can concentrate on the pure/chunking/filter logic and the ``retrieve``
flow end-to-end with a stubbed Milvus layer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from catalog_retriever.src import retriever as retriever_mod
from catalog_retriever.src.retriever import (
    ImageEmbeddings,
    Retriever,
    RetrieverConfig,
    TextEmbeddings,
)


# --------------------------------------------------------------------------->
# Fixtures
# --------------------------------------------------------------------------->


@pytest.fixture
def retriever_config() -> RetrieverConfig:
    return RetrieverConfig(
        text_embed_port="http://localhost:9000/v1",
        image_embed_port="http://localhost:9001/v1",
        text_model_name="text-model",
        image_model_name="image-model",
        db_port="http://localhost:19530",
        db_name="catalog",
        sim_threshold=0.1,
        text_collection="text_col",
        image_collection="image_col",
    )


@pytest.fixture
def retriever(
    retriever_config: RetrieverConfig, monkeypatch: pytest.MonkeyPatch
) -> Retriever:
    """Build a ``Retriever`` with Milvus and OpenAI dependencies stubbed."""

    class _FakeOpenAI:
        def __init__(self, *_, **__) -> None:
            self.embeddings = SimpleNamespace(create=lambda **_: None)

    class _FakeMilvus:
        def __init__(self, *_, **__) -> None:
            self.col = None

        def similarity_search_with_relevance_scores(self, *_, **__):  # pragma: no cover - replaced per test
            raise NotImplementedError

    monkeypatch.setattr(retriever_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(retriever_mod, "Milvus", _FakeMilvus)
    return Retriever(config=retriever_config)


def _doc(
    name: str,
    price: Any = 49.99,
    category: str = "dress",
    subcategory: str | None = None,
) -> SimpleNamespace:
    """Build a minimal object that quacks like a LangChain ``Document``.

    ``subcategory`` defaults to ``category`` so tests that only override one
    axis don't accidentally get cross-category metadata (which would alias
    through the substring-based category filter in ``retrieve``).
    """
    sub = subcategory if subcategory is not None else category
    page_content = f"{name} | desc | {category},{sub}"
    return SimpleNamespace(
        page_content=page_content,
        metadata={
            "pk": id(name),
            "name": name,
            "price": price,
            "image": f"{name.lower().replace(' ', '_')}.jpg",
        },
    )


# --------------------------------------------------------------------------->
# RetrieverConfig pydantic contract
# --------------------------------------------------------------------------->


class TestRetrieverConfig:
    def test_all_fields_required(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            RetrieverConfig()  # type: ignore[call-arg]

    def test_valid_config_builds(self) -> None:
        cfg = RetrieverConfig(
            text_embed_port="http://a",
            image_embed_port="http://b",
            text_model_name="t",
            image_model_name="i",
            db_port="http://c",
            db_name="d",
            sim_threshold=0.5,
            text_collection="tc",
            image_collection="ic",
        )
        assert cfg.sim_threshold == 0.5


# --------------------------------------------------------------------------->
# Pure helpers
# --------------------------------------------------------------------------->


class TestCoerceFloat:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (42, 42.0),
            (3.14, 3.14),
            ("19.99", 19.99),
            ("$1,299.00", 1299.0),
            ("  $25  ", 25.0),
            ("nonsense", None),
            (["list"], None),
            ({}, None),
        ],
    )
    def test_coercion(self, value: Any, expected: float | None) -> None:
        assert Retriever._coerce_float(value) == (
            expected if expected is None else pytest.approx(expected)
        )


class TestApplyStructuredFilters:
    def test_returns_input_when_filters_empty(self, retriever: Retriever) -> None:
        results = [(_doc("A", 10), 0.9), (_doc("B", 20), 0.8)]
        assert retriever._apply_structured_filters(results, filters=None) == results
        assert retriever._apply_structured_filters(results, filters={}) == results

    def test_min_price_filter(self, retriever: Retriever) -> None:
        results = [
            (_doc("Cheap", 10), 0.9),
            (_doc("Mid", 50), 0.8),
            (_doc("Pricey", 200), 0.7),
        ]
        filtered = retriever._apply_structured_filters(
            results, filters={"min_price": 30}
        )
        assert [r[0].metadata["name"] for r in filtered] == ["Mid", "Pricey"]

    def test_max_price_filter(self, retriever: Retriever) -> None:
        results = [
            (_doc("Cheap", 10), 0.9),
            (_doc("Mid", 50), 0.8),
            (_doc("Pricey", 200), 0.7),
        ]
        filtered = retriever._apply_structured_filters(
            results, filters={"max_price": 100}
        )
        assert [r[0].metadata["name"] for r in filtered] == ["Cheap", "Mid"]

    def test_range_filter(self, retriever: Retriever) -> None:
        results = [
            (_doc("Cheap", 10), 0.9),
            (_doc("Mid", 50), 0.8),
            (_doc("Pricey", 200), 0.7),
        ]
        filtered = retriever._apply_structured_filters(
            results, filters={"min_price": 20, "max_price": 100}
        )
        assert [r[0].metadata["name"] for r in filtered] == ["Mid"]

    def test_missing_price_metadata_excluded(self, retriever: Retriever) -> None:
        results = [
            (_doc("A", 50), 0.9),
            (_doc("No Price", None), 0.8),
            (_doc("Bad Price", "free"), 0.7),
        ]
        filtered = retriever._apply_structured_filters(
            results, filters={"min_price": 10}
        )
        assert [r[0].metadata["name"] for r in filtered] == ["A"]

    def test_no_price_filters_returns_input(self, retriever: Retriever) -> None:
        # ``min_price`` / ``max_price`` both missing → just passthrough.
        results = [(_doc("A", 10), 0.9)]
        assert retriever._apply_structured_filters(results, filters={"color": "red"}) == results


# --------------------------------------------------------------------------->
# Text chunking / batching
# --------------------------------------------------------------------------->


class TestCreateTextChunks:
    def test_empty_input(self, retriever: Retriever) -> None:
        chunks, counts = retriever._create_text_chunks([])
        assert chunks == []
        assert counts == []

    def test_short_text_single_chunk(self, retriever: Retriever) -> None:
        chunks, counts = retriever._create_text_chunks(["short text"])
        assert len(chunks) == 1
        assert counts == [1]
        assert "short text" in chunks[0]

    def test_long_text_multiple_chunks(self, retriever: Retriever) -> None:
        long = "word " * 500  # ~2500 chars, chunk_size=1000 → multiple chunks
        chunks, counts = retriever._create_text_chunks([long])
        assert len(chunks) >= 2
        assert counts == [len(chunks)]

    def test_multiple_texts(self, retriever: Retriever) -> None:
        chunks, counts = retriever._create_text_chunks(["short", "also short"])
        assert len(chunks) == 2
        assert counts == [1, 1]


class TestEmbedChunksInBatches:
    def test_batches_and_collects_embeddings(self, retriever: Retriever) -> None:
        chunks = ["a", "b", "c", "d", "e"]

        def _create(**kwargs):
            # Return one embedding per input text.
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in kwargs["input"]]
            )

        retriever.text_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        embeddings = retriever._embed_chunks_in_batches(chunks, query_type="query", batch_size=2)

        assert len(embeddings) == 5
        for emb in embeddings:
            assert emb == [1.0, 2.0]

    def test_batch_error_fills_with_none(self, retriever: Retriever) -> None:
        def _create(**_):
            raise RuntimeError("API down")

        retriever.text_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        embeddings = retriever._embed_chunks_in_batches(["a", "b"], query_type="query")
        assert embeddings == [None, None]


class TestReconstructEmbeddings:
    def test_averages_chunk_embeddings_per_text(self, retriever: Retriever) -> None:
        embeddings = [
            [1.0, 2.0],  # text 0 chunk 0
            [3.0, 4.0],  # text 0 chunk 1
            [5.0, 6.0],  # text 1 chunk 0
        ]
        counts = [2, 1]

        result = retriever._reconstruct_embeddings(
            ["t0", "t1"], embeddings, counts
        )

        assert len(result) == 2
        assert list(result[0]) == [2.0, 3.0]
        assert list(result[1]) == [5.0, 6.0]

    def test_zero_chunk_text_gets_none(self, retriever: Retriever) -> None:
        result = retriever._reconstruct_embeddings(["t"], [], [0])
        assert result == [None]

    def test_all_failed_embeddings_gives_none(self, retriever: Retriever) -> None:
        result = retriever._reconstruct_embeddings(["t"], [None, None], [2])
        assert result == [None]


class TestTextEmbeddings:
    def test_empty_input_returns_empty_list(self, retriever: Retriever) -> None:
        assert retriever.text_embeddings([]) == []

    def test_happy_path_single_text(self, retriever: Retriever) -> None:
        def _create(**kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in kwargs["input"]]
            )

        retriever.text_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        embeddings = retriever.text_embeddings(["hello world"])
        assert len(embeddings) == 1
        assert embeddings[0] is not None


# --------------------------------------------------------------------------->
# embed_chunk (single query embedding)
# --------------------------------------------------------------------------->


class TestEmbedChunk:
    def test_calls_client_with_query_type(self, retriever: Retriever) -> None:
        captured: Dict[str, Any] = {}

        def _create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.5])])

        retriever.text_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        out = retriever.embed_chunk("hi")

        assert out == [0.5, 0.5]
        assert captured["input"] == "hi"
        assert captured["model"] == retriever.text_model_name
        assert captured["extra_body"]["input_type"] == "query"


# --------------------------------------------------------------------------->
# embeddings_exist
# --------------------------------------------------------------------------->


class TestEmbeddingsExist:
    def test_returns_false_when_no_collections(self, retriever: Retriever) -> None:
        retriever.text_db.col = None
        retriever.image_db.col = None
        assert retriever.embeddings_exist() is False

    def test_returns_true_when_both_populated(self, retriever: Retriever) -> None:
        retriever.text_db.col = SimpleNamespace(
            flush=lambda: None, num_entities=5
        )
        retriever.image_db.col = SimpleNamespace(
            flush=lambda: None, num_entities=7
        )
        assert retriever.embeddings_exist() is True

    def test_returns_false_when_only_one_populated(self, retriever: Retriever) -> None:
        retriever.text_db.col = SimpleNamespace(flush=lambda: None, num_entities=5)
        retriever.image_db.col = SimpleNamespace(flush=lambda: None, num_entities=0)
        assert retriever.embeddings_exist() is False

    def test_exception_surface_as_false(self, retriever: Retriever) -> None:
        def _boom():
            raise RuntimeError("milvus down")

        retriever.text_db.col = SimpleNamespace(flush=_boom, num_entities=0)
        assert retriever.embeddings_exist() is False


# --------------------------------------------------------------------------->
# retrieve() end-to-end with mocked Milvus
# --------------------------------------------------------------------------->


class TestImageEmbeddings:
    def test_empty_input_returns_empty_list(self, retriever: Retriever) -> None:
        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **_: None)
        )
        assert retriever.image_embeddings([]) == []

    def test_base64_inputs_passed_through_to_client(
        self, retriever: Retriever
    ) -> None:
        captured: Dict[str, Any] = {}

        def _create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in kwargs["input"]]
            )

        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        b64 = "data:image/jpeg;base64,AAA"
        embeddings = retriever.image_embeddings([b64, b64])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert captured["model"] == retriever.image_model_name
        assert captured["input"] == [b64, b64]

    def test_url_input_fetched_and_encoded(
        self, retriever: Retriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            retriever_mod,
            "image_url_to_base64",
            lambda url, *a, **k: "data:image/jpeg;base64,fromURL",
        )
        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.5]) for _ in kwargs["input"]]
                )
            )
        )

        embeddings = retriever.image_embeddings(["http://example.com/a.jpg"])
        assert embeddings == [[0.5]]

    def test_path_input_loaded_from_disk(
        self, retriever: Retriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            retriever_mod,
            "image_path_to_base64",
            lambda path, *a, **k: "data:image/jpeg;base64,fromPATH",
        )
        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.9]) for _ in kwargs["input"]]
                )
            )
        )

        embeddings = retriever.image_embeddings(["/app/shared/img.jpg"])
        assert embeddings == [[0.9]]

    def test_too_large_image_resized_then_embedded(
        self, retriever: Retriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        huge = "data:image/jpeg;base64," + ("A" * 70000)
        resized = "data:image/jpeg;base64,small"

        monkeypatch.setattr(
            retriever_mod,
            "resize_base64_image",
            lambda *a, **k: resized,
        )
        captured: Dict[str, Any] = {}

        def _create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])

        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        result = retriever.image_embeddings([huge])

        assert result == [[1.0]]
        assert captured["input"] == [resized]

    def test_failed_resize_turns_input_into_none(
        self, retriever: Retriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        huge = "data:image/jpeg;base64," + ("A" * 70000)
        monkeypatch.setattr(
            retriever_mod, "resize_base64_image", lambda *a, **k: None
        )
        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_: SimpleNamespace(data=[])
            )
        )

        result = retriever.image_embeddings([huge])
        assert result == [None]

    def test_embedding_api_error_returns_none_per_item(
        self, retriever: Retriever
    ) -> None:
        def _create(**_):
            raise RuntimeError("service down")

        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        result = retriever.image_embeddings(
            ["data:image/jpeg;base64,x", "data:image/jpeg;base64,y"]
        )
        assert result == [None, None]

    def test_webp_error_branch_still_returns_none(
        self, retriever: Retriever
    ) -> None:
        def _create(**_):
            raise RuntimeError("webp not supported")

        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=_create)
        )

        result = retriever.image_embeddings(
            ["data:image/webp;base64,bad"], verbose=True
        )
        assert result == [None]

    def test_mixed_valid_and_invalid_inputs(
        self, retriever: Retriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The first item raises at preprocessing → None; the second is valid.
        def _raise_url(_, *a, **k):
            raise ValueError("bad url")

        monkeypatch.setattr(retriever_mod, "image_url_to_base64", _raise_url)
        retriever.image_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.42]) for _ in kwargs["input"]]
                )
            )
        )

        result = retriever.image_embeddings(
            ["http://bad.example.com/a.jpg", "data:image/jpeg;base64,ok"]
        )

        assert result == [None, [0.42]]


class TestMilvusFromCsv:
    def test_skips_when_embeddings_already_exist(
        self, retriever: Retriever, tmp_path
    ) -> None:
        retriever.embeddings_exist = lambda: True  # type: ignore[assignment]
        retriever.text_db.add_embeddings = MagicMock()
        retriever.image_db.add_embeddings = MagicMock()

        retriever.milvus_from_csv(csv_path=str(tmp_path / "anything.csv"))

        retriever.text_db.add_embeddings.assert_not_called()
        retriever.image_db.add_embeddings.assert_not_called()

    def test_populates_collections_when_empty(
        self, retriever: Retriever, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever.embeddings_exist = lambda: False  # type: ignore[assignment]

        # Minimal CSV with the columns read by milvus_from_csv.
        csv_path = tmp_path / "products.csv"
        csv_path.write_text(
            "name,description,category,subcategory,image\n"
            "Silk Dress,elegant,dress,dress,silk.jpg\n"
        )

        retriever.text_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        retriever.image_embeddings = MagicMock(return_value=[[0.3, 0.4]])

        retriever.text_db.add_embeddings = MagicMock()
        retriever.image_db.add_embeddings = MagicMock()

        retriever.milvus_from_csv(csv_path=str(csv_path))

        retriever.text_db.add_embeddings.assert_called_once()
        retriever.image_db.add_embeddings.assert_called_once()

    def test_skips_failed_embeddings_when_populating(
        self, retriever: Retriever, tmp_path
    ) -> None:
        retriever.embeddings_exist = lambda: False  # type: ignore[assignment]

        csv_path = tmp_path / "p.csv"
        csv_path.write_text(
            "name,description,category,subcategory,image\n"
            "A,x,cat,sub,a.jpg\n"
            "B,y,cat,sub,b.jpg\n"
        )

        # One text embedding fails; one image embedding fails.
        retriever.text_embeddings = MagicMock(return_value=[[0.1], None])
        retriever.image_embeddings = MagicMock(return_value=[None, [0.4]])

        retriever.text_db.add_embeddings = MagicMock()
        retriever.image_db.add_embeddings = MagicMock()

        retriever.milvus_from_csv(csv_path=str(csv_path))

        text_call = retriever.text_db.add_embeddings.call_args.kwargs
        assert text_call["embeddings"] == [[0.1]]
        assert len(text_call["texts"]) == 1

        image_call = retriever.image_db.add_embeddings.call_args.kwargs
        assert image_call["embeddings"] == [[0.4]]


class TestRetrieve:
    async def test_text_only_category_match_filters(self, retriever: Retriever) -> None:
        retriever.text_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[
                (_doc("Silk Dress", price=49.99, category="dress"), 0.9),
                (_doc("Leather Bag", price=199, category="bag"), 0.7),
            ]
        )

        texts, ids, sims, names, images = await retriever.retrieve(
            query=["summer outfit"],
            categories=["dress"],
            filters=None,
            image="",
            k=5,
            image_bool=False,
            verbose=False,
        )

        assert names == ["Silk Dress"]
        assert "Silk Dress" in texts[0]
        assert "PRICE: 49.99" in texts[0]
        assert sims == [0.9]
        assert len(ids) == 1
        assert images == ["silk_dress.jpg"]

    async def test_text_only_no_categories_returns_empty(
        self, retriever: Retriever
    ) -> None:
        retriever.text_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(_doc("Silk Dress"), 0.9)]
        )

        texts, ids, sims, names, images = await retriever.retrieve(
            query=["summer outfit"],
            categories=[],
            k=5,
            image_bool=False,
            verbose=False,
        )

        assert (texts, ids, sims, names, images) == ([], [], [], [], [])

    async def test_similarity_threshold_drops_low_scores(
        self, retriever: Retriever
    ) -> None:
        retriever.sim_threshold = 0.5
        retriever.text_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[
                (_doc("A", category="dress"), 0.8),
                (_doc("B", category="dress"), 0.2),
            ]
        )

        texts, _ids, _sims, names, _images = await retriever.retrieve(
            query=["dresses"],
            categories=["dress"],
            k=5,
            image_bool=False,
            verbose=False,
        )

        assert names == ["A"]

    async def test_min_price_filter_applied_before_category(
        self, retriever: Retriever
    ) -> None:
        retriever.text_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[
                (_doc("Cheap", price=5, category="dress"), 0.9),
                (_doc("Mid", price=60, category="dress"), 0.85),
            ]
        )

        _texts, _ids, _sims, names, _images = await retriever.retrieve(
            query=["q"],
            categories=["dress"],
            filters={"min_price": 20},
            k=5,
            image_bool=False,
            verbose=False,
        )

        assert names == ["Mid"]

    async def test_image_search_skips_category_filter(
        self, retriever: Retriever
    ) -> None:
        retriever.text_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(_doc("Text Match", category="dress"), 0.6)]
        )
        retriever.image_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(_doc("Image Match", category="bag"), 0.95)]
        )

        _, _, sims, names, _ = await retriever.retrieve(
            query=["thing"],
            categories=["dress"],
            image="data:image/jpeg;base64,AAA",
            k=5,
            image_bool=True,
            verbose=False,
        )

        # Image search returns results ordered by similarity without category gating.
        assert "Image Match" in names
        assert sims[0] == pytest.approx(0.95)

    async def test_blank_query_replaced_with_dummy(
        self, retriever: Retriever
    ) -> None:
        captured: Dict[str, Any] = {}

        def _search(query: str, k: int = 4) -> List[Tuple[Any, float]]:
            captured["query"] = query
            return [(_doc("A"), 0.9)]

        retriever.text_db.similarity_search_with_relevance_scores = _search
        retriever.image_db.similarity_search_with_relevance_scores = _search

        await retriever.retrieve(
            query=[],
            categories=["dress"],
            image="data:image/jpeg;base64,AAA",
            k=2,
            image_bool=True,
            verbose=False,
        )

        # The dummy query text is injected when query is empty.
        assert "image" in captured["query"].lower()

    async def test_deduplicates_by_pk(self, retriever: Retriever) -> None:
        shared = _doc("Same", category="dress")
        retriever.text_db.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(shared, 0.9), (shared, 0.85)]
        )

        _texts, ids, _sims, names, _images = await retriever.retrieve(
            query=["q"],
            categories=["dress"],
            k=5,
            image_bool=False,
            verbose=False,
        )

        assert len(names) == 1
        assert len(ids) == 1
