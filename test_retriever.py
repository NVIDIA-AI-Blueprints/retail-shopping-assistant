import numpy as np
import pytest

from catalog_retriever.src.retriever import TextEmbeddings


class DummyRetriever:
    def text_embeddings(self, texts):
        # Return deterministic 3-D vectors for each input text.
        return [[float(i + 1), float(i + 2), float(i + 3)] for i in range(len(texts))]


def test_embed_documents_returns_flat_list_of_lists():
    embeddings = TextEmbeddings(DummyRetriever())
    texts = ["first document", "second document", "third document"]
    result = embeddings.embed_documents(texts)

    # The bug returned [[vec1, vec2, vec3]] instead of [vec1, vec2, vec3].
    assert isinstance(result, list)
    assert len(result) == len(texts)
    for vec in result:
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) == 3

