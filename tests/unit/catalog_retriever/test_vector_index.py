# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vector index the catalog builds, and the search parameters that go with it."""

from __future__ import annotations

import math

import pytest
from catalog_retriever.src.vector_index import GPU_CAGRA_MAX_LIMIT, VectorIndexSettings
from pydantic import ValidationError


def test_the_default_is_the_cpu_index_the_catalog_has_always_built() -> None:
    settings = VectorIndexSettings()

    assert settings.index_params() == {
        "metric_type": "COSINE",
        "index_type": "AUTOINDEX",
        "params": {},
    }
    assert settings.search_params(205) == {"metric_type": "COSINE", "params": {}}
    assert settings.max_limit is None


def test_the_default_leaves_vectors_as_they_arrive() -> None:
    assert VectorIndexSettings().prepare([3, 4]) == [3.0, 4.0]


def test_gpu_cagra_uses_its_build_parameters_and_ip() -> None:
    """GPU indexes take L2 or IP, not COSINE."""

    params = VectorIndexSettings(type="GPU_CAGRA").index_params()

    assert params == {
        "metric_type": "IP",
        "index_type": "GPU_CAGRA",
        "params": {
            "intermediate_graph_degree": 128,
            "graph_degree": 100,
            "build_algo": "NN_DESCENT",
            "cache_dataset_on_device": "true",
            "adapt_for_cpu": "true",
        },
    }


def test_gpu_search_keeps_the_graph_on_the_gpu() -> None:
    params = VectorIndexSettings(type="GPU_CAGRA", gpu_search=True).index_params()

    assert params["params"]["adapt_for_cpu"] == "false"


def test_a_cpu_search_of_a_gpu_graph_sends_ef_no_smaller_than_the_limit() -> None:
    """Milvus requires ef once adapt_for_cpu is on, and rejects ef below limit."""

    settings = VectorIndexSettings(type="GPU_CAGRA", ef=100)

    assert settings.search_params(10) == {"metric_type": "IP", "params": {"ef": 100}}
    assert settings.search_params(400) == {"metric_type": "IP", "params": {"ef": 400}}


@pytest.mark.parametrize(
    ("limit", "itopk_size"),
    [(4, 64), (64, 64), (65, 128), (205, 256), (1024, 1024)],
)
def test_a_gpu_search_keeps_at_least_limit_intermediate_results(limit, itopk_size) -> None:
    settings = VectorIndexSettings(type="GPU_CAGRA", gpu_search=True)

    assert settings.search_params(limit) == {
        "metric_type": "IP",
        "params": {"itopk_size": itopk_size},
    }


def test_gpu_cagra_caps_a_search_at_its_limit() -> None:
    assert VectorIndexSettings(type="GPU_CAGRA").max_limit == GPU_CAGRA_MAX_LIMIT == 1024


def test_ip_vectors_are_normalized_so_ip_equals_cosine() -> None:
    vector = VectorIndexSettings(type="GPU_CAGRA").prepare([3, 4])

    assert vector == pytest.approx([0.6, 0.8])
    assert math.hypot(*vector) == pytest.approx(1.0)


def test_a_zero_vector_is_refused_rather_than_divided_by_zero() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        VectorIndexSettings(type="GPU_CAGRA").prepare([0.0, 0.0])


def test_an_unknown_index_type_is_refused() -> None:
    with pytest.raises(ValidationError):
        VectorIndexSettings(type="HNSW")


def test_ef_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        VectorIndexSettings(type="GPU_CAGRA", ef=0)


def test_the_signature_names_everything_that_changes_the_built_index() -> None:
    assert VectorIndexSettings().signature == "AUTOINDEX/COSINE"
    assert VectorIndexSettings(type="GPU_CAGRA").signature == (
        "GPU_CAGRA/IP/adapt_for_cpu=True"
    )
    assert VectorIndexSettings(type="GPU_CAGRA", gpu_search=True).signature == (
        "GPU_CAGRA/IP/adapt_for_cpu=False"
    )
    assert VectorIndexSettings(ef=400).signature == VectorIndexSettings().signature
