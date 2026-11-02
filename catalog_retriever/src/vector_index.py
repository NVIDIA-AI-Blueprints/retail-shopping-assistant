# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which Milvus vector index the catalog builds, and how it is searched.

AUTOINDEX with COSINE is the default and runs on the CPU Milvus image.
GPU_CAGRA needs the ``-gpu`` image. docs/VECTOR_SEARCH.md covers when to use
which.
"""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: Milvus refuses a GPU_CAGRA search whose ``limit`` exceeds this.
GPU_CAGRA_MAX_LIMIT = 1024


class VectorIndexSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["AUTOINDEX", "GPU_CAGRA"] = "AUTOINDEX"
    #: GPU_CAGRA only. False builds on the GPU and searches on the CPU
    #: (``adapt_for_cpu``).
    gpu_search: bool = False
    #: Search breadth when a CPU searches a GPU-built graph. Raised to the
    #: request's limit when smaller, since Milvus rejects ef below it.
    ef: int = Field(default=100, ge=1)

    @property
    def metric_type(self) -> str:
        # Milvus GPU indexes take L2 or IP, not COSINE. On unit vectors IP is
        # the cosine, so scores and their ordering are unchanged.
        return "COSINE" if self.type == "AUTOINDEX" else "IP"

    @property
    def normalizes_vectors(self) -> bool:
        return self.metric_type == "IP"

    @property
    def max_limit(self) -> int | None:
        return GPU_CAGRA_MAX_LIMIT if self.type == "GPU_CAGRA" else None

    @property
    def signature(self) -> str:
        """Everything that changes the built index, for the catalog fingerprint."""

        if self.type == "AUTOINDEX":
            return f"{self.type}/{self.metric_type}"
        return f"{self.type}/{self.metric_type}/adapt_for_cpu={not self.gpu_search}"

    def index_params(self) -> dict[str, Any]:
        if self.type == "AUTOINDEX":
            return {"metric_type": self.metric_type, "index_type": "AUTOINDEX", "params": {}}
        return {
            "metric_type": self.metric_type,
            "index_type": "GPU_CAGRA",
            "params": {
                "intermediate_graph_degree": 128,
                "graph_degree": 100,
                "build_algo": "NN_DESCENT",
                "cache_dataset_on_device": "true",
                "adapt_for_cpu": "false" if self.gpu_search else "true",
            },
        }

    def search_params(self, limit: int) -> dict[str, Any]:
        if self.type == "AUTOINDEX":
            params: dict[str, Any] = {}
        elif self.gpu_search:
            params = {"itopk_size": _itopk_size(limit)}
        else:
            params = {"ef": max(self.ef, limit)}
        return {"metric_type": self.metric_type, "params": params}

    def prepare(self, embedding: list[float]) -> list[float]:
        vector = [float(value) for value in embedding]
        if not self.normalizes_vectors:
            return vector
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Cannot index or search with a zero-length embedding.")
        return [value / norm for value in vector]


def _itopk_size(limit: int) -> int:
    # CAGRA keeps itopk_size intermediate results and must keep at least
    # `limit`; Milvus documents it as a power of two.
    size = 64
    while size < limit:
        size *= 2
    return min(size, GPU_CAGRA_MAX_LIMIT)
