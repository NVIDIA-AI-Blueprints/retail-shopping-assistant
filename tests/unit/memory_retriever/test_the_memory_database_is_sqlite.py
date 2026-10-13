# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The memory service runs on SQLite only, and says so at startup."""

from __future__ import annotations

import pytest
from memory_retriever.src.database import build_engine


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://memory:memory@localhost:5432/memory",
        "postgresql+psycopg://memory:memory@localhost:5432/memory",
        "mysql://memory@localhost/memory",
    ],
)
def test_a_non_sqlite_url_is_refused_before_any_connection(url: str) -> None:
    with pytest.raises(ValueError, match="must use SQLite"):
        build_engine(url)
