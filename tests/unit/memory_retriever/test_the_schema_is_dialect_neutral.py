# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema properties that must hold on every dialect, checked without a server.

These read the model definitions, so they run everywhere and cost nothing. That
matters: the bug below was invisible to a suite that only ever ran on SQLite,
and a guard that needs PostgreSQL to be installed is a guard that gets skipped.
"""

from __future__ import annotations


def test_every_partial_index_is_partial_on_both_dialects() -> None:
    """The class of bug, not the one instance of it.

    A partial index declared with only `sqlite_where` is created *whole*
    everywhere else. `uq_conversation_started` was, so on PostgreSQL it became a
    unique index on conversation_id alone and allowed one turn per conversation
    for all time -- while the tests, all on SQLite, passed.

    This is deliberately not skipped when PostgreSQL is absent: it reads the
    model definitions and needs no server, and it is the guard that would have
    caught the original.
    """

    from memory_retriever.src.models import Base

    missing = []
    for table in Base.metadata.tables.values():
        for index in table.indexes:
            dialects = index.dialect_options
            sqlite_where = dialects["sqlite"]._non_defaults.get("where")
            postgres_where = dialects["postgresql"]._non_defaults.get("where")
            if (sqlite_where is None) != (postgres_where is None):
                missing.append(
                    f"{index.name}: sqlite_where="
                    f"{sqlite_where is not None}, postgresql_where="
                    f"{postgres_where is not None}"
                )

    assert missing == []
