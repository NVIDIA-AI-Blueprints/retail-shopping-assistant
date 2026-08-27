# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database configuration for the memory service.

SQLite stays the default because it is what local work and the test suite
use. Postgres is chosen by URL, and is what removes the two ceilings SQLite
cannot: a file has one writer, and the volume holding it can be mounted by
one pod. Neither is a tuning limit, so neither has a setting that fixes it.

Everything dialect-specific is decided here in `build_engine`, so the rest of
the service never asks which database it is talking to.
"""

from __future__ import annotations

import os
from typing import Any

from hashlib import sha256

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker


DEFAULT_DATABASE_URL = "sqlite:///./context.db"
DEFAULT_BUSY_TIMEOUT_MS = 5000

#: How many requests the service will work on at once. One number, because the
#: connection pool, the threadpool that runs the synchronous dependencies and
#: uvicorn's admission limit all have to agree: a request that has been admitted
#: must be certain of getting a connection. Where they disagree, the surplus
#: waits for a connection it can never be promised, and waiting is the failure
#: -- see the pool timeout below. 64 is comfortably above where this service
#: saturates (reads plateau near 460/s, writes near 230/s) and low enough that
#: refusing the 65th is quick.
DEFAULT_MAX_CONCURRENT_REQUESTS = 64

#: Short on purpose. This is only reachable if the three limits above have
#: drifted apart, and then a fast, loud failure is worth far more than a request
#: that blocks for the SQLAlchemy default of thirty seconds.
_POOL_TIMEOUT_SECONDS = 5.0


def _configured_database_url() -> str:
    configured_url = os.environ.get(
        "MEMORY_DATABASE_URL",
        DEFAULT_DATABASE_URL,
    ).strip()
    database_url = configured_url or DEFAULT_DATABASE_URL
    _require_supported_dialect(database_url)
    return database_url


def _require_supported_dialect(url: str) -> None:
    """Fail at startup, not on the first query of a live deployment."""

    if not url.startswith(("sqlite:", "postgresql:", "postgresql+")):
        raise ValueError(
            "MEMORY_DATABASE_URL must use SQLite or PostgreSQL, got "
            f"{url.split(':', 1)[0]!r}"
        )


def configured_max_concurrent_requests() -> int:
    """Return the one number the pool, the threadpool and uvicorn all use."""

    value = int(
        os.environ.get(
            "MEMORY_MAX_CONCURRENT_REQUESTS",
            str(DEFAULT_MAX_CONCURRENT_REQUESTS),
        )
    )
    if value < 1:
        raise ValueError("MEMORY_MAX_CONCURRENT_REQUESTS must be at least 1")
    return value


def _configured_busy_timeout_ms() -> int:
    timeout = int(
        os.environ.get(
            "MEMORY_SQLITE_BUSY_TIMEOUT_MS",
            str(DEFAULT_BUSY_TIMEOUT_MS),
        )
    )
    if timeout < 0:
        raise ValueError("MEMORY_SQLITE_BUSY_TIMEOUT_MS must be non-negative")
    return timeout


def build_engine(
    database_url: str | None = None,
    *,
    busy_timeout_ms: int | None = None,
    max_concurrent_requests: int | None = None,
    poolclass: Any | None = None,
) -> Engine:
    """Build a SQLite engine with per-connection safety settings."""

    url = database_url or _configured_database_url()
    _require_supported_dialect(url)
    is_sqlite = url.startswith("sqlite:")
    timeout = (
        _configured_busy_timeout_ms() if busy_timeout_ms is None else busy_timeout_ms
    )
    if timeout < 0:
        raise ValueError("busy_timeout_ms must be non-negative")

    engine_kwargs: dict[str, Any] = {}
    if is_sqlite:
        # check_same_thread because the endpoints run in a threadpool, and
        # `timeout` is how long a writer waits for the one write lock.
        engine_kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": timeout / 1000,
        }
    else:
        # A Postgres connection crosses a network and can be closed by
        # something in the middle of it -- a pooler, a failover, an idle
        # reaper. pre_ping spends one round trip finding that out here rather
        # than raising it at whoever was about to run a query.
        engine_kwargs["pool_pre_ping"] = True
    if poolclass is not None:
        engine_kwargs["poolclass"] = poolclass
    else:
        # A connection each for every request that can be in flight, and no
        # overflow: overflow only postpones the question of whether there are
        # enough. SQLite connections are file handles, so holding sixty-four
        # idle ones costs almost nothing, and a WAL database serves readers
        # concurrently, which is what they are mostly for.
        concurrency = (
            configured_max_concurrent_requests()
            if max_concurrent_requests is None
            else max_concurrent_requests
        )
        if concurrency < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        engine_kwargs["pool_size"] = concurrency
        engine_kwargs["max_overflow"] = 0
        engine_kwargs["pool_timeout"] = _POOL_TIMEOUT_SECONDS
    database_engine = create_engine(url, **engine_kwargs)

    if is_sqlite:
        # Both are SQLite defaults that are wrong for a service: foreign keys
        # are off unless asked for, and a writer that finds the lock held gives
        # up at once instead of waiting.
        @event.listens_for(database_engine, "connect")
        def _configure_connection(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={timeout}")
            cursor.close()

    return database_engine


DATABASE_URL = _configured_database_url()
engine = build_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


def begin_write_transaction(db: Any, scope: str) -> None:
    """Take the write lock for `scope` before reading what will be written.

    Several mutations here read a row, decide from it, and then write. Without a
    lock held across both halves, two of them interleave and the second decides
    from state the first has already invalidated.

    SQLite has one write lock for the whole file. `BEGIN IMMEDIATE` takes it at
    the start of the transaction rather than on the first write, which is the
    difference between waiting and failing: a transaction that reads first and
    then tries to upgrade to a writer gets SQLITE_BUSY instead of queueing.
    `scope` is ignored, because there is nothing finer to take.

    PostgreSQL has no such statement and needs none for the whole database --
    readers and writers do not block each other. What it does need is the same
    guarantee for the scope, and an advisory lock gives exactly that: it
    serialises the transactions that name the same scope and nothing else. So
    the swap is not a loosening. Two shoppers in different conversations
    serialise on SQLite and do not here, which is the point of moving.

    The lock is transaction-scoped and released on commit or rollback, so there
    is nothing to unlock and nothing leaks if the request fails.
    """

    if db.bind.dialect.name == "sqlite":
        db.execute(text("BEGIN IMMEDIATE"))
        return

    db.execute(
        text("SELECT pg_advisory_xact_lock(:key)"),
        {"key": advisory_lock_key(scope)},
    )


def advisory_lock_key(scope: str) -> int:
    """A stable signed 64-bit key for a scope name.

    Hashed here rather than with PostgreSQL's `hashtext`, which is an internal
    function with no compatibility promise across versions -- a key that changed
    under an upgrade would silently stop excluding anything.
    """

    digest = sha256(scope.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)
