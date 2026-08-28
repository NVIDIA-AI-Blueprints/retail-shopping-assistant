# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A request that has been admitted is certain of getting a connection.

Sixteen concurrent readers used to stop this service answering anything at all,
`/health` included, until it was restarted. The pool held fifteen connections;
`get_db` is a synchronous dependency, so FastAPI ran it in a forty-thread pool
that knew nothing about that number. Demand outran supply and the surplus waited
thirty seconds each for connections that were never coming back.

Three limits decide whether that can happen -- how many requests uvicorn admits,
how many threads run the dependencies, and how many connections exist -- and the
bug was that they were three unrelated defaults. These tests are about them
being one number.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from memory_retriever.src.database import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    build_engine,
    configured_max_concurrent_requests,
)


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    # A file, not `:memory:`: the pooling this is about only applies to the
    # engine the service actually builds.
    return f"sqlite:///{tmp_path / 'pool.db'}"


def test_the_pool_holds_a_connection_for_every_request_that_can_be_in_flight(
    database_url: str,
) -> None:
    engine = build_engine(database_url, max_concurrent_requests=7)

    assert engine.pool.size() == 7
    # No overflow. Overflow does not answer the question of whether there are
    # enough connections, it only moves the moment the answer arrives.
    assert engine.pool._max_overflow == 0


def test_every_admitted_request_gets_one_and_the_next_is_refused_quickly(
    database_url: str,
) -> None:
    engine = build_engine(database_url, max_concurrent_requests=3)

    held = [engine.connect() for _ in range(3)]
    try:
        with pytest.raises(SQLAlchemyTimeoutError):
            engine.connect()
    finally:
        for connection in held:
            connection.close()

    # Reachable only if the three limits have drifted, and then the point is to
    # say so rather than to block a caller for the thirty-second default.
    assert engine.pool._timeout <= 5.0


def test_the_three_limits_read_the_same_setting(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_MAX_CONCURRENT_REQUESTS", "11")

    assert configured_max_concurrent_requests() == 11
    assert build_engine(database_url).pool.size() == 11


def test_the_threadpool_is_matched_to_the_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anyio
    import anyio.to_thread

    from memory_retriever.src.main import _match_threadpool_to_the_connection_pool

    monkeypatch.setenv("MEMORY_MAX_CONCURRENT_REQUESTS", "9")

    async def on_a_loop() -> int:
        # The limiter belongs to the running event loop, which is why the real
        # call site is the lifespan rather than import time.
        _match_threadpool_to_the_connection_pool()
        return anyio.to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(on_a_loop) == 9


def test_the_server_admits_no_more_requests_than_it_has_connections() -> None:
    """The third limit lives in the Dockerfile, so it is checked there.

    uvicorn's default is unlimited, which is what let requests pile up behind an
    exhausted pool. With this flag the surplus is refused immediately and a
    caller can retry, rather than holding a connection slot while it waits.
    """

    dockerfile = (
        Path(__file__).resolve().parents[3] / "memory_retriever" / "Dockerfile"
    ).read_text()

    assert "--limit-concurrency" in dockerfile
    # The shell is only there to expand the variable. Without `exec` it stays
    # as PID 1 and swallows the SIGTERM that should drain in-flight requests.
    assert "exec uvicorn" in dockerfile
    assert "MEMORY_MAX_CONCURRENT_REQUESTS" in dockerfile
    assert str(DEFAULT_MAX_CONCURRENT_REQUESTS) in dockerfile


def test_a_concurrency_of_zero_is_refused_rather_than_silently_serialising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_MAX_CONCURRENT_REQUESTS", "0")

    with pytest.raises(ValueError):
        configured_max_concurrent_requests()


def test_the_default_survives_an_unset_environment(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEMORY_MAX_CONCURRENT_REQUESTS", raising=False)

    assert configured_max_concurrent_requests() == DEFAULT_MAX_CONCURRENT_REQUESTS
    assert build_engine(database_url).pool.size() == DEFAULT_MAX_CONCURRENT_REQUESTS


def test_an_explicit_poolclass_still_wins(database_url: str) -> None:
    """Tests build engines with StaticPool; sizing must not take that away."""

    from sqlalchemy.pool import StaticPool

    engine = build_engine(database_url, poolclass=StaticPool)

    assert isinstance(engine.pool, StaticPool)
