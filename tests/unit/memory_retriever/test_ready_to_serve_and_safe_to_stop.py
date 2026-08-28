# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an orchestrator needs to know: can this pod serve, and may it be stopped?

Three separate questions that were all answered by one static "healthy":

  liveness   is the process alive, or should it be restarted
  readiness  should this pod be sent traffic yet
  shutdown   will a stop drain in-flight work or cut it off

Kubernetes asks all three and gets a different answer to each. Answering them
with one endpoint that always says yes means a pod takes traffic before its
migrations finish, and every rollout drops whatever was in flight.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from memory_retriever.src import main as memory_main
from memory_retriever.src.migrations import (
    expected_schema_version,
    run_schema_migrations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def migrated_db(monkeypatch: pytest.MonkeyPatch):
    """A database built the way a deployed one is: through the migrations.

    `create_all` would not leave a schema_migrations table at all, and that
    table is the whole subject of the readiness check.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(memory_main, "engine", engine)
    monkeypatch.setattr(memory_main, "SessionLocal", sessionmaker(bind=engine))
    run_schema_migrations(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(migrated_db) -> TestClient:
    return TestClient(memory_main.app)


def test_a_pod_behind_on_migrations_says_it_is_not_ready(client) -> None:
    with memory_main.engine.begin() as connection:
        connection.execute(text("DELETE FROM schema_migrations"))

    response = client.get("/ready")

    assert response.status_code == 503
    assert "needs" in response.json()["detail"]


def test_a_migrated_pod_says_it_is_ready(client) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["schema_version"] == expected_schema_version()


def test_liveness_still_answers_when_readiness_does_not(client) -> None:
    """They have to be able to disagree, or there was no point separating them.

    A pod mid-migration is alive and must not be restarted, but must not be sent
    traffic either. One endpoint cannot say both.
    """

    with memory_main.engine.begin() as connection:
        connection.execute(text("DELETE FROM schema_migrations"))

    assert client.get("/ready").status_code == 503
    assert client.get("/health").status_code == 200


def test_the_database_endpoints_do_not_run_on_the_event_loop() -> None:
    """`async def` on a blocking endpoint is a lock over the whole service.

    FastAPI runs a plain `def` endpoint in the threadpool and an `async def` one
    on the event loop. Every endpoint here does blocking database work, so
    `async def` meant one request at a time for the entire process, reads
    included. This is the guard against one drifting back.
    """

    source = (REPO_ROOT / "memory_retriever" / "src" / "main.py").read_text()
    blocking_but_async = [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef)
        and any(
            getattr(arg, "arg", None) == "db"
            for arg in node.args.args + node.args.kwonlyargs
        )
    ]

    assert blocking_but_async == []


def test_liveness_stays_on_the_event_loop_on_purpose() -> None:
    """The one endpoint that should not move into the threadpool.

    A liveness probe queued behind sixty-four database calls reports a busy pod
    as a dead one and gets it killed. On the loop it answers instantly unless
    the loop itself is stuck, which is the only thing a restart actually fixes.
    """

    source = (REPO_ROOT / "memory_retriever" / "src" / "main.py").read_text()
    health = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == "health_check"
    )

    assert isinstance(health, ast.AsyncFunctionDef)


@pytest.mark.parametrize(
    "service",
    ["memory_retriever", "catalog_retriever", "chain_server"],
)
def test_a_stop_drains_instead_of_cutting_off(service: str) -> None:
    dockerfile = (REPO_ROOT / service / "Dockerfile").read_text()

    assert "--timeout-graceful-shutdown" in dockerfile
    # Without `exec` the shell stays PID 1 and swallows the SIGTERM that starts
    # the drain, so the graceful window above would never begin.
    assert "exec uvicorn" in dockerfile


def test_the_reload_watcher_is_off_unless_asked_for() -> None:
    """--reload runs a supervising parent, which breaks the drain above.

    The server becomes a child of the file watcher, so SIGTERM goes to the
    watcher and in-flight turns are cut off however long the graceful window is.
    """

    dockerfile = (REPO_ROOT / "chain_server" / "Dockerfile").read_text()

    assert "CHAIN_SERVER_RELOAD" in dockerfile
    assert "--port 8009 --reload" not in dockerfile


def test_the_chain_server_window_outlasts_a_whole_turn() -> None:
    """A turn may run to DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS, 150 here.

    A shorter window means a rollout kills shoppers mid-answer, which is the
    failure this is for.
    """

    dockerfile = (REPO_ROOT / "chain_server" / "Dockerfile").read_text()

    assert "SHUTDOWN_GRACE_SECONDS:-160" in dockerfile
