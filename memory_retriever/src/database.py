# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite configuration for the memory service."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker


DEFAULT_DATABASE_URL = "sqlite:///./context.db"
DEFAULT_BUSY_TIMEOUT_MS = 5000


def _configured_database_url() -> str:
    configured_url = os.environ.get(
        "MEMORY_DATABASE_URL",
        DEFAULT_DATABASE_URL,
    ).strip()
    database_url = configured_url or DEFAULT_DATABASE_URL
    if not database_url.startswith("sqlite:"):
        raise ValueError("MEMORY_DATABASE_URL must use SQLite")
    return database_url


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
    poolclass: Any | None = None,
) -> Engine:
    """Build a SQLite engine with per-connection safety settings."""

    url = database_url or _configured_database_url()
    if not url.startswith("sqlite:"):
        raise ValueError("The memory service requires SQLite")
    timeout = (
        _configured_busy_timeout_ms() if busy_timeout_ms is None else busy_timeout_ms
    )
    if timeout < 0:
        raise ValueError("busy_timeout_ms must be non-negative")

    engine_kwargs: dict[str, Any] = {
        "connect_args": {
            "check_same_thread": False,
            "timeout": timeout / 1000,
        }
    }
    if poolclass is not None:
        engine_kwargs["poolclass"] = poolclass
    database_engine = create_engine(url, **engine_kwargs)

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
