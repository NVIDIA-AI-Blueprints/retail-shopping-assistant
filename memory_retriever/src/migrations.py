# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered, idempotent SQLite migrations for the memory service."""

from __future__ import annotations

import json
import time
from hashlib import sha256

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .models import (
    CartItem,
    CartMutation,
    CartQuantityIdempotency,
    ConversationEvent,
    ConversationProjection,
    ConversationTurn,
    SchemaMigration,
    ShopperProfile,
    User,
    new_cart_line_id,
    new_turn_attempt_id,
)


def cart_mutation_digest(
    operation: str,
    stable_target_id: str,
    request_body: dict,
) -> str:
    canonical = json.dumps(
        {
            "operation": operation,
            "stable_target_id": stable_target_id,
            "request_body": request_body,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _table_columns(connection: Connection, table_name: str) -> set[str]:
    escaped_name = table_name.replace("'", "''")
    return {
        str(row[1])
        for row in connection.execute(
            text(f"PRAGMA table_info('{escaped_name}')")
        ).fetchall()
    }


def ensure_price_column(connection: Connection) -> None:
    if "price" not in _table_columns(connection, "cart_items"):
        connection.execute(text("ALTER TABLE cart_items ADD COLUMN price REAL"))


def ensure_cart_size_column(connection: Connection) -> None:
    """Add the cart size column for databases created before sizes existed.

    Existing rows stay null. They were added when nothing recorded a size, and
    backfilling one would invent a choice the shopper never made.
    """

    if "size" not in _table_columns(connection, "cart_items"):
        connection.execute(text("ALTER TABLE cart_items ADD COLUMN size TEXT"))


def ensure_cart_line_id_column(connection: Connection) -> None:
    columns = _table_columns(connection, "cart_items")
    if "cart_line_id" not in columns:
        connection.execute(text("ALTER TABLE cart_items ADD COLUMN cart_line_id TEXT"))
    rows = connection.execute(
        text(
            "SELECT id FROM cart_items WHERE cart_line_id IS NULL OR cart_line_id = ''"
        )
    ).fetchall()
    for row in rows:
        connection.execute(
            text("UPDATE cart_items SET cart_line_id = :cart_line_id WHERE id = :id"),
            {"cart_line_id": new_cart_line_id(), "id": row[0]},
        )
    connection.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_cart_items_cart_line_id "
            "ON cart_items (cart_line_id)"
        )
    )


def ensure_product_id_column(connection: Connection) -> None:
    if "product_id" not in _table_columns(connection, "cart_items"):
        connection.execute(text("ALTER TABLE cart_items ADD COLUMN product_id TEXT"))
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_cart_items_product_id "
            "ON cart_items (product_id)"
        )
    )


def migrate_quantity_idempotency(connection: Connection) -> None:
    rows = connection.execute(
        text(
            "SELECT idempotency_key, user_id, cart_line_id, quantity, "
            "response_body FROM cart_quantity_idempotency"
        )
    ).mappings()
    for row in rows:
        connection.execute(
            text(
                "INSERT OR IGNORE INTO cart_mutations "
                "(user_id, idempotency_key, operation, canonical_digest, "
                "stable_target_id, response_body) VALUES "
                "(:user_id, :idempotency_key, 'update', :canonical_digest, "
                ":stable_target_id, :response_body)"
            ),
            {
                "user_id": row["user_id"],
                "idempotency_key": row["idempotency_key"],
                "canonical_digest": cart_mutation_digest(
                    "update",
                    row["cart_line_id"],
                    {"quantity": row["quantity"]},
                ),
                "stable_target_id": row["cart_line_id"],
                "response_body": row["response_body"],
            },
        )


def _legacy_schema(connection: Connection) -> None:
    for table in (
        User.__table__,
        CartItem.__table__,
        CartQuantityIdempotency.__table__,
        CartMutation.__table__,
    ):
        table.create(bind=connection, checkfirst=True)
    ensure_price_column(connection)
    ensure_cart_line_id_column(connection)
    ensure_product_id_column(connection)
    migrate_quantity_idempotency(connection)


def _conversation_schema(connection: Connection) -> None:
    for table in (
        ConversationTurn.__table__,
        ConversationEvent.__table__,
        ConversationProjection.__table__,
    ):
        table.create(bind=connection, checkfirst=True)


def _conversation_output(connection: Connection) -> None:
    if "output_json" not in _table_columns(connection, "conversation_turns"):
        connection.execute(
            text("ALTER TABLE conversation_turns ADD COLUMN output_json TEXT")
        )


def _conversation_attempt_id(connection: Connection) -> None:
    if "attempt_id" not in _table_columns(connection, "conversation_turns"):
        connection.execute(
            text("ALTER TABLE conversation_turns ADD COLUMN attempt_id TEXT")
        )
    rows = connection.execute(
        text(
            "SELECT turn_id FROM conversation_turns "
            "WHERE attempt_id IS NULL OR attempt_id = ''"
        )
    ).fetchall()
    for row in rows:
        connection.execute(
            text(
                "UPDATE conversation_turns SET attempt_id = :attempt_id "
                "WHERE turn_id = :turn_id"
            ),
            {"attempt_id": new_turn_attempt_id(), "turn_id": row[0]},
        )


def _shopper_profiles_schema(connection: Connection) -> None:
    ShopperProfile.__table__.create(bind=connection, checkfirst=True)


def _conversation_shopper_profile(connection: Connection) -> None:
    if "shopper_profile_id" not in _table_columns(connection, "conversation_turns"):
        connection.execute(
            text(
                "ALTER TABLE conversation_turns "
                "ADD COLUMN shopper_profile_id VARCHAR(64) "
                "REFERENCES shopper_profiles(shopper_profile_id) "
                "ON DELETE RESTRICT ON UPDATE RESTRICT"
            )
        )


def _cart_size(connection: Connection) -> None:
    """Give cart lines a size.

    Its own version rather than a line inside `_legacy_schema`: version 1 has
    already been applied to every existing database, so anything added there
    now would never run.
    """

    ensure_cart_size_column(connection)


def _cart_line_uniqueness(connection: Connection) -> None:
    """Make one cart line per (user, product, size) a rule the database keeps.

    Written to run on a database that already has rows. It builds the indexes
    directly rather than through `create_all`, so it applies to tables created
    before the model declared them.

    It does not deduplicate first. Checked on the live volume on 2026-08-27:
    2,061 cart rows and zero duplicate groups, so the indexes build cleanly. A
    week earlier it was 249 rows -- the count grows with use, and a database
    that has drifted will need a merge step before this runs. Failing loudly
    here is right: silently dropping a shopper's second line to make an index
    build is worse than a migration that stops and asks.

    Two indexes because `size` is nullable and NULLs do not collide in a unique
    index. `IF NOT EXISTS` so a re-run is a no-op, matching the rest of this
    file.
    """

    connection.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_cart_line_sized "
        "ON cart_items (user_id, product_id, size) WHERE size IS NOT NULL"
    )
    connection.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_cart_line_unsized "
        "ON cart_items (user_id, product_id) WHERE size IS NULL"
    )


_MIGRATIONS = (
    (1, _legacy_schema),
    (2, _conversation_schema),
    (3, _conversation_output),
    (4, _conversation_attempt_id),
    (5, _shopper_profiles_schema),
    (6, _conversation_shopper_profile),
    (7, _cart_size),
    (8, _cart_line_uniqueness),
)


def expected_schema_version() -> int:
    """The version a pod must have reached before it can serve."""

    return max(version for version, _ in _MIGRATIONS)


def run_schema_migrations(database_engine: Engine) -> None:
    """Apply each schema version once without losing existing data."""

    with database_engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        connection.commit()
    SchemaMigration.__table__.create(bind=database_engine, checkfirst=True)

    for version, migrate in _MIGRATIONS:
        with database_engine.begin() as connection:
            applied = connection.execute(
                text("SELECT 1 FROM schema_migrations WHERE version = :version"),
                {"version": version},
            ).first()
            if applied is not None:
                continue
            migrate(connection)
            connection.execute(
                text(
                    "INSERT INTO schema_migrations (version, applied_at) "
                    "VALUES (:version, :applied_at)"
                ),
                {"version": version, "applied_at": time.time()},
            )
