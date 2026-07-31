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


def _remove_obsolete_conversation_turn_columns(connection: Connection) -> None:
    """Remove superseded caches and restore the active-turn uniqueness gate."""

    for column_name in (
        "diagnostics_json",
        "start_response_body",
        "finalize_response_body",
    ):
        if column_name in _table_columns(connection, "conversation_turns"):
            connection.execute(
                text(
                    "ALTER TABLE conversation_turns "
                    f"DROP COLUMN {column_name}"
                )
            )
    connection.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_conversation_started "
            "ON conversation_turns (conversation_id) "
            "WHERE status = 'started'"
        )
    )


def _conversation_summary_projection(connection: Connection) -> None:
    """Add the durable rolling-summary boundary without rewriting projections."""

    columns = _table_columns(connection, "conversation_projection")
    if "summary_text" not in columns:
        connection.execute(
            text(
                "ALTER TABLE conversation_projection "
                "ADD COLUMN summary_text TEXT NOT NULL DEFAULT ''"
            )
        )
    if "summary_through_sequence" not in columns:
        connection.execute(
            text(
                "ALTER TABLE conversation_projection "
                "ADD COLUMN summary_through_sequence INTEGER NOT NULL DEFAULT 0"
            )
        )


def _conversation_active_receipts_projection(connection: Connection) -> None:
    """Add the bounded typed-receipt projection with an empty default."""

    columns = _table_columns(connection, "conversation_projection")
    if "active_receipts_json" not in columns:
        connection.execute(
            text(
                "ALTER TABLE conversation_projection "
                "ADD COLUMN active_receipts_json TEXT NOT NULL DEFAULT '[]'"
            )
        )


def _conversation_current_weather_scope_projection(
    connection: Connection,
) -> None:
    """Add the singleton weather-planning scope with a rollback-safe default."""

    columns = _table_columns(connection, "conversation_projection")
    if "current_weather_scope_json" not in columns:
        connection.execute(
            text(
                "ALTER TABLE conversation_projection "
                "ADD COLUMN current_weather_scope_json TEXT NOT NULL "
                "DEFAULT '{\"revision\"\\:0}'"
            )
        )


def _conversation_current_weather_pending_projection(
    connection: Connection,
) -> None:
    """Move v4 pending bindings out of the rollback-readable v3 scope lane."""

    columns = _table_columns(connection, "conversation_projection")
    if "current_weather_pending_json" not in columns:
        connection.execute(
            text(
                "ALTER TABLE conversation_projection "
                "ADD COLUMN current_weather_pending_json TEXT NOT NULL "
                "DEFAULT '{}'"
            )
        )

    pending_keys = (
        "pending_question",
        "pending_source_turn_id",
        "pending_source_sequence",
    )
    rows = connection.execute(
        text(
            "SELECT conversation_id, current_weather_scope_json, "
            "current_weather_pending_json FROM conversation_projection"
        )
    ).mappings()
    for row in rows:
        scope_payload = json.loads(
            row["current_weather_scope_json"] or '{"revision":0}'
        )
        if not isinstance(scope_payload, dict):
            raise ValueError("current weather scope projection must be an object")
        present_pending_keys = [
            key for key in pending_keys if key in scope_payload
        ]
        if not present_pending_keys:
            continue
        existing_pending = json.loads(
            row["current_weather_pending_json"] or "{}"
        )
        if not isinstance(existing_pending, dict):
            raise ValueError(
                "current weather pending projection must be an object"
            )
        if existing_pending:
            raise ValueError("pending weather scope projection is not empty")

        if len(present_pending_keys) != len(pending_keys):
            for key in present_pending_keys:
                scope_payload.pop(key)
            connection.execute(
                text(
                    "UPDATE conversation_projection "
                    "SET current_weather_scope_json = :scope_json, "
                    "current_weather_pending_json = '{}' "
                    "WHERE conversation_id = :conversation_id"
                ),
                {
                    "conversation_id": row["conversation_id"],
                    "scope_json": json.dumps(
                        scope_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            )
            continue

        pending_payload = {
            "scope_revision": scope_payload["revision"],
            **{
                key: scope_payload.pop(key)
                for key in pending_keys
            },
        }
        connection.execute(
            text(
                "UPDATE conversation_projection "
                "SET current_weather_scope_json = :scope_json, "
                "current_weather_pending_json = :pending_json "
                "WHERE conversation_id = :conversation_id"
            ),
            {
                "conversation_id": row["conversation_id"],
                "scope_json": json.dumps(
                    scope_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "pending_json": json.dumps(
                    pending_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )


_MIGRATIONS = (
    (1, _legacy_schema),
    (2, _conversation_schema),
    (3, _conversation_output),
    (4, _conversation_attempt_id),
    (5, _shopper_profiles_schema),
    (6, _conversation_shopper_profile),
    (7, _remove_obsolete_conversation_turn_columns),
    (8, _conversation_summary_projection),
    (9, _conversation_active_receipts_projection),
    (10, _conversation_current_weather_scope_projection),
    (11, _conversation_current_weather_pending_projection),
)


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
