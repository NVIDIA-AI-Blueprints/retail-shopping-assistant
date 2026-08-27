#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy a memory database from SQLite to PostgreSQL.

Switching MEMORY_DATABASE_URL points the service at an empty database. Every
shopper's cart and conversation history is still in the SQLite file, and nothing
moves it, so this does.

It reads with the models rather than raw SQL, which means the destination gets
rows the current schema agrees with, and anything the schema rejects is found
here instead of on a live request. It refuses to write into a destination that
already holds rows -- the answer to a half-finished run is to drop the schema
and start again, not to merge two partial copies and hope.

Rehearse it against a copy of the volume before running it against the real one:

    docker compose --profile postgres up -d memory-db
    python scripts/copy_memory_to_postgres.py \\
        --source sqlite:////data/context.db \\
        --target postgresql+psycopg://memory:memory@localhost:5432/memory \\
        --dry-run

Drop --dry-run to write. The source is opened read-only and is never modified,
so a failed run costs nothing but the destination.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run as a script from anywhere: the service package is a sibling of scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from memory_retriever.src.database import build_engine
from memory_retriever.src.migrations import run_schema_migrations
from memory_retriever.src.models import (
    CartItem,
    CartMutation,
    CartQuantityIdempotency,
    ConversationEvent,
    ConversationProjection,
    ConversationTurn,
    ShopperProfile,
    User,
)


#: Parents before children: conversation_turns points at shopper_profiles, and
#: the conversation tables point at conversation_turns. Copying a child first
#: fails the foreign key, which SQLite would have allowed and Postgres will not.
TABLES_IN_DEPENDENCY_ORDER = (
    ShopperProfile,
    User,
    CartItem,
    CartMutation,
    CartQuantityIdempotency,
    ConversationTurn,
    ConversationEvent,
    ConversationProjection,
)

#: Rows per flush. Large enough that this is not slow on a hundred thousand
#: rows, small enough not to hold the whole table in memory at once.
BATCH = 500


def _count(session, model) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar() or 0


def _copy_model(source_session, target_session, model, *, dry_run: bool) -> int:
    columns = [column.name for column in model.__table__.columns]
    copied = 0
    for row in source_session.execute(select(model)).scalars().yield_per(BATCH):
        if not dry_run:
            target_session.add(
                model(**{name: getattr(row, name) for name in columns})
            )
        copied += 1
        if copied % BATCH == 0 and not dry_run:
            target_session.flush()
    if not dry_run:
        target_session.flush()
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="sqlite:// URL to read")
    parser.add_argument("--target", required=True, help="postgresql:// URL to write")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count what would be copied and write nothing",
    )
    args = parser.parse_args()

    if not args.source.startswith("sqlite:"):
        parser.error("--source must be a SQLite URL")
    if args.target.startswith("sqlite:"):
        parser.error("--target must not be SQLite; this copies off it")

    source_engine = build_engine(args.source)
    target_engine = build_engine(args.target)

    # The destination needs the schema before it can receive anything, and the
    # service would run these on its next start anyway.
    if not args.dry_run:
        run_schema_migrations(target_engine)

    SourceSession = sessionmaker(bind=source_engine)
    TargetSession = sessionmaker(bind=target_engine)

    with SourceSession() as source, TargetSession() as target:
        if not args.dry_run:
            occupied = {
                model.__tablename__: _count(target, model)
                for model in TABLES_IN_DEPENDENCY_ORDER
                if _count(target, model)
            }
            if occupied:
                print(
                    "refusing to write: the destination already holds rows -- "
                    f"{occupied}.\nDrop the schema and run again rather than "
                    "merging two partial copies.",
                    file=sys.stderr,
                )
                return 1

        print(f"{'table':32s} {'source':>9s} {'copied':>9s}")
        totals = []
        for model in TABLES_IN_DEPENDENCY_ORDER:
            available = _count(source, model)
            copied = _copy_model(source, target, model, dry_run=args.dry_run)
            print(f"{model.__tablename__:32s} {available:9d} {copied:9d}")
            totals.append((model, available, copied))

        if args.dry_run:
            print("\ndry run: nothing was written")
            return 0

        target.commit()

        # Count again from the destination rather than trusting the loop: the
        # number that matters is what is in the database, not what was sent.
        print(f"\n{'table':32s} {'source':>9s} {'target':>9s}")
        mismatched = False
        for model, available, _ in totals:
            landed = _count(target, model)
            flag = "" if landed == available else "   MISMATCH"
            if flag:
                mismatched = True
            print(f"{model.__tablename__:32s} {available:9d} {landed:9d}{flag}")

        if mismatched:
            print("\nrow counts do not agree; do not switch over", file=sys.stderr)
            return 1
        print("\nevery row accounted for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
