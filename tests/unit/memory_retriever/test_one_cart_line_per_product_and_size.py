# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One cart line per (user, product, size), kept by the database.

The add reads then writes. It is correct today only because the service
serialises every request -- 13 `async def` endpoints doing blocking database
work make the event loop a lock over the whole process. Both changes that let
this scale remove that lock: Postgres swaps database-wide write serialisation
for MVCC, and threadpool-bound endpoints free the loop. After either, two
concurrent adds of the same product and size are a lost update or a duplicate
line.

So the invariant lives in the schema, and these tests are about the schema
keeping it rather than about the add path being careful.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from memory_retriever.src.migrations import run_schema_migrations
from memory_retriever.src.models import CartItem


@pytest.fixture
def session() -> Iterator[object]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Through the migrations, not `create_all`: the index has to reach a
    # database whose tables already existed, which is every deployed one.
    run_schema_migrations(engine)
    maker = sessionmaker(bind=engine)
    db = maker()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _line(db, user_id=1, product_id="p1", size=None, amount=1):
    item = CartItem(
        user_id=user_id,
        product_id=product_id,
        item="A Dress",
        amount=amount,
        price=10.0,
        size=size,
    )
    db.add(item)
    db.flush()
    return item


def test_a_sized_line_cannot_be_inserted_twice(session) -> None:
    _line(session, size="8")
    with pytest.raises(IntegrityError):
        _line(session, size="8")


def test_a_one_size_line_cannot_be_inserted_twice(session) -> None:
    """The case a single three-column index would miss.

    `size` is nullable and NULLs are distinct in a unique index on both SQLite
    and Postgres, so `(user_id, product_id, NULL)` twice would not collide.
    That is bags, sunglasses and jewellery -- 38 of the 215 products here, and
    the ones most likely to be added twice, since nothing asks the shopper a
    size question for them.
    """

    _line(session, size=None)
    with pytest.raises(IntegrityError):
        _line(session, size=None)


def test_two_sizes_of_one_product_are_still_two_lines(session) -> None:
    """The invariant must not become "one line per product".

    A 6 and an 8 of one dress are two things the shopper owns.
    """

    _line(session, size="6")
    _line(session, size="8")
    assert session.query(CartItem).count() == 2


def test_the_same_product_for_two_shoppers_is_two_lines(session) -> None:
    _line(session, user_id=1, size="8")
    _line(session, user_id=2, size="8")
    assert session.query(CartItem).count() == 2


def test_the_migration_adds_the_index_to_a_database_that_predates_it() -> None:
    """The case that matters, and the one `create_all` hides.

    Every deployed database has `cart_items` already. Building the schema from
    today's model creates the indexes as a side effect, so a test that does
    that proves nothing about the migration -- removing migration 8 entirely
    left the other tests in this file green.

    So this one builds the table the way an older version left it, with no
    unique index, and asserts the migration puts one there.
    """

    from sqlalchemy import inspect, text

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE cart_items ("
            " id INTEGER PRIMARY KEY,"
            " cart_line_id VARCHAR,"
            " user_id INTEGER,"
            " product_id VARCHAR,"
            " item VARCHAR,"
            " size VARCHAR,"
            " amount INTEGER,"
            " price FLOAT)"
        )
        # A database in use, not an empty one.
        connection.exec_driver_sql(
            "INSERT INTO cart_items (user_id, product_id, item, size, amount)"
            " VALUES (1, 'p1', 'A Dress', '8', 1)"
        )

    before = {i["name"] for i in inspect(engine).get_indexes("cart_items")}
    assert "uq_cart_line_sized" not in before

    from memory_retriever.src.migrations import _cart_line_uniqueness

    with engine.begin() as connection:
        _cart_line_uniqueness(connection)

    after = {i["name"] for i in inspect(engine).get_indexes("cart_items")}
    assert {"uq_cart_line_sized", "uq_cart_line_unsized"} <= after

    # And it is enforced, on the row that was already there.
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO cart_items (user_id, product_id, item, size,"
                    " amount) VALUES (1, 'p1', 'A Dress', '8', 1)"
                )
            )
    engine.dispose()
