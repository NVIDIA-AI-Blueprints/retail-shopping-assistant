# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The schema and its invariants, against a real PostgreSQL.

SQLite is permissive in ways that hide bugs until the day someone moves off it,
and three of these tests exist because it hid one:

  * ``user_id`` was ``Integer``. SQLite stores 64-bit integers whatever the
    declaration says, so the largest live shopper id -- 1.08e18, half a billion
    times PostgreSQL's ``INTEGER`` limit -- round-tripped fine and would have
    failed on the first insert after a migration.
  * ``conversation_turns`` was created before ``shopper_profiles``, which it has
    a foreign key to. SQLite does not check that a key's target exists until an
    insert; PostgreSQL checks at ``CREATE`` and refuses the table.
  * The zipcode check used ``GLOB``, which only SQLite has.

None of these can be caught against SQLite by definition, which is why this
module wants a real server. It is skipped when there is not one, so it is not a
test everybody has to have PostgreSQL to run -- but the skip is the point of
failure to watch, because a skipped test proves nothing.

    docker run -d --name pg -e POSTGRES_PASSWORD=test -e POSTGRES_USER=memory \\
        -e POSTGRES_DB=memory -p 127.0.0.1:55432:5432 postgres:16-alpine
    MEMORY_TEST_POSTGRES_URL=postgresql+psycopg://memory:test@127.0.0.1:55432/memory \\
        pytest tests/unit/memory_retriever/test_the_schema_works_on_postgres.py
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from memory_retriever.src.database import build_engine
from memory_retriever.src.migrations import (
    expected_schema_version,
    run_schema_migrations,
)
from memory_retriever.src.models import CartItem, ShopperProfile, new_cart_line_id


POSTGRES_URL = os.environ.get("MEMORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="MEMORY_TEST_POSTGRES_URL is not set; see this module's docstring",
)

#: The largest shopper id in the live database, and the reason for BigInteger.
#: PostgreSQL's INTEGER stops at 2,147,483,647.
LARGEST_LIVE_USER_ID = 1_077_059_276_034_060_207


@pytest.fixture
def engine() -> Iterator[object]:
    database_engine = build_engine(POSTGRES_URL, max_concurrent_requests=8)
    with database_engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    run_schema_migrations(database_engine)
    yield database_engine
    database_engine.dispose()


@pytest.fixture
def session(engine) -> Iterator[object]:
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


def test_every_migration_runs_on_a_fresh_postgres(engine) -> None:
    with engine.connect() as connection:
        applied = connection.execute(
            text("SELECT MAX(version) FROM schema_migrations")
        ).scalar()

    assert applied == expected_schema_version()


def test_the_conversation_tables_survive_their_foreign_key(engine) -> None:
    """Ordering that SQLite forgave and PostgreSQL does not.

    conversation_turns references shopper_profiles, so it cannot be created
    first. SQLite allows a key pointing at a table that does not exist yet and
    only complains on insert.
    """

    tables = set(inspect(engine).get_table_names())

    assert {"conversation_turns", "shopper_profiles"} <= tables


def test_a_real_shopper_id_does_not_overflow(session) -> None:
    session.add(
        CartItem(
            cart_line_id=new_cart_line_id(),
            user_id=LARGEST_LIVE_USER_ID,
            product_id="p1",
            item="Shoe",
            size="8",
            amount=1,
            price=59.99,
        )
    )
    session.commit()

    stored = session.query(CartItem).one()

    assert stored.user_id == LARGEST_LIVE_USER_ID


def test_a_price_keeps_its_cents(session) -> None:
    """REAL is 8 bytes in SQLite and 4 in PostgreSQL, where it rounds."""

    session.add(
        CartItem(
            cart_line_id=new_cart_line_id(),
            user_id=LARGEST_LIVE_USER_ID,
            product_id="p1",
            item="Shoe",
            amount=1,
            price=1234.56,
        )
    )
    session.commit()

    assert session.query(CartItem).one().price == 1234.56


def test_one_cart_line_per_product_and_size_holds_here_too(session) -> None:
    for _ in range(2):
        session.add(
            CartItem(
                cart_line_id=new_cart_line_id(),
                user_id=1,
                product_id="p1",
                item="Shoe",
                size="8",
                amount=1,
                price=1.0,
            )
        )

    with pytest.raises(IntegrityError):
        session.commit()


def test_two_sizes_of_one_product_are_still_two_lines(session) -> None:
    for size in ("8", "9"):
        session.add(
            CartItem(
                cart_line_id=new_cart_line_id(),
                user_id=1,
                product_id="p1",
                item="Shoe",
                size=size,
                amount=1,
                price=1.0,
            )
        )
    session.commit()

    assert session.query(CartItem).count() == 2


def test_the_unsized_index_is_partial_and_not_a_whole_table_rule(session) -> None:
    """Two sized lines of one product are fine; two unsized ones are not.

    A single unique index over (user_id, product_id) would have rejected the
    pair above. Both indexes are partial, and this is the difference.
    """

    session.add(
        CartItem(
            cart_line_id=new_cart_line_id(),
            user_id=1,
            product_id="p2",
            item="Belt",
            amount=1,
            price=1.0,
        )
    )
    session.commit()
    session.add(
        CartItem(
            cart_line_id=new_cart_line_id(),
            user_id=1,
            product_id="p2",
            item="Belt",
            amount=1,
            price=1.0,
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize("zipcode", ["1234x", "abcde", "1234"])
def test_a_zipcode_that_is_not_five_digits_is_refused(session, zipcode: str) -> None:
    """The check that used to be written with SQLite's GLOB.

    PostgreSQL has no GLOB and rejects the statement, so before this the table
    could not be created here at all.
    """

    session.add(
        ShopperProfile(
            shopper_profile_id="p",
            display_name="P",
            shopper_type="p",
            behavior="b",
            zipcode=zipcode,
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_an_over_long_zipcode_is_refused_by_the_column_itself(session) -> None:
    """A different rejection, from a different place, worth keeping separate.

    The check constraint never sees this one: PostgreSQL enforces VARCHAR(5) and
    raises DataError before any CHECK runs. SQLite ignores declared lengths
    entirely, so there the same value reaches the constraint and fails the
    length test instead. Both refuse it; only one of them is the constraint.
    """

    from sqlalchemy.exc import DataError

    session.add(
        ShopperProfile(
            shopper_profile_id="p",
            display_name="P",
            shopper_type="p",
            behavior="b",
            zipcode="123456",
        )
    )

    with pytest.raises(DataError):
        session.commit()


def test_a_five_digit_zipcode_is_accepted(session) -> None:
    session.add(
        ShopperProfile(
            shopper_profile_id="p",
            display_name="P",
            shopper_type="p",
            behavior="b",
            zipcode="94025",
        )
    )
    session.commit()

    assert session.query(ShopperProfile).one().zipcode == "94025"


def test_migrations_are_idempotent_here_as_well(engine) -> None:
    run_schema_migrations(engine)
    run_schema_migrations(engine)

    with engine.connect() as connection:
        versions = connection.execute(
            text("SELECT COUNT(*) FROM schema_migrations")
        ).scalar()

    assert versions == expected_schema_version()
