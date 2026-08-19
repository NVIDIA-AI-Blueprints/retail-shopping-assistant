# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``memory_retriever.src.main``.

The service creates a module-level SQLite engine bound to ``./context.db``.
For tests we reconfigure it to an in-memory engine and rebuild the schema
against that engine. Every test gets a fresh database through the
``isolated_memory_db`` fixture so state never leaks between cases.
"""

from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from memory_retriever.src import main as memory_main


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def isolated_memory_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swap the module-level SQLite engine for a per-test in-memory one.

    We use ``StaticPool`` + the ``:memory:`` URL so all sessions opened
    during a single test share the same connection and therefore see the
    same data. At teardown the module's original globals are restored so
    subsequent tests see fresh tables.
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_local = sessionmaker(bind=test_engine)

    monkeypatch.setattr(memory_main, "engine", test_engine)
    monkeypatch.setattr(memory_main, "SessionLocal", test_session_local)

    memory_main.Base.metadata.create_all(bind=test_engine)

    yield

    memory_main.Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def client(isolated_memory_db) -> TestClient:
    return TestClient(memory_main.app)


_mutation_sequence = count()


def _add_cart(
    client: TestClient,
    user_id: int,
    item: str,
    amount: int,
    *,
    price: float | None = None,
    product_id: str | None = None,
) -> object:
    payload = {
        "product_id": product_id or f"test:{item}",
        "item": item,
        "amount": amount,
        "idempotency_key": f"test-add-{next(_mutation_sequence)}",
    }
    if price is not None:
        payload["price"] = price
    return client.post(f"/user/{user_id}/cart/add", json=payload)


def _remove_cart(
    client: TestClient,
    user_id: int,
    item: str,
    amount: int,
) -> object:
    matches = [
        line
        for line in client.get(f"/user/{user_id}/cart").json()["cart"]
        if line["item"] == item
    ]
    cart_line_id = matches[0]["cart_line_id"] if matches else "missing-line"
    return client.post(
        f"/user/{user_id}/cart/remove",
        json={
            "cart_line_id": cart_line_id,
            "amount": amount,
            "idempotency_key": f"test-remove-{next(_mutation_sequence)}",
        },
    )


# --------------------------------------------------------------------------->
# Health
# --------------------------------------------------------------------------->


class TestHealth:
    def test_health_returns_200_with_status(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body
        assert body["version"] == "1.0.0"


def test_database_sessions_return_connections_after_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    session_factory = sessionmaker(bind=test_engine)
    retained_sessions = []

    def retained_session():
        session = session_factory()
        retained_sessions.append(session)
        return session

    monkeypatch.setattr(memory_main, "engine", test_engine)
    monkeypatch.setattr(memory_main, "SessionLocal", retained_session)
    monkeypatch.setenv(
        "SHARED_CONFIG_ROOT",
        str(REPO_ROOT / "shared" / "configs"),
    )
    memory_main.Base.metadata.create_all(bind=test_engine)

    with TestClient(memory_main.app) as request_client:
        requests = [
            request_client.get("/user/1/context"),
            request_client.get("/user/1/cart"),
            request_client.post(
                "/user/1/context/replace",
                json={"new_context": "request scoped"},
            ),
            request_client.get("/user/404"),
        ]

    assert [response.status_code for response in requests] == [200, 200, 200, 404]
    assert test_engine.pool.checkedout() == 0

    for session in retained_sessions:
        session.close()
    test_engine.dispose()


# --------------------------------------------------------------------------->
# Cart endpoints
# --------------------------------------------------------------------------->


class TestCartFlows:
    def test_empty_cart_returns_empty_list(self, client: TestClient) -> None:
        response = client.get("/user/1/cart")
        assert response.status_code == 200
        assert response.json() == {"user_id": 1, "cart": []}

    def test_add_single_item(self, client: TestClient) -> None:
        response = _add_cart(client, 1, "Silk Dress", 1, price=49.99)
        assert response.status_code == 200
        assert "added 1" in response.json()["message"]

        cart_line = client.get("/user/1/cart").json()["cart"][0]
        assert len(cart_line["cart_line_id"]) == 32
        int(cart_line["cart_line_id"], 16)
        assert cart_line["item"] == "Silk Dress"
        assert cart_line["product_id"] == "test:Silk Dress"
        assert cart_line["amount"] == 1
        assert cart_line["price"] == 49.99

    def test_repeated_add_increments_existing_amount(self, client: TestClient) -> None:
        _add_cart(client, 1, "Silk Dress", 1, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart == [
            {
                "cart_line_id": cart_line_id,
                "product_id": "test:Silk Dress",
                "item": "Silk Dress",
                "amount": 3,
                "price": 49.99,
            }
        ]

    def test_duplicate_add_with_same_key_mutates_once(
        self,
        client: TestClient,
    ) -> None:
        payload = {
            "product_id": "prod_silk_dress",
            "item": "Silk Dress",
            "amount": 1,
            "price": 49.99,
            "idempotency_key": "add-silk-once",
        }

        first = client.post("/user/1/cart/add", json=payload)
        replay = client.post("/user/1/cart/add", json=payload)

        assert replay.json() == first.json()
        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["amount"] == 1

    def test_add_updates_price_when_newer_provided(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 1, price=49.99)
        _add_cart(client, 1, "Silk Dress", 1, price=69.99)

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["price"] == pytest.approx(69.99)
        assert cart[0]["amount"] == 2

    def test_add_without_price_keeps_existing_price(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 1, price=49.99)
        # Second call omits price; existing 49.99 should be preserved.
        _add_cart(client, 1, "Silk Dress", 1)

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["price"] == pytest.approx(49.99)

    def test_remove_reduces_amount(self, client: TestClient) -> None:
        _add_cart(client, 1, "Silk Dress", 3, price=49.99)
        response = _remove_cart(client, 1, "Silk Dress", 1)
        assert response.status_code == 200

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["amount"] == 2

    def test_remove_with_same_key_replays_original_mutation(
        self,
        client: TestClient,
    ) -> None:
        _add_cart(
            client,
            1,
            "Silk Dress",
            3,
            price=49.99,
            product_id="prod_silk_dress",
        )
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]
        payload = {
            "cart_line_id": cart_line_id,
            "amount": 1,
            "idempotency_key": "remove-silk-once",
        }

        first = client.post("/user/1/cart/remove", json=payload)
        replay = client.post("/user/1/cart/remove", json=payload)

        assert replay.json() == first.json()
        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["amount"] == 2

    def test_reused_add_key_for_different_product_is_rejected(
        self,
        client: TestClient,
    ) -> None:
        first = client.post(
            "/user/1/cart/add",
            json={
                "product_id": "prod_silk_dress",
                "item": "Silk Dress",
                "amount": 1,
                "idempotency_key": "shared-add-key",
            },
        )
        conflict = client.post(
            "/user/1/cart/add",
            json={
                "product_id": "prod_leather_bag",
                "item": "Leather Bag",
                "amount": 1,
                "idempotency_key": "shared-add-key",
            },
        )

        assert first.status_code == 200
        assert conflict.status_code == 409
        assert client.get("/user/1/cart").json()["cart"] == [
            first.json()["cart_line"]
        ]

    def test_remove_targets_opaque_line_id_not_display_name(
        self,
        client: TestClient,
    ) -> None:
        for product_id in ("prod_silk_one", "prod_silk_two"):
            client.post(
                "/user/1/cart/add",
                json={
                    "product_id": product_id,
                    "item": "Silk Dress",
                    "amount": 1,
                    "idempotency_key": f"add-{product_id}",
                },
            )
        cart = client.get("/user/1/cart").json()["cart"]

        response = client.post(
            "/user/1/cart/remove",
            json={
                "cart_line_id": cart[0]["cart_line_id"],
                "amount": 1,
                "idempotency_key": "remove-first-silk",
            },
        )

        assert response.status_code == 200
        assert client.get("/user/1/cart").json()["cart"] == [cart[1]]

    def test_remove_deletes_when_amount_exceeded(self, client: TestClient) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        _remove_cart(client, 1, "Silk Dress", 5)

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart == []

    def test_remove_unknown_line_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/user/1/cart/remove",
            json={
                "cart_line_id": "missing-line",
                "amount": 1,
                "idempotency_key": "remove-missing-line",
            },
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Item not in cart"

    def test_update_quantity_sets_absolute_amount(self, client: TestClient) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]

        response = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 4, "idempotency_key": "set-silk-to-four"},
        )

        assert response.status_code == 200
        assert response.json()["cart_line"] == {
            "cart_line_id": cart_line_id,
            "product_id": "test:Silk Dress",
            "item": "Silk Dress",
            "amount": 4,
            "price": 49.99,
        }
        duplicate = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 4, "idempotency_key": "set-silk-to-four"},
        )
        assert duplicate.status_code == 200
        assert duplicate.json() == response.json()
        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["cart_line_id"] == cart_line_id
        assert cart[0]["amount"] == 4

    def test_update_quantity_zero_deletes_line(self, client: TestClient) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]

        response = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 0, "idempotency_key": "delete-silk"},
        )

        assert response.status_code == 200
        assert response.json()["cart_line"]["amount"] == 0
        assert client.get("/user/1/cart").json()["cart"] == []

    def test_deleted_line_id_is_not_reused_and_zero_retry_is_safe(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        first_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]
        client.put(
            f"/user/1/cart/{first_line_id}/quantity",
            json={"quantity": 0, "idempotency_key": "delete-first-line"},
        )
        _add_cart(client, 1, "Leather Bag", 1, price=99.0)
        second_line = client.get("/user/1/cart").json()["cart"][0]

        assert second_line["cart_line_id"] != first_line_id
        retry = client.put(
            f"/user/1/cart/{first_line_id}/quantity",
            json={"quantity": 0, "idempotency_key": "delete-first-line"},
        )

        assert retry.status_code == 200
        assert retry.json()["cart_line"] == {
            "cart_line_id": first_line_id,
            "product_id": "test:Silk Dress",
            "item": "Silk Dress",
            "amount": 0,
            "price": 49.99,
        }
        assert client.get("/user/1/cart").json()["cart"] == [second_line]

        absent_retry = client.put(
            f"/user/1/cart/{first_line_id}/quantity",
            json={"quantity": 0, "idempotency_key": "confirm-first-absent"},
        )
        assert absent_retry.status_code == 404
        assert client.get("/user/1/cart").json()["cart"] == [second_line]

    def test_idempotency_key_conflict_does_not_mutate_cart(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]
        first = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 4, "idempotency_key": "shared-key"},
        )

        conflict = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 5, "idempotency_key": "shared-key"},
        )

        assert first.status_code == 200
        assert conflict.status_code == 409
        assert "different cart mutation" in conflict.json()["detail"]
        assert client.get("/user/1/cart").json()["cart"][0]["amount"] == 4

    def test_out_of_order_replay_does_not_revert_newer_quantity(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 1, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]
        first = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 2, "idempotency_key": "first-update"},
        )
        client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 3, "idempotency_key": "newer-update"},
        )

        replay = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 2, "idempotency_key": "first-update"},
        )

        assert replay.status_code == 200
        assert replay.json() == first.json()
        assert client.get("/user/1/cart").json()["cart"][0]["amount"] == 3

    def test_update_quantity_unknown_line_returns_404(
        self, client: TestClient
    ) -> None:
        response = client.put(
            "/user/1/cart/999/quantity",
            json={"quantity": 2, "idempotency_key": "missing-line"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Cart line not found"

    def test_update_quantity_cannot_cross_user_boundary(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]

        response = client.put(
            f"/user/2/cart/{cart_line_id}/quantity",
            json={"quantity": 4, "idempotency_key": "cross-user"},
        )

        assert response.status_code == 404
        assert client.get("/user/1/cart").json()["cart"][0]["amount"] == 2

    def test_update_quantity_rejects_negative_amount_without_mutation(
        self, client: TestClient
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        cart_line = client.get("/user/1/cart").json()["cart"][0]

        response = client.put(
            f"/user/1/cart/{cart_line['cart_line_id']}/quantity",
            json={"quantity": -1, "idempotency_key": "negative"},
        )

        assert response.status_code == 422
        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["amount"] == 2

    def test_update_quantity_rolls_back_when_commit_fails(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _add_cart(client, 1, "Silk Dress", 2, price=49.99)
        cart_line_id = client.get("/user/1/cart").json()["cart"][0][
            "cart_line_id"
        ]
        session_factory = memory_main.SessionLocal
        failing_session = session_factory()
        rollback_calls = []
        original_rollback = failing_session.rollback

        def fail_commit() -> None:
            raise RuntimeError("commit failed")

        def record_rollback() -> None:
            rollback_calls.append(True)
            original_rollback()

        monkeypatch.setattr(failing_session, "commit", fail_commit)
        monkeypatch.setattr(failing_session, "rollback", record_rollback)
        with monkeypatch.context() as scoped:
            scoped.setattr(memory_main, "SessionLocal", lambda: failing_session)
            with pytest.raises(RuntimeError, match="commit failed"):
                client.put(
                    f"/user/1/cart/{cart_line_id}/quantity",
                    json={
                        "quantity": 4,
                        "idempotency_key": "rollback-update",
                    },
                )

        assert rollback_calls == [True]
        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["amount"] == 2
        retry = client.put(
            f"/user/1/cart/{cart_line_id}/quantity",
            json={"quantity": 4, "idempotency_key": "rollback-update"},
        )
        assert retry.status_code == 200
        assert client.get("/user/1/cart").json()["cart"][0]["amount"] == 4

    @pytest.mark.parametrize("operation", ["add", "remove"])
    def test_add_and_remove_roll_back_with_their_replay_record(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
    ) -> None:
        if operation == "remove":
            _add_cart(client, 1, "Silk Dress", 1, price=49.99)
        before = client.get("/user/1/cart").json()["cart"]
        if operation == "add":
            path = "/user/1/cart/add"
            payload = {
                "product_id": "prod_silk_dress",
                "item": "Silk Dress",
                "amount": 1,
                "idempotency_key": "rollback-add",
            }
        else:
            path = "/user/1/cart/remove"
            payload = {
                "cart_line_id": before[0]["cart_line_id"],
                "amount": 1,
                "idempotency_key": "rollback-remove",
            }

        failing_session = memory_main.SessionLocal()
        rollback_calls = []
        original_rollback = failing_session.rollback

        def fail_commit() -> None:
            raise RuntimeError("commit failed")

        def record_rollback() -> None:
            rollback_calls.append(True)
            original_rollback()

        monkeypatch.setattr(failing_session, "commit", fail_commit)
        monkeypatch.setattr(failing_session, "rollback", record_rollback)
        with monkeypatch.context() as scoped:
            scoped.setattr(memory_main, "SessionLocal", lambda: failing_session)
            with pytest.raises(RuntimeError, match="commit failed"):
                client.post(path, json=payload)

        assert rollback_calls == [True]
        assert client.get("/user/1/cart").json()["cart"] == before
        with memory_main.SessionLocal() as db:
            assert db.query(memory_main.CartMutation).filter_by(
                user_id=1,
                idempotency_key=payload["idempotency_key"],
            ).first() is None

    def test_clear_cart_removes_all_items(self, client: TestClient) -> None:
        _add_cart(client, 1, "A", 1, price=10.0)
        _add_cart(client, 1, "B", 2, price=20.0)

        response = client.post("/user/1/cart/clear")
        assert response.status_code == 200

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart == []

    def test_clear_empty_cart_returns_404(self, client: TestClient) -> None:
        response = client.post("/user/999/cart/clear")
        assert response.status_code == 404

    def test_carts_are_partitioned_per_user(self, client: TestClient) -> None:
        _add_cart(client, 1, "A", 1, price=10.0)
        _add_cart(client, 2, "B", 1, price=20.0)

        assert client.get("/user/1/cart").json()["cart"][0]["item"] == "A"
        assert client.get("/user/2/cart").json()["cart"][0]["item"] == "B"

    def test_validation_error_on_missing_required_fields(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/user/1/cart/add",
            json={"amount": 1},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "path,payload",
        [
            ("/user/1/cart/add", {"item": "Silk Dress", "amount": 1}),
            ("/user/1/cart/remove", {"cart_line_id": "line-1", "amount": 1}),
        ],
    )
    def test_mutations_require_idempotency_keys(
        self,
        client: TestClient,
        path: str,
        payload: dict,
    ) -> None:
        assert client.post(path, json=payload).status_code == 422


def test_cart_line_id_migration_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with legacy_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE cart_items ("
                "id INTEGER PRIMARY KEY, user_id INTEGER, item TEXT, "
                "amount INTEGER, price REAL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO cart_items (id, user_id, item, amount, price) "
                "VALUES (1, 7, 'Silk Dress', 1, 49.99), "
                "(2, 7, 'Leather Bag', 1, 99.0)"
            )
        )
    monkeypatch.setattr(memory_main, "engine", legacy_engine)

    memory_main.Base.metadata.create_all(bind=legacy_engine)
    with legacy_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO cart_quantity_idempotency "
                "(idempotency_key, user_id, cart_line_id, quantity, response_body) "
                "VALUES ('legacy-update', 7, 'legacy-line', 2, "
                "'{\"user_id\": 7, \"message\": \"updated\"}')"
            )
        )
    memory_main._ensure_cart_line_id_column()
    memory_main._ensure_product_id_column()
    memory_main._migrate_quantity_idempotency()
    memory_main._migrate_quantity_idempotency()
    with legacy_engine.connect() as conn:
        first_ids = conn.execute(
            text("SELECT cart_line_id FROM cart_items ORDER BY id")
        ).scalars().all()
        indexes = conn.execute(text("PRAGMA index_list(cart_items)")).fetchall()
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table'")
        ).scalars().all()
        product_columns = conn.execute(
            text("PRAGMA table_info(cart_items)")
        ).fetchall()
        migrated_mutations = conn.execute(
            text(
                "SELECT user_id, idempotency_key, operation, stable_target_id "
                "FROM cart_mutations"
            )
        ).fetchall()

    memory_main._ensure_cart_line_id_column()
    with legacy_engine.connect() as conn:
        second_ids = conn.execute(
            text("SELECT cart_line_id FROM cart_items ORDER BY id")
        ).scalars().all()

    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    assert all(len(cart_line_id) == 32 for cart_line_id in first_ids)
    assert any(
        row[1] == "ix_cart_items_cart_line_id" and row[2] == 1
        for row in indexes
    )
    assert "cart_quantity_idempotency" in tables
    assert "cart_mutations" in tables
    assert any(column[1] == "product_id" for column in product_columns)
    assert migrated_mutations == [(7, "legacy-update", "update", "legacy-line")]
    legacy_engine.dispose()


# --------------------------------------------------------------------------->
# Context endpoints
# --------------------------------------------------------------------------->


class TestContextFlows:
    def test_context_empty_for_unknown_user(self, client: TestClient) -> None:
        response = client.get("/user/42/context")
        assert response.status_code == 200
        assert response.json() == {"user_id": 42, "context": ""}

    def test_add_context_creates_user(self, client: TestClient) -> None:
        response = client.post(
            "/user/1/context/add",
            json={"new_context": "hello"},
        )
        assert response.status_code == 200

        assert client.get("/user/1/context").json()["context"] == "hello"

    def test_add_context_appends_to_existing(self, client: TestClient) -> None:
        client.post("/user/1/context/add", json={"new_context": "first"})
        client.post("/user/1/context/add", json={"new_context": "second"})

        stored = client.get("/user/1/context").json()["context"]
        assert stored == "first second"

    def test_replace_context_overwrites_existing(
        self, client: TestClient
    ) -> None:
        client.post("/user/1/context/add", json={"new_context": "old"})
        client.post(
            "/user/1/context/replace",
            json={"new_context": "fresh"},
        )

        assert client.get("/user/1/context").json()["context"] == "fresh"

    def test_replace_context_creates_user_when_absent(
        self, client: TestClient
    ) -> None:
        client.post(
            "/user/99/context/replace",
            json={"new_context": "brand-new"},
        )
        assert (
            client.get("/user/99/context").json()["context"] == "brand-new"
        )

    def test_clear_context_deletes_user(self, client: TestClient) -> None:
        client.post("/user/1/context/add", json={"new_context": "sticky"})
        response = client.post("/user/1/context/clear")
        assert response.status_code == 200

        # After clear the user no longer exists: GET falls back to empty.
        assert client.get("/user/1/context").json() == {
            "user_id": 1,
            "context": "",
        }

    def test_clear_unknown_user_returns_404(self, client: TestClient) -> None:
        response = client.post("/user/404/context/clear")
        assert response.status_code == 404


# --------------------------------------------------------------------------->
# User-level endpoints
# --------------------------------------------------------------------------->


class TestUserEndpoints:
    def test_get_user_404_for_missing_user(self, client: TestClient) -> None:
        response = client.get("/user/7")
        assert response.status_code == 404

    def test_get_user_returns_context(self, client: TestClient) -> None:
        client.post("/user/7/context/add", json={"new_context": "hi"})

        response = client.get("/user/7")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == 7
        assert body["context"] == "hi"

    def test_clear_user_removes_record(self, client: TestClient) -> None:
        client.post("/user/1/context/add", json={"new_context": "will be gone"})
        response = client.post("/user/1/clear")
        assert response.status_code == 200

        # After clear the user no longer exists.
        assert client.get("/user/1").status_code == 404

    def test_clear_user_404_for_missing_user(self, client: TestClient) -> None:
        response = client.post("/user/1234/clear")
        assert response.status_code == 404


def test_the_cart_is_read_in_the_order_its_lines_were_added(
    client: TestClient,
) -> None:
    """A shopper's "that one" is answered from this order.

    Left unordered it was whatever the engine happened to return -- usually
    insertion order, and never promised to be. The assistant reads the cart
    back to decide which line "that one" means, so the order is part of the
    answer rather than a detail of storage.
    """

    for item in ("First Added", "Second Added", "Third Added"):
        assert _add_cart(client, 4242, item, 1).status_code == 200

    cart = client.get("/user/4242/cart").json()["cart"]

    assert [line["item"] for line in cart] == [
        "First Added",
        "Second Added",
        "Third Added",
    ]
