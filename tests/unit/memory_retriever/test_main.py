# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``memory_retriever.src.main``.

The service creates a module-level SQLite engine bound to ``./context.db``.
For tests we reconfigure it to an in-memory engine and rebuild the schema
against that engine. Every test gets a fresh database through the
``isolated_memory_db`` fixture so state never leaks between cases.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from memory_retriever.src import main as memory_main


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


# --------------------------------------------------------------------------->
# Cart endpoints
# --------------------------------------------------------------------------->


class TestCartFlows:
    def test_empty_cart_returns_empty_list(self, client: TestClient) -> None:
        response = client.get("/user/1/cart")
        assert response.status_code == 200
        assert response.json() == {"user_id": 1, "cart": []}

    def test_add_single_item(self, client: TestClient) -> None:
        response = client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 1, "price": 49.99},
        )
        assert response.status_code == 200
        assert "added 1" in response.json()["message"]

        listed = client.get("/user/1/cart").json()
        assert listed["cart"] == [
            {"item": "Silk Dress", "amount": 1, "price": 49.99}
        ]

    def test_repeated_add_increments_existing_amount(self, client: TestClient) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 1, "price": 49.99},
        )
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 2, "price": 49.99},
        )

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart == [
            {"item": "Silk Dress", "amount": 3, "price": 49.99}
        ]

    def test_add_updates_price_when_newer_provided(
        self, client: TestClient
    ) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 1, "price": 49.99},
        )
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 1, "price": 69.99},
        )

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["price"] == pytest.approx(69.99)
        assert cart[0]["amount"] == 2

    def test_add_without_price_keeps_existing_price(
        self, client: TestClient
    ) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 1, "price": 49.99},
        )
        # Second call omits price; existing 49.99 should be preserved.
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 1},
        )

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["price"] == pytest.approx(49.99)

    def test_remove_reduces_amount(self, client: TestClient) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 3, "price": 49.99},
        )
        response = client.post(
            "/user/1/cart/remove",
            json={"item": "Silk Dress", "amount": 1},
        )
        assert response.status_code == 200

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart[0]["amount"] == 2

    def test_remove_deletes_when_amount_exceeded(self, client: TestClient) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "Silk Dress", "amount": 2, "price": 49.99},
        )
        client.post(
            "/user/1/cart/remove",
            json={"item": "Silk Dress", "amount": 5},
        )

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart == []

    def test_remove_unknown_item_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/user/1/cart/remove",
            json={"item": "Ghost", "amount": 1},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Item not in cart"

    def test_clear_cart_removes_all_items(self, client: TestClient) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "A", "amount": 1, "price": 10.0},
        )
        client.post(
            "/user/1/cart/add",
            json={"item": "B", "amount": 2, "price": 20.0},
        )

        response = client.post("/user/1/cart/clear")
        assert response.status_code == 200

        cart = client.get("/user/1/cart").json()["cart"]
        assert cart == []

    def test_clear_empty_cart_returns_404(self, client: TestClient) -> None:
        response = client.post("/user/999/cart/clear")
        assert response.status_code == 404

    def test_carts_are_partitioned_per_user(self, client: TestClient) -> None:
        client.post(
            "/user/1/cart/add",
            json={"item": "A", "amount": 1, "price": 10.0},
        )
        client.post(
            "/user/2/cart/add",
            json={"item": "B", "amount": 1, "price": 20.0},
        )

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
