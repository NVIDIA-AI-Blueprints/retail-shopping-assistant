# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cart a shopper can touch, and the two ways it could quietly go wrong.

A browser holds an opaque ``cart_id``; the memory service keys carts on an
integer derived from it. If these routes accepted that integer, any caller
could read or mutate any cart, because the memory service has no
authentication. And an idempotency conflict reported as a server fault invites
a retry of a request that can only fail the same way.
"""

import importlib
import sys
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from shared.commerce_contracts import (
    Cart,
    CartLine,
    CartMutationResult,
    CommerceError,
    GetCartResult,
    Money,
)


@pytest.fixture
def cart_module(monkeypatch: pytest.MonkeyPatch, base_config) -> Iterator[Any]:
    from chain_server.src import config as config_mod
    from chain_server.src import deepagents_runtime as runtime_mod

    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: base_config)
    monkeypatch.setattr(runtime_mod, "DeepAgentsRuntime", lambda *_: object())

    sys.modules.pop("chain_server.src.main", None)
    module = importlib.import_module("chain_server.src.main")
    yield module
    sys.modules.pop("chain_server.src.main", None)


@pytest.fixture
def cart_client(cart_module) -> TestClient:
    return TestClient(cart_module.app)


def _cart(quantity: int = 1) -> Cart:
    return Cart(
        user_id="1",
        lines=[
            CartLine(
                cart_line_id="line-1",
                product_id="p-1",
                display_name="Black Satin Lace-Up Dress",
                quantity=quantity,
                size="2",
                unit_price=Money(amount=69.99),
            )
        ],
        subtotal=Money(amount=69.99 * quantity),
    )


def _ok_read(*_args, **_kwargs) -> GetCartResult:
    return GetCartResult(ok=True, cart=_cart())


class TestReadCart:
    def test_a_cart_line_carries_its_size_and_price(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        monkeypatch.setattr(cart_module, "get_cart", _ok_read)

        response = cart_client.get("/cart", params={"cart_id": "cart-abc"})

        assert response.status_code == 200
        body = response.json()
        assert body["lines"][0]["size"] == "2"
        assert body["lines"][0]["unit_price"] == 69.99
        assert body["lines"][0]["cart_line_id"] == "line-1"
        assert body["subtotal"] == pytest.approx(69.99)

    def test_the_opaque_cart_handle_is_hashed_server_side(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        """The browser must never be able to name another shopper's cart.

        The memory service authenticates nothing, so the integer owner id has
        to be derived here from the handle rather than accepted from a caller.
        """

        seen: dict = {}

        def capture(request, *_args, **_kwargs) -> GetCartResult:
            seen["user_id"] = request.user_id
            return GetCartResult(ok=True, cart=_cart())

        monkeypatch.setattr(cart_module, "get_cart", capture)

        cart_client.get("/cart", params={"cart_id": "cart-abc"})
        first = seen["user_id"]
        cart_client.get("/cart", params={"cart_id": "cart-xyz"})
        second = seen["user_id"]

        assert first.isdigit()
        assert first != "cart-abc"
        assert first != second

    def test_a_missing_cart_id_is_rejected(self, cart_client) -> None:
        assert cart_client.get("/cart").status_code == 422
        assert (
            cart_client.get("/cart", params={"cart_id": "   "}).status_code == 422
        )


class TestUpdateQuantity:
    def _patch(self, client, **body):
        payload = {"quantity": 2, "idempotency_key": "k-1"}
        payload.update(body)
        return client.patch(
            "/cart/lines/line-1", params={"cart_id": "cart-abc"}, json=payload
        )

    def test_a_successful_update_returns_the_whole_cart(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        """The mutation returns only the changed line; a panel needs all of it."""

        monkeypatch.setattr(
            cart_module,
            "update_cart_item",
            lambda *a, **k: CartMutationResult(ok=True, changed_line=None),
        )
        monkeypatch.setattr(cart_module, "get_cart", _ok_read)

        response = self._patch(cart_client)

        assert response.status_code == 200
        assert len(response.json()["lines"]) == 1

    def test_a_reused_key_for_a_different_quantity_is_a_conflict(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        """409, not 502.

        Reported as a server fault, a caller retries a request that can only
        fail the same way, and the shopper watches a quantity refuse to move
        with no way to tell why.
        """

        monkeypatch.setattr(
            cart_module,
            "update_cart_item",
            lambda *a, **k: CartMutationResult(
                ok=False,
                error=CommerceError(
                    code="cart_update_failed",
                    message="Cart quantity update was rejected.",
                    details={"status_code": 409},
                ),
            ),
        )

        response = self._patch(cart_client)

        assert response.status_code == 409
        assert "idempotency key" in response.json()["detail"]

    def test_an_unknown_line_is_not_found(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            cart_module,
            "update_cart_item",
            lambda *a, **k: CartMutationResult(
                ok=False,
                error=CommerceError(
                    code="cart_line_not_found", message="No such line."
                ),
            ),
        )

        assert self._patch(cart_client).status_code == 404

    def test_a_transport_failure_is_a_bad_gateway(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            cart_module,
            "update_cart_item",
            lambda *a, **k: CartMutationResult(
                ok=False,
                error=CommerceError(
                    code="cart_update_failed", message="Cart update failed."
                ),
            ),
        )

        assert self._patch(cart_client).status_code == 502

    def test_quantity_zero_is_allowed_and_a_negative_one_is_not(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        """Zero removes the line -- the same contract the agent's tool uses."""

        monkeypatch.setattr(
            cart_module,
            "update_cart_item",
            lambda *a, **k: CartMutationResult(ok=True, changed_line=None),
        )
        monkeypatch.setattr(
            cart_module,
            "get_cart",
            lambda *a, **k: GetCartResult(ok=True, cart=Cart(user_id="1", lines=[])),
        )

        assert self._patch(cart_client, quantity=0).status_code == 200
        assert self._patch(cart_client, quantity=-1).status_code == 422

    def test_an_idempotency_key_is_required(
        self, cart_module, cart_client, monkeypatch
    ) -> None:
        response = cart_client.patch(
            "/cart/lines/line-1",
            params={"cart_id": "cart-abc"},
            json={"quantity": 2},
        )

        assert response.status_code == 422
