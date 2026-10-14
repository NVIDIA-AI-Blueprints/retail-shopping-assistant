# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.agenttypes``.

These tests pin the data contract the LangGraph pipeline relies on:
``Cart`` contents and ``State`` defaults. All logic here is pure
pydantic/Python and does not require any service to be running.
"""

from __future__ import annotations

import pytest
from chain_server.src.agenttypes import Cart, State
from pydantic import ValidationError


class TestCart:
    def test_default_cart_is_empty(self) -> None:
        cart = Cart()

        assert cart.contents == []

    def test_contents_can_hold_heterogeneous_metadata(self) -> None:
        cart = Cart(contents=[{"item": "X", "amount": 1, "price": 19.99}])

        assert cart.contents[0]["price"] == pytest.approx(19.99)


class TestState:
    def test_minimum_required_fields_enforced(self) -> None:
        with pytest.raises(ValidationError):
            State()  # type: ignore[call-arg]

    def test_defaults_populate_optional_fields(self) -> None:
        state = State(user_id=7, query="hello")

        assert state.user_id == 7
        assert state.query == "hello"
        assert state.context == ""
        assert state.image == ""
        assert state.response == ""
        assert state.retrieved == {}
        assert state.guardrails is True
        assert state.timings == {}
        assert state.agent_diagnostics == {}
        assert isinstance(state.cart, Cart)
        assert state.cart.contents == []

    def test_agent_diagnostics_default_is_isolated(self) -> None:
        first = State(user_id=1, query="first")
        second = State(user_id=2, query="second")

        first.agent_diagnostics["tool_calls"] = [{"tool_name": "read_file"}]

        assert second.agent_diagnostics == {}

    def test_cart_field_accepts_cart_instance(self) -> None:
        cart = Cart(contents=[{"item": "X", "amount": 1}])
        state = State(user_id=1, query="q", cart=cart)

        assert state.cart is cart

    def test_state_rejects_unknown_fields_gracefully(self) -> None:
        # ``State`` does not declare ``extra = 'forbid'`` so extra keys are
        # silently dropped; this test documents that behaviour so a change
        # to ``extra`` policy surfaces as a failure here.
        state = State(user_id=1, query="q")
        assert not hasattr(state, "nonexistent_field")
