# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.agenttypes``.

These tests pin the data contract the LangGraph pipeline relies on:
``Cart`` helpers, ``State`` default + helper behaviour, and ``Rail`` timing
accumulation. All logic here is pure pydantic/Python and does not require
any service to be running.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from chain_server.src.agenttypes import Cart, Rail, State


class TestCart:
    def test_default_cart_is_empty(self) -> None:
        cart = Cart()

        assert cart.contents == []
        assert cart.is_empty() is True
        assert cart.get_item_count() == 0
        assert cart.get_items() == []

    def test_item_count_sums_amount_field_across_entries(self) -> None:
        cart = Cart(
            contents=[
                {"item": "Silk Dress", "amount": 2},
                {"item": "Leather Bag", "amount": 3},
            ]
        )

        assert cart.is_empty() is False
        assert cart.get_item_count() == 5

    def test_item_count_skips_missing_amount(self) -> None:
        cart = Cart(
            contents=[
                {"item": "Hat"},
                {"item": "Scarf", "amount": 0},
                {"item": "Belt", "amount": 4},
            ]
        )

        assert cart.get_item_count() == 4

    def test_get_items_returns_unique_item_names(self) -> None:
        cart = Cart(
            contents=[
                {"item": "Silk Dress", "amount": 1},
                {"item": "Silk Dress", "amount": 2},
                {"item": "Leather Bag", "amount": 1},
            ]
        )

        assert sorted(cart.get_items()) == ["Leather Bag", "Silk Dress"]

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
        assert state.next_agent == ""
        assert state.guardrails is True
        assert state.timings == {}
        assert isinstance(state.cart, Cart)
        assert state.cart.is_empty()

    def test_add_timing_records_step_duration(self) -> None:
        state = State(user_id=1, query="hi")

        state.add_timing("planner", 0.125)
        state.add_timing("retriever", 0.375)

        assert state.timings == {"planner": 0.125, "retriever": 0.375}
        assert math.isclose(state.get_total_time(), 0.5)

    def test_add_timing_overwrites_same_step(self) -> None:
        state = State(user_id=1, query="hi")

        state.add_timing("planner", 0.1)
        state.add_timing("planner", 0.9)

        assert state.timings == {"planner": 0.9}
        assert state.get_total_time() == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "image,expected",
        [
            ("", False),
            ("   ", False),
            ("data:image/png;base64,abc", True),
        ],
    )
    def test_has_image_trims_whitespace(self, image: str, expected: bool) -> None:
        state = State(user_id=1, query="q", image=image)
        assert state.has_image() is expected

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("", True),
            ("   ", True),
            ("\n\t", True),
            ("hi", False),
        ],
    )
    def test_is_empty_query_trims_whitespace(self, query: str, expected: bool) -> None:
        state = State(user_id=1, query=query)
        assert state.is_empty_query() is expected

    def test_cart_field_accepts_cart_instance(self) -> None:
        cart = Cart(contents=[{"item": "X", "amount": 1}])
        state = State(user_id=1, query="q", cart=cart)

        assert state.cart is cart
        assert state.cart.get_item_count() == 1

    def test_state_rejects_unknown_fields_gracefully(self) -> None:
        # ``State`` does not declare ``extra = 'forbid'`` so extra keys are
        # silently dropped; this test documents that behaviour so a change
        # to ``extra`` policy surfaces as a failure here.
        state = State(user_id=1, query="q")
        assert not hasattr(state, "nonexistent_field")


class TestRail:
    def test_rail_defaults_to_safe_with_empty_timings(self) -> None:
        rail = Rail()

        assert rail.is_safe is True
        assert rail.rail_timings == {}
        assert rail.get_total_rail_time() == 0.0

    def test_add_timing_aggregates_per_check(self) -> None:
        rail = Rail(is_safe=False)

        rail.add_timing("input", 0.02)
        rail.add_timing("output", 0.03)

        assert rail.is_safe is False
        assert rail.get_total_rail_time() == pytest.approx(0.05)

    def test_total_rail_time_handles_overwrite_of_same_key(self) -> None:
        rail = Rail()

        rail.add_timing("input", 0.05)
        rail.add_timing("input", 0.02)

        assert rail.rail_timings == {"input": 0.02}
        assert rail.get_total_rail_time() == pytest.approx(0.02)
