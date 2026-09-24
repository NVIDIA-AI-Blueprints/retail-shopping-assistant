# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An add is checked against the cart it would change.

J06, P08 and J04: "size 2 please" after the dress went in as a 2 wrote it again,
and the shopper had two. J06 t9: "make those a 7" was sent as an add, and the
heels were in the cart as a 6 and a 7. The skill says what to do in both; these
hold what the add tool does when the model does not.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from chain_server.src.agenttypes import Cart, State
from chain_server.src.conversation_products import (
    ConversationProductMatch,
    ProductReferenceResolution,
    ResolveConversationProductsResult,
)
from chain_server.src.runtime import identity as identity_mod
from shared.commerce_contracts import (
    CartMutationResult,
    GetProductDetailsResult,
    Money,
    ProductDetail,
    ProductSummary,
)


def _text(result: Any) -> str:
    return result[0] if isinstance(result, tuple) else result


def _effects(result: Any) -> list[dict[str, Any]]:
    artifact = result[1] if isinstance(result, tuple) else None
    return (artifact or {}).get("committed_effects") or []


_DRESS = dict(
    product_id="prod_dress",
    display_name="Black Satin Lace-Up Dress",
    category="apparel",
    price=Money(amount=69.99),
    attributes={"sizes": "2, 4, 6, 8"},
)


def _line(line_id: str, size: str | None, quantity: int = 1) -> dict[str, Any]:
    return {
        "cart_line_id": line_id,
        "product_id": "prod_dress",
        "item": "Black Satin Lace-Up Dress",
        "amount": quantity,
        "price": 69.99,
        **({"size": size} if size else {}),
    }


class _Cart:
    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self.lines = lines
        self.writes: list[Any] = []

    def read(self, _user_id: int) -> Cart:
        return Cart(contents=[dict(line) for line in self.lines])

    def add(self, request, _port) -> CartMutationResult:
        self.writes.append(request)
        self.lines.append(
            {
                "cart_line_id": f"line_new_{request.size}",
                "product_id": request.product_id,
                "item": request.display_name,
                "amount": request.quantity,
                "price": 69.99,
                **({"size": request.size} if request.size else {}),
            }
        )
        return CartMutationResult(ok=True)


@pytest.fixture
def add_tool(base_config, monkeypatch: pytest.MonkeyPatch):
    from chain_server.src import cart_operations as cart_ops_mod
    from chain_server.src.runtime import runtime as runtime_mod

    captured: dict[str, Any] = {}
    deepagents_mod = ModuleType("deepagents")
    tools_mod = ModuleType("langchain_core.tools")
    openai_mod = ModuleType("langchain_openai")

    class FakeProfile:
        def __init__(self, *args, **kwargs) -> None:
            pass

    def fake_tool(*, args_schema=None, return_direct: bool = False, **_kw):
        def decorate(fn):
            fn.args_schema = args_schema
            return fn

        return decorate

    deepagents_mod.GeneralPurposeSubagentProfile = FakeProfile
    deepagents_mod.HarnessProfile = FakeProfile
    deepagents_mod.create_deep_agent = lambda **kwargs: (
        captured.update(kwargs) or SimpleNamespace()
    )
    deepagents_mod.register_harness_profile = lambda *_: None
    tools_mod.tool = fake_tool
    openai_mod.ChatOpenAI = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    runtime._conversation_products = SimpleNamespace(
        resolve=lambda *_: ResolveConversationProductsResult(
            results=[
                ProductReferenceResolution(
                    reference_id="cart-add",
                    status="resolved",
                    match_count=1,
                    matches=[
                        ConversationProductMatch(
                            product=ProductSummary(**_DRESS),
                            turn_sequence=1,
                            position=1,
                            candidate_set_id="set-1",
                        )
                    ],
                )
            ]
        )
    )
    identity = identity_mod.RequestIdentity(
        session_id="session-a",
        conversation_id="conversation-a",
        cart_id="cart-a",
        context_user_id=111,
        cart_user_id=222,
        request_id="request-a",
    )
    cart = _Cart([])
    monkeypatch.setattr(
        cart_ops_mod,
        "get_product_details",
        lambda *_a, **_k: GetProductDetailsResult(
            ok=True, product=ProductDetail(**_DRESS)
        ),
    )
    monkeypatch.setattr(cart_ops_mod, "add_cart_item", cart.add)
    monkeypatch.setattr(runtime, "_read_cart", cart.read)

    def start(query: str, lines: list[dict[str, Any]]):
        cart.lines = lines
        state = State(user_id=111, query=query)
        state.cart = cart.read(222)
        runtime._create_agent(state, identity)
        return {fn.__name__: fn for fn in captured["tools"]}["add_cart_items_tool"]

    return SimpleNamespace(start=start, cart=cart)


def _add(size: str, quantity: int = 1) -> list[dict[str, Any]]:
    return [{"product_ref": "prod_dress", "size": size, "quantity": quantity}]


def test_a_size_the_cart_already_holds_is_not_added_again(add_tool) -> None:
    add = add_tool.start("size 2 please", [_line("line_dress_2", "2")])

    result = add(items=_add("2"))

    assert add_tool.cart.writes == []
    assert [line["amount"] for line in add_tool.cart.lines] == [1]
    assert _effects(result) == []
    assert "already in the cart" in _text(result)
    assert "line_dress_2" in _text(result)


def test_more_of_a_line_is_pointed_at_the_quantity(add_tool) -> None:
    """"Add another one" still has a route: the total, on the line it names."""

    add = add_tool.start("add another one in a 2", [_line("line_dress_2", "2")])

    text = _text(add(items=_add("2", quantity=1)))

    assert "update_cart_items_tool" in text
    assert add_tool.cart.writes == []


def test_another_size_is_added_and_the_old_line_named(add_tool) -> None:
    """"Add it in a 4 as well" and "make those a 4" both arrive as this add."""

    add = add_tool.start("make it a 4", [_line("line_dress_2", "2")])

    result = add(items=_add("4"))

    assert [write.size for write in add_tool.cart.writes] == ["4"]
    assert sorted(line["size"] for line in add_tool.cart.lines) == ["2", "4"]
    text = _text(result)
    assert "already held size 2 (CART_LINE_ID line_dress_2)" in text
    assert "remove_cart_item_tool" in text


def test_an_add_to_a_cart_without_it_says_nothing_extra(add_tool) -> None:
    add = add_tool.start("add it in a 4", [])

    text = _text(add(items=_add("4")))

    assert [write.size for write in add_tool.cart.writes] == ["4"]
    assert "already" not in text
