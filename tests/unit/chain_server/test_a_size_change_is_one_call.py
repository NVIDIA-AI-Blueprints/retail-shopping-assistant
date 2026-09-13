# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J07 t4, "can you change the heels to an 8".

A size is a separate cart line rather than a property of one, and the cart
service has no operation that changes it. That was once the model's problem to
solve: `update_cart_items_tool` refused a size and spelled out the sequence --
add the new size, confirm it, remove the old line. Neither reader got it right.
A turn that went straight to `add_cart_items_tool` never saw the refusal, so it
added the 8, said the cart now held both, and asked which to keep -- leaving a
pair the shopper had just replaced in their cart, with no CART_LINE_ID to
remove it by. A turn that did see the refusal had three calls to sequence while
the shopper waited.

The sequence needed nothing the model knows. These hold what it does instead:
one call, the cart it leaves behind, and the two failures that must not lose a
line.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from chain_server.src.agenttypes import Cart, State
from shared.commerce_contracts import (
    CartLine,
    CartMutationResult,
    CommerceError,
    GetProductDetailsResult,
    Money,
    ProductDetail,
)


def _text(result: Any) -> str:
    return result[0] if isinstance(result, tuple) else result


def _artifact(result: Any) -> dict[str, Any]:
    return result[1] if isinstance(result, tuple) else {}


def _heels(sizes: str = "6, 7, 8") -> ProductDetail:
    return ProductDetail(
        product_id="prod_heels",
        display_name="Jade Suede Heels",
        category="shoes",
        price=Money(amount=129.99),
        attributes={"sizes": sizes},
    )


def _line(size: str | None, quantity: int = 1) -> dict[str, Any]:
    line: dict[str, Any] = {
        "cart_line_id": "line_heels_7",
        "product_id": "prod_heels",
        "item": "Jade Suede Heels",
        "amount": quantity,
        "price": 129.99,
    }
    if size:
        line["size"] = size
    return line


class _Cart:
    """The cart as the shopper's lines, read back after every mutation.

    The tool reads the cart it changed rather than trusting the mutation
    result, so a stub that reports success without moving a line would pass
    tests the live cart fails.
    """

    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self.lines = lines

    def read(self, _user_id: int) -> Cart:
        return Cart(contents=[dict(line) for line in self.lines])

    def add(self, request, _port) -> CartMutationResult:
        self.lines.append(
            {
                "cart_line_id": f"line_new_{request.size}",
                "product_id": request.product_id,
                "item": request.display_name,
                "amount": request.quantity,
                "price": 129.99,
                **({"size": request.size} if request.size else {}),
            }
        )
        return CartMutationResult(ok=True)

    def remove(self, request, _port) -> CartMutationResult:
        self.lines = [
            line
            for line in self.lines
            if line["cart_line_id"] != request.cart_line_id
        ]
        return CartMutationResult(ok=True)


@pytest.fixture
def cart_tools(base_config, monkeypatch: pytest.MonkeyPatch):
    """The registered cart tools, with the cart and the catalog stubbed."""

    from chain_server.src import deepagents_runtime as runtime_mod
    from chain_server.src import turn_support as runtime_mod_support

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
    runtime._conversation_products = SimpleNamespace(resolve=lambda *_: None)
    identity = runtime_mod_support.RequestIdentity(
        session_id="session-a",
        conversation_id="conversation-a",
        cart_id="cart-a",
        context_user_id=111,
        cart_user_id=222,
        request_id="request-a",
    )

    cart = _Cart([_line("7")])
    catalog: dict[str, Any] = {
        "result": GetProductDetailsResult(ok=True, product=_heels())
    }
    monkeypatch.setattr(
        runtime_mod,
        "get_product_details",
        lambda *_a, **_k: catalog["result"],
    )
    monkeypatch.setattr(runtime_mod, "add_cart_item", cart.add)
    monkeypatch.setattr(runtime_mod, "remove_cart_item", cart.remove)
    monkeypatch.setattr(runtime, "_read_cart", cart.read)

    state = State(user_id=111, query="change the heels to an 8")
    state.cart = cart.read(222)
    runtime._create_agent(state, identity)

    tools = {fn.__name__: fn for fn in captured["tools"]}
    return SimpleNamespace(
        update=tools["update_cart_items_tool"],
        cart=cart,
        catalog=catalog,
        runtime_mod=runtime_mod,
        state=state,
    )


class TestASizeChangeIsOneCall:
    def test_the_line_moves_to_the_new_size(self, cart_tools) -> None:
        """One call, and the cart holds one pair of heels in an 8."""

        result = cart_tools.update(
            cart_line_id="line_heels_7", quantity=1, size="8"
        )

        sizes = [line.get("size") for line in cart_tools.cart.lines]
        assert sizes == ["8"]
        assert "CART SIZE CHANGED" in _text(result)
        assert "size 8" in _text(result)

    def test_the_add_runs_before_the_remove(self, cart_tools, monkeypatch) -> None:
        """Order is the whole safety property, so it is pinned rather than read.

        Removing first and failing the add would leave the shopper with
        nothing. Adding first and failing the remove leaves them one line too
        many, which is visible and recoverable.
        """

        order: list[str] = []
        add, remove = cart_tools.cart.add, cart_tools.cart.remove
        monkeypatch.setattr(
            cart_tools.runtime_mod,
            "add_cart_item",
            lambda *a, **k: (order.append("add"), add(*a, **k))[1],
        )
        monkeypatch.setattr(
            cart_tools.runtime_mod,
            "remove_cart_item",
            lambda *a, **k: (order.append("remove"), remove(*a, **k))[1],
        )

        cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="8")

        assert order == ["add", "remove"]

    def test_both_mutations_are_recorded_as_committed(self, cart_tools) -> None:
        """A turn that dies after this still has to say what the cart holds."""

        result = cart_tools.update(
            cart_line_id="line_heels_7", quantity=1, size="8"
        )

        effects = _artifact(result).get("committed_effects") or []
        assert [effect["operation"] for effect in effects] == [
            "added to cart",
            "removed from cart",
        ]

    def test_a_size_and_a_quantity_change_together(self, cart_tools) -> None:
        """"make those an 8, and two pairs" is one change to one line."""

        cart_tools.cart.lines = [_line("7", quantity=1)]
        cart_tools.state.cart = cart_tools.cart.read(222)

        cart_tools.update(cart_line_id="line_heels_7", quantity=2, size="8")

        assert [
            (line.get("size"), line["amount"]) for line in cart_tools.cart.lines
        ] == [("8", 2)]

    def test_the_whole_line_goes_with_it(self, cart_tools) -> None:
        """Two of a size, resized, is still two -- and the old line is gone."""

        cart_tools.cart.lines = [_line("7", quantity=3)]
        cart_tools.state.cart = cart_tools.cart.read(222)

        cart_tools.update(cart_line_id="line_heels_7", quantity=3, size="8")

        assert [line["amount"] for line in cart_tools.cart.lines] == [3]


class TestWhatIsStillRefused:
    def test_a_size_the_catalog_does_not_sell(self, cart_tools) -> None:
        """The same fact the add path checks, on the route a size arrives by.

        Skipping it would seat a size the shop does not sell by a route the
        add refuses -- and this is the route a shopper naming a size reaches.
        """

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="9")
        )

        assert result.startswith("CART_UPDATE_REFUSED")
        assert "6, 7, 8" in result
        assert [line.get("size") for line in cart_tools.cart.lines] == ["7"]

    def test_a_cart_line_id_this_cart_does_not_have(self, cart_tools) -> None:
        """A guessed id must not be answered with a new line."""

        result = _text(
            cart_tools.update(cart_line_id="line_guessed", quantity=1, size="8")
        )

        assert result.startswith("CART_UPDATE_REFUSED")
        assert "get_cart_tool" in result
        assert [line["cart_line_id"] for line in cart_tools.cart.lines] == [
            "line_heels_7"
        ]

    def test_a_quantity_of_zero_with_a_size(self, cart_tools) -> None:
        """Delete and replace are different intents, and both are available."""

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=0, size="8")
        )

        assert result.startswith("CART_UPDATE_REFUSED")
        assert "remove_cart_item_tool" in result
        assert [line.get("size") for line in cart_tools.cart.lines] == ["7"]

    def test_the_size_they_already_have_is_a_quantity_change(
        self, cart_tools, monkeypatch
    ) -> None:
        """Routing it through the add would put a second line on one size."""

        updates: list[Any] = []

        def fake_update(request, _port) -> CartMutationResult:
            updates.append(request)
            return CartMutationResult(
                ok=True,
                changed_line=CartLine(
                    cart_line_id=request.cart_line_id,
                    product_id="prod_heels",
                    display_name="Jade Suede Heels",
                    quantity=request.quantity,
                ),
            )

        monkeypatch.setattr(
            cart_tools.runtime_mod, "update_cart_item", fake_update
        )

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=2, size="7")
        )

        assert "CART UPDATED" in result
        assert [request.quantity for request in updates] == [2]
        assert len(cart_tools.cart.lines) == 1


class TestNothingIsLostWhenAStepFails:
    def test_a_failed_add_leaves_the_line_they_have(
        self, cart_tools, monkeypatch
    ) -> None:
        """Adding first is what makes this failure safe: nothing is half-done."""

        monkeypatch.setattr(
            cart_tools.runtime_mod,
            "add_cart_item",
            lambda *_a, **_k: CartMutationResult(
                ok=False,
                error=CommerceError(code="upstream", message="Cart is unavailable."),
            ),
        )

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="8")
        )

        assert result.startswith("CART_UPDATE_FAILED")
        assert "nothing was removed" in result
        assert [line.get("size") for line in cart_tools.cart.lines] == ["7"]

    def test_a_failed_remove_says_the_cart_holds_both(
        self, cart_tools, monkeypatch
    ) -> None:
        """The extra line is the shopper's to pay for, so the turn names it.

        Reporting a clean replacement here would be a lie the shopper finds at
        checkout, so the report carries the id that finishes the job instead.
        """

        monkeypatch.setattr(
            cart_tools.runtime_mod,
            "remove_cart_item",
            lambda *_a, **_k: CartMutationResult(
                ok=False,
                error=CommerceError(code="conflict", message="Line is locked."),
            ),
        )

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="8")
        )

        assert result.startswith("CART SIZE PARTIALLY CHANGED")
        assert "line_heels_7" in result
        assert "remove_cart_item_tool" in result
        assert sorted(
            line.get("size") for line in cart_tools.cart.lines
        ) == ["7", "8"]

    def test_a_catalog_that_cannot_be_read_changes_nothing(
        self, cart_tools
    ) -> None:
        """Whether the size is sold is unknown, not assumed either way."""

        cart_tools.catalog["result"] = GetProductDetailsResult(
            ok=False,
            error=CommerceError(code="timeout", message="Catalog timed out."),
        )

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="8")
        )

        assert result.startswith("CART_UPDATE_FAILED")
        assert [line.get("size") for line in cart_tools.cart.lines] == ["7"]


class TestAOneSizeLineHasNoSizeToChange:
    def test_a_one_size_line_is_refused_a_size(self, cart_tools) -> None:
        """J06 t9: "make those a 7" arrived with the tote bag's line id.

        The add path is permissive about a product the catalog states one size
        for, because refusing there would block a cart on missing data. A
        resize cannot be: a tote sold in one size has no second size to move
        to, and it became "size 7" while the heels the shopper meant stayed a
        6. Which line they meant is the model's to read, so the refusal sends
        it to the cart rather than guessing for it.
        """

        cart_tools.catalog["result"] = GetProductDetailsResult(
            ok=True,
            product=ProductDetail(
                product_id="prod_tote",
                display_name="Linen Canvas Tote Bag",
                category="bags",
                price=Money(amount=59.99),
                attributes={"sizes": "onesize"},
            ),
        )
        cart_tools.cart.lines = [_line(None)]
        cart_tools.state.cart = cart_tools.cart.read(222)

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="7")
        )

        assert result.startswith("CART_UPDATE_REFUSED")
        assert "sold in one size" in result
        assert "get_cart_tool" in result
        assert all(line.get("size") is None for line in cart_tools.cart.lines)

    def test_a_product_with_no_stated_sizes_is_refused_too(
        self, cart_tools
    ) -> None:
        """Nothing in this catalog is sizeless, so a blank run is a bad read."""

        cart_tools.catalog["result"] = GetProductDetailsResult(
            ok=True, product=_heels(sizes="")
        )

        result = _text(
            cart_tools.update(cart_line_id="line_heels_7", quantity=1, size="8")
        )

        assert result.startswith("CART_UPDATE_REFUSED")
        assert [line.get("size") for line in cart_tools.cart.lines] == ["7"]
