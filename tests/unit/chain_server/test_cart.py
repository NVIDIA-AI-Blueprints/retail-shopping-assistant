# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.cart``.

The cart agent mixes pure helper logic (price extraction, name resolution,
pronoun handling) with HTTP/LLM I/O. This suite exhaustively exercises the
helpers and then covers the ``invoke`` branches by stubbing the OpenAI
client and the catalog/memory HTTP calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from chain_server.src import cart as cart_mod
from chain_server.src.agenttypes import Cart, State
from chain_server.src.cart import (
    CartAgent,
    _extract_price,
    _normalize_name,
    _resolve_catalog_match,
)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestExtractPrice:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Name | desc | cat\nPRICE: 29.99", 29.99),
            ("PRICE: 0", 0.0),
            ("price: 45.9", 45.9),  # case-insensitive
            ("noise PRICE: 100 more noise", 100.0),
        ],
    )
    def test_valid_prices_extracted(self, text: str, expected: float) -> None:
        assert _extract_price(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "no price here",
            "PRICE: free",
            "price tag 10",  # missing colon
        ],
    )
    def test_missing_or_malformed_returns_none(self, text: Any) -> None:
        assert _extract_price(text) is None


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,normalized",
        [
            ("", ""),
            ("Silk Dress", "silk dress"),
            ("  Silk   Dress  ", "silk dress"),
            ("Alpine-Waterproof Boot!", "alpine waterproof boot"),
            ("ITEM!!!", "item"),
        ],
    )
    def test_normalization_rules(self, raw: str, normalized: str) -> None:
        assert _normalize_name(raw) == normalized


class TestResolveCatalogMatch:
    def test_returns_none_for_empty_catalog(self) -> None:
        assert _resolve_catalog_match("Silk Dress", [], []) is None

    def test_exact_name_match_wins_over_similarity(self) -> None:
        # Exact match should be picked even if another row has higher similarity.
        names = ["Alpine Hiking Boot", "Silk Dress"]
        sims = [0.9, 0.4]
        assert _resolve_catalog_match("Silk Dress", names, sims) == 1

    def test_substring_match_picks_candidate(self) -> None:
        names = ["Deluxe Leather Handbag"]
        assert _resolve_catalog_match("Leather Handbag", names, [0.1]) == 0

    def test_jaccard_overlap_triggers_fallback(self) -> None:
        # No exact/substring match but strong token overlap.
        names = ["Bamboo Slim Fit Chinos", "Cotton Cardigan"]
        sims = [0.2, 0.9]
        assert _resolve_catalog_match("Bamboo Chinos", names, sims) == 0

    def test_low_overlap_falls_back_to_similarity_when_above_threshold(self) -> None:
        names = ["Completely Unrelated Widget"]
        sims = [0.7]
        assert _resolve_catalog_match(
            "totally different query", names, sims, min_similarity=0.5
        ) == 0

    def test_low_overlap_and_low_similarity_returns_none(self) -> None:
        names = ["Completely Unrelated Widget"]
        sims = [0.1]
        assert _resolve_catalog_match("totally different", names, sims) is None

    def test_empty_query_falls_back_to_top_similarity(self) -> None:
        names = ["Hat", "Scarf"]
        sims = [0.8, 0.7]
        assert _resolve_catalog_match("", names, sims) == 0

    def test_empty_query_returns_none_when_top_below_threshold(self) -> None:
        assert (
            _resolve_catalog_match("", ["Hat"], [0.1], min_similarity=0.5) is None
        )


# ---------------------------------------------------------------------------
# Name-scan helpers on CartAgent
# ---------------------------------------------------------------------------


class TestLooksLikeProductName:
    @pytest.mark.parametrize(
        "value",
        [
            "Silk Dress",
            "Alpine-Waterproof Hiking Boot",
            "O'Henry Scarf",
            "Slim-Fit Chinos",
        ],
    )
    def test_accepts_valid_product_names(self, value: str) -> None:
        assert CartAgent._looks_like_product_name(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "Price: $69.99",
            "Tip:",
            "single",
            "$45.00",
            "99 bottles",  # starts with a digit
        ],
    )
    def test_rejects_non_product_spans(self, value: str) -> None:
        assert CartAgent._looks_like_product_name(value) is False


class TestCollectKnownProducts:
    def test_collects_from_cart_and_context_sources(self) -> None:
        state = State(
            user_id=1,
            query="hi",
            cart=Cart(
                contents=[
                    {"item": "Silk Dress", "amount": 1},
                    {"item": "Leather Bag", "amount": 2},
                ]
            ),
            context=(
                "These are the products:\n"
                "Alpine Hiking Boot | desc | shoes\n"
                "**Pearl Drop Stud Earrings**\n"
                "**Price: $69.99**\n"  # bold span that is NOT a product
            ),
        )

        known = CartAgent._collect_known_products(state)

        assert "Silk Dress" in known
        assert "Leather Bag" in known
        assert "Alpine Hiking Boot" in known
        assert "Pearl Drop Stud Earrings" in known
        # bold span matching a non-product shape must be filtered out.
        assert not any(n.lower().startswith("price:") for n in known)

    def test_dedupes_case_insensitively_preserving_first_casing(self) -> None:
        state = State(
            user_id=1,
            query="hi",
            cart=Cart(contents=[{"item": "Silk Dress", "amount": 1}]),
            context="**silk dress**\n**SILK DRESS**",
        )

        known = CartAgent._collect_known_products(state)

        # Cart is trusted source, and since it's inserted first its casing
        # wins against later context mentions.
        assert "Silk Dress" in known
        assert len([n for n in known if n.lower() == "silk dress"]) == 1

    def test_empty_state_returns_empty(self) -> None:
        state = State(user_id=1, query="hi")
        assert CartAgent._collect_known_products(state) == []


class TestIsPronounReference:
    @pytest.mark.parametrize(
        "q",
        ["add it", "take this", "buy that", "pair those", "grab both"],
    )
    def test_matches_supported_pronouns(self, q: str) -> None:
        assert CartAgent._is_pronoun_reference(q) is True

    @pytest.mark.parametrize("q", ["add the silk dress", "", "remove everything"])
    def test_no_pronoun_returns_false(self, q: str) -> None:
        assert CartAgent._is_pronoun_reference(q) is False


class TestFindNamedProduct:
    def test_exact_contained_name_wins(self) -> None:
        assert (
            CartAgent._find_named_product(
                "please add the silk dress to my cart", ["Silk Dress", "Hiking Boot"]
            )
            == "Silk Dress"
        )

    def test_returns_best_candidate_above_threshold(self) -> None:
        result = CartAgent._find_named_product(
            "please add the bamboo chinos",
            ["Bamboo Slim Fit Chinos", "Cotton Cardigan"],
        )
        assert result == "Bamboo Slim Fit Chinos"

    def test_ambiguous_score_tie_returns_none(self) -> None:
        # Two candidates share an identical single-word score: the anti-
        # ambiguity guard must abstain rather than silently pick one.
        result = CartAgent._find_named_product(
            "add the dress", ["Silk Dress", "Summer Dress"]
        )
        assert result is None

    def test_no_shared_tokens_returns_none(self) -> None:
        assert (
            CartAgent._find_named_product(
                "add the hat", ["Silk Dress", "Leather Bag"]
            )
            is None
        )

    def test_empty_inputs_return_none(self) -> None:
        assert CartAgent._find_named_product("", ["Silk Dress"]) is None
        assert CartAgent._find_named_product("add to cart", []) is None


class TestLastMentionedProduct:
    def test_returns_name_with_rightmost_occurrence(self) -> None:
        context = (
            "Earlier we discussed **Silk Dress**.\n"
            "Later I recommended **Leather Bag** for you."
        )
        result = CartAgent._last_mentioned_product(
            ["Silk Dress", "Leather Bag"], context
        )
        assert result == "Leather Bag"

    def test_none_when_no_context(self) -> None:
        assert (
            CartAgent._last_mentioned_product(["Silk Dress"], "") is None
        )

    def test_none_when_no_known(self) -> None:
        assert CartAgent._last_mentioned_product([], "text") is None


class TestExtractRecentDiscussion:
    def test_returns_placeholder_when_empty(self) -> None:
        assert CartAgent._extract_recent_discussion("") == "(no prior discussion)"

    def test_short_context_returned_verbatim(self) -> None:
        ctx = "A short context."
        assert CartAgent._extract_recent_discussion(ctx) == ctx

    def test_long_context_truncated_with_prefix(self) -> None:
        ctx = "a\n" + "x" * 5000
        out = CartAgent._extract_recent_discussion(ctx, max_chars=2000)

        assert out.startswith("...\n") or len(out) <= 2000 + len("...\n")
        assert len(out) <= 2000 + len("...\n")


# ---------------------------------------------------------------------------
# ResolveTargetItem integration (pure logic, uses State only)
# ---------------------------------------------------------------------------


@pytest.fixture
def cart_agent(base_config, monkeypatch: pytest.MonkeyPatch) -> CartAgent:
    """Build a CartAgent without constructing a real OpenAI client.

    ``CartAgent.__init__`` eagerly instantiates ``OpenAI(...)``, which is
    expensive to mock per test. We replace it with a dummy class so every
    test gets a fresh agent object with predictable attributes.
    """

    class _FakeOpenAI:
        def __init__(self, *_, **__) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None))

    monkeypatch.setattr(cart_mod, "OpenAI", _FakeOpenAI)
    return CartAgent(config=base_config)


class TestResolveTargetItem:
    def test_returns_named_product_when_query_contains_name(
        self, cart_agent: CartAgent
    ) -> None:
        state = State(
            user_id=1,
            query="please add the silk dress to my cart",
            cart=Cart(contents=[{"item": "Leather Bag", "amount": 1}]),
            context="**Silk Dress**",
        )
        assert cart_agent._resolve_target_item(state) == "Silk Dress"

    def test_returns_last_mentioned_for_pronoun_only_query(
        self, cart_agent: CartAgent
    ) -> None:
        state = State(
            user_id=1,
            query="add it to my cart",
            cart=Cart(contents=[]),
            context=(
                "Earlier: **Silk Dress**.\n"
                "Later: **Leather Bag** was shown."
            ),
        )
        assert cart_agent._resolve_target_item(state) == "Leather Bag"

    def test_no_known_products_returns_none(self, cart_agent: CartAgent) -> None:
        state = State(user_id=1, query="add it to my cart")
        assert cart_agent._resolve_target_item(state) is None

    def test_query_without_pronoun_or_name_returns_none(
        self, cart_agent: CartAgent
    ) -> None:
        state = State(
            user_id=1,
            query="add something nice",
            context="**Silk Dress**",
        )
        assert cart_agent._resolve_target_item(state) is None


# ---------------------------------------------------------------------------
# HTTP + LLM integration paths (requests are stubbed)
# ---------------------------------------------------------------------------


class _Recorder:
    """Collect outgoing HTTP calls so assertions can inspect them."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, Dict[str, Any]]] = []


def _install_http_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    catalog_response: Dict[str, Any] | None = None,
    cart_before: List[Dict[str, Any]] | None = None,
    cart_after: List[Dict[str, Any]] | None = None,
    add_message: str = "added 1 Silk Dress",
    remove_message: str = "removed 1 Silk Dress",
    add_status: int = 200,
    remove_status: int = 200,
) -> _Recorder:
    """Install stubs for the ``requests`` surface used by CartAgent.

    The two surfaces are:

    * ``requests.Session().post`` to the catalog retriever for name lookups.
    * ``requests.get`` / ``requests.post`` to the memory service for cart CRUD.

    Returning a recorder lets tests assert on the payload content.
    """
    recorder = _Recorder()

    catalog_response = catalog_response or {
        "names": ["Silk Dress"],
        "similarities": [0.95],
        "texts": ["Silk Dress | lovely dress | dress\nPRICE: 49.99"],
    }

    class _FakeSession:
        def __init__(self) -> None:
            self.mounts: List[str] = []

        def mount(self, prefix: str, _adapter: Any) -> None:
            self.mounts.append(prefix)

        def post(self, url: str, json: Dict[str, Any]):
            recorder.calls.append(("POST", url, json))
            return cart_mod.requests.Response  # unused; replaced below

    class _FakeResponse:
        def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
            self._payload = payload
            self.status_code = status

        def json(self) -> Dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise cart_mod.requests.exceptions.HTTPError(
                    f"HTTP {self.status_code}"
                )

        @property
        def text(self) -> str:
            return json.dumps(self._payload)

    def _fake_session_ctor() -> _FakeSession:
        return _FakeSession()

    def _fake_session_post(self: _FakeSession, url: str, json: Dict[str, Any]):
        recorder.calls.append(("POST", url, json))
        # The catalog retriever's /query/text is the only session.post target.
        return _FakeResponse(catalog_response)

    def _fake_requests_get(url: str, timeout: int = 10):
        recorder.calls.append(("GET", url, {}))
        cart_state = cart_before if cart_before is not None else []
        return _FakeResponse({"cart": cart_state})

    def _fake_requests_post(url: str, json: Dict[str, Any], timeout: int = 10):
        recorder.calls.append(("POST", url, json))
        # Which memory endpoint is being invoked?
        if url.endswith("/cart/add"):
            return _FakeResponse({"message": add_message}, status=add_status)
        if url.endswith("/cart/remove"):
            return _FakeResponse({"message": remove_message}, status=remove_status)
        return _FakeResponse({"status": "ok"})

    # Install the session stub. We overwrite the two methods individually
    # to avoid re-implementing the full Session protocol.
    monkeypatch.setattr(cart_mod.requests, "Session", _fake_session_ctor)
    monkeypatch.setattr(_FakeSession, "post", _fake_session_post)

    monkeypatch.setattr(cart_mod.requests, "get", _fake_requests_get)
    monkeypatch.setattr(cart_mod.requests, "post", _fake_requests_post)

    if cart_after is not None:
        # For flows where the cart is refreshed after mutation, overlay the
        # "after" state on the single GET path. Tests that care about state
        # toggling between GETs swap in their own implementation via
        # ``monkeypatch``.
        def _fake_requests_get_after(url: str, timeout: int = 10):  # noqa: ARG001
            recorder.calls.append(("GET", url, {}))
            return _FakeResponse({"cart": cart_after})

        monkeypatch.setattr(cart_mod.requests, "get", _fake_requests_get_after)

    return recorder


class TestLookupInCatalog:
    def test_returns_match_dict_on_success(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(monkeypatch)

        match = cart_agent._lookup_in_catalog("Silk Dress")

        assert match is not None
        assert match["name"] == "Silk Dress"
        assert match["similarity"] == pytest.approx(0.95)
        assert "PRICE: 49.99" in match["text"]

    def test_returns_none_when_no_plausible_match(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            catalog_response={"names": ["Unrelated Widget"], "similarities": [0.05], "texts": ["x"]},
        )

        assert cart_agent._lookup_in_catalog("Silk Dress") is None


class TestAddRemoveCart:
    def test_add_calls_memory_with_price_when_available(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_http_stubs(
            monkeypatch,
            add_message="added 2 Silk Dress",
        )

        msg = cart_agent._add_to_cart(user_id=42, item_name="Silk Dress", quantity=2)

        assert msg == "added 2 Silk Dress"
        add_calls = [c for c in recorder.calls if c[1].endswith("/cart/add")]
        assert len(add_calls) == 1
        _, url, payload = add_calls[0]
        assert url.endswith("/user/42/cart/add")
        assert payload == {"item": "Silk Dress", "amount": 2, "price": 49.99}

    def test_add_reports_catalog_miss(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            catalog_response={"names": ["Other"], "similarities": [0.01], "texts": ["x"]},
        )

        msg = cart_agent._add_to_cart(user_id=1, item_name="Unknown Item", quantity=1)
        assert "No such item" in msg

    def test_add_surfaces_memory_failure(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(monkeypatch, add_status=500)

        msg = cart_agent._add_to_cart(user_id=1, item_name="Silk Dress", quantity=3)
        assert msg == "Failed to add 3 Silk Dress to cart."

    def test_remove_passes_item_name_and_amount(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_http_stubs(
            monkeypatch,
            remove_message="removed 1 Silk Dress",
        )

        msg = cart_agent._remove_from_cart(user_id=7, item_name="Silk Dress", quantity=1)

        assert msg == "removed 1 Silk Dress"
        remove_calls = [c for c in recorder.calls if c[1].endswith("/cart/remove")]
        _, url, payload = remove_calls[0]
        assert url.endswith("/user/7/cart/remove")
        assert payload == {"item": "Silk Dress", "amount": 1}

    def test_remove_reports_catalog_miss(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            catalog_response={"names": ["Other"], "similarities": [0.01], "texts": [""]},
        )

        msg = cart_agent._remove_from_cart(user_id=1, item_name="Unknown", quantity=2)
        assert "No such item" in msg


class TestViewCartTotal:
    def test_empty_cart_reports_zero(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(monkeypatch, cart_before=[])

        out = cart_agent._view_cart_total(user_id=1)
        assert "$0.00" in out
        assert "empty" in out.lower()

    def test_totals_and_line_items_rendered(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            cart_before=[
                {"item": "Silk Dress", "amount": 2, "price": 49.99},
                {"item": "Leather Bag", "amount": 1, "price": 199.00},
            ],
        )

        out = cart_agent._view_cart_total(user_id=1)

        assert "2 x Silk Dress @ $49.99 = $99.98" in out
        assert "1 x Leather Bag @ $199.00 = $199.00" in out
        assert "Cart total: $298.98" in out

    def test_missing_price_line_excluded_from_total(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            cart_before=[
                {"item": "Silk Dress", "amount": 1, "price": 49.99},
                {"item": "Mystery Item", "amount": 2, "price": None},
            ],
        )

        out = cart_agent._view_cart_total(user_id=1)

        assert "1 x Silk Dress @ $49.99 = $49.99" in out
        assert "2 x Mystery Item: price unavailable" in out
        assert "Cart total: $49.99" in out
        assert "Mystery Item" in out  # listed in missing-price note
        assert "Re-add them" in out


# ---------------------------------------------------------------------------
# invoke() dispatch
# ---------------------------------------------------------------------------


def _install_llm_tool_response(
    cart_agent: CartAgent, tool_name: str | None, arguments: Dict[str, Any] | None, *, fallback_content: str = ""
) -> None:
    """Replace the agent's OpenAI client with a tool-call-emitting stub."""
    message = SimpleNamespace(
        content=fallback_content,
        tool_calls=None
        if tool_name is None
        else [
            SimpleNamespace(
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=json.dumps(arguments or {}),
                )
            )
        ],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    cart_agent.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )


class TestCartAgentInvoke:
    def test_add_to_cart_path_updates_state(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            cart_before=[{"item": "Silk Dress", "amount": 1, "price": 49.99}],
            add_message="added 1 Silk Dress",
        )
        _install_llm_tool_response(
            cart_agent, "add_to_cart", {"item_name": "Silk Dress", "quantity": 1}
        )
        state = State(user_id=42, query="add silk dress")

        result = cart_agent.invoke(state, verbose=False)

        assert "added 1 Silk Dress" in result.response
        assert result.cart.get_item_count() == 1
        assert "cart" in result.timings

    def test_view_cart_empty_returns_empty_message(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(monkeypatch, cart_before=[])
        _install_llm_tool_response(cart_agent, "view_cart", {})
        state = State(user_id=1, query="what is in my cart")

        result = cart_agent.invoke(state, verbose=False)

        assert result.response == "Your cart is empty."

    def test_view_cart_non_empty_includes_items(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            cart_before=[
                {"item": "Silk Dress", "amount": 2, "price": 49.99},
            ],
        )
        _install_llm_tool_response(cart_agent, "view_cart", {})
        state = State(user_id=1, query="show my cart")

        result = cart_agent.invoke(state, verbose=False)

        assert "Silk Dress" in result.response

    def test_view_cart_total_dispatched(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            cart_before=[
                {"item": "Silk Dress", "amount": 2, "price": 49.99},
            ],
        )
        _install_llm_tool_response(cart_agent, "view_cart_total", {})
        state = State(user_id=1, query="what is my total?")

        result = cart_agent.invoke(state, verbose=False)

        assert "Cart total:" in result.response
        assert "99.98" in result.response

    def test_remove_from_cart_updates_state(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(
            monkeypatch,
            cart_before=[],
            remove_message="removed 1 Silk Dress",
        )
        _install_llm_tool_response(
            cart_agent, "remove_from_cart", {"item_name": "Silk Dress", "quantity": 1}
        )
        state = State(user_id=5, query="remove silk dress")

        result = cart_agent.invoke(state, verbose=False)

        assert "removed 1 Silk Dress" in result.response

    def test_invoke_handles_no_tool_call_gracefully(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(monkeypatch)
        _install_llm_tool_response(
            cart_agent, None, None, fallback_content="I don't know what to do."
        )
        state = State(user_id=1, query="garbled request")

        result = cart_agent.invoke(state, verbose=False)

        assert "couldn't process" in result.response.lower()
        assert "cart" in result.timings

    def test_pronoun_override_rewrites_llm_item_name(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # LLM picks the "wrong" product; deterministic resolution should
        # overwrite item_name to the last-mentioned product in context.
        recorder = _install_http_stubs(monkeypatch)
        _install_llm_tool_response(
            cart_agent,
            "add_to_cart",
            {"item_name": "Hiking Boot", "quantity": 1},
        )
        state = State(
            user_id=11,
            query="add it to my cart",
            context="Earlier: **Hiking Boot**.\nLater: **Silk Dress** was shown.",
        )

        cart_agent.invoke(state, verbose=False)

        # The catalog lookup should have been called with the overridden name.
        catalog_posts = [
            payload for _, url, payload in recorder.calls if url.endswith("/query/text")
        ]
        assert any(p.get("text") == ["Silk Dress"] for p in catalog_posts)

    def test_fallback_tool_call_parsed_from_content(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_http_stubs(monkeypatch, cart_before=[])
        _install_llm_tool_response(
            cart_agent,
            None,
            None,
            fallback_content='{"name": "view_cart", "arguments": {}}',
        )
        state = State(user_id=1, query="what is in my cart")

        result = cart_agent.invoke(state, verbose=False)

        assert result.response == "Your cart is empty."


# ---------------------------------------------------------------------------
# Bulk cart tool dispatch
# ---------------------------------------------------------------------------


def _install_bulk_http_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    catalog_by_query: Dict[str, Dict[str, Any]],
    cart_after: List[Dict[str, Any]] | None = None,
) -> _Recorder:
    """HTTP stubs for bulk flows where each item resolves to a different catalog row.

    The regular ``_install_http_stubs`` returns one catalog response for every
    lookup. For bulk tests we need the catalog retriever mock to return a
    different ``(name, price)`` based on the query string, so both items map
    to real distinct products.
    """
    recorder = _Recorder()

    class _FakeResponse:
        def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
            self._payload = payload
            self.status_code = status

        def json(self) -> Dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise cart_mod.requests.exceptions.HTTPError(
                    f"HTTP {self.status_code}"
                )

        @property
        def text(self) -> str:
            return json.dumps(self._payload)

    class _FakeSession:
        def mount(self, prefix: str, _adapter: Any) -> None:
            pass

        def post(self, url: str, json: Dict[str, Any]):
            recorder.calls.append(("POST", url, json))
            query_list = json.get("text") or []
            query = query_list[0] if query_list else ""
            key = _normalize_name(query)
            entry = catalog_by_query.get(key)
            if entry is None:
                return _FakeResponse(
                    {"names": [], "similarities": [], "texts": []}
                )
            return _FakeResponse(entry)

    def _fake_session_ctor() -> _FakeSession:
        return _FakeSession()

    def _fake_requests_get(url: str, timeout: int = 10):  # noqa: ARG001
        recorder.calls.append(("GET", url, {}))
        return _FakeResponse({"cart": cart_after or []})

    def _fake_requests_post(url: str, json: Dict[str, Any], timeout: int = 10):  # noqa: ARG001
        recorder.calls.append(("POST", url, json))
        if url.endswith("/cart/add"):
            item = json.get("item", "item")
            amount = json.get("amount", 1)
            return _FakeResponse(
                {"message": f"added {amount} of '{item}' to cart"}
            )
        if url.endswith("/cart/remove"):
            item = json.get("item", "item")
            amount = json.get("amount", 1)
            return _FakeResponse(
                {"message": f"removed {amount} of '{item}' from cart"}
            )
        return _FakeResponse({"status": "ok"})

    monkeypatch.setattr(cart_mod.requests, "Session", _fake_session_ctor)
    monkeypatch.setattr(cart_mod.requests, "get", _fake_requests_get)
    monkeypatch.setattr(cart_mod.requests, "post", _fake_requests_post)

    return recorder


class TestBulkCartDispatch:
    """Covers the bulk_add_to_cart / bulk_remove_from_cart branches of
    :meth:`CartAgent.invoke`. These tests exist specifically to pin down the
    behavior that every item named in a single bulk tool call reaches the
    memory service (the original bug was that only the first item did)."""

    def test_bulk_add_to_cart_writes_every_item(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_bulk_http_stubs(
            monkeypatch,
            catalog_by_query={
                "honey floral print midi skirt": {
                    "names": ["Honey Floral Print Midi Skirt"],
                    "similarities": [0.99],
                    "texts": ["Honey Floral Print Midi Skirt | skirt | dress\nPRICE: 69.99"],
                },
                "lace and silk blouse": {
                    "names": ["Lace and Silk Blouse"],
                    "similarities": [0.97],
                    "texts": ["Lace and Silk Blouse | blouse | top\nPRICE: 49.99"],
                },
                "pearl bracelet": {
                    "names": ["Pearl Bracelet"],
                    "similarities": [0.95],
                    "texts": ["Pearl Bracelet | jewellery | bracelet\nPRICE: 29.99"],
                },
            },
        )
        _install_llm_tool_response(
            cart_agent,
            "bulk_add_to_cart",
            {
                "items": [
                    {"item_name": "Honey Floral Print Midi Skirt", "quantity": 1},
                    {"item_name": "Lace and Silk Blouse", "quantity": 1},
                    {"item_name": "Pearl Bracelet", "quantity": 1},
                ]
            },
        )
        state = State(
            user_id=77,
            query=(
                "add the Honey Floral Print Midi Skirt, the Lace and Silk Blouse, "
                "and the Pearl Bracelet to my cart"
            ),
        )

        result = cart_agent.invoke(state, verbose=False)

        add_posts = [
            payload for _, url, payload in recorder.calls if url.endswith("/cart/add")
        ]
        items_written = [p.get("item") for p in add_posts]
        assert items_written == [
            "Honey Floral Print Midi Skirt",
            "Lace and Silk Blouse",
            "Pearl Bracelet",
        ]
        # Prices are cached for deterministic cart totals on every line.
        assert [p.get("price") for p in add_posts] == [69.99, 49.99, 29.99]
        # The surfaced response mentions every item so the chatter's
        # grounding can accurately report the multi-item outcome.
        for name in ["Honey Floral Print Midi Skirt", "Lace and Silk Blouse", "Pearl Bracelet"]:
            assert name in result.response

    def test_bulk_add_reports_per_item_catalog_miss_but_continues(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Silk Dress resolves; Unknown Widget does not. The second line should
        # still be written, and the miss should surface in the response.
        recorder = _install_bulk_http_stubs(
            monkeypatch,
            catalog_by_query={
                "silk dress": {
                    "names": ["Silk Dress"],
                    "similarities": [0.99],
                    "texts": ["Silk Dress | dress | dress\nPRICE: 49.99"],
                },
            },
        )
        _install_llm_tool_response(
            cart_agent,
            "bulk_add_to_cart",
            {
                "items": [
                    {"item_name": "Silk Dress", "quantity": 1},
                    {"item_name": "Unknown Widget", "quantity": 2},
                ]
            },
        )
        state = State(user_id=1, query="add silk dress and unknown widget")

        result = cart_agent.invoke(state, verbose=False)

        add_posts = [
            payload for _, url, payload in recorder.calls if url.endswith("/cart/add")
        ]
        # Only the resolved item reaches memory; the miss is surfaced in text.
        assert [p.get("item") for p in add_posts] == ["Silk Dress"]
        assert "Silk Dress" in result.response
        assert "Unknown Widget" in result.response
        assert "No such item" in result.response

    def test_bulk_add_rewrites_paraphrased_item_names(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The LLM returns abbreviated names (``skirt``/``blouse``). The
        # deterministic resolver should rewrite them back to the catalog
        # names known from RECENT DISCUSSION / cart before the catalog
        # lookup ever runs.
        recorder = _install_bulk_http_stubs(
            monkeypatch,
            catalog_by_query={
                "honey floral print midi skirt": {
                    "names": ["Honey Floral Print Midi Skirt"],
                    "similarities": [0.95],
                    "texts": ["Honey Floral Print Midi Skirt | skirt | dress\nPRICE: 69.99"],
                },
                "lace and silk blouse": {
                    "names": ["Lace and Silk Blouse"],
                    "similarities": [0.95],
                    "texts": ["Lace and Silk Blouse | blouse | top\nPRICE: 49.99"],
                },
            },
        )
        _install_llm_tool_response(
            cart_agent,
            "bulk_add_to_cart",
            {
                "items": [
                    {"item_name": "Honey Skirt", "quantity": 1},
                    {"item_name": "Silk Blouse", "quantity": 1},
                ]
            },
        )
        state = State(
            user_id=9,
            query="add those two",
            context=(
                "Earlier I recommended **Honey Floral Print Midi Skirt**.\n"
                "Then I showed **Lace and Silk Blouse**."
            ),
        )

        cart_agent.invoke(state, verbose=False)

        # After override, both add_to_cart calls should hit memory with the
        # full catalog names, not the abbreviated ones the LLM returned.
        add_posts = [
            payload for _, url, payload in recorder.calls if url.endswith("/cart/add")
        ]
        assert [p.get("item") for p in add_posts] == [
            "Honey Floral Print Midi Skirt",
            "Lace and Silk Blouse",
        ]

    def test_bulk_remove_writes_every_item(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_bulk_http_stubs(
            monkeypatch,
            catalog_by_query={
                "silk dress": {
                    "names": ["Silk Dress"],
                    "similarities": [0.99],
                    "texts": ["Silk Dress | dress | dress\nPRICE: 49.99"],
                },
                "leather bag": {
                    "names": ["Leather Bag"],
                    "similarities": [0.98],
                    "texts": ["Leather Bag | bag | accessory\nPRICE: 99.99"],
                },
            },
        )
        _install_llm_tool_response(
            cart_agent,
            "bulk_remove_from_cart",
            {
                "items": [
                    {"item_name": "Silk Dress", "quantity": 1},
                    {"item_name": "Leather Bag", "quantity": 2},
                ]
            },
        )
        state = State(user_id=3, query="remove the silk dress and the leather bag")

        cart_agent.invoke(state, verbose=False)

        remove_posts = [
            payload for _, url, payload in recorder.calls if url.endswith("/cart/remove")
        ]
        assert [(p.get("item"), p.get("amount")) for p in remove_posts] == [
            ("Silk Dress", 1),
            ("Leather Bag", 2),
        ]

    def test_bulk_add_handles_empty_items_gracefully(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _install_bulk_http_stubs(monkeypatch, catalog_by_query={})
        _install_llm_tool_response(
            cart_agent, "bulk_add_to_cart", {"items": []}
        )
        state = State(user_id=1, query="add these")

        result = cart_agent.invoke(state, verbose=False)

        # No memory writes should happen when the LLM sends an empty list.
        assert not any(url.endswith("/cart/add") for _, url, _ in recorder.calls)
        assert "No items" in result.response

    def test_bulk_add_accepts_stringified_items_from_xml_fallback(
        self, cart_agent: CartAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nemotron's XML tool-call path delivers ``items`` as a string repr.
        # ``_coerce_value`` already parses that into a list; this test also
        # exercises the defensive re-parse in ``_coerce_bulk_items`` by
        # passing the arguments verbatim as an XML tool_call content blob.
        _install_bulk_http_stubs(
            monkeypatch,
            catalog_by_query={
                "silk dress": {
                    "names": ["Silk Dress"],
                    "similarities": [0.99],
                    "texts": ["Silk Dress | dress | dress\nPRICE: 49.99"],
                },
            },
        )
        xml_content = (
            "<tool_call><function=bulk_add_to_cart>"
            "<parameter=items>[{'item_name': 'Silk Dress', 'quantity': 1}]</parameter>"
            "</function></tool_call>"
        )
        _install_llm_tool_response(
            cart_agent, None, None, fallback_content=xml_content
        )
        state = State(user_id=1, query="add silk dress")

        result = cart_agent.invoke(state, verbose=False)

        assert "Silk Dress" in result.response
