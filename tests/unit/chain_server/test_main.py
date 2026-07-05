# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.main``.

The module does expensive work at import time: it calls ``load_config`` and
constructs the assistant runtime. For unit tests we replace that runtime with
a lightweight stub before importing the module.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from chain_server.src.agenttypes import Cart, State
from shared.commerce_contracts import Cart as CommerceCart
from shared.commerce_contracts import (
    CartLine,
    CatalogCapabilities,
    CatalogFilterCapability,
    Money,
    ProductSummary,
    SearchCatalogResult,
)


class _StubRuntime:
    """Replacement for the Deep Agents runtime."""

    def __init__(self, response_text: str = "ok") -> None:
        self.response_text = response_text
        self.astream_calls: List[Any] = []
        self.ainvoke_calls: List[Any] = []

    async def astream(self, state: State, identity):
        self.astream_calls.append((state, identity))
        for piece in ["hello ", "world"]:
            yield piece

    async def ainvoke(self, state: State, identity) -> Dict[str, Any]:
        self.ainvoke_calls.append((state, identity))
        return {
            "response": self.response_text,
            "timings": {"chatter": 0.1, "memory": 0.01},
        }

    def catalog_capabilities(self, *, force_refresh: bool = False) -> CatalogCapabilities:
        return CatalogCapabilities(
            catalog_id="test_catalog",
            retrieval_modes=["text"],
            filters={
                "category": CatalogFilterCapability(
                    type="enum",
                    operators=["in"],
                    source_fields=["subcategory"],
                    values=["bag", "dress"],
                )
            },
        )


@pytest.fixture
def main_module(
    monkeypatch: pytest.MonkeyPatch, base_config
) -> Iterator[Any]:
    """Import ``chain_server.src.main`` with all heavy deps stubbed."""
    from chain_server.src import config as config_mod
    from chain_server.src import deepagents_runtime as runtime_mod

    # Config loader returns our pre-baked config rather than reading YAML.
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: base_config)

    runtime = _StubRuntime()
    monkeypatch.setattr(runtime_mod, "DeepAgentsRuntime", lambda *_: runtime)

    # Force a fresh import so our stubs are actually used.
    sys.modules.pop("chain_server.src.main", None)
    main_module = importlib.import_module("chain_server.src.main")
    main_module._test_runtime = runtime  # type: ignore[attr-defined]

    yield main_module

    sys.modules.pop("chain_server.src.main", None)


@pytest.fixture
def client(main_module) -> TestClient:
    return TestClient(main_module.app)


# ---------------------------------------------------------------------------
# create_initial_state
# ---------------------------------------------------------------------------


class TestCreateInitialState:
    def test_defaults_fill_empty_strings_and_empty_cart(
        self, main_module
    ) -> None:
        request = main_module.QueryRequest(user_id=1, query="hi")
        state = main_module.create_initial_state(request)

        assert state.user_id == 1
        assert state.query == "hi"
        assert state.context == ""
        assert state.image == ""
        assert isinstance(state.cart, Cart)
        assert state.cart.is_empty()
        assert state.guardrails is True

    def test_cart_passthrough(self, main_module) -> None:
        cart = Cart(contents=[{"item": "X", "amount": 2, "price": 9.99}])
        request = main_module.QueryRequest(user_id=1, query="hi", cart=cart)
        state = main_module.create_initial_state(request)

        assert state.cart.contents == cart.contents

    def test_none_context_becomes_empty(self, main_module) -> None:
        request = main_module.QueryRequest(user_id=1, query="hi", context=None)
        state = main_module.create_initial_state(request)
        assert state.context == ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TestHealthAndRoot:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"] == "1.0.0"

    def test_capabilities_return_media_config(self, client: TestClient) -> None:
        response = client.get("/capabilities")
        assert response.status_code == 200

        body = response.json()
        media = body["media_input"]
        assert media["enabled"] is True
        assert media["max_images_per_turn"] == 1
        assert media["max_videos_per_turn"] == 1
        assert media["video_mime_types"] == ["video/mp4"]
        assert media["vlm_enabled"] is True
        assert body["catalog"]["catalog_id"] == "test_catalog"
        assert body["catalog"]["filters"]["category"]["values"] == ["bag", "dress"]

    def test_root_describes_endpoints(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

        body = response.json()
        assert body["message"] == "Shopping Assistant API"
        assert body["version"] == "1.0.0"
        for key in ["query", "stream", "timing", "capabilities", "health", "docs"]:
            assert key in body["endpoints"]


class TestTimingEndpoint:
    def test_returns_response_and_timings(
        self, main_module, client: TestClient
    ) -> None:
        response = client.post(
            "/query/timing",
            json={"user_id": 1, "query": "hello"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == main_module._test_runtime.response_text
        assert "total" in body["timings"]
        assert body["timings"]["total"] > 0


class TestStreamEndpoint:
    def test_stream_returns_sse_body_with_done_marker(
        self, main_module, client: TestClient
    ) -> None:
        with client.stream(
            "POST",
            "/query/stream",
            json={"user_id": 1, "query": "hi"},
        ) as stream_response:
            assert stream_response.status_code == 200
            chunks: List[str] = []
            for line in stream_response.iter_lines():
                if line:
                    chunks.append(line)

        joined = "\n".join(chunks)
        assert "data: hello " in joined
        assert "data: world" in joined
        assert "[DONE]" in joined

    def test_image_only_query_populates_placeholder(
        self, main_module, client: TestClient
    ) -> None:
        # Image-only requests should get a placeholder query injected so that
        # the graph has something to work with.
        runtime = main_module._test_runtime
        runtime.astream_calls.clear()

        with client.stream(
            "POST",
            "/query/stream",
            json={"user_id": 1, "query": "", "image": "data:image/jpeg;base64,AAA"},
        ) as stream_response:
            # Drain the stream so the generator actually runs.
            for _ in stream_response.iter_lines():
                pass

        assert runtime.astream_calls
        state_arg, _ = runtime.astream_calls[-1]
        assert state_arg.image.startswith("data:image/jpeg")
        assert "media" in state_arg.query.lower()

    def test_video_only_query_populates_media_placeholder(
        self, main_module, client: TestClient
    ) -> None:
        runtime = main_module._test_runtime
        runtime.astream_calls.clear()

        with client.stream(
            "POST",
            "/query/stream",
            json={
                "user_id": 1,
                "query": "",
                "media": [
                    {
                        "type": "video",
                        "mime_type": "video/mp4",
                        "data": "data:video/mp4;base64,QUFB",
                    }
                ],
            },
        ) as stream_response:
            for _ in stream_response.iter_lines():
                pass

        assert runtime.astream_calls
        state_arg, _ = runtime.astream_calls[-1]
        assert state_arg.image == ""
        assert state_arg.query == "The user submitted visual media without additional text."
        assert state_arg.media[0]["type"] == "video"
        assert state_arg.media[0]["mime_type"] == "video/mp4"

    def test_rejects_too_many_images(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/query/timing",
            json={
                "user_id": 1,
                "query": "find these",
                "image": "data:image/jpeg;base64,QUFB",
                "media": [
                    {
                        "type": "image",
                        "mime_type": "image/png",
                        "data": "data:image/png;base64,QkJC",
                    }
                ],
            },
        )

        assert response.status_code == 400
        assert "At most 1 image" in response.json()["detail"]

    def test_optional_identity_fields_reach_runtime(
        self, main_module, client: TestClient
    ) -> None:
        response = client.post(
            "/query/timing",
            json={
                "user_id": 1,
                "query": "hello",
                "session_id": "session-a",
                "conversation_id": "conversation-a",
                "cart_id": "cart-a",
            },
        )

        assert response.status_code == 200
        _, identity = main_module._test_runtime.ainvoke_calls[-1]
        assert identity.session_id == "session-a"
        assert identity.conversation_id == "conversation-a"
        assert identity.cart_id == "cart-a"
        assert identity.context_user_id != 1
        assert identity.cart_user_id != 1


class TestRequestIdentity:
    def test_missing_explicit_ids_keep_legacy_user_scope(self) -> None:
        from chain_server.src.deepagents_runtime import create_request_identity

        identity = create_request_identity(legacy_user_id=42)

        assert identity.session_id == "legacy-session-42"
        assert identity.conversation_id == "legacy-conversation-42"
        assert identity.cart_id == "legacy-cart-42"
        assert identity.context_user_id == 42
        assert identity.cart_user_id == 42
        assert identity.legacy_user_id == 42

    def test_cart_scope_can_survive_across_conversations(self) -> None:
        from chain_server.src.deepagents_runtime import create_request_identity

        first = create_request_identity(
            legacy_user_id=1,
            conversation_id="conversation-a",
            cart_id="cart-shared",
        )
        second = create_request_identity(
            legacy_user_id=2,
            conversation_id="conversation-b",
            cart_id="cart-shared",
        )
        different_cart = create_request_identity(
            legacy_user_id=1,
            conversation_id="conversation-a",
            cart_id="cart-other",
        )

        assert first.context_user_id != second.context_user_id
        assert first.cart_user_id == second.cart_user_id
        assert first.cart_user_id != different_cart.cart_user_id

    def test_missing_cart_id_keeps_cart_on_legacy_user_scope(self) -> None:
        from chain_server.src.deepagents_runtime import create_request_identity

        identity = create_request_identity(
            legacy_user_id=42,
            conversation_id="conversation-a",
        )

        assert identity.context_user_id != 42
        assert identity.cart_user_id == 42


class TestDeepAgentsRuntimeScopes:
    def test_load_and_persist_memory_use_context_scope_while_cart_uses_cart_scope(
        self,
        base_config,
        fake_response_cls,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        identity = runtime_mod.RequestIdentity(
            session_id="session-a",
            conversation_id="conversation-a",
            cart_id="cart-a",
            context_user_id=111,
            cart_user_id=222,
            request_id="request-a",
        )
        state = State(user_id=999, query="hello", guardrails=False)
        cart_user_ids: List[int] = []
        get_urls: List[str] = []
        post_urls: List[str] = []

        def fake_get(url: str, timeout: int):
            get_urls.append(url)
            return fake_response_cls({"context": "prior context"})

        def fake_post(url: str, json: Dict[str, str], timeout: int):
            post_urls.append(url)
            return fake_response_cls({"message": "ok"})

        def fake_read_cart(user_id: int) -> Cart:
            cart_user_ids.append(user_id)
            return Cart(contents=[{"item": "Bag", "amount": 1, "price": 20.0}])

        monkeypatch.setattr(runtime_mod.requests, "get", fake_get)
        monkeypatch.setattr(runtime_mod.requests, "post", fake_post)
        monkeypatch.setattr(runtime, "_read_cart", fake_read_cart)

        runtime._load_memory(state, identity)
        runtime._persist_context(state, identity)

        assert get_urls == [f"{base_config.memory_port}/user/111/context"]
        assert post_urls == [f"{base_config.memory_port}/user/111/context/replace"]
        assert cart_user_ids == [222]
        assert state.context == "prior context"
        assert state.cart.contents == [{"item": "Bag", "amount": 1, "price": 20.0}]


class TestDeepAgentsRuntimeRefs:
    def test_search_and_cart_read_tools_are_chainable(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        captured: Dict[str, Any] = {}
        deepagents_mod = ModuleType("deepagents")
        tools_mod = ModuleType("langchain_core.tools")
        openai_mod = ModuleType("langchain_openai")

        class FakeProfile:
            def __init__(self, *args, **kwargs) -> None:
                pass

        class FakeChatOpenAI:
            def __init__(self, *args, **kwargs) -> None:
                pass

        def fake_tool(*, return_direct: bool = False):
            def decorate(fn):
                fn.return_direct = return_direct
                return fn

            return decorate

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        deepagents_mod.GeneralPurposeSubagentProfile = FakeProfile
        deepagents_mod.HarnessProfile = FakeProfile
        deepagents_mod.create_deep_agent = fake_create_deep_agent
        deepagents_mod.register_harness_profile = lambda *args, **kwargs: None
        tools_mod.tool = fake_tool
        openai_mod.ChatOpenAI = FakeChatOpenAI

        monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        runtime._catalog_capabilities = SimpleNamespace(
            get=lambda: CatalogCapabilities(
                catalog_id="custom_catalog",
                retrieval_modes=["text"],
                filters={
                    "category": CatalogFilterCapability(
                        type="enum",
                        operators=["in"],
                        source_fields=["subcategory"],
                        values=["dress"],
                    ),
                    "color": CatalogFilterCapability(
                        type="enum",
                        operators=["in"],
                        source_fields=["color"],
                        values=["blue"],
                    )
                },
            )
        )
        identity = runtime_mod.RequestIdentity(
            session_id="session-a",
            conversation_id="conversation-a",
            cart_id="cart-a",
            context_user_id=111,
            cart_user_id=222,
            request_id="request-a",
        )

        runtime._create_agent(State(user_id=111, query="hello"), identity)

        tools_by_name = {fn.__name__: fn for fn in captured["tools"]}
        assert tools_by_name["search_catalog_tool"].return_direct is False
        assert tools_by_name["get_product_details_tool"].return_direct is False
        assert tools_by_name["get_cart_tool"].return_direct is False
        assert tools_by_name["add_cart_item_tool"].return_direct is True
        assert tools_by_name["remove_cart_item_tool"].return_direct is True
        assert tools_by_name["view_cart_total_tool"].return_direct is True
        assert "Catalog ID: custom_catalog" in captured["system_prompt"]
        assert "values dress" in captured["system_prompt"]
        assert "top blouse sweater" not in captured["system_prompt"]

    def test_search_catalog_tool_executes_structured_plan(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        captured: Dict[str, Any] = {}
        deepagents_mod = ModuleType("deepagents")
        tools_mod = ModuleType("langchain_core.tools")
        openai_mod = ModuleType("langchain_openai")

        class FakeProfile:
            def __init__(self, *args, **kwargs) -> None:
                pass

        class FakeChatOpenAI:
            def __init__(self, *args, **kwargs) -> None:
                pass

        def fake_tool(*, return_direct: bool = False):
            def decorate(fn):
                fn.return_direct = return_direct
                return fn

            return decorate

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        capabilities = CatalogCapabilities(
            catalog_id="fashion",
            retrieval_modes=["text", "image", "hybrid"],
            image_search_enabled=True,
            filters={
                "category": CatalogFilterCapability(
                    type="enum",
                    operators=["in"],
                    source_fields=["subcategory"],
                    values=["bag", "dress"],
                ),
                "price": CatalogFilterCapability(
                    type="number",
                    operators=["gte", "lte"],
                    source_fields=["price"],
                ),
                "color": CatalogFilterCapability(
                    type="enum",
                    operators=["in"],
                    source_fields=["color"],
                    values=["blue", "black"],
                ),
            },
        )
        captured_plan = {}

        def fake_execute_catalog_search(plan, *args, **kwargs):
            captured_plan["plan"] = plan
            return SimpleNamespace(
                result=SearchCatalogResult(
                    ok=True,
                    products=[
                        ProductSummary(
                            product_id="prod_1",
                            display_name="Work Bag",
                            image_url="bag.jpg",
                            price=Money(amount=59.0),
                        )
                    ],
                ),
                fallback_used=False,
            )

        deepagents_mod.GeneralPurposeSubagentProfile = FakeProfile
        deepagents_mod.HarnessProfile = FakeProfile
        deepagents_mod.create_deep_agent = fake_create_deep_agent
        deepagents_mod.register_harness_profile = lambda *args, **kwargs: None
        tools_mod.tool = fake_tool
        openai_mod.ChatOpenAI = FakeChatOpenAI

        monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)
        monkeypatch.setattr(
            runtime_mod,
            "execute_catalog_search",
            fake_execute_catalog_search,
        )

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        runtime._catalog_capabilities = SimpleNamespace(get=lambda: capabilities)
        identity = runtime_mod.RequestIdentity(
            session_id="session-a",
            conversation_id="conversation-a",
            cart_id="cart-a",
            context_user_id=111,
            cart_user_id=222,
            request_id="request-a",
        )
        state = State(
            user_id=111,
            query="show me practical work bags under $60",
        )

        runtime._create_agent(state, identity)
        tools_by_name = {fn.__name__: fn for fn in captured["tools"]}

        result = tools_by_name["search_catalog_tool"](
            "practical work bag",
            filters={
                "category": ["bag"],
                "price": {"max": 60},
                "color": ["blue"],
            },
            strictness="hard",
        )

        assert "PRODUCT_REF: prod_1" in result
        assert state.retrieved == {"Work Bag": "bag.jpg"}
        assert captured_plan["plan"].hard_filters == {
            "category": ["bag"],
            "price": {"max": 60.0},
            "color": ["blue"],
        }
        assert captured_plan["plan"].strictness == "hard"

    def test_product_refs_are_cached_by_conversation(
        self,
        base_config,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        identity_a = runtime_mod.RequestIdentity(
            session_id="session-a",
            conversation_id="conversation-a",
            cart_id="cart-a",
            context_user_id=111,
            cart_user_id=222,
            request_id="request-a",
        )
        identity_b = runtime_mod.RequestIdentity(
            session_id="session-b",
            conversation_id="conversation-b",
            cart_id="cart-b",
            context_user_id=333,
            cart_user_id=444,
            request_id="request-b",
        )
        product = ProductSummary(
            product_id="prod_123",
            display_name="Silk Dress",
            price=Money(amount=49.99),
        )

        runtime._remember_products(identity_a, [product])

        assert runtime._product_from_ref(identity_a, "prod_123") == product
        assert runtime._product_from_ref(identity_b, "prod_123") is None

    def test_product_details_tool_reads_cached_product_ref(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        captured: Dict[str, Any] = {}
        deepagents_mod = ModuleType("deepagents")
        tools_mod = ModuleType("langchain_core.tools")
        openai_mod = ModuleType("langchain_openai")

        class FakeProfile:
            def __init__(self, *args, **kwargs) -> None:
                pass

        class FakeChatOpenAI:
            def __init__(self, *args, **kwargs) -> None:
                pass

        def fake_tool(*, return_direct: bool = False):
            def decorate(fn):
                fn.return_direct = return_direct
                return fn

            return decorate

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        deepagents_mod.GeneralPurposeSubagentProfile = FakeProfile
        deepagents_mod.HarnessProfile = FakeProfile
        deepagents_mod.create_deep_agent = fake_create_deep_agent
        deepagents_mod.register_harness_profile = lambda *args, **kwargs: None
        tools_mod.tool = fake_tool
        openai_mod.ChatOpenAI = FakeChatOpenAI

        monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        runtime._catalog_capabilities = SimpleNamespace(
            get=lambda: CatalogCapabilities(
                catalog_id="fashion",
                retrieval_modes=["text"],
                filters={},
            )
        )
        identity = runtime_mod.RequestIdentity(
            session_id="session-a",
            conversation_id="conversation-a",
            cart_id="cart-a",
            context_user_id=111,
            cart_user_id=222,
            request_id="request-a",
        )
        runtime._remember_products(
            identity,
            [
                ProductSummary(
                    product_id="prod_123",
                    display_name="Work Bag",
                    description="structured tote",
                    category="bag",
                    price=Money(amount=59.0),
                    attributes={"catalog_text": "Work Bag | structured tote | bag"},
                )
            ],
        )

        runtime._create_agent(State(user_id=111, query="tell me more"), identity)
        tools_by_name = {fn.__name__: fn for fn in captured["tools"]}

        details = tools_by_name["get_product_details_tool"]("prod_123")
        missing = tools_by_name["get_product_details_tool"]("Work Bag")

        assert "PRODUCT_REF: prod_123" in details
        assert "DESCRIPTION: structured tote" in details
        assert "No product with PRODUCT_REF 'Work Bag'" in missing

    def test_format_product_exposes_product_ref(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        product = ProductSummary(
            product_id="prod_456",
            display_name="Leather Bag",
            description="structured tote",
            price=Money(amount=129.0),
        )

        formatted = runtime_mod._format_product(product)

        assert "PRODUCT_REF: prod_456" in formatted
        assert "Leather Bag" in formatted

    def test_format_cart_exposes_cart_line_id(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        cart = Cart(
            contents=[
                {
                    "cart_line_id": "line_1",
                    "product_id": "prod_123",
                    "item": "Silk Dress",
                    "amount": 2,
                    "price": 49.99,
                }
            ]
        )

        formatted = runtime_mod._format_cart(cart)

        assert "CART_LINE_ID: line_1" in formatted
        assert "2 x Silk Dress" in formatted

    def test_read_cart_preserves_cart_line_and_product_refs(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        contract_cart = CommerceCart(
            user_id="222",
            lines=[
                CartLine(
                    cart_line_id="line_1",
                    product_id="prod_123",
                    display_name="Silk Dress",
                    quantity=2,
                    unit_price=Money(amount=49.99),
                )
            ],
        )

        monkeypatch.setattr(
            runtime_mod,
            "get_cart",
            lambda request, memory_port: SimpleNamespace(ok=True, cart=contract_cart),
        )

        cart = runtime._read_cart(222)

        assert cart.contents == [
            {
                "cart_line_id": "line_1",
                "product_id": "prod_123",
                "item": "Silk Dress",
                "amount": 2,
                "price": 49.99,
            }
        ]

    def test_cart_line_lookup_uses_exact_cart_line_id(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        cart = Cart(
            contents=[
                {"cart_line_id": "line_1", "item": "Silk Dress", "amount": 1},
                {"cart_line_id": "line_2", "item": "Silk Dress", "amount": 1},
            ]
        )

        assert runtime_mod._cart_line_by_id("line_2", cart) == cart.contents[1]
        assert runtime_mod._cart_line_by_id("Silk Dress", cart) is None


class TestValidation:
    def test_missing_user_id_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/query/timing",
            json={"query": "hi"},
        )
        assert response.status_code == 422

    def test_bad_payload_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/query/timing",
            json={"user_id": "not-an-int", "query": "hi"},
        )
        assert response.status_code == 422
