# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.main``.

The module does expensive work at import time: it calls ``load_config`` and
constructs the assistant runtime. For unit tests we replace that runtime with
a lightweight stub before importing the module.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from chain_server.src.agenttypes import Cart, State
from shared.commerce_contracts import Cart as CommerceCart
from shared.commerce_contracts import (
    CartLine,
    CartMutationResult,
    CatalogCapabilities,
    CatalogFilterCapability,
    CommerceError,
    GetProductDetailsResult,
    Money,
    ProductDetail,
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

    def catalog_capabilities(self) -> CatalogCapabilities:
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

    def test_guardrails_request_overrides_config_default(self, main_module) -> None:
        request = main_module.QueryRequest(user_id=1, query="hi", guardrails=False)
        state = main_module.create_initial_state(request)

        assert state.guardrails is False

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
        assert body["models"]["app_llm"]["model"] == "test-model"
        assert body["models"]["app_llm"]["enabled"] is True
        assert body["models"]["vlm"]["model"] == "test-vlm"
        assert body["models"]["vlm"]["enabled"] is True
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
        assert body["cart"] == {"contents": []}
        assert "total" in body["timings"]
        assert body["timings"]["total"] > 0
        assert body["model_usage"] == {}


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


class TestDeepAgentsRuntimeTokenUsage:
    def test_collects_normalized_usage_metadata_without_double_counting(self) -> None:
        from chain_server.src.deepagents_runtime import _collect_token_usage

        result = {
            "messages": [
                SimpleNamespace(
                    content="thinking",
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                    },
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        }
                    },
                ),
                {
                    "content": "final",
                    "response_metadata": {
                        "token_usage": {
                            "prompt_tokens": 40,
                            "completion_tokens": 10,
                            "total_tokens": 50,
                        }
                    },
                },
            ]
        }

        assert _collect_token_usage(result) == {
            "input_tokens": 140,
            "output_tokens": 30,
            "total_tokens": 170,
            "model_calls": 2,
        }

    def test_collect_token_usage_defaults_when_metadata_is_absent(self) -> None:
        from chain_server.src.deepagents_runtime import _collect_token_usage

        assert _collect_token_usage({"messages": [{"content": "hello"}]}) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model_calls": 0,
        }


class TestDeepAgentsRuntimeModelUsage:
    def test_safety_model_usage_matches_guardrails_flows(self) -> None:
        from chain_server.src.deepagents_runtime import _record_safety_model_usage

        state = State(user_id=1, query="hello")

        _record_safety_model_usage(state, "input")
        _record_safety_model_usage(state, "output")

        assert state.model_usage["content_safety"]["status"] == "used"
        assert state.model_usage["content_safety"]["calls"] == 2
        assert state.model_usage["topic_control"]["status"] == "used"
        assert state.model_usage["topic_control"]["calls"] == 1

    def test_safety_model_usage_marks_transport_failures(self) -> None:
        from chain_server.src.deepagents_runtime import _record_safety_model_usage

        state = State(user_id=1, query="hello")

        _record_safety_model_usage(state, "input", ok=False)

        assert state.model_usage["content_safety"]["status"] == "failed"
        assert state.model_usage["content_safety"]["calls"] == 1
        assert state.model_usage["topic_control"]["status"] == "failed"
        assert state.model_usage["topic_control"]["calls"] == 1

    def test_safety_check_transport_error_fails_open_with_failed_usage_signal(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        runtime = runtime_mod.DeepAgentsRuntime(base_config)

        def fake_post(*args, **kwargs):
            raise runtime_mod.requests.RequestException("rails down")

        monkeypatch.setattr(runtime_mod.requests, "post", fake_post)

        safe, check_ok = runtime._check_safety("input", 1, "hello")

        assert safe is True
        assert check_ok is False

    def test_language_model_failure_usage_is_explicit(self) -> None:
        from chain_server.src.deepagents_runtime import _record_language_model_failure

        state = State(user_id=1, query="hello")

        _record_language_model_failure(state)

        assert state.model_usage["app_llm"]["status"] == "failed"
        assert state.model_usage["app_llm"]["calls"] == 1


class TestDeepAgentsRuntimeMediaFailures:
    def test_video_dependent_query_short_circuits_when_vlm_is_unavailable(self) -> None:
        from chain_server.src.deepagents_runtime import (
            _media_failure_response,
            _should_short_circuit_media_failure,
        )

        state = State(
            user_id=1,
            query="I love the shoes she is wearing in this video. Do you have something like that?",
            media=[
                {
                    "type": "video",
                    "mime_type": "video/mp4",
                    "data": "data:video/mp4;base64,QUFB",
                }
            ],
            media_analysis=(
                '{"summary": "Media was attached, but the configured VLM could not '
                'authenticate. Video/image understanding is unavailable for this turn.", '
                '"uncertainties": ["VLM authentication failed."]}'
            ),
        )

        assert _should_short_circuit_media_failure(state) is True
        response = _media_failure_response(state.media_analysis)
        assert "Please describe the item in text" in response
        assert "turn.. Please" not in response

    def test_explicit_text_query_can_continue_when_media_is_unavailable(self) -> None:
        from chain_server.src.deepagents_runtime import _should_short_circuit_media_failure

        state = State(
            user_id=1,
            query="Find black patent heels under $100",
            media=[
                {
                    "type": "video",
                    "mime_type": "video/mp4",
                    "data": "data:video/mp4;base64,QUFB",
                }
            ],
            media_analysis=(
                '{"summary": "Video was attached, but VLM media understanding is not configured.", '
                '"uncertainties": ["Video understanding requires an enabled VLM."]}'
            ),
        )

        assert _should_short_circuit_media_failure(state) is False

    def test_image_similarity_query_continues_when_vlm_is_unavailable(self) -> None:
        from chain_server.src.deepagents_runtime import _should_short_circuit_media_failure

        image_data = "data:image/jpeg;base64,QUFB"
        state = State(
            user_id=1,
            query="Find products similar to this image",
            image=image_data,
            media=[
                {
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "data": image_data,
                }
            ],
            media_analysis=(
                '{"summary": "Media was attached, but the configured VLM could not '
                'authenticate. Video/image understanding is unavailable for this turn.", '
                '"uncertainties": ["VLM authentication failed."]}'
            ),
        )

        assert _should_short_circuit_media_failure(state) is False


class TestDeepAgentsRuntimeRefs:
    @pytest.mark.parametrize("legacy_field", ["filters", "strictness"])
    def test_search_catalog_tool_input_rejects_legacy_constraint_fields(
        self, legacy_field: str
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            runtime_mod.SearchCatalogToolInput.model_validate(
                {
                    "semantic_query": "dresses",
                    legacy_field: (
                        {"price": {"max": 100}}
                        if legacy_field == "filters"
                        else "hard"
                    ),
                }
            )

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
        registered_profile: Dict[str, Any] = {}

        class FakeProfile:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

        class FakeChatOpenAI:
            def __init__(self, *args, **kwargs) -> None:
                pass

        def fake_tool(*, args_schema=None, return_direct: bool = False):
            def decorate(fn):
                fn.args_schema = args_schema
                fn.return_direct = return_direct
                return fn

            return decorate

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        def fake_register_harness_profile(name, profile):
            registered_profile["name"] = name
            registered_profile["profile"] = profile

        deepagents_mod.GeneralPurposeSubagentProfile = FakeProfile
        deepagents_mod.HarnessProfile = FakeProfile
        deepagents_mod.create_deep_agent = fake_create_deep_agent
        deepagents_mod.register_harness_profile = fake_register_harness_profile
        tools_mod.tool = fake_tool
        openai_mod.ChatOpenAI = FakeChatOpenAI

        monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        runtime._catalog_capabilities = SimpleNamespace(
            get=lambda **_: CatalogCapabilities(
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
        assert set(tools_by_name) == {
            "search_catalog_tool",
            "get_product_details_tool",
            "get_cart_tool",
            "add_cart_items_tool",
            "remove_cart_item_tool",
            "view_cart_total_tool",
        }
        assert (
            tools_by_name["search_catalog_tool"].args_schema
            is runtime_mod.SearchCatalogToolInput
        )
        assert set(runtime_mod.SearchCatalogToolInput.model_fields) == {
            "semantic_query",
            "required_constraints",
            "search_mode",
        }
        assert (
            tools_by_name["add_cart_items_tool"].args_schema
            is runtime_mod.AddCartItemsToolInput
        )
        assert tools_by_name["search_catalog_tool"].return_direct is False
        assert tools_by_name["get_product_details_tool"].return_direct is False
        assert tools_by_name["get_cart_tool"].return_direct is False
        assert tools_by_name["add_cart_items_tool"].return_direct is False
        assert tools_by_name["remove_cart_item_tool"].return_direct is False
        assert tools_by_name["view_cart_total_tool"].return_direct is False
        assert captured["skills"] == ["/shopper"]
        assert captured["backend"].cwd == (
            Path(__file__).resolve().parents[3] / "chain_server" / "skills"
        )
        assert captured["backend"].virtual_mode is True
        excluded_tools = registered_profile["profile"].kwargs["excluded_tools"]
        assert "read_file" not in excluded_tools
        assert "write_file" in excluded_tools
        assert "execute" in excluded_tools
        assert "Retrieval modes: text" in captured["system_prompt"]
        assert "values dress" in captured["system_prompt"]
        assert "Put every shopper must-have in `required_constraints`" in (
            captured["system_prompt"]
        )
        assert "Semantic relevance ranks candidates" in captured["system_prompt"]
        assert "top blouse sweater" not in captured["system_prompt"]
        assert "Cart mutation scope must match" in captured["system_prompt"]
        assert "Selection, approval, or styling preference is not cart intent" in (
            captured["system_prompt"]
        )
        assert "If cart mutation scope is ambiguous" in captured["system_prompt"]
        assert "ask one concise clarification" in captured["system_prompt"]
        assert "For an explicit cart swap" in captured["system_prompt"]
        assert "remove the rejected cart line" in captured["system_prompt"]
        assert "Product comparison tables" in captured["system_prompt"]
        assert "require get_product_details_tool" in captured["system_prompt"]
        assert "Do not upgrade shopper assumptions" in captured["system_prompt"]
        assert "Do not\nshow them to shoppers" in captured["system_prompt"]
        assert "Do not group leather, rubber, metal" in captured["system_prompt"]
        assert "Shopper wording is not product evidence" in captured["system_prompt"]
        assert "making unsupported whole-outfit claims" in captured["system_prompt"]
        assert "Initial recommendations should use product name" in (
            captured["system_prompt"]
        )
        assert "Search-only product names are display names" in (
            captured["system_prompt"]
        )
        assert "Do not make group-level claims" in captured["system_prompt"]
        assert "Do not enumerate materials" in captured["system_prompt"]
        assert "Tax, shipping fees, delivery dates" in captured["system_prompt"]
        assert "real-time stock or inventory status" in captured["system_prompt"]
        assert "Outdoor-practicality claims require exact support" in (
            captured["system_prompt"]
        )
        assert "stable on grass or gravel" in captured["system_prompt"]
        assert "will stay comfortable all evening" in captured["system_prompt"]
        assert "Rubber sole means" in captured["system_prompt"]
        assert "maximum breathability" in captured["system_prompt"]
        assert "best-in-category performance" in captured["system_prompt"]
        assert "compare only confirmed construction facts" in captured["system_prompt"]

    def test_shopper_agent_tool_registry_matches_registered_tool_names(self) -> None:
        registry_path = (
            Path(__file__).resolve().parents[3]
            / "docs"
            / "SHOPPER_AGENT_TOOL_REGISTRY.md"
        )
        registry = registry_path.read_text()
        registered_lines = [
            line
            for line in registry.splitlines()
            if line.startswith("| `") and line.rstrip().endswith("| Registered |")
        ]
        registered_tools = {
            line.split("|", 2)[1].strip().strip("`") for line in registered_lines
        }

        assert registered_tools == {
            "search_catalog_tool",
            "get_product_details_tool",
            "get_cart_tool",
            "view_cart_total_tool",
            "add_cart_items_tool",
            "remove_cart_item_tool",
        }
        for planned_tool in (
            "get_store_policy_tool",
            "update_cart_item_tool",
            "load_customer_persona_tool",
        ):
            assert f"| `{planned_tool}` |" in registry
            assert f"| `{planned_tool}` | Planned" in registry

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

        def fake_tool(*, args_schema=None, return_direct: bool = False):
            def decorate(fn):
                fn.args_schema = args_schema
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
            captured_plan["calls"] = captured_plan.get("calls", 0) + 1
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
        runtime._catalog_capabilities = SimpleNamespace(get=lambda **_: capabilities)
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
            semantic_query="practical work bag",
            required_constraints={
                "category": ["bag"],
                "price": {"max": 60},
                "color": ["blue"],
            },
        )

        assert "SEARCH_RESULT_GROUNDING_NOTE" in result
        assert "Call get_product_details_tool" in result
        assert "PRODUCT_REF: prod_1" in result
        assert state.retrieved == {"Work Bag": "bag.jpg"}
        assert [product["product_id"] for product in state.product_results] == ["prod_1"]
        assert state.model_usage["text_embedding"]["status"] == "used"
        assert state.model_usage["text_embedding"]["calls"] == 1
        assert "image_embedding" not in state.model_usage
        assert captured_plan["plan"].semantic_queries == ["practical work bag"]
        assert captured_plan["plan"].hard_filters == {
            "category": ["bag"],
            "price": {"max": 60.0},
            "color": ["blue"],
        }
        assert captured_plan["calls"] == 1

        required_constraint_failure = tools_by_name["search_catalog_tool"](
            semantic_query="dresses",
            required_constraints={"composition": "cotton"},
        )

        assert "catalog requirement cannot be enforced" in required_constraint_failure
        assert "'composition' is not an advertised hard filter" in (
            required_constraint_failure
        )
        assert captured_plan["calls"] == 1

        image_state = State(
            user_id=111,
            query="find products similar to this image",
            image="data:image/jpeg;base64,QUFB",
        )
        runtime._create_agent(image_state, identity)
        image_search_tool = {fn.__name__: fn for fn in captured["tools"]}["search_catalog_tool"]

        image_result = image_search_tool(semantic_query="", required_constraints={})

        assert "SEARCH_RESULT_GROUNDING_NOTE" in image_result
        assert "PRODUCT_REF: prod_1" in image_result
        assert captured_plan["plan"].search_mode == "hybrid"
        assert image_state.model_usage["text_embedding"]["status"] == "used"
        assert image_state.model_usage["text_embedding"]["calls"] == 1
        assert image_state.model_usage["image_embedding"]["status"] == "used"
        assert image_state.model_usage["image_embedding"]["calls"] == 1

    def test_search_catalog_tool_enforces_per_turn_cap(
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

        def fake_tool(*, args_schema=None, return_direct: bool = False):
            def decorate(fn):
                fn.args_schema = args_schema
                fn.return_direct = return_direct
                return fn

            return decorate

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        calls = 0

        def fake_execute_catalog_search(plan, *args, **kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                result=SearchCatalogResult(
                    ok=True,
                    products=[
                        ProductSummary(
                            product_id=f"prod_{calls}",
                            display_name=f"Product {calls}",
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

        base_config.max_catalog_searches_per_turn = 1
        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        capability_calls = []

        def capabilities_for_turn(**kwargs):
            capability_calls.append(kwargs)
            return CatalogCapabilities(
                catalog_id="fashion",
                retrieval_modes=["text"],
                filters={},
            )

        runtime._catalog_capabilities = SimpleNamespace(
            get=capabilities_for_turn
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
        search_tool = {fn.__name__: fn for fn in captured["tools"]}["search_catalog_tool"]

        first = search_tool(semantic_query="dress")
        second = search_tool(semantic_query="shoes")

        assert "PRODUCT_REF: prod_1" in first
        assert "STOP_TOOL_USE: Catalog search limit reached" in second
        assert calls == 1
        assert capability_calls == [{}]

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

    def test_cached_cart_images_are_rehydrated_from_product_refs(
        self,
        base_config,
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
        runtime._remember_products(
            identity,
            [
                ProductSummary(
                    product_id="prod_tote",
                    display_name="Linen Canvas Tote Bag",
                    price=Money(amount=59.99),
                    image_url="/images/Linen_Canvas_Tote_Bag.jpg",
                )
            ],
        )
        retrieved: dict[str, str] = {}
        cart = Cart(
            contents=[
                {
                    "cart_line_id": "Linen Canvas Tote Bag",
                    "product_id": "Linen Canvas Tote Bag",
                    "item": "Linen Canvas Tote Bag",
                    "amount": 1,
                    "price": 59.99,
                }
            ]
        )

        runtime._append_cached_cart_images(retrieved, cart, identity)

        assert retrieved == {
            "Linen Canvas Tote Bag": "/images/Linen_Canvas_Tote_Bag.jpg"
        }

    @pytest.mark.asyncio
    async def test_multi_intent_request_reaches_agent_without_pre_agent_mutation(
        self,
        base_config,
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
        captured = {}

        async def fake_analyze(state):
            return ""

        def fake_add_cart_item(request, memory_port):
            raise AssertionError("cart mutation must happen through agent tool calls")

        class FakeAgent:
            async def ainvoke(self, payload, config):
                captured["payload"] = payload
                captured["config"] = config
                return {
                    "messages": [
                        {
                            "content": (
                                "I can style this under $100, then add the selected "
                                "item once the tool has a valid product ref."
                            )
                        }
                    ]
                }

        def fake_create_agent(state, identity, turn_capabilities=None):
            captured["state_query"] = state.query
            return FakeAgent()

        monkeypatch.setattr(runtime._media_perception, "analyze", fake_analyze)
        monkeypatch.setattr(runtime, "_load_memory", lambda state, identity: None)
        monkeypatch.setattr(runtime, "_persist_context", lambda state, identity: None)
        monkeypatch.setattr(runtime, "_create_agent", fake_create_agent)
        monkeypatch.setattr(runtime_mod, "add_cart_item", fake_add_cart_item)

        state = State(
            user_id=111,
            query=(
                "Help me style this and keep it under budget 100 and if you do "
                "that add it to cart"
            ),
            guardrails=False,
        )

        output = await runtime._run_turn(state, identity)

        assert output.response.startswith("I can style this under $100")
        user_message = captured["payload"]["messages"][0]["content"]
        assert "USER QUERY: Help me style this" in user_message
        assert captured["config"]["configurable"]["thread_id"] == "conversation-a"

    def test_partial_product_results_response_is_grounded(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        state = State(
            user_id=111,
            query="style this",
            product_results=[
                {
                    "product_id": "prod_1",
                    "display_name": "Yonder Floral Maxi Dress",
                    "category": "dress",
                    "price": {"amount": 119.99, "currency": "USD"},
                }
            ],
        )

        response = runtime_mod._partial_product_results_response(state)

        assert "**Yonder Floral Maxi Dress** — dress — $119.99 USD" in response
        assert "overstate outdoor performance" in response
        assert "100% cotton" not in response
        assert "grass" not in response

    @pytest.mark.asyncio
    async def test_deepagents_failure_uses_partial_results_and_resets_thread(
        self,
        base_config,
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
        reset_threads: list[str] = []
        persisted: list[str] = []

        async def fake_analyze(state):
            return ""

        class FakeCheckpointer:
            def delete_thread(self, thread_id):
                reset_threads.append(thread_id)

        class FailingAgent:
            async def ainvoke(self, payload, config):
                raise RuntimeError("recursion limit")

        monkeypatch.setattr(runtime._media_perception, "analyze", fake_analyze)
        monkeypatch.setattr(runtime, "_load_memory", lambda state, identity: None)
        monkeypatch.setattr(
            runtime,
            "_persist_context",
            lambda state, identity: persisted.append(state.context),
        )
        monkeypatch.setattr(
            runtime,
            "_create_agent",
            lambda state, identity, turn_capabilities=None: FailingAgent(),
        )
        runtime._checkpointer = FakeCheckpointer()

        state = State(
            user_id=111,
            query="What shoes work?",
            guardrails=False,
            product_results=[
                {
                    "product_id": "prod_1",
                    "display_name": "Aimee Ankle Strap Sandals",
                    "category": "shoes",
                    "price": {"amount": 59.99, "currency": "USD"},
                }
            ],
        )

        output = await runtime._run_turn(state, identity)

        assert output.response.startswith("I found these grounded catalog options")
        assert "**Aimee Ankle Strap Sandals** — shoes — $59.99 USD" in output.response
        assert output.model_usage["app_llm"]["status"] == "failed"
        assert "deepagents_error" in output.timings
        assert reset_threads == ["conversation-a"]
        assert persisted and "Aimee Ankle Strap Sandals" in persisted[-1]

    @pytest.mark.asyncio
    async def test_grounding_rewrite_edits_internal_refs_and_surface_overclaims(
        self,
        base_config,
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
        captured: Dict[str, Any] = {}

        async def fake_analyze(state):
            return ""

        class FakeAgent:
            async def ainvoke(self, payload, config):
                return {
                    "messages": [
                        {
                            "role": "tool",
                            "content": (
                                "SEARCH_RESULT_GROUNDING_NOTE: Use search results "
                                "for candidate names and prices.\n"
                                "PRODUCT_REF: prod_sandal\n"
                                "NAME: Flat Strappy Black Sandals\n"
                                "PRICE: $49.90 USD\n"
                                "IMAGE_URL: /images/sandal.jpg\n"
                                "DETAILS: Call get_product_details_tool before "
                                "outdoor-practicality claims."
                            ),
                        },
                        {
                            "content": (
                                "The Flat Strappy Black Sandals (PRODUCT_REF: "
                                "prod_sandal) will not sink in grass or gravel and "
                                "will stay comfortable all evening."
                            ),
                            "usage_metadata": {
                                "input_tokens": 10,
                                "output_tokens": 8,
                                "total_tokens": 18,
                            },
                        },
                    ]
                }

        class FakeGroundingModel:
            def invoke(self, messages):
                captured["rewrite_messages"] = messages
                return SimpleNamespace(
                    content=(
                        "The **Flat Strappy Black Sandals** are $49.90 and are a "
                        "candidate for the outfit. I would avoid promising grass, "
                        "gravel, or all-evening comfort without product details."
                    ),
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 10,
                        "total_tokens": 22,
                    },
                )

        monkeypatch.setattr(runtime._media_perception, "analyze", fake_analyze)
        monkeypatch.setattr(runtime, "_load_memory", lambda state, identity: None)
        monkeypatch.setattr(runtime, "_persist_context", lambda state, identity: None)
        monkeypatch.setattr(
            runtime,
            "_create_agent",
            lambda state, identity, turn_capabilities=None: FakeAgent(),
        )
        monkeypatch.setattr(runtime, "_create_chat_model", lambda: FakeGroundingModel())

        state = State(
            user_id=111,
            query="Style practical sandals for an outdoor dinner.",
            guardrails=False,
        )

        output = await runtime._run_turn(state, identity)

        assert "PRODUCT_REF" not in output.response
        assert "will not sink" not in output.response
        assert "stay comfortable all evening" not in output.response
        assert output.response.startswith("The **Flat Strappy Black Sandals**")
        rewrite_prompt = captured["rewrite_messages"][1]["content"]
        assert "TOOL EVIDENCE:" in rewrite_prompt
        assert "DRAFT RESPONSE:" in rewrite_prompt
        evidence_section = rewrite_prompt.split("DRAFT RESPONSE:", 1)[0]
        assert "CUSTOMER_SAFE_SEARCH_EVIDENCE" in evidence_section
        assert "Flat Strappy Black Sandals | price: $49.90 USD | image: available" in (
            evidence_section
        )
        assert "PRODUCT_REF: prod_sandal" not in evidence_section
        assert "Call get_product_details_tool" not in evidence_section
        assert output.model_usage["app_llm_grounding_editor"]["status"] == "used"
        assert output.model_usage["app_llm_grounding_editor"]["calls"] == 1
        assert output.token_usage == {
            "input_tokens": 22,
            "output_tokens": 18,
            "total_tokens": 40,
            "model_calls": 2,
        }
        assert "grounding_rewrite" in output.timings

    @pytest.mark.asyncio
    async def test_grounding_rewrite_can_be_disabled(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        base_config.grounding_rewrite_enabled = False
        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        identity = runtime_mod.RequestIdentity(
            session_id="session-a",
            conversation_id="conversation-a",
            cart_id="cart-a",
            context_user_id=111,
            cart_user_id=222,
            request_id="request-a",
        )

        async def fake_analyze(state):
            return ""

        class FakeAgent:
            async def ainvoke(self, payload, config):
                return {
                    "messages": [
                        {"role": "tool", "content": "PRODUCT_REF: prod_sandal"},
                        {"content": "Draft with PRODUCT_REF: prod_sandal"},
                    ]
                }

        def fail_chat_model():
            raise AssertionError("grounding editor should not run")

        monkeypatch.setattr(runtime._media_perception, "analyze", fake_analyze)
        monkeypatch.setattr(runtime, "_load_memory", lambda state, identity: None)
        monkeypatch.setattr(runtime, "_persist_context", lambda state, identity: None)
        monkeypatch.setattr(
            runtime,
            "_create_agent",
            lambda state, identity, turn_capabilities=None: FakeAgent(),
        )
        monkeypatch.setattr(runtime, "_create_chat_model", fail_chat_model)

        output = await runtime._run_turn(
            State(user_id=111, query="hello", guardrails=False),
            identity,
        )

        assert output.response == "Draft with PRODUCT_REF: prod_sandal"
        assert "app_llm_grounding_editor" not in output.model_usage

    def test_collect_tool_grounding_evidence_uses_customer_safe_summary(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        result = {
            "messages": [
                {
                    "role": "tool",
                    "content": (
                        "PRODUCT_DETAIL_GROUNDING_NOTE: State only facts.\n"
                        "PRODUCT_REF: prod_skirt\n"
                        "NAME: Zephyr Linen Skirt\n"
                        "CATEGORY: skirt\n"
                        "PRICE: $39.99 USD\n"
                        "IMAGE_URL: /images/zephyr.jpg\n"
                        "DETAILS:\n"
                        "- care: Machine wash cold.\n"
                        "- composition: 100% linen\n"
                        "DESCRIPTION: 100% linen and breathable for all-day comfort."
                    ),
                }
            ]
        }

        evidence = runtime_mod._collect_tool_grounding_evidence(
            result,
            max_chars=12000,
        )

        assert "CUSTOMER_SAFE_PRODUCT_DETAIL_EVIDENCE" in evidence
        assert "Zephyr Linen Skirt | category: skirt | price: $39.99 USD" in evidence
        assert "image: available" in evidence
        assert "details: care: Machine wash cold.; composition: 100% linen" in evidence
        assert "PRODUCT_REF" not in evidence
        assert "DESCRIPTION" not in evidence
        assert "all-day comfort" not in evidence

    def test_collect_search_evidence_forbids_name_based_attribute_inference(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        result = {
            "messages": [
                {
                    "role": "tool",
                    "content": (
                        "SEARCH_RESULT_GROUNDING_NOTE: Use search results.\n"
                        "PRODUCT_REF: prod_ocean\n"
                        "NAME: Ocean Breeze Maxi Dress\n"
                        "CATEGORY: dress\n"
                        "PRICE: $189.99 USD\n"
                        "IMAGE_URL: /images/ocean.jpg\n"
                        "PRODUCT_REF: prod_gazelle\n"
                        "NAME: Gazelle Gingham Dress\n"
                        "CATEGORY: dress\n"
                        "PRICE: $149.99 USD"
                    ),
                }
            ]
        }

        evidence = runtime_mod._collect_tool_grounding_evidence(
            result,
            max_chars=12000,
        )

        assert "CUSTOMER_SAFE_SEARCH_EVIDENCE" in evidence
        assert "Treat names as display names, not attribute evidence" in evidence
        assert "group claims require product-detail evidence for every item" in evidence
        assert "Ocean Breeze Maxi Dress | category: dress | price: $189.99 USD" in (
            evidence
        )
        assert "Gazelle Gingham Dress | category: dress | price: $149.99 USD" in (
            evidence
        )

    def test_explicit_product_matching_allows_specific_abbreviated_names(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        sandals = ProductSummary(
            product_id="prod_sandals",
            display_name="Flat Strappy Black Sandals with Buckle Embellishment",
        )
        layer = ProductSummary(
            product_id="prod_layer",
            display_name="Gentle Meadow Blouse Sweater",
        )
        other = ProductSummary(
            product_id="prod_green",
            display_name="Green Meadow Sweater Top",
        )

        matches = runtime_mod._explicitly_named_products(
            (
                "Please add the Flat Strappy Sandals and Gentle Meadow Blouse "
                "Sweater to my cart."
            ),
            [sandals, layer, other],
        )

        assert [product.product_id for product in matches] == [
            "prod_layer",
            "prod_sandals",
        ]

    def test_scrub_internal_shopper_language_removes_tool_mechanics(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        scrubbed = runtime_mod._scrub_internal_shopper_language(
            (
                "The product detail tool doesn't return fabric composition, "
                "and the sandals weren't added because the tool requires an "
                "exact match."
            )
        )

        assert "tool" not in scrubbed.lower()
        assert "I don't have fabric composition" in scrubbed
        assert "because I need an exact match" in scrubbed

    def test_add_cart_items_tool_requires_cached_refs_and_batches_adds(
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

        def fake_tool(*, args_schema=None, return_direct: bool = False):
            def decorate(fn):
                fn.args_schema = args_schema
                fn.return_direct = return_direct
                return fn

            return decorate

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        added = []

        def fake_add_cart_item(request, memory_port):
            added.append(request)
            return CartMutationResult(ok=True, message="memory service message")

        deepagents_mod.GeneralPurposeSubagentProfile = FakeProfile
        deepagents_mod.HarnessProfile = FakeProfile
        deepagents_mod.create_deep_agent = fake_create_deep_agent
        deepagents_mod.register_harness_profile = lambda *args, **kwargs: None
        tools_mod.tool = fake_tool
        openai_mod.ChatOpenAI = FakeChatOpenAI

        monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)
        monkeypatch.setattr(runtime_mod, "add_cart_item", fake_add_cart_item)

        runtime = runtime_mod.DeepAgentsRuntime(base_config)
        runtime._catalog_capabilities = SimpleNamespace(
            get=lambda **_: CatalogCapabilities(
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
        product = ProductSummary(
            product_id="prod_flats",
            display_name="Felicity Flats",
            price=Money(amount=49.9),
        )
        bag = ProductSummary(
            product_id="prod_bag",
            display_name="Work Bag",
            price=Money(amount=59.0),
        )
        luminous = ProductSummary(
            product_id="prod_luminous",
            display_name="Luminous Lace Blouse Sweater",
            price=Money(amount=59.99),
        )
        green = ProductSummary(
            product_id="prod_green",
            display_name="Green Meadow Sweater Top",
            price=Money(amount=49.99),
        )
        active_products = {
            item.product_id: ProductDetail.model_validate(item.model_dump())
            for item in (product, bag, luminous, green)
        }
        def fake_product_details(request, *args, **kwargs):
            product_detail = active_products.get(request.product_id)
            if product_detail is not None:
                return GetProductDetailsResult(ok=True, product=product_detail)
            return GetProductDetailsResult(
                ok=False,
                error=CommerceError(
                    code="product_not_found",
                    message="not found",
                ),
            )

        monkeypatch.setattr(
            runtime_mod,
            "get_product_details",
            fake_product_details,
        )
        monkeypatch.setattr(
            runtime,
            "_read_cart",
            lambda user_id: Cart(
                contents=[
                    {
                        "cart_line_id": "line_flats",
                        "product_id": "prod_flats",
                        "item": "Felicity Flats",
                        "amount": 3,
                        "price": 49.9,
                    },
                    {
                        "cart_line_id": "line_bag",
                        "product_id": "prod_bag",
                        "item": "Work Bag",
                        "amount": 1,
                        "price": 59.0,
                    },
                ]
            ),
        )

        runtime._create_agent(State(user_id=111, query="hello"), identity)
        add_tool = {fn.__name__: fn for fn in captured["tools"]}["add_cart_items_tool"]

        missing = add_tool(items=[{"product_ref": "missing", "quantity": 1}])
        assert "Search the catalog first" in missing
        assert added == []

        runtime._remember_products(identity, [product, bag])
        added_response = add_tool(
            items=[
                {
                    "product_ref": "prod_flats",
                    "quantity": 2,
                    "expected_display_name": "Felicity Flats",
                },
                {"product_ref": "missing", "quantity": 1},
                {
                    "product_ref": "prod_bag",
                    "quantity": 1,
                    "expected_display_name": "Work Bag",
                },
                {"product_ref": "prod_flats", "quantity": 1},
            ]
        )

        assert "CART_ADD_RESULT" in added_response
        assert "Added:" in added_response
        assert "3 x Felicity Flats (PRODUCT_REF: prod_flats)" in added_response
        assert "1 x Work Bag (PRODUCT_REF: prod_bag)" in added_response
        assert "Failed:" in added_response
        assert "PRODUCT_REF 'missing'" in added_response
        assert "Cart total:" in added_response
        assert len(added) == 2
        assert added[0].product_id == "prod_flats"
        assert added[0].quantity == 3
        assert added[1].product_id == "prod_bag"
        assert added[1].quantity == 1

        added.clear()
        del active_products["prod_bag"]
        stale_response = add_tool(
            items=[
                {
                    "product_ref": "prod_bag",
                    "quantity": 1,
                    "expected_display_name": "Work Bag",
                }
            ]
        )
        assert "no longer present in the active catalog" in stale_response
        assert added == []

        active_products["prod_bag"] = ProductDetail(
            product_id="prod_bag",
            display_name="Different Product",
        )
        reused_id_response = add_tool(
            items=[
                {
                    "product_ref": "prod_bag",
                    "quantity": 1,
                    "expected_display_name": "Work Bag",
                }
            ]
        )
        assert "resolves to a different product" in reused_id_response
        assert added == []

        monkeypatch.setattr(
            runtime_mod,
            "get_product_details",
            lambda *args, **kwargs: GetProductDetailsResult(
                ok=False,
                error=CommerceError(
                    code="catalog_request_failed",
                    message="temporary",
                    retryable=True,
                ),
            ),
        )
        transient_response = add_tool(
            items=[
                {
                    "product_ref": "prod_flats",
                    "quantity": 1,
                    "expected_display_name": "Felicity Flats",
                }
            ]
        )
        assert "temporarily unavailable" in transient_response
        assert "no longer present" not in transient_response
        assert added == []

        monkeypatch.setattr(runtime_mod, "get_product_details", fake_product_details)

        runtime._remember_products(identity, [luminous, green])
        runtime._create_agent(
            State(
                user_id=111,
                query=(
                    "Could you add Felicity Flats and Luminous Lace Blouse "
                    "Sweater to my cart?"
                ),
            ),
            identity,
        )
        add_tool = {fn.__name__: fn for fn in captured["tools"]}[
            "add_cart_items_tool"
        ]
        blocked_response = add_tool(
            items=[
                {
                    "product_ref": "prod_flats",
                    "quantity": 1,
                    "expected_display_name": "Felicity Flats",
                },
                {
                    "product_ref": "prod_green",
                    "quantity": 1,
                    "expected_display_name": "Green Meadow Sweater Top",
                },
            ]
        )

        assert added == []
        assert "outside the current explicit add request" in blocked_response
        assert "Luminous Lace Blouse Sweater" in blocked_response
        assert "Green Meadow Sweater Top" in blocked_response

        mismatch_response = add_tool(
            items=[
                {
                    "product_ref": "prod_green",
                    "quantity": 1,
                    "expected_display_name": "Luminous Lace Blouse Sweater",
                }
            ]
        )
        assert added == []
        assert "expected 'Luminous Lace Blouse Sweater'" in mismatch_response
        assert "resolves to 'Green Meadow Sweater Top'" in mismatch_response

    def test_product_details_tool_reads_cached_product_ref(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        base_config.max_product_detail_reads_per_turn = 4
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

        def fake_tool(*, args_schema=None, return_direct: bool = False):
            def decorate(fn):
                fn.args_schema = args_schema
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
            get=lambda **_: CatalogCapabilities(
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
                    image_url="/images/work_bag.jpg",
                    attributes={"catalog_text": "Work Bag | structured tote | bag"},
                )
            ],
        )

        monkeypatch.setattr(
            runtime_mod,
            "get_product_details",
            lambda *args, **kwargs: GetProductDetailsResult(
                ok=True,
                product=ProductDetail(
                    product_id="prod_123",
                    display_name="Work Bag",
                    category="bag",
                    price=Money(amount=59.0),
                    image_url="/images/work_bag.jpg",
                    attributes={
                        "care": "Spot clean with a damp cloth.",
                        "composition": "patent leather",
                    },
                ),
            ),
        )
        state = State(user_id=111, query="tell me more")
        runtime._create_agent(state, identity)
        tools_by_name = {fn.__name__: fn for fn in captured["tools"]}

        details = tools_by_name["get_product_details_tool"]("prod_123")
        missing = tools_by_name["get_product_details_tool"]("Work Bag")

        assert "PRODUCT_DETAIL_GROUNDING_NOTE" in details
        assert "PRODUCT_REF: prod_123" in details
        assert "IMAGE_URL: /images/work_bag.jpg" in details
        assert "- care: Spot clean with a damp cloth." in details
        assert "- composition: patent leather" in details
        assert "structured tote" not in details
        assert state.retrieved == {"Work Bag": "/images/work_bag.jpg"}
        assert "No product with PRODUCT_REF 'Work Bag'" in missing

        monkeypatch.setattr(
            runtime_mod,
            "get_product_details",
            lambda *args, **kwargs: GetProductDetailsResult(
                ok=False,
                error=CommerceError(
                    code="catalog_request_failed",
                    message="temporary",
                    retryable=True,
                ),
            ),
        )
        transient = tools_by_name["get_product_details_tool"]("prod_123")
        assert "temporarily unavailable" in transient
        assert "no longer available" not in transient

    def test_product_details_tool_stops_after_read_budget(
        self,
        base_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        base_config.max_product_detail_reads_per_turn = 1
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

        def fake_tool(*, args_schema=None, return_direct: bool = False):
            def decorate(fn):
                fn.args_schema = args_schema
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
        monkeypatch.setattr(
            runtime_mod,
            "get_product_details",
            lambda request, *args, **kwargs: GetProductDetailsResult(
                ok=True,
                product=ProductDetail(
                    product_id=request.product_id,
                    display_name=(
                        "Skirt One"
                        if request.product_id == "prod_1"
                        else "Skirt Two"
                    ),
                    attributes={},
                ),
            ),
        )
        runtime._catalog_capabilities = SimpleNamespace(
            get=lambda **_: CatalogCapabilities(
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
                ProductSummary(product_id="prod_1", display_name="Skirt One"),
                ProductSummary(product_id="prod_2", display_name="Skirt Two"),
            ],
        )

        runtime._create_agent(State(user_id=111, query="compare these"), identity)
        detail_tool = {fn.__name__: fn for fn in captured["tools"]}[
            "get_product_details_tool"
        ]

        first = detail_tool("prod_1")
        second = detail_tool("prod_2")

        assert "PRODUCT_DETAIL_GROUNDING_NOTE" in first
        assert "STOP_TOOL_USE: Product-detail read limit reached" in second

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
        assert "structured tote" not in formatted
        assert "Call get_product_details_tool" in formatted

    def test_format_product_details_warns_against_performance_overclaims(self) -> None:
        from chain_server.src import deepagents_runtime as runtime_mod

        product = ProductDetail(
            product_id="prod_789",
            display_name="Outdoor Sandal",
            description="Rubber sole and ankle strap.",
            price=Money(amount=59.0),
            attributes={"sole": "rubber", "fastening": "ankle strap"},
        )

        formatted = runtime_mod._format_product_details(product)

        assert "PRODUCT_DETAIL_GROUNDING_NOTE" in formatted
        assert "- sole: rubber" in formatted
        assert "- fastening: ankle strap" in formatted
        assert "Rubber sole and ankle strap" not in formatted

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
