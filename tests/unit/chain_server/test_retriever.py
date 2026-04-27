# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.retriever``.

Covers
------
- ``_normalize_numeric_filter`` / ``_normalize_filters`` (pure logic).
- ``_sanitize_categories`` (pure logic).
- ``_extract_retrieval_inputs`` with a stubbed OpenAI async client.
- ``invoke`` with both text and image paths, success and HTTP failure.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import requests

from chain_server.src import retriever as retriever_mod
from chain_server.src.agenttypes import State
from chain_server.src.retriever import RetrieverAgent


@pytest.fixture
def retriever_agent(base_config, monkeypatch: pytest.MonkeyPatch) -> RetrieverAgent:
    class _FakeOpenAI:
        def __init__(self, *_, **__) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: None)
            )

    monkeypatch.setattr(retriever_mod, "OpenAI", _FakeOpenAI)
    return RetrieverAgent(config=base_config)


# ---------------------------------------------------------------------------
# Pure helper methods
# ---------------------------------------------------------------------------


class TestNormalizeNumericFilter:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (10, 10.0),
            (10.5, 10.5),
            ("25", 25.0),
            ("$25", 25.0),
            ("1,250.50", 1250.5),
            ("  $19.99  ", 19.99),
            ("not a number", None),
            (["list"], None),
        ],
    )
    def test_coercion(self, value: Any, expected: float | None) -> None:
        result = RetrieverAgent._normalize_numeric_filter(value)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestNormalizeFilters:
    def test_keeps_positive_prices(self, retriever_agent: RetrieverAgent) -> None:
        assert retriever_agent._normalize_filters({"min_price": 20, "max_price": 100}) == {
            "min_price": 20.0,
            "max_price": 100.0,
        }

    def test_drops_non_positive_prices(self, retriever_agent: RetrieverAgent) -> None:
        assert retriever_agent._normalize_filters({"min_price": 0, "max_price": 0}) == {}

    def test_drops_both_when_min_exceeds_max(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        assert retriever_agent._normalize_filters({"min_price": 200, "max_price": 50}) == {}

    def test_ignores_unknown_keys(self, retriever_agent: RetrieverAgent) -> None:
        assert retriever_agent._normalize_filters(
            {"max_price": 100, "color": "red"}
        ) == {"max_price": 100.0}

    def test_empty_dict_returns_empty(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        assert retriever_agent._normalize_filters({}) == {}


class TestSanitizeCategories:
    def test_keeps_allowlisted_in_order_dedupe(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        result = retriever_agent._sanitize_categories(["bag", "dress", "bag", "shoes"])
        assert result == ["bag", "dress", "shoes"]

    def test_drops_unknown_categories(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        assert retriever_agent._sanitize_categories(["apparel", "clothing"]) == []

    def test_skips_non_strings_and_blank_values(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        result = retriever_agent._sanitize_categories(
            ["bag", "", "   ", None, 123, "dress"]  # type: ignore[list-item]
        )
        assert result == ["bag", "dress"]


# ---------------------------------------------------------------------------
# extract_retrieval_inputs
# ---------------------------------------------------------------------------


def _stub_extraction_response(
    retriever_agent: RetrieverAgent, arguments: Dict[str, Any]
) -> None:
    """Replace the retriever's OpenAI client with a tool-call response."""
    tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="extract_retrieval_inputs",
                arguments=json.dumps(arguments),
            )
        )
    ]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))
        ]
    )
    retriever_agent.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )


class TestExtractRetrievalInputs:
    async def test_empty_query_short_circuits(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        entities, categories, filters = await retriever_agent._extract_retrieval_inputs(
            State(user_id=1, query="")
        )

        # With no query we fall back to [] entities, full category list, and no filters.
        assert entities == []
        assert categories == retriever_agent.categories
        assert filters == {}

    async def test_tool_call_parsing_happy_path(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": ["summer dress"],
                "category_one": "dress",
                "category_two": "dress",
                "category_three": "dress",
                "max_price": 100,
            },
        )

        entities, categories, filters = await retriever_agent._extract_retrieval_inputs(
            State(user_id=1, query="show me a summer dress under $100")
        )

        assert entities == ["summer dress"]
        assert categories == ["dress"]  # dedup + allowlist
        assert filters == {"max_price": 100.0}

    async def test_extractor_prompt_rejects_generic_anything_entities(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        captured: Dict[str, Any] = {}
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="extract_retrieval_inputs",
                                    arguments=json.dumps(
                                        {
                                            "search_entities": [],
                                            "category_one": "dress",
                                            "category_two": "bag",
                                            "category_three": "shoes",
                                            "max_price": 100,
                                        }
                                    ),
                                )
                            )
                        ],
                    )
                )
            ]
        )

        def _create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return response

        retriever_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
        )

        entities, _categories, filters = await retriever_agent._extract_retrieval_inputs(
            State(user_id=1, query="show me anything under $100")
        )

        prompt = captured["messages"][0]["content"]
        assert "Do NOT use generic browse words" in prompt
        assert "show me anything under $100" in prompt
        assert entities == []
        assert filters == {"max_price": 100.0}

    async def test_invalid_categories_fall_back_to_allowlist(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": ["summer top"],
                "category_one": "apparel",
                "category_two": "clothing",
                "category_three": "apparel",
            },
        )

        entities, categories, filters = await retriever_agent._extract_retrieval_inputs(
            State(user_id=1, query="anything summer-y")
        )

        assert entities == ["summer top"]
        # All LLM categories were invented → fall back to agent's full list.
        assert categories == retriever_agent.categories
        assert filters == {}

    async def test_string_entity_list_gets_split(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": "['earrings', 'necklace']",
                "category_one": "earrings",
                "category_two": "necklace",
                "category_three": "necklace",
            },
        )

        entities, _, _ = await retriever_agent._extract_retrieval_inputs(
            State(user_id=1, query="accessories please")
        )

        assert entities == ["earrings", "necklace"]

    async def test_falls_back_to_content_when_no_tool_call(
        self, retriever_agent: RetrieverAgent
    ) -> None:
        # Model emits raw JSON content; the retriever must parse it.
        content = json.dumps(
            {
                "name": "extract_retrieval_inputs",
                "arguments": {
                    "search_entities": ["bag"],
                    "category_one": "bag",
                    "category_two": "bag",
                    "category_three": "bag",
                },
            }
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=None)
                )
            ]
        )
        retriever_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
        )

        entities, categories, _ = await retriever_agent._extract_retrieval_inputs(
            State(user_id=1, query="need a bag")
        )

        assert entities == ["bag"]
        assert categories == ["bag"]


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


class _FakeCatalogResponse:
    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _install_session_post(
    monkeypatch: pytest.MonkeyPatch,
    payload: Dict[str, Any] | None = None,
    *,
    raise_exc: Exception | None = None,
):
    """Stub ``requests.Session.post`` used inside RetrieverAgent.invoke."""
    captured: Dict[str, Any] = {}

    class _FakeSession:
        def mount(self, *_args, **_kwargs) -> None:
            return None

        def post(self, url: str, json: Dict[str, Any]):
            captured["url"] = url
            captured["json"] = json
            if raise_exc is not None:
                raise raise_exc
            return _FakeCatalogResponse(payload or {})

    monkeypatch.setattr(retriever_mod.requests, "Session", lambda: _FakeSession())
    return captured


class TestRetrieverInvoke:
    async def test_text_query_populates_state_with_results(
        self,
        retriever_agent: RetrieverAgent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": ["summer dress"],
                "category_one": "dress",
                "category_two": "dress",
                "category_three": "dress",
            },
        )
        captured = _install_session_post(
            monkeypatch,
            payload={
                "texts": ["Summer Dress | breezy | dress"],
                "names": ["Summer Dress"],
                "images": ["img1.jpg"],
                "similarities": [0.9],
            },
        )

        state = State(user_id=1, query="show me a summer dress")
        out = await retriever_agent.invoke(state, verbose=False)

        assert captured["url"].endswith("/query/text")
        # Filters omitted when none are requested.
        assert captured["json"]["text"] == ["summer dress"]
        assert captured["json"]["k"] == retriever_agent.k_value
        assert "Summer Dress" in out.response
        assert "Summer Dress" in out.retrieved
        assert out.retrieved["Summer Dress"] == "img1.jpg"
        assert "retriever_retrieval" in out.timings
        assert "retriever_categories" in out.timings

    async def test_image_query_uses_image_endpoint(
        self,
        retriever_agent: RetrieverAgent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": ["bag"],
                "category_one": "bag",
                "category_two": "bag",
                "category_three": "bag",
            },
        )
        captured = _install_session_post(
            monkeypatch,
            payload={
                "texts": ["Leather Bag | elegant | bag"],
                "names": ["Leather Bag"],
                "images": ["bag.jpg"],
                "similarities": [0.7],
            },
        )

        state = State(
            user_id=1,
            query="find a similar bag",
            image="data:image/jpeg;base64,AAAA",
        )
        out = await retriever_agent.invoke(state, verbose=False)

        assert captured["url"].endswith("/query/image")
        assert captured["json"]["image_base64"] == "data:image/jpeg;base64,AAAA"
        assert "Leather Bag" in out.response

    async def test_no_results_produces_polite_message(
        self,
        retriever_agent: RetrieverAgent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": ["watch"],
                "category_one": "bag",
                "category_two": "bag",
                "category_three": "bag",
            },
        )
        _install_session_post(
            monkeypatch,
            payload={
                "texts": [],
                "names": [],
                "images": [],
                "similarities": [],
            },
        )

        state = State(user_id=1, query="looking for a watch")
        out = await retriever_agent.invoke(state, verbose=False)

        assert "no products closely matching" in out.response.lower()

    async def test_http_error_sets_error_response(
        self,
        retriever_agent: RetrieverAgent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_extraction_response(
            retriever_agent,
            {
                "search_entities": ["bag"],
                "category_one": "bag",
                "category_two": "bag",
                "category_three": "bag",
            },
        )
        _install_session_post(
            monkeypatch,
            raise_exc=requests.exceptions.ConnectionError("boom"),
        )

        state = State(user_id=1, query="any bag")
        out = await retriever_agent.invoke(state, verbose=False)

        assert "encountered an error" in out.response.lower()
