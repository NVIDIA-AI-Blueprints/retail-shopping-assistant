# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.chatter``.

Splits into two groups:

1. ``_format_*`` helpers that render the grounded prompt blocks.
2. ``invoke`` with an async-iterator stand-in for the OpenAI streaming
   response, using ``stream_writer_capture`` to observe emitted events.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import AsyncIterator, Iterable, List

import pytest

from chain_server.src import chatter as chatter_mod
from chain_server.src.agenttypes import Cart, State
from chain_server.src.chatter import ChatterAgent


@pytest.fixture
def chatter_agent(base_config, monkeypatch: pytest.MonkeyPatch) -> ChatterAgent:
    class _FakeAsyncOpenAI:
        def __init__(self, *_, **__) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

    monkeypatch.setattr(chatter_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    return ChatterAgent(config=base_config)


# ---------------------------------------------------------------------------
# Pure formatting helpers
# ---------------------------------------------------------------------------


class TestFormatCart:
    def test_empty_cart_returns_placeholder(self) -> None:
        state = State(user_id=1, query="q")
        assert ChatterAgent._format_cart(state) == "(empty)"

    def test_renders_line_per_item_with_price(self) -> None:
        state = State(
            user_id=1,
            query="q",
            cart=Cart(
                contents=[
                    {"item": "Silk Dress", "amount": 2, "price": 49.99},
                    {"item": "Leather Bag", "amount": 1, "price": 199.0},
                ]
            ),
        )

        rendered = ChatterAgent._format_cart(state)

        assert "- 2 x Silk Dress @ $49.99" in rendered
        assert "- 1 x Leather Bag @ $199.00" in rendered

    def test_renders_line_without_price_when_missing(self) -> None:
        state = State(
            user_id=1,
            query="q",
            cart=Cart(contents=[{"item": "Hat", "amount": 1}]),
        )

        rendered = ChatterAgent._format_cart(state)
        assert rendered == "- 1 x Hat"

    def test_handles_non_numeric_price_gracefully(self) -> None:
        state = State(
            user_id=1,
            query="q",
            cart=Cart(
                contents=[{"item": "Weird Thing", "amount": 1, "price": "NaN$$"}]
            ),
        )

        rendered = ChatterAgent._format_cart(state)
        # Non-coercible price collapses to the no-price branch rather than raising.
        assert rendered == "- 1 x Weird Thing"


class TestFormatAvailableCatalog:
    def test_empty_retrieved_returns_placeholder(self) -> None:
        state = State(user_id=1, query="q")
        assert (
            ChatterAgent._format_available_catalog(state)
            == "(no fresh catalog results this turn)"
        )

    def test_lists_names_one_per_line(self) -> None:
        state = State(
            user_id=1,
            query="q",
            retrieved={"Silk Dress": "a.jpg", "Hiking Boot": "b.jpg"},
        )

        out = ChatterAgent._format_available_catalog(state)
        assert "- Silk Dress" in out
        assert "- Hiking Boot" in out


class TestDescribePrecedingAgent:
    @pytest.mark.parametrize(
        "next_agent,expected",
        [
            ("cart", "cart"),
            ("CART", "cart"),
            ("retriever", "retriever"),
            ("", "none"),
            ("chatter", "none"),
            ("unknown", "none"),
        ],
    )
    def test_mapping(self, next_agent: str, expected: str) -> None:
        state = State(user_id=1, query="q", next_agent=next_agent)
        assert ChatterAgent._describe_preceding_agent(state) == expected


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


class _FakeStream:
    """Minimal async iterator emitting chat-completion-like delta chunks."""

    def __init__(self, pieces: Iterable[str]) -> None:
        self._pieces: List[str] = list(pieces)

    def __aiter__(self) -> AsyncIterator["_FakeStream"]:
        return self

    async def __anext__(self):
        if not self._pieces:
            raise StopAsyncIteration
        content = self._pieces.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
        )


def _install_async_stream(
    chatter_agent: ChatterAgent, pieces: Iterable[str]
) -> None:
    async def _create(**_: object) -> _FakeStream:
        return _FakeStream(pieces)

    chatter_agent.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )


class TestChatterInvoke:
    async def test_invoke_streams_content_updates_state(
        self,
        chatter_agent: ChatterAgent,
        stream_writer_capture: list,
    ) -> None:
        _install_async_stream(chatter_agent, ["Hello ", "world", "!"])
        state = State(
            user_id=1,
            query="hi there",
            retrieved={"Silk Dress": "a.jpg"},
        )

        out = await chatter_agent.invoke(state, verbose=False)

        assert out.response == "Hello world!"
        assert out.context.endswith("Hello world!")
        assert "chatter" in out.timings
        assert "first_token" in out.timings

        # The first emitted payload is the images frame.
        first_payload = json.loads(stream_writer_capture[0])
        assert first_payload["type"] == "images"
        assert first_payload["payload"] == {"Silk Dress": "a.jpg"}

        # Every subsequent event is a content frame.
        content_frames = [
            json.loads(ev) for ev in stream_writer_capture[1:]
        ]
        assert [frame["type"] for frame in content_frames] == ["content"] * 3
        assert [frame["payload"] for frame in content_frames] == [
            "Hello ",
            "world",
            "!",
        ]

    async def test_invoke_handles_empty_stream_gracefully(
        self,
        chatter_agent: ChatterAgent,
        stream_writer_capture: list,
    ) -> None:
        _install_async_stream(chatter_agent, [])
        state = State(user_id=1, query="hello")

        out = await chatter_agent.invoke(state, verbose=False)

        assert out.response == ""
        # No timings for first_token since nothing streamed.
        assert "first_token" not in out.timings

    async def test_invoke_skips_empty_delta_chunks(
        self,
        chatter_agent: ChatterAgent,
        stream_writer_capture: list,
    ) -> None:
        # ``None`` content chunks arrive when only tool deltas are in flight;
        # the chatter must ignore them without crashing.
        class _NullAwareStream(_FakeStream):
            def __init__(self) -> None:
                super().__init__([None, "Hi", None, "!"])

        async def _create(**_: object) -> _NullAwareStream:
            return _NullAwareStream()

        chatter_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
        )

        state = State(user_id=1, query="hello")
        out = await chatter_agent.invoke(state, verbose=False)

        assert out.response == "Hi!"

    async def test_invoke_injects_image_query_placeholder(
        self,
        chatter_agent: ChatterAgent,
        stream_writer_capture: list,
    ) -> None:
        # When the user submits only an image, the chatter prompt should
        # still contain a coherent USER QUERY block even though state.query
        # is empty. The simplest thing to verify is that the async create
        # receives a messages list with a user message that includes the
        # placeholder text.
        captured: dict = {}

        async def _create(**kwargs):  # noqa: ANN003 - signature mirrors openai
            captured.update(kwargs)
            return _FakeStream(["ok"])

        chatter_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
        )

        state = State(user_id=1, query="")
        out = await chatter_agent.invoke(state, verbose=False)

        assert out.response == "ok"
        user_message = next(
            m for m in captured["messages"] if m["role"] == "user"
        )
        assert "image" in user_message["content"].lower()
