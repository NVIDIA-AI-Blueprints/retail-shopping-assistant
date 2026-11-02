# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.summarizer``.

The summarizer is an LLM + HTTP agent: it condenses ``state.context`` when it
exceeds ``memory_length``, then POSTs the (possibly new) context back to the
memory service. Both surfaces are stubbed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from chain_server.src import summarizer as summarizer_mod
from chain_server.src.agenttypes import State
from chain_server.src.summarizer import SummaryAgent


@dataclass
class _PostRecord:
    """Capture of a single outgoing ``requests.post`` call."""

    url: str
    payload: Dict[str, Any]


@dataclass
class _PostRecorder:
    calls: List[_PostRecord] = field(default_factory=list)


@pytest.fixture
def summary_agent(base_config, monkeypatch: pytest.MonkeyPatch) -> SummaryAgent:
    class _FakeOpenAI:
        def __init__(self, *_, **__) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: None)
            )

    monkeypatch.setattr(summarizer_mod, "OpenAI", _FakeOpenAI)
    # memory_length is pinned small so we can exercise both branches with
    # short strings rather than 16KB fixtures.
    tiny_config = SimpleNamespace(
        llm_name=base_config.llm_name,
        llm_port=base_config.llm_port,
        memory_length=50,
        memory_port=base_config.memory_port,
    )
    return SummaryAgent(config=tiny_config)


@pytest.fixture
def post_recorder(monkeypatch: pytest.MonkeyPatch) -> _PostRecorder:
    recorder = _PostRecorder()

    class _FakeResponse:
        status_code = 200

        def json(self) -> Dict[str, Any]:
            return {"status": "ok"}

        def raise_for_status(self) -> None:
            return None

    def _fake_post(url: str, json: Dict[str, Any] | None = None, **_: Any):
        recorder.calls.append(_PostRecord(url=url, payload=dict(json or {})))
        return _FakeResponse()

    monkeypatch.setattr(summarizer_mod.requests, "post", _fake_post)
    return recorder


def _stub_tool_call_response(
    summary_agent: SummaryAgent, summary_text: str
) -> None:
    """Simulate the LLM returning a ``summarizer`` tool call."""
    tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="summarizer",
                arguments=json.dumps({"summary": summary_text}),
            )
        )
    ]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))
        ]
    )
    summary_agent.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )


def _stub_content_response(
    summary_agent: SummaryAgent, content: str | None
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ]
    )
    summary_agent.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )


class TestSummaryAgentInvoke:
    def test_short_context_skips_llm_and_persists_as_is(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        def _explode(**_: Any) -> Any:  # pragma: no cover - regression guard
            raise AssertionError("LLM must not be called for short context")

        summary_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_explode))
        )

        state = State(user_id=1, query="hi", context="short context")

        out = summary_agent.invoke(state, verbose=False)

        assert out.context == "short context"
        assert len(post_recorder.calls) == 1
        call = post_recorder.calls[0]
        assert call.url.endswith("/user/1/context/replace")
        assert call.payload == {"new_context": "short context"}

    def test_long_context_uses_llm_tool_call(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        long_context = "x" * 500  # exceeds memory_length=50
        _stub_tool_call_response(summary_agent, "summary-of-x-500")

        state = State(user_id=2, query="hi", context=long_context)
        out = summary_agent.invoke(state, verbose=False)

        assert out.context == "summary-of-x-500"
        assert len(post_recorder.calls) == 1
        assert post_recorder.calls[0].payload == {"new_context": "summary-of-x-500"}

    def test_fallback_json_parse_used_when_tool_calls_missing(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        long_context = "y" * 500
        fallback_content = json.dumps(
            {"name": "summarizer", "arguments": {"summary": "fallback-summary"}}
        )
        _stub_content_response(summary_agent, fallback_content)

        state = State(user_id=3, query="hi", context=long_context)
        out = summary_agent.invoke(state, verbose=False)

        assert out.context == "fallback-summary"
        assert post_recorder.calls[0].payload == {"new_context": "fallback-summary"}

    def test_unparseable_content_preserves_original_context(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        long_context = "z" * 500
        _stub_content_response(summary_agent, "just some freeform prose")

        state = State(user_id=4, query="hi", context=long_context)
        out = summary_agent.invoke(state, verbose=False)

        # Both branches prefer *keeping* the existing context to corrupting it
        # with raw model output.
        assert out.context == long_context
        assert post_recorder.calls[0].payload == {"new_context": long_context}

    def test_none_content_and_no_tool_calls_keeps_existing_context(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        long_context = "abcd" * 200
        _stub_content_response(summary_agent, None)

        state = State(user_id=5, query="hi", context=long_context)
        out = summary_agent.invoke(state, verbose=False)

        assert out.context == long_context
        assert post_recorder.calls[0].payload == {"new_context": long_context}

    def test_tool_call_without_summary_key_preserves_existing_context(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        long_context = "w" * 500
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="summarizer",
                    arguments=json.dumps({"other_field": "oops"}),
                )
            )
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, tool_calls=tool_calls)
                )
            ]
        )
        summary_agent.model = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
        )

        state = State(user_id=6, query="hi", context=long_context)
        out = summary_agent.invoke(state, verbose=False)

        # No ``summary`` -> summarizer falls back to the existing context.
        assert out.context == long_context

    def test_memory_port_url_is_constructed_per_user(
        self, summary_agent: SummaryAgent, post_recorder: _PostRecorder
    ) -> None:
        state = State(user_id=42, query="hi", context="ok")
        summary_agent.invoke(state, verbose=False)

        call = post_recorder.calls[0]
        assert call.url.startswith(summary_agent.memory_port)
        assert call.url.endswith("/user/42/context/replace")
