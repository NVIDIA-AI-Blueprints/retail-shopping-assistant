# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracing must record what the turn did without changing what the turn does.

Two failures these cover are both silent. Reading the redacted diagnostics view
traces an empty object in every default deployment -- the exporter works, the
spans arrive, and every one of them is blank. And a tracing error that escapes
turns an observability feature into an outage.
"""

import json
from types import SimpleNamespace

from chain_server.src.agenttypes import State
from chain_server.src.deepagents_runtime import (
    _record_turn_diagnostics,
    _turn_span,
    _turn_trace_session,
)
from chain_server.src.turn_support import RequestIdentity


def _identity() -> RequestIdentity:
    return RequestIdentity(
        session_id="session-a",
        conversation_id="conversation-a",
        cart_id="cart-a",
        context_user_id=7,
        cart_user_id=7,
        request_id="request-a",
    )


class _RecordingSpan:
    """A span that remembers what was set on it."""

    def __init__(self) -> None:
        self.attributes: dict = {}

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value


class _RaisingSpan:
    def set_attribute(self, key: str, value) -> None:
        raise RuntimeError("exporter is unhappy")


def _state_with_diagnostics(**overrides) -> State:
    state = State(user_id=7, query="show me a black dress", guardrails=False)
    diagnostics = {
        "skill_files_read": ["/shopper/product-discovery/SKILL.md"],
        "tool_calls": [
            {"sequence": 1, "tool_name": "activate_shopper_skills_tool", "status": "completed"},
            {"sequence": 2, "tool_name": "search_catalog_tool", "status": "rejected"},
        ],
        "rejected_tool_calls": [2],
        "duplicate_tool_calls": [],
        "product_evidence": [{"product_ref": "p1"}, {"product_ref": "p2"}],
        "product_evidence_truncated": False,
        "catalog_scope_outcomes": [{"outcome": "zero_results"}],
        "final_termination_reason": "completed",
        "partial_graph_messages": [{"type": "ai", "content": "x" * 5000}],
    }
    diagnostics.update(overrides)
    state.agent_diagnostics = diagnostics
    return state


def _metadata(span: _RecordingSpan) -> dict:
    return json.loads(span.attributes["metadata"])


def test_diagnostics_are_read_from_state_not_from_the_redacted_view() -> None:
    """`_exposed_agent_diagnostics` returns {} unless the operator flag is on.

    That flag is false in docker-compose, so a tracer reading it would export a
    perfectly healthy span carrying nothing, in every normal deployment.
    """

    span = _RecordingSpan()
    state = _state_with_diagnostics()

    _record_turn_diagnostics(span, state)

    metadata = _metadata(span)
    assert metadata["termination_reason"] == "completed"
    assert metadata["tools"] == [
        "activate_shopper_skills_tool",
        "search_catalog_tool",
    ]


def test_the_scalar_counts_match_the_blob() -> None:
    span = _RecordingSpan()

    _record_turn_diagnostics(span, _state_with_diagnostics())

    metadata = _metadata(span)
    assert metadata["tool_calls"] == 2
    assert metadata["tool_calls_rejected"] == 1
    assert metadata["products_shown"] == 2
    assert metadata["zero_result_scopes"] == 1
    assert metadata["skills"] == ["/shopper/product-discovery/SKILL.md"]


def test_partial_graph_messages_are_not_attached() -> None:
    """It is the largest field, and the span tree already says what it says."""

    span = _RecordingSpan()

    _record_turn_diagnostics(span, _state_with_diagnostics())

    diagnostics = json.loads(_metadata(span)["diagnostics_json"])
    assert "partial_graph_messages" not in diagnostics
    assert "skill_files_read" in diagnostics


def test_the_full_blob_is_one_attribute_not_a_hundred() -> None:
    """A viewer flattens nested metadata into one attribute per leaf.

    Left nested, the diagnostics explode into roughly 120 keys and the few
    fields worth reading sort to the bottom of a wall of them.
    """

    span = _RecordingSpan()

    _record_turn_diagnostics(span, _state_with_diagnostics())

    metadata = _metadata(span)
    assert isinstance(metadata["diagnostics_json"], str)
    # The summary stays flat and readable; only the blob is serialised.
    assert set(metadata) == {
        "termination_reason",
        "tool_calls",
        "tool_calls_rejected",
        "products_shown",
        "zero_result_scopes",
        "skills",
        "tools",
        "diagnostics_json",
    }


def test_a_raising_span_does_not_reach_the_turn() -> None:
    """Observability never changes turn behaviour, including by failing."""

    _record_turn_diagnostics(_RaisingSpan(), _state_with_diagnostics())


def test_a_turn_with_no_diagnostics_still_records() -> None:
    state = State(user_id=7, query="hello", guardrails=False)
    state.agent_diagnostics = {}
    span = _RecordingSpan()

    _record_turn_diagnostics(span, state)

    metadata = _metadata(span)
    assert metadata["tool_calls"] == 0
    assert metadata["termination_reason"] is None


def test_recording_against_no_span_is_a_no_op() -> None:
    _record_turn_diagnostics(None, _state_with_diagnostics())


def test_the_session_is_the_conversation_not_the_graph_thread() -> None:
    """The checkpoint thread is [conversation_id, request_id] -- per turn.

    A tracer left to infer a session from it files a twenty-turn conversation
    as twenty sessions, which is the opposite of the point.
    """

    identity = _identity()
    assert identity.checkpoint_thread_id != identity.conversation_id

    captured: dict = {}

    class _Recorder:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_using_attributes(**kwargs):
        captured.update(kwargs)
        return _Recorder()

    import chain_server.src.deepagents_runtime as runtime
    import sys

    module = SimpleNamespace(using_attributes=fake_using_attributes)
    saved = sys.modules.get("openinference.instrumentation")
    sys.modules["openinference.instrumentation"] = module
    try:
        with runtime._turn_trace_session(identity):
            pass
    finally:
        if saved is None:
            sys.modules.pop("openinference.instrumentation", None)
        else:
            sys.modules["openinference.instrumentation"] = saved

    assert captured["session_id"] == "conversation-a"
    assert captured["user_id"] == "7"


def test_tracing_absent_is_a_null_context_not_an_error() -> None:
    """A deployment that never exports a span pays nothing and breaks nothing."""

    with _turn_trace_session(_identity()):
        pass
    with _turn_span(_identity()):
        pass
