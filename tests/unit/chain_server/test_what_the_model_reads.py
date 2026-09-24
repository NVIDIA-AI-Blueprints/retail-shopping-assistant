# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything the agent is built with that reaches the model, byte for byte.

Tool names, order, descriptions and schemas, the system prompt, and the text
the middleware carries. The model runs at temperature 0 and several of its
choices are near ties, so any byte of this can change which tools a turn
calls -- whitespace included. A change here is a behaviour change: refresh the
snapshot with REFRESH_MODEL_FACING_SNAPSHOT=1 and replay before merging it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chain_server.src.agenttypes import DialogueTurn, State
from chain_server.src.runtime.identity import RequestIdentity
from chain_server.src.runtime.runtime import DeepAgentsRuntime
from chain_server.src.weather import WeatherConfig

SNAPSHOT = Path(__file__).with_name("model_facing_snapshot.json")


def _built_agent(base_config: Any) -> dict[str, Any]:
    base_config.weather = WeatherConfig(enabled=True)
    runtime = DeepAgentsRuntime(base_config)
    identity = RequestIdentity(
        request_id="r1",
        session_id="s1",
        conversation_id="c1",
        cart_id="cart1",
        context_user_id=1,
        cart_user_id=1,
    )
    state = State(
        user_id=1,
        query="We're going to Italy at the weekend, what should I wear?",
        dialogue=[
            DialogueTurn(sequence=1, shopper_text="hi", assistant_text="hello")
        ],
    )
    captured: dict[str, Any] = {}
    with patch("deepagents.create_deep_agent", lambda **kw: captured.update(kw)):
        runtime._create_agent(state, identity)
    return captured


def _tool_record(tool: Any) -> dict[str, Any]:
    schema = tool.tool_call_schema
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    return {
        "name": tool.name,
        "description": tool.description,
        "schema": schema,
        "return_direct": tool.return_direct,
        "response_format": getattr(tool, "response_format", None),
        "handles_validation_error": (
            getattr(tool, "handle_validation_error", None) is not None
        ),
    }


def _strings(value: Any) -> Any:
    """The text inside a value; objects without text are not model-facing."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        kept = {str(k): _strings(v) for k, v in value.items()}
        return {k: v for k, v in kept.items() if v is not None}
    if isinstance(value, (list, tuple)):
        kept = [_strings(v) for v in value]
        return [v for v in kept if v is not None] or None
    return None


def _model_facing(captured: dict[str, Any]) -> dict[str, Any]:
    middleware = {}
    for layer in captured.get("middleware", []):
        attrs = {k: _strings(v) for k, v in sorted(vars(layer).items())}
        middleware[type(layer).__name__] = {
            k: v for k, v in attrs.items() if v not in (None, {}, [])
        }
    return {
        "tool_order": [tool.name for tool in captured["tools"]],
        "tools": [_tool_record(tool) for tool in captured["tools"]],
        "system_prompt": captured.get("system_prompt"),
        "kwargs": sorted(captured),
        "middleware": middleware,
    }


def test_what_the_model_reads_is_the_snapshot(base_config: Any) -> None:
    current = json.dumps(
        _model_facing(_built_agent(base_config)), indent=1, sort_keys=True
    ) + "\n"
    if os.environ.get("REFRESH_MODEL_FACING_SNAPSHOT") == "1":
        SNAPSHOT.write_text(current)
    assert current == SNAPSHOT.read_text(), (
        "What the model reads changed. If that is intended, refresh with "
        "REFRESH_MODEL_FACING_SNAPSHOT=1 and replay before merging."
    )
