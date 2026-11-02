# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""While tools run, the browser hears what the turn is doing, in shopper words."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

from orchestrator.src.runtime.streaming import (
    _TOOL_PROGRESS_LABELS,
    CHECKING_MESSAGE_LABEL,
    CHECKING_REPLY_LABEL,
    WRITING_REPLY_LABEL,
    ToolProgressMiddleware,
    _emit_progress_label,
    media_progress_label,
)

SNAPSHOT = Path(__file__).with_name("model_facing_snapshot.json")


def _call(name: str) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": name, "args": {}, "id": "call-1"})


def _labels(events: list[str]) -> list[str]:
    return [json.loads(event)["payload"]["label"] for event in events]


def test_each_allowed_call_is_announced_once_and_passes_through() -> None:
    async def scenario() -> tuple[list[str], list[object]]:
        events: list[str] = []
        middleware = ToolProgressMiddleware(events.append)
        results = [
            await middleware.awrap_tool_call(_call("search_catalog_tool"), _async_result("a")),
            await middleware.awrap_tool_call(_call("search_catalog_tool"), _async_result("b")),
            middleware.wrap_tool_call(_call("add_cart_items_tool"), lambda request: "c"),
            await middleware.awrap_tool_call(_call("activate_shopper_skills_tool"), _async_result("d")),
        ]
        await asyncio.sleep(0)
        return events, results

    events, results = asyncio.run(scenario())

    assert results == ["a", "b", "c", "d"]
    assert _labels(events) == ["Searching the catalog", "Updating your cart"]
    assert all(json.loads(event)["type"] == "progress" for event in events)


def test_no_tool_name_reaches_the_browser() -> None:
    for label in [
        *_TOOL_PROGRESS_LABELS.values(),
        WRITING_REPLY_LABEL,
        CHECKING_MESSAGE_LABEL,
        CHECKING_REPLY_LABEL,
    ]:
        assert "_" not in label and "tool" not in label.lower()


def test_every_label_names_a_tool_the_model_can_call() -> None:
    model_tools = set(re.findall(r'"name": "([a-z_]+_tool)"', SNAPSHOT.read_text()))

    assert set(_TOOL_PROGRESS_LABELS) <= model_tools
    assert model_tools - set(_TOOL_PROGRESS_LABELS) == {"activate_shopper_skills_tool"}


def test_perception_names_what_the_shopper_shared() -> None:
    image, video = {"type": "image"}, {"type": "video"}

    assert media_progress_label([image]) == "Looking at your photo"
    assert media_progress_label([image, image]) == "Looking at your photos"
    assert media_progress_label([video]) == "Looking at your video"
    assert media_progress_label([image, video]) == "Looking at what you shared"


def test_a_failing_listener_never_breaks_the_turn() -> None:
    def broken(_: str) -> None:
        raise RuntimeError("queue closed")

    _emit_progress_label(broken, WRITING_REPLY_LABEL)
    _emit_progress_label(None, WRITING_REPLY_LABEL)


def _async_result(value: str):
    async def handler(request: object) -> str:
        return value

    return handler
