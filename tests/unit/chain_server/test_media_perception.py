# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chain_server.src.agenttypes import State
from chain_server.src.media_perception import MediaPerceptionClient


@pytest.mark.asyncio
async def test_disabled_vlm_returns_video_capability_note(base_config) -> None:
    config = SimpleNamespace(
        **{
            **base_config.__dict__,
            "vlm_enabled": False,
            "vlm_port": None,
            "vlm_name": None,
            "vlm_api_key_env": None,
        }
    )
    client = MediaPerceptionClient(config)
    state = State(
        user_id=1,
        query="",
        media=[
            {
                "type": "video",
                "mime_type": "video/mp4",
                "data": "data:video/mp4;base64,QUFB",
            }
        ],
    )

    analysis = json.loads(await client.analyze(state))

    assert "not configured" in analysis["summary"]
    assert analysis["search_queries"] == []


@pytest.mark.asyncio
async def test_disabled_vlm_leaves_image_only_path_empty(base_config) -> None:
    config = SimpleNamespace(
        **{
            **base_config.__dict__,
            "vlm_enabled": False,
            "vlm_port": None,
            "vlm_name": None,
            "vlm_api_key_env": None,
        }
    )
    client = MediaPerceptionClient(config)
    state = State(
        user_id=1,
        query="find this",
        image="data:image/jpeg;base64,QUFB",
        media=[
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": "data:image/jpeg;base64,QUFB",
            }
        ],
    )

    assert await client.analyze(state) == ""


@pytest.mark.asyncio
async def test_enabled_vlm_returns_normalized_json(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    from chain_server.src import media_perception as media_perception_mod

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(
                content=json.dumps(
                    {
                        "summary": "black strappy sandal",
                        "fashion_items": ["sandals"],
                        "style_terms": ["dressy"],
                        "colors": ["black"],
                        "materials_or_textures": [],
                        "occasion": [],
                        "search_queries": ["black strappy sandals"],
                        "constraints_detected": {},
                        "uncertainties": [],
                        "safety_notes": [],
                    }
                )
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(media_perception_mod, "OpenAI", FakeOpenAI)
    config = SimpleNamespace(
        **{
            **base_config.__dict__,
            "vlm_enabled": True,
            "vlm_port": "http://vlm.example/v1",
            "vlm_name": "vlm-model",
            "vlm_api_key_env": None,
        }
    )
    client = MediaPerceptionClient(config)
    state = State(
        user_id=1,
        query="find shoes like this",
        media=[
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": "data:image/jpeg;base64,QUFB",
            }
        ],
    )

    analysis = json.loads(await client.analyze(state))

    assert analysis["search_queries"] == ["black strappy sandals"]
    assert captured["model"] == "vlm-model"
    assert captured["messages"][1]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_enabled_vlm_auth_failure_returns_unavailable_analysis(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chain_server.src import media_perception as media_perception_mod

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("401 Unauthorized: Authentication failed")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(media_perception_mod, "OpenAI", FakeOpenAI)
    config = SimpleNamespace(
        **{
            **base_config.__dict__,
            "vlm_enabled": True,
            "vlm_port": "http://vlm.example/v1",
            "vlm_name": "vlm-model",
            "vlm_api_key_env": None,
        }
    )
    client = MediaPerceptionClient(config)
    state = State(
        user_id=1,
        query="find shoes like the video",
        media=[
            {
                "type": "video",
                "mime_type": "video/mp4",
                "data": "data:video/mp4;base64,QUFB",
            }
        ],
    )

    analysis = json.loads(await client.analyze(state))

    assert "could not authenticate" in analysis["summary"]
    assert analysis["search_queries"] == []
    assert analysis["uncertainties"] == ["VLM authentication failed."]
