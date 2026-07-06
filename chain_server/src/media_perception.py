# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM-backed media perception for user-attached shopping media."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from openai import OpenAI

from .agenttypes import State


logger = logging.getLogger(__name__)

MEDIA_ONLY_QUERY = "The user submitted visual media without additional text."


class MediaPerceptionClient:
    """Small adapter around the configured VLM endpoint."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.enabled = bool(
            getattr(config, "vlm_enabled", False)
            and getattr(config, "vlm_port", None)
            and getattr(config, "vlm_name", None)
        )
        self.model_name = getattr(config, "vlm_name", None)
        api_key_env = getattr(config, "vlm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        self.client = (
            OpenAI(base_url=getattr(config, "vlm_port"), api_key=api_key or "not-needed")
            if self.enabled
            else None
        )

    async def analyze(self, state: State) -> str:
        """Return compact structured media analysis for the current turn."""

        media = state.media or []
        if not media:
            return ""

        if not self.enabled or self.client is None or not self.model_name:
            return _disabled_analysis(media)

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=self._messages(state),
                temperature=0,
                max_tokens=1200,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - media perception is optional.
            logger.warning("VLM media perception failed: %s", exc)
            return _failed_analysis(exc)

        return _normalize_vlm_content(content)

    def _messages(self, state: State) -> list[dict[str, Any]]:
        prompt = _perception_prompt(state.query, state.context)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in state.media:
            media_type = str(item.get("type") or "").strip().lower()
            data = str(item.get("data") or "").strip()
            if not data:
                continue
            if media_type == "image":
                content.append({"type": "image_url", "image_url": {"url": data}})
            elif media_type == "video":
                content.append({"type": "video_url", "video_url": {"url": data}})

        return [
            {
                "role": "system",
                "content": (
                    "You are a visual media perception module for a fashion retail "
                    "shopping assistant. Return compact JSON only."
                ),
            },
            {"role": "user", "content": content},
        ]


def _perception_prompt(query: str, context: str) -> str:
    query_text = (query or "").strip()
    if not query_text or query_text == MEDIA_ONLY_QUERY:
        task = (
            "No user text was supplied. Study the attached media neutrally for a "
            "fashion retail assistant."
        )
    else:
        task = f"User request: {query_text}"

    recent = (context or "").strip()
    if len(recent) > 2000:
        recent = recent[-2000:]

    return f"""\
{task}

Recent conversation context:
{recent or "(none)"}

Focus on visible apparel, accessories, colors, materials or textures, silhouette,
style, occasion, and shopping-relevant attributes. Do not identify people or
infer sensitive traits. Separate visual observations from catalog facts; you do
not know catalog availability.

Return JSON with exactly these keys:
summary, fashion_items, style_terms, colors, materials_or_textures, occasion,
search_queries, constraints_detected, uncertainties, safety_notes.
"""


def _disabled_analysis(media: list[dict[str, Any]]) -> str:
    has_video = any(str(item.get("type") or "").lower() == "video" for item in media)
    if not has_video:
        return ""
    return json.dumps(
        {
            "summary": "Video was attached, but VLM media understanding is not configured.",
            "fashion_items": [],
            "style_terms": [],
            "colors": [],
            "materials_or_textures": [],
            "occasion": [],
            "search_queries": [],
            "constraints_detected": {},
            "uncertainties": ["Video understanding requires an enabled VLM."],
            "safety_notes": [],
        },
        sort_keys=True,
    )


def _failed_analysis(exc: Exception) -> str:
    message = str(exc)
    lower_message = message.lower()
    if "401" in message or "unauthorized" in lower_message or "authentication" in lower_message:
        summary = (
            "Media was attached, but the configured VLM could not authenticate. "
            "Video/image understanding is unavailable for this turn."
        )
        uncertainty = "VLM authentication failed."
    elif "access" in lower_message and "model" in lower_message:
        summary = (
            "Media was attached, but the configured key is not allowed to access "
            "the VLM. Video/image understanding is unavailable for this turn."
        )
        uncertainty = "VLM model access is not available for the configured key."
    else:
        summary = "Media was attached, but VLM media analysis failed."
        uncertainty = "VLM media analysis failed."

    return json.dumps(
        {
            "summary": summary,
            "fashion_items": [],
            "style_terms": [],
            "colors": [],
            "materials_or_textures": [],
            "occasion": [],
            "search_queries": [],
            "constraints_detected": {},
            "uncertainties": [uncertainty],
            "safety_notes": [],
        },
        sort_keys=True,
    )


def _normalize_vlm_content(content: str) -> str:
    cleaned = (content or "").strip()
    if not cleaned:
        return json.dumps(
            {
                "summary": "Media was attached, but the VLM returned no analysis.",
                "fashion_items": [],
                "style_terms": [],
                "colors": [],
                "materials_or_textures": [],
                "occasion": [],
                "search_queries": [],
                "constraints_detected": {},
                "uncertainties": ["VLM returned an empty response."],
                "safety_notes": [],
            },
            sort_keys=True,
        )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return json.dumps(
            {
                "summary": cleaned[:1000],
                "fashion_items": [],
                "style_terms": [],
                "colors": [],
                "materials_or_textures": [],
                "occasion": [],
                "search_queries": [],
                "constraints_detected": {},
                "uncertainties": ["VLM did not return structured JSON."],
                "safety_notes": [],
            },
            sort_keys=True,
        )

    if not isinstance(parsed, dict):
        parsed = {"summary": str(parsed)}
    return json.dumps(parsed, sort_keys=True)[:4000]
