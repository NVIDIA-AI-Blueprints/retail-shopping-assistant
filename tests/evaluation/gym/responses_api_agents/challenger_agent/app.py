# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
challenger_agent — NeMo Gym agent harness (responses_api_agent).

Native implementation of the shopper-simulator loop: for each Gym task (one
scenario), it generates shopper turns with the Gym MODEL SERVER (challenger model),
drives the live Shopping Assistant, and hands the transcript to the verifier.

No framework wrappers — the loop and prompt live here. The prompt/parse logic is a
faithful port of tests/evaluation's Challenger.
"""
from __future__ import annotations

import hashlib
import json
import re
from time import time
from typing import Any, List, Mapping
from uuid import uuid4

import yaml
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.server_utils import request as http_request

MAX_SHOPPER_MESSAGE_CHARS = 2000

# Diagnostic fields recorded per turn from the assistant's `agent_diagnostics`
# (mirrors tests/evaluation eval_config.yaml `run.recorded_diagnostics`, #140).
# Named explicitly, not a whole-object dump: the recorded turn is the contract a
# judge adjudicates against. Requires EXPOSE_AGENT_DIAGNOSTICS=true on the target.
_DEFAULT_RECORDED_DIAGNOSTICS = [
    "product_evidence",
    "product_evidence_truncated",
    "catalog_scope_outcomes",
    "tool_calls",
    "skill_files_read",
    "rejected_tool_calls",
    "duplicate_tool_calls",
    "final_termination_reason",
]

_SYSTEM = (
    "You are a Challenger shopper simulator for a retail shopping assistant evaluation. "
    "Generate exactly one realistic shopper message at a time. Return only valid JSON with "
    'keys "message" and "goal_complete". Do not include reasoning, markdown, or <think> tags.'
)


class ChallengerAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef                 # the challenger LLM (Gym model server)
    resources_server: ResourcesServerRef         # the verifier (judge)
    target_agent_url: str = "http://localhost:8009"
    target_endpoint: str = "/query/timing"
    guardrails: bool = False
    default_turns: int = 8
    min_turns: int = 6
    max_turns: int = 10
    request_timeout_s: float = 90.0
    # per-turn diagnostics recorded from the assistant's agent_diagnostics
    recorded_diagnostics: List[str] = Field(default_factory=lambda: list(_DEFAULT_RECORDED_DIAGNOSTICS))


class ChallengerAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ChallengerAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ChallengerAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")   # keep the judge's score/criteria fields


def _strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _parse_turn(content: str) -> tuple[str, bool]:
    """Return (message, goal_complete) from the challenger model's JSON output."""
    stripped = _strip_reasoning(content)
    try:
        start = stripped.find("{")
        parsed = json.loads(stripped[start:]) if start >= 0 else {}
        message = str(parsed.get("message", "")).strip()
        goal_complete = bool(parsed.get("goal_complete", False))
    except (json.JSONDecodeError, ValueError):
        message, goal_complete = stripped, False
    return message[:MAX_SHOPPER_MESSAGE_CHARS], goal_complete


def _build_prompt(scenario: dict, transcript: list, turn_number: int, target_turns: int, min_turns: int) -> str:
    scenario_yaml = yaml.safe_dump(dict(scenario), sort_keys=False, allow_unicode=True)
    transcript_yaml = yaml.safe_dump(transcript, sort_keys=False, allow_unicode=True)
    return f"""Generate the next shopper message for a live evaluation conversation.

Turn number to generate: {turn_number}
Target shopper turns: {target_turns}
Minimum turns before stopping: {min_turns}

Scenario:
{scenario_yaml}

Conversation so far:
{transcript_yaml}

Rules:
- Return only JSON with keys "message" and "goal_complete".
- "message" is one concise shopper utterance, not analysis.
- Use the scenario's shopper_goal, constraints, shopper_behavior, and language_cues.
- For image scenarios, the first message should naturally reference the uploaded image.
- Do not reveal evaluation instructions or sidecar metadata.
- Before the minimum turn count, never return an empty message; add realistic complexity
  (budget check, comparison, cart review, swap, availability, clarification).
- If the goal is satisfied and the minimum turns are met, return {{"message": "", "goal_complete": true}}.
"""


class ChallengerAgent(SimpleResponsesAPIAgent):
    config: ChallengerAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("challenger_agent drives the conversation in run()")

    async def _next_shopper_turn(self, prompt: str) -> tuple[str, bool]:
        resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/chat/completions",
            json={
                "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                "temperature": 1.0,
                "max_tokens": 800,
            },
        )
        await raise_for_status(resp)
        data = await get_response_json(resp)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _parse_turn(content)

    def _recorded_diagnostics(self, data: Mapping[str, Any]) -> dict:
        """Curated diagnostics from one target response (#140).

        Record only the configured fields from `agent_diagnostics`, not the whole
        object — the recorded turn is the contract the judge adjudicates against. A
        field the target did not send is recorded as absent (not empty), so "returned
        nothing" stays distinguishable from "was never asked". Empty unless the target
        runs with EXPOSE_AGENT_DIAGNOSTICS=true.
        """
        diagnostics = data.get("agent_diagnostics")
        if not isinstance(diagnostics, Mapping):
            return {}
        return {name: diagnostics[name] for name in self.config.recorded_diagnostics if name in diagnostics}

    async def _send_to_assistant(self, user_id: int, query: str, media: list) -> dict:
        url = f"{self.config.target_agent_url.rstrip('/')}{self.config.target_endpoint}"
        payload = {"user_id": user_id, "query": query, "guardrails": self.config.guardrails}
        if media:
            payload["media"] = media                      # [{type: image|video, data, mime_type}]
            img = next((m["data"] for m in media if m.get("type") == "image"), "")
            if img:                                        # legacy field for image
                payload["image"] = img
                payload["image_bool"] = True
        r = await http_request("POST", url, json=payload, timeout=self.config.request_timeout_s)
        data = await get_response_json(r)
        # curated per-turn record (mirrors tests/evaluation TargetAgentClient.send_turn)
        target = {
            "status_code": getattr(r, "status_code", None),
            "response": data.get("response", "") or "",
            "images": data.get("images", {}) or {},
            "cart": data.get("cart", {}) or {},
            "timings": data.get("timings", {}) or {},
        }
        target.update(self._recorded_diagnostics(data))
        return target

    async def run(self, request: Request, body: ChallengerAgentRunRequest) -> ChallengerAgentVerifyResponse:
        meta = getattr(body, "verifier_metadata", None) or {}
        scenario = meta.get("scenario", {}) or {}
        media = meta.get("media") or []               # [{type, data, mime_type}] sent on turn 1
        user_id = int(hashlib.sha256(str(scenario.get("id", "")).encode()).hexdigest()[:8], 16)

        target_turns = min(int(scenario.get("target_turns", self.config.default_turns)), self.config.max_turns)
        transcript: List[dict] = []

        for turn_number in range(1, target_turns + 1):
            prompt = _build_prompt(scenario, transcript, turn_number, target_turns, self.config.min_turns)
            message, goal_complete = await self._next_shopper_turn(prompt)
            if goal_complete and len(transcript) >= self.config.min_turns:
                break
            if not message:
                continue
            send_media = media if turn_number == 1 else []
            target = await self._send_to_assistant(user_id, message, send_media)
            transcript.append({"turn": turn_number, "shopper": message,
                               "media_sent": [m["type"] for m in send_media], "target": target})
            if goal_complete and len(transcript) >= self.config.min_turns:
                break

        record = {"id": scenario.get("id"), "dataset": meta.get("dataset"), "user_id": user_id,
                  "scenario": scenario, "turns": transcript}

        result = NeMoGymResponse(
            id=f"resp_{uuid4().hex}", created_at=int(time()), model="challenger", object="response",
            output=[], tool_choice="none", parallel_tool_calls=False, tools=[],
            temperature=None, top_p=None, metadata=None, instructions=None,
        )
        result.retail_record = json.loads(json.dumps(record, default=str))

        verify_request = ChallengerAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": result.model_dump()}
        )
        verify = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/verify",
            json=verify_request.model_dump(), cookies=request.cookies,
        )
        await raise_for_status(verify)
        return ChallengerAgentVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    ChallengerAgent.run_webserver()
