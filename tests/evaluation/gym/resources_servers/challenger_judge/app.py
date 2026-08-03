# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
challenger_judge — NeMo Gym verifier (resources server).

Native: applies the judge_rules.md rubric with the Gym judge model server. No
framework wrapper. Gym only needs `reward`, so:
    reward = 1.0 if pass else 0.0
and the 1-5 score, 11 criteria, and critical_failures ride along as extra fields.

When judging is disabled (default, matching eval_config), verify() passes the
transcript through with a neutral reward.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import get_response_json

_RULES_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "judge_rules.md")
)


class ChallengerJudgeConfig(BaseResourcesServerConfig):
    name: str = "challenger_judge"
    judge_model_server: Optional[ModelServerRef] = None
    judge_enabled: bool | str = False          # env resolves to the string "true"/"false"
    max_tokens: int = 2048

    @property
    def judging_on(self) -> bool:
        return str(self.judge_enabled).strip().lower() in ("1", "true", "yes", "on")


class ChallengerJudgeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ChallengerJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    score: Optional[int] = None
    passed: Optional[bool] = None
    criteria: Optional[dict] = None
    critical_failures: Optional[list] = None


def _strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _parse_judgment(text: str) -> dict:
    s = _strip_reasoning(text)
    start = s.find("{")
    return json.loads(s[start:]) if start >= 0 else {}


class ChallengerJudgeResourcesServer(SimpleResourcesServer):
    config: ChallengerJudgeConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rules = ""
        if os.path.exists(_RULES_PATH):
            self._rules = open(_RULES_PATH).read()

    async def verify(self, body: ChallengerJudgeVerifyRequest) -> ChallengerJudgeVerifyResponse:
        payload = body.model_dump()

        if not self.config.judging_on or self.config.judge_model_server is None:
            return ChallengerJudgeVerifyResponse(**payload, reward=0.0)

        record = (getattr(body, "response", None) or {})
        if hasattr(record, "get"):
            record = record.get("retail_record", record)
        transcript = json.dumps(record, default=str)[:20000]

        prompt = (
            f"{self._rules}\n\n--- CONVERSATION RECORD ---\n{transcript}\n\n"
            "Score this scenario per the rules above. Output ONLY JSON: "
            '{"score": <1-5>, "pass": <bool>, "reason": "...", '
            '"criteria": {...}, "critical_failures": [...]}.'
        )
        resp = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                  "max_tokens": self.config.max_tokens},
        )
        data = await get_response_json(resp)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        try:
            j = _parse_judgment(content)
        except (json.JSONDecodeError, ValueError):
            j = {}

        passed = bool(j.get("pass", False))
        return ChallengerJudgeVerifyResponse(
            **payload,
            reward=1.0 if passed else 0.0,
            score=j.get("score"),
            passed=passed,
            criteria=j.get("criteria"),
            critical_failures=j.get("critical_failures"),
        )


if __name__ == "__main__":
    ChallengerJudgeResourcesServer.run_webserver()
