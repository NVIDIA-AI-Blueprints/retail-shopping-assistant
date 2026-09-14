# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral guardrail boundary for shopper turns.

Only typed decisions cross this module. Request and response bodies are never
logged or attached to traces; diagnostics are deliberately limited to policy,
modality, status, latency, and a sanitized error code.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import httpx
from opentelemetry import trace
from pydantic import BaseModel, Field, ValidationError

GuardrailStage = Literal["input", "output"]
GuardrailStatus = Literal["allow", "block", "error"]


def stops_turn(
    decision: GuardrailDecision, failure_mode: Literal["open", "closed"]
) -> bool:
    """Apply the deployment failure policy without changing the provider result."""

    return decision.status == "block" or (
        decision.status == "error" and failure_mode == "closed"
    )


class GuardrailAttachment(BaseModel):
    type: Literal["image", "video"]
    data: str
    mime_type: str
    filename: str = ""


class GuardrailConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GuardrailDecision(BaseModel):
    status: GuardrailStatus
    stage: GuardrailStage
    policy: str = "configured"
    violated_categories: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    diagnostic_code: str | None = None
    modalities: list[Literal["text", "image", "video"]] = Field(default_factory=list)
    model_calls: dict[str, int] = Field(default_factory=dict)


class GuardrailProvider(Protocol):
    async def check_input(
        self,
        *,
        text: str,
        media: Sequence[dict[str, Any]],
        conversation: Sequence[dict[str, str]] = (),
    ) -> GuardrailDecision: ...

    async def check_output(self, *, text: str) -> GuardrailDecision: ...


class GuardrailServiceClient:
    """HTTP client for the independently deployed guardrail service."""

    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/checks"
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._tracer = trace.get_tracer(__name__)

    async def check_input(
        self,
        *,
        text: str,
        media: Sequence[dict[str, Any]],
        conversation: Sequence[dict[str, str]] = (),
    ) -> GuardrailDecision:
        attachments = [GuardrailAttachment.model_validate(item) for item in media]
        messages = [
            GuardrailConversationMessage.model_validate(item) for item in conversation
        ]
        return await self._check(
            stage="input",
            shopper_text=text,
            attachments=attachments,
            conversation=messages,
        )

    async def check_output(self, *, text: str) -> GuardrailDecision:
        return await self._check(stage="output", assistant_text=text)

    async def _check(
        self,
        *,
        stage: GuardrailStage,
        shopper_text: str = "",
        assistant_text: str = "",
        attachments: Sequence[GuardrailAttachment] = (),
        conversation: Sequence[GuardrailConversationMessage] = (),
    ) -> GuardrailDecision:
        started = time.monotonic()
        modalities = (["text"] if shopper_text or assistant_text else []) + [
            item.type for item in attachments
        ]
        code: str | None = None
        try:
            response = await self._client.post(
                self._url,
                json={
                    "stage": stage,
                    "shopper_text": shopper_text,
                    "attachments": [item.model_dump() for item in attachments],
                    "assistant_text": assistant_text,
                    "conversation": [item.model_dump() for item in conversation],
                },
            )
            response.raise_for_status()
            decision = GuardrailDecision.model_validate(response.json())
            if decision.stage != stage:
                raise ValueError("stage mismatch")
        except httpx.TimeoutException:
            code = "timeout"
            decision = self._error(stage, modalities, code, started)
        except httpx.HTTPStatusError as exc:
            code = f"http_{exc.response.status_code}"
            decision = self._error(stage, modalities, code, started)
        except (httpx.HTTPError, ValidationError, ValueError):
            code = "invalid_or_unavailable"
            decision = self._error(stage, modalities, code, started)

        elapsed_ms = (time.monotonic() - started) * 1000
        decision.latency_ms = elapsed_ms
        with self._tracer.start_as_current_span("guardrails.decision") as span:
            span.set_attribute("guardrails.stage", stage)
            span.set_attribute("guardrails.policy", decision.policy)
            span.set_attribute("guardrails.status", decision.status)
            span.set_attribute("guardrails.modalities", ",".join(dict.fromkeys(modalities)))
            span.set_attribute("guardrails.latency_ms", elapsed_ms)
            if decision.diagnostic_code:
                span.set_attribute("guardrails.failure_code", decision.diagnostic_code)
        return decision

    @staticmethod
    def _error(
        stage: GuardrailStage,
        modalities: Sequence[str],
        code: str,
        started: float,
    ) -> GuardrailDecision:
        return GuardrailDecision(
            status="error",
            stage=stage,
            policy="provider",
            diagnostic_code=code,
            modalities=list(dict.fromkeys(modalities)),
            latency_ms=(time.monotonic() - started) * 1000,
        )
