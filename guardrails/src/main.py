# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal typed guardrail service. Bodies are never logged or traced."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from rails import GuardrailEngine


class Attachment(BaseModel):
    type: Literal["image", "video"]
    data: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    filename: str = ""


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class CheckRequest(BaseModel):
    stage: Literal["input", "output"]
    shopper_text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    assistant_text: str = ""
    conversation: list[ConversationMessage] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_stage_content(self):
        if self.stage == "output" and (
            self.shopper_text or self.attachments or self.conversation
        ):
            raise ValueError("output checks accept assistant_text only")
        if self.stage == "input" and self.assistant_text:
            raise ValueError("input checks do not accept assistant_text")
        return self


class CheckResponse(BaseModel):
    status: Literal["allow", "block", "error"]
    stage: Literal["input", "output"]
    policy: str
    violated_categories: list[str] = Field(default_factory=list)
    latency_ms: float
    diagnostic_code: str | None = None
    modalities: list[Literal["text", "image", "video"]] = Field(default_factory=list)
    model_calls: dict[str, int] = Field(default_factory=dict)


class CapabilitiesResponse(BaseModel):
    input_execution_mode: Literal["parallel", "sequential", "unknown"]


def create_app(engine: GuardrailEngine | None = None) -> FastAPI:
    app = FastAPI(title="Retail Guardrails", version="1.0.0")
    evaluator = engine or GuardrailEngine()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            input_execution_mode=getattr(evaluator, "input_execution_mode", "unknown")
        )

    @app.post("/v1/checks", response_model=CheckResponse)
    async def checks(request: CheckRequest) -> CheckResponse:
        try:
            decision, latency_ms, modalities, model_calls = await evaluator.check(request)
        except Exception:  # noqa: BLE001 - keep diagnostics sanitized.
            return CheckResponse(
                status="error",
                stage=request.stage,
                policy="service",
                latency_ms=0,
                diagnostic_code="evaluation_failed",
                modalities=list(dict.fromkeys(
                    (["text"] if request.shopper_text or request.assistant_text else [])
                    + [item.type for item in request.attachments]
                )),
                model_calls={},
            )
        return CheckResponse(
            status=decision.status,
            stage=request.stage,
            policy=decision.policy,
            violated_categories=decision.violated_categories,
            diagnostic_code=decision.diagnostic_code,
            latency_ms=latency_ms,
            modalities=modalities,
            model_calls=model_calls,
        )

    return app


app = create_app()
