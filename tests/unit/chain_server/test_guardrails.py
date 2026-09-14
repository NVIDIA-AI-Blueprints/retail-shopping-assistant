"""Tests for the provider-neutral chain-server guardrail boundary."""

from __future__ import annotations

import httpx
import pytest
from chain_server.src.guardrails import (
    GuardrailDecision,
    NemoGuardrailProvider,
    stops_turn,
)


@pytest.mark.parametrize(
    ("status", "failure_mode", "expected"),
    [
        ("allow", "open", False),
        ("allow", "closed", False),
        ("block", "open", True),
        ("block", "closed", True),
        ("error", "open", False),
        ("error", "closed", True),
    ],
)
def test_failure_policy(status, failure_mode, expected):
    decision = GuardrailDecision(status=status, stage="input")
    assert stops_turn(decision, failure_mode) is expected


@pytest.mark.asyncio
async def test_input_contract_includes_every_normalized_attachment():
    recorded = []

    def handler(request: httpx.Request):
        import json
        payload = json.loads(request.content)
        recorded.append(payload)
        return httpx.Response(200, json={
            "status": "allow", "stage": "input", "policy": "combined",
            "violated_categories": [], "latency_ms": 1,
            "modalities": ["text", "image", "video"],
        })

    provider = NemoGuardrailProvider("http://rails", timeout_seconds=1)
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    decision = await provider.check_input(
        text="find this look",
        conversation=[
            {"role": "user", "content": "Show me jackets"},
            {"role": "assistant", "content": "Here are three options."},
        ],
        media=[
            {"type": "image", "data": "data:image/png;base64,AAAA", "mime_type": "image/png"},
            {"type": "video", "data": "data:video/mp4;base64,AAAA", "mime_type": "video/mp4"},
        ],
    )
    await provider._client.aclose()

    assert decision.status == "allow"
    assert [item["type"] for item in recorded[0]["attachments"]] == ["image", "video"]
    assert recorded[0]["conversation"] == [
        {"role": "user", "content": "Show me jackets"},
        {"role": "assistant", "content": "Here are three options."},
    ]


@pytest.mark.asyncio
async def test_invalid_provider_payload_is_a_sanitized_error():
    provider = NemoGuardrailProvider("http://rails", timeout_seconds=1)
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"raw": "unsafe body"})
        )
    )
    decision = await provider.check_output(text="private assistant body")
    await provider._client.aclose()

    assert decision.status == "error"
    assert decision.diagnostic_code == "invalid_or_unavailable"
    assert "private" not in decision.model_dump_json()
