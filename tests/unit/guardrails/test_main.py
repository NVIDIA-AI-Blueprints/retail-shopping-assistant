"""Contract tests for the internal /v1/checks guardrail API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDRAILS_SRC = str(REPO_ROOT / "guardrails" / "src")


@pytest.fixture
def main_module(monkeypatch):
    fake_rails = ModuleType("rails")
    fake_rails.GuardrailEngine = object
    monkeypatch.setitem(sys.modules, "rails", fake_rails)
    if GUARDRAILS_SRC not in sys.path:
        sys.path.insert(0, GUARDRAILS_SRC)
    sys.modules.pop("guardrails.src.main", None)
    return importlib.import_module("guardrails.src.main")


class FakeEngine:
    def __init__(self, status="allow"):
        self.status = status
        self.requests = []
        self.input_execution_mode = "parallel"

    async def check(self, request):
        self.requests.append(request)
        return (
            SimpleNamespace(
                status=self.status,
                policy="combined",
                violated_categories=["test"] if self.status == "block" else [],
                diagnostic_code=None,
            ),
            4.5,
            list(dict.fromkeys(
                (["text"] if request.shopper_text or request.assistant_text else [])
                + [item.type for item in request.attachments]
            )),
            {"content_safety": 1, "topic_control": 1},
        )


def test_health_and_typed_input_contract(main_module):
    engine = FakeEngine()
    client = TestClient(main_module.create_app(engine))
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/capabilities").json() == {
        "input_execution_mode": "parallel"
    }

    response = client.post("/v1/checks", json={
        "stage": "input",
        "shopper_text": "find a jacket",
        "attachments": [{
            "type": "video", "data": "data:video/mp4;base64,AAAA",
            "mime_type": "video/mp4",
        }],
    })
    assert response.status_code == 200
    assert response.json() == {
        "status": "allow",
        "stage": "input",
        "policy": "combined",
        "violated_categories": [],
        "latency_ms": 4.5,
        "diagnostic_code": None,
        "modalities": ["text", "video"],
        "model_calls": {"content_safety": 1, "topic_control": 1},
    }


def test_contract_rejects_media_on_output(main_module):
    client = TestClient(main_module.create_app(FakeEngine()))
    response = client.post("/v1/checks", json={
        "stage": "output",
        "assistant_text": "safe",
        "attachments": [{
            "type": "image", "data": "data:image/png;base64,AAAA",
            "mime_type": "image/png",
        }],
    })
    assert response.status_code == 422


def test_service_exception_is_sanitized(main_module):
    class Broken:
        async def check(self, _request):
            raise RuntimeError("raw shopper content must not escape")

    client = TestClient(main_module.create_app(Broken()))
    body = client.post("/v1/checks", json={
        "stage": "output", "assistant_text": "private response"
    }).json()
    assert body["status"] == "error"
    assert body["diagnostic_code"] == "evaluation_failed"
    assert body["model_calls"] == {}
    assert "private response" not in str(body)
