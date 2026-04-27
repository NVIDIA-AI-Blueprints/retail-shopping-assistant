# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``guardrails.src.rails``.

The module is organised for an in-container layout (``/app/src`` on
``PYTHONPATH``) so its imports are flat (``from config_utils import ...``)
and it instantiates ``GuardRails`` at module load time. To make it
testable as a library we:

* Add ``guardrails/src/`` to ``sys.path`` so the flat import resolves.
* Register a dummy ``nemoguardrails`` module in ``sys.modules`` *before*
  importing ``rails`` so the real package is never required.
* Let the module run its import-time ``GuardRails(config_path)`` against
  the stubbed classes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Module loading helper with full dependency isolation
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDRAILS_SRC = str(REPO_ROOT / "guardrails" / "src")


def _install_fake_nemoguardrails(
    created: Dict[str, Any],
) -> ModuleType:
    """Create a stub ``nemoguardrails`` package and register it in sys.modules.

    ``RailsConfig.from_path`` just records the path and returns a config-like
    object. ``LLMRails.generate_async`` is a coroutine that echoes the user
    input, which the real guardrails does when inputs are considered safe.
    """

    class _RailsConfig:
        def __init__(self, config_path: str) -> None:
            self.config_path = config_path
            # Mirror the real type's ``.models`` structure so
            # ``apply_endpoint_overrides`` could operate on it if invoked.
            self.models = []

        @classmethod
        def from_path(cls, path: str) -> "_RailsConfig":
            created["from_path_arg"] = path
            return cls(path)

    class _LLMRails:
        def __init__(self, config: Any) -> None:
            created.setdefault("llm_rails_instances", []).append(config)
            self._generate_result: Dict[str, Any] | None = None

        def configure_result(self, result: Dict[str, Any]) -> None:
            self._generate_result = result

        async def generate_async(
            self, messages: List[Dict[str, Any]], options: Dict[str, Any]
        ) -> Dict[str, Any]:
            created.setdefault("calls", []).append(
                {"messages": messages, "options": options}
            )
            if self._generate_result is not None:
                return self._generate_result
            # Default: echo the last user message as assistant content.
            last = messages[-1]
            return {
                "response": [
                    {"role": "assistant", "content": last["content"]}
                ]
            }

    fake_module = ModuleType("nemoguardrails")
    fake_module.RailsConfig = _RailsConfig
    fake_module.LLMRails = _LLMRails

    sys.modules["nemoguardrails"] = fake_module
    return fake_module


@pytest.fixture
def rails_module(monkeypatch: pytest.MonkeyPatch):
    """Import ``guardrails.src.rails`` with all external deps stubbed out.

    The fixture rebuilds a fresh module per test so state from one case
    never leaks into another.
    """
    created: Dict[str, Any] = {}
    _install_fake_nemoguardrails(created)

    # Prepend the in-container-style src dir so ``from config_utils import ...``
    # resolves without a package prefix.
    if GUARDRAILS_SRC not in sys.path:
        sys.path.insert(0, GUARDRAILS_SRC)

    # Force a fresh import of both names to pick up our injected stubs.
    for name in ("rails", "guardrails.src.rails"):
        sys.modules.pop(name, None)

    import importlib

    rails_module = importlib.import_module("guardrails.src.rails")
    # Attach the recorder to the module for per-test access.
    rails_module._test_created = created  # type: ignore[attr-defined]

    yield rails_module

    # Clean up import cache so other test files get a pristine module.
    for name in ("rails", "guardrails.src.rails"):
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModuleImportWiring:
    def test_module_exposes_guardrails_singleton(
        self, rails_module
    ) -> None:
        assert hasattr(rails_module, "guardRails")
        assert isinstance(rails_module.guardRails, rails_module.GuardRails)

    def test_rails_accessor_returns_singleton(
        self, rails_module
    ) -> None:
        rails = rails_module.Rails().getGuardRails()
        assert rails is rails_module.guardRails

    def test_rails_config_from_path_called_with_container_path(
        self, rails_module
    ) -> None:
        created = rails_module._test_created
        # Singleton is built in the module body with this hardcoded path.
        assert created["from_path_arg"] == "/app/shared/configs/rails"


class TestBaseRails:
    async def test_base_rails_methods_are_noops(self, rails_module) -> None:
        base = rails_module.BaseRails()
        assert await base.call_input_content_rails("hi") is None
        assert await base.call_output_content_rails("hi") is None


class TestGuardRails:
    async def test_input_check_generates_with_input_rails_option(
        self, rails_module
    ) -> None:
        rails = rails_module.guardRails
        result = await rails.call_input_content_rails("hello")

        # Default stub echoes the input back as the assistant message.
        assert result["response"][0]["content"] == "hello"

        calls = rails_module._test_created["calls"]
        assert calls[-1]["options"] == {"rails": ["input"]}
        assert calls[-1]["messages"] == [
            {"role": "user", "content": "hello"}
        ]

    async def test_output_check_builds_expected_messages_shape(
        self, rails_module
    ) -> None:
        rails = rails_module.guardRails
        await rails.call_output_content_rails("safe bot response")

        calls = rails_module._test_created["calls"]
        assert calls[-1]["options"] == {"rails": ["output"]}
        assert calls[-1]["messages"] == [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "safe bot response"},
        ]

    async def test_unsafe_input_rewrites_response(
        self, rails_module
    ) -> None:
        # Configure the stub LLMRails to simulate an unsafe classification.
        rails = rails_module.guardRails
        rails.app.configure_result(
            {
                "response": [
                    {"role": "assistant", "content": "I can't help with that."}
                ]
            }
        )

        result = await rails.call_input_content_rails("prompt injection attempt")
        assert result["response"][0]["content"] == "I can't help with that."
