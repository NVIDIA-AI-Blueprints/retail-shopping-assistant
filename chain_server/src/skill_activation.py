# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-phase shopper-skill activation for the Deep Agents runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from threading import Lock
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ContentBlock, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest


SKILL_ACTIVATION_TOOL_NAME = "activate_shopper_skills_tool"
SKILL_ACTIVATION_COMPLETE = "SHOPPER_SKILL_ACTIVATION_COMPLETE:"
SKILL_ACTIVATION_REQUIRED = (
    "SKILL_ACTIVATION_REQUIRED: Shopper skills must be selected and loaded "
    "before any shopping tool can run. Retry after activation completes."
)


class ShopperSkillActivationError(RuntimeError):
    """Raised when a turn tries to finish before required skill activation."""


_ACTIVATION_PROMPT = f"""## Required Shopper Skill Selection

Before answering this turn or using any shopping tool, call
`{SKILL_ACTIVATION_TOOL_NAME}` exactly once. Select the smallest set of
registered skills that fully covers the shopper's current intent. Use the
registered descriptions below and the full conversation context; this is
semantic selection, not keyword matching. Do not attempt another tool in the
same response. The runtime will load the complete selected instructions before
the next model step. Outfit styling and product discovery are alternative
primary procedures, so never select both; budget shopping may accompany either
only when the shopper states a budget. Keep the primary procedure aligned with
the active conversation task: an outfit-building or styling thread continues
to use outfit styling for piece-by-piece searches until the shopper changes
tasks. Do not switch to product discovery merely because the current turn asks
for one product type. Do not mention this activation step or skill names to the
shopper."""

_ACTIVATION_FAILED_PROMPT = """## Shopper Skill Activation Failed

Shopper skill instructions could not be loaded for this turn, so no shopping
tools are available. Tell the shopper that the assistant cannot complete the
request right now and ask them to try again. Do not invent catalog, cart,
policy, or availability information."""


class ShopperSkillActivationMiddleware(AgentMiddleware):
    """Require selected skill content before exposing or executing tools."""

    def __init__(
        self,
        *,
        request_id: str,
        gated_tools: frozenset[str],
        skill_descriptions: Mapping[str, str],
    ) -> None:
        self._request_id = request_id
        self._gated_tools = gated_tools
        self._skill_descriptions = dict(skill_descriptions)
        self._status = "pending"
        self._skill_files: dict[str, str] = {}
        self._lock = Lock()

    def activate(self, skill_files: Mapping[str, str]) -> bool:
        """Store complete selected skill files for later model steps."""

        if not skill_files or any(
            not content.strip() for content in skill_files.values()
        ):
            raise ValueError("Activated skill files must contain instructions.")
        with self._lock:
            if self._status != "pending":
                return False
            self._skill_files = dict(skill_files)
            self._status = "active"
        return True

    def fail(self) -> None:
        """Fail closed when selected skill instructions cannot be loaded."""

        with self._lock:
            if self._status == "pending":
                self._status = "failed"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Apply the activation phase to a synchronous model request."""

        response = handler(self._prepare_model_request(request))
        self._validate_model_response(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Apply the activation phase to an asynchronous model request."""

        response = await handler(self._prepare_model_request(request))
        self._validate_model_response(response)
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Reject a synchronous shopping call without prior activation."""

        if self._tool_call_is_allowed(request):
            return handler(request)
        return _activation_required_message(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Reject an asynchronous shopping call without prior activation."""

        if self._tool_call_is_allowed(request):
            return await handler(request)
        return _activation_required_message(request)

    def _prepare_model_request(self, request: ModelRequest) -> ModelRequest:
        with self._lock:
            status = self._status
            skill_files = dict(self._skill_files)
        if status == "active":
            tools = [
                candidate
                for candidate in request.tools
                if _tool_name(candidate) != SKILL_ACTIVATION_TOOL_NAME
            ]
            return request.override(
                tools=tools,
                tool_choice=None,
                model_settings={
                    **request.model_settings,
                    "parallel_tool_calls": False,
                },
                system_message=_append_system_text(
                    request.system_message,
                    _active_skills_prompt(skill_files),
                ),
            )
        if status == "failed":
            return request.override(
                tools=[],
                tool_choice=None,
                system_message=_append_system_text(
                    request.system_message,
                    _ACTIVATION_FAILED_PROMPT,
                ),
            )

        activation_tools = [
            candidate
            for candidate in request.tools
            if _tool_name(candidate) == SKILL_ACTIVATION_TOOL_NAME
        ]
        if len(activation_tools) != 1:
            raise RuntimeError(
                "Shopper skill activation tool is not registered exactly once."
            )
        return request.override(
            tools=activation_tools,
            tool_choice=SKILL_ACTIVATION_TOOL_NAME,
            model_settings={**request.model_settings, "parallel_tool_calls": False},
            system_message=_append_system_text(
                request.system_message,
                _activation_prompt(self._skill_descriptions),
            ),
        )

    def _tool_call_is_allowed(self, request: ToolCallRequest) -> bool:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name == SKILL_ACTIVATION_TOOL_NAME:
            return True
        if tool_name not in self._gated_tools:
            return True
        return _has_current_turn_activation(request.state, self._request_id)

    def _validate_model_response(self, response: ModelResponse) -> None:
        with self._lock:
            activation_pending = self._status == "pending"
        if activation_pending and not _response_requests_activation(response):
            raise ShopperSkillActivationError(
                "The model did not complete required shopper skill activation."
            )


def _active_skills_prompt(skill_files: Mapping[str, str]) -> str:
    sections = [
        "## Active Shopper Skills",
        "The complete instructions below are mandatory for this turn. Apply them "
        "before constructing tool arguments or the final response. They are already "
        "loaded; do not read these SKILL.md files again or mention them to the shopper.",
    ]
    for path, content in skill_files.items():
        sections.extend((f"### {path}", content.strip()))
    return "\n\n".join(sections)


def _activation_prompt(skill_descriptions: Mapping[str, str]) -> str:
    lines = [_ACTIVATION_PROMPT, "", "Registered shopper skills:"]
    lines.extend(
        f"- {name}: {description}" for name, description in skill_descriptions.items()
    )
    return "\n".join(lines)


def _activation_required_message(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(
        content=SKILL_ACTIVATION_REQUIRED,
        name=str(request.tool_call.get("name") or "unknown"),
        tool_call_id=str(request.tool_call.get("id") or ""),
    )


def _response_requests_activation(response: ModelResponse) -> bool:
    for message in response.result:
        for tool_call in _value(message, "tool_calls") or []:
            if _value(tool_call, "name") == SKILL_ACTIVATION_TOOL_NAME:
                return True
    return False


def _has_current_turn_activation(state: Any, request_id: str) -> bool:
    messages = _state_messages(state)
    marker = f"REQUEST ID: {request_id}"
    start = len(messages)
    for index, message in enumerate(messages):
        if _message_type(message) == "human" and marker in _message_text(message):
            start = index + 1
    for message in messages[start:]:
        if _message_type(message) != "tool":
            continue
        if _value(message, "name") != SKILL_ACTIVATION_TOOL_NAME:
            continue
        if _value(message, "status") == "error":
            continue
        if _message_text(message).startswith(SKILL_ACTIVATION_COMPLETE):
            return True
    return False


def _append_system_text(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    content: list[ContentBlock] = (
        list(system_message.content_blocks) if system_message else []
    )
    if content:
        text = f"\n\n{text}"
    content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=content)


def _state_messages(state: Any) -> list[Any]:
    messages = _value(state, "messages") or []
    return messages if isinstance(messages, list) else []


def _message_type(message: Any) -> str:
    return str(_value(message, "type") or _value(message, "role") or "").lower()


def _message_text(message: Any) -> str:
    content = _value(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(_value(block, "text") or "") for block in content
        ).strip()
    return str(content or "")


def _tool_name(candidate: Any) -> str:
    if isinstance(candidate, dict):
        function = candidate.get("function") or {}
        return str(candidate.get("name") or function.get("name") or "")
    return str(getattr(candidate, "name", ""))


def _value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
