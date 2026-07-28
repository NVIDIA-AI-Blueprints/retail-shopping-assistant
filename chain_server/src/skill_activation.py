# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-phase shopper-skill activation for the Deep Agents runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Mapping
from threading import Lock
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    ContentBlock,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt.tool_node import ToolCallRequest

from .tool_policy import (
    SHOPPING_TOOL_POLICIES,
    granted_tools_for_skills,
    tool_is_granted,
    validate_skill_tool_grants,
)


SKILL_ACTIVATION_TOOL_NAME = "activate_shopper_skills_tool"
SKILL_ACTIVATION_COMPLETE = "SHOPPER_SKILL_ACTIVATION_COMPLETE:"
SKILL_ACTIVATION_INVALID = "SHOPPER_SKILL_ACTIVATION_INVALID:"
SKILL_ACTIVATION_CLARIFICATION_REQUIRED = (
    "SHOPPER_SKILL_ACTIVATION_CLARIFICATION_REQUIRED:"
)
SKILL_ACTIVATION_MULTIPLE_PRIMARY = "multiple_primary_procedures"
SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY = "modifier_requires_primary"
SKILL_ACTIVATION_EVENT_CONTEXT_REQUIRES_STYLING = (
    "event_context_requires_styling"
)
SKILL_ACTIVATION_REQUIRED = (
    "SKILL_ACTIVATION_REQUIRED: Shopper skills must be selected and loaded "
    "before any shopping tool can run. Retry after activation completes."
)
SKILL_TOOL_NOT_GRANTED = (
    "SHOPPER_SKILL_TOOL_NOT_GRANTED: The active shopper skills do not grant "
    "this tool. Continue using only the tools available for this turn."
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
only when the shopper states a budget. Event context may accompany outfit
styling only. Select it whenever an event destination or venue is stated, or
when the response would otherwise ask about or branch on missing destination or
venue context. An occasion-led styling request with no established setting
qualifies when location or venue could change the direction; generic advice is
not a reason to omit it. Keep the primary procedure aligned with the active
conversation task:
an outfit-building or styling thread continues to use outfit styling for
piece-by-piece searches until the shopper changes tasks. Do not switch to
product discovery merely because the current turn asks for one product type; a
terse item-only follow-up does not by itself end an active outfit task. Do not
mention this activation step or skill names to the shopper."""

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
        skill_descriptions: Mapping[str, str],
        skill_tool_grants: Mapping[str, Collection[str]],
        previous_selected_skills: Collection[str] = (),
    ) -> None:
        self._request_id = request_id
        self._skill_descriptions = dict(skill_descriptions)
        self._skill_tool_grants = {
            name: frozenset(tool_names)
            for name, tool_names in skill_tool_grants.items()
        }
        validate_skill_tool_grants(self._skill_tool_grants)
        self._status = "pending"
        self._skill_files: dict[str, str] = {}
        self._selected_skills: frozenset[str] = frozenset()
        self._granted_tools: frozenset[str] = frozenset()
        self._activation_validation_failures = 0
        self._clarification_response = ""
        self._previous_selected_skills = tuple(
            dict.fromkeys(
                name
                for name in previous_selected_skills
                if name in self._skill_descriptions
            )
        )
        self._lock = Lock()

    def activate(
        self,
        skill_files: Mapping[str, str],
        selected_skills: Collection[str],
    ) -> bool:
        """Store complete selected skill files for later model steps."""

        if not skill_files or any(
            not content.strip() for content in skill_files.values()
        ):
            raise ValueError("Activated skill files must contain instructions.")
        selected = frozenset(selected_skills)
        if not selected:
            raise ValueError("At least one shopper skill must be selected.")
        granted_tools = granted_tools_for_skills(
            selected,
            self._skill_tool_grants,
        )
        with self._lock:
            if self._status != "pending":
                return False
            self._skill_files = dict(skill_files)
            self._selected_skills = selected
            self._granted_tools = granted_tools
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

        clarification = self._clarification_model_response()
        if clarification is not None:
            return clarification
        response = handler(self._prepare_model_request(request))
        return self._validate_model_response(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Apply the activation phase to an asynchronous model request."""

        clarification = self._clarification_model_response()
        if clarification is not None:
            return clarification
        response = await handler(self._prepare_model_request(request))
        return self._validate_model_response(response)

    def handle_activation_validation_error(self, error: Any) -> str:
        """Return bounded model feedback for an invalid skill selection."""

        issue = _activation_validation_issue(error)
        feedback = _activation_validation_feedback(issue)
        with self._lock:
            if self._status != "pending":
                return f"{SKILL_ACTIVATION_INVALID} {feedback}"
            self._activation_validation_failures += 1
            if self._activation_validation_failures == 1:
                return (
                    f"{SKILL_ACTIVATION_INVALID} {feedback} "
                    "Retry the activation once with the smallest valid skill set."
                )
            self._status = "clarification"
            self._clarification_response = _activation_clarification(issue)
        return (
            f"{SKILL_ACTIVATION_CLARIFICATION_REQUIRED} {feedback} "
            "No shopping tool was run."
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Reject a synchronous shopping call without prior activation."""

        rejection = self._tool_call_rejection(request)
        if rejection is None:
            return handler(request)
        return _tool_rejection_message(request, rejection)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Reject an asynchronous shopping call without prior activation."""

        rejection = self._tool_call_rejection(request)
        if rejection is None:
            return await handler(request)
        return _tool_rejection_message(request, rejection)

    def _prepare_model_request(self, request: ModelRequest) -> ModelRequest:
        with self._lock:
            status = self._status
            skill_files = dict(self._skill_files)
            selected_skills = self._selected_skills
            granted_tools = self._granted_tools
        if status == "active":
            tools = [
                candidate
                for candidate in request.tools
                if _tool_is_visible(
                    _tool_name(candidate),
                    selected_skills,
                    granted_tools,
                )
            ]
            tool_choice = request.tool_choice
            if isinstance(tool_choice, str) and not _tool_is_visible(
                tool_choice,
                selected_skills,
                granted_tools,
            ):
                tool_choice = None
            return request.override(
                tools=tools,
                tool_choice=tool_choice,
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
                _activation_prompt(
                    self._skill_descriptions,
                    previous_skills=self._previous_selected_skills,
                ),
            ),
        )

    def _clarification_model_response(self) -> ModelResponse | None:
        with self._lock:
            if self._status != "clarification":
                return None
            clarification = self._clarification_response
        return ModelResponse(result=[AIMessage(content=clarification)])

    def _tool_call_rejection(self, request: ToolCallRequest) -> str | None:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name == SKILL_ACTIVATION_TOOL_NAME:
            return None
        if tool_name not in SHOPPING_TOOL_POLICIES:
            return None
        if not _has_current_turn_activation(request.state, self._request_id):
            return SKILL_ACTIVATION_REQUIRED
        with self._lock:
            status = self._status
            selected_skills = self._selected_skills
            granted_tools = self._granted_tools
        if status != "active" or not tool_is_granted(
            tool_name,
            selected_skills,
            granted_tools,
        ):
            return SKILL_TOOL_NOT_GRANTED
        return None

    def _validate_model_response(self, response: ModelResponse) -> ModelResponse:
        with self._lock:
            status = self._status
        if status == "active":
            shopping_calls = [
                tool_call
                for message in response.result
                for tool_call in _value(message, "tool_calls") or []
                if _value(tool_call, "name") in SHOPPING_TOOL_POLICIES
            ]
            if len(shopping_calls) > 1:
                raise ShopperSkillActivationError(
                    "The model requested multiple shopping tools in one step."
                )
            return response
        if status != "pending":
            return response

        activation_calls = _activation_calls(response)
        if not activation_calls:
            raise ShopperSkillActivationError(
                "The model did not complete required shopper skill activation."
            )
        if len(activation_calls) == 1:
            return response

        clarification = _activation_clarification("invalid_selection")
        with self._lock:
            self._status = "clarification"
            self._clarification_response = clarification
        return ModelResponse(result=[AIMessage(content=clarification)])


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


def _activation_prompt(
    skill_descriptions: Mapping[str, str],
    *,
    previous_skills: tuple[str, ...] = (),
) -> str:
    lines = [_ACTIVATION_PROMPT, "", "Registered shopper skills:"]
    lines.extend(
        f"- {name}: {description}" for name, description in skill_descriptions.items()
    )
    if previous_skills:
        lines.extend(
            (
                "",
                "Previous turn's selected shopper skills: "
                + ", ".join(previous_skills)
                + ". Keep the same primary skill when the current request "
                "continues that task; change it only when the shopper changes "
                "tasks.",
            )
        )
    return "\n".join(lines)


def selected_skill_names_for_turn(
    messages: list[Any],
    request_id: str,
) -> tuple[str, ...]:
    """Return skills whose activation completed in the current turn."""

    marker = f"REQUEST ID: {request_id}"
    current_turn_start: int | None = None
    for index, message in enumerate(messages):
        if _message_type(message) == "human" and marker in _message_text(message):
            current_turn_start = index
    if current_turn_start is None:
        return ()
    turn_messages = messages[current_turn_start:]
    completed_call_ids = {
        str(_value(message, "tool_call_id") or "")
        for message in turn_messages
        if _message_type(message) == "tool"
        and _value(message, "name") == SKILL_ACTIVATION_TOOL_NAME
        and _message_text(message).startswith(SKILL_ACTIVATION_COMPLETE)
    }
    for message in turn_messages:
        if _message_type(message) != "ai":
            continue
        for tool_call in _value(message, "tool_calls") or []:
            if _value(tool_call, "name") != SKILL_ACTIVATION_TOOL_NAME:
                continue
            if str(_value(tool_call, "id") or "") not in completed_call_ids:
                continue
            arguments = _value(tool_call, "args") or {}
            names = _value(arguments, "skill_names") or []
            return tuple(
                name
                for name in names
                if isinstance(name, str) and name.strip()
            )
    return ()


def _tool_rejection_message(
    request: ToolCallRequest,
    content: str,
) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=str(request.tool_call.get("name") or "unknown"),
        tool_call_id=str(request.tool_call.get("id") or ""),
    )


def _activation_calls(response: ModelResponse) -> list[Any]:
    return [
        tool_call
        for message in response.result
        for tool_call in _value(message, "tool_calls") or []
        if _value(tool_call, "name") == SKILL_ACTIVATION_TOOL_NAME
    ]


def _activation_validation_issue(error: Any) -> str:
    errors = error.errors() if callable(getattr(error, "errors", None)) else []
    for detail in errors:
        issue = str(_value(detail, "type") or "")
        if issue in {
            SKILL_ACTIVATION_EVENT_CONTEXT_REQUIRES_STYLING,
            SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
            SKILL_ACTIVATION_MULTIPLE_PRIMARY,
        }:
            return issue
    return "invalid_selection"


def _activation_validation_feedback(issue: str) -> str:
    if issue == SKILL_ACTIVATION_EVENT_CONTEXT_REQUIRES_STYLING:
        return (
            "event-context is a modifier and requires outfit-styling. "
            "Pair it with outfit-styling for occasion-led fashion guidance; "
            "otherwise remove it."
        )
    if issue == SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY:
        return (
            "budget-shopping is a modifier and requires exactly one primary "
            "procedure: outfit-styling or product-discovery."
        )
    if issue == SKILL_ACTIVATION_MULTIPLE_PRIMARY:
        return (
            "Select exactly one primary procedure: outfit-styling or "
            "product-discovery, never both."
        )
    return "The selected shopper-skill combination is invalid."


def _activation_clarification(issue: str) -> str:
    if issue == SKILL_ACTIVATION_EVENT_CONTEXT_REQUIRES_STYLING:
        return "What outfit or event would you like help styling?"
    if issue == SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY:
        return "What product or outfit would you like to find within your budget?"
    return "What product or shopping task would you like help with?"


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


def _tool_is_visible(
    tool_name: str,
    selected_skills: Collection[str],
    granted_tools: Collection[str],
) -> bool:
    if tool_name == SKILL_ACTIVATION_TOOL_NAME:
        return False
    return tool_is_granted(tool_name, selected_skills, granted_tools)


def _value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
