# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic tool-loop termination for the Deep Agents runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ContentBlock, SystemMessage, ToolMessage


SEARCH_TOOL_NAME = "search_catalog_tool"
SEARCH_VALIDATION_ERROR_PREFIX = (
    f"Error invoking tool '{SEARCH_TOOL_NAME}' with kwargs "
)
STOP_TOOL_USE_PREFIX = "STOP_TOOL_USE:"
SEARCH_SCOPE_COMPLETE_PREFIX = "SEARCH_SCOPE_COMPLETE:"
CONSTRAINT_REVIEW_PREFIX = "REVIEW_REQUIRED_CONSTRAINT:"
UNSUPPORTED_TAXONOMY_PREFIX = "The requested catalog taxonomy cannot be enforced:"
UNSUPPORTED_CONSTRAINT_PREFIX = "The requested catalog requirement cannot be enforced:"
_SYNTHESIS_PROMPT = """## Tool Loop Closed

Do not call or describe another tool. Produce the best concise shopper-facing
answer now from the tool outcome and any evidence already in this turn. If that
is insufficient, say what the catalog could not establish and offer one next
step. When the outcome says no faithful advertised taxonomy matches, do not
claim a search ran and do not name alternative product types; ask permission
before searching a different advertised type."""
_REPAIR_PROMPT = """## Catalog Search Repair

Correct the previous catalog call once using the tool result and advertised
search schema. Send exactly one category per call; category and subcategory are
JSON arrays. Send one focused product role per call. For a broad styling or
discovery request that named no product type, use agent_selected_type with
one focused role in one category and every advertised subcategory that serves
that role. A broad rainy-day or wet-weather
outfit is context, not a water-resistance requirement. Before using no_direct_catalog_match,
separate the requested product type from its modifiers: an unavailable
attribute does not erase an advertised type, subjective style stays in the
semantic query, and a supported alternative branch must still be searched. A
no-direct result has no required constraints. If the result asks for constraint
review, preserve a product attribute directly stated for the target product and
stop; remove a requirement inferred from broad season, weather, occasion, or
style context. Do not repeat the same invalid arguments."""
_SYNTHESIS_FALLBACK = (
    "I couldn't establish a reliable catalog match for that request. I can "
    "explain the gap or search a different advertised product type if you'd like."
)


class ToolLoopControlMiddleware(AgentMiddleware):
    """Allow one search-schema repair, then require answer synthesis."""

    def __init__(self) -> None:
        self._repair_pending = False
        self._repair_in_flight = False
        self._repair_used = False
        self._synthesis_required = False
        self._observed_tool_results: set[str] = set()
        self._lock = Lock()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Control a synchronous model step."""

        response = handler(self._prepare_model_request(request))
        return self._enforce_synthesis(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Control an asynchronous model step."""

        response = await handler(self._prepare_model_request(request))
        return self._enforce_synthesis(response)

    def _prepare_model_request(self, request: ModelRequest) -> ModelRequest:
        with self._lock:
            self._observe_tool_results(request.messages)
            if self._synthesis_required:
                return request.override(
                    tools=[],
                    tool_choice="none",
                    system_message=_append_system_text(
                        request.system_message,
                        _SYNTHESIS_PROMPT,
                    ),
                )
            if not self._repair_pending:
                return request
            self._repair_pending = False
            self._repair_in_flight = True
            self._repair_used = True

        search_tools = [
            tool for tool in request.tools if _tool_name(tool) == SEARCH_TOOL_NAME
        ]
        return request.override(
            tools=search_tools,
            tool_choice=None,
            model_settings={**request.model_settings, "parallel_tool_calls": False},
            system_message=_append_system_text(
                request.system_message,
                _REPAIR_PROMPT,
            ),
        )

    def _observe_tool_results(self, messages: list[Any]) -> None:
        """Observe current-turn graph results, including schema failures."""

        start = 0
        for index, message in enumerate(messages):
            if str(getattr(message, "type", "")) == "human":
                start = index + 1

        for result in messages[start:]:
            if not isinstance(result, ToolMessage):
                continue
            tool_call_id = str(result.tool_call_id or "")
            if tool_call_id in self._observed_tool_results:
                continue
            self._observed_tool_results.add(tool_call_id)
            content = result.content
            if not isinstance(content, str):
                continue
            content = content.strip()
            if content.startswith(STOP_TOOL_USE_PREFIX):
                self._synthesis_required = True
                continue
            if content.startswith(
                (UNSUPPORTED_TAXONOMY_PREFIX, UNSUPPORTED_CONSTRAINT_PREFIX)
            ):
                self._synthesis_required = True
                continue
            if self._repair_in_flight:
                self._repair_in_flight = False
                if (
                    "SEARCH_RESULT_GROUNDING_NOTE" not in content
                    or SEARCH_SCOPE_COMPLETE_PREFIX in content
                ):
                    self._synthesis_required = True
                continue
            if SEARCH_SCOPE_COMPLETE_PREFIX in content:
                self._synthesis_required = True
            elif content.startswith(
                (SEARCH_VALIDATION_ERROR_PREFIX, CONSTRAINT_REVIEW_PREFIX)
            ):
                if self._repair_used:
                    self._synthesis_required = True
                else:
                    self._repair_pending = True

    def _enforce_synthesis(self, response: ModelResponse) -> ModelResponse:
        with self._lock:
            synthesis_required = self._synthesis_required
        if not synthesis_required:
            return response

        result = []
        for message in response.result:
            if not (
                getattr(message, "tool_calls", None)
                or getattr(message, "invalid_tool_calls", None)
            ):
                result.append(message)
                continue
            update: dict[str, Any] = {
                "tool_calls": [],
                "invalid_tool_calls": [],
            }
            if not _message_text(message):
                update["content"] = _SYNTHESIS_FALLBACK
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if (
                isinstance(additional_kwargs, dict)
                and "tool_calls" in additional_kwargs
            ):
                update["additional_kwargs"] = {
                    key: value
                    for key, value in additional_kwargs.items()
                    if key != "tool_calls"
                }
            result.append(message.model_copy(update=update))
        return ModelResponse(
            result=result,
            structured_response=response.structured_response,
        )


def _tool_name(candidate: Any) -> str:
    if isinstance(candidate, dict):
        function = candidate.get("function") or {}
        return str(candidate.get("name") or function.get("name") or "")
    return str(getattr(candidate, "name", ""))


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


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict)
        ).strip()
    return ""
