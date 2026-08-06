# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed chain-server boundary for durable conversation turns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from .agenttypes import SHOPPER_PROFILE_ID_PATTERN, DialogueTurn, ShopperContext
from shared.commerce_contracts import ProductSummary


TurnStatus = Literal["started", "completed", "failed", "blocked", "abandoned"]
FinalTurnStatus = Literal["completed", "failed", "blocked"]
EventSourceKind = Literal["shopper", "compiler", "catalog", "cart", "runtime"]
EventType = Literal[
    "candidate_set_presented",
    "product_detail_confirmed",
    "historical_reference_resolved",
    "reference_clarification_required",
    "product_selected",
    "product_rejected",
    "preference_added",
    "preference_superseded",
    "catalog_scope_no_match",
    "wearer_audience_declared",
    "audience_assumption_disclosed",
]

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_CONTEXT_MAX_CHARS = 16_384
_TRUNCATION_MARKER = "…"


class _MemoryModel(BaseModel):
    """Closed model for the conversation-memory wire contract."""

    model_config = ConfigDict(extra="forbid")


class TurnStartRequest(_MemoryModel):
    """Idempotent input for one durable turn start."""

    request_id: str = Field(..., min_length=1, max_length=128)
    shopper_text: str = Field(..., min_length=1, max_length=100_000)
    cart_user_id: int = Field(..., ge=0)
    request_digest: str = Field(
        ...,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    shopper_profile_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=SHOPPER_PROFILE_ID_PATTERN,
    )
    catalog_revision: str | None = Field(default=None, min_length=1, max_length=512)


class RecentConversationTurn(_MemoryModel):
    """One bounded raw turn returned for current-turn prompt context."""

    sequence: int = Field(..., ge=1)
    shopper_text: str = Field(..., min_length=1, max_length=100_000)
    assistant_text: str | None = Field(default=None, max_length=100_000)
    status: TurnStatus | None = None


class ConversationProjection(_MemoryModel):
    """Per-conversation derived summary rebuilt at each finalization.

    Only ``product_reference_index`` is consumed by the runtime. The remaining
    lanes are accepted so the memory contract stays stable, and are dropped
    deliberately rather than by omission; see the field notes below.
    """

    #: Written by the memory service, deliberately unread by the runtime.
    version: int = Field(default=0, ge=0)
    #: Reserved and never written. Would carry the item an outfit is built
    #: around, which the model currently re-derives each turn.
    active_anchors: list[JsonValue] = Field(default_factory=list, max_length=50)
    #: Reserved and never written. Would carry standing shopper constraints
    #: that today survive only as long as the dialogue window.
    effective_preferences: list[JsonValue] = Field(
        default_factory=list,
        max_length=100,
    )
    #: Consumed. The only lane with a full-conversation horizon.
    product_reference_index: list[JsonValue] = Field(
        default_factory=list,
        max_length=100,
    )
    #: Written by the memory service, deliberately unread by the runtime.
    last_turn_id: str | None = Field(default=None, min_length=1, max_length=256)


class ConversationCartItem(_MemoryModel):
    """One authoritative cart item returned with a turn-start snapshot."""

    cart_line_id: str = Field(..., min_length=1)
    product_id: str | None = Field(default=None, min_length=1)
    item: str = Field(..., min_length=1)
    amount: int = Field(..., ge=1)
    price: float | None = None


class TurnReplayOutput(_MemoryModel):
    """Shopper-visible artifacts restored for one finalized replay."""

    product_results: list[ProductSummary]
    retrieved: dict[str, str]
    agent_diagnostics: dict[str, JsonValue]
    selected_skill_names: list[str] = Field(default_factory=list, max_length=5)


class TurnStartResult(_MemoryModel):
    """Combined conversation context and cart for one turn."""

    turn_id: str = Field(..., min_length=1, max_length=256)
    attempt_id: str = Field(..., min_length=1, max_length=128)
    sequence: int = Field(..., ge=1)
    replayed: bool = False
    status: TurnStatus = "started"
    #: The audience most recently declared for who is being shopped for,
    #: carried forward so a wearer named one turn ago still has standing.
    wearer_audience: list[str] = Field(default_factory=list, max_length=8)
    assumed_audience: list[str] = Field(default_factory=list, max_length=8)
    recent_turns: list[RecentConversationTurn] = Field(
        default_factory=list,
        max_length=100,
    )
    previous_selected_skill_names: list[str] = Field(
        default_factory=list,
        max_length=5,
    )
    shopper_context: ShopperContext | None
    projection: ConversationProjection = Field(default_factory=ConversationProjection)
    cart: list[ConversationCartItem] = Field(default_factory=list)
    assistant_text: str | None = Field(default=None, max_length=100_000)
    termination_reason: str | None = Field(default=None, max_length=1_024)
    output: TurnReplayOutput | None = None


class ConversationEvent(_MemoryModel):
    """One append-only structured event submitted during finalization."""

    event_key: str = Field(..., min_length=1, max_length=256)
    event_type: EventType
    source_kind: EventSourceKind
    source_ref: str | None = Field(default=None, min_length=1, max_length=512)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class TurnFinalizeRequest(_MemoryModel):
    """Idempotent input for one durable turn finalization."""

    request_id: str = Field(..., min_length=1, max_length=128)
    attempt_id: str = Field(..., min_length=1, max_length=128)
    assistant_text: str = Field(..., max_length=100_000)
    status: FinalTurnStatus
    termination_reason: str | None = Field(default=None, max_length=1_024)
    events: list[ConversationEvent] = Field(default_factory=list, max_length=128)
    output: TurnReplayOutput | None = None


class TurnFinalizeResult(_MemoryModel):
    """Durable completion receipt for one conversation turn."""

    turn_id: str = Field(..., min_length=1, max_length=256)
    attempt_id: str = Field(..., min_length=1, max_length=128)
    sequence: int = Field(..., ge=1)
    replayed: bool = False
    status: FinalTurnStatus
    assistant_text: str = Field(..., max_length=100_000)
    termination_reason: str | None = Field(default=None, max_length=1_024)


class ConversationMemoryError(RuntimeError):
    """A stable failure at the chain-to-memory service boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


_ResponseModel = TypeVar("_ResponseModel", bound=_MemoryModel)


class ConversationMemoryClient:
    """Start and finalize durable conversation turns through the memory API."""

    def __init__(
        self,
        memory_retriever_url: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        self.memory_retriever_url = memory_retriever_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests

    def start_turn(
        self,
        conversation_id: str,
        *,
        request_id: str,
        shopper_text: str,
        media: Sequence[Mapping[str, Any]] = (),
        cart_user_id: int,
        shopper_profile_id: str | None = None,
        catalog_revision: str | None = None,
    ) -> TurnStartResult:
        """Start one turn without sending raw media to the memory service."""

        request = TurnStartRequest(
            request_id=request_id,
            shopper_text=shopper_text,
            cart_user_id=cart_user_id,
            request_digest=build_request_digest(
                shopper_text,
                media,
                shopper_profile_id=shopper_profile_id,
            ),
            shopper_profile_id=shopper_profile_id,
            catalog_revision=catalog_revision,
        )
        payload = self._post(
            f"/conversations/{_path_segment(conversation_id)}/turn/start",
            request.model_dump(mode="json"),
        )
        result = self._validate_response(payload, TurnStartResult)
        if (shopper_profile_id is None) != (result.shopper_context is None):
            raise ConversationMemoryError(
                "shopper_context_invalid",
                "Conversation memory returned mismatched shopper context.",
            )
        return result

    def finalize_turn(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        request_id: str,
        attempt_id: str,
        assistant_text: str,
        status: FinalTurnStatus,
        termination_reason: str | None,
        events: Sequence[ConversationEvent] = (),
        output: TurnReplayOutput | None = None,
    ) -> TurnFinalizeResult:
        """Finalize one turn with deterministic structured events."""

        request = TurnFinalizeRequest(
            request_id=request_id,
            attempt_id=attempt_id,
            assistant_text=assistant_text,
            status=status,
            termination_reason=termination_reason,
            events=list(events),
            output=output,
        )
        payload = self._post(
            (
                f"/conversations/{_path_segment(conversation_id)}/turns/"
                f"{_path_segment(turn_id)}/finalize"
            ),
            request.model_dump(mode="json", exclude_none=True),
        )
        return self._validate_response(payload, TurnFinalizeResult)

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = self.session.post(
                f"{self.memory_retriever_url}{path}",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ConversationMemoryError(
                "memory_request_failed",
                "Conversation memory request failed.",
                retryable=True,
            ) from exc

        status_code = int(getattr(response, "status_code", 500))
        if status_code >= 400:
            detail = ""
            try:
                error_payload = response.json()
                if isinstance(error_payload, dict):
                    detail = str(error_payload.get("detail") or "")
            except (TypeError, ValueError):
                pass
            raise _http_error(status_code, detail)
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise ConversationMemoryError(
                "memory_response_invalid",
                "Conversation memory returned an invalid response.",
                status_code=status_code,
            ) from exc

    @staticmethod
    def _validate_response(
        payload: Any,
        model: type[_ResponseModel],
    ) -> _ResponseModel:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise ConversationMemoryError(
                "memory_response_invalid",
                "Conversation memory returned an invalid response.",
            ) from exc


def build_request_digest(
    shopper_text: str,
    media: Sequence[Mapping[str, Any]] = (),
    *,
    shopper_profile_id: str | None = None,
) -> str:
    """Fingerprint shopper input, selected profile, and ordered media hashes.

    Guest requests retain the pre-profile canonical shape so finalized turns
    created before the shopper-profile migration remain exactly replayable.
    """

    media_fingerprints = [
        {
            "type": str(item.get("type") or ""),
            "mime_type": str(item.get("mime_type") or ""),
            "sha256": hashlib.sha256(_media_bytes(item.get("data"))).hexdigest(),
        }
        for item in media
    ]
    digest_payload: dict[str, Any] = {
        "shopper_text": shopper_text,
        "media": media_fingerprints,
    }
    if shopper_profile_id is not None:
        digest_payload["shopper_profile_id"] = shopper_profile_id
    canonical = _canonical_json(digest_payload)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def build_dialogue_context(
    recent_turns: Sequence[RecentConversationTurn],
    *,
    max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
) -> tuple[list[DialogueTurn], str]:
    """Select eligible turns within budget and render them in one pass.

    Returns the typed turns that were actually rendered together with their
    rendered text. Producing both from one selection is what stops the typed
    lane and the prompt from drifting apart, and removes any need to parse the
    rendered text back into state.
    """

    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    selected = sorted(
        (
            turn
            for turn in recent_turns
            if turn.status not in {"abandoned", "blocked"}
            and turn.assistant_text is not None
        ),
        key=lambda turn: turn.sequence,
    )
    if not selected:
        return [], ""

    heading = "RECENT CONVERSATION:"
    retained: list[tuple[DialogueTurn, str]] = []
    used = len(heading) + 1
    for turn in reversed(selected):
        typed, block = _render_recent_turn(turn)
        separator = 1 if retained else 0
        available = max_chars - used - separator
        if len(block) <= available:
            retained.append((typed, block))
            used += len(block) + separator
            continue
        if not retained and available >= 48:
            retained.append(_render_recent_turn(turn, max_chars=available))
        break

    if not retained:
        return [], heading
    ordered = list(reversed(retained))
    rendered = f"{heading}\n" + "\n".join(block for _, block in ordered)
    return [typed for typed, _ in ordered], rendered


def format_conversation_context(
    recent_turns: Sequence[RecentConversationTurn],
    *,
    max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
) -> str:
    """Render model-safe service-bounded turns without merging speaker lines."""

    return build_dialogue_context(recent_turns, max_chars=max_chars)[1]


def _render_recent_turn(
    turn: RecentConversationTurn,
    *,
    max_chars: int | None = None,
) -> tuple[DialogueTurn, str]:
    """Return one turn as typed context and as its exact rendered block."""

    shopper = _inline_text(turn.shopper_text) or "(empty)"
    assistant = _inline_text(turn.assistant_text or "") or "(none)"
    prefix = f"[turn {turn.sequence}]\nUser: "
    separator = "\nAssistant: "
    rendered = f"{prefix}{shopper}{separator}{assistant}"
    if max_chars is None or len(rendered) <= max_chars:
        return _dialogue_turn(turn.sequence, shopper, assistant), rendered

    value_budget = max_chars - len(prefix) - len(separator)
    if value_budget < 2:
        # Preserved edge case: the block is hard-sliced, so the typed lane keeps
        # the untruncated values rather than inventing a split that never rendered.
        return _dialogue_turn(turn.sequence, shopper, assistant), rendered[:max_chars]
    shopper_budget = max(1, value_budget // 2)
    assistant_budget = max(1, value_budget - shopper_budget)
    shopper = _truncate(shopper, shopper_budget)
    assistant = _truncate(assistant, assistant_budget)
    return (
        _dialogue_turn(turn.sequence, shopper, assistant),
        f"{prefix}{shopper}{separator}{assistant}",
    )


def _dialogue_turn(sequence: int, shopper: str, assistant: str) -> DialogueTurn:
    return DialogueTurn(
        sequence=sequence,
        shopper_text=shopper,
        assistant_text=assistant,
    )


def _http_error(status_code: int, detail: str = "") -> ConversationMemoryError:
    if status_code == 409:
        if detail == "conversation_profile_mismatch":
            return ConversationMemoryError(
                detail,
                "Conversation is bound to another shopper profile.",
                status_code=status_code,
            )
        if detail in {"turn_in_progress", "conversation_turn_in_progress"}:
            return ConversationMemoryError(
                detail,
                "Conversation turn is already in progress.",
                status_code=status_code,
                retryable=True,
            )
        if detail == "turn_abandoned":
            return ConversationMemoryError(
                detail,
                "The earlier conversation turn was interrupted.",
                status_code=status_code,
            )
        if detail in {"turn_superseded", "turn_attempt_superseded"}:
            return ConversationMemoryError(
                detail,
                "The conversation turn was superseded by a newer turn or attempt.",
                status_code=status_code,
            )
        return ConversationMemoryError(
            "memory_turn_conflict",
            "Conversation turn conflicts with an existing request.",
            status_code=status_code,
        )
    if status_code == 404 and detail == "shopper_profile_not_found":
        return ConversationMemoryError(
            detail,
            "Shopper profile was not found.",
            status_code=status_code,
        )
    if status_code in {400, 404, 422}:
        return ConversationMemoryError(
            "memory_request_invalid",
            "Conversation memory rejected the request.",
            status_code=status_code,
        )
    if status_code >= 500:
        return ConversationMemoryError(
            "memory_service_unavailable",
            "Conversation memory is unavailable.",
            status_code=status_code,
            retryable=True,
        )
    return ConversationMemoryError(
        "memory_http_error",
        "Conversation memory request failed.",
        status_code=status_code,
    )


def _path_segment(value: str) -> str:
    return quote(value, safe="")


def _media_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value or "").encode("utf-8")


def _inline_text(value: str) -> str:
    return " ".join(value.split())


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[:max_chars]
    return value[: max_chars - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
