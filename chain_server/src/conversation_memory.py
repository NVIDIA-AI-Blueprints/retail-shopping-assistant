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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from .agenttypes import SHOPPER_PROFILE_ID_PATTERN, ShopperContext
from shared.commerce_contracts import ProductSummary
from shared.weather_receipts import (
    MAX_ACTIVE_WEATHER_RECEIPTS,
    WeatherForecastReceipt,
    WeatherReceiptPromotion,
)
from shared.weather_scope import (
    MAX_CURRENT_WEATHER_SCOPE_SOURCE_TURNS,
    CurrentWeatherScope,
    CurrentWeatherScopeResolution,
    CurrentWeatherScopeSourceTurn,
    CurrentWeatherScopeTransition,
    current_weather_scope_source_references,
)


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
]

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_CONTEXT_MAX_CHARS = 16_384
_MAX_CONVERSATION_SUMMARY_CHARS = 16_384
_MEMORY_RESPONSE_CONTRACT = 5
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


class SummaryCompactionTurn(_MemoryModel):
    """One exact context-eligible turn in memory's oldest raw prefix."""

    sequence: int = Field(..., ge=1)
    shopper_text: str = Field(..., min_length=1, max_length=100_000)
    assistant_text: str = Field(..., max_length=100_000)
    status: Literal["completed", "failed"]


class SummaryCompactionSource(_MemoryModel):
    """Memory-owned exact oldest prefix eligible for one summary advance."""

    expected_projection_version: int = Field(..., ge=0)
    after_sequence: int = Field(..., ge=0)
    through_sequence: int = Field(..., ge=1)
    turns: list[SummaryCompactionTurn] = Field(..., min_length=1, max_length=100)

    @model_validator(mode="after")
    def _turns_match_boundary(self) -> "SummaryCompactionSource":
        sequences = [turn.sequence for turn in self.turns]
        if sequences != sorted(set(sequences)):
            raise ValueError("summary compaction turns must be strictly ordered")
        if (
            sequences[0] <= self.after_sequence
            or sequences[-1] != self.through_sequence
        ):
            raise ValueError("summary compaction turns must match their boundary")
        return self


class ConversationProjection(_MemoryModel):
    """Durable conversation-level summary and grounding projections."""

    version: int = Field(default=0, ge=0)
    summary_text: str = Field(
        default="",
        max_length=_MAX_CONVERSATION_SUMMARY_CHARS,
    )
    summary_through_sequence: int = Field(default=0, ge=0)
    active_anchors: list[JsonValue] = Field(default_factory=list, max_length=50)
    effective_preferences: list[JsonValue] = Field(
        default_factory=list,
        max_length=100,
    )
    product_reference_index: list[JsonValue] = Field(
        default_factory=list,
        max_length=100,
    )
    active_receipts: list[WeatherForecastReceipt] = Field(
        default_factory=list,
        max_length=MAX_ACTIVE_WEATHER_RECEIPTS,
    )
    current_weather_scope: CurrentWeatherScope = Field(
        default_factory=CurrentWeatherScope
    )
    last_turn_id: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("summary_text")
    @classmethod
    def _require_trimmed_summary(cls, value: str) -> str:
        if value and value != value.strip():
            raise ValueError("summary_text must not contain outer whitespace")
        return value

    @model_validator(mode="after")
    def _summary_pair_is_consistent(self) -> "ConversationProjection":
        if bool(self.summary_text) != (self.summary_through_sequence > 0):
            raise ValueError(
                "summary_text and summary_through_sequence must be set together"
            )
        return self


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
    contract_version: Literal[1, 2, 3, 4, 5] = 1
    attempt_id: str = Field(..., min_length=1, max_length=128)
    sequence: int = Field(..., ge=1)
    replayed: bool = False
    status: TurnStatus = "started"
    recent_turns: list[RecentConversationTurn] = Field(
        default_factory=list,
        max_length=100,
    )
    unsummarized_turn_count: int = Field(default=0, ge=0)
    summary_compaction_source: SummaryCompactionSource | None = None
    current_weather_scope_source_turns: list[
        CurrentWeatherScopeSourceTurn
    ] = Field(
        default_factory=list,
        max_length=MAX_CURRENT_WEATHER_SCOPE_SOURCE_TURNS,
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

    @model_validator(mode="before")
    @classmethod
    def _contract_lanes_match_version(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        version = value.get("contract_version", 1)
        if not isinstance(version, int):
            return value
        projection = value.get("projection")
        projection_keys = (
            set(projection) if isinstance(projection, Mapping) else set()
        )
        v2_top_level = {"unsummarized_turn_count", "summary_compaction_source"}
        v2_projection = {
            "summary_text",
            "summary_through_sequence",
            "active_receipts",
        }
        if version < 2 and (
            v2_top_level.intersection(value)
            or v2_projection.intersection(projection_keys)
        ):
            raise ValueError("memory contract v1 response contains v2-only lanes")
        if version < 3 and "current_weather_scope" in projection_keys:
            raise ValueError(
                "memory response contains a weather scope before contract v3"
            )
        if version >= 3 and "current_weather_scope" not in projection_keys:
            raise ValueError("memory contract v3 response omitted weather scope")
        if (
            version < 5
            and "current_weather_scope_source_turns" in value
        ):
            raise ValueError(
                "memory response contains weather scope sources before "
                "contract v5"
            )
        if (
            version >= 5
            and "current_weather_scope_source_turns" not in value
        ):
            raise ValueError(
                "memory contract v5 response omitted weather scope sources"
            )
        return value

    @model_validator(mode="after")
    def _summary_sources_are_consistent(self) -> "TurnStartResult":
        watermark = self.projection.summary_through_sequence
        recent_sequences = [turn.sequence for turn in self.recent_turns]
        if recent_sequences != sorted(set(recent_sequences)):
            raise ValueError(
                "recent conversation turns must be strictly ordered"
            )
        if self.contract_version >= 2 and any(
            turn.sequence <= watermark
            or turn.status not in {"completed", "failed"}
            or turn.assistant_text is None
            for turn in self.recent_turns
        ):
            raise ValueError(
                "recent conversation turns must be eligible and post-summary"
            )
        source = self.summary_compaction_source
        if source is not None and (
            source.expected_projection_version != self.projection.version
            or source.after_sequence != watermark
            or source.through_sequence >= self.sequence
        ):
            raise ValueError(
                "summary compaction source must match the turn projection"
            )
        source_length = len(source.turns) if source is not None else 0
        if self.unsummarized_turn_count < source_length:
            raise ValueError(
                "unsummarized_turn_count cannot be smaller than compaction source"
            )
        if self.contract_version >= 5:
            source_references = [
                (turn.turn_id, turn.sequence)
                for turn in self.current_weather_scope_source_turns
            ]
            if source_references != sorted(
                set(source_references),
                key=lambda item: (item[1], item[0]),
            ):
                raise ValueError(
                    "weather scope source turns must be strictly ordered"
                )
            if set(source_references) != set(
                current_weather_scope_source_references(
                    self.projection.current_weather_scope
                )
            ):
                raise ValueError(
                    "weather scope source turns must exactly match the scope"
                )
            if not self.replayed and any(
                turn.sequence >= self.sequence
                for turn in self.current_weather_scope_source_turns
            ):
                raise ValueError(
                    "weather scope source turns must precede a new active turn"
                )
        return self


class ConversationEvent(_MemoryModel):
    """One append-only structured event submitted during finalization."""

    event_key: str = Field(..., min_length=1, max_length=256)
    event_type: EventType
    source_kind: EventSourceKind
    source_ref: str | None = Field(default=None, min_length=1, max_length=512)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class ConversationSummaryAdvance(_MemoryModel):
    """One complete compare-and-swap update to the rolling summary boundary."""

    expected_projection_version: int = Field(..., ge=0)
    summary_text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_CONVERSATION_SUMMARY_CHARS,
    )
    summary_through_sequence: int = Field(..., ge=1)

    @field_validator("summary_text")
    @classmethod
    def _require_trimmed_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary_text must not contain outer whitespace")
        return value


class TurnFinalizeRequest(_MemoryModel):
    """Idempotent input for one durable turn finalization."""

    request_id: str = Field(..., min_length=1, max_length=128)
    attempt_id: str = Field(..., min_length=1, max_length=128)
    assistant_text: str = Field(..., max_length=100_000)
    status: FinalTurnStatus
    termination_reason: str | None = Field(default=None, max_length=1_024)
    events: list[ConversationEvent] = Field(default_factory=list, max_length=128)
    output: TurnReplayOutput | None = None
    summary_advance: ConversationSummaryAdvance | None = None
    weather_receipt_promotion: WeatherReceiptPromotion | None = None
    current_weather_scope_transition: CurrentWeatherScopeTransition | None = None
    current_weather_scope_resolution: CurrentWeatherScopeResolution | None = None

    @model_validator(mode="after")
    def _weather_scope_updates_are_mutually_exclusive(
        self,
    ) -> "TurnFinalizeRequest":
        if (
            self.current_weather_scope_transition is not None
            and self.current_weather_scope_resolution is not None
        ):
            raise ValueError(
                "legacy weather transition and atomic resolution are "
                "mutually exclusive"
            )
        return self


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
            (
                f"/conversations/{_path_segment(conversation_id)}/turn/start"
                f"?response_contract={_MEMORY_RESPONSE_CONTRACT}"
            ),
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
        summary_advance: ConversationSummaryAdvance | None = None,
        weather_receipt_promotion: WeatherReceiptPromotion | None = None,
        current_weather_scope_transition: (
            CurrentWeatherScopeTransition | None
        ) = None,
        current_weather_scope_resolution: (
            CurrentWeatherScopeResolution | None
        ) = None,
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
            summary_advance=summary_advance,
            weather_receipt_promotion=weather_receipt_promotion,
            current_weather_scope_transition=current_weather_scope_transition,
            current_weather_scope_resolution=current_weather_scope_resolution,
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


def format_conversation_context(
    recent_turns: Sequence[RecentConversationTurn],
    *,
    max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
) -> str:
    """Render model-safe service-bounded turns without merging speaker lines."""

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
        return ""
    return _format_recent_turns(selected, max_chars)


def _format_recent_turns(
    turns: Sequence[RecentConversationTurn],
    max_chars: int,
) -> str:
    heading = "RECENT CONVERSATION:"
    if not turns:
        return f"{heading}\n(none)"

    retained: list[str] = []
    used = len(heading) + 1
    for turn in reversed(turns):
        block = _format_recent_turn(turn)
        separator = 1 if retained else 0
        available = max_chars - used - separator
        if len(block) <= available:
            retained.append(block)
            used += len(block) + separator
            continue
        if not retained and available >= 48:
            retained.append(_format_recent_turn(turn, max_chars=available))
        break

    if not retained:
        return heading
    return f"{heading}\n" + "\n".join(reversed(retained))


def _format_recent_turn(
    turn: RecentConversationTurn,
    *,
    max_chars: int | None = None,
) -> str:
    shopper = _inline_text(turn.shopper_text) or "(empty)"
    assistant = _inline_text(turn.assistant_text or "") or "(none)"
    prefix = f"[turn {turn.sequence}]\nUser: "
    separator = "\nAssistant: "
    rendered = f"{prefix}{shopper}{separator}{assistant}"
    if max_chars is None or len(rendered) <= max_chars:
        return rendered

    value_budget = max_chars - len(prefix) - len(separator)
    if value_budget < 2:
        return rendered[:max_chars]
    shopper_budget = max(1, value_budget // 2)
    assistant_budget = max(1, value_budget - shopper_budget)
    return (
        f"{prefix}{_truncate(shopper, shopper_budget)}"
        f"{separator}{_truncate(assistant, assistant_budget)}"
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
        if detail in {
            "projection_version_conflict",
            "summary_boundary_conflict",
        }:
            return ConversationMemoryError(
                detail,
                "Conversation summary state changed before finalization.",
                status_code=status_code,
                retryable=True,
            )
        if detail in {
            "weather_receipt_stale",
            "weather_receipt_status_conflict",
            "weather_receipt_scope_conflict",
        }:
            return ConversationMemoryError(
                detail,
                "Optional weather receipt could not be promoted.",
                status_code=status_code,
            )
        if detail in {
            "current_weather_scope_revision_conflict",
            "current_weather_scope_resolution_conflict",
            "current_weather_scope_status_conflict",
            "current_weather_scope_saved_area_unavailable",
        }:
            return ConversationMemoryError(
                detail,
                "Current weather scope could not be finalized atomically.",
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
