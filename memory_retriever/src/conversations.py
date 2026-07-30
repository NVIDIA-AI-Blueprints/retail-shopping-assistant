# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable conversation-turn API for the memory service."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import func, text

from .models import (
    CartItem,
    ConversationEvent,
    ConversationProjection,
    ConversationTurn,
    ShopperProfile,
)
from .product_references import (
    PRESENTED_PRODUCTS_EVENT_KEY,
    ProductResolutionRequest,
    append_presented_products_event,
    rebuild_product_reference_index,
    resolve_product_references,
)
from .shopper_profiles import SHOPPER_PROFILE_ID_PATTERN
from shared.weather_receipts import (
    MAX_ACTIVE_WEATHER_RECEIPTS,
    WEATHER_TOOL_NAME,
    WeatherForecastReceipt,
    WeatherReceiptPromotion,
    weather_receipt_id,
    weather_scope_key,
)


DEFAULT_ABANDONED_SECONDS = 300
DEFAULT_RECENT_TURNS_LIMIT = 8
SUMMARY_COMPACTION_SOURCE_TURNS = 4
MAX_RECENT_TURNS_LIMIT = 50
MAX_CONVERSATION_SUMMARY_CHARS = 16_384
SHOPPER_PROFILE_NOT_FOUND = "shopper_profile_not_found"
CONVERSATION_PROFILE_MISMATCH = "conversation_profile_mismatch"
REQUEST_INPUT_CONFLICT = "request_id was already used for different turn input"
PROJECTION_VERSION_CONFLICT = "projection_version_conflict"
SUMMARY_BOUNDARY_CONFLICT = "summary_boundary_conflict"
WEATHER_RECEIPT_STATUS_CONFLICT = "weather_receipt_status_conflict"
WEATHER_RECEIPT_STALE = "weather_receipt_stale"
MEMORY_RESPONSE_CONTRACT_V2 = 2


class TurnStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=128)
    shopper_text: str = Field(..., min_length=1, max_length=100_000)
    cart_user_id: int = Field(..., ge=0)
    request_digest: str = Field(..., min_length=1, max_length=128)
    catalog_revision: str | None = Field(default=None, max_length=512)
    shopper_profile_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=SHOPPER_PROFILE_ID_PATTERN,
    )


class ShopperContext(BaseModel):
    """Profile guidance resolved from the immutable memory-service registry."""

    model_config = ConfigDict(extra="forbid", from_attributes=True, strict=True)

    shopper_type: str = Field(
        ...,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    behavior: str = Field(..., min_length=1, max_length=512)
    zipcode: str = Field(..., pattern=r"^[0-9]{5}$")

    @field_validator("shopper_type", "behavior", "zipcode")
    @classmethod
    def _reject_outer_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("shopper context must not contain outer whitespace")
        return value

    @field_validator("behavior")
    @classmethod
    def _require_single_line_behavior(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("shopper behavior must be one line")
        return value


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


class ConversationEventInput(BaseModel):
    event_key: str = Field(..., min_length=1, max_length=256)
    event_type: EventType
    source_kind: Literal["shopper", "compiler", "catalog", "cart", "runtime"]
    source_ref: str | None = Field(default=None, max_length=512)
    payload: dict[str, Any] = Field(default_factory=dict)


class TurnReplayOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_results: list[dict[str, Any]]
    retrieved: dict[str, str]
    agent_diagnostics: dict[str, Any]
    selected_skill_names: list[str] = Field(default_factory=list, max_length=5)


class ConversationSummaryAdvance(BaseModel):
    """One complete compare-and-swap update to the rolling summary boundary."""

    model_config = ConfigDict(extra="forbid")

    expected_projection_version: int = Field(..., ge=0)
    summary_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONVERSATION_SUMMARY_CHARS,
    )
    summary_through_sequence: int = Field(..., ge=1)

    @field_validator("summary_text")
    @classmethod
    def _require_trimmed_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary_text must not contain outer whitespace")
        return value


class TurnFinalizeRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=128)
    attempt_id: str = Field(..., min_length=1, max_length=128)
    assistant_text: str = Field(..., max_length=100_000)
    status: Literal["completed", "failed", "blocked"]
    termination_reason: str | None = Field(default=None, max_length=1024)
    events: list[ConversationEventInput] = Field(
        default_factory=list,
        max_length=128,
    )
    output: TurnReplayOutput | None = None
    summary_advance: ConversationSummaryAdvance | None = None
    weather_receipt_promotion: WeatherReceiptPromotion | None = None

    @model_validator(mode="after")
    def _event_keys_are_unique(self):
        keys = [event.event_key for event in self.events]
        if len(keys) != len(set(keys)):
            raise ValueError("event_key values must be unique within a turn")
        if PRESENTED_PRODUCTS_EVENT_KEY in keys:
            raise ValueError("event_key is reserved for runtime product cards")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _finalize_digest(request: TurnFinalizeRequest) -> str:
    payload = request.model_dump(mode="json")
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_conversation_id(conversation_id: str) -> None:
    if (
        not conversation_id
        or len(conversation_id) > 256
        or conversation_id != conversation_id.strip()
        or any(ord(character) < 32 for character in conversation_id)
    ):
        raise HTTPException(status_code=422, detail="Invalid conversation_id")


def _active_weather_receipts(
    projection: ConversationProjection,
    *,
    current_time: float,
) -> list[WeatherForecastReceipt]:
    """Return only valid, fresh, uniquely scoped bounded receipts."""

    try:
        raw_receipts = json.loads(projection.active_receipts_json or "[]")
    except (TypeError, ValueError, RecursionError):
        raw_receipts = []
    if not isinstance(raw_receipts, list):
        raw_receipts = []

    now = datetime.fromtimestamp(current_time, tz=timezone.utc)
    newest_by_scope: dict[str, WeatherForecastReceipt] = {}
    for raw_receipt in raw_receipts:
        try:
            receipt = WeatherForecastReceipt.model_validate_json(
                _canonical_json(raw_receipt)
            )
        except (TypeError, ValueError, ValidationError):
            continue
        if receipt.valid_until <= now:
            continue
        previous = newest_by_scope.get(receipt.scope_key)
        if previous is None or _receipt_order_key(receipt) > (
            _receipt_order_key(previous)
        ):
            newest_by_scope[receipt.scope_key] = receipt

    return sorted(
        newest_by_scope.values(),
        key=_receipt_order_key,
        reverse=True,
    )[:MAX_ACTIVE_WEATHER_RECEIPTS]


def _receipt_order_key(
    receipt: WeatherForecastReceipt,
) -> tuple[int, datetime, str]:
    return (
        receipt.source_sequence,
        receipt.evidence.fetched_at,
        receipt.receipt_id,
    )


def _store_active_weather_receipts(
    projection: ConversationProjection,
    receipts: list[WeatherForecastReceipt],
) -> None:
    projection.active_receipts_json = _canonical_json(
        [
            receipt.model_dump(mode="json", exclude_none=True)
            for receipt in receipts
        ]
    )


def _projection_dict(
    projection: ConversationProjection,
    *,
    current_time: float,
    include_v2_fields: bool = True,
) -> dict[str, Any]:
    summary_text = projection.summary_text or ""
    summary_through_sequence = projection.summary_through_sequence or 0
    if (not summary_text) != (summary_through_sequence == 0):
        raise HTTPException(
            status_code=500,
            detail="conversation_summary_projection_invalid",
        )
    result = {
        "version": projection.version,
        "active_anchors": json.loads(projection.active_anchors_json),
        "effective_preferences": json.loads(projection.effective_preferences_json),
        "product_reference_index": json.loads(projection.product_reference_index_json),
        "last_turn_id": projection.last_turn_id,
    }
    if include_v2_fields:
        result.update(
            {
                "summary_text": summary_text,
                "summary_through_sequence": summary_through_sequence,
                "active_receipts": [
                    receipt.model_dump(mode="json", exclude_none=True)
                    for receipt in _active_weather_receipts(
                        projection,
                        current_time=current_time,
                    )
                ],
            }
        )
    return result


def _get_or_create_projection(
    db,
    conversation_id: str,
) -> ConversationProjection:
    projection = (
        db.query(ConversationProjection)
        .filter_by(conversation_id=conversation_id)
        .first()
    )
    if projection is None:
        projection = ConversationProjection(conversation_id=conversation_id)
        db.add(projection)
        db.flush()
    return projection


def _cart_item_dict(item: CartItem) -> dict[str, Any]:
    result = {
        "cart_line_id": item.cart_line_id,
        "item": item.item,
        "amount": item.amount,
        "price": item.price,
    }
    if item.product_id:
        result["product_id"] = item.product_id
    return result


def _cart_for_user(db, user_id: int) -> list[dict[str, Any]]:
    items = (
        db.query(CartItem)
        .filter(CartItem.user_id == user_id)
        .order_by(CartItem.id)
        .all()
    )
    return [_cart_item_dict(item) for item in items]


def _recent_turns_limit() -> int:
    configured = int(
        os.environ.get(
            "MEMORY_RECENT_TURNS",
            str(DEFAULT_RECENT_TURNS_LIMIT),
        )
    )
    return min(MAX_RECENT_TURNS_LIMIT, max(1, configured))


def _recent_turns(
    db,
    conversation_id: str,
    *,
    after_sequence: int,
    before_sequence: int,
) -> list[dict[str, Any]]:
    """Return a bounded raw tail strictly after the durable summary boundary."""

    rows = (
        db.query(ConversationTurn)
        .filter(
            ConversationTurn.conversation_id == conversation_id,
            ConversationTurn.sequence > after_sequence,
            ConversationTurn.sequence < before_sequence,
            ConversationTurn.status.in_(("completed", "failed")),
            ConversationTurn.assistant_text.is_not(None),
        )
        .order_by(ConversationTurn.sequence.desc())
        .limit(_recent_turns_limit())
        .all()
    )
    return [_context_turn_dict(row) for row in reversed(rows)]


def _context_turn_dict(turn: ConversationTurn) -> dict[str, Any]:
    return {
        "sequence": turn.sequence,
        "shopper_text": turn.shopper_text,
        "assistant_text": turn.assistant_text,
        "status": turn.status,
    }


def _summary_compaction_state(
    db,
    conversation_id: str,
    *,
    after_sequence: int,
    before_sequence: int,
    projection_version: int,
) -> tuple[int, dict[str, Any] | None]:
    """Return the count and bounded oldest prefix of unsummarized raw turns."""

    eligibility = (
        ConversationTurn.conversation_id == conversation_id,
        ConversationTurn.sequence > after_sequence,
        ConversationTurn.sequence < before_sequence,
        ConversationTurn.status.in_(("completed", "failed")),
        ConversationTurn.assistant_text.is_not(None),
    )
    count = (
        db.query(func.count(ConversationTurn.turn_id))
        .filter(*eligibility)
        .scalar()
        or 0
    )
    rows = (
        db.query(ConversationTurn)
        .filter(*eligibility)
        .order_by(ConversationTurn.sequence.asc())
        .limit(SUMMARY_COMPACTION_SOURCE_TURNS)
        .all()
    )
    if not rows:
        return int(count), None
    return int(count), {
        "expected_projection_version": projection_version,
        "after_sequence": after_sequence,
        "through_sequence": rows[-1].sequence,
        "turns": [_context_turn_dict(row) for row in rows],
    }


def _turn_selected_skill_names(turn: ConversationTurn) -> list[str]:
    """Read the typed skill-selection hint from one finalized turn."""

    if not turn.output_json:
        return []
    try:
        output = json.loads(turn.output_json)
    except (TypeError, ValueError):
        return []
    names = output.get("selected_skill_names") if isinstance(output, dict) else None
    if not isinstance(names, list):
        return []
    return [name for name in names[:5] if isinstance(name, str) and name.strip()]


def _previous_selected_skill_names(
    db,
    turn: ConversationTurn,
) -> list[str]:
    """Return only the immediately preceding eligible turn's skill hint."""

    if turn.sequence <= 1:
        return []
    previous = (
        db.query(ConversationTurn)
        .filter_by(
            conversation_id=turn.conversation_id,
            sequence=turn.sequence - 1,
        )
        .first()
    )
    if previous is None or previous.status in {"started", "blocked", "abandoned"}:
        return []
    return _turn_selected_skill_names(previous)


def _shopper_context(
    profile: ShopperProfile | None,
) -> dict[str, str] | None:
    if profile is None:
        return None
    try:
        return ShopperContext.model_validate(profile).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail="shopper_context_invalid",
        ) from exc


def _start_response(
    db,
    turn: ConversationTurn,
    projection: ConversationProjection,
    shopper_profile: ShopperProfile | None,
    *,
    replayed: bool,
    current_time: float,
    response_contract: Literal[1, 2],
) -> dict[str, Any]:
    v2_response = response_contract == MEMORY_RESPONSE_CONTRACT_V2
    result = {
        "turn_id": turn.turn_id,
        "attempt_id": turn.attempt_id,
        "sequence": turn.sequence,
        "replayed": replayed,
        "status": turn.status,
        "recent_turns": _recent_turns(
            db,
            turn.conversation_id,
            after_sequence=(
                projection.summary_through_sequence if v2_response else 0
            ),
            before_sequence=turn.sequence,
        ),
        "previous_selected_skill_names": _previous_selected_skill_names(db, turn),
        "projection": _projection_dict(
            projection,
            current_time=current_time,
            include_v2_fields=v2_response,
        ),
        "cart": _cart_for_user(db, turn.cart_user_id),
        "assistant_text": turn.assistant_text,
        "termination_reason": turn.termination_reason,
        "output": json.loads(turn.output_json) if turn.output_json else None,
        "shopper_context": _shopper_context(shopper_profile),
    }
    if v2_response:
        unsummarized_turn_count, summary_compaction_source = (
            _summary_compaction_state(
                db,
                turn.conversation_id,
                after_sequence=projection.summary_through_sequence,
                before_sequence=turn.sequence,
                projection_version=projection.version,
            )
        )
        result.update(
            {
                "contract_version": MEMORY_RESPONSE_CONTRACT_V2,
                "unsummarized_turn_count": unsummarized_turn_count,
                "summary_compaction_source": summary_compaction_source,
            }
        )
    return result


def _finalize_response(
    turn: ConversationTurn,
    *,
    replayed: bool,
) -> dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "attempt_id": turn.attempt_id,
        "sequence": turn.sequence,
        "replayed": replayed,
        "status": turn.status,
        "assistant_text": turn.assistant_text,
        "termination_reason": turn.termination_reason,
    }


def _mark_stale_started_turns(
    db,
    *,
    current_time: float,
    timeout_seconds: int,
    termination_reason: str,
    conversation_id: str | None = None,
) -> int:
    query = db.query(ConversationTurn).filter(
        ConversationTurn.status == "started",
        ConversationTurn.started_at < current_time - timeout_seconds,
    )
    if conversation_id is not None:
        query = query.filter(ConversationTurn.conversation_id == conversation_id)
    turns = query.all()
    for turn in turns:
        turn.status = "abandoned"
        turn.termination_reason = termination_reason
        turn.completed_at = current_time
    return len(turns)


def _restart_abandoned_turn(
    turn: ConversationTurn,
    request: TurnStartRequest,
    *,
    started_at: float,
) -> None:
    turn.status = "started"
    turn.attempt_id = uuid4().hex
    turn.started_at = started_at
    turn.completed_at = None
    turn.assistant_text = None
    turn.termination_reason = None
    turn.finalize_digest = None
    turn.output_json = None
    turn.catalog_revision = request.catalog_revision


def _resolve_shopper_profile(
    db,
    shopper_profile_id: str | None,
) -> ShopperProfile | None:
    if shopper_profile_id is None:
        return None
    profile = (
        db.query(ShopperProfile)
        .filter_by(shopper_profile_id=shopper_profile_id)
        .first()
    )
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=SHOPPER_PROFILE_NOT_FOUND,
        )
    return profile


def _require_conversation_profile(
    db,
    conversation_id: str,
    shopper_profile_id: str | None,
) -> None:
    bindings = {
        row[0]
        for row in db.query(ConversationTurn.shopper_profile_id)
        .filter(ConversationTurn.conversation_id == conversation_id)
        .distinct()
        .all()
    }
    if bindings and bindings != {shopper_profile_id}:
        raise HTTPException(
            status_code=409,
            detail=CONVERSATION_PROFILE_MISMATCH,
        )


def _start_turn(
    db,
    conversation_id: str,
    request: TurnStartRequest,
    *,
    response_contract: Literal[1, 2],
) -> dict[str, Any]:
    db.execute(text("BEGIN IMMEDIATE"))
    current_time = time.time()
    _mark_stale_started_turns(
        db,
        current_time=current_time,
        timeout_seconds=abandoned_timeout_seconds(),
        termination_reason="turn_start_abandoned_timeout",
        conversation_id=conversation_id,
    )
    existing = (
        db.query(ConversationTurn)
        .filter_by(
            conversation_id=conversation_id,
            request_id=request.request_id,
        )
        .first()
    )
    if existing is not None:
        if (
            existing.request_digest != request.request_digest
            or existing.shopper_text != request.shopper_text
            or existing.cart_user_id != request.cart_user_id
            or existing.shopper_profile_id != request.shopper_profile_id
        ):
            raise HTTPException(
                status_code=409,
                detail=REQUEST_INPUT_CONFLICT,
            )
        shopper_profile = _resolve_shopper_profile(
            db,
            request.shopper_profile_id,
        )
        _require_conversation_profile(
            db,
            conversation_id,
            request.shopper_profile_id,
        )
        if existing.status == "started":
            raise HTTPException(status_code=409, detail="turn_in_progress")
        if existing.status == "abandoned":
            latest_sequence = (
                db.query(func.max(ConversationTurn.sequence))
                .filter(ConversationTurn.conversation_id == conversation_id)
                .scalar()
            )
            if existing.sequence != latest_sequence:
                raise HTTPException(status_code=409, detail="turn_superseded")
            active = (
                db.query(ConversationTurn)
                .filter_by(
                    conversation_id=conversation_id,
                    status="started",
                )
                .first()
            )
            if active is not None:
                raise HTTPException(
                    status_code=409,
                    detail="conversation_turn_in_progress",
                )
            _restart_abandoned_turn(existing, request, started_at=current_time)
            projection = _get_or_create_projection(db, conversation_id)
            response = _start_response(
                db,
                existing,
                projection,
                shopper_profile,
                replayed=False,
                current_time=current_time,
                response_contract=response_contract,
            )
            db.commit()
            return response
        projection = _get_or_create_projection(db, conversation_id)
        response = _start_response(
            db,
            existing,
            projection,
            shopper_profile,
            replayed=True,
            current_time=current_time,
            response_contract=response_contract,
        )
        db.commit()
        return response

    shopper_profile = _resolve_shopper_profile(
        db,
        request.shopper_profile_id,
    )
    _require_conversation_profile(
        db,
        conversation_id,
        request.shopper_profile_id,
    )
    active = (
        db.query(ConversationTurn)
        .filter_by(
            conversation_id=conversation_id,
            status="started",
        )
        .first()
    )
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail="conversation_turn_in_progress",
        )

    sequence = (
        db.query(func.max(ConversationTurn.sequence))
        .filter(ConversationTurn.conversation_id == conversation_id)
        .scalar()
        or 0
    ) + 1
    turn = ConversationTurn(
        turn_id=uuid4().hex,
        conversation_id=conversation_id,
        sequence=sequence,
        request_id=request.request_id,
        request_digest=request.request_digest,
        attempt_id=uuid4().hex,
        cart_user_id=request.cart_user_id,
        shopper_profile_id=request.shopper_profile_id,
        shopper_text=request.shopper_text,
        status="started",
        catalog_revision=request.catalog_revision,
        started_at=current_time,
    )
    db.add(turn)
    db.flush()
    projection = _get_or_create_projection(db, conversation_id)
    response = _start_response(
        db,
        turn,
        projection,
        shopper_profile,
        replayed=False,
        current_time=current_time,
        response_contract=response_contract,
    )
    db.commit()
    return response


def _append_events(
    db,
    turn: ConversationTurn,
    events: list[ConversationEventInput],
    now: float,
) -> None:
    next_order = (
        db.query(func.max(ConversationEvent.logical_order))
        .filter(ConversationEvent.turn_id == turn.turn_id)
        .scalar()
        or 0
    ) + 1
    existing_keys = {
        row[0]
        for row in db.query(ConversationEvent.event_key)
        .filter(ConversationEvent.turn_id == turn.turn_id)
        .all()
    }
    if existing_keys.intersection(event.event_key for event in events):
        raise HTTPException(status_code=409, detail="event_key already exists")
    for offset, event in enumerate(events):
        db.add(
            ConversationEvent(
                event_id=uuid4().hex,
                turn_id=turn.turn_id,
                event_key=event.event_key,
                logical_order=next_order + offset,
                event_type=event.event_type,
                source_kind=event.source_kind,
                source_ref=event.source_ref,
                payload_json=_canonical_json(event.payload),
                created_at=now,
            )
        )


def _validate_summary_advance(
    db,
    turn: ConversationTurn,
    projection: ConversationProjection,
    advance: ConversationSummaryAdvance | None,
) -> None:
    if advance is None:
        return
    if advance.expected_projection_version != projection.version:
        raise HTTPException(
            status_code=409,
            detail=PROJECTION_VERSION_CONFLICT,
        )
    _count, source = _summary_compaction_state(
        db,
        turn.conversation_id,
        after_sequence=projection.summary_through_sequence,
        before_sequence=turn.sequence,
        projection_version=projection.version,
    )
    if (
        source is None
        or advance.summary_through_sequence != source["through_sequence"]
    ):
        raise HTTPException(
            status_code=409,
            detail=SUMMARY_BOUNDARY_CONFLICT,
        )


def _prepare_weather_receipt(
    turn: ConversationTurn,
    projection: ConversationProjection,
    request: TurnFinalizeRequest,
    *,
    current_time: float,
) -> WeatherForecastReceipt | None:
    """Validate one promotion and stamp its conversation-owned identity."""

    promotion = request.weather_receipt_promotion
    if promotion is None:
        return None
    if request.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=WEATHER_RECEIPT_STATUS_CONFLICT,
        )
    if promotion.expected_projection_version != projection.version:
        raise HTTPException(
            status_code=409,
            detail=PROJECTION_VERSION_CONFLICT,
        )
    if (
        promotion.location_scope.kind == "confirmed_saved_zip"
        and turn.shopper_profile_id is None
    ):
        raise HTTPException(
            status_code=409,
            detail=WEATHER_RECEIPT_STATUS_CONFLICT,
        )

    valid_until = promotion.evidence.fetched_at + timedelta(
        seconds=promotion.ttl_seconds
    )
    now = datetime.fromtimestamp(current_time, tz=timezone.utc)
    if valid_until <= now:
        raise HTTPException(
            status_code=409,
            detail=WEATHER_RECEIPT_STALE,
        )

    scope_key = weather_scope_key(
        promotion.location_scope,
        promotion.evidence,
    )
    return WeatherForecastReceipt(
        receipt_id=weather_receipt_id(
            source_turn_id=turn.turn_id,
            source_tool_call_id=promotion.source_tool_call_id,
            scope_key=scope_key,
            fetched_at=promotion.evidence.fetched_at,
        ),
        scope_key=scope_key,
        source_turn_id=turn.turn_id,
        source_sequence=turn.sequence,
        source_tool=WEATHER_TOOL_NAME,
        source_tool_call_id=promotion.source_tool_call_id,
        location_scope=promotion.location_scope,
        evidence=promotion.evidence,
        valid_until=valid_until,
    )


def _advance_active_weather_receipts(
    projection: ConversationProjection,
    receipt: WeatherForecastReceipt | None,
    *,
    current_time: float,
) -> None:
    """Prune, exact-scope upsert, and cap the active receipt projection."""

    active = _active_weather_receipts(
        projection,
        current_time=current_time,
    )
    if receipt is not None:
        active = [
            existing
            for existing in active
            if existing.scope_key != receipt.scope_key
        ]
        active.append(receipt)
    active.sort(key=_receipt_order_key, reverse=True)
    _store_active_weather_receipts(
        projection,
        active[:MAX_ACTIVE_WEATHER_RECEIPTS],
    )


def _finalize_turn(
    db,
    conversation_id: str,
    turn_id: str,
    request: TurnFinalizeRequest,
) -> dict[str, Any]:
    db.execute(text("BEGIN IMMEDIATE"))
    turn = (
        db.query(ConversationTurn)
        .filter_by(
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        .first()
    )
    if turn is None:
        raise HTTPException(status_code=404, detail="Conversation turn not found")
    if turn.request_id != request.request_id:
        raise HTTPException(
            status_code=409,
            detail="request_id does not match the conversation turn",
        )
    if turn.attempt_id != request.attempt_id:
        raise HTTPException(status_code=409, detail="turn_attempt_superseded")

    digest = _finalize_digest(request)
    if turn.finalize_digest is not None:
        if turn.finalize_digest != digest:
            raise HTTPException(
                status_code=409,
                detail="Conversation turn was finalized with different data",
            )
        response = _finalize_response(turn, replayed=True)
        db.commit()
        return response
    if turn.status != "started":
        raise HTTPException(
            status_code=409,
            detail=f"Conversation turn cannot be finalized from {turn.status}",
        )

    projection = _get_or_create_projection(db, conversation_id)
    now = time.time()
    _validate_summary_advance(
        db,
        turn,
        projection,
        request.summary_advance,
    )
    weather_receipt = _prepare_weather_receipt(
        turn,
        projection,
        request,
        current_time=now,
    )
    _append_events(db, turn, request.events, now)
    db.flush()
    append_presented_products_event(
        db,
        turn,
        request.output.product_results if request.output is not None else [],
        created_at=now,
    )
    db.flush()
    rebuild_product_reference_index(db, projection)
    if request.summary_advance is not None:
        projection.summary_text = request.summary_advance.summary_text
        projection.summary_through_sequence = (
            request.summary_advance.summary_through_sequence
        )
    _advance_active_weather_receipts(
        projection,
        weather_receipt,
        current_time=now,
    )
    projection.version += 1
    projection.last_turn_id = turn.turn_id
    turn.assistant_text = request.assistant_text
    turn.output_json = (
        _canonical_json(request.output.model_dump(mode="json"))
        if request.output is not None
        else None
    )
    turn.status = request.status
    turn.termination_reason = request.termination_reason
    turn.finalize_digest = digest
    turn.completed_at = now
    db.flush()
    response = _finalize_response(turn, replayed=False)
    db.commit()
    return response


def _delete_conversation(db, conversation_id: str) -> dict[str, Any]:
    db.execute(text("BEGIN IMMEDIATE"))
    turn_ids = db.query(ConversationTurn.turn_id).filter_by(
        conversation_id=conversation_id
    )
    event_count = (
        db.query(ConversationEvent)
        .filter(ConversationEvent.turn_id.in_(turn_ids))
        .count()
    )
    projection_count = (
        db.query(ConversationProjection)
        .filter_by(conversation_id=conversation_id)
        .delete(synchronize_session=False)
    )
    turn_count = (
        db.query(ConversationTurn)
        .filter_by(conversation_id=conversation_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return {
        "conversation_id": conversation_id,
        "deleted_turns": turn_count,
        "deleted_events": event_count,
        "deleted_projection": bool(projection_count),
    }


def abandoned_timeout_seconds() -> int:
    timeout = int(
        os.environ.get(
            "MEMORY_TURN_ABANDON_SECONDS",
            str(DEFAULT_ABANDONED_SECONDS),
        )
    )
    if timeout < 1:
        raise ValueError("MEMORY_TURN_ABANDON_SECONDS must be positive")
    return timeout


def sweep_abandoned_turns(
    session_factory,
    *,
    now: float | None = None,
    timeout_seconds: int | None = None,
) -> int:
    """Release turns left active by a stopped memory-service process."""

    current_time = time.time() if now is None else now
    timeout = (
        abandoned_timeout_seconds() if timeout_seconds is None else timeout_seconds
    )
    with session_factory() as db:
        try:
            db.execute(text("BEGIN IMMEDIATE"))
            abandoned_count = _mark_stale_started_turns(
                db,
                current_time=current_time,
                timeout_seconds=timeout,
                termination_reason="startup_abandoned_turn_sweep",
            )
            db.commit()
            return abandoned_count
        except Exception:
            db.rollback()
            raise


def create_conversation_router(get_db) -> APIRouter:
    router = APIRouter()

    @router.post("/conversations/{conversation_id}/turn/start")
    def start_conversation_turn(
        conversation_id: str,
        request: TurnStartRequest,
        response_contract: Literal["1", "2"] = "1",
        db=Depends(get_db),
    ):
        _validate_conversation_id(conversation_id)
        try:
            return _start_turn(
                db,
                conversation_id,
                request,
                response_contract=int(response_contract),
            )
        except Exception:
            db.rollback()
            raise

    @router.post("/conversations/{conversation_id}/turns/{turn_id}/finalize")
    def finalize_conversation_turn(
        conversation_id: str,
        turn_id: str,
        request: TurnFinalizeRequest,
        db=Depends(get_db),
    ):
        _validate_conversation_id(conversation_id)
        try:
            return _finalize_turn(db, conversation_id, turn_id, request)
        except Exception:
            db.rollback()
            raise

    @router.post("/conversations/{conversation_id}/products/resolve")
    def resolve_conversation_products(
        conversation_id: str,
        request: ProductResolutionRequest,
        db=Depends(get_db),
    ):
        _validate_conversation_id(conversation_id)
        return resolve_product_references(db, conversation_id, request)

    @router.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, db=Depends(get_db)):
        _validate_conversation_id(conversation_id)
        try:
            return _delete_conversation(db, conversation_id)
        except Exception:
            db.rollback()
            raise

    return router
