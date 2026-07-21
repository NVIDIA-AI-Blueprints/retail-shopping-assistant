# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable conversation-turn API for the memory service."""

from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, text

from .models import (
    CartItem,
    ConversationEvent,
    ConversationProjection,
    ConversationTurn,
)


DEFAULT_ABANDONED_SECONDS = 300
DEFAULT_RECENT_TURNS_LIMIT = 8
MAX_RECENT_TURNS_LIMIT = 50


class TurnStartRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=128)
    shopper_text: str = Field(..., min_length=1, max_length=100_000)
    cart_user_id: int = Field(..., ge=0)
    request_digest: str = Field(..., min_length=1, max_length=128)
    catalog_revision: str | None = Field(default=None, max_length=512)


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

    @model_validator(mode="after")
    def _event_keys_are_unique(self):
        keys = [event.event_key for event in self.events]
        if len(keys) != len(set(keys)):
            raise ValueError("event_key values must be unique within a turn")
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


def _projection_dict(projection: ConversationProjection) -> dict[str, Any]:
    return {
        "version": projection.version,
        "active_anchors": json.loads(projection.active_anchors_json),
        "effective_preferences": json.loads(projection.effective_preferences_json),
        "product_reference_index": json.loads(projection.product_reference_index_json),
        "last_turn_id": projection.last_turn_id,
    }


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


def _recent_turns(db, conversation_id: str) -> list[dict[str, Any]]:
    rows = (
        db.query(ConversationTurn)
        .filter(
            ConversationTurn.conversation_id == conversation_id,
            ConversationTurn.status != "started",
        )
        .order_by(ConversationTurn.sequence.desc())
        .limit(_recent_turns_limit())
        .all()
    )
    return [
        {
            "sequence": row.sequence,
            "shopper_text": row.shopper_text,
            "assistant_text": row.assistant_text,
            "status": row.status,
        }
        for row in reversed(rows)
    ]


def _start_response(
    db,
    turn: ConversationTurn,
    projection: ConversationProjection,
    *,
    replayed: bool,
) -> dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "attempt_id": turn.attempt_id,
        "sequence": turn.sequence,
        "replayed": replayed,
        "status": turn.status,
        "recent_turns": _recent_turns(db, turn.conversation_id),
        "projection": _projection_dict(projection),
        "cart": _cart_for_user(db, turn.cart_user_id),
        "assistant_text": turn.assistant_text,
        "termination_reason": turn.termination_reason,
        "output": json.loads(turn.output_json) if turn.output_json else None,
    }


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


def _start_turn(db, conversation_id: str, request: TurnStartRequest) -> dict[str, Any]:
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
        ):
            raise HTTPException(
                status_code=409,
                detail="request_id was already used for different turn input",
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
            response = _start_response(db, existing, projection, replayed=False)
            db.commit()
            return response
        projection = _get_or_create_projection(db, conversation_id)
        response = _start_response(db, existing, projection, replayed=True)
        db.commit()
        return response

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
        shopper_text=request.shopper_text,
        status="started",
        catalog_revision=request.catalog_revision,
        started_at=current_time,
    )
    db.add(turn)
    db.flush()
    projection = _get_or_create_projection(db, conversation_id)
    response = _start_response(db, turn, projection, replayed=False)
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

    now = time.time()
    _append_events(db, turn, request.events, now)
    projection = _get_or_create_projection(db, conversation_id)
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
        db=Depends(get_db),
    ):
        _validate_conversation_id(conversation_id)
        try:
            return _start_turn(db, conversation_id, request)
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

    @router.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, db=Depends(get_db)):
        _validate_conversation_id(conversation_id)
        try:
            return _delete_conversation(db, conversation_id)
        except Exception:
            db.rollback()
            raise

    return router
