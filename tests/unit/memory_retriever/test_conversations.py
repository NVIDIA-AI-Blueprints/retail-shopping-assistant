# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for durable conversation storage."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from memory_retriever.src import main as memory_main
from memory_retriever.src import conversations as conversation_store
from memory_retriever.src import product_references


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def conversation_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    test_engine = memory_main.build_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
    monkeypatch.setattr(memory_main, "engine", test_engine)
    monkeypatch.setattr(memory_main, "SessionLocal", session_factory)
    monkeypatch.setenv(
        "SHARED_CONFIG_ROOT",
        str(REPO_ROOT / "shared" / "configs"),
    )

    with TestClient(memory_main.app) as client:
        yield client

    memory_main.Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def _start_turn(
    client: TestClient,
    conversation_id: str,
    *,
    request_id: str,
    shopper_text: str = "Show me a bag",
    cart_user_id: int = 7,
    request_digest: str | None = None,
    shopper_profile_id: str | None = None,
    include_null_profile: bool = False,
    response_contract: int | None = 2,
):
    payload = {
        "request_id": request_id,
        "shopper_text": shopper_text,
        "cart_user_id": cart_user_id,
        "request_digest": request_digest or f"digest:{request_id}",
        "catalog_revision": "catalog-v1",
    }
    if shopper_profile_id is not None or include_null_profile:
        payload["shopper_profile_id"] = shopper_profile_id
    path = f"/conversations/{conversation_id}/turn/start"
    if response_contract is not None:
        path += f"?response_contract={response_contract}"
    return client.post(path, json=payload)


def test_unversioned_start_returns_exact_v1_shape(
    conversation_db: TestClient,
) -> None:
    response = _start_turn(
        conversation_db,
        "conversation-v1-shape",
        request_id="request-v1",
        response_contract=None,
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "turn_id",
        "attempt_id",
        "sequence",
        "replayed",
        "status",
        "recent_turns",
        "previous_selected_skill_names",
        "shopper_context",
        "projection",
        "cart",
        "assistant_text",
        "termination_reason",
        "output",
    }
    assert set(payload["projection"]) == {
        "version",
        "active_anchors",
        "effective_preferences",
        "product_reference_index",
        "last_turn_id",
    }


def test_turn_start_negotiates_v2_and_future_requests_to_server_max(
    conversation_db: TestClient,
) -> None:
    v2 = _start_turn(
        conversation_db,
        "conversation-contract-v2",
        request_id="request-v2",
        response_contract=2,
    )
    future = _start_turn(
        conversation_db,
        "conversation-contract-v3",
        request_id="request-v3",
        response_contract=999,
    )

    assert v2.status_code == 200
    assert v2.json()["contract_version"] == 2
    assert "current_weather_scope" not in v2.json()["projection"]
    assert future.status_code == 200
    assert future.json()["contract_version"] == 3
    assert future.json()["projection"]["current_weather_scope"] == {
        "revision": 0
    }


def test_v1_start_reads_raw_tail_before_v2_summary_watermark(
    conversation_db: TestClient,
) -> None:
    first = _start_turn(
        conversation_db,
        "conversation-v1-rollback",
        request_id="request-1",
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-v1-rollback",
            first["turn_id"],
            request_id="request-1",
            attempt_id=first["attempt_id"],
            assistant_text="First assistant turn.",
        ).status_code
        == 200
    )
    second = _start_turn(
        conversation_db,
        "conversation-v1-rollback",
        request_id="request-2",
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-v1-rollback",
            second["turn_id"],
            request_id="request-2",
            attempt_id=second["attempt_id"],
            assistant_text="Second assistant turn.",
            summary_advance={
                "expected_projection_version": second["projection"]["version"],
                "summary_text": "The first turn is summarized.",
                "summary_through_sequence": 1,
            },
        ).status_code
        == 200
    )

    rollback_start = _start_turn(
        conversation_db,
        "conversation-v1-rollback",
        request_id="request-3",
        response_contract=None,
    )

    assert rollback_start.status_code == 200
    payload = rollback_start.json()
    assert [turn["sequence"] for turn in payload["recent_turns"]] == [1, 2]
    assert "summary_text" not in payload["projection"]
    assert "summary_compaction_source" not in payload


def _finalize_turn(
    client: TestClient,
    conversation_id: str,
    turn_id: str,
    *,
    request_id: str,
    attempt_id: str,
    assistant_text: str = "Here is a bag.",
    status: str = "completed",
    events: list[dict] | None = None,
    output: dict | None = None,
    summary_advance: dict | None = None,
    weather_receipt_promotion: dict | None = None,
    current_weather_scope_transition: dict | None = None,
):
    payload = {
        "request_id": request_id,
        "attempt_id": attempt_id,
        "assistant_text": assistant_text,
        "status": status,
        "termination_reason": status,
        "events": events or [],
        "output": output,
    }
    if summary_advance is not None:
        payload["summary_advance"] = summary_advance
    if weather_receipt_promotion is not None:
        payload["weather_receipt_promotion"] = weather_receipt_promotion
    if current_weather_scope_transition is not None:
        payload["current_weather_scope_transition"] = (
            current_weather_scope_transition
        )
    return client.post(
        f"/conversations/{conversation_id}/turns/{turn_id}/finalize",
        json=payload,
    )


def _weather_receipt_promotion(
    *,
    expected_projection_version: int,
    fetched_at: datetime,
    location: str = "NYC",
    location_query: str | None = "NYC, NY",
    resolved_location: str | None = "New York, NY, United States",
    forecast_date: date = date(2026, 8, 3),
    ttl_seconds: int = 3_600,
    saved_area: bool = False,
    source_tool_call_id: str = "weather-call-1",
) -> dict:
    location_scope = (
        {"kind": "confirmed_saved_zip"}
        if saved_area
        else {
            "kind": "shopper_provided_location",
            "location": location,
            **(
                {"location_query": location_query}
                if location_query is not None
                else {}
            ),
        }
    )
    return {
        "expected_projection_version": expected_projection_version,
        "source_tool_call_id": source_tool_call_id,
        "location_scope": location_scope,
        "evidence": {
            "ok": True,
            "provider": "visual_crossing",
            "fetched_at": fetched_at.isoformat(),
            "requested_window": {
                "start_date": forecast_date.isoformat(),
                "end_date": forecast_date.isoformat(),
            },
            **(
                {"resolved_location": resolved_location}
                if resolved_location is not None
                else {}
            ),
            "days": [
                {
                    "date": forecast_date.isoformat(),
                    "condition": "rain",
                    "precipitation_probability_pct": 70.0,
                    "precipitation_types": ["rain"],
                    "temperature_low_f": 65.0,
                    "temperature_high_f": 78.0,
                }
            ],
            "attribution": {
                "label": "Weather Data Provided by Visual Crossing",
                "url": "https://www.visualcrossing.com/",
            },
        },
        "ttl_seconds": ttl_seconds,
    }


def _weather_scope_transition(
    *,
    expected_projection_version: int,
    action: str,
    location: str | None = None,
    location_query: str | None = None,
    forecast_date: date | None = None,
    saved_area: bool = False,
) -> dict:
    transition: dict = {
        "expected_projection_version": expected_projection_version,
        "action": action,
    }
    if saved_area:
        transition["location_scope"] = {"kind": "confirmed_saved_zip"}
    elif location is not None:
        transition["location_scope"] = {
            "kind": "shopper_provided_location",
            "location": location,
        }
        if location_query is not None:
            transition["location_scope"]["location_query"] = location_query
    if forecast_date is not None:
        transition["requested_window"] = {
            "start_date": forecast_date.isoformat(),
            "end_date": forecast_date.isoformat(),
        }
    return transition


def _present_products(
    client: TestClient,
    conversation_id: str,
    *,
    request_id: str,
    products: list[dict],
) -> tuple[dict, str]:
    started = _start_turn(
        client,
        conversation_id,
        request_id=request_id,
        shopper_text=f"Show products for {request_id}",
    ).json()
    finalized = _finalize_turn(
        client,
        conversation_id,
        started["turn_id"],
        request_id=request_id,
        attempt_id=started["attempt_id"],
        output={
            "product_results": products,
            "retrieved": {},
            "agent_diagnostics": {},
        },
    )
    assert finalized.status_code == 200
    with memory_main.SessionLocal() as db:
        event = (
            db.query(memory_main.ConversationEvent)
            .filter_by(
                turn_id=started["turn_id"],
                event_type="candidate_set_presented",
            )
            .one()
        )
        candidate_set_id = event.event_id
    return started, candidate_set_id


def test_start_returns_recent_turns_projection_and_authoritative_cart(
    conversation_db: TestClient,
) -> None:
    added = conversation_db.post(
        "/user/7/cart/add",
        json={
            "product_id": "bag-1",
            "item": "Structured Bag",
            "amount": 1,
            "price": 45.0,
            "idempotency_key": "add-bag-1",
        },
    )
    first = _start_turn(
        conversation_db,
        "conversation-a",
        request_id="request-1",
    )

    assert first.status_code == 200
    assert first.json() == {
        "turn_id": first.json()["turn_id"],
        "contract_version": 2,
        "attempt_id": first.json()["attempt_id"],
        "sequence": 1,
        "replayed": False,
        "status": "started",
        "recent_turns": [],
        "unsummarized_turn_count": 0,
        "summary_compaction_source": None,
        "previous_selected_skill_names": [],
        "projection": {
            "version": 0,
            "summary_text": "",
            "summary_through_sequence": 0,
            "active_receipts": [],
            "active_anchors": [],
            "effective_preferences": [],
            "product_reference_index": [],
            "last_turn_id": None,
        },
        "cart": [added.json()["cart_line"]],
        "assistant_text": None,
        "termination_reason": None,
        "output": None,
        "shopper_context": None,
    }
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-a",
            first.json()["turn_id"],
            request_id="request-1",
            attempt_id=first.json()["attempt_id"],
            output={
                "product_results": [],
                "retrieved": {},
                "agent_diagnostics": {},
                "selected_skill_names": ["outfit-styling"],
            },
        ).status_code
        == 200
    )

    second = _start_turn(
        conversation_db,
        "conversation-a",
        request_id="request-2",
        shopper_text="Show me shoes",
    )

    assert second.status_code == 200
    assert second.json()["sequence"] == 2
    assert second.json()["recent_turns"] == [
        {
            "sequence": 1,
            "shopper_text": "Show me a bag",
            "assistant_text": "Here is a bag.",
            "status": "completed",
        }
    ]
    assert second.json()["previous_selected_skill_names"] == ["outfit-styling"]


def test_summary_advance_is_atomic_replayable_and_excludes_covered_turns(
    conversation_db: TestClient,
) -> None:
    first = _start_turn(
        conversation_db,
        "conversation-summary",
        request_id="request-1",
        shopper_text="I need a wedding outfit.",
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-summary",
            first["turn_id"],
            request_id="request-1",
            attempt_id=first["attempt_id"],
            assistant_text="Here are two dress directions.",
        ).status_code
        == 200
    )
    second = _start_turn(
        conversation_db,
        "conversation-summary",
        request_id="request-2",
        shopper_text="The wedding is in New York.",
    ).json()
    summary_advance = {
        "expected_projection_version": second["projection"]["version"],
        "summary_text": "The shopper is assembling a wedding outfit.",
        "summary_through_sequence": 1,
    }
    output = {
        "product_results": [
            {
                "product_id": "dress-1",
                "display_name": "Satin Dress",
                "category": "dresses",
            }
        ],
        "retrieved": {},
        "agent_diagnostics": {},
    }

    finalized = _finalize_turn(
        conversation_db,
        "conversation-summary",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        assistant_text="I will plan around New York.",
        output=output,
        summary_advance=summary_advance,
    )
    replay = _finalize_turn(
        conversation_db,
        "conversation-summary",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        assistant_text="I will plan around New York.",
        output=output,
        summary_advance=summary_advance,
    )
    changed_replay = _finalize_turn(
        conversation_db,
        "conversation-summary",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        assistant_text="I will plan around New York.",
        output=output,
        summary_advance={
            **summary_advance,
            "summary_text": "A different summary.",
        },
    )
    third = _start_turn(
        conversation_db,
        "conversation-summary",
        request_id="request-3",
        shopper_text="Compare those dresses.",
    )

    assert finalized.status_code == 200
    assert replay.json() == {**finalized.json(), "replayed": True}
    assert changed_replay.status_code == 409
    assert third.status_code == 200
    assert third.json()["projection"]["summary_text"] == (
        "The shopper is assembling a wedding outfit."
    )
    assert third.json()["projection"]["summary_through_sequence"] == 1
    assert [turn["sequence"] for turn in third.json()["recent_turns"]] == [2]
    assert (
        third.json()["projection"]["product_reference_index"][0]["products"][0]["ref"]
        == "dress-1"
    )
    with memory_main.SessionLocal() as db:
        projection = db.query(memory_main.ConversationProjection).one()
        assert projection.version == 2
        assert projection.last_turn_id == second["turn_id"]


def test_invalid_summary_advance_rolls_back_the_complete_finalize(
    conversation_db: TestClient,
) -> None:
    first = _start_turn(
        conversation_db,
        "conversation-summary-conflict",
        request_id="request-1",
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-summary-conflict",
            first["turn_id"],
            request_id="request-1",
            attempt_id=first["attempt_id"],
        ).status_code
        == 200
    )
    second = _start_turn(
        conversation_db,
        "conversation-summary-conflict",
        request_id="request-2",
    ).json()
    event = {
        "event_key": "detail-1",
        "event_type": "product_detail_confirmed",
        "source_kind": "catalog",
        "source_ref": "dress-1",
        "payload": {"material": "satin"},
    }

    wrong_version = _finalize_turn(
        conversation_db,
        "conversation-summary-conflict",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        events=[event],
        summary_advance={
            "expected_projection_version": 999,
            "summary_text": "Summary.",
            "summary_through_sequence": 1,
        },
    )
    future_boundary = _finalize_turn(
        conversation_db,
        "conversation-summary-conflict",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        events=[event],
        summary_advance={
            "expected_projection_version": second["projection"]["version"],
            "summary_text": "Summary.",
            "summary_through_sequence": second["sequence"],
        },
    )

    assert wrong_version.status_code == 409
    assert wrong_version.json()["detail"] == "projection_version_conflict"
    assert future_boundary.status_code == 409
    assert future_boundary.json()["detail"] == "summary_boundary_conflict"
    with memory_main.SessionLocal() as db:
        turn = (
            db.query(memory_main.ConversationTurn)
            .filter_by(turn_id=second["turn_id"])
            .one()
        )
        projection = db.query(memory_main.ConversationProjection).one()
        assert turn.status == "started"
        assert turn.finalize_digest is None
        assert turn.output_json is None
        assert db.query(memory_main.ConversationEvent).count() == 0
        assert projection.version == 1
        assert projection.summary_text == ""
        assert projection.summary_through_sequence == 0

    completed_without_advance = _finalize_turn(
        conversation_db,
        "conversation-summary-conflict",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        events=[event],
    )
    assert completed_without_advance.status_code == 200


def test_current_weather_scope_continue_replace_and_clear_are_atomic(
    conversation_db: TestClient,
) -> None:
    first = _start_turn(
        conversation_db,
        "conversation-weather-scope",
        request_id="request-1",
        response_contract=3,
    ).json()
    first_transition = _weather_scope_transition(
        expected_projection_version=first["projection"]["version"],
        action="replace",
        location="NYC",
        location_query="NYC, NY",
        forecast_date=date(2026, 8, 3),
    )
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-scope",
            first["turn_id"],
            request_id="request-1",
            attempt_id=first["attempt_id"],
            current_weather_scope_transition=first_transition,
        ).status_code
        == 200
    )

    second = _start_turn(
        conversation_db,
        "conversation-weather-scope",
        request_id="request-2",
        response_contract=3,
    ).json()
    scope = second["projection"]["current_weather_scope"]
    assert scope["revision"] == 1
    assert scope["location"]["value"]["location"] == "NYC"
    assert scope["location"]["source_turn_id"] == first["turn_id"]
    assert scope["window"]["source_turn_id"] == first["turn_id"]
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-scope",
            second["turn_id"],
            request_id="request-2",
            attempt_id=second["attempt_id"],
            current_weather_scope_transition=_weather_scope_transition(
                expected_projection_version=second["projection"]["version"],
                action="continue",
                forecast_date=date(2026, 8, 4),
            ),
        ).status_code
        == 200
    )

    third = _start_turn(
        conversation_db,
        "conversation-weather-scope",
        request_id="request-3",
        response_contract=3,
    ).json()
    scope = third["projection"]["current_weather_scope"]
    assert scope["revision"] == 2
    assert scope["location"]["source_turn_id"] == first["turn_id"]
    assert scope["window"]["source_turn_id"] == second["turn_id"]
    assert scope["window"]["value"]["start_date"] == "2026-08-04"
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-scope",
            third["turn_id"],
            request_id="request-3",
            attempt_id=third["attempt_id"],
            current_weather_scope_transition=_weather_scope_transition(
                expected_projection_version=third["projection"]["version"],
                action="replace",
                location="Seattle",
                location_query="Seattle, WA",
            ),
        ).status_code
        == 200
    )

    fourth = _start_turn(
        conversation_db,
        "conversation-weather-scope",
        request_id="request-4",
        response_contract=3,
    ).json()
    scope = fourth["projection"]["current_weather_scope"]
    assert scope["revision"] == 3
    assert scope["location"]["value"]["location"] == "Seattle"
    assert "window" not in scope
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-scope",
            fourth["turn_id"],
            request_id="request-4",
            attempt_id=fourth["attempt_id"],
            current_weather_scope_transition=_weather_scope_transition(
                expected_projection_version=fourth["projection"]["version"],
                action="replace",
            ),
        ).status_code
        == 200
    )

    cleared = _start_turn(
        conversation_db,
        "conversation-weather-scope",
        request_id="request-5",
        response_contract=3,
    ).json()["projection"]["current_weather_scope"]
    assert cleared == {"revision": 4}


def test_weather_scope_rejects_stale_failed_and_unbound_saved_area_updates(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-weather-scope-conflict",
        request_id="request-1",
        response_contract=3,
    ).json()
    transition = _weather_scope_transition(
        expected_projection_version=999,
        action="replace",
        location="Denver",
    )

    stale = _finalize_turn(
        conversation_db,
        "conversation-weather-scope-conflict",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        current_weather_scope_transition=transition,
    )
    failed = _finalize_turn(
        conversation_db,
        "conversation-weather-scope-conflict",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        status="failed",
        current_weather_scope_transition={
            **transition,
            "expected_projection_version": 0,
        },
    )
    saved_area = _finalize_turn(
        conversation_db,
        "conversation-weather-scope-conflict",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        current_weather_scope_transition=_weather_scope_transition(
            expected_projection_version=0,
            action="replace",
            saved_area=True,
        ),
    )

    assert stale.status_code == 409
    assert stale.json()["detail"] == "projection_version_conflict"
    assert failed.status_code == 409
    assert failed.json()["detail"] == "current_weather_scope_status_conflict"
    assert saved_area.status_code == 409
    assert (
        saved_area.json()["detail"]
        == "current_weather_scope_saved_area_unavailable"
    )
    with memory_main.SessionLocal() as db:
        turn = db.query(memory_main.ConversationTurn).one()
        projection = db.query(memory_main.ConversationProjection).one()
        assert turn.status == "started"
        assert projection.current_weather_scope_json == '{"revision":0}'


def test_scope_matched_receipt_is_atomic_and_replacement_drops_it(
    conversation_db: TestClient,
) -> None:
    first = _start_turn(
        conversation_db,
        "conversation-scoped-weather-receipt",
        request_id="request-1",
        response_contract=3,
    ).json()
    fetched_at = datetime.now(timezone.utc)
    scope_transition = _weather_scope_transition(
        expected_projection_version=0,
        action="replace",
        location="NYC",
        location_query="NYC, NY",
        forecast_date=date(2026, 8, 3),
    )
    finalized = _finalize_turn(
        conversation_db,
        "conversation-scoped-weather-receipt",
        first["turn_id"],
        request_id="request-1",
        attempt_id=first["attempt_id"],
        current_weather_scope_transition=scope_transition,
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=0,
            fetched_at=fetched_at,
        ),
    )

    assert finalized.status_code == 200
    second = _start_turn(
        conversation_db,
        "conversation-scoped-weather-receipt",
        request_id="request-2",
        response_contract=3,
    ).json()
    assert len(second["projection"]["active_receipts"]) == 1
    replaced = _finalize_turn(
        conversation_db,
        "conversation-scoped-weather-receipt",
        second["turn_id"],
        request_id="request-2",
        attempt_id=second["attempt_id"],
        current_weather_scope_transition=_weather_scope_transition(
            expected_projection_version=second["projection"]["version"],
            action="replace",
            location="Seattle",
            location_query="Seattle, WA",
        ),
    )
    assert replaced.status_code == 200

    third = _start_turn(
        conversation_db,
        "conversation-scoped-weather-receipt",
        request_id="request-3",
        response_contract=3,
    ).json()
    assert third["projection"]["active_receipts"] == []
    with memory_main.SessionLocal() as db:
        projection = db.query(memory_main.ConversationProjection).one()
        assert projection.active_receipts_json == "[]"


def test_scope_mismatched_receipt_rolls_back_finalize(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-scope-receipt-mismatch",
        request_id="request-1",
        response_contract=3,
    ).json()

    finalized = _finalize_turn(
        conversation_db,
        "conversation-scope-receipt-mismatch",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        current_weather_scope_transition=_weather_scope_transition(
            expected_projection_version=0,
            action="replace",
            location="Seattle",
            location_query="Seattle, WA",
            forecast_date=date(2026, 8, 3),
        ),
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=0,
            fetched_at=datetime.now(timezone.utc),
        ),
    )

    assert finalized.status_code == 409
    assert finalized.json()["detail"] == "weather_receipt_scope_conflict"
    with memory_main.SessionLocal() as db:
        turn = db.query(memory_main.ConversationTurn).one()
        projection = db.query(memory_main.ConversationProjection).one()
        assert turn.status == "started"
        assert projection.current_weather_scope_json == '{"revision":0}'
        assert projection.active_receipts_json == "[]"


def test_weather_receipt_promotion_is_atomic_replayable_and_hydrated(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-weather-receipt",
        request_id="request-1",
        shopper_text="The NYC patio wedding is next week.",
    ).json()
    fetched_at = datetime.now(timezone.utc)
    promotion = _weather_receipt_promotion(
        expected_projection_version=started["projection"]["version"],
        fetched_at=fetched_at,
    )
    event = {
        "event_key": "weather-advice-prepared",
        "event_type": "preference_added",
        "source_kind": "runtime",
        "payload": {"kind": "event_styling"},
    }

    finalized = _finalize_turn(
        conversation_db,
        "conversation-weather-receipt",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        assistant_text="I used the current NYC forecast.",
        events=[event],
        weather_receipt_promotion=promotion,
    )
    replay = _finalize_turn(
        conversation_db,
        "conversation-weather-receipt",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        assistant_text="I used the current NYC forecast.",
        events=[event],
        weather_receipt_promotion=promotion,
    )
    changed_replay = _finalize_turn(
        conversation_db,
        "conversation-weather-receipt",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        assistant_text="I used the current NYC forecast.",
        events=[event],
        weather_receipt_promotion={
            **promotion,
            "ttl_seconds": 1_800,
        },
    )
    next_turn = _start_turn(
        conversation_db,
        "conversation-weather-receipt",
        request_id="request-2",
        shopper_text="Compare the dresses for that event.",
    )

    assert finalized.status_code == 200
    assert replay.json() == {**finalized.json(), "replayed": True}
    assert changed_replay.status_code == 409
    assert next_turn.status_code == 200
    receipts = next_turn.json()["projection"]["active_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["receipt_type"] == "weather_forecast.v1"
    assert receipt["source_turn_id"] == started["turn_id"]
    assert receipt["source_sequence"] == started["sequence"]
    assert receipt["source_tool"] == "get_weather_forecast_tool"
    assert receipt["source_tool_call_id"] == "weather-call-1"
    assert receipt["location_scope"] == {
        "kind": "shopper_provided_location",
        "location": "NYC",
        "location_query": "NYC, NY",
    }
    assert receipt["evidence"]["requested_window"] == {
        "start_date": "2026-08-03",
        "end_date": "2026-08-03",
    }
    with memory_main.SessionLocal() as db:
        projection = db.query(memory_main.ConversationProjection).one()
        assert len(json.loads(projection.active_receipts_json)) == 1
        assert db.query(memory_main.ConversationEvent).count() == 1


def test_weather_receipt_conflicts_roll_back_the_complete_finalize(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-weather-conflict",
        request_id="request-1",
    ).json()
    event = {
        "event_key": "detail-1",
        "event_type": "product_detail_confirmed",
        "source_kind": "catalog",
        "source_ref": "dress-1",
        "payload": {"material": "satin"},
    }
    wrong_version = _finalize_turn(
        conversation_db,
        "conversation-weather-conflict",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        events=[event],
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=999,
            fetched_at=datetime.now(timezone.utc),
        ),
    )

    assert wrong_version.status_code == 409
    assert wrong_version.json()["detail"] == "projection_version_conflict"
    with memory_main.SessionLocal() as db:
        turn = db.query(memory_main.ConversationTurn).one()
        projection = db.query(memory_main.ConversationProjection).one()
        assert turn.status == "started"
        assert turn.finalize_digest is None
        assert turn.output_json is None
        assert db.query(memory_main.ConversationEvent).count() == 0
        assert projection.version == 0
        assert projection.active_receipts_json == "[]"

    failed_promotion = _finalize_turn(
        conversation_db,
        "conversation-weather-conflict",
        started["turn_id"],
        request_id="request-1",
        attempt_id=started["attempt_id"],
        status="failed",
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=0,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    assert failed_promotion.status_code == 409
    assert failed_promotion.json()["detail"] == (
        "weather_receipt_status_conflict"
    )

    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-conflict",
            started["turn_id"],
            request_id="request-1",
            attempt_id=started["attempt_id"],
            events=[event],
        ).status_code
        == 200
    )


def test_saved_area_receipt_requires_bound_profile_and_never_stores_zip(
    conversation_db: TestClient,
) -> None:
    guest = _start_turn(
        conversation_db,
        "guest-weather-receipt",
        request_id="guest-request",
    ).json()
    guest_rejected = _finalize_turn(
        conversation_db,
        "guest-weather-receipt",
        guest["turn_id"],
        request_id="guest-request",
        attempt_id=guest["attempt_id"],
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=0,
            fetched_at=datetime.now(timezone.utc),
            saved_area=True,
            resolved_location=None,
        ),
    )
    assert guest_rejected.status_code == 409
    assert guest_rejected.json()["detail"] == "weather_receipt_status_conflict"

    selected = _start_turn(
        conversation_db,
        "selected-weather-receipt",
        request_id="selected-request",
        shopper_profile_id="shopper_jordan",
    ).json()
    finalized = _finalize_turn(
        conversation_db,
        "selected-weather-receipt",
        selected["turn_id"],
        request_id="selected-request",
        attempt_id=selected["attempt_id"],
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=0,
            fetched_at=datetime.now(timezone.utc),
            saved_area=True,
            resolved_location=None,
        ),
    )
    next_turn = _start_turn(
        conversation_db,
        "selected-weather-receipt",
        request_id="selected-request-2",
        shopper_profile_id="shopper_jordan",
    )

    assert finalized.status_code == 200
    receipt = next_turn.json()["projection"]["active_receipts"][0]
    assert receipt["location_scope"] == {"kind": "confirmed_saved_zip"}
    assert "resolved_location" not in receipt["evidence"]
    with memory_main.SessionLocal() as db:
        stored = db.query(memory_main.ConversationProjection).filter_by(
            conversation_id="selected-weather-receipt"
        ).one()
        assert "10001" not in stored.active_receipts_json


def test_weather_receipts_supersede_exact_scope_and_cap_distinct_scopes(
    conversation_db: TestClient,
) -> None:
    conversation_id = "conversation-weather-cap"
    base_fetched_at = datetime.now(timezone.utc)

    for index in range(5):
        started = _start_turn(
            conversation_db,
            conversation_id,
            request_id=f"request-{index + 1}",
        ).json()
        finalized = _finalize_turn(
            conversation_db,
            conversation_id,
            started["turn_id"],
            request_id=f"request-{index + 1}",
            attempt_id=started["attempt_id"],
            weather_receipt_promotion=_weather_receipt_promotion(
                expected_projection_version=started["projection"]["version"],
                fetched_at=base_fetched_at + timedelta(seconds=index),
                location=f"City {index + 1}",
                location_query=None,
                resolved_location=f"Resolved City {index + 1}",
                source_tool_call_id=f"weather-call-{index + 1}",
            ),
        )
        assert finalized.status_code == 200

    sixth = _start_turn(
        conversation_db,
        conversation_id,
        request_id="request-6",
    ).json()
    capped = sixth["projection"]["active_receipts"]
    assert len(capped) == 4
    assert [receipt["source_sequence"] for receipt in capped] == [5, 4, 3, 2]
    assert all(
        receipt["location_scope"]["location"] != "City 1"
        for receipt in capped
    )

    replaced = _finalize_turn(
        conversation_db,
        conversation_id,
        sixth["turn_id"],
        request_id="request-6",
        attempt_id=sixth["attempt_id"],
        weather_receipt_promotion=_weather_receipt_promotion(
            expected_projection_version=sixth["projection"]["version"],
            fetched_at=base_fetched_at + timedelta(seconds=10),
            location="City 2",
            location_query=None,
            resolved_location="New Provider Resolution for City 2",
            source_tool_call_id="weather-call-6",
        ),
    )
    seventh = _start_turn(
        conversation_db,
        conversation_id,
        request_id="request-7",
    )

    assert replaced.status_code == 200
    receipts = seventh.json()["projection"]["active_receipts"]
    assert len(receipts) == 4
    city_two = [
        receipt
        for receipt in receipts
        if receipt["location_scope"]["location"] == "City 2"
    ]
    assert len(city_two) == 1
    assert city_two[0]["source_sequence"] == 6
    assert city_two[0]["evidence"]["resolved_location"] == (
        "New Provider Resolution for City 2"
    )


def test_expired_weather_receipts_are_filtered_then_pruned_on_finalize(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_time = 1_800_000_000.0
    clock = {"now": base_time}
    monkeypatch.setattr(
        conversation_store,
        "time",
        SimpleNamespace(time=lambda: clock["now"]),
    )
    fetched_at = datetime.fromtimestamp(base_time, tz=timezone.utc)
    first = _start_turn(
        conversation_db,
        "conversation-weather-expiry",
        request_id="request-1",
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-expiry",
            first["turn_id"],
            request_id="request-1",
            attempt_id=first["attempt_id"],
            weather_receipt_promotion=_weather_receipt_promotion(
                expected_projection_version=0,
                fetched_at=fetched_at,
                ttl_seconds=1,
            ),
        ).status_code
        == 200
    )
    with memory_main.SessionLocal() as db:
        projection = db.query(memory_main.ConversationProjection).one()
        assert len(json.loads(projection.active_receipts_json)) == 1

    clock["now"] = base_time + 2
    second = _start_turn(
        conversation_db,
        "conversation-weather-expiry",
        request_id="request-2",
    ).json()
    assert second["projection"]["active_receipts"] == []
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-weather-expiry",
            second["turn_id"],
            request_id="request-2",
            attempt_id=second["attempt_id"],
        ).status_code
        == 200
    )
    with memory_main.SessionLocal() as db:
        projection = db.query(memory_main.ConversationProjection).one()
        assert projection.active_receipts_json == "[]"


def test_summary_raw_tail_contains_only_later_context_eligible_turns(
    conversation_db: TestClient,
) -> None:
    first = _start_turn(
        conversation_db,
        "conversation-summary-tail",
        request_id="request-1",
    ).json()
    _finalize_turn(
        conversation_db,
        "conversation-summary-tail",
        first["turn_id"],
        request_id="request-1",
        attempt_id=first["attempt_id"],
    )
    blocked = _start_turn(
        conversation_db,
        "conversation-summary-tail",
        request_id="request-2",
    ).json()
    _finalize_turn(
        conversation_db,
        "conversation-summary-tail",
        blocked["turn_id"],
        request_id="request-2",
        attempt_id=blocked["attempt_id"],
        assistant_text="Blocked response.",
        status="blocked",
    )
    failed = _start_turn(
        conversation_db,
        "conversation-summary-tail",
        request_id="request-3",
    ).json()
    _finalize_turn(
        conversation_db,
        "conversation-summary-tail",
        failed["turn_id"],
        request_id="request-3",
        attempt_id=failed["attempt_id"],
        assistant_text="Safe failed response.",
        status="failed",
    )
    fourth = _start_turn(
        conversation_db,
        "conversation-summary-tail",
        request_id="request-4",
    ).json()
    skipped_eligible_boundary = _finalize_turn(
        conversation_db,
        "conversation-summary-tail",
        fourth["turn_id"],
        request_id="request-4",
        attempt_id=fourth["attempt_id"],
        summary_advance={
            "expected_projection_version": fourth["projection"]["version"],
            "summary_text": "Invalid summary boundary.",
            "summary_through_sequence": 2,
        },
    )
    assert skipped_eligible_boundary.status_code == 409
    assert (
        skipped_eligible_boundary.json()["detail"]
        == "summary_boundary_conflict"
    )
    finalized = _finalize_turn(
        conversation_db,
        "conversation-summary-tail",
        fourth["turn_id"],
        request_id="request-4",
        attempt_id=fourth["attempt_id"],
        summary_advance={
            "expected_projection_version": fourth["projection"]["version"],
            "summary_text": "The first and failed turns are summarized.",
            "summary_through_sequence": 3,
        },
    )
    next_turn = _start_turn(
        conversation_db,
        "conversation-summary-tail",
        request_id="request-5",
    )

    assert finalized.status_code == 200
    assert [
        (turn["sequence"], turn["status"]) for turn in next_turn.json()["recent_turns"]
    ] == [(4, "completed")]


def test_summary_compaction_accepts_any_offered_oldest_prefix_boundary(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_RECENT_TURNS", "4")
    for sequence in range(1, 7):
        started = _start_turn(
            conversation_db,
            "conversation-summary-source",
            request_id=f"request-{sequence}",
        ).json()
        finalized = _finalize_turn(
            conversation_db,
            "conversation-summary-source",
            started["turn_id"],
            request_id=f"request-{sequence}",
            attempt_id=started["attempt_id"],
            assistant_text=f"Assistant response {sequence}.",
        )
        assert finalized.status_code == 200

    seventh = _start_turn(
        conversation_db,
        "conversation-summary-source",
        request_id="request-7",
    ).json()
    source = seventh["summary_compaction_source"]

    assert seventh["unsummarized_turn_count"] == 6
    assert [turn["sequence"] for turn in seventh["recent_turns"]] == [3, 4, 5, 6]
    assert source["after_sequence"] == 0
    assert source["through_sequence"] == 4
    assert source["expected_projection_version"] == seventh["projection"]["version"]
    assert [turn["sequence"] for turn in source["turns"]] == [1, 2, 3, 4]

    partial_advance = _finalize_turn(
        conversation_db,
        "conversation-summary-source",
        seventh["turn_id"],
        request_id="request-7",
        attempt_id=seventh["attempt_id"],
        summary_advance={
            "expected_projection_version": seventh["projection"]["version"],
            "summary_text": "The first two turns are summarized.",
            "summary_through_sequence": 2,
        },
    )
    assert partial_advance.status_code == 200

    next_turn = _start_turn(
        conversation_db,
        "conversation-summary-source",
        request_id="request-8",
    ).json()
    assert next_turn["projection"]["summary_through_sequence"] == 2
    assert next_turn["projection"]["summary_text"] == (
        "The first two turns are summarized."
    )
    assert all(
        turn["sequence"] > 2 for turn in next_turn["recent_turns"]
    )
    assert [
        turn["sequence"]
        for turn in next_turn["summary_compaction_source"]["turns"]
    ] == [3, 4, 5, 6]


def test_selected_profile_is_bound_and_returns_authoritative_context(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-profile",
        request_id="request-profile",
        shopper_profile_id="shopper_morgan",
    )

    assert started.status_code == 200
    assert started.json()["shopper_context"] == {
        "shopper_type": "skeptical_researcher",
        "behavior": (
            "Probes for material, care burden, and repeated-wear practicality "
            "before choosing."
        ),
        "zipcode": "60601",
    }
    assert set(started.json()["shopper_context"]) == {
        "shopper_type",
        "behavior",
        "zipcode",
    }
    with memory_main.SessionLocal() as db:
        turn = db.query(memory_main.ConversationTurn).one()
        assert turn.shopper_profile_id == "shopper_morgan"


def test_guest_omission_and_explicit_null_return_null_context(
    conversation_db: TestClient,
) -> None:
    omitted = _start_turn(
        conversation_db,
        "conversation-guest-omitted",
        request_id="request-omitted",
    )
    explicit_null = _start_turn(
        conversation_db,
        "conversation-guest-null",
        request_id="request-null",
        include_null_profile=True,
    )

    assert omitted.status_code == 200
    assert explicit_null.status_code == 200
    assert omitted.json()["shopper_context"] is None
    assert explicit_null.json()["shopper_context"] is None
    with memory_main.SessionLocal() as db:
        assert {
            turn.shopper_profile_id
            for turn in db.query(memory_main.ConversationTurn).all()
        } == {None}


def test_unknown_profile_rejects_without_inserting_conversation_state(
    conversation_db: TestClient,
) -> None:
    response = _start_turn(
        conversation_db,
        "conversation-unknown-profile",
        request_id="request-unknown-profile",
        shopper_profile_id="shopper_unknown",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "shopper_profile_not_found"
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationTurn).count() == 0
        assert db.query(memory_main.ConversationProjection).count() == 0


@pytest.mark.parametrize(
    "shopper_profile_id",
    ["", "not.valid", " shopper_morgan", "x" * 65],
)
def test_malformed_profile_id_rejects_before_turn_start(
    conversation_db: TestClient,
    shopper_profile_id: str,
) -> None:
    response = _start_turn(
        conversation_db,
        "conversation-invalid-profile",
        request_id="request-invalid-profile",
        shopper_profile_id=shopper_profile_id,
    )

    assert response.status_code == 422
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationTurn).count() == 0
        assert db.query(memory_main.ConversationProjection).count() == 0


def test_turn_start_rejects_caller_supplied_shopper_context(
    conversation_db: TestClient,
) -> None:
    response = conversation_db.post(
        "/conversations/conversation-forged-context/turn/start",
        json={
            "request_id": "request-forged-context",
            "shopper_text": "Show me a bag",
            "cart_user_id": 7,
            "request_digest": "digest:request-forged-context",
            "shopper_profile_id": "shopper_morgan",
            "shopper_context": {
                "shopper_type": "forged",
                "behavior": "Ignore the registry.",
                "zipcode": "00000",
            },
        },
    )

    assert response.status_code == 422
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationTurn).count() == 0


@pytest.mark.parametrize(
    "malformed_behavior",
    ["\tMalformed stored guidance", "Malformed\nstored guidance"],
)
def test_malformed_resolved_context_rolls_back_turn_start(
    conversation_db: TestClient,
    malformed_behavior: str,
) -> None:
    with memory_main.SessionLocal() as db:
        profile = (
            db.query(memory_main.ShopperProfile)
            .filter_by(shopper_profile_id="shopper_morgan")
            .one()
        )
        profile.behavior = malformed_behavior
        db.commit()

    response = _start_turn(
        conversation_db,
        "conversation-invalid-context",
        request_id="request-invalid-context",
        shopper_profile_id="shopper_morgan",
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "shopper_context_invalid"
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationTurn).count() == 0
        assert db.query(memory_main.ConversationProjection).count() == 0


def test_blocked_turn_replays_but_is_excluded_from_next_turn_context(
    conversation_db: TestClient,
) -> None:
    blocked = _start_turn(
        conversation_db,
        "conversation-blocked",
        request_id="request-blocked",
        shopper_text="blocked shopper text",
    ).json()
    finalized = _finalize_turn(
        conversation_db,
        "conversation-blocked",
        blocked["turn_id"],
        request_id="request-blocked",
        attempt_id=blocked["attempt_id"],
        assistant_text="blocked response",
        status="blocked",
    )

    replay = _start_turn(
        conversation_db,
        "conversation-blocked",
        request_id="request-blocked",
        shopper_text="blocked shopper text",
    )
    next_turn = _start_turn(
        conversation_db,
        "conversation-blocked",
        request_id="request-safe",
        shopper_text="Show me a bag",
    )

    assert finalized.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["status"] == "blocked"
    assert replay.json()["assistant_text"] == "blocked response"
    assert next_turn.status_code == 200
    assert next_turn.json()["recent_turns"] == []
    assert next_turn.json()["previous_selected_skill_names"] == []
    with memory_main.SessionLocal() as db:
        stored = (
            db.query(memory_main.ConversationTurn)
            .filter_by(turn_id=blocked["turn_id"])
            .one()
        )
        assert stored.shopper_text == "blocked shopper text"
        assert stored.status == "blocked"


def test_start_is_idempotent_and_rejects_active_or_conflicting_reuse(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-idempotent",
        request_id="request-1",
    )
    active_replay = _start_turn(
        conversation_db,
        "conversation-idempotent",
        request_id="request-1",
    )
    another_request = _start_turn(
        conversation_db,
        "conversation-idempotent",
        request_id="request-2",
    )

    assert active_replay.status_code == 409
    assert active_replay.json()["detail"] == "turn_in_progress"
    assert another_request.status_code == 409
    assert another_request.json()["detail"] == "conversation_turn_in_progress"

    finalized = _finalize_turn(
        conversation_db,
        "conversation-idempotent",
        started.json()["turn_id"],
        request_id="request-1",
        attempt_id=started.json()["attempt_id"],
        output={
            "product_results": [{"product_ref": "bag-1"}],
            "retrieved": {"Structured Bag": "/images/bag.png"},
            "agent_diagnostics": {"final_termination_reason": "completed"},
        },
    )
    replay = _start_turn(
        conversation_db,
        "conversation-idempotent",
        request_id="request-1",
    )
    conflict = _start_turn(
        conversation_db,
        "conversation-idempotent",
        request_id="request-1",
        shopper_text="Different request",
    )

    assert finalized.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["status"] == "completed"
    assert replay.json()["assistant_text"] == "Here is a bag."
    assert replay.json()["output"] == {
        "product_results": [{"product_ref": "bag-1"}],
        "retrieved": {"Structured Bag": "/images/bag.png"},
        "agent_diagnostics": {"final_termination_reason": "completed"},
        "selected_skill_names": [],
    }
    assert conflict.status_code == 409


def test_same_profile_exact_retry_replays_context_and_allows_next_turn(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-profile-retry",
        request_id="request-profile-1",
        shopper_profile_id="shopper_casey",
    ).json()
    finalized = _finalize_turn(
        conversation_db,
        "conversation-profile-retry",
        started["turn_id"],
        request_id="request-profile-1",
        attempt_id=started["attempt_id"],
    )
    replay = _start_turn(
        conversation_db,
        "conversation-profile-retry",
        request_id="request-profile-1",
        shopper_profile_id="shopper_casey",
    )
    next_turn = _start_turn(
        conversation_db,
        "conversation-profile-retry",
        request_id="request-profile-2",
        shopper_text="Show me shoes",
        shopper_profile_id="shopper_casey",
    )

    assert finalized.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["turn_id"] == started["turn_id"]
    assert replay.json()["shopper_context"] == started["shopper_context"]
    assert next_turn.status_code == 200
    assert next_turn.json()["sequence"] == 2
    assert next_turn.json()["shopper_context"] == started["shopper_context"]
    with memory_main.SessionLocal() as db:
        assert {
            turn.shopper_profile_id
            for turn in db.query(memory_main.ConversationTurn).all()
        } == {"shopper_casey"}


def test_same_request_id_with_another_profile_is_input_conflict(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-cross-profile-retry",
        request_id="request-cross-profile",
        request_digest="same-digest",
        shopper_profile_id="shopper_morgan",
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-cross-profile-retry",
            started["turn_id"],
            request_id="request-cross-profile",
            attempt_id=started["attempt_id"],
        ).status_code
        == 200
    )

    conflict = _start_turn(
        conversation_db,
        "conversation-cross-profile-retry",
        request_id="request-cross-profile",
        request_digest="same-digest",
        shopper_profile_id="shopper_riley",
    )

    assert conflict.status_code == 409
    assert conflict.json()["detail"] == (
        "request_id was already used for different turn input"
    )
    with memory_main.SessionLocal() as db:
        turns = db.query(memory_main.ConversationTurn).all()
        assert len(turns) == 1
        assert turns[0].shopper_profile_id == "shopper_morgan"


@pytest.mark.parametrize(
    ("original_profile", "replacement_profile"),
    [
        (None, "shopper_morgan"),
        ("shopper_morgan", None),
        ("shopper_morgan", "shopper_riley"),
    ],
)
def test_conversation_profile_binding_cannot_change(
    conversation_db: TestClient,
    original_profile: str | None,
    replacement_profile: str | None,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-profile-mismatch",
        request_id="request-original",
        shopper_profile_id=original_profile,
    ).json()
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-profile-mismatch",
            started["turn_id"],
            request_id="request-original",
            attempt_id=started["attempt_id"],
        ).status_code
        == 200
    )

    mismatch = _start_turn(
        conversation_db,
        "conversation-profile-mismatch",
        request_id="request-replacement",
        shopper_profile_id=replacement_profile,
    )

    assert mismatch.status_code == 409
    assert mismatch.json()["detail"] == "conversation_profile_mismatch"
    with memory_main.SessionLocal() as db:
        turns = db.query(memory_main.ConversationTurn).all()
        assert len(turns) == 1
        assert turns[0].shopper_profile_id == original_profile


def test_finalize_indexes_ordered_presented_products_once(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-presented-products",
        request_id="request-products",
    ).json()
    products = [
        {
            "product_id": "bag-1",
            "display_name": "Structured Tote",
            "category": "tote_bags",
            "price": {"amount": 59.99, "currency": "USD"},
            "attributes": {"taxonomy": {"category": "bags"}},
        },
        {
            "product_id": "bag-2",
            "display_name": "Cobalt Crossbody",
            "category": "crossbody_bags",
            "image_url": "/images/bag-2.png",
        },
    ]
    output = {
        "product_results": products,
        "retrieved": {"Cobalt Crossbody": "/images/bag-2.png"},
        "agent_diagnostics": {},
    }

    first = _finalize_turn(
        conversation_db,
        "conversation-presented-products",
        started["turn_id"],
        request_id="request-products",
        attempt_id=started["attempt_id"],
        output=output,
    )
    replay = _finalize_turn(
        conversation_db,
        "conversation-presented-products",
        started["turn_id"],
        request_id="request-products",
        attempt_id=started["attempt_id"],
        output=output,
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    with memory_main.SessionLocal() as db:
        events = db.query(memory_main.ConversationEvent).all()
        assert len(events) == 1
        assert events[0].event_type == "candidate_set_presented"
        assert events[0].source_kind == "runtime"
        assert events[0].logical_order == 1
        assert json.loads(events[0].payload_json) == {"products": products}
        candidate_set_id = events[0].event_id

    next_turn = _start_turn(
        conversation_db,
        "conversation-presented-products",
        request_id="request-next",
    )

    assert next_turn.status_code == 200
    assert next_turn.json()["projection"]["product_reference_index"] == [
        {
            "candidate_set_id": candidate_set_id,
            "catalog_revision": "catalog-v1",
            "turn_seq": 1,
            "products": [
                {
                    "ref": "bag-1",
                    "name": "Structured Tote",
                    "category": "tote_bags",
                    "position": 1,
                },
                {
                    "ref": "bag-2",
                    "name": "Cobalt Crossbody",
                    "category": "crossbody_bags",
                    "position": 2,
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "products",
    [
        [],
        [
            {"product_id": "", "display_name": "Missing ID"},
            {"product_id": "bag-1", "display_name": " "},
            {"display_name": "Missing product ID"},
        ],
    ],
)
def test_finalize_does_not_index_empty_or_malformed_products(
    conversation_db: TestClient,
    products: list[dict],
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-no-products",
        request_id="request-no-products",
    ).json()

    finalized = _finalize_turn(
        conversation_db,
        "conversation-no-products",
        started["turn_id"],
        request_id="request-no-products",
        attempt_id=started["attempt_id"],
        output={
            "product_results": products,
            "retrieved": {},
            "agent_diagnostics": {},
        },
    )

    assert finalized.status_code == 200
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationEvent).count() == 0
        projection = db.query(memory_main.ConversationProjection).one()
        assert projection.product_reference_index_json == "[]"


def test_product_reference_index_keeps_newest_complete_sets_within_budget(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(product_references, "_MAX_REFERENCE_INDEX_CHARS", 400)
    for index in range(1, 5):
        _present_products(
            conversation_db,
            "conversation-bounded-index",
            request_id=f"request-{index}",
            products=[
                {
                    "product_id": f"product-{index}",
                    "display_name": f"Product {index} " + ("x" * 100),
                }
            ],
        )

    with memory_main.SessionLocal() as db:
        stored = db.query(memory_main.ConversationProjection).one()
        reference_sets = json.loads(stored.product_reference_index_json)

    assert len(stored.product_reference_index_json) <= 400
    assert reference_sets[-1]["products"][0]["ref"] == "product-4"
    assert all(
        reference_set["products"][0]["ref"] != "product-1"
        for reference_set in reference_sets
    )


def test_product_resolution_batches_unique_ambiguous_and_missing_results(
    conversation_db: TestClient,
) -> None:
    products = [
        {
            "product_id": "bag-1",
            "display_name": "Structured Tote",
            "category": "tote_bags",
            "price": {"amount": 59.99, "currency": "USD"},
        },
        {
            "product_id": "bag-2",
            "display_name": "Canvas Tote",
            "category": "tote_bags",
        },
    ]
    _, candidate_set_id = _present_products(
        conversation_db,
        "conversation-resolve",
        request_id="request-resolve",
        products=products,
    )

    response = conversation_db.post(
        "/conversations/conversation-resolve/products/resolve",
        json={
            "references": [
                {
                    "reference_id": "unique",
                    "display_name": "  structured tote  ",
                },
                {"reference_id": "ambiguous", "category": "TOTE_BAGS"},
                {"reference_id": "missing", "display_name": "Missing Bag"},
            ]
        },
    )

    assert response.status_code == 200
    results = {result["reference_id"]: result for result in response.json()["results"]}
    assert results["unique"] == {
        "reference_id": "unique",
        "status": "resolved",
        "matches": [
            {
                "product": products[0],
                "candidate_set_id": candidate_set_id,
                "turn_sequence": 1,
                "position": 1,
                "catalog_revision": "catalog-v1",
            }
        ],
        "match_count": 1,
    }
    assert results["ambiguous"]["status"] == "ambiguous"
    assert results["ambiguous"]["match_count"] == 2
    assert [
        match["product"]["product_id"]
        for match in results["ambiguous"]["matches"]
    ] == ["bag-1", "bag-2"]
    assert results["missing"] == {
        "reference_id": "missing",
        "status": "not_found",
        "matches": [],
        "match_count": 0,
    }


def test_product_resolution_uses_candidate_set_ordinal(
    conversation_db: TestClient,
) -> None:
    products = [
        {"product_id": "bag-1", "display_name": "First Bag"},
        {"product_id": "bag-2", "display_name": "Second Bag"},
    ]
    _, candidate_set_id = _present_products(
        conversation_db,
        "conversation-ordinal",
        request_id="request-ordinal",
        products=products,
    )

    response = conversation_db.post(
        "/conversations/conversation-ordinal/products/resolve",
        json={
            "references": [
                {
                    "reference_id": "second",
                    "candidate_set_id": candidate_set_id,
                    "ordinal": 2,
                }
            ]
        },
    )

    result = response.json()["results"][0]
    assert response.status_code == 200
    assert result["status"] == "resolved"
    assert result["matches"][0]["product"] == products[1]
    assert result["matches"][0]["position"] == 2


def test_product_resolution_deduplicates_repeated_ref_using_latest_occurrence(
    conversation_db: TestClient,
) -> None:
    earlier = {
        "product_id": "bag-1",
        "display_name": "Structured Tote",
        "price": {"amount": 59.99, "currency": "USD"},
    }
    latest = {
        "product_id": "bag-1",
        "display_name": "Structured Tote",
        "price": {"amount": 49.99, "currency": "USD"},
    }
    _present_products(
        conversation_db,
        "conversation-repeated-ref",
        request_id="request-earlier",
        products=[earlier],
    )
    _, latest_set_id = _present_products(
        conversation_db,
        "conversation-repeated-ref",
        request_id="request-latest",
        products=[latest],
    )

    response = conversation_db.post(
        "/conversations/conversation-repeated-ref/products/resolve",
        json={"references": [{"reference_id": "bag", "product_ref": "bag-1"}]},
    )

    result = response.json()["results"][0]
    assert response.status_code == 200
    assert result["status"] == "resolved"
    assert result["match_count"] == 1
    assert result["matches"] == [
        {
            "product": latest,
            "candidate_set_id": latest_set_id,
            "turn_sequence": 2,
            "position": 1,
            "catalog_revision": "catalog-v1",
        }
    ]


def test_product_resolution_is_conversation_scoped(
    conversation_db: TestClient,
) -> None:
    _, candidate_set_id = _present_products(
        conversation_db,
        "conversation-private",
        request_id="request-private",
        products=[{"product_id": "bag-private", "display_name": "Private Bag"}],
    )

    response = conversation_db.post(
        "/conversations/conversation-other/products/resolve",
        json={
            "references": [
                {
                    "reference_id": "cross-conversation",
                    "candidate_set_id": candidate_set_id,
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == [
        {
            "reference_id": "cross-conversation",
            "status": "not_found",
            "matches": [],
            "match_count": 0,
        }
    ]


def test_product_resolution_reads_history_outside_recent_turn_window(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_RECENT_TURNS", "1")
    _present_products(
        conversation_db,
        "conversation-distant",
        request_id="request-1",
        products=[{"product_id": "bag-old", "display_name": "Earlier Bag"}],
    )
    _present_products(
        conversation_db,
        "conversation-distant",
        request_id="request-2",
        products=[{"product_id": "shoe-2", "display_name": "Second Turn Shoe"}],
    )
    _present_products(
        conversation_db,
        "conversation-distant",
        request_id="request-3",
        products=[{"product_id": "watch-3", "display_name": "Third Turn Watch"}],
    )
    next_turn = _start_turn(
        conversation_db,
        "conversation-distant",
        request_id="request-4",
    )

    response = conversation_db.post(
        "/conversations/conversation-distant/products/resolve",
        json={
            "references": [
                {"reference_id": "old", "display_name": "Earlier Bag"}
            ]
        },
    )

    assert next_turn.status_code == 200
    assert [turn["sequence"] for turn in next_turn.json()["recent_turns"]] == [3]
    result = response.json()["results"][0]
    assert result["status"] == "resolved"
    assert result["matches"][0]["turn_sequence"] == 1
    assert result["matches"][0]["product"]["product_id"] == "bag-old"


def test_product_resolution_requires_selector_and_scoped_ordinal(
    conversation_db: TestClient,
) -> None:
    path = "/conversations/conversation-validation/products/resolve"

    missing_selector = conversation_db.post(
        path,
        json={"references": [{"reference_id": "missing"}]},
    )
    unscoped_ordinal = conversation_db.post(
        path,
        json={
            "references": [
                {"reference_id": "ordinal", "category": "bags", "ordinal": 2}
            ]
        },
    )

    assert missing_selector.status_code == 422
    assert unscoped_ordinal.status_code == 422


def test_finalize_persists_ordered_events_and_replays_exactly(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-events",
        request_id="request-events",
    ).json()
    events = [
        {
            "event_key": "candidate-set-1",
            "event_type": "candidate_set_presented",
            "source_kind": "catalog",
            "source_ref": "search-1",
            "payload": {
                "products": [
                    {"product_id": "bag-1", "display_name": "Bag One"}
                ]
            },
        },
        {
            "event_key": "detail-1",
            "event_type": "product_detail_confirmed",
            "source_kind": "catalog",
            "source_ref": "bag-1",
            "payload": {"material": "cotton"},
        },
    ]
    first = _finalize_turn(
        conversation_db,
        "conversation-events",
        started["turn_id"],
        request_id="request-events",
        attempt_id=started["attempt_id"],
        events=events,
    )
    replay = _finalize_turn(
        conversation_db,
        "conversation-events",
        started["turn_id"],
        request_id="request-events",
        attempt_id=started["attempt_id"],
        events=events,
    )
    conflict = _finalize_turn(
        conversation_db,
        "conversation-events",
        started["turn_id"],
        request_id="request-events",
        attempt_id=started["attempt_id"],
        assistant_text="A different answer.",
        events=events,
    )

    assert first.status_code == 200
    assert replay.json() == {**first.json(), "replayed": True}
    assert conflict.status_code == 409
    with memory_main.SessionLocal() as db:
        stored_events = (
            db.query(memory_main.ConversationEvent)
            .order_by(memory_main.ConversationEvent.logical_order)
            .all()
        )
        projection = db.query(memory_main.ConversationProjection).one()
        assert [event.event_key for event in stored_events] == [
            "candidate-set-1",
            "detail-1",
        ]
        assert [event.logical_order for event in stored_events] == [1, 2]
        assert projection.version == 1
        assert projection.last_turn_id == started["turn_id"]
        assert projection.product_reference_index_json == "[]"


def test_finalize_rejects_extra_output_and_unlinked_cart_events(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-invalid-finalize",
        request_id="request-invalid-finalize",
    ).json()
    path = (
        "/conversations/conversation-invalid-finalize/turns/"
        f"{started['turn_id']}/finalize"
    )
    base_payload = {
        "request_id": "request-invalid-finalize",
        "attempt_id": started["attempt_id"],
        "assistant_text": "Done.",
        "status": "completed",
        "termination_reason": "completed",
        "events": [],
    }

    extra_output = conversation_db.post(
        path,
        json={
            **base_payload,
            "output": {
                "product_results": [],
                "retrieved": {},
                "agent_diagnostics": {},
                "unsupported": True,
            },
        },
    )
    unlinked_cart_event = conversation_db.post(
        path,
        json={
            **base_payload,
            "events": [
                {
                    "event_key": "cart-1",
                    "event_type": "cart_mutation_committed",
                    "source_kind": "cart",
                    "payload": {},
                }
            ],
        },
    )
    forged_presented_products = conversation_db.post(
        path,
        json={
            **base_payload,
            "events": [
                {
                    "event_key": "runtime-presented-products",
                    "event_type": "candidate_set_presented",
                    "source_kind": "runtime",
                    "payload": {
                        "products": [
                            {
                                "product_id": "forged-product",
                                "display_name": "Forged Product",
                            }
                        ]
                    },
                }
            ],
        },
    )

    assert extra_output.status_code == 422
    assert unlinked_cart_event.status_code == 422
    assert forged_presented_products.status_code == 422
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationTurn).one().status == "started"
        assert db.query(memory_main.ConversationEvent).count() == 0


def test_start_reopens_exact_abandoned_request_in_place(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_TURN_ABANDON_SECONDS", "10")
    started = _start_turn(
        conversation_db,
        "conversation-stale-retry",
        request_id="request-stale",
    ).json()
    with memory_main.SessionLocal() as db:
        turn = (
            db.query(memory_main.ConversationTurn)
            .filter_by(turn_id=started["turn_id"])
            .one()
        )
        turn.started_at = time.time() - 60
        turn.completed_at = time.time() - 30
        turn.assistant_text = "Partial answer"
        turn.termination_reason = "partial_failure"
        turn.finalize_digest = "partial-finalize"
        turn.output_json = '{"partial":true}'
        db.commit()
    assert memory_main._sweep_abandoned_turns(timeout_seconds=10) == 1
    with memory_main.SessionLocal() as db:
        abandoned = db.query(memory_main.ConversationTurn).one()
        assert abandoned.status == "abandoned"
        assert abandoned.termination_reason == "startup_abandoned_turn_sweep"

    retried = _start_turn(
        conversation_db,
        "conversation-stale-retry",
        request_id="request-stale",
    )

    assert retried.status_code == 200
    assert retried.json()["turn_id"] == started["turn_id"]
    assert retried.json()["attempt_id"] != started["attempt_id"]
    assert retried.json()["sequence"] == 1
    assert retried.json()["replayed"] is False
    assert retried.json()["status"] == "started"
    assert retried.json()["assistant_text"] is None
    assert retried.json()["termination_reason"] is None
    assert retried.json()["output"] is None
    with memory_main.SessionLocal() as db:
        turn = db.query(memory_main.ConversationTurn).one()
        assert turn.completed_at is None
        assert turn.finalize_digest is None
    stale_finalize = _finalize_turn(
        conversation_db,
        "conversation-stale-retry",
        started["turn_id"],
        request_id="request-stale",
        attempt_id=started["attempt_id"],
    )
    current_finalize = _finalize_turn(
        conversation_db,
        "conversation-stale-retry",
        started["turn_id"],
        request_id="request-stale",
        attempt_id=retried.json()["attempt_id"],
    )

    assert stale_finalize.status_code == 409
    assert stale_finalize.json()["detail"] == "turn_attempt_superseded"
    assert current_finalize.status_code == 200


def test_start_abandons_stale_other_request_and_advances_sequence(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_TURN_ABANDON_SECONDS", "10")
    started = _start_turn(
        conversation_db,
        "conversation-stale-replacement",
        request_id="request-stale",
    ).json()
    with memory_main.SessionLocal() as db:
        turn = (
            db.query(memory_main.ConversationTurn)
            .filter_by(turn_id=started["turn_id"])
            .one()
        )
        turn.started_at = time.time() - 60
        db.commit()

    replacement = _start_turn(
        conversation_db,
        "conversation-stale-replacement",
        request_id="request-replacement",
    )
    assert (
        _finalize_turn(
            conversation_db,
            "conversation-stale-replacement",
            replacement.json()["turn_id"],
            request_id="request-replacement",
            attempt_id=replacement.json()["attempt_id"],
        ).status_code
        == 200
    )
    superseded_retry = _start_turn(
        conversation_db,
        "conversation-stale-replacement",
        request_id="request-stale",
    )

    assert replacement.status_code == 200
    assert replacement.json()["sequence"] == 2
    assert replacement.json()["recent_turns"] == []
    assert superseded_retry.status_code == 409
    assert superseded_retry.json()["detail"] == "turn_superseded"
    with memory_main.SessionLocal() as db:
        abandoned = (
            db.query(memory_main.ConversationTurn)
            .filter_by(turn_id=started["turn_id"])
            .one()
        )
        assert abandoned.status == "abandoned"
        assert abandoned.termination_reason == "turn_start_abandoned_timeout"


def test_stale_retry_reuses_cart_mutation_idempotency_key(
    conversation_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_TURN_ABANDON_SECONDS", "10")
    request_id = "request-cart-crash"
    started = _start_turn(
        conversation_db,
        "conversation-cart-crash",
        request_id=request_id,
    ).json()
    mutation = {
        "product_id": "bag-1",
        "item": "Structured Bag",
        "amount": 1,
        "price": 45.0,
        "idempotency_key": f"{request_id}:add:bag-1:1",
    }
    first_add = conversation_db.post("/user/7/cart/add", json=mutation)
    with memory_main.SessionLocal() as db:
        turn = (
            db.query(memory_main.ConversationTurn)
            .filter_by(turn_id=started["turn_id"])
            .one()
        )
        turn.started_at = time.time() - 60
        db.commit()

    retried = _start_turn(
        conversation_db,
        "conversation-cart-crash",
        request_id=request_id,
    )
    repeated_add = conversation_db.post("/user/7/cart/add", json=mutation)

    assert first_add.status_code == 200
    assert retried.status_code == 200
    assert retried.json()["turn_id"] == started["turn_id"]
    assert repeated_add.status_code == 200
    assert repeated_add.json() == first_add.json()
    assert conversation_db.get("/user/7/cart").json()["cart"][0]["amount"] == 1
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.CartMutation).count() == 1


def test_delete_cascades_conversation_rows_but_preserves_cart_and_context(
    conversation_db: TestClient,
) -> None:
    conversation_db.post("/user/7/context/replace", json={"new_context": "legacy"})
    conversation_db.post(
        "/user/7/cart/add",
        json={
            "product_id": "bag-1",
            "item": "Structured Bag",
            "amount": 1,
            "idempotency_key": "keep-cart",
        },
    )
    started = _start_turn(
        conversation_db,
        "conversation-delete",
        request_id="request-delete",
    ).json()
    _finalize_turn(
        conversation_db,
        "conversation-delete",
        started["turn_id"],
        request_id="request-delete",
        attempt_id=started["attempt_id"],
        events=[
            {
                "event_key": "event-delete",
                "event_type": "candidate_set_presented",
                "source_kind": "catalog",
                "payload": {},
            }
        ],
    )

    deleted = conversation_db.delete("/conversations/conversation-delete")

    assert deleted.status_code == 200
    assert deleted.json() == {
        "conversation_id": "conversation-delete",
        "deleted_turns": 1,
        "deleted_events": 1,
        "deleted_projection": True,
    }
    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ConversationTurn).count() == 0
        assert db.query(memory_main.ConversationEvent).count() == 0
        assert db.query(memory_main.ConversationProjection).count() == 0
        assert db.query(memory_main.CartMutation).count() == 1
    assert conversation_db.get("/user/7/cart").json()["cart"][0]["item"] == (
        "Structured Bag"
    )
    assert conversation_db.get("/user/7/context").json()["context"] == "legacy"


def test_turn_profile_foreign_key_restricts_delete_and_primary_key_update(
    conversation_db: TestClient,
) -> None:
    started = _start_turn(
        conversation_db,
        "conversation-profile-fk",
        request_id="request-profile-fk",
        shopper_profile_id="shopper_morgan",
    )
    assert started.status_code == 200

    with memory_main.SessionLocal() as db:
        with pytest.raises(IntegrityError):
            db.execute(
                text(
                    "DELETE FROM shopper_profiles "
                    "WHERE shopper_profile_id = 'shopper_morgan'"
                )
            )
        db.rollback()
        with pytest.raises(IntegrityError):
            db.execute(
                text(
                    "UPDATE shopper_profiles "
                    "SET shopper_profile_id = 'shopper_changed' "
                    "WHERE shopper_profile_id = 'shopper_morgan'"
                )
            )
        db.rollback()

        turn = db.query(memory_main.ConversationTurn).one()
        profile = db.query(memory_main.ShopperProfile).filter_by(
            shopper_profile_id="shopper_morgan"
        ).one()
        assert turn.shopper_profile_id == profile.shopper_profile_id


def test_versioned_migrations_upgrade_legacy_schema_once_without_data_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with legacy_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE cart_items ("
                "id INTEGER PRIMARY KEY, user_id INTEGER, item TEXT, amount INTEGER)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO cart_items (id, user_id, item, amount) "
                "VALUES (1, 7, 'Legacy Bag', 1)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE conversation_turns ("
                "turn_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, "
                "sequence INTEGER NOT NULL, request_id TEXT NOT NULL, "
                "request_digest TEXT NOT NULL, finalize_digest TEXT, "
                "cart_user_id INTEGER NOT NULL, shopper_text TEXT NOT NULL, "
                "assistant_text TEXT, status TEXT NOT NULL, "
                "termination_reason TEXT, catalog_revision TEXT, "
                "diagnostics_json TEXT, start_response_body TEXT NOT NULL, "
                "finalize_response_body TEXT, "
                "started_at REAL NOT NULL, completed_at REAL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO conversation_turns ("
                "turn_id, conversation_id, sequence, request_id, "
                "request_digest, cart_user_id, shopper_text, status, "
                "start_response_body, started_at"
                ") VALUES ("
                "'legacy-turn', 'legacy-conversation', 1, 'legacy-request', "
                "'legacy-digest', 7, 'Remember this', 'completed', '{}', 1.0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE conversation_projection ("
                "conversation_id TEXT PRIMARY KEY, version INTEGER NOT NULL, "
                "active_anchors_json TEXT NOT NULL, "
                "effective_preferences_json TEXT NOT NULL, "
                "product_reference_index_json TEXT NOT NULL, "
                "last_turn_id TEXT)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO conversation_projection ("
                "conversation_id, version, active_anchors_json, "
                "effective_preferences_json, product_reference_index_json, "
                "last_turn_id"
                ") VALUES ("
                ":conversation_id, :version, :active_anchors, "
                ":effective_preferences, :product_reference_index, "
                ":last_turn_id)"
            ),
            {
                "conversation_id": "legacy-conversation",
                "version": 3,
                "active_anchors": '["anchor"]',
                "effective_preferences": ('[{"field":"color","value":"blue"}]'),
                "product_reference_index": '[{"turn_seq":1}]',
                "last_turn_id": "legacy-turn",
            },
        )
    monkeypatch.setattr(memory_main, "engine", legacy_engine)
    monkeypatch.setattr(
        memory_main,
        "SessionLocal",
        sessionmaker(bind=legacy_engine, expire_on_commit=False),
    )

    memory_main._run_schema_migrations()
    memory_main._run_schema_migrations()

    with memory_main.SessionLocal() as db:
        db.add(
            memory_main.ConversationTurn(
                turn_id="current-turn",
                conversation_id="current-conversation",
                sequence=1,
                request_id="current-request",
                request_digest="current-digest",
                cart_user_id=8,
                shopper_text="Start a current turn",
                status="started",
                started_at=2.0,
            )
        )
        db.commit()

    with legacy_engine.connect() as connection:
        versions = (
            connection.execute(
                text("SELECT version FROM schema_migrations ORDER BY version")
            )
            .scalars()
            .all()
        )
        row = connection.execute(
            text(
                "SELECT item, cart_line_id, product_id, price "
                "FROM cart_items WHERE id = 1"
            )
        ).one()
        attempt_id = connection.execute(
            text(
                "SELECT attempt_id FROM conversation_turns "
                "WHERE turn_id = 'legacy-turn'"
            )
        ).scalar_one()
        shopper_profile_id = connection.execute(
            text(
                "SELECT shopper_profile_id FROM conversation_turns "
                "WHERE turn_id = 'legacy-turn'"
            )
        ).scalar_one_or_none()
        turn_columns = set(
            connection.execute(
                text("PRAGMA table_info('conversation_turns')")
            ).scalars(index=1)
        )
        turn_count = connection.execute(
            text("SELECT COUNT(*) FROM conversation_turns")
        ).scalar_one()
        turn_indexes = set(
            connection.execute(
                text("PRAGMA index_list('conversation_turns')")
            ).scalars(index=1)
        )
        profile_foreign_key = next(
            row
            for row in connection.execute(
                text("PRAGMA foreign_key_list('conversation_turns')")
            ).mappings()
            if row["from"] == "shopper_profile_id"
        )
        tables = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).scalars()
        )
        projection_row = connection.execute(
            text(
                "SELECT version, summary_text, summary_through_sequence, "
                "active_receipts_json, current_weather_scope_json, "
                "active_anchors_json, effective_preferences_json, "
                "product_reference_index_json, last_turn_id "
                "FROM conversation_projection "
                "WHERE conversation_id = 'legacy-conversation'"
            )
        ).one()

    assert versions == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert row[0] == "Legacy Bag"
    assert len(row[1]) == 32
    assert row[2:] == (None, None)
    assert len(attempt_id) == 32
    assert shopper_profile_id is None
    assert {
        "diagnostics_json",
        "start_response_body",
        "finalize_response_body",
    }.isdisjoint(turn_columns)
    assert turn_count == 2
    assert "uq_conversation_started" in turn_indexes
    assert profile_foreign_key["table"] == "shopper_profiles"
    assert profile_foreign_key["to"] == "shopper_profile_id"
    assert profile_foreign_key["on_delete"] == "RESTRICT"
    assert profile_foreign_key["on_update"] == "RESTRICT"
    assert projection_row == (
        3,
        "",
        0,
        "[]",
        '{"revision":0}',
        '["anchor"]',
        '[{"field":"color","value":"blue"}]',
        '[{"turn_seq":1}]',
        "legacy-turn",
    )
    assert {
        "conversation_turns",
        "conversation_events",
        "conversation_projection",
        "shopper_profiles",
    } <= tables
    legacy_engine.dispose()


def test_file_database_reopens_with_sqlite_safety_settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "memory.db"
    database_url = f"sqlite:///{database_path}"
    first_engine = memory_main.build_engine(database_url, busy_timeout_ms=2500)
    monkeypatch.setattr(memory_main, "engine", first_engine)
    monkeypatch.setattr(
        memory_main,
        "SessionLocal",
        sessionmaker(bind=first_engine, expire_on_commit=False),
    )
    memory_main._run_schema_migrations()
    with memory_main.SessionLocal() as db:
        db.add(
            memory_main.ConversationTurn(
                turn_id="durable-turn",
                conversation_id="durable-conversation",
                sequence=1,
                request_id="durable-request",
                request_digest="durable-digest",
                cart_user_id=7,
                shopper_text="Remember this",
                status="completed",
                assistant_text="Remembered.",
                started_at=1.0,
                completed_at=2.0,
            )
        )
        db.commit()
    first_engine.dispose()

    reopened_engine = memory_main.build_engine(database_url, busy_timeout_ms=2500)
    monkeypatch.setattr(memory_main, "engine", reopened_engine)
    monkeypatch.setattr(
        memory_main,
        "SessionLocal",
        sessionmaker(bind=reopened_engine, expire_on_commit=False),
    )
    memory_main._run_schema_migrations()

    with reopened_engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 2500
        assert (
            connection.execute(
                text("SELECT assistant_text FROM conversation_turns")
            ).scalar_one()
            == "Remembered."
        )
        assert (
        connection.execute(
            text("SELECT COUNT(*) FROM schema_migrations")
        ).scalar_one()
        == 10
    )
    reopened_engine.dispose()


def test_fresh_projection_schema_supports_staging_shaped_insert() -> None:
    fresh_engine = memory_main.build_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
    )
    memory_main.run_schema_migrations(fresh_engine)

    with fresh_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO conversation_projection ("
                "conversation_id, version, active_anchors_json, "
                "effective_preferences_json, product_reference_index_json, "
                "last_turn_id"
                ") VALUES ("
                "'rollback-conversation', 0, '[]', '[]', '[]', NULL)"
            )
        )
        values = connection.execute(
            text(
                "SELECT summary_text, summary_through_sequence, "
                "active_receipts_json, current_weather_scope_json "
                "FROM conversation_projection "
                "WHERE conversation_id = 'rollback-conversation'"
            )
        ).one()
        defaults = {
            row["name"]: row["dflt_value"]
            for row in connection.execute(
                text("PRAGMA table_info('conversation_projection')")
            ).mappings()
            if row["name"]
            in {
                "summary_text",
                "summary_through_sequence",
                "active_receipts_json",
                "current_weather_scope_json",
            }
        }

    assert values == ("", 0, "[]", '{"revision":0}')
    assert all(default is not None for default in defaults.values())
    fresh_engine.dispose()
