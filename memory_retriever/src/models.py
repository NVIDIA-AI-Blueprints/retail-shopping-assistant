# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database models owned by the memory service."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)

from .database import Base


def new_cart_line_id() -> str:
    return uuid4().hex


def new_turn_attempt_id() -> str:
    return uuid4().hex


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    context = Column(String, default="")


class CartItem(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True)
    cart_line_id = Column(
        String,
        default=new_cart_line_id,
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(Integer, index=True)
    product_id = Column(String, nullable=True, index=True)
    item = Column(String)
    amount = Column(Integer)
    price = Column(Float, nullable=True)


class CartQuantityIdempotency(Base):
    """Legacy quantity-only ledger retained as a migration source."""

    __tablename__ = "cart_quantity_idempotency"
    idempotency_key = Column(String, primary_key=True)
    user_id = Column(Integer, nullable=False)
    cart_line_id = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    response_body = Column(String, nullable=False)


class CartMutation(Base):
    __tablename__ = "cart_mutations"
    user_id = Column(Integer, primary_key=True)
    idempotency_key = Column(String, primary_key=True)
    operation = Column(String, nullable=False)
    canonical_digest = Column(String, nullable=False)
    stable_target_id = Column(String, nullable=False)
    response_body = Column(String, nullable=False)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"
    version = Column(Integer, primary_key=True)
    applied_at = Column(Float, nullable=False)


class ShopperProfile(Base):
    """One immutable representative shopper managed by startup bootstrap."""

    __tablename__ = "shopper_profiles"
    __table_args__ = (
        UniqueConstraint(
            "shopper_type",
            name="uq_shopper_profiles_shopper_type",
        ),
        CheckConstraint(
            "length(shopper_profile_id) BETWEEN 1 AND 64 "
            "AND shopper_profile_id = trim(shopper_profile_id)",
            name="ck_shopper_profiles_id",
        ),
        CheckConstraint(
            "length(display_name) BETWEEN 1 AND 80 "
            "AND display_name = trim(display_name)",
            name="ck_shopper_profiles_display_name",
        ),
        CheckConstraint(
            "length(shopper_type) BETWEEN 1 AND 80 "
            "AND shopper_type = trim(shopper_type)",
            name="ck_shopper_profiles_type",
        ),
        CheckConstraint(
            "length(behavior) BETWEEN 1 AND 512 "
            "AND behavior = trim(behavior)",
            name="ck_shopper_profiles_behavior",
        ),
        CheckConstraint(
            "length(zipcode) = 5 AND zipcode NOT GLOB '*[^0-9]*'",
            name="ck_shopper_profiles_zipcode",
        ),
    )

    shopper_profile_id = Column(String(64), primary_key=True, nullable=False)
    display_name = Column(String(80), nullable=False)
    shopper_type = Column(String(80), nullable=False)
    behavior = Column(Text, nullable=False)
    zipcode = Column(String(5), nullable=False)


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_conversation_turn_sequence",
        ),
        UniqueConstraint(
            "conversation_id",
            "request_id",
            name="uq_conversation_turn_request",
        ),
        CheckConstraint(
            "status IN ('started', 'completed', 'failed', 'blocked', 'abandoned')",
            name="ck_conversation_turn_status",
        ),
        Index(
            "uq_conversation_started",
            "conversation_id",
            unique=True,
            sqlite_where=text("status = 'started'"),
        ),
    )

    turn_id = Column(String, primary_key=True)
    conversation_id = Column(String, nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    request_id = Column(String, nullable=False)
    request_digest = Column(String, nullable=False)
    attempt_id = Column(
        String,
        default=new_turn_attempt_id,
        nullable=False,
    )
    finalize_digest = Column(String, nullable=True)
    cart_user_id = Column(Integer, nullable=False)
    shopper_text = Column(Text, nullable=False)
    assistant_text = Column(Text, nullable=True)
    output_json = Column(Text, nullable=True)
    status = Column(String, nullable=False, index=True)
    termination_reason = Column(String, nullable=True)
    catalog_revision = Column(String, nullable=True)
    started_at = Column(Float, nullable=False)
    completed_at = Column(Float, nullable=True)


class ConversationEvent(Base):
    __tablename__ = "conversation_events"
    __table_args__ = (
        UniqueConstraint("turn_id", "event_key", name="uq_turn_event_key"),
        UniqueConstraint("turn_id", "logical_order", name="uq_turn_event_order"),
    )

    event_id = Column(String, primary_key=True)
    turn_id = Column(
        String,
        ForeignKey("conversation_turns.turn_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_key = Column(String, nullable=False)
    logical_order = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)
    source_kind = Column(String, nullable=False)
    source_ref = Column(String, nullable=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(Float, nullable=False)


class ConversationProjection(Base):
    __tablename__ = "conversation_projection"

    conversation_id = Column(String, primary_key=True)
    version = Column(Integer, nullable=False, default=0)
    active_anchors_json = Column(Text, nullable=False, default="[]")
    effective_preferences_json = Column(Text, nullable=False, default="[]")
    product_reference_index_json = Column(Text, nullable=False, default="[]")
    last_turn_id = Column(
        String,
        ForeignKey("conversation_turns.turn_id", ondelete="SET NULL"),
        nullable=True,
    )
