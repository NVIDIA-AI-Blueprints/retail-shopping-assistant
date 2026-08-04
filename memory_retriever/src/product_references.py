# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presented-product events and their compact conversation projection."""

from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func

from .models import ConversationEvent, ConversationProjection, ConversationTurn


PRESENTED_PRODUCTS_EVENT_KEY = "runtime-presented-products"
# Query safety bound only. The character budget below is the effective limit:
# a compact set of eight products is roughly 1KB, so ~15 sets survive and this
# row cap is never reached. Resolution itself is unbounded and still sees every
# presented-product event in the conversation.
_MAX_PRESENTED_EVENT_ROWS = 100
_MAX_REFERENCE_INDEX_CHARS = 16_384
_MAX_CLARIFICATION_MATCHES = 5


class _ReferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductReferenceDescriptor(_ReferenceModel):
    reference_id: str = Field(..., min_length=1, max_length=128)
    product_ref: str | None = Field(default=None, min_length=1, max_length=512)
    display_name: str | None = Field(default=None, min_length=1, max_length=512)
    category: str | None = Field(default=None, min_length=1, max_length=256)
    turn_sequence: int | None = Field(default=None, ge=1)
    candidate_set_id: str | None = Field(default=None, min_length=1, max_length=64)
    ordinal: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_selectors(self):
        selectors = (
            self.product_ref,
            self.display_name,
            self.category,
            self.turn_sequence,
            self.candidate_set_id,
        )
        if not any(selector is not None for selector in selectors):
            raise ValueError("At least one product reference selector is required")
        if self.ordinal is not None and (
            self.turn_sequence is None and self.candidate_set_id is None
        ):
            raise ValueError("ordinal requires turn_sequence or candidate_set_id")
        return self


class ProductResolutionRequest(_ReferenceModel):
    references: list[ProductReferenceDescriptor] = Field(
        ...,
        min_length=1,
        max_length=20,
    )


class ProductReferenceMatch(_ReferenceModel):
    product: dict[str, Any]
    candidate_set_id: str = Field(..., min_length=1, max_length=64)
    turn_sequence: int
    position: int
    catalog_revision: str | None = Field(default=None, max_length=512)


class ProductResolutionResult(_ReferenceModel):
    reference_id: str = Field(..., min_length=1, max_length=128)
    status: Literal["resolved", "ambiguous", "not_found"]
    matches: list[ProductReferenceMatch]
    match_count: int
    #: The one supplied field that stopped an otherwise clear match, when
    #: exactly one is responsible. Diagnosis only; it names no product.
    blocking_field: str | None = None


class ProductResolutionResponse(_ReferenceModel):
    results: list[ProductResolutionResult]


def append_presented_products_event(
    db,
    turn: ConversationTurn,
    product_results: list[dict[str, Any]],
    *,
    created_at: float,
) -> ConversationEvent | None:
    """Append one event for the ordered products returned to the shopper."""

    products = [
        dict(product)
        for product in product_results
        if _is_referenceable_product(product)
    ]
    if not products:
        return None

    logical_order = (
        db.query(func.max(ConversationEvent.logical_order))
        .filter(ConversationEvent.turn_id == turn.turn_id)
        .scalar()
        or 0
    ) + 1
    event = ConversationEvent(
        event_id=uuid4().hex,
        turn_id=turn.turn_id,
        event_key=PRESENTED_PRODUCTS_EVENT_KEY,
        logical_order=logical_order,
        event_type="candidate_set_presented",
        source_kind="runtime",
        source_ref=turn.catalog_revision,
        payload_json=_canonical_json({"products": products}),
        created_at=created_at,
    )
    db.add(event)
    return event


def rebuild_product_reference_index(
    db,
    projection: ConversationProjection,
) -> None:
    """Rebuild the compact index from durable presented-product events."""

    rows = _recent_presented_event_rows(db, projection.conversation_id)
    reference_sets = []
    for event, turn in rows:
        products = _compact_products(event.payload_json)
        if not products:
            continue
        reference_set: dict[str, Any] = {
            "candidate_set_id": event.event_id,
            "turn_seq": turn.sequence,
            "products": products,
        }
        if turn.catalog_revision:
            reference_set["catalog_revision"] = turn.catalog_revision
        reference_sets.append(reference_set)

    projection.product_reference_index_json = _canonical_json(
        _newest_reference_sets_within_budget(reference_sets)
    )


def resolve_product_references(
    db,
    conversation_id: str,
    request: ProductResolutionRequest,
) -> ProductResolutionResponse:
    """Resolve typed descriptors against durable presented-product events."""

    occurrences = _product_occurrences(_presented_event_rows(db, conversation_id))
    return ProductResolutionResponse(
        results=[
            _resolve_descriptor(descriptor, occurrences)
            for descriptor in request.references
        ]
    )


#: Descriptor fields that narrow a match, in the order a reader would check them.
_DESCRIPTOR_FIELDS = (
    "product_ref",
    "display_name",
    "category",
    "turn_sequence",
    "candidate_set_id",
    "ordinal",
)


def _blocking_field(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> str | None:
    """Return the one supplied field that stopped an otherwise clear match.

    Matching is conjunctive, so a descriptor carrying five correct identifiers
    and one wrong one resolves nothing and reports not_found -- which tells the
    model only that the product is missing, when in fact it named it correctly
    five ways. Naming the field lets the model correct the call instead of
    repeating it, which is what it did for four turns.

    This only diagnoses. It never resolves to the product it found, because a
    descriptor the shopper's assistant got wrong is not authority to pick one.
    """

    supplied = [
        name
        for name in _DESCRIPTOR_FIELDS
        if getattr(descriptor, name, None) is not None
    ]
    if len(supplied) < 2:
        return None
    blocking: str | None = None
    for name in supplied:
        relaxed = descriptor.model_copy(update={name: None})
        if any(_matches_descriptor(o, relaxed) for o in occurrences):
            if blocking is not None:
                # More than one field is wrong; naming one would mislead.
                return None
            blocking = name
    return blocking


def _resolve_descriptor(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> ProductResolutionResult:
    blocking_field: str | None = None
    matches_by_ref: dict[str, ProductReferenceMatch] = {}
    for occurrence in occurrences:
        if not _matches_descriptor(occurrence, descriptor):
            continue
        product_ref = _identifier(occurrence.product["product_id"])
        matches_by_ref.pop(product_ref, None)
        matches_by_ref[product_ref] = occurrence

    matches = list(matches_by_ref.values())
    if not matches:
        status = "not_found"
        blocking_field = _blocking_field(descriptor, occurrences)
    elif len(matches) == 1:
        status = "resolved"
    else:
        status = "ambiguous"
    return ProductResolutionResult(
        reference_id=descriptor.reference_id,
        status=status,
        matches=matches[-_MAX_CLARIFICATION_MATCHES:],
        match_count=len(matches),
        blocking_field=blocking_field,
    )


def _matches_descriptor(
    match: ProductReferenceMatch,
    descriptor: ProductReferenceDescriptor,
) -> bool:
    product = match.product
    if descriptor.product_ref is not None and _identifier(
        product["product_id"]
    ) != _identifier(descriptor.product_ref):
        return False
    if descriptor.display_name is not None and _normalized(
        product["display_name"]
    ) != _normalized(descriptor.display_name):
        return False
    if descriptor.category is not None:
        category = product.get("category")
        if not isinstance(category, str) or _normalized(category) != _normalized(
            descriptor.category
        ):
            return False
    if (
        descriptor.turn_sequence is not None
        and match.turn_sequence != descriptor.turn_sequence
    ):
        return False
    if descriptor.candidate_set_id is not None and _identifier(
        match.candidate_set_id
    ) != _identifier(descriptor.candidate_set_id):
        return False
    if descriptor.ordinal is not None and match.position != descriptor.ordinal:
        return False
    return True


def _product_occurrences(rows) -> list[ProductReferenceMatch]:
    occurrences = []
    for event, turn in rows:
        for position, product in _event_products(event.payload_json):
            occurrences.append(
                ProductReferenceMatch(
                    product=product,
                    candidate_set_id=event.event_id,
                    turn_sequence=turn.sequence,
                    position=position,
                    catalog_revision=turn.catalog_revision,
                )
            )
    return occurrences


def _presented_event_rows(db, conversation_id: str):
    return (
        db.query(ConversationEvent, ConversationTurn)
        .join(ConversationTurn, ConversationTurn.turn_id == ConversationEvent.turn_id)
        .filter(
            ConversationTurn.conversation_id == conversation_id,
            ConversationEvent.event_type == "candidate_set_presented",
            ConversationEvent.event_key == PRESENTED_PRODUCTS_EVENT_KEY,
            ConversationEvent.source_kind == "runtime",
        )
        .order_by(ConversationTurn.sequence, ConversationEvent.logical_order)
        .all()
    )


def _recent_presented_event_rows(db, conversation_id: str):
    rows = (
        db.query(ConversationEvent, ConversationTurn)
        .join(ConversationTurn, ConversationTurn.turn_id == ConversationEvent.turn_id)
        .filter(
            ConversationTurn.conversation_id == conversation_id,
            ConversationEvent.event_type == "candidate_set_presented",
            ConversationEvent.event_key == PRESENTED_PRODUCTS_EVENT_KEY,
            ConversationEvent.source_kind == "runtime",
        )
        .order_by(
            ConversationTurn.sequence.desc(), ConversationEvent.logical_order.desc()
        )
        .limit(_MAX_PRESENTED_EVENT_ROWS)
        .all()
    )
    return list(reversed(rows))


def _newest_reference_sets_within_budget(
    reference_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    for reference_set in reversed(reference_sets):
        candidate = list(reversed([*selected, reference_set]))
        if len(_canonical_json(candidate)) > _MAX_REFERENCE_INDEX_CHARS:
            break
        selected.append(reference_set)
    return list(reversed(selected))


def _compact_products(payload_json: str) -> list[dict[str, Any]]:
    products = []
    for position, product in _event_products(payload_json):
        compact = {
            "ref": product["product_id"],
            "name": product["display_name"],
            "position": position,
        }
        category = product.get("category")
        if isinstance(category, str) and category.strip():
            compact["category"] = category
        products.append(compact)
    return products


def _event_products(payload_json: str) -> list[tuple[int, dict[str, Any]]]:
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return []
    raw_products = payload.get("products") if isinstance(payload, dict) else None
    if not isinstance(raw_products, list):
        return []

    products: list[tuple[int, dict[str, Any]]] = []
    for position, product in enumerate(raw_products, start=1):
        if not _is_referenceable_product(product):
            continue
        products.append((position, dict(product)))
    return products


def _is_referenceable_product(product: Any) -> bool:
    if not isinstance(product, dict):
        return False
    product_id = product.get("product_id")
    display_name = product.get("display_name")
    return (
        isinstance(product_id, str)
        and bool(product_id.strip())
        and isinstance(display_name, str)
        and bool(display_name.strip())
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


#: Characters a model may wrap an opaque identifier in. Nothing in the tool
#: schema or the rendered index shows a quoted ref, so a model that quotes one
#: is following a convention, not disobeying an instruction.
_REFERENCE_WRAPPERS = "<>[]{}\"'`"


def _identifier(value: str) -> str:
    """Compare the identifier, not the punctuation around it.

    A model sent `<generated:add69d96c548b4a3>` for a product it had displayed
    one turn earlier. The exact comparison missed, resolution returned
    not_found, and four turns in a row told the shopper the listing was
    unavailable. Stripping the wrapper compares what the index stored; it does
    not change which product is named, so nothing is guessed by doing it.
    """

    return value.strip().strip(_REFERENCE_WRAPPERS).strip()
