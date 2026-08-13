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
    #: Corroborating fields that disagreed with the record of the product the
    #: ref identified. Reported so nothing is relaxed silently.
    corroboration_mismatch: list[str] = Field(default_factory=list)


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


#: Fields that describe where a product was seen rather than which product it
#: is. They stay part of the descriptor and are still checked -- but once an
#: exact ``product_ref`` has identified a product, they can no longer overrule
#: it, because a mismatch here is a vocabulary difference and not a different
#: product.
#:
#: The assistant asked for a dress by ref, name, set, turn and position, all
#: five correct, and added ``category: "apparel"`` -- the catalog's word for the
#: department, where the index stores the subcategory ``"dresses"``. Matching
#: was conjunctive, so the sixth field cancelled the other five and the answer
#: came back not_found. Being more specific was what broke it.
_CORROBORATING_FIELDS = (
    "category",
    "turn_sequence",
    "candidate_set_id",
    "ordinal",
)


def _matched_occurrences(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> list[ProductReferenceMatch]:
    """Collapse matching occurrences to one per product, newest kept."""

    matches_by_ref: dict[str, ProductReferenceMatch] = {}
    for occurrence in occurrences:
        if not _matches_descriptor(occurrence, descriptor):
            continue
        product_ref = _identifier(occurrence.product["product_id"])
        matches_by_ref.pop(product_ref, None)
        matches_by_ref[product_ref] = occurrence
    return list(matches_by_ref.values())


def _corroboration_mismatch(
    descriptor: ProductReferenceDescriptor,
    match: ProductReferenceMatch,
) -> list[str]:
    """Name the supplied corroborating fields that disagree with the record."""

    mismatched = []
    if descriptor.category is not None and _normalized(
        str(match.product.get("category") or "")
    ) != _normalized(descriptor.category):
        mismatched.append("category")
    if (
        descriptor.turn_sequence is not None
        and match.turn_sequence != descriptor.turn_sequence
    ):
        mismatched.append("turn_sequence")
    if descriptor.candidate_set_id is not None and _identifier(
        match.candidate_set_id
    ) != _identifier(descriptor.candidate_set_id):
        mismatched.append("candidate_set_id")
    if descriptor.ordinal is not None and match.position != descriptor.ordinal:
        mismatched.append("ordinal")
    return mismatched


def _resolve_descriptor(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> ProductResolutionResult:
    blocking_field: str | None = None
    corroboration_mismatch: list[str] = []
    matches = _matched_occurrences(descriptor, occurrences)

    if not matches and descriptor.product_ref is not None:
        # Strictly a second chance: a descriptor that resolves today resolves
        # identically above, and only one that resolves nothing gets here. The
        # ref is an identifier this system minted and printed itself, so a ref
        # that matches has identified the product; display_name is deliberately
        # not relaxed, because a name that contradicts the ref is the confused
        # assistant this gate was built to catch.
        relaxed = descriptor.model_copy(
            update={name: None for name in _CORROBORATING_FIELDS}
        )
        candidates = [
            occurrence
            for occurrence in occurrences
            if _matches_descriptor(occurrence, relaxed)
        ]
        if candidates:
            # A product shown more than once has an occurrence per showing, and
            # they differ in exactly the fields just relaxed. Take the one the
            # descriptor describes best, so the facts reported back belong to
            # the showing the assistant referred to -- and so the mismatch names
            # only what was really wrong. Taking the newest instead reported
            # four wrong fields when one was.
            best = min(
                reversed(candidates),
                key=lambda occurrence: len(
                    _corroboration_mismatch(descriptor, occurrence)
                ),
            )
            matches = [best]
            corroboration_mismatch = _corroboration_mismatch(descriptor, best)

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
        corroboration_mismatch=corroboration_mismatch,
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
        # The one attribute a reference is usually made of. "The black one"
        # cannot be resolved against a list of names: two of four dresses shown
        # together had "Black" in the name and all four were black, so the model
        # guessed by name and put a navy dress in the cart. Colour is recorded
        # when the product is shown, so carrying it costs nothing and is the
        # difference between resolving the reference and inventing one.
        attributes = product.get("attributes")
        colour = (
            attributes.get("primary_color")
            if isinstance(attributes, dict)
            else None
        )
        if isinstance(colour, str) and colour.strip():
            compact["color"] = colour
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
