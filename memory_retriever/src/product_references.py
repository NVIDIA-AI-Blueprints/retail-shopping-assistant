# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presented-product events and their compact conversation projection."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func

from .models import ConversationEvent, ConversationProjection, ConversationTurn

PRESENTED_PRODUCTS_EVENT_KEY = "runtime-presented-products"
#: Where the product sat on the screen, kept beside it rather than recounted.
_SCREEN_POSITION_KEY = "screen_position"
#: The product itself, inside a recorded entry. Anything the record knows about
#: how a turn presented a product is a sibling of this key, never a field
#: within it: the runtime's product contract admits product fields only, and
#: refuses -- for the whole object -- anything it does not recognise.
_STORED_PRODUCT_KEY = "product"
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
    #: The heading the shopper counted under: "the second shoes" is
    #: ordinal 2 in group "shoes". An ordinal is numbered from one inside each
    #: group, so without this it names one product per group rather than one
    #: product.
    group: str | None = Field(default=None, min_length=1, max_length=256)
    #: What the shopper described rather than named, in advertised attribute
    #: terms: "the black one" is {"primary_color": "black"}. Compared against
    #: the attributes the catalog confirmed when the product was shown, so the
    #: reading stays the model's and the matching stays exact.
    attributes: dict[str, str] | None = Field(default=None, max_length=12)

    @model_validator(mode="after")
    def _validate_selectors(self):
        selectors = (
            self.product_ref,
            self.display_name,
            self.category,
            self.turn_sequence,
            self.candidate_set_id,
            self.ordinal,
            self.group,
            self.attributes,
        )
        if not any(selector is not None for selector in selectors):
            raise ValueError("At least one product reference selector is required")
        # An ordinal used to require a turn or a candidate set, which the
        # shopper saying "the second one" does not supply and the model can
        # only guess at. Alone it counts within the most recent showing, which
        # is the one they are looking at.
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
    #: Where this product sat under its heading. Numbered from one inside each
    #: group, so it identifies a product only together with the group.
    position: int
    #: The heading it was shown under, empty where the showing had none.
    group: str = Field(default="", max_length=256)
    #: Which group, in the order they were shown. Carried because a bare
    #: ordinal with nothing to narrow it means the first group on screen.
    group_index: int = 0
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


#: Attribute keys carried out of retrieval that no reader wants back.
#: ``catalog_text`` is the prose serialisation of the very attributes stored
#: beside it -- half of every event, and every consumer filters it out again on
#: the way to the model. ``similarity`` is the retrieval score for the search
#: that produced the row, which means nothing once the turn is over.
_UNSTORED_ATTRIBUTE_KEYS = frozenset({"catalog_text", "similarity"})


def _persistable(product: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one product for the record, without the parts nothing reads."""

    stored = dict(product)
    attributes = stored.get("attributes")
    if isinstance(attributes, dict):
        stored["attributes"] = {
            name: value
            for name, value in attributes.items()
            if name not in _UNSTORED_ATTRIBUTE_KEYS
        }
    return stored


def append_presented_products_event(
    db,
    turn: ConversationTurn,
    product_results: list[dict[str, Any]],
    *,
    product_groups: list[dict[str, Any]] | None = None,
    created_at: float,
) -> ConversationEvent | None:
    """Append one event for the groups of products returned to the shopper.

    A showing is a list of headed groups, because that is what the shopper
    reads: dresses, then shoes, counted from one inside each. Stored as a flat
    queue it was eight products in a row, and "the second shoes" was a question
    with no structural answer -- only a guess from category strings, which two
    scopes out of one category defeat.

    The place each product holds under its heading is recorded beside it,
    counted over everything that group presented and not over what survives
    this filter. The number the shopper reads is stamped on the streamed list,
    which is not filtered, so counting the kept ones would shift every position
    after a dropped product: they would say "the fifth" and be handed the
    sixth. Numbering first leaves a gap instead, and a gap resolves to nothing
    rather than to the wrong garment.

    Beside it, and not on it, because a product record holds product facts. The
    position was previously stamped into the product itself and popped back off
    by name when read, which works only while every annotation is remembered in
    both places. One was not: a second annotation was added, the read path did
    not know to remove it, and the product came back out of the record carrying
    a field the runtime's product contract forbids. The runtime refused the
    whole object, the refusal was read as "no such product", and a shopper
    asking for a bag they had been shown four turns earlier was told the
    reference was not valid. Nesting makes that leak unrepresentable rather
    than remembered.
    """

    groups = _groups_as_shown(product_results, product_groups or [])
    if not groups:
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
        payload_json=_canonical_json({"groups": groups}),
        created_at=created_at,
    )
    db.add(event)
    return event


def _groups_as_shown(
    product_results: list[dict[str, Any]],
    product_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The showing, as headed groups of numbered products.

    The groups name their products by id; the products themselves come from
    the turn's ordered results. Anything no group claims is one unheaded group
    at the end, which is what a name lookup produces and what a turn with no
    groups at all becomes.
    """

    held = {
        str(product.get("product_id") or ""): product
        for product in product_results
        if isinstance(product, Mapping) and str(product.get("product_id") or "")
    }
    groups: list[dict[str, Any]] = []
    placed: set[str] = set()
    for group in product_groups:
        if not isinstance(group, Mapping):
            continue
        members = []
        for product_id in group.get("product_ids") or []:
            product = held.get(str(product_id))
            if product is not None and str(product_id) not in placed:
                members.append(product)
                placed.add(str(product_id))
        recorded = _numbered_under_a_heading(members)
        if recorded:
            groups.append(
                {
                    "heading": str(group.get("heading") or ""),
                    "products": recorded,
                }
            )
    unclaimed = _numbered_under_a_heading(
        [
            product
            for product in product_results
            if not isinstance(product, Mapping)
            or str(product.get("product_id") or "") not in placed
        ]
    )
    if unclaimed:
        groups.append({"heading": "", "products": unclaimed})
    return groups


def _numbered_under_a_heading(
    products: list[Any],
) -> list[dict[str, Any]]:
    """One group's products, each with the place it held under its heading."""

    recorded = []
    for position, product in enumerate(products, start=1):
        if not _is_referenceable_product(product):
            continue
        recorded.append(
            {_STORED_PRODUCT_KEY: _persistable(product), _SCREEN_POSITION_KEY: position}
        )
    return recorded


def _system_identifications_by_turn(db, conversation_id: str) -> list[tuple[int, list[str]]]:
    """Which products the record itself picked, and on which turn.

    A shopper who says "the first pairing" has chosen by a coordinate this
    service wrote down. The choice is durable for the same reason the showing
    is.
    """

    rows = (
        db.query(ConversationEvent, ConversationTurn)
        .join(ConversationTurn, ConversationTurn.turn_id == ConversationEvent.turn_id)
        .filter(
            ConversationTurn.conversation_id == conversation_id,
            ConversationEvent.event_type == "historical_reference_resolved",
            ConversationEvent.source_kind == "runtime",
        )
        .order_by(ConversationTurn.sequence, ConversationEvent.logical_order)
        .all()
    )
    identifications: list[tuple[int, list[str]]] = []
    for event, turn in rows:
        try:
            payload = json.loads(event.payload_json or "{}")
        except (TypeError, ValueError):
            continue
        refs = [
            str(ref)
            for ref in (payload.get("product_refs") or [])
            if isinstance(ref, (str, int))
        ]
        if refs:
            identifications.append((turn.sequence, refs))
    return identifications


def _shopper_sizes_of(turn) -> list[str]:
    """The sizes this turn's searches were filtered by, if exactly recorded.

    Read from the turn's own diagnostics, which already cross the service
    boundary and are already stored. Only one size qualifies a showing: two
    means the shopper was comparing, and neither is the size they want.
    """

    try:
        output = json.loads(turn.output_json or "{}")
    except (TypeError, ValueError):
        return []
    diagnostics = output.get("agent_diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    sizes = diagnostics.get("shopper_sizes")
    if not isinstance(sizes, list):
        return []
    return [str(value).strip() for value in sizes if str(value).strip()]


def rebuild_product_reference_index(
    db,
    projection: ConversationProjection,
) -> None:
    """Rebuild the compact index from durable presented-product events."""

    rows = _recent_presented_event_rows(db, projection.conversation_id)
    identifications = _system_identifications_by_turn(db, projection.conversation_id)
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
        # A choice belongs to the set it was made from, so it is filed against
        # the newest showing at or before the turn that made it. That is also
        # what retires it: once a newer set is shown, this set is no longer the
        # one in front of the shopper and its choices no longer answer for what
        # is.
        shown = {str(item.get("ref")) for item in products if isinstance(item, dict)}
        picked = [
            ref
            for sequence, refs in identifications
            if sequence >= turn.sequence
            for ref in refs
            if ref in shown
        ]
        if picked:
            reference_set["system_identified"] = sorted(dict.fromkeys(picked))
        # A showing made under a size filter is size-qualified: those four
        # sandals came back because they come in a 7. The size belongs to this
        # set and to nothing else -- a later showing carries its own or none,
        # and they never merge, so no size follows the shopper around.
        sizes = _shopper_sizes_of(turn)
        if len(sizes) == 1:
            reference_set["shopper_size"] = sizes[0]
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


def _identified(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> tuple[list[ProductReferenceMatch], bool]:
    """Occurrences the ref or the name points at, and whether they contradict.

    These two identify; nothing else in a descriptor does. A ``product_ref`` is
    an identifier this system minted and printed into the prompt, and a
    display_name is the catalog's own. Neither is the model's reading of what
    the shopper wanted, which is what makes them worth trusting over every
    other field.

    Supplied together they are a checksum on each other, and disagreement is
    reported rather than resolved. Reading the ref off the wrong line of a
    numbered index is an ordinary mistake, and letting it win silently puts a
    dress in the cart that the shopper named and did not ask for.
    """

    ref_hits = name_hits = None
    if descriptor.product_ref is not None:
        ref_hits = [
            occurrence
            for occurrence in occurrences
            if _same_reference(
                occurrence.product["product_id"], descriptor.product_ref
            )
        ]
    if descriptor.display_name is not None:
        name_hits = [
            occurrence
            for occurrence in occurrences
            if _normalized(occurrence.product["display_name"])
            == _normalized(descriptor.display_name)
        ]
    if ref_hits is not None and name_hits is not None:
        # Offered both, they have to agree, and one of them finding nothing is
        # a disagreement like any other: a ref that matches nothing is as much
        # a confused caller as a ref that matches the wrong thing. Whichever
        # way round it is, there is no rule for choosing between two
        # identifiers neither of which the model authored -- only a caller to
        # correct, and _blocking_field says which half to correct.
        named = {
            _identifier(occurrence.product["product_id"])
            for occurrence in name_hits
        }
        agreed = [
            occurrence
            for occurrence in ref_hits
            if _identifier(occurrence.product["product_id"]) in named
        ]
        return agreed, not agreed

    hits = ref_hits if ref_hits is not None else name_hits
    if hits is None:
        # Nothing here identifies. Whatever describes the product does.
        return [], False
    if hits:
        return hits, False
    if descriptor.display_name is not None and descriptor.attributes:
        # A phrase can be read as a name or as a description, and the model
        # sends both readings of the same one: display_name "black one" beside
        # {"primary_color": "black"}. Nothing is called "black one", so the
        # reading that can match is the one meant, and the name steps aside.
        return [], False
    # An identifier was offered and matched nothing shown. With no description
    # behind it there is nothing else to go on, and offering every product in
    # the conversation instead would answer "the Missing Bag" with a menu.
    return [], True


def _narrowed_without_emptying(
    descriptor: ProductReferenceDescriptor,
    pool: list[ProductReferenceMatch],
    *,
    identified: bool,
) -> tuple[list[ProductReferenceMatch], list[str]]:
    """Apply every remaining field, in order, but never down to nothing.

    These fields describe: where the product was seen, what it is called a
    department at a time, what colour the model read into the request. They
    are worth using to choose between several candidates and are not worth
    losing a candidate over, because each is the model's account of the
    shopper rather than the shopper's own words.

    Conjunction made them vetoes. Asked for the first sweater it had shown one
    turn earlier, the assistant sent the ref, the name, the set, the turn and
    the position -- all five right -- and added a size run it invented. The
    sixth field cancelled the other five, the answer came back not_found, and
    the turn searched the catalog by name for the product whose ref it was
    already holding. Being more specific was what broke it.

    A field that would empty the pool is reported instead, so the reply is
    told which of its own values the record does not share.
    """

    ignored: list[str] = []

    def narrow(field: str, keep: Callable[[ProductReferenceMatch], bool]) -> None:
        nonlocal pool
        narrowed = [occurrence for occurrence in pool if keep(occurrence)]
        if narrowed or not soft:
            pool = narrowed
        else:
            ignored.append(field)

    # Where a product was seen is this system's own bookkeeping, so while
    # nothing has identified the product it is doing the identifying and a miss
    # is a real miss: "the third one" over a showing of two is not the showing
    # of two. Once a ref or a name has named the product, the same fields are
    # only corroborating it, and a wrong one must not cancel it.
    soft = identified

    if descriptor.turn_sequence is not None:
        narrow(
            "turn_sequence",
            lambda o: o.turn_sequence == descriptor.turn_sequence,
        )
    if descriptor.candidate_set_id is not None:
        narrow(
            "candidate_set_id",
            lambda o: _identifier(o.candidate_set_id)
            == _identifier(descriptor.candidate_set_id),
        )
    if descriptor.ordinal is not None:
        # "The second one" counts within the showing the shopper is looking at,
        # and restarts under each heading inside it. Unscoped it counted across
        # every showing at once, so a conversation with four of them offered
        # four second ones. Occurrences arrive oldest first.
        if (
            descriptor.turn_sequence is None
            and descriptor.candidate_set_id is None
            and pool
        ):
            newest = _identifier(pool[-1].candidate_set_id)
            pool = [
                occurrence
                for occurrence in pool
                if _identifier(occurrence.candidate_set_id) == newest
            ]
        if pool:
            pool = _the_group_the_ordinal_counts_in(descriptor, pool)
        narrow("ordinal", lambda o: o.position == descriptor.ordinal)
    # From here the fields describe rather than identify: a department name and
    # the colours and sizes the model read into the request. Those are worth
    # choosing between candidates with, and never worth losing one over.
    soft = True
    if descriptor.category is not None:
        narrow(
            "category",
            lambda o: isinstance(o.product.get("category"), str)
            and _normalized(o.product["category"])
            == _normalized(descriptor.category),
        )
    # Named one at a time, so the reply learns that sizes disagreed rather than
    # that "attributes" did. The record's own values are printed with the
    # match, so naming the field points straight at the correction.
    for name, value in (descriptor.attributes or {}).items():
        narrow(
            f"attributes.{name}",
            lambda o, n=name, v=value: _attributes_agree(o.product, {n: v}),
        )

    return pool, ignored


def _one_per_product(
    occurrences: list[ProductReferenceMatch],
) -> list[ProductReferenceMatch]:
    """Collapse occurrences to one per product, keeping the newest showing."""

    matches_by_ref: dict[str, ProductReferenceMatch] = {}
    for occurrence in occurrences:
        product_ref = _identifier(occurrence.product["product_id"])
        matches_by_ref.pop(product_ref, None)
        matches_by_ref[product_ref] = occurrence
    return list(matches_by_ref.values())


def _matched_occurrences(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> list[ProductReferenceMatch]:
    """Which products this descriptor refers to, one entry each."""

    matches, _ignored, _refused = _resolution_pool(descriptor, occurrences)
    return matches


def _resolution_pool(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> tuple[list[ProductReferenceMatch], list[str], bool]:
    """Resolve a reference: identify first, then narrow, never to nothing."""

    identified, refused = _identified(descriptor, occurrences)
    if refused:
        return [], [], True
    pool, ignored = _narrowed_without_emptying(
        descriptor,
        identified if identified else list(occurrences),
        identified=bool(identified),
    )
    return _one_per_product(pool), ignored, False


def _the_group_the_ordinal_counts_in(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> list[ProductReferenceMatch]:
    """Narrow one showing to the group the shopper is counting inside.

    A number restarts under each heading, so "the first one" over a showing of
    dresses and shoes names two products, and the reference that could not be
    clearer came back as a question to ask.

    Named, the group decides it: "the second shoes" is the shoes. Unnamed, the
    first group on screen does, which is the one the reply anchors on and the
    only group there is when the shopper asked for one kind. Assume and
    disclose: guessing which of four dresses they meant is recoverable in
    three words, and stopping to ask is not free.
    """

    if descriptor.group is not None:
        wanted = _normalized(descriptor.group)
        named = [
            occurrence
            for occurrence in occurrences
            if _normalized(occurrence.group) == wanted
        ]
        # A heading this showing does not know falls through to the default
        # below rather than narrowing to nothing. The shopper's word for a
        # group is not always the word the search was asked for -- "frocks"
        # over a group headed "dresses" -- and leaving it unnarrowed would
        # answer a clearer reference with a question than a vaguer one.
        if named:
            return named
    groups = {occurrence.group_index for occurrence in occurrences}
    if len(groups) <= 1:
        return occurrences
    first = min(groups)
    return [
        occurrence for occurrence in occurrences if occurrence.group_index == first
    ]


def _resolve_descriptor(
    descriptor: ProductReferenceDescriptor,
    occurrences: list[ProductReferenceMatch],
) -> ProductResolutionResult:
    blocking_field: str | None = None
    matches, corroboration_mismatch, refused = _resolution_pool(
        descriptor, occurrences
    )

    if refused:
        return ProductResolutionResult(
            reference_id=descriptor.reference_id,
            status="not_found",
            matches=[],
            match_count=0,
            blocking_field=_blocking_field(descriptor, occurrences),
            corroboration_mismatch=[],
        )

    if len(matches) > 1 and descriptor.turn_sequence is None and (
        descriptor.candidate_set_id is None
    ):
        # Several still fit and the shopper never said which showing, so they
        # mean the one in front of them: "the black one" a turn after a black
        # dress was shown is not the black dress from nine turns before. Within
        # one showing recency says nothing -- four black dresses on a screen are
        # equally recent -- and those stay a question worth asking.
        newest = _identifier(
            max(matches, key=lambda match: match.turn_sequence).candidate_set_id
        )
        matches = [
            match
            for match in matches
            if _identifier(match.candidate_set_id) == newest
        ]

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
    if descriptor.product_ref is not None and not _same_reference(
        product["product_id"], descriptor.product_ref
    ):
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
    if descriptor.attributes:
        return _attributes_agree(match.product, descriptor.attributes)
    return True


def _attributes_agree(product: Any, wanted: dict[str, str]) -> bool:
    """Whether a shown product carries every attribute the shopper described.

    "Add the black one in a 2" could be asked of this index and never answered
    from it: a description is not a PRODUCT_REF and not a product name, so the
    two comparisons on offer both missed, the reference came back NOT FOUND,
    and a catalog lookup went off and ranked products by how much their names
    resembled the string "black one".

    Every fact needed was already recorded. The attributes the catalog
    confirmed when the product was shown are stored with the showing, so
    matching them is a comparison against this system's own record. The model
    reads "black" into primary_color; nothing here reads anything.

    A stored list holds each value the product offers -- sizes 2, 4, 6 -- so
    the shopper's size matches by membership. Anything else matches whole.
    """

    recorded = getattr(product, "attributes", None)
    if recorded is None and isinstance(product, dict):
        recorded = product.get("attributes")
    if not isinstance(recorded, dict):
        return False
    for name, value in wanted.items():
        if name not in recorded:
            return False
        held = recorded[name]
        if isinstance(held, (list, tuple, set)):
            if not any(_normalized(str(item)) == _normalized(value) for item in held):
                return False
        elif _normalized(str(held)) != _normalized(value):
            return False
    return True


def _product_occurrences(rows) -> list[ProductReferenceMatch]:
    occurrences = []
    for event, turn in rows:
        for group_index, (heading, products) in enumerate(
            _event_groups(event.payload_json)
        ):
            for position, product in products:
                occurrences.append(
                    ProductReferenceMatch(
                        product=product,
                        candidate_set_id=event.event_id,
                        turn_sequence=turn.sequence,
                        position=position,
                        group=heading,
                        group_index=group_index,
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
    """Every product of one showing, each naming its group and its number.

    The number restarts under each heading, so a position without its group
    names one product per group. Flattened without the heading, a showing of
    dresses and shoes offered two first ones and "the first one" resolved to
    neither.
    """

    products = []
    for heading, numbered in _event_groups(payload_json):
        for position, product in numbered:
            compact = {
                "ref": product["product_id"],
                "name": product["display_name"],
                "position": position,
            }
            if heading:
                compact["group"] = heading
            category = product.get("category")
            if isinstance(category, str) and category.strip():
                compact["category"] = category
            # The sizes this product is sold in, so a later turn can tell which
            # of the things on screen the shopper's "in a 2" could even mean.
            # Whether a product comes in a 2 is a catalog fact; which one they
            # meant is not, and only the first belongs in this record.
            sizes = (product.get("attributes") or {}).get("sizes")
            if isinstance(sizes, list):
                kept = [str(value).strip() for value in sizes if str(value).strip()]
                if kept:
                    compact["sizes"] = kept
            products.append(compact)
    return products


def _event_groups(payload_json: str) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    """Each group the shopper was shown, with its heading and its products.

    Two shapes of payload, because conversations already written keep theirs.
    Groups are what is written now. A flat `products` list is what came before,
    and it reads back as a single unheaded group -- which is honest: the turn
    did show those products in that order, and nothing recorded where one kind
    ended and the next began.
    """

    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    raw_groups = payload.get("groups")
    if isinstance(raw_groups, list):
        groups = []
        for group in raw_groups:
            if not isinstance(group, dict):
                continue
            products = _entry_products(group.get("products"))
            if products:
                groups.append((str(group.get("heading") or ""), products))
        return groups
    products = _entry_products(payload.get("products"))
    return [("", products)] if products else []


def _entry_products(raw_products: Any) -> list[tuple[int, dict[str, Any]]]:
    """One recorded list of products, each with the place it was shown in.

    Three shapes of entry, because conversations already written keep theirs.
    The product nested under its own key is what is written now. Before that it
    was the product itself with the position stamped into it, and before that
    the product alone. The two older shapes are read by taking the entry as the
    product and lifting the position off it, which is also what keeps a product
    recorded the old way from coming back out with a stray field on it.
    """

    if not isinstance(raw_products, list):
        return []

    products: list[tuple[int, dict[str, Any]]] = []
    for counted, entry in enumerate(raw_products, start=1):
        if not isinstance(entry, dict):
            continue
        nested = entry.get(_STORED_PRODUCT_KEY)
        if isinstance(nested, dict):
            product, position = dict(nested), entry.get(_SCREEN_POSITION_KEY)
        else:
            product = dict(entry)
            position = product.pop(_SCREEN_POSITION_KEY, None)
        if not _is_referenceable_product(product):
            continue
        # Recorded before the list was filtered, so it is the number the shopper
        # was shown. Counting here is the fallback for a conversation written
        # before the number was kept at all.
        products.append(
            (position if isinstance(position, int) and position > 0 else counted, product)
        )
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


def _same_reference(stored: str, given: str) -> bool:
    """Whether these name the same product, however the model wrote it.

    The wrapper tolerance above exists because a model sent
    `<generated:add69d96c548b4a3>`. The same model drops the other half: asked
    to add a tote it had been shown one turn earlier it sent
    `92a114b74aaa39ea` for `generated:92a114b74aaa39ea`, was told the ref did
    not match, retried the identical value until its budget ran out, and then
    told the shopper the bag was in their cart.

    A bare identifier names what the index stored just as exactly as the
    qualified form does, so accepting it guesses nothing. Two *different*
    schemes are two different references and still do not match.
    """

    left, right = _identifier(stored), _identifier(given)
    if left == right:
        return True
    for bare, qualified in ((left, right), (right, left)):
        if ":" in bare or ":" not in qualified:
            continue
        if qualified.split(":", 1)[1] == bare:
            return True
    return False
