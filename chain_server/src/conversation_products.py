# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed boundary for products presented earlier in a conversation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shared.commerce_contracts import ProductSummary


_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_INDEX_MAX_CHARS = 12_000


class _ConversationProductModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductReferenceDescriptor(_ConversationProductModel):
    """Structured selectors for one historical product reference."""

    reference_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Short label used to identify this result.",
    )
    product_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Exact PRODUCT_REF from the historical product index.",
    )
    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Exact product name from the historical product index.",
    )
    category: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Optional. The value in square brackets in the historical product "
            "index, such as 'dresses' -- not the catalog department such as "
            "'apparel'. Omit it when sending a product_ref."
        ),
    )
    turn_sequence: int | None = Field(
        default=None,
        ge=1,
        description="Exact turn number from the historical product index.",
    )
    candidate_set_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Exact set ID from the historical product index.",
    )
    ordinal: int | None = Field(
        default=None,
        ge=1,
        description=(
            "One-based product position within the selected turn or set. This "
            "is how a positional reference resolves: 'the first one', 'the "
            "second', 'the last one you showed' send `ordinal` with the "
            "`candidate_set_id` or `turn_sequence` they were shown in, never a "
            "guessed PRODUCT_REF. The index owns the order and you do not: "
            "asked to add 'the first one' of four sandals, a guessed ref "
            "picked the fourth, the name did not match it, nothing was added, "
            "and the shopper was re-shown the same four in a different order."
        ),
    )

    @model_validator(mode="after")
    def _selectors_are_valid(self):
        selectors = (
            self.product_ref,
            self.display_name,
            self.category,
            self.turn_sequence,
            self.candidate_set_id,
        )
        if not any(value is not None for value in selectors):
            # `reference_id` is meant to be a label -- "first_sandals" -- but a
            # model asked to add the Southwest Bracelet put the product's name
            # there and nothing anywhere else. The whole call was refused, the
            # name lookup that exists for exactly this never ran, and the turn
            # died on "I could not complete that shopping request" with the
            # product's name sitting one field over.
            #
            # A label that is plainly a name is a name. Reading it as one
            # guesses nothing: it is looked up in the catalog and comes back
            # labelled as found by name rather than shown before.
            object.__setattr__(self, "display_name", self.reference_id)
            return self
        if self.ordinal is not None and (
            self.turn_sequence is None and self.candidate_set_id is None
        ):
            raise ValueError("ordinal requires turn_sequence or candidate_set_id")
        return self


class ResolveConversationProductsRequest(_ConversationProductModel):
    """One batched historical-product resolution request."""

    references: list[ProductReferenceDescriptor] = Field(
        ...,
        min_length=1,
        max_length=20,
    )


class ConversationProductMatch(_ConversationProductModel):
    """One presented product and its durable presentation coordinates."""

    product: ProductSummary
    candidate_set_id: str = Field(..., min_length=1, max_length=64)
    turn_sequence: int = Field(..., ge=1)
    position: int = Field(..., ge=1)
    catalog_revision: str | None = Field(default=None, max_length=512)


class ProductReferenceResolution(_ConversationProductModel):
    """Deterministic outcome for one reference descriptor."""

    reference_id: str = Field(..., min_length=1, max_length=128)
    status: Literal["resolved", "ambiguous", "not_found"]
    matches: list[ConversationProductMatch] = Field(default_factory=list)
    match_count: int = Field(..., ge=0)
    #: Which supplied field stopped the match, when exactly one is responsible.
    blocking_field: str | None = Field(default=None, max_length=64)
    #: Corroborating fields that disagreed with the record of the product the
    #: ref identified. The reference still resolved; this is what was odd about
    #: it, reported rather than relaxed silently.
    corroboration_mismatch: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_result(self):
        if self.status == "resolved" and (
            self.match_count != 1 or len(self.matches) != 1
        ):
            raise ValueError("resolved references require exactly one match")
        if self.status == "ambiguous" and self.match_count < 2:
            raise ValueError("ambiguous references require multiple matches")
        if self.status == "not_found" and (self.match_count or self.matches):
            raise ValueError("not_found references cannot contain matches")
        return self


class ResolveConversationProductsResult(_ConversationProductModel):
    """Batch response from conversation memory."""

    results: list[ProductReferenceResolution] = Field(..., min_length=1)


class ConversationProductsError(RuntimeError):
    """Stable failure at the historical-product service boundary."""

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


class ConversationProductsClient:
    """Resolve products from durable conversation events in one request."""

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

    def resolve(
        self,
        conversation_id: str,
        references: Sequence[ProductReferenceDescriptor],
    ) -> ResolveConversationProductsResult:
        """Resolve a nonempty descriptor batch with one memory-service call."""

        request = ResolveConversationProductsRequest(references=list(references))
        try:
            response = self.session.post(
                (
                    f"{self.memory_retriever_url}/conversations/"
                    f"{quote(conversation_id, safe='')}/products/resolve"
                ),
                json=request.model_dump(mode="json", exclude_none=True),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ConversationProductsError(
                "conversation_products_request_failed",
                "Historical product resolution request failed.",
                retryable=True,
            ) from exc

        status_code = int(getattr(response, "status_code", 500))
        if status_code >= 400:
            raise ConversationProductsError(
                "conversation_products_unavailable"
                if status_code >= 500
                else "conversation_products_request_rejected",
                "Historical product resolution was not available.",
                status_code=status_code,
                retryable=status_code >= 500,
            )
        try:
            payload = response.json()
            return ResolveConversationProductsResult.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ConversationProductsError(
                "conversation_products_response_invalid",
                "Historical product resolution returned an invalid response.",
                status_code=status_code,
            ) from exc


#: Selectors that identify a product without the model choosing which one: the
#: coordinates recorded when the products were shown -- which turn, which set,
#: which position. A product_ref is deliberately not among them. The refs are
#: printed into the prompt, so picking one out of four black dresses is the
#: model choosing and then quoting itself, which is the move this exists to
#: catch.
_SYSTEM_SELECTORS = ("ordinal", "candidate_set_id", "turn_sequence")


def _descriptor_value(descriptor: Any, name: str) -> Any:
    if isinstance(descriptor, dict):
        return descriptor.get(name)
    return getattr(descriptor, name, None)


def _identified_by_the_system(descriptor: Any) -> bool:
    if descriptor is None:
        return False
    return any(
        _descriptor_value(descriptor, name) is not None
        for name in _SYSTEM_SELECTORS
    )


class ProductEvidence:
    """Products authorized for deterministic tools in the current turn."""

    def __init__(self, products: Iterable[ProductSummary] = ()) -> None:
        self._products = {product.product_id: product for product in products}
        self._system_identified: set[str] = set()

    def add(self, products: Iterable[ProductSummary]) -> None:
        for product in products:
            self._products[product.product_id] = product

    def add_resolutions(
        self,
        resolutions: Iterable[ProductReferenceResolution],
        descriptors: Iterable[Any] = (),
    ) -> None:
        by_reference = {
            _descriptor_value(descriptor, "reference_id"): descriptor
            for descriptor in descriptors or ()
        }
        for resolution in resolutions:
            if resolution.status != "resolved" or len(resolution.matches) != 1:
                continue
            product = resolution.matches[0].product
            self.add([product])
            # How the product was identified, not just that it was. A ref the
            # system minted, or a position it recorded, is the system picking
            # the product. A display name is the model asserting one -- which is
            # right when the shopper said the name and wrong when they said
            # "the black one" and the model chose which black one they meant.
            if _identified_by_the_system(by_reference.get(resolution.reference_id)):
                self._system_identified.add(product.product_id)

    def identified_by_the_system(self, product_ref: str) -> bool:
        """Whether the record picked this product, rather than the model."""

        wanted = _reference_identifier(product_ref)
        if wanted in self._system_identified:
            return True
        return any(
            _same_reference(ref, wanted) for ref in self._system_identified
        )

    def system_identified(self) -> tuple[str, ...]:
        """Every product the record picked this turn, in a stable order.

        Read at finalization so the choice can be recorded. A shopper who says
        "add the first pairing" has chosen, by a coordinate the system itself
        wrote down -- and that choice used to be forgotten the moment the turn
        ended, leaving them to say it again, and again, until they typed the
        catalog's own names.
        """

        return tuple(sorted(self._system_identified))

    def get(self, product_ref: str) -> ProductSummary | None:
        wanted = _reference_identifier(product_ref)
        if not wanted:
            return None
        exact = self._products.get(wanted)
        if exact is not None:
            return exact
        # A model that drops the scheme is still naming what the index stored.
        # Only when it names exactly one: two matches is a real ambiguity and
        # must miss, so the shopper is asked rather than guessed at.
        matches = [
            product
            for ref, product in self._products.items()
            if _same_reference(ref, wanted)
        ]
        return matches[0] if len(matches) == 1 else None

    def values(self) -> tuple[ProductSummary, ...]:
        return tuple(self._products.values())


#: Characters a model may wrap an opaque identifier in. Mirrors the resolver's
#: own tolerance in `memory_retriever/src/product_references.py`, because a ref
#: that resolves in one lane and misses in the other is the worst of both.
_REFERENCE_WRAPPERS = "<>[]{}\"'`"


def _reference_identifier(value: str) -> str:
    """The identifier, without the punctuation a model may wrap it in."""

    return (value or "").strip().strip(_REFERENCE_WRAPPERS).strip()


def _same_reference(stored: str, given: str) -> bool:
    """Whether these name the same product, however the model wrote it.

    Asked to add a tote it had been shown one turn earlier, the model sent
    `92a114b74aaa39ea` for `generated:92a114b74aaa39ea`, was told the ref did
    not match, retried the identical value until its budget ran out, and then
    told the shopper the bag was in their cart.

    A bare identifier names what the index stored just as exactly as the
    qualified form, so accepting it guesses nothing. Two different schemes are
    two different references and still do not match.
    """

    left, right = _reference_identifier(stored), _reference_identifier(given)
    if left == right:
        return True
    for bare, qualified in ((left, right), (right, left)):
        if ":" in bare or ":" not in qualified:
            continue
        if qualified.split(":", 1)[1] == bare:
            return True
    return False


#: Fields the catalog returns alongside attributes that are not product facts:
#: marketing copy and retrieval scores must never reach the model as evidence.
_NON_ATTRIBUTE_KEYS = frozenset({"catalog_text", "similarity", "taxonomy"})


def _presented_attribute_facts(product: Any) -> dict[str, str]:
    """Attributes the catalog confirmed when this product was shown."""

    attributes = getattr(product, "attributes", None)
    if not isinstance(attributes, dict):
        return {}
    facts: dict[str, str] = {}
    for name, value in sorted(attributes.items()):
        if name in _NON_ATTRIBUTE_KEYS:
            continue
        text = str(value).strip() if not isinstance(value, (list, dict)) else ", ".join(
            str(v) for v in (value if isinstance(value, list) else value.values())
        ).strip()
        if text:
            facts[str(name)] = text
    return facts


def format_product_resolution(result: ResolveConversationProductsResult) -> str:
    """Render resolution outcomes without authorizing a guessed product."""

    lines: list[str] = []
    for resolution in result.results:
        if resolution.status == "resolved":
            match = resolution.matches[0]
            product = match.product
            # The durable presentation event stores the whole ProductSummary, so
            # these facts are already resolved and in hand. Rendering only the
            # ref and the name discarded them, and the assistant then had no way
            # to answer "which of those was cheapest" about products it had
            # itself shown -- it re-searched, and still came back without the
            # prices it had displayed a few turns earlier.
            lines.extend(
                (
                    f"REFERENCE {resolution.reference_id}: RESOLVED",
                    f"PRODUCT_REF: {product.product_id}",
                    f"NAME: {product.display_name}",
                )
            )
            if product.category:
                lines.append(f"CATEGORY: {product.category}")
            if product.price:
                # Stated as what was shown, with the turn, because a stored
                # presentation is evidence of what the shopper saw and not of
                # today's price.
                lines.append(
                    f"PRICE_WHEN_SHOWN: ${product.price.amount:.2f} "
                    f"{product.price.currency} (turn {match.turn_sequence})"
                )
            if product.image_url:
                lines.append("IMAGE_AVAILABLE: yes")
            # The presented-product event stores the whole ProductSummary, so
            # the attributes the catalog confirmed when this product was shown
            # are already in hand. Withholding them told the model to spend a
            # round trip fetching what the lane had recorded -- and the message
            # below used to say details were required for material and care,
            # which was never true of this data.
            facts = _presented_attribute_facts(product)
            if facts:
                lines.append("CONFIRMED WHEN SHOWN:")
                lines.extend(f"- {name}: {value}" for name, value in facts.items())
            lines.append(
                "These are the facts presented earlier, including the "
                "attributes the catalog confirmed at that time. Read details "
                "only for a fact not listed above. Confirm price with a fresh "
                "read before a cart action or a budget claim."
            )
            if resolution.corroboration_mismatch:
                # Resolved on the ref, so the product is not in doubt. Saying
                # which describing field disagreed is what keeps the relaxation
                # visible instead of silent -- and the values above are the
                # record, not what the call claimed.
                lines.append(
                    "NOTE: resolved by PRODUCT_REF. These fields you supplied "
                    "do not match the record and were not used: "
                    + ", ".join(resolution.corroboration_mismatch)
                    + ". Use the values above."
                )
            continue
        if resolution.status == "ambiguous":
            # The candidates, with what the catalog confirmed about each when it
            # was shown. A list of names cannot answer "the black one": four
            # dresses shown together were all black and only two said so in
            # their names, so the model guessed, reached back fourteen turns to
            # a navy dress and put it in the cart.
            #
            # The record already holds this. Handing it back is what lets the
            # model tell the candidates apart -- and it needs no new descriptor
            # field to ask with, and no copy of the catalog in the index.
            lines.append(
                f"REFERENCE {resolution.reference_id}: CLARIFICATION REQUIRED. "
                "These were shown; tell them apart on the facts below and ask "
                "the shopper which one. Do not guess."
            )
            for match in resolution.matches:
                product = match.product
                lines.append(
                    f"- {product.display_name} "
                    f"(PRODUCT_REF: {product.product_id}, turn "
                    f"{match.turn_sequence})"
                )
                for name, value in _presented_attribute_facts(product).items():
                    lines.append(f"    {name}: {value}")
            continue
        if resolution.blocking_field:
            # Naming the field is what lets the model correct the call. Reporting
            # only NOT FOUND told it the product was gone when it had in fact
            # identified it correctly by every other field, so it repeated the
            # same call and kept telling the shopper the listing was unavailable.
            lines.append(
                f"REFERENCE {resolution.reference_id}: NOT FOUND. Every other "
                f"field you supplied matched one earlier product; "
                f"'{resolution.blocking_field}' did not. Correct that field and "
                "retry, or ask which earlier product the shopper means. Do not "
                "guess."
            )
            continue
        # Nothing shown in this conversation matches -- which is a different
        # situation from an ambiguous or near-miss reference above, and used to
        # get the same answer: stop and ask. A shopper who names a product that
        # was never shown was making a search request, and the turn ended with a
        # full search budget unspent, offering to accept a product link the
        # assistant cannot read.
        #
        # Which recovery fits depends on whether the shopper named a product or
        # pointed at one, and the model is the only reader that can tell.
        lines.append(
            f"REFERENCE {resolution.reference_id}: NOT FOUND. Nothing shown in "
            "this conversation matches it. If the shopper named a product, "
            "search the catalog and show the closest matches, then ask which to "
            "add. If they pointed at an earlier item, ask which one. Never add "
            "a product the shopper has not been shown, and do not ask for a "
            "product link or a price -- neither identifies a catalog product."
        )
    return "\n".join(lines)


def format_historical_product_index(
    reference_sets: Sequence[Any],
    *,
    max_chars: int = _DEFAULT_INDEX_MAX_CHARS,
) -> str:
    """Render the compact projection as bounded read-only model context."""

    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    # Most recent first, because that is how the shopper refers to things. "The
    # black one" means the most recent black thing they were shown, not the
    # oldest -- and this list used to open with turn 1 and bury the latest
    # showing at the bottom of a long prompt. Asked for "the black one in a 2"
    # one turn after four black dresses were shown, the assistant reached back
    # fourteen turns for a navy dress and put it in the cart.
    heading = (
        "HISTORICAL PRODUCT INDEX (read-only, most recently shown first):"
    )
    formatted_sets = []
    for raw_set in reference_sets:
        line = _format_reference_set(raw_set)
        if line:
            formatted_sets.append(line)
    if not formatted_sets:
        return ""

    remaining = max_chars - len(heading) - 1
    selected_newest_first: list[str] = []
    for line in reversed(formatted_sets):
        separator = 1 if selected_newest_first else 0
        if len(line) + separator > remaining:
            break
        selected_newest_first.append(line)
        remaining -= len(line) + separator

    omitted = len(selected_newest_first) < len(formatted_sets)
    marker = "(earlier historical products omitted)"
    if omitted:
        while selected_newest_first and len(marker) + 1 > remaining:
            removed = selected_newest_first.pop()
            remaining += len(removed) + 1
    lines = [heading]
    lines.extend(selected_newest_first)
    if omitted:
        # At the end now: what was dropped is the oldest, and it belongs where
        # the oldest entries would have been.
        lines.append(marker)
    return "\n".join(lines)


def _format_reference_set(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    set_id = _one_line(value.get("candidate_set_id"))
    shopper_size = _one_line(value.get("shopper_size"))
    turn = value.get("turn_seq")
    products = value.get("products")
    if not set_id or not isinstance(turn, int) or not isinstance(products, list):
        return ""
    rendered = []
    for product in products:
        if not isinstance(product, dict):
            continue
        ref = _one_line(product.get("ref"))
        name = _one_line(product.get("name"))
        position = product.get("position")
        if not ref or not name or not isinstance(position, int):
            continue
        category = _one_line(product.get("category"))
        rendered.append(
            f"{position}:{name} [{category}] <{ref}>"
            if category
            else f"{position}:{name} <{ref}>"
        )
    if not rendered:
        return ""
    # The showing remembers the size it was made under. "Show me sandals in a
    # 7" then "add the first one" asked which size, with the answer one turn
    # back. It qualifies this set and nothing else: a later showing carries its
    # own size or none, so no size follows the shopper from one to the next.
    qualifier = (
        f" [shopper asked for size {shopper_size}]" if shopper_size else ""
    )
    return f"- set={set_id} turn={turn}{qualifier}: " + "; ".join(rendered)


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())[:256]
