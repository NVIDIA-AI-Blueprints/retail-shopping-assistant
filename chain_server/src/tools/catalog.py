# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that search the catalog and read what it holds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from shared.commerce_contracts import CatalogCapabilities, GetProductDetailsInput

from ..agenttypes import State
from ..catalog_capabilities import format_catalog_capabilities_for_prompt
from ..catalog_execution import execute_catalog_search
from ..catalog_request import CatalogSearchPlan
from ..catalog_scope import CATALOG_SEARCH_RULES
from ..catalog_search import SearchContext, search_catalog
from ..commerce_tools import get_product_details
from ..control_signals import ControlSignal, control, normalize_tool_result
from ..conversation_products import (
    ConversationProductsError,
    ProductReferenceDescriptor,
    ResolveConversationProductsRequest,
    format_historical_product_index,
    format_product_resolution,
)
from ..response_format import _format_product_detail_record, format_catalog_shape
from ..turn_scope import TurnScope
from ..turn_support import (
    _ONE_SIZE,
    RequestIdentity,
    _advertised_sizes,
    _append_product_results,
    _detail_fields_already_held,
    _product_detail_failure_message,
    _product_detail_record,
    _same_product_display_name,
    _where_a_product_was_already_shown,
)
from .evidence import ProductDetailEvidence
from .schemas import (
    _search_catalog_scopes_input_model,
    _search_catalog_tool_input_model,
)

if TYPE_CHECKING:
    from ..deepagents_runtime import DeepAgentsRuntime


#: Historical-product resolutions allowed per turn while none has resolved. A
#: resolution that succeeds ends the budget immediately; this only bounds the
#: corrections a failing one may attempt.
_MAX_PRODUCT_RESOLUTION_ATTEMPTS = 2


#: Characters a model may wrap an opaque identifier in, matching the resolver's
#: own tolerance so both lanes read a ref the same way.
_REFERENCE_WRAPPERS = "<>[]{}\"'`"


_MAX_NAME_LOOKUPS = 2


class _DescribeCatalogInput(BaseModel):
    """No arguments. The shape is published; there is nothing to narrow."""


def catalog_prompt_section(capabilities: CatalogCapabilities) -> str:
    """The catalog's schema and the rules for searching it, as one block.

    Held out of the static prompt and handed to the skill gate instead, so
    it reaches only a model request that was granted `search_catalog_tool`.
    The activation step is granted nothing and paid for this every call; a
    cart read and a policy question paid too, for a search they cannot run.

    The two travel together because the rules are only true beside the
    capabilities: they say a filter value comes from the enum above them.
    """

    return (
        "Catalog capabilities:\n"
        f"{format_catalog_capabilities_for_prompt(capabilities)}\n"
        f"{CATALOG_SEARCH_RULES}"
    )


def build_catalog_tools(
    runtime: DeepAgentsRuntime,
    state: State,
    identity: RequestIdentity,
    scope: TurnScope,
    turn_capabilities: CatalogCapabilities,
):
    """Search, product-detail, earlier-product and catalog-shape tools,
    validated against this turn's `turn_capabilities`."""

    from langchain_core.tools import tool

    wearer_audience_field = str(
        getattr(runtime.config, "wearer_audience_field", "") or ""
    )
    search_input_model = _search_catalog_tool_input_model(
        turn_capabilities,
        wearer_audience_field=wearer_audience_field,
    )
    search_tool_arguments_model = _search_catalog_scopes_input_model(
        turn_capabilities,
        max_scopes=max(
            1, int(getattr(runtime.config, "max_search_scopes_per_call", 1) or 1)
        ),
        wearer_audience_field=wearer_audience_field,
    )
    constraint_input_model = search_input_model.model_fields[
        "required_constraints"
    ].annotation

    search_context = SearchContext(
        config=runtime.config,
        state=state,
        scope=scope,
        capabilities=turn_capabilities,
        search_input_model=search_input_model,
        constraint_input_model=constraint_input_model,
        vocabulary_judge=runtime._vocabulary_judge,
    )

    def _search_catalog_impl(scopes, not_covered=None):
        """Execute one catalog search per product role; may return signals."""

        return search_catalog(search_context, scopes, not_covered=not_covered)

    @tool(
        args_schema=search_tool_arguments_model,
        return_direct=False,
        response_format="content_and_artifact",
    )
    def search_catalog_tool(scopes, not_covered=None):
        """Find products by description, advertised taxonomy, or constraints.

        Use for browse, search, and recommendation requests after product
        discovery or outfit styling is active. Select exact values from the
        current Catalog capabilities. Do not use for a product already
        established in this conversation, and do not repeat a completed hard-
        filter scope with different semantic wording.
        """

        return normalize_tool_result(
            _search_catalog_impl(scopes, not_covered)
        )

    def _get_product_details_impl(product_ref: str):
        """Get detailed facts (material, care, dimensions, closures) for a
        product established in this turn by search or historical-product
        resolution. Requires a PRODUCT_REF — not a product name. Do NOT call
        for initial recommendations. Stop immediately if STOP_TOOL_USE is
        returned.
        """

        held = scope.answer_already_given(
            "get_product_details_tool",
            product_ref,
        )
        if held is not None:
            return held
        if (
            scope.product_detail_reads
            >= runtime.config.max_product_detail_reads_per_turn
        ):
            return control(
                "STOP_TOOL_USE: Product-detail read limit reached for this "
                "turn. Do not call more tools this turn. Answer now from the "
                "details already read and keep any other products to names, "
                "prices, categories, image availability, and styling role.",
                ControlSignal.STOP_TOOL_USE,
            )
        cached_product = scope.product_evidence.get(product_ref)
        if cached_product is None:
            return (
                f"No product with PRODUCT_REF '{product_ref}' is available. "
                "Search this turn or resolve the earlier product first."
            )
        if _detail_fields_already_held(cached_product, turn_capabilities):
            # The search that produced this product already returned every
            # detail field its category advertises. Answering from that
            # evidence is not a guess; it is the same data, without an ~8.7s
            # round trip. A product recovered from the historical index
            # carries identity only, so it fails this check and still reads.
            record = _product_detail_record(cached_product)
            evidence = ProductDetailEvidence(products=[record])
            answer = _format_product_detail_record(record)
            scope.remember_answer(
                "get_product_details_tool",
                product_ref,
                answer,
            )
            return (answer, evidence.as_artifact())
        scope.product_detail_reads += 1
        detail_result = get_product_details(
            GetProductDetailsInput(product_id=cached_product.product_id),
            runtime.config.retriever_port,
            timeout_seconds=runtime.config.catalog_search_timeout_seconds,
        )
        if not detail_result.ok or detail_result.product is None:
            return _product_detail_failure_message(
                detail_result.error,
                cart_validation=False,
            )
        product = detail_result.product
        if not _same_product_display_name(
            product.display_name,
            cached_product.display_name,
        ):
            return (
                "That product reference now resolves to a different item. "
                "Search the catalog again before using its details."
            )
        if product.image_url:
            scope.retrieved[product.display_name] = product.image_url
        record = _product_detail_record(product)
        evidence = ProductDetailEvidence(products=[record])
        answer = _format_product_detail_record(record)
        scope.remember_answer(
            "get_product_details_tool",
            product_ref,
            answer,
        )
        return (answer, evidence.as_artifact())

    @tool(return_direct=False, response_format="content_and_artifact")
    def get_product_details_tool(product_ref: str):
        """Get detailed facts (material, care, dimensions, closures) for a
        product established in this turn by search or historical-product
        resolution. Requires a PRODUCT_REF — not a product name. Do NOT call
        for initial recommendations. Stop immediately if STOP_TOOL_USE is
        returned.
        """

        return normalize_tool_result(_get_product_details_impl(product_ref))

    def _descriptor_field(descriptor: Any, name: str) -> Any:
        """Read one descriptor field whether it arrived typed or as a dict."""

        if isinstance(descriptor, dict):
            return descriptor.get(name)
        return getattr(descriptor, name, None)

    def _established_this_turn(references: Any) -> list[str]:
        """Answer from this turn's evidence, for refs and names alike.

        The shopper asked about something this turn already searched for and
        found. Nothing needs resolving: the answer is in hand, and going to
        history for it returns nothing because the durable index is written
        when the turn ends.
        """

        answers: list[str] = []
        for descriptor in references or []:
            product_ref = _descriptor_field(descriptor, "product_ref")
            display_name = _descriptor_field(descriptor, "display_name")
            product = None
            if product_ref:
                product = scope.product_evidence.get(
                    str(product_ref).strip().strip(_REFERENCE_WRAPPERS)
                )
            if product is None and display_name:
                wanted = " ".join(str(display_name).casefold().split())
                product = next(
                    (
                        item
                        for item in scope.product_evidence.values()
                        if " ".join(str(item.display_name).casefold().split())
                        == wanted
                    ),
                    None,
                )
            if product is None:
                return []
            answers.append(
                f"REFERENCE {_descriptor_field(descriptor, 'reference_id')}: "
                f"ALREADY ESTABLISHED THIS TURN. "
                f"PRODUCT_REF: {product.product_id}. "
                f"NAME: {product.display_name}. "
                "Use it directly; it needs no resolution and no search."
            )
        return answers

    def _catalog_name_lookup(result: Any, references: Any) -> str:
        """Look the shopper's product name up in the catalog, and say so.

        Nothing in this conversation matched, which means the shopper named
        a product the assistant never showed -- and that is a search
        request. Telling the model to search was a sentence in a tool
        result, so it was advisory: measured across full conversations it
        was obeyed most of the time and, when it was not, the assistant
        offered products it had shown earlier and the shopper never got the
        one they asked for.

        So the runtime does it. A name lookup needs no taxonomy and no
        filters -- nothing that belongs to the model -- so it can be
        composed here without deciding anything on the model's behalf.

        What comes back is labelled for what it is: found by name, not
        shown before. Whether one of these IS the product the shopper named
        or merely resembles it is a judgement about language, which the
        model makes; supplying honest facts to judge from is our job.
        """

        unresolved = {
            item.reference_id
            for item in result.results
            if item.status == "not_found" and not item.blocking_field
        }
        names: list[str] = []
        for descriptor in references or []:
            reference_id = _descriptor_field(descriptor, "reference_id")
            display_name = _descriptor_field(descriptor, "display_name")
            if reference_id not in unresolved:
                continue
            # The name is usually in `display_name`, but the model may put
            # it in `reference_id` instead -- "Southwest Bracelet" as the
            # label rather than the name. Both fields are the model's own
            # free text; either may carry it.
            text = str(display_name or reference_id or "").strip()
            if text and text not in names:
                names.append(text)
        if not names:
            return ""

        sections: list[str] = []
        for name in names[:_MAX_NAME_LOOKUPS]:
            try:
                execution = execute_catalog_search(
                    CatalogSearchPlan(
                        should_search=True,
                        semantic_queries=[name],
                        hard_filters={},
                        search_mode="text",
                        top_k=4,
                    ),
                    runtime.config.retriever_port,
                    timeout_seconds=getattr(
                        runtime.config, "catalog_search_timeout_seconds", None
                    ),
                )
            except Exception:  # pragma: no cover - retrieval already degrades
                continue
            found = execution.result
            if not found.ok or not found.products:
                sections.append(
                    f'CATALOG NAME LOOKUP "{name}": the catalog returned '
                    "nothing for that name. Tell the shopper it is not "
                    "carried. Do not offer a different product as though it "
                    "were the one they named."
                )
                continue
            # Registered exactly as a search result is, so these are
            # addable this turn and resolvable in the next one.
            scope.product_evidence.add(found.products)
            _append_product_results(state, found.products)
            for product in found.products:
                if product.image_url:
                    scope.retrieved[product.display_name] = product.image_url
            # Whether a product was shown is a fact this record holds, so
            # it is read rather than assumed: a resolvable reference can
            # arrive here for a product an earlier turn did show.
            shown_before = {
                product.product_id: _where_a_product_was_already_shown(
                    state.historical_product_sets, product.product_id
                )
                for product in found.products
            }
            lines = [
                f'CATALOG NAME LOOKUP "{name}": the catalog was searched '
                "by that name; these are the closest matches in rank "
                "order. Each line records whether you had already shown "
                "it, and only what the line says is true.",
            ]
            for rank, product in enumerate(found.products, start=1):
                price = (
                    f" - ${product.price.amount:.2f} {product.price.currency}"
                    if getattr(product, "price", None)
                    else ""
                )
                seen = shown_before.get(product.product_id)
                if seen:
                    under = f" under {seen['group']}" if seen["group"] else ""
                    where = (
                        f" -- SHOWN EARLIER, turn {seen['turn_sequence']} "
                        f"as #{seen['position']}{under}"
                    )
                else:
                    where = " -- not shown earlier"
                lines.append(
                    f"{rank}. {product.display_name}{price} "
                    f"[PRODUCT_REF {product.product_id}]{where}"
                )
            exact = [
                product
                for product in found.products
                if _same_product_display_name(name, product.display_name)
            ]
            if len(exact) == 1:
                # Naming a product by the name the catalog gives it is not
                # a resemblance to be judged -- it is the same product, and
                # choosing it is the shopper's to do. Told to "offer it and
                # ask which size", the assistant answered "add the
                # Southwest Bracelet" with "I found a Southwest Bracelet
                # for $169.99. Would you like me to add that?" -- asking
                # permission for the thing it had just been asked to do,
                # about a bracelet that has no size to ask about.
                match = exact[0]
                sizes = _advertised_sizes(match)
                lines.append(
                    f"'{match.display_name}' is the product they named, by "
                    "the catalog's own name for it. They have chosen it. "
                    + (
                        "You showed it earlier; do not suggest otherwise. "
                        "Then "
                        if shown_before.get(match.product_id)
                        else "Say plainly that it was not among the ones "
                        "you had shown, then "
                    )
                    + (
                        # Only a catalog that says "onesize" settles it.
                        # Silence about sizes is not evidence of having
                        # none, and a garment added in a size nobody chose
                        # is the failure this must not reintroduce.
                        "add it."
                        if sizes == [_ONE_SIZE]
                        else "ask which size"
                        + (
                            ", offering " + ", ".join(sizes)
                            if sizes
                            else ""
                        )
                        + " -- unless they already said one, in which case "
                        "add it."
                    )
                    + " Do not ask whether to add what they asked you to add."
                )
            else:
                lines.append(
                    (
                        "Some of these you have already shown -- the lines "
                        "above say which. Do not claim otherwise about "
                        "those. "
                        if any(shown_before.values())
                        else "Say plainly that this was not something you "
                        "had shown. "
                    )
                    + "If one of these is the product the shopper named, "
                    "offer it and ask which size before adding. If none "
                    "is, say you do not carry that one and name the "
                    "closest you do -- never present a different product "
                    "as the one they asked for."
                )
            sections.append("\n".join(lines))
        return "\n\n".join(section for section in sections if section)

    def _resolve_conversation_products_impl(
        references: list[ProductReferenceDescriptor],
    ):
        """Resolve products the shopper refers to from earlier in this
        conversation. Use only when a needed product was not established
        in the current turn. Submit exact descriptors from the historical
        product index.

        Multiple matches need one concise clarification: never guess, and
        never mutate the cart on a guess. Zero matches means the shopper
        referred to something never shown -- if they named a product,
        search for it and show the closest matches, then ask which they
        meant; if they pointed at an earlier item, ask which one. Never add
        a product the shopper has not been shown, and never accept a
        product link or a price as identification.

        Written here rather than in each skill because all three skills
        that hold this tool need the same rule, and three copies of it had
        already begun to differ.
        """

        # This turn's own evidence first, before the memory service is
        # called at all. A product this turn searched for is not in the
        # durable index yet -- that is written when the turn finalizes --
        # so asking history about it returns nothing, and the turn spends a
        # round trip rediscovering what it already holds. The two records
        # disagree only inside the turn that created one of them; reading
        # the nearer one first is what makes them agree.
        established = _established_this_turn(references)
        if established:
            return "\n".join(established)

        with scope.resolution_lock:
            # A resolution that found something ends the budget. One that
            # found nothing does not: the failure itself says "correct that
            # field and retry", and refusing the retry would have the turn
            # ask the shopper to name a product the assistant already named.
            # Attempts are still counted, so a call that keeps missing
            # terminates.
            if (
                scope.product_resolution_used
                or scope.product_resolution_attempts
                >= _MAX_PRODUCT_RESOLUTION_ATTEMPTS
            ):
                return control(
                    "STOP_TOOL_USE: Historical product resolution limit "
                    "reached for this turn. Use the resolution results you "
                    "have and ask one concise clarification if needed.",
                    ControlSignal.STOP_TOOL_USE,
                )
            scope.product_resolution_attempts += 1

        try:
            result = runtime._conversation_products.resolve(
                identity.conversation_id,
                references,
            )
        except (ConversationProductsError, ValidationError):
            return (
                "REFERENCE RESOLUTION UNAVAILABLE: Ask which earlier product "
                "the shopper means; do not guess or search for a substitute."
            )
        scope.product_evidence.add_resolutions(result.results, references)
        state.system_identified_products = list(
            scope.product_evidence.system_identified()
        )
        for resolution in result.results:
            if resolution.status != "resolved":
                continue
            product = resolution.matches[0].product
            if product.image_url:
                scope.retrieved[product.display_name] = product.image_url
        if any(item.status == "resolved" for item in result.results):
            scope.product_resolution_used = True
            return format_product_resolution(result)
        # Nothing resolved. The products this conversation has shown are
        # already recorded, so the next attempt can be a lookup in that
        # record rather than another guess at a descriptor.
        #
        # Only for a near miss. A blocking field means the call pointed at
        # something it had seen and got one field wrong, and the record is
        # what corrects it. When nothing matches at all, the shopper named a
        # product that was never shown -- that is a search request, and
        # handing back a list of earlier products reads as a menu and
        # suppresses the search: asked for a dress by name, the assistant
        # offered four it had shown before and never looked in the catalog.
        near_miss = any(item.blocking_field for item in result.results)
        if not near_miss:
            looked_up = _catalog_name_lookup(result, references)
            if looked_up:
                return "\n\n".join(
                    (format_product_resolution(result), looked_up)
                )
        return "\n\n".join(
            value
            for value in (
                format_product_resolution(result),
                format_historical_product_index(state.historical_product_sets)
                if near_miss
                else "",
            )
            if value
        )

    @tool(
        args_schema=ResolveConversationProductsRequest,
        return_direct=False,
        response_format="content_and_artifact",
    )
    def resolve_conversation_products_tool(
        references: list[ProductReferenceDescriptor],
    ):
        """Resolve products the shopper refers to from earlier in this
        conversation. Use only when a needed product was not established
        in the current turn. Submit exact descriptors from the historical
        product index. If a reference is ambiguous, ask one concise
        clarification and do not guess. If nothing matches at all, the
        result says what to do next.
        """

        return normalize_tool_result(
            _resolve_conversation_products_impl(references)
        )

    @tool(args_schema=_DescribeCatalogInput, return_direct=False)
    def describe_catalog_tool() -> str:
        """What this shop holds: how many products, which categories, the
        price range of each, and their subcategories. Use for questions
        about the SHOP rather than about a product -- the most or least
        expensive thing, whether anything falls in a price range, what
        departments exist. Takes no arguments and searches nothing.

        A fact about the catalog comes from here, never from the results of
        one search: the dearest item a search happened to return is that
        search's maximum, not the shop's. To name the actual item, read the
        range here and then search that category at that bound.
        """

        return format_catalog_shape(runtime._catalog_capabilities.get())

    return search_catalog_tool, get_product_details_tool, resolve_conversation_products_tool, describe_catalog_tool
