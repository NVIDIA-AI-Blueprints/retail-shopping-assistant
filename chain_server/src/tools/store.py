# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools for store questions: policies, stock and active promotions."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from shared.commerce_contracts import (
    CheckActivePromotionsResult,
    CheckProductAvailabilityInput,
    CheckProductAvailabilityResult,
    GetStorePolicyInput,
    GetStorePolicyResult,
)

from ..commerce_tools import (
    check_active_promotions,
    check_product_availability,
    get_store_policy,
)
from ..config import ChainServerConfig
from ..control_signals import normalize_tool_result
from ..turn_scope import TurnScope


class _GetStorePolicyInput(BaseModel):
    topic: Literal[
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    ] = Field(description="Policy topic to look up.")


class _AvailabilityItemInput(BaseModel):
    product_ref: str = Field(
        description=(
            "PRODUCT_REF established by current-turn search or historical-product "
            "resolution."
        )
    )
    variant_hint: str | None = Field(
        default=None,
        description="Requested size wording, such as 'size 8'.",
    )


_MAX_AVAILABILITY_ITEMS: int = int(
    ChainServerConfig.model_fields["search_products_per_call"].default
)


class _CheckAvailabilityInput(BaseModel):
    items: list[_AvailabilityItemInput] = Field(
        ...,
        min_length=1,
        max_length=_MAX_AVAILABILITY_ITEMS,
        description=(
            "Every product the shopper asked about, in one call. They are "
            "checked together, so four products cost one round trip, not four."
        ),
    )


_SHARED_CONFIG_ROOT_ENV = "SHARED_CONFIG_ROOT"


_STORE_POLICIES_RELATIVE_PATH = Path("chain_server/store_policies.yaml")


def _store_policies_path() -> Path:
    """Resolve controlled policy content outside the agent-readable skill root."""

    configured_root = os.environ.get(_SHARED_CONFIG_ROOT_ENV, "").strip()
    if configured_root:
        return Path(configured_root) / _STORE_POLICIES_RELATIVE_PATH

    deployed_path = Path("/app/shared/configs") / _STORE_POLICIES_RELATIVE_PATH
    if deployed_path.is_file():
        return deployed_path

    return (
        Path(__file__).resolve().parents[3]
        / "shared"
        / "configs"
        / _STORE_POLICIES_RELATIVE_PATH
    )


def _format_policy_result(result: GetStorePolicyResult) -> str:
    if not result.ok or result.policy is None:
        message = result.error.message if result.error else "unknown error"
        return f"POLICY NOT AVAILABLE: {message}"
    policy = result.policy
    return f"STORE POLICY — {policy.title}\n{policy.body}"


def _format_availability_result(result: CheckProductAvailabilityResult) -> str:
    return f"AVAILABILITY ({result.product_ref}): {result.message}"


def _format_promotions_result(result: CheckActivePromotionsResult) -> str:
    status = "YES" if result.active else "NO"
    return f"ACTIVE PROMOTIONS: {status}\n{result.message}"


def build_store_tools(
    scope: TurnScope,
):
    """Store policy, availability and promotion tools, answering repeats from `scope`."""

    from langchain_core.tools import tool

    @tool(args_schema=_GetStorePolicyInput, return_direct=False)
    def get_store_policy_tool(
        topic: Literal[
            "returns",
            "shipping",
            "sizing",
            "payment",
            "price_match",
            "gift_cards",
        ],
    ) -> str:
        """Look up store policy for: returns, shipping, sizing, payment,
        price_match, or gift_cards. Use ONLY for these policy topics. Do
        NOT use for product facts, prices, or availability. If this tool
        returns a not-found error, relay the message to the shopper and
        direct them to the retailer's help center. Do NOT substitute model
        knowledge for a missing policy.
        """

        result = get_store_policy(
            GetStorePolicyInput(topic=topic),
            _store_policies_path(),
        )
        return _format_policy_result(result)

    # ``content_and_artifact`` because a repeat is refused with a typed
    # control signal, and a signal rides on the artifact. Returning the
    # tuple from a tool declared without it put the pair in the content
    # instead: the model read ``["STOP_TOOL_USE: ...", {...}]``, the
    # runtime saw no signal at all, and one turn made this call sixteen
    # times.
    @tool(
        args_schema=_CheckAvailabilityInput,
        return_direct=False,
        response_format="content_and_artifact",
    )
    def check_product_availability_tool(items):
        """Check whether products are available or in stock. Use ONLY when
        the shopper explicitly asks about availability, stock, or a specific
        size. Requires a PRODUCT_REF established by search or
        historical-product resolution. Do NOT use for browsing. Pass every
        product being asked about in one call. The deterministic stub
        reports general availability, sized availability for apparel and
        footwear, and one-size availability for other product categories.
        """

        requests = [
            item if isinstance(item, dict) else item.model_dump()
            for item in items
        ]
        asked = json.dumps(requests, sort_keys=True)
        held = scope.answer_already_given(
            "check_product_availability_tool",
            asked,
        )
        if held is not None:
            return normalize_tool_result(held)

        def _one(entry: dict[str, Any]) -> str:
            product_ref = entry.get("product_ref") or ""
            product = scope.product_evidence.get(product_ref)
            if product is None:
                return (
                    f"PRODUCT_REF '{product_ref}' is unknown in this "
                    "conversation. Search this turn or resolve the earlier "
                    "product first."
                )
            return _format_availability_result(
                check_product_availability(
                    CheckProductAvailabilityInput(
                        product_ref=product_ref,
                        variant_hint=entry.get("variant_hint"),
                    ),
                    product,
                )
            )

        # Each check stands in for an inventory-system lookup, so they go out
        # together. Asking about four products cost four model round trips at
        # roughly 8.7s each -- enough to exhaust a turn's step budget before
        # the shopper got an answer.
        if len(requests) == 1:
            answer = _one(requests[0])
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(requests), 8)
            ) as pool:
                answer = "\n\n".join(pool.map(_one, requests))
        scope.remember_answer(
            "check_product_availability_tool",
            asked,
            answer,
        )
        return normalize_tool_result(answer)

    @tool(return_direct=False, response_format="content_and_artifact")
    def check_active_promotions_tool():
        """Check whether a sale, discount, or promotion is currently active.
        Use ONLY when the shopper explicitly asks about promotion status. Do
        NOT use for ordinary affordable browsing, a price ceiling, price
        matching, or product availability. Catalog search does not establish
        sale status.
        """

        held = scope.answer_already_given("check_active_promotions_tool", "")
        if held is not None:
            return normalize_tool_result(held)
        answer = _format_promotions_result(check_active_promotions())
        scope.remember_answer("check_active_promotions_tool", "", answer)
        return normalize_tool_result(answer)

    return get_store_policy_tool, check_product_availability_tool, check_active_promotions_tool
