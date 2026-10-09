# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who a request is for: session, conversation, cart and user ids."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestIdentity:
    """Server-owned identity used to scope one assistant turn."""

    session_id: str
    conversation_id: str
    cart_id: str
    context_user_id: int
    cart_user_id: int
    request_id: str
    shopper_profile_id: str | None = None

    @property
    def legacy_user_id(self) -> int:
        return self.context_user_id

    @property
    def checkpoint_thread_id(self) -> str:
        return json.dumps(
            [self.conversation_id, self.request_id],
            separators=(",", ":"),
        )


def create_request_identity(
    *,
    conversation_id: str,
    cart_id: str,
    session_id: str | None = None,
    request_id: str | None = None,
    shopper_profile_id: str | None = None,
) -> RequestIdentity:
    """Scope one turn to the caller's opaque conversation and cart handles.

    There is no fallback to the request's user_id: it is guessable, and the
    cart routes trust whoever holds a cart handle.
    """

    conversation = (conversation_id or "").strip()
    cart = (cart_id or "").strip()
    if not conversation or not cart:
        raise ValueError("conversation_id and cart_id are required")
    return RequestIdentity(
        session_id=session_id or conversation,
        conversation_id=conversation,
        cart_id=cart,
        context_user_id=_stable_numeric_id("conversation", conversation),
        cart_user_id=cart_user_id_for(cart),
        request_id=request_id or str(uuid.uuid4()),
        shopper_profile_id=shopper_profile_id,
    )


def cart_user_id_for(cart_id: str) -> int:
    """The memory service's integer key for an opaque cart handle."""

    return _stable_numeric_id("cart", cart_id)


def _stable_numeric_id(namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()
    return int(digest[:15], 16)
