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
    legacy_user_id: int,
    session_id: str | None = None,
    conversation_id: str | None = None,
    cart_id: str | None = None,
    request_id: str | None = None,
    shopper_profile_id: str | None = None,
) -> RequestIdentity:
    """Create scoped request identity while preserving legacy user_id behavior."""

    session = session_id or f"legacy-session-{legacy_user_id}"
    conversation = conversation_id or f"legacy-conversation-{legacy_user_id}"
    cart = cart_id or f"legacy-cart-{legacy_user_id}"
    return RequestIdentity(
        session_id=session,
        conversation_id=conversation,
        cart_id=cart,
        context_user_id=(
            _stable_numeric_id("conversation", conversation_id)
            if conversation_id
            else legacy_user_id
        ),
        cart_user_id=_stable_numeric_id("cart", cart_id) if cart_id else legacy_user_id,
        request_id=request_id or str(uuid.uuid4()),
        shopper_profile_id=shopper_profile_id,
    )


def _stable_numeric_id(namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()
    return int(digest[:15], 16)
