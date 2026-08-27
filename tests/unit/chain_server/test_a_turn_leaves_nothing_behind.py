# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A finished turn leaves nothing in the process, however it finished.

The checkpointer is `MemorySaver`: a dict in this process, with no eviction and
no size bound. Its thread is keyed on (conversation_id, request_id), so it
belongs to exactly one request and nothing will ever read it again once that
request is done.

Freeing it used to be conditional on the turn finalizing. Every turn that did
not -- a superseded attempt, a memory service blip, a turn that never started --
left its whole message history behind for the life of the process. Pods are
long-lived, so that ends in an OOMKill, and running more pods only means more of
them leaking.

These tests drive each way a turn can end and assert the same thing every time:
nothing is left. They are about the failure paths, because the success path was
the only one that ever worked.
"""

from __future__ import annotations

from typing import Any

import pytest


class _CountingCheckpointer:
    """Enough of a checkpointer to see what a turn leaves behind."""

    def __init__(self) -> None:
        self.threads: set[str] = set()

    def remember(self, thread_id: str) -> None:
        self.threads.add(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.threads.discard(thread_id)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A runtime with the turn body stubbed and the checkpointer observable.

    The real `_run_turn_inner` reaches a model, a catalog and a memory service.
    What is under test is the lifecycle around it, so the body is replaced per
    test with whichever ending is being checked.
    """

    from chain_server.src.deepagents_runtime import DeepAgentsRuntime

    runtime = DeepAgentsRuntime.__new__(DeepAgentsRuntime)
    runtime._checkpointer = _CountingCheckpointer()
    return runtime


def _identity() -> Any:
    from chain_server.src.turn_support import RequestIdentity

    return RequestIdentity(
        session_id="s1",
        conversation_id="c1",
        cart_id="cart1",
        context_user_id=1,
        cart_user_id=1,
        request_id="r1",
    )


async def _run(runtime: Any, identity: Any, body: Any) -> Any:
    """Drive `_run_turn` with a stubbed inner body."""

    from types import SimpleNamespace

    state = SimpleNamespace(agent_diagnostics={}, response="")
    runtime._run_turn_inner = body
    runtime._checkpointer.remember(identity.checkpoint_thread_id)
    return await runtime._run_turn(state, identity)


@pytest.mark.asyncio
async def test_a_turn_that_succeeds_leaves_nothing(runtime) -> None:
    identity = _identity()

    async def body(state, ident, on_progress=None):
        return state

    await _run(runtime, identity, body)

    assert runtime._checkpointer.threads == set()


@pytest.mark.asyncio
async def test_a_turn_that_raises_leaves_nothing(runtime) -> None:
    """The path that leaked. An unexpected error used to keep the checkpoint."""

    identity = _identity()

    async def body(state, ident, on_progress=None):
        raise RuntimeError("model went away")

    with pytest.raises(RuntimeError):
        await _run(runtime, identity, body)

    assert runtime._checkpointer.threads == set()


@pytest.mark.asyncio
async def test_a_cancelled_turn_leaves_nothing(runtime) -> None:
    """A shopper closing the tab is the commonest way a turn ends early."""

    import asyncio

    identity = _identity()

    async def body(state, ident, on_progress=None):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _run(runtime, identity, body)

    assert runtime._checkpointer.threads == set()


@pytest.mark.asyncio
async def test_the_thread_belongs_to_one_request_and_never_to_a_conversation(
    runtime,
) -> None:
    """Why no sticky sessions, and why freeing it unconditionally is safe.

    The key includes request_id, so two turns of one conversation can never
    share a thread. There is nothing for session affinity to preserve, and
    nothing that could still want this thread after its request ends.
    """

    from chain_server.src.turn_support import RequestIdentity

    common = dict(
        session_id="s1", conversation_id="c1", cart_id="cart1",
        context_user_id=1, cart_user_id=1,
    )
    first = RequestIdentity(request_id="r1", **common)
    second = RequestIdentity(request_id="r2", **common)

    assert first.checkpoint_thread_id != second.checkpoint_thread_id


@pytest.mark.asyncio
async def test_a_genuinely_cancelled_task_still_frees_its_checkpoint() -> None:
    """Cancellation for real, not a raised CancelledError.

    The two are not the same. Freeing happens in a `finally` that awaits, and
    inside a task that is actually being cancelled an await can behave
    differently from one following a plain raise. A shopper closing the tab
    cancels the task, so this is the path that happens in production, and the
    other test would not notice if it stopped working.
    """

    import asyncio

    from chain_server.src.deepagents_runtime import DeepAgentsRuntime
    from types import SimpleNamespace

    runtime = DeepAgentsRuntime.__new__(DeepAgentsRuntime)
    runtime._checkpointer = _CountingCheckpointer()
    identity = _identity()
    runtime._checkpointer.remember(identity.checkpoint_thread_id)

    started = asyncio.Event()

    async def body(state, ident, on_progress=None):
        started.set()
        await asyncio.sleep(3600)  # as a turn waiting on a model would

    runtime._run_turn_inner = body
    state = SimpleNamespace(agent_diagnostics={}, response="")

    task = asyncio.create_task(runtime._run_turn(state, identity))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime._checkpointer.threads == set()
