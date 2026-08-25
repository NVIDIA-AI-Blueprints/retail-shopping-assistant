
from pathlib import Path

# Resolved from this file, not the working directory: CI runs pytest with
# `working-directory: tests`, where a path relative to the repo root does not
# exist. The rest of the suite already does this.
_REPO_ROOT = Path(__file__).resolve().parents[3]
"""One stalled model request must not cost the whole turn.

J03 turn 5 -- "add the Southwest Bracelet", a turn that normally costs ten
seconds -- spent 135.0s and returned "This request took too long to complete."
The trace shows why: a single `model` span of 133.8s. Not a loop, not many
calls. One request that never came back, with no deadline to stop it.
"""

import types

import pytest

from chain_server.src.deepagents_runtime import (

    _MODEL_REQUEST_TIMEOUT_CEILING_SECONDS,
    DeepAgentsRuntime,
)


def _timeout_for(budget: float, reserve: float = 15.0) -> float:
    runtime = types.SimpleNamespace(
        config=types.SimpleNamespace(
            deepagents_execution_timeout_seconds=budget,
            grounding_editor_reserve_seconds=reserve,
        )
    )
    return DeepAgentsRuntime._model_request_timeout(runtime)


@pytest.mark.parametrize("budget", [45.0, 150.0, 300.0])
def test_two_attempts_fit_inside_the_agent_loop(budget: float) -> None:
    """The retry only helps if the turn survives it.

    LangChain's `max_retries` defaults to 2 and fires on errors; a hang is not
    an error, so the deadline is what makes the retry reachable. A fixed forty
    seconds fits a 150-second deployment and swallows a 45-second one whole.
    """

    reserve = min(15.0, budget / 2)
    allowance = max(budget - reserve, budget / 2)
    assert _timeout_for(budget) * 2 <= allowance


def test_it_never_waits_longer_than_the_ceiling() -> None:
    """A request slower than this has stalled, not thought."""

    assert _timeout_for(3600.0) == _MODEL_REQUEST_TIMEOUT_CEILING_SECONDS


def test_a_tiny_budget_still_leaves_a_usable_deadline() -> None:
    assert _timeout_for(4.0) >= 5.0


def test_the_client_is_built_with_it() -> None:
    source = open(_REPO_ROOT / "chain_server/src/deepagents_runtime.py").read()
    block = source[source.index("def _create_chat_model") :][:1600]
    assert "timeout=self._model_request_timeout()" in block
