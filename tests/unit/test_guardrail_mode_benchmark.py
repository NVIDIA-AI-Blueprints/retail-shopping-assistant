"""Regression tests for guardrail benchmark validity classification."""

from benchmarks.guardrail_mode_benchmark import (
    guardrail_failed,
    response_was_blocked,
)


def test_guardrail_failure_is_not_an_allowed_response() -> None:
    body = {
        "response": "Guardrails are temporarily unavailable.",
        "model_usage": {
            "content_safety": {"status": "failed", "calls": 1},
        },
    }

    assert guardrail_failed(body) is True
    assert response_was_blocked(body) is False


def test_typed_diagnostics_identify_a_block_without_matching_copy() -> None:
    body = {
        "response": "Deployment-specific refusal text.",
        "agent_diagnostics": {
            "final_termination_reason": "input_guardrail_blocked",
        },
    }

    assert guardrail_failed(body) is False
    assert response_was_blocked(body) is True
