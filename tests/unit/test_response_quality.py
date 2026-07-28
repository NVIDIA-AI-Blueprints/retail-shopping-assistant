from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RESPONSE_QUALITY_PATH = REPO_ROOT / "tests" / "integration" / "response_quality.py"


def _load_response_quality(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "test-judge")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.test/v1")
    monkeypatch.setenv("JUDGE_API_KEY_ENV", "TEST_JUDGE_API_KEY")
    monkeypatch.setenv("TEST_JUDGE_API_KEY", "test-key")

    spec = importlib.util.spec_from_file_location("response_quality_under_test", RESPONSE_QUALITY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _judge_response(score: int = 4, reasoning: str = "Contextually consistent."):
    arguments = f'{{"judgement": {score}, "reasoning": "{reasoning}"}}'
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    message = SimpleNamespace(tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_judge_prompt_uses_actual_prior_turns_as_authoritative_context(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _judge_response()

    response_quality.LLM_CLIENT = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response_quality.judge_test(
        query="What top would go with that?",
        answer="A cream blouse would balance the black skirt.",
        ideal_answer="Pair it with a top for a blush satin skirt.",
        prior_turns=[
            {
                "shopper": "Show me skirts.",
                "assistant": "I found a black skirt and a green skirt.",
            },
            {
                "shopper": "The black one is my favorite.",
                "assistant": "Great, we'll use the black skirt.",
            },
        ],
        verbose=False,
    )

    prompt = captured["messages"][1]["content"]
    assert prompt.index("Shopper: Show me skirts.") < prompt.index(
        "Assistant: I found a black skirt and a green skirt."
    )
    assert prompt.index("The black one is my favorite.") < prompt.index(
        "Great, we'll use the black skirt."
    )
    assert "actual conversation history is authoritative" in prompt
    assert "reference answer is guidance" in prompt
    assert "must not override the actual conversation history" in prompt
    assert "matches the reference in content and clarity" not in prompt


def test_judge_test_remains_compatible_without_prior_turns(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _judge_response(score=5, reasoning="Clear and complete.")

    response_quality.LLM_CLIENT = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = response_quality.judge_test(
        query="Show me bags.",
        answer="Here are two bags.",
        ideal_answer="Here are some bags.",
        verbose=False,
    )

    assert result == {"score": 5, "justification": "Clear and complete."}
    assert "ACTUAL PRIOR CONVERSATION" not in captured["messages"][1]["content"]


def test_diagnostic_expectations_accept_redacted_event_weather_trace(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)

    response_quality._validate_diagnostic_expectations(
        {
            "required_skills": ["outfit-styling", "event-context"],
            "forbidden_skills": ["product-discovery"],
            "required_tools": ["get_weather_forecast_tool"],
            "forbidden_tools": ["search_catalog_tool"],
            "tool_call_counts": {"get_weather_forecast_tool": 1},
            "weather_tool_calls": 1,
        },
        {
            "skill_files_read": [
                "/shopper/outfit-styling/SKILL.md",
                "/shopper/event-context/SKILL.md",
            ],
            "tool_calls": [
                {
                    "tool_name": "get_weather_forecast_tool",
                    "arguments": {"redacted": True},
                    "status": "completed",
                }
            ],
        },
        label="weather turn",
    )


def test_diagnostic_expectations_enforce_exact_tool_call_count(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)

    with pytest.raises(
        AssertionError,
        match="expected 1 search_catalog_tool calls, found 2",
    ):
        response_quality._validate_diagnostic_expectations(
            {"tool_call_counts": {"search_catalog_tool": 1}},
            {
                "tool_calls": [
                    {
                        "tool_name": "search_catalog_tool",
                        "status": "rejected",
                    },
                    {
                        "tool_name": "search_catalog_tool",
                        "status": "completed",
                    },
                ]
            },
            label="occasion turn",
        )


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (
            {
                "skill_files_read": ["/shopper/outfit-styling/SKILL.md"],
                "tool_calls": [],
            },
            "missing required skill event-context",
        ),
        (
            {
                "skill_files_read": [
                    "/shopper/outfit-styling/SKILL.md",
                    "/shopper/event-context/SKILL.md",
                ],
                "tool_calls": [
                    {
                        "tool_name": "get_weather_forecast_tool",
                        "arguments": {
                            "location": "Seattle",
                            "location_query": "Seattle, WA",
                            "relative_date": "next_week",
                        },
                        "status": "completed",
                    }
                ],
            },
            "weather tool arguments were not redacted",
        ),
    ],
)
def test_diagnostic_expectations_fail_before_judging(
    monkeypatch,
    diagnostics,
    message,
):
    response_quality = _load_response_quality(monkeypatch)

    with pytest.raises(AssertionError, match=message):
        response_quality._validate_diagnostic_expectations(
            {
                "required_skills": ["outfit-styling", "event-context"],
                "required_tools": ["get_weather_forecast_tool"],
                "weather_tool_calls": 1,
            },
            diagnostics,
            label="weather turn",
        )
