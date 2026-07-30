from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from chain_server.src.tool_policy import SHOPPING_TOOL_POLICIES


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "tests" / "integration" / "diagnostic_validation.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "diagnostic_validation_under_test",
        VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _diagnostics() -> dict:
    return {
        "skill_files_read": [
            "/shopper/outfit-styling/SKILL.md",
            "/shopper/event-context/SKILL.md",
        ],
        "tool_calls": [
            {
                "tool_name": "activate_shopper_skills_tool",
                "status": "completed",
                "arguments": {
                    "skill_names": ["outfit-styling", "event-context"],
                    "event_context_next_question": "none",
                },
            },
            {
                "tool_name": "resolve_conversation_products_tool",
                "status": "completed",
            },
            {
                "tool_name": "get_product_details_tool",
                "status": "completed",
            },
            {
                "tool_name": "get_product_details_tool",
                "status": "completed",
            },
        ],
        "product_evidence": [
            {
                "product_name": "Intricate Lace Gown",
                "source_tool": "get_product_details_tool",
                "evidence_type": "product_detail",
            },
            {
                "product_name": "Wavy Hem Satin Dress",
                "source_tool": "get_product_details_tool",
                "evidence_type": "product_detail",
            },
        ],
        "weather_receipt_status": "bound",
    }


def _expectations() -> dict:
    return {
        "required_skills": ["outfit-styling", "event-context"],
        "forbidden_skills": ["product-discovery"],
        "required_event_context_next_question": "none",
        "required_business_sequence": [
            "resolve_conversation_products_tool",
            "get_product_details_tool",
            "get_product_details_tool",
        ],
        "required_product_detail_names": [
            "Intricate Lace Gown",
            "Wavy Hem Satin Dress",
        ],
        "required_weather_receipt_status": "bound",
        "required_response_phrases": [
            "Intricate Lace Gown",
            "Wavy Hem Satin Dress",
        ],
        "forbidden_response_phrases": ["Previously shown options still in play"],
    }


def test_standalone_validator_needs_no_judge_environment(
    monkeypatch,
) -> None:
    for name in (
        "JUDGE_BASE_URL",
        "JUDGE_MODEL",
        "JUDGE_API_KEY_ENV",
        "TEST_JUDGE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    validator = _load_validator()

    validator._validate_diagnostic_expectations(
        _expectations(),
        _diagnostics(),
        response=(
            "Intricate Lace Gown is more formal; Wavy Hem Satin Dress is the "
            "cleaner semi-formal choice."
        ),
        label="comparison",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda trace: trace["tool_calls"].reverse(),
            "expected business sequence",
        ),
        (
            lambda trace: trace["tool_calls"][2].update(status="rejected"),
            "non-completed calls",
        ),
        (
            lambda trace: trace["tool_calls"][0]["arguments"].update(
                event_context_next_question="event_date"
            ),
            "expected event-context next question",
        ),
        (
            lambda trace: trace.update(weather_receipt_status="not_applicable"),
            "expected weather receipt status",
        ),
    ],
)
def test_standalone_validator_rejects_wrong_or_incomplete_sequence(
    mutate,
    message: str,
) -> None:
    validator = _load_validator()
    diagnostics = _diagnostics()
    mutate(diagnostics)

    with pytest.raises(AssertionError, match=message):
        validator._validate_diagnostic_expectations(
            _expectations(),
            diagnostics,
            response=("Intricate Lace Gown and Wavy Hem Satin Dress are compared."),
            label="comparison",
        )


def test_standalone_validator_checks_weather_outcome_and_response() -> None:
    validator = _load_validator()
    expectations = {
        "required_business_sequence": ["get_weather_forecast_tool"],
        "required_weather_trace": {
            "request_shape": "relative_range",
            "location_source": "shopper_provided_location",
            "provider_input": "location_query",
            "outcome": "success",
        },
        "required_weather_receipt_status": "promotion_prepared",
        "required_response_phrases": ["Weather Data Provided by Visual Crossing"],
        "forbidden_response_phrases": ["valid live forecast"],
    }
    diagnostics = {
        "weather_receipt_status": "promotion_prepared",
        "tool_calls": [
            {
                "tool_name": "get_weather_forecast_tool",
                "status": "completed",
                "arguments": {"redacted": True},
                "weather": {
                    "request_shape": "relative_range",
                    "location_source": "shopper_provided_location",
                    "provider_input": "location_query",
                    "outcome": "success",
                },
            }
        ]
    }

    validator._validate_diagnostic_expectations(
        expectations,
        diagnostics,
        response="Weather Data Provided by Visual Crossing",
        label="weather",
    )

    diagnostics["tool_calls"][0]["weather"]["outcome"] = "provider_unavailable"
    with pytest.raises(AssertionError, match="expected exact weather trace"):
        validator._validate_diagnostic_expectations(
            expectations,
            diagnostics,
            response="Weather Data Provided by Visual Crossing",
            label="weather",
        )

    diagnostics["tool_calls"][0]["weather"]["outcome"] = "success"
    diagnostics["tool_calls"][0]["weather"]["resolved_location"] = "must not leak"
    with pytest.raises(AssertionError, match="expected exact weather trace"):
        validator._validate_diagnostic_expectations(
            expectations,
            diagnostics,
            response="Weather Data Provided by Visual Crossing",
            label="weather",
        )


def test_standalone_validator_rejects_missing_or_structured_receipt_status() -> None:
    validator = _load_validator()
    expectations = {"required_weather_receipt_status": "bound"}

    with pytest.raises(AssertionError, match="expected weather receipt status"):
        validator._validate_diagnostic_expectations(
            expectations,
            {"tool_calls": []},
            label="receipt",
        )

    with pytest.raises(AssertionError, match="expected weather receipt status"):
        validator._validate_diagnostic_expectations(
            expectations,
            {
                "tool_calls": [],
                "weather_receipt_status": {
                    "status": "bound",
                    "location": "must not be exposed",
                },
            },
            label="receipt",
        )


def test_business_sequence_registry_matches_runtime_policy() -> None:
    validator = _load_validator()

    assert validator._BUSINESS_TOOLS == set(SHOPPING_TOOL_POLICIES)


def test_standalone_main_validates_real_yaml_without_judge(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    validator = _load_validator()
    conversation_dir = tmp_path / "conversations" / "focused_diagnostics"
    result_dir = conversation_dir / "results"
    result_dir.mkdir(parents=True)
    (conversation_dir / "conv_0.yaml").write_text(
        yaml.safe_dump(
            {
                "queries": ["Compare those."],
                "diagnostic_expectations": [
                    {
                        "required_business_sequence": ["get_product_details_tool"],
                        "required_response_phrases": ["Compared"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "conv_0.yaml").write_text(
        yaml.safe_dump(
            {
                "results": [
                    {
                        "content": "Compared both products.",
                        "agent_diagnostics": {
                            "tool_calls": [
                                {
                                    "tool_name": "get_product_details_tool",
                                    "status": "completed",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_PATH", "focused_diagnostics")
    monkeypatch.setenv("RESULT_DIRECTORY", "results")

    assert validator.main() == 0
    assert "Diagnostic validation passed: 1 scenario file(s)." in (
        capsys.readouterr().out
    )

    (result_dir / "extra.yaml").write_text("results: []\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Mismatch between query"):
        validator.main()
