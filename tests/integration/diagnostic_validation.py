"""Validate live skill, tool, evidence, and response expectations without a Judge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_BUSINESS_TOOLS = {
    "search_catalog_tool",
    "get_product_details_tool",
    "check_product_availability_tool",
    "check_active_promotions_tool",
    "resolve_conversation_products_tool",
    "get_cart_tool",
    "view_cart_total_tool",
    "add_cart_items_tool",
    "remove_cart_item_tool",
    "update_cart_items_tool",
    "get_store_policy_tool",
    "get_weather_forecast_tool",
}


def _validate_diagnostic_expectations(
    expectations: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None,
    *,
    response: str = "",
    label: str = "turn",
) -> None:
    """Fail when one live turn violates its committed deterministic contract."""

    expected = expectations or {}
    trace = diagnostics or {}
    skill_files = set(trace.get("skill_files_read") or [])
    tool_calls = trace.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise AssertionError(f"{label}: agent_diagnostics.tool_calls must be a list")

    normalized_calls = [call for call in tool_calls if isinstance(call, dict)]
    activation_calls = [
        call
        for call in normalized_calls
        if call.get("tool_name") == "activate_shopper_skills_tool"
        and call.get("status") == "completed"
    ]
    called_tools = {str(call.get("tool_name") or "") for call in normalized_calls}
    completed_tools = {
        str(call.get("tool_name") or "")
        for call in normalized_calls
        if call.get("status") == "completed"
    }

    for skill_name in expected.get("required_skills", []):
        path = f"/shopper/{skill_name}/SKILL.md"
        if path not in skill_files:
            raise AssertionError(f"{label}: missing required skill {skill_name}")
    for skill_name in expected.get("forbidden_skills", []):
        path = f"/shopper/{skill_name}/SKILL.md"
        if path in skill_files:
            raise AssertionError(f"{label}: forbidden skill selected {skill_name}")

    required_next_question = expected.get("required_event_context_next_question")
    if required_next_question is not None:
        if len(activation_calls) != 1:
            raise AssertionError(
                f"{label}: expected one completed skill activation, "
                f"found {len(activation_calls)}"
            )
        actual_next_question = activation_calls[0].get(
            "accepted_event_context_next_question",
            (activation_calls[0].get("arguments") or {}).get(
                "event_context_next_question"
            ),
        )
        if actual_next_question != required_next_question:
            raise AssertionError(
                f"{label}: expected event-context next question "
                f"{required_next_question!r}, found "
                f"{actual_next_question!r}"
            )

    required_weather_scope = expected.get("required_weather_scope")
    if required_weather_scope is not None:
        if len(activation_calls) != 1:
            raise AssertionError(
                f"{label}: expected one completed skill activation, "
                f"found {len(activation_calls)}"
            )
        actual_weather_scope = (activation_calls[0].get("arguments") or {}).get(
            "weather_scope"
        )
        if actual_weather_scope != dict(required_weather_scope):
            raise AssertionError(
                f"{label}: expected weather scope "
                f"{dict(required_weather_scope)!r}, found "
                f"{actual_weather_scope!r}"
            )

    for tool_name in expected.get("required_tools", []):
        if tool_name not in completed_tools:
            raise AssertionError(f"{label}: required tool did not complete {tool_name}")
    for tool_name in expected.get("forbidden_tools", []):
        if tool_name in called_tools:
            raise AssertionError(f"{label}: forbidden tool called {tool_name}")

    for tool_name, expected_count in (expected.get("tool_call_counts") or {}).items():
        actual_count = sum(
            call.get("tool_name") == tool_name for call in normalized_calls
        )
        if actual_count != expected_count:
            raise AssertionError(
                f"{label}: expected {expected_count} {tool_name} calls, "
                f"found {actual_count}"
            )

    required_sequence = expected.get("required_business_sequence")
    if required_sequence is not None:
        business_calls = [
            call
            for call in normalized_calls
            if str(call.get("tool_name") or "") in _BUSINESS_TOOLS
        ]
        actual_sequence = [str(call.get("tool_name") or "") for call in business_calls]
        if actual_sequence != list(required_sequence):
            raise AssertionError(
                f"{label}: expected business sequence {list(required_sequence)}, "
                f"found {actual_sequence}"
            )
        incomplete = [
            str(call.get("tool_name") or "")
            for call in business_calls
            if call.get("status") != "completed"
        ]
        if incomplete:
            raise AssertionError(
                f"{label}: business sequence contains non-completed calls {incomplete}"
            )

    expected_detail_names = expected.get("required_product_detail_names")
    if expected_detail_names is not None:
        product_evidence = trace.get("product_evidence") or []
        if not isinstance(product_evidence, list):
            raise AssertionError(
                f"{label}: agent_diagnostics.product_evidence must be a list"
            )
        actual_detail_names = {
            str(record.get("product_name") or "")
            for record in product_evidence
            if isinstance(record, dict)
            and record.get("source_tool") == "get_product_details_tool"
            and record.get("evidence_type") == "product_detail"
            and record.get("product_name")
        }
        required_detail_names = {str(name) for name in expected_detail_names}
        if actual_detail_names != required_detail_names:
            raise AssertionError(
                f"{label}: expected product detail evidence "
                f"{sorted(required_detail_names)}, found "
                f"{sorted(actual_detail_names)}"
            )

    weather_calls = [
        call
        for call in normalized_calls
        if call.get("tool_name") == "get_weather_forecast_tool"
    ]
    expected_weather_calls = expected.get("weather_tool_calls")
    if (
        expected_weather_calls is not None
        and len(weather_calls) != expected_weather_calls
    ):
        raise AssertionError(
            f"{label}: expected {expected_weather_calls} weather calls, "
            f"found {len(weather_calls)}"
        )
    for call in weather_calls:
        if call.get("arguments") != {"redacted": True}:
            raise AssertionError(f"{label}: weather tool arguments were not redacted")

    required_weather_trace = expected.get("required_weather_trace")
    if required_weather_trace is not None:
        actual_weather_traces = [call.get("weather") or {} for call in weather_calls]
        if len(actual_weather_traces) != 1:
            raise AssertionError(
                f"{label}: expected one weather trace, found "
                f"{len(actual_weather_traces)}"
            )
        actual_weather_trace = actual_weather_traces[0]
        expected_weather_trace = dict(required_weather_trace)
        if actual_weather_trace != expected_weather_trace:
            raise AssertionError(
                f"{label}: expected exact weather trace "
                f"{expected_weather_trace!r}, found {actual_weather_trace!r}"
            )

    required_weather_receipt_status = expected.get(
        "required_weather_receipt_status"
    )
    if required_weather_receipt_status is not None:
        actual_weather_receipt_status = trace.get("weather_receipt_status")
        if (
            not isinstance(actual_weather_receipt_status, str)
            or actual_weather_receipt_status != required_weather_receipt_status
        ):
            raise AssertionError(
                f"{label}: expected weather receipt status "
                f"{required_weather_receipt_status!r}, found "
                f"{actual_weather_receipt_status!r}"
            )

    normalized_response = response.casefold()
    for phrase in expected.get("required_response_phrases", []):
        if str(phrase).casefold() not in normalized_response:
            raise AssertionError(
                f"{label}: response is missing required phrase {phrase!r}"
            )
    for phrase in expected.get("forbidden_response_phrases", []):
        if str(phrase).casefold() in normalized_response:
            raise AssertionError(
                f"{label}: response contains forbidden phrase {phrase!r}"
            )


def _preflight_diagnostic_expectations(
    query_dir: str | Path,
    result_dir: str | Path,
    filenames: Sequence[str],
) -> None:
    """Validate every collected trace and response before optional judging."""

    query_root = Path(query_dir)
    result_root = Path(result_dir)
    for filename in filenames:
        query_data = (
            yaml.safe_load((query_root / filename).read_text(encoding="utf-8")) or {}
        )
        result_data = (
            yaml.safe_load((result_root / filename).read_text(encoding="utf-8")) or {}
        )
        expectations = query_data.get("diagnostic_expectations") or [
            {} for _ in query_data.get("queries", [])
        ]
        result_entries = result_data.get("results") or []
        if len(expectations) != len(result_entries):
            raise AssertionError(
                f"Mismatch in diagnostic expectation counts in {filename}"
            )
        for index, (expected, result) in enumerate(zip(expectations, result_entries)):
            _validate_diagnostic_expectations(
                expected,
                result.get("agent_diagnostics"),
                response=str(result.get("response") or result.get("content") or ""),
                label=f"{filename} turn {index}",
            )


def main() -> int:
    conversation = os.environ["TEST_PATH"]
    result_directory = os.environ.get("RESULT_DIRECTORY", "results")
    query_dir = Path("conversations") / conversation
    result_dir = query_dir / result_directory
    query_files = sorted(path.name for path in query_dir.glob("*.yaml"))
    result_files = sorted(path.name for path in result_dir.glob("*.yaml"))
    if query_files != result_files:
        raise AssertionError("Mismatch between query and result filenames!")
    _preflight_diagnostic_expectations(query_dir, result_dir, query_files)
    print(f"Diagnostic validation passed: {len(query_files)} scenario file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
