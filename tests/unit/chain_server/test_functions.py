# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.functions``.

The parsing fallbacks in this module exist because certain NIM-hosted models
emit tool calls in non-standard formats. These tests lock down both paths
(XML / JSON) and the tool schema surface used by agents.
"""

from __future__ import annotations

import pytest

from chain_server.src.functions import (
    _coerce_value,
    _parse_json_tool_call,
    _parse_xml_tool_call,
    add_to_cart_function,
    bulk_add_to_cart_function,
    bulk_remove_from_cart_function,
    parse_tool_call_fallback,
    remove_from_cart_function,
    retrieval_extraction_function,
    summary_function,
    view_cart_function,
    view_cart_total_function,
)


class TestCoerceValue:
    def test_empty_string_returned_as_is(self) -> None:
        assert _coerce_value("") == ""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("42", 42),
            ("-7", -7),
            ("0", 0),
        ],
    )
    def test_integer_coercion(self, raw: str, expected: int) -> None:
        value = _coerce_value(raw)
        assert value == expected
        assert isinstance(value, int)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("3.14", 3.14),
            ("-0.5", -0.5),
        ],
    )
    def test_float_coercion(self, raw: str, expected: float) -> None:
        value = _coerce_value(raw)
        assert value == pytest.approx(expected)
        assert isinstance(value, float)

    @pytest.mark.parametrize("raw", ["foo", "12abc", "abc12"])
    def test_non_numeric_strings_pass_through(self, raw: str) -> None:
        assert _coerce_value(raw) == raw

    def test_literal_list_parsed_with_ast(self) -> None:
        # NIM XML tool-call emits list args as their Python repr.
        assert _coerce_value("['red', 'blue']") == ["red", "blue"]

    def test_literal_dict_parsed_with_ast(self) -> None:
        assert _coerce_value("{'min_price': 50}") == {"min_price": 50}

    def test_json_list_parsed_when_ast_fails(self) -> None:
        # JSON-style quoting falls through ast to json.
        # ast.literal_eval handles JSON-like double-quoted strings too, but
        # this test documents that both parse paths succeed.
        assert _coerce_value('["red", "blue"]') == ["red", "blue"]

    def test_invalid_bracket_content_returns_original(self) -> None:
        # The raw string is returned unchanged when neither parser succeeds.
        raw = "[this is not valid"
        assert _coerce_value(raw) == raw

    def test_float_without_dot_is_coerced_as_int(self) -> None:
        assert _coerce_value("100") == 100
        assert isinstance(_coerce_value("100"), int)

    def test_whitespace_only_string_returned_as_is(self) -> None:
        # Empty-ish strings skip the numeric conversion paths.
        assert _coerce_value("   ") == "   "


class TestXmlToolCall:
    def test_parses_single_parameter_call(self) -> None:
        xml = (
            "<tool_call><function=add_to_cart>"
            "<parameter=item_name>Silk Dress</parameter>"
            "<parameter=quantity>2</parameter>"
            "</function></tool_call>"
        )

        name, args = _parse_xml_tool_call(xml)

        assert name == "add_to_cart"
        assert args == {"item_name": "Silk Dress", "quantity": 2}

    def test_multiline_parameter_preserved_with_dotall(self) -> None:
        xml = (
            "<function=summarizer>"
            "<parameter=summary>line one\nline two</parameter>"
            "</function>"
        )

        name, args = _parse_xml_tool_call(xml)

        assert name == "summarizer"
        assert args == {"summary": "line one\nline two"}

    def test_returns_none_when_function_tag_missing(self) -> None:
        assert _parse_xml_tool_call("not a tool call") == (None, {})

    def test_float_parameter_coerced(self) -> None:
        xml = (
            "<function=extract_retrieval_inputs>"
            "<parameter=max_price>129.99</parameter>"
            "</function>"
        )

        name, args = _parse_xml_tool_call(xml)

        assert name == "extract_retrieval_inputs"
        assert args == {"max_price": pytest.approx(129.99)}


class TestJsonToolCall:
    def test_parses_standard_arguments_key(self) -> None:
        payload = '{"name": "add_to_cart", "arguments": {"item_name": "Hat", "quantity": 1}}'

        name, args = _parse_json_tool_call(payload)

        assert name == "add_to_cart"
        assert args == {"item_name": "Hat", "quantity": 1}

    def test_parses_parameters_alias(self) -> None:
        payload = '{"name": "view_cart", "parameters": {"note": "ok"}}'

        name, args = _parse_json_tool_call(payload)

        assert name == "view_cart"
        assert args == {"note": "ok"}

    def test_extracts_embedded_json_object_from_prose(self) -> None:
        payload = (
            'Model prelude text {"name": "remove_from_cart", '
            '"arguments": {"item_name": "Bag", "quantity": 1}} trailing prose'
        )

        name, args = _parse_json_tool_call(payload)

        assert name == "remove_from_cart"
        assert args == {"item_name": "Bag", "quantity": 1}

    def test_returns_none_when_no_json_object_present(self) -> None:
        assert _parse_json_tool_call("completely free text") == (None, {})

    def test_returns_none_for_invalid_json(self) -> None:
        assert _parse_json_tool_call('{"name": "foo"') == (None, {})

    def test_returns_none_for_list_payload(self) -> None:
        # The parser must reject non-dict root values even if they parse.
        assert _parse_json_tool_call("[1, 2, 3]") == (None, {})

    def test_returns_none_when_name_missing(self) -> None:
        assert _parse_json_tool_call('{"arguments": {"x": 1}}') == (None, {})

    def test_falls_back_to_empty_args_when_arguments_not_a_dict(self) -> None:
        payload = '{"name": "summarizer", "arguments": "raw-string"}'

        name, args = _parse_json_tool_call(payload)

        assert name == "summarizer"
        assert args == {}


class TestParseToolCallFallback:
    def test_prefers_xml_format_when_present(self) -> None:
        payload = (
            '<function=view_cart_total></function>\n'
            '{"name": "view_cart", "arguments": {}}'
        )

        name, args = parse_tool_call_fallback(payload)

        assert name == "view_cart_total"
        assert args == {}

    def test_falls_back_to_json_when_xml_absent(self) -> None:
        name, args = parse_tool_call_fallback(
            '{"name": "view_cart", "arguments": {}}'
        )

        assert name == "view_cart"
        assert args == {}

    def test_empty_content_returns_none(self) -> None:
        assert parse_tool_call_fallback("") == (None, {})

    def test_none_content_returns_none(self) -> None:
        assert parse_tool_call_fallback(None) == (None, {})  # type: ignore[arg-type]

    def test_unrecognized_content_returns_none(self) -> None:
        assert parse_tool_call_fallback("just regular prose") == (None, {})


class TestToolSpecs:
    """Guards on the tool-call spec shape consumed by the OpenAI client.

    The specs are exported to the LLM verbatim as ``tools=[...]`` in the
    chat.completions.create call. A regression here (missing required keys,
    renamed parameters) would silently break agent routing and are worth
    pinning explicitly.
    """

    @pytest.mark.parametrize(
        "spec",
        [
            add_to_cart_function,
            remove_from_cart_function,
            bulk_add_to_cart_function,
            bulk_remove_from_cart_function,
            view_cart_function,
            view_cart_total_function,
            summary_function,
            retrieval_extraction_function,
        ],
    )
    def test_spec_top_level_shape(self, spec: dict) -> None:
        assert spec["type"] == "function"
        assert "function" in spec
        assert isinstance(spec["function"]["name"], str)
        assert spec["function"]["name"]

    @pytest.mark.parametrize(
        "spec,expected_required",
        [
            (add_to_cart_function, ["item_name", "quantity"]),
            (remove_from_cart_function, ["item_name", "quantity"]),
            (bulk_add_to_cart_function, ["items"]),
            (bulk_remove_from_cart_function, ["items"]),
            (summary_function, ["summary"]),
        ],
    )
    def test_required_parameters_declared(
        self, spec: dict, expected_required: list[str]
    ) -> None:
        required = spec["function"]["parameters"]["required"]
        assert required == expected_required

    def test_view_cart_functions_have_no_parameters(self) -> None:
        assert "parameters" not in view_cart_function["function"]
        assert "parameters" not in view_cart_total_function["function"]

    @pytest.mark.parametrize(
        "spec", [bulk_add_to_cart_function, bulk_remove_from_cart_function]
    )
    def test_bulk_tools_have_array_items_schema(self, spec: dict) -> None:
        # The bulk tools must declare ``items`` as an array of {item_name,
        # quantity} objects so the LLM emits a list rather than a flat dict.
        items = spec["function"]["parameters"]["properties"]["items"]
        assert items["type"] == "array"
        item_obj = items["items"]
        assert item_obj["type"] == "object"
        assert set(item_obj["required"]) == {"item_name", "quantity"}
        assert item_obj["properties"]["item_name"]["type"] == "string"
        assert item_obj["properties"]["quantity"]["type"] == "integer"

    def test_retrieval_extraction_requires_entities_and_categories(self) -> None:
        required = retrieval_extraction_function["function"]["parameters"]["required"]

        assert "search_entities" in required
        assert "category_one" in required
        assert "category_two" in required
        assert "category_three" in required

    def test_retrieval_extraction_price_filters_are_optional(self) -> None:
        properties = retrieval_extraction_function["function"]["parameters"]["properties"]
        required = retrieval_extraction_function["function"]["parameters"]["required"]

        assert "min_price" in properties
        assert "max_price" in properties
        assert "min_price" not in required
        assert "max_price" not in required
