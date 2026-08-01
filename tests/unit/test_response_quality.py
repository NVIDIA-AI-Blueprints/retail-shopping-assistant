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


def test_judge_prompt_includes_bounded_ref_free_catalog_evidence(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _judge_response()

    response_quality.LLM_CLIENT = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response_quality.judge_test(
        query="Compare the dresses.",
        answer="The lace gown is 90% silk.",
        ideal_answer="Compare only confirmed facts.",
        product_evidence=[
            {
                "product_ref": "internal-ref",
                "product_name": "Intricate Lace Gown",
                "source_tool": "get_product_details_tool",
                "evidence_type": "product_detail",
                "facts": {"composition": "90% silk, 10% spandex"},
            }
        ],
        verbose=False,
    )

    prompt = captured["messages"][1]["content"]
    assert "CURRENT-TURN STRUCTURED CATALOG EVIDENCE" in prompt
    assert "authoritative for product-specific facts" in prompt
    assert "Intricate Lace Gown" in prompt
    assert "90% silk, 10% spandex" in prompt
    assert "internal-ref" not in prompt


def test_diagnostic_expectations_require_exact_product_detail_set(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)
    expectations = {
        "required_skills": ["outfit-styling"],
        "required_tools": [
            "resolve_conversation_products_tool",
            "get_product_details_tool",
        ],
        "forbidden_tools": ["search_catalog_tool"],
        "tool_call_counts": {
            "resolve_conversation_products_tool": 1,
            "get_product_details_tool": 2,
            "search_catalog_tool": 0,
        },
        "required_product_detail_names": [
            "Intricate Lace Gown",
            "Wavy Hem Satin Dress",
        ],
    }
    diagnostics = {
        "skill_files_read": ["/shopper/outfit-styling/SKILL.md"],
        "tool_calls": [
            {
                "tool_name": "resolve_conversation_products_tool",
                "status": "completed",
            },
            {"tool_name": "get_product_details_tool", "status": "completed"},
            {"tool_name": "get_product_details_tool", "status": "completed"},
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
    }

    response_quality._validate_diagnostic_expectations(
        expectations,
        diagnostics,
        label="comparison turn",
    )

    diagnostics["product_evidence"].append(
        {
            "product_name": "Elegant Embroidered Lace Dress",
            "source_tool": "get_product_details_tool",
            "evidence_type": "product_detail",
        }
    )
    with pytest.raises(
        AssertionError,
        match="expected product detail evidence",
    ):
        response_quality._validate_diagnostic_expectations(
            expectations,
            diagnostics,
            label="comparison turn",
        )


def test_diagnostic_expectations_validate_summary_compaction(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)
    expectations = {
        "conversation_summary_compaction": "prepared",
        "conversation_summary_input_projection": "exact",
    }
    diagnostics = {
        "conversation_summary_compaction": "prepared",
        "conversation_summary_input_projection": "exact",
    }

    response_quality._validate_diagnostic_expectations(
        expectations,
        diagnostics,
        label="summary turn",
    )

    diagnostics["conversation_summary_input_projection"] = "bounded_head_tail"
    with pytest.raises(
        AssertionError,
        match="expected conversation summary input projection",
    ):
        response_quality._validate_diagnostic_expectations(
            expectations,
            diagnostics,
            label="summary turn",
        )


def test_diagnostic_expectations_match_structural_tool_argument_subset(monkeypatch):
    response_quality = _load_response_quality(monkeypatch)
    expectations = {
        "tool_call_expectations": [
            {
                "tool_name": "search_catalog_tool",
                "status": "completed",
                "arguments": {
                    "taxonomy": {
                        "category": ["footwear"],
                        "subcategory": ["heels", "flats"],
                    },
                    "required_constraints": {},
                },
            }
        ]
    }
    diagnostics = {
        "tool_calls": [
            {
                "sequence": 2,
                "tool_name": "search_catalog_tool",
                "status": "completed",
                "arguments": {
                    "semantic_query": "polished heels or flats",
                    "requested_product_type": "shoes",
                    "taxonomy": {
                        "category": ["footwear"],
                        "subcategory": ["flats", "heels"],
                    },
                    "required_constraints": {},
                },
            }
        ]
    }

    response_quality._validate_diagnostic_expectations(
        expectations,
        diagnostics,
        label="catalog turn",
    )

    expectations["tool_call_expectations"][0]["arguments"][
        "required_constraints"
    ] = {"primary_color": ["black"]}
    with pytest.raises(AssertionError, match="no tool call matched expectation"):
        response_quality._validate_diagnostic_expectations(
            expectations,
            diagnostics,
            label="catalog turn",
        )


def test_shopping_judge_preflight_requires_exposed_agent_diagnostics(
    monkeypatch,
    tmp_path,
):
    response_quality = _load_response_quality(monkeypatch)
    query_dir = tmp_path / "golden"
    result_dir = tmp_path / "results"
    query_dir.mkdir()
    result_dir.mkdir()
    (query_dir / "conv_0.yaml").write_text(
        "queries:\n- Show me bags.\nanswers:\n- Here are some bags.\n",
        encoding="utf-8",
    )
    (result_dir / "conv_0.yaml").write_text(
        "results:\n"
        "- query: Show me bags.\n"
        "  response: Here are two bags.\n"
        "  agent_diagnostics: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="EXPOSE_AGENT_DIAGNOSTICS=true"):
        response_quality._preflight_diagnostic_expectations(
            str(query_dir),
            str(result_dir),
            ["conv_0.yaml"],
            require_agent_diagnostics=True,
        )
