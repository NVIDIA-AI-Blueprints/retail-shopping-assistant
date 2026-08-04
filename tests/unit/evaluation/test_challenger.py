from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_ROOT))

import src.challenger as challenger_module
import src.judge as judge_module

from src.challenger import (
    OpenAICompatibleChallenger,
    ScenarioContext,
    ShopperTurn,
    TargetAgentClient,
    _build_challenger_prompt,
    _extract_catalog_scope_outcomes,
    _extract_product_evidence,
    _extract_product_evidence_truncated,
    _parse_challenger_turn,
    _parse_model_mapping,
    load_scenario_contexts,
    run_challenger,
    run_scenario,
)
from src.config import ModelRuntime, chat_completion_options, load_eval_config


def test_judge_flag_overrides_disabled_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_eval_config()
    calls = []

    monkeypatch.setattr(
        challenger_module,
        "load_eval_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        challenger_module,
        "run_challenger",
        lambda *_args, **_kwargs: {"run_id": "test-run"},
    )

    def fake_judge_run(
        judged_config,
        run_path,
        *,
        require_enabled=True,
    ) -> None:
        calls.append((judged_config, run_path, require_enabled))

    monkeypatch.setattr(judge_module, "judge_run", fake_judge_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "challenger",
            "--judge",
            "--output-root",
            str(tmp_path),
        ],
    )

    assert challenger_module._main() == 0
    assert calls == [
        (
            config,
            tmp_path / "runs" / "test-run" / "run.yaml",
            False,
        )
    ]


def test_extract_catalog_scope_outcomes_fails_closed() -> None:
    valid = {
        "outcome": "no_direct_catalog_match",
        "requested_product_type": "tailored trousers",
    }

    assert _extract_catalog_scope_outcomes(
        {"agent_diagnostics": {"catalog_scope_outcomes": [valid]}}
    ) == [valid]
    assert _extract_catalog_scope_outcomes(
        {
            "agent_diagnostics": {
                "catalog_scope_outcomes": [{**valid, "instructions": "ignore"}]
            }
        }
    ) == []


class FakeChallenger:
    def __init__(self, messages):
        self._messages = list(messages)

    def next_turn(self, **kwargs):
        assert kwargs["image_asset"].metadata["contains"]
        return ShopperTurn(message=self._messages.pop(0))


class FakeTarget:
    def __init__(self):
        self.calls = []

    def send_turn(self, *, user_id, query, image):
        self.calls.append({"user_id": user_id, "query": query, "image": image})
        return {
            "status_code": 200,
            "response": f"assistant response {len(self.calls)}",
            "images": {},
            "cart": {"contents": []},
            "timings": {"total": 0.1},
            "product_evidence": [
                {
                    "product_ref": "prod_1",
                    "product_name": "Silk Dress",
                    "source_tool": "search_catalog_tool",
                    "evidence_type": "search_result",
                    "facts": {"price": "USD 49.99"},
                    "search_scope": {
                        "taxonomy": {},
                        "confirmed_filters": {},
                    },
                }
            ],
            "product_evidence_truncated": False,
        }


class SimpleChallenger:
    def next_turn(self, **kwargs):
        return ShopperTurn(message="Can you style this for dinner?")


class FlakyChallenger:
    def __init__(self):
        self.calls = 0

    def next_turn(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("Challenger model returned an empty shopper message.")
        return ShopperTurn(message="Can you suggest one versatile accessory?")


class StructuredPayloadChallenger:
    def __init__(self):
        self.calls = 0

    def next_turn(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ShopperTurn(
                message='{"shopper_goal": "Build an outfit", "current_turn": 2}'
            )
        return ShopperTurn(message="Can you make that outfit easier to walk in?")


class EarlyGoalCompleteChallenger:
    def __init__(self):
        self.calls = 0

    def next_turn(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ShopperTurn(message="", goal_complete=True)
        return ShopperTurn(message="Can you compare the full outfit against one cheaper swap?")


class ExhaustingChallenger:
    def __init__(self):
        self.calls = 0

    def next_turn(self, **kwargs):
        self.calls += 1
        if self.calls <= 2:
            return ShopperTurn(message=f"Question {self.calls}?")
        raise ValueError("Challenger model returned an empty shopper message.")


def test_load_scenario_contexts_resolves_image_sidecar():
    config = load_eval_config()

    contexts = load_scenario_contexts(
        config,
        datasets=["image_shopping"],
        scenario_ids={"image_find_similar_under_budget"},
    )

    assert len(contexts) == 1
    assert contexts[0].image_asset is not None
    assert contexts[0].image_asset.id == "classic_black_patent_leather_purse"
    assert contexts[0].image_asset.file_path.exists()


def test_run_challenger_dry_run_can_select_all_scenarios():
    config = load_eval_config()

    result = run_challenger(
        config,
        datasets=["text_shopping", "image_shopping"],
        scenario_limit_per_dataset=0,
        dry_run=True,
    )

    assert result["scenario_count"] == 20
    assert result["estimated_target_turns"] == 160


def test_run_challenger_dry_run_can_select_one_scenario():
    config = load_eval_config()

    result = run_challenger(
        config,
        datasets=["text_shopping", "image_shopping"],
        scenario_ids={"text_budget_work_bag"},
        scenario_limit_per_dataset=0,
        dry_run=True,
    )

    assert result["scenario_count"] == 1
    assert result["estimated_target_turns"] == 8
    assert result["scenarios"] == [
        {"dataset": "text_shopping", "id": "text_budget_work_bag"}
    ]


def test_style_guide_scenarios_are_selectable_in_dry_run():
    config = load_eval_config()

    result = run_challenger(
        config,
        datasets=["style_guide"],
        scenario_limit_per_dataset=0,
        dry_run=True,
    )

    assert result["scenario_count"] == 8
    assert result["estimated_target_turns"] == 64


def test_eval_model_config_exposes_model_call_timeout():
    config = load_eval_config()

    assert config.challenger_model.timeout_seconds == 45
    assert config.judge_model.timeout_seconds == 60


def test_challenger_client_uses_model_call_timeout(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, timeout):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["timeout"] = timeout

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    OpenAICompatibleChallenger(
        ModelRuntime(
            provider="openai_compatible",
            base_url="http://localhost:8000/v1",
            model="challenger-model",
            api_key=None,
            disable_thinking=True,
            json_mode=True,
            temperature=0.7,
            max_tokens=512,
            timeout_seconds=12,
        )
    )

    assert captured == {
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
        "timeout": 12,
    }


def test_challenger_client_uses_reasoning_content_when_content_is_empty(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            reasoning_content=(
                                '{"message": "Can you help me style this?", '
                                '"goal_complete": false}'
                            ),
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, timeout):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    challenger = OpenAICompatibleChallenger(
        ModelRuntime(
            provider="openai_compatible",
            base_url="http://localhost:8000/v1",
            model="challenger-model",
            api_key=None,
            disable_thinking=False,
            json_mode=True,
            temperature=1.0,
            max_tokens=512,
            timeout_seconds=12,
        )
    )

    turn = challenger.next_turn(
        scenario={"id": "style_case"},
        dataset="style_guide",
        image_asset=None,
        transcript=[],
        turn_number=1,
        target_turns=8,
        min_turns=6,
    )

    assert turn.message == "Can you help me style this?"


def test_challenger_prompt_instructs_turn_sequence_when_present():
    prompt = _build_challenger_prompt(
        scenario={
            "id": "cart_case",
            "shopper_goal": "Seed cart before styling.",
            "turn_sequence": ["Find the dress.", "Add the dress."],
        },
        dataset="style_guide",
        image_asset=None,
        transcript=[],
        turn_number=1,
        target_turns=2,
        min_turns=1,
    )

    assert "turn_sequence" in prompt
    assert "follow the item for the current turn number exactly in order" in prompt
    assert "do not compress setup steps into one message" in prompt
    assert "Before the minimum turn count has been met" in prompt
    assert "never return an empty message" in prompt


def test_run_scenario_preserves_style_metadata(tmp_path):
    config = load_eval_config()
    config = replace(
        config,
        conversation=replace(
            config.conversation,
            default_turns=1,
            min_turns=1,
            max_turns=1,
        ),
    )
    source_context = load_scenario_contexts(
        config,
        datasets=["style_guide"],
        scenario_ids={"style_anchor_product_work_to_evening"},
    )[0]
    scenario = dict(source_context.scenario)
    scenario["target_turns"] = 1
    context = ScenarioContext(
        dataset=source_context.dataset,
        dataset_dir=source_context.dataset_dir,
        scenario=scenario,
        image_asset=None,
    )

    record = run_scenario(
        config=config,
        context=context,
        challenger=SimpleChallenger(),
        target=FakeTarget(),
        run_id="testrun",
        run_dir=tmp_path,
    )

    assert record["entry_mode"] == "anchor_product"
    assert record["secondary_entry_pattern"] == "product_page_anchor"
    assert record["catalog_dependency"]["level"] == "seed_anchor"
    assert record["turns"][0]["target"]["cart"] == {"contents": []}
    assert record["turns"][0]["target"]["product_evidence"][0][
        "product_ref"
    ] == "prod_1"
    assert (
        record["turns"][0]["target"]["product_evidence_truncated"] is False
    )
    assert record["success_criteria"]
    assert record["failure_modes"]


def test_run_scenario_retries_empty_challenger_turn(tmp_path):
    config = load_eval_config()
    config = replace(
        config,
        conversation=replace(config.conversation, default_turns=1, min_turns=1, max_turns=1),
    )
    context = ScenarioContext(
        dataset="style_guide",
        dataset_dir=tmp_path,
        scenario={
            "id": "style_retry_case",
            "brief": "Retry case.",
            "shopper_goal": "Ask a style question.",
            "target_turns": 1,
        },
        image_asset=None,
    )
    challenger = FlakyChallenger()
    target = FakeTarget()

    record = run_scenario(
        config=config,
        context=context,
        challenger=challenger,
        target=target,
        run_id="testrun",
        run_dir=tmp_path,
    )

    assert record["error"] is None
    assert challenger.calls == 2
    assert record["challenger_retry_errors"] == [
        {
            "turn": 1,
            "errors": [
                "attempt 1: Challenger model returned an empty shopper message."
            ],
        }
    ]
    assert record["turns"][0]["shopper"] == "Can you suggest one versatile accessory?"


def test_run_scenario_retries_structured_challenger_payload(tmp_path):
    config = load_eval_config()
    config = replace(
        config,
        conversation=replace(config.conversation, default_turns=1, min_turns=1, max_turns=1),
    )
    context = ScenarioContext(
        dataset="style_guide",
        dataset_dir=tmp_path,
        scenario={
            "id": "style_structured_payload_case",
            "brief": "Structured payload retry case.",
            "shopper_goal": "Ask a style question.",
            "target_turns": 1,
        },
        image_asset=None,
    )
    challenger = StructuredPayloadChallenger()
    target = FakeTarget()

    record = run_scenario(
        config=config,
        context=context,
        challenger=challenger,
        target=target,
        run_id="testrun",
        run_dir=tmp_path,
    )

    assert record["error"] is None
    assert challenger.calls == 2
    assert record["challenger_retry_errors"] == [
        {
            "turn": 1,
            "errors": [
                "attempt 1: Challenger model returned a structured payload "
                "instead of a shopper message."
            ],
        }
    ]
    assert len(target.calls) == 1
    assert target.calls[0]["query"] == "Can you make that outfit easier to walk in?"


def test_run_scenario_retries_goal_complete_before_min_turns(tmp_path):
    config = load_eval_config()
    config = replace(
        config,
        conversation=replace(config.conversation, default_turns=2, min_turns=2, max_turns=2),
    )
    context = ScenarioContext(
        dataset="style_guide",
        dataset_dir=tmp_path,
        scenario={
            "id": "style_early_completion_case",
            "brief": "Early completion case.",
            "shopper_goal": "Build a full outfit and keep the conversation complex.",
            "target_turns": 2,
        },
        image_asset=None,
    )
    challenger = EarlyGoalCompleteChallenger()
    target = FakeTarget()

    record = run_scenario(
        config=config,
        context=context,
        challenger=challenger,
        target=target,
        run_id="testrun",
        run_dir=tmp_path,
    )

    assert record["error"] is None
    assert challenger.calls == 3
    assert record["challenger_retry_errors"] == [
        {
            "turn": 1,
            "errors": [
                "attempt 1: goal_complete returned before minimum turns "
                "with no shopper message (0/2 turns complete)"
            ],
        }
    ]
    assert record["turns"][0]["shopper"] == (
        "Can you compare the full outfit against one cheaper swap?"
    )


def test_run_scenario_stops_after_challenger_exhausts_near_minimum(tmp_path):
    config = load_eval_config()
    config = replace(
        config,
        conversation=replace(config.conversation, default_turns=4, min_turns=3, max_turns=4),
    )
    context = ScenarioContext(
        dataset="style_guide",
        dataset_dir=tmp_path,
        scenario={
            "id": "style_exhaustion_case",
            "brief": "Exhaustion case.",
            "shopper_goal": "Ask enough style questions.",
            "target_turns": 4,
        },
        image_asset=None,
    )

    record = run_scenario(
        config=config,
        context=context,
        challenger=ExhaustingChallenger(),
        target=FakeTarget(),
        run_id="testrun",
        run_dir=tmp_path,
    )

    assert record["error"] is None
    assert record["stopped_reason"] == "challenger_exhausted_after_partial_completion"
    assert len(record["turns"]) == 2
    assert record["challenger_retry_errors"] == [
        {
            "turn": 3,
            "errors": [
                "attempt 1: Challenger model returned an empty shopper message.",
                "attempt 2: Challenger model returned an empty shopper message.",
                "attempt 3: Challenger model returned an empty shopper message.",
            ],
        }
    ]


def test_run_scenario_sends_image_only_on_first_turn(tmp_path):
    config = load_eval_config()
    config = replace(
        config,
        conversation=replace(config.conversation, default_turns=2, min_turns=1, max_turns=2),
    )
    source_context = load_scenario_contexts(
        config,
        datasets=["image_shopping"],
        scenario_ids={"image_find_similar_under_budget"},
    )[0]
    scenario = dict(source_context.scenario)
    scenario["target_turns"] = 2
    context = ScenarioContext(
        dataset=source_context.dataset,
        dataset_dir=source_context.dataset_dir,
        scenario=scenario,
        image_asset=source_context.image_asset,
    )
    target = FakeTarget()

    record = run_scenario(
        config=config,
        context=context,
        challenger=FakeChallenger(["Find this under $60.", "What is the cheapest match?"]),
        target=target,
        run_id="testrun",
        run_dir=tmp_path,
    )

    assert record["error"] is None
    assert len(record["turns"]) == 2
    assert target.calls[0]["image"].startswith("data:image/jpeg;base64,")
    assert target.calls[1]["image"] == ""
    assert (tmp_path / record["input_assets"][0]["copied_to"]).exists()


def test_parse_model_mapping_accepts_first_json_object_with_trailing_text():
    parsed = _parse_model_mapping(
        '{"message": "Show me bags under $60.", "goal_complete": false}\n'
        '{"message": "extra object"}'
    )

    assert parsed == {"message": "Show me bags under $60.", "goal_complete": False}


def test_parse_challenger_turn_accepts_plain_text_message():
    turn = _parse_challenger_turn("I need a polished bag under $60, no exceptions.")

    assert turn == ShopperTurn(
        message="I need a polished bag under $60, no exceptions.",
        goal_complete=False,
    )


def test_parse_challenger_turn_strips_completed_think_block():
    turn = _parse_challenger_turn(
        '<think>I should not send this.</think>\n'
        '{"message": "I need a work bag under $60.", "goal_complete": false}'
    )

    assert turn == ShopperTurn(
        message="I need a work bag under $60.",
        goal_complete=False,
    )


def test_parse_challenger_turn_rejects_thinking_only_response():
    with pytest.raises(ValueError, match="empty shopper message"):
        _parse_challenger_turn("<think>I am still reasoning and got cut off")


def test_parse_challenger_turn_rejects_structured_message_payload():
    with pytest.raises(ValueError) as excinfo:
        _parse_challenger_turn(
            '{"message": "{\\"shopper_goal\\": \\"Build an outfit\\", '
            '\\"current_turn\\": 3}", "goal_complete": false}'
        )

    assert str(excinfo.value) == (
        "Challenger model returned a structured payload instead of a shopper message."
    )


def test_parse_challenger_turn_rejects_metadata_mapping_message():
    with pytest.raises(ValueError) as excinfo:
        _parse_challenger_turn(
            '{"message": "shopper_goal: Build an outfit", "goal_complete": false}'
        )

    assert str(excinfo.value) == (
        "Challenger model returned evaluation metadata instead of a shopper message."
    )


def test_chat_completion_options_can_request_json_and_disable_thinking():
    runtime = ModelRuntime(
        provider="openai_compatible",
        base_url="http://localhost:8000/v1",
        model="local-model",
        api_key=None,
        disable_thinking=True,
        json_mode=True,
        temperature=0.7,
        max_tokens=512,
        timeout_seconds=45,
    )

    assert chat_completion_options(runtime) == {
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_target_agent_client_uses_configured_guardrails(monkeypatch):
    config = load_eval_config()
    config = replace(config, target_agent=replace(config.target_agent, guardrails=False))
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": "ok",
                "images": {},
                "cart": {"contents": [{"item": "Bag", "amount": 1}]},
                "timings": {},
                "agent_diagnostics": {
                    "product_evidence": [
                        {
                            "product_ref": "prod_bag",
                            "product_name": "Structured Bag",
                            "source_tool": "search_catalog_tool",
                            "evidence_type": "search_result",
                            "facts": {"price": "USD 59.00"},
                            "search_scope": {
                                "taxonomy": {"category": ["bags"]},
                                "confirmed_filters": {
                                    "primary_color": ["black", "brown"]
                                },
                            },
                        }
                    ],
                    "product_evidence_truncated": True,
                    "ordered_tool_calls": ["must not be copied"],
                },
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.challenger.requests.post", fake_post)

    result = TargetAgentClient(config).send_turn(user_id=7, query="hello", image="")

    assert captured["json"]["guardrails"] is False
    assert captured["json"]["image_bool"] is False
    assert result["cart"] == {"contents": [{"item": "Bag", "amount": 1}]}
    assert result["product_evidence"] == [
        {
            "product_ref": "prod_bag",
            "product_name": "Structured Bag",
            "source_tool": "search_catalog_tool",
            "evidence_type": "search_result",
            "facts": {"price": "USD 59.00"},
            "search_scope": {
                "taxonomy": {"category": ["bags"]},
                "confirmed_filters": {
                    "primary_color": ["black", "brown"]
                },
            },
        }
    ]
    assert result["product_evidence_truncated"] is True
    assert "ordered_tool_calls" not in result


@pytest.mark.parametrize(
    "diagnostics",
    [
        {},
        {"product_evidence": "not-a-list"},
        {"product_evidence": [{"product_ref": "incomplete"}]},
        {
            "product_evidence": [
                {
                    "product_ref": "prod_1",
                    "product_name": "x" * 501,
                    "source_tool": "search_catalog_tool",
                    "evidence_type": "search_result",
                    "facts": {},
                }
            ]
        },
    ],
)
def test_extract_product_evidence_fails_closed_for_invalid_data(diagnostics):
    assert _extract_product_evidence({"agent_diagnostics": diagnostics}) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", False), (1, False), (None, False)],
)
def test_extract_product_evidence_truncated_requires_boolean(value, expected):
    assert (
        _extract_product_evidence_truncated(
            {
                "agent_diagnostics": {
                    "product_evidence": [],
                    "product_evidence_truncated": value,
                }
            }
        )
        is expected
    )
    assert _extract_product_evidence_truncated({}) is False


def test_extract_product_evidence_truncated_requires_valid_evidence():
    assert (
        _extract_product_evidence_truncated(
            {
                "agent_diagnostics": {
                    "product_evidence": "invalid",
                    "product_evidence_truncated": True,
                }
            }
        )
        is False
    )


def test_extract_product_evidence_enforces_record_and_fact_bounds():
    record = {
        "product_ref": "prod_1",
        "product_name": "Silk Dress",
        "source_tool": "search_catalog_tool",
        "evidence_type": "search_result",
        "facts": {"price": "USD 49.99"},
        "search_scope": {"taxonomy": {}, "confirmed_filters": {}},
    }

    accepted = [dict(record) for _ in range(24)]
    assert _extract_product_evidence(
        {"agent_diagnostics": {"product_evidence": accepted}}
    ) == accepted
    assert _extract_product_evidence(
        {"agent_diagnostics": {"product_evidence": accepted + [dict(record)]}}
    ) == []
    assert _extract_product_evidence(
        {
            "agent_diagnostics": {
                "product_evidence": [
                    {**record, "facts": {str(index): index for index in range(41)}}
                ]
            }
        }
    ) == []
    assert _extract_product_evidence(
        {
            "agent_diagnostics": {
                "product_evidence": [{**record, "evidence_type": []}]
            }
        }
    ) == []
    assert _extract_product_evidence(
        {
            "agent_diagnostics": {
                "product_evidence": [{**record, "unexpected": "field"}]
            }
        }
    ) == []
    assert _extract_product_evidence(
        {
            "agent_diagnostics": {
                "product_evidence": [
                    {
                        **record,
                        "source_tool": "get_product_details_tool",
                    }
                ]
            }
        }
    ) == []


def test_extract_product_evidence_rejects_oversized_aggregate():
    large_facts = {
        f"field-{index}": "v" * 500
        for index in range(40)
    }
    record = {
        "product_ref": "prod_1",
        "product_name": "Silk Dress",
        "source_tool": "get_product_details_tool",
        "evidence_type": "product_detail",
        "facts": large_facts,
    }
    data = {
        "agent_diagnostics": {
            "product_evidence": [record, {**record, "product_ref": "prod_2"}],
            "product_evidence_truncated": True,
        }
    }

    assert _extract_product_evidence(data) == []
    assert _extract_product_evidence_truncated(data) is False


class TestRecordedDiagnostics:
    """What a turn records is the contract a judge adjudicates against.

    Recording three fields meant a failing turn could not be explained
    afterwards -- which skill was active, how many searches it spent, whether a
    call was rejected -- so every diagnosis reproduced the conversation live.
    """

    def _response(self, **diagnostics):
        return {
            "response": "here are some dresses",
            "agent_diagnostics": {
                "product_evidence": [{"product_ref": "p1"}],
                "tool_calls": [
                    {"sequence": 1, "tool_name": "search_catalog_tool"}
                ],
                "skill_files_read": ["/shopper/product-discovery/SKILL.md"],
                "final_termination_reason": "completed",
                **diagnostics,
            },
        }

    def test_configured_fields_are_recorded(self) -> None:
        from src.challenger import _recorded_diagnostics

        recorded = _recorded_diagnostics(
            self._response(),
            ["product_evidence", "tool_calls", "skill_files_read"],
        )

        assert recorded["tool_calls"][0]["tool_name"] == "search_catalog_tool"
        assert recorded["skill_files_read"] == [
            "/shopper/product-discovery/SKILL.md"
        ]
        assert recorded["product_evidence"] == [{"product_ref": "p1"}]

    def test_unconfigured_fields_are_not_recorded(self) -> None:
        """The turn stays a contract, not a dump of whatever was emitted."""

        from src.challenger import _recorded_diagnostics

        recorded = _recorded_diagnostics(self._response(), ["product_evidence"])

        assert set(recorded) == {"product_evidence"}
        assert "tool_calls" not in recorded

    def test_absent_field_is_absent_not_empty(self) -> None:
        """"the runtime sent nothing" must stay distinct from "never asked"."""

        from src.challenger import _recorded_diagnostics

        recorded = _recorded_diagnostics(
            self._response(), ["product_evidence", "catalog_scope_outcomes"]
        )

        assert "catalog_scope_outcomes" not in recorded

    def test_diagnostics_disabled_on_the_target_records_nothing(self) -> None:
        """EXPOSE_AGENT_DIAGNOSTICS=false yields an empty object, not a crash.

        A whole run was once scored against an empty trace, so this path is
        asserted rather than assumed.
        """

        from src.challenger import _recorded_diagnostics

        assert _recorded_diagnostics({"agent_diagnostics": {}}, ["tool_calls"]) == {}
        assert _recorded_diagnostics({}, ["tool_calls"]) == {}

    def test_default_field_list_covers_skills_and_tools(self) -> None:
        from src.config import _DEFAULT_RECORDED_DIAGNOSTICS

        assert "tool_calls" in _DEFAULT_RECORDED_DIAGNOSTICS
        assert "skill_files_read" in _DEFAULT_RECORDED_DIAGNOSTICS
        assert "product_evidence" in _DEFAULT_RECORDED_DIAGNOSTICS
