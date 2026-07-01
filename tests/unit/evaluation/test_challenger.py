from dataclasses import replace
from pathlib import Path
import sys

import pytest


EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_ROOT))

from src.challenger import (
    ScenarioContext,
    ShopperTurn,
    TargetAgentClient,
    _parse_challenger_turn,
    _parse_model_mapping,
    load_scenario_contexts,
    run_challenger,
    run_scenario,
)
from src.config import ModelRuntime, chat_completion_options, load_eval_config


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
            "timings": {"total": 0.1},
        }


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
            return {"response": "ok", "images": {}, "timings": {}}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.challenger.requests.post", fake_post)

    TargetAgentClient(config).send_turn(user_id=7, query="hello", image="")

    assert captured["json"]["guardrails"] is False
    assert captured["json"]["image_bool"] is False
