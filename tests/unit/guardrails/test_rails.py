"""Unit tests for typed guardrail aggregation and required-check coverage."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDRAILS_SRC = str(REPO_ROOT / "guardrails" / "src")


@pytest.fixture
def rails_module(monkeypatch):
    fake_nemo = ModuleType("nemoguardrails")
    fake_nemo.LLMRails = object
    fake_nemo.RailsConfig = object
    fake_nemo.RailStatus = SimpleNamespace(PASSED=SimpleNamespace(value="passed"), BLOCKED=SimpleNamespace(value="blocked"), MODIFIED=SimpleNamespace(value="modified"))
    fake_nemo.RailType = SimpleNamespace(INPUT="input", OUTPUT="output")
    monkeypatch.setitem(sys.modules, "nemoguardrails", fake_nemo)
    fake_config = ModuleType("config_utils")
    fake_config.apply_model_config = lambda *_args: None
    fake_config.resolve_guardrail_model_config = lambda _role: None
    monkeypatch.setitem(sys.modules, "config_utils", fake_config)
    if GUARDRAILS_SRC not in sys.path:
        sys.path.insert(0, GUARDRAILS_SRC)
    sys.modules.pop("guardrails.src.rails", None)
    return importlib.import_module("guardrails.src.rails")


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["allow", "allow"], "allow"),
        (["allow", "error"], "error"),
        (["error", "block"], "block"),
    ],
)
def test_aggregation_precedence(rails_module, statuses, expected):
    decisions = [rails_module.PolicyDecision(status, "test") for status in statuses]
    assert rails_module.aggregate_decisions(decisions).status == expected


@pytest.mark.asyncio
async def test_mixed_input_requires_text_and_every_attachment_check(rails_module):
    calls = []

    class Text:
        async def check(self, stage, text, conversation):
            calls.append(("text", stage, text))
            assert list(conversation) == []
            return rails_module.PolicyDecision("allow", "text")

    class Media:
        async def check(self, text, attachments, conversation):
            calls.append(("media", text, [item.type for item in attachments]))
            assert conversation == []
            return rails_module.PolicyDecision("block", "multimodal", ["unsafe_media"])

    engine = rails_module.GuardrailEngine(text=Text(), multimodal=Media())
    request = SimpleNamespace(
        stage="input",
        shopper_text="safe words",
        assistant_text="",
        attachments=[SimpleNamespace(type="image"), SimpleNamespace(type="video")],
    )
    decision, _, modalities, model_calls = await engine.check(request)

    assert decision.status == "block"
    assert calls == [
        ("text", "input", "safe words"),
        ("media", "safe words", ["image", "video"]),
    ]
    assert modalities == ["text", "image", "video"]
    assert model_calls == {}


@pytest.mark.asyncio
async def test_media_only_input_still_runs_multimodal_check(rails_module):
    class Text:
        async def check(self, *_args):
            raise AssertionError("empty text must not trigger the text rail")

    class Media:
        async def check(self, text, attachments, conversation):
            assert text == ""
            assert conversation == []
            assert [item.type for item in attachments] == ["video"]
            return rails_module.PolicyDecision("allow", "multimodal")

    engine = rails_module.GuardrailEngine(text=Text(), multimodal=Media())
    decision, _, _, _ = await engine.check(SimpleNamespace(
        stage="input", shopper_text="", assistant_text="",
        attachments=[SimpleNamespace(type="video")],
    ))
    assert decision.status == "allow"


@pytest.mark.asyncio
async def test_modified_text_is_blocked_until_transformation_policy_exists(rails_module):
    class App:
        async def check_async(self, **_kwargs):
            return SimpleNamespace(
                status=SimpleNamespace(value="modified"), rail="content safety"
            )

    evaluator = object.__new__(rails_module.TextRailsEvaluator)
    evaluator._rails = App()
    decision = await evaluator.check("input", "original text")

    assert decision.status == "block"
    assert decision.violated_categories == ["content_safety"]


def test_retail_false_positive_guidance_is_explicit(rails_module):
    prompt = " ".join(
        rails_module._video_safety_prompt("find a swimsuit").split()
    )
    for allowed_case in ("swimwear", "mannequins", "children's apparel"):
        assert allowed_case in prompt
    assert "untrusted shopper data" in prompt


def test_model_categories_are_sanitized_and_bounded(rails_module):
    categories = rails_module._sanitize_categories([
        "unsafe media\nRAW SHOPPER BODY " + ("x" * 200)
    ])
    assert categories == ["policy_violation"]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_environment_rejects_invalid_values(
    rails_module,
    monkeypatch,
    value,
):
    monkeypatch.setenv("TEST_POSITIVE_FLOAT", value)

    with pytest.raises(ValueError, match="finite and positive"):
        rails_module._positive_float_env("TEST_POSITIVE_FLOAT", 1.0)


@pytest.mark.asyncio
async def test_video_is_error_when_judge_does_not_support_full_video(rails_module):
    evaluator = object.__new__(rails_module.MultimodalSafetyEvaluator)
    evaluator._supported_modalities = {"image"}
    decision = await evaluator.check(
        "show me this look",
        [SimpleNamespace(type="video", data="data:video/mp4;base64,AAAA")],
    )

    assert decision.status == "error"
    assert decision.diagnostic_code == "unsupported_video_safety"


@pytest.mark.asyncio
async def test_nemotron_omni_uses_compact_instruct_mode(rails_module):
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"status":"allow","violated_categories":[]}')
            )])

    evaluator = object.__new__(rails_module.MultimodalSafetyEvaluator)
    evaluator._supported_modalities = {"image", "video"}
    evaluator._model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    evaluator._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    evaluator._video_fps = 2.0
    decision = await evaluator.check(
        "find this look",
        [SimpleNamespace(type="video", data="data:video/mp4;base64,AAAA")],
    )

    assert decision.status == "allow"
    assert captured["temperature"] == 0.2
    assert captured["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert captured["extra_body"]["mm_processor_kwargs"] == {
        "use_audio_in_video": True
    }
    assert captured["extra_body"]["media_io_kwargs"] == {
        "video": {"fps": 2.0, "num_frames": -1}
    }


@pytest.mark.asyncio
async def test_current_content_safety_uses_documented_fixed_shape(rails_module):
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="User Safety: unsafe\nSafety Categories: PII/Privacy"
                )
            )])

    evaluator = object.__new__(rails_module.NemotronSafetyEvaluator)
    evaluator._content_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    evaluator._content_model = "nvidia/nemotron-3.5-content-safety"

    decision = await evaluator.check_content("input", "show me this", "data:image/png;base64,AAAA")

    assert decision.status == "block"
    assert decision.violated_categories == ["sensitive_data"]
    assert captured["model"] == "nvidia/nemotron-3.5-content-safety"
    assert captured["extra_body"]["chat_template_kwargs"] == {
        "request_categories": "/categories",
        "enable_thinking": False,
    }
    assert captured["max_tokens"] == 100
    assert captured["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_retail_topic_uses_custom_policy_and_deterministic_category(rails_module):
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="User Safety: unsafe")
            )])

    evaluator = object.__new__(rails_module.NemotronSafetyEvaluator)
    evaluator._topic_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    evaluator._topic_model = "nvidia/nemotron-3.5-content-safety"
    evaluator._topic_policy = "### Policy\nName: Retail"

    decision = await evaluator.check_topic(
        "what about the first one?",
        conversation=[
            {"role": "user", "content": "Show me dresses"},
            {"role": "assistant", "content": "Here are two dresses."},
        ],
    )

    assert decision.status == "block"
    assert decision.violated_categories == ["non_retail"]
    assert captured["extra_body"]["chat_template_kwargs"]["custom_policy"] == (
        "### Policy\nName: Retail"
    )
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert captured["max_tokens"] == 400
    prompt_text = captured["messages"][0]["content"][0]["text"]
    assert "Show me dresses" in prompt_text
    assert "Current shopper request: what about the first one?" in prompt_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_label", "expected_status"),
    [("on-topic ", "allow"), ("off-topic\n", "block")],
)
async def test_dedicated_topic_control_uses_binary_contract(
    rails_module,
    raw_label,
    expected_status,
):
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=raw_label)
            )])

    evaluator = object.__new__(rails_module.NemotronSafetyEvaluator)
    evaluator._topic_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    evaluator._topic_model = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    evaluator._topic_policy = "Only discuss retail shopping."

    decision = await evaluator.check_topic(
        "what about the first one?",
        conversation=[
            {"role": "user", "content": "Show me dresses"},
            {"role": "assistant", "content": "Here are two dresses."},
        ],
    )

    assert decision.status == expected_status
    assert decision.violated_categories == (
        [] if expected_status == "allow" else ["non_retail"]
    )
    assert captured["messages"] == [
        {
            "role": "system",
            "content": (
                "Only discuss retail shopping.\n\n"
                f"{rails_module.TOPIC_CONTROL_OUTPUT_RESTRICTION}"
            ),
        },
        {"role": "user", "content": "Show me dresses"},
        {"role": "assistant", "content": "Here are two dresses."},
        {"role": "user", "content": "what about the first one?"},
    ]
    assert captured["temperature"] == 0
    assert captured["top_p"] == 1
    assert captured["max_tokens"] == 20
    assert "extra_body" not in captured


@pytest.mark.asyncio
async def test_dedicated_topic_control_combines_text_model_and_image_policy(
    rails_module,
):
    calls = []

    class TopicCompletions:
        async def create(self, **kwargs):
            calls.append(("dedicated", kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="on-topic")
            )])

    class ContentCompletions:
        async def create(self, **kwargs):
            calls.append(("custom", kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="User Safety: safe")
            )])

    evaluator = object.__new__(rails_module.NemotronSafetyEvaluator)
    evaluator._topic_client = SimpleNamespace(
        chat=SimpleNamespace(completions=TopicCompletions())
    )
    evaluator._topic_model = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    evaluator._content_client = SimpleNamespace(
        chat=SimpleNamespace(completions=ContentCompletions())
    )
    evaluator._content_model = "nvidia/nemotron-3.5-content-safety"
    evaluator._topic_policy = "Only discuss retail shopping."

    decision = await evaluator.check_topic(
        "find this look",
        "data:image/png;base64,AAAA",
    )

    assert decision.status == "allow"
    assert {kind for kind, _ in calls} == {"dedicated", "custom"}
    custom_call = next(kwargs for kind, kwargs in calls if kind == "custom")
    assert custom_call["extra_body"]["chat_template_kwargs"]["custom_policy"] == (
        "Only discuss retail shopping."
    )
    assert custom_call["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_dedicated_topic_block_wins_if_image_policy_errors(rails_module):
    class TopicCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="off-topic"))]
            )

    class ContentCompletions:
        async def create(self, **_kwargs):
            raise RuntimeError("content endpoint unavailable")

    evaluator = object.__new__(rails_module.NemotronSafetyEvaluator)
    evaluator._topic_client = SimpleNamespace(
        chat=SimpleNamespace(completions=TopicCompletions())
    )
    evaluator._topic_model = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    evaluator._content_client = SimpleNamespace(
        chat=SimpleNamespace(completions=ContentCompletions())
    )
    evaluator._content_model = "nvidia/nemotron-3.5-content-safety"
    evaluator._topic_policy = "Only discuss retail shopping."

    decision = await evaluator.check_topic(
        "configure Kubernetes",
        "data:image/png;base64,AAAA",
    )

    assert decision.status == "block"
    assert decision.violated_categories == ["non_retail"]


@pytest.mark.asyncio
async def test_dedicated_topic_control_error_falls_back_to_custom_policy(
    rails_module,
):
    class TopicCompletions:
        async def create(self, **_kwargs):
            raise RuntimeError("topic endpoint unavailable")

    class ContentCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="User Safety: safe")
            )])

    evaluator = object.__new__(rails_module.NemotronSafetyEvaluator)
    evaluator._topic_client = SimpleNamespace(
        chat=SimpleNamespace(completions=TopicCompletions())
    )
    evaluator._topic_model = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    evaluator._content_client = SimpleNamespace(
        chat=SimpleNamespace(completions=ContentCompletions())
    )
    evaluator._content_model = "nvidia/nemotron-3.5-content-safety"
    evaluator._topic_policy = "Only discuss retail shopping."

    decision = await evaluator.check_topic("show me a dress")

    assert decision.status == "allow"


def test_retail_topic_policy_pins_off_topic_regressions_and_retail_counterexamples(
    rails_module,
):
    policy = rails_module._load_topic_policy(
        str(REPO_ROOT / "shared" / "configs" / "rails")
    )

    for unsafe_example in (
        "What do you think of the most recent WSJ article on AI?",
        "what about politics?",
        "Explain the difference between the United States Senate and House of Representatives.",
    ):
        assert unsafe_example in policy

    for safe_example in (
        "Do you sell campaign-themed T-shirts?",
        "What should I wear to a political fundraiser?",
        "Which jacket should I wear to an AI conference?",
        "What about the first one?",
    ):
        assert safe_example in policy


def test_text_input_guardrails_run_in_parallel():
    config = yaml.safe_load(
        (REPO_ROOT / "shared" / "configs" / "rails" / "config.yml").read_text(
            encoding="utf-8"
        )
    )

    assert config["rails"]["input"]["parallel"] is True
    assert config["rails"]["input"]["flows"] == [
        "content safety check input $model=content_safety",
        "topic safety check input $model=topic_control",
    ]


def test_input_execution_mode_environment_overrides_yaml(
    rails_module,
    monkeypatch,
):
    config = SimpleNamespace(
        rails=SimpleNamespace(input=SimpleNamespace(parallel=True))
    )

    class FakeRails:
        def register_action(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        rails_module,
        "RailsConfig",
        SimpleNamespace(from_path=lambda _path: config),
    )
    monkeypatch.setattr(rails_module, "LLMRails", lambda _config: FakeRails())
    monkeypatch.setenv("GUARDRAILS_INPUT_EXECUTION_MODE", "sequential")

    evaluator = rails_module.TextRailsEvaluator(
        "unused",
        safety=SimpleNamespace(),
    )

    assert config.rails.input.parallel is False
    assert evaluator.input_execution_mode == "sequential"


@pytest.mark.asyncio
async def test_every_image_gets_content_and_topic_checks(rails_module):
    calls = []

    class ImageSafety:
        async def check_content(self, stage, text, image_data):
            calls.append(("content", stage, text, image_data))
            return rails_module.PolicyDecision("allow", "content_safety")

        async def check_topic(self, text, image_data, conversation):
            calls.append(("topic", text, image_data))
            assert list(conversation) == []
            status = "block" if image_data.endswith("BBBB") else "allow"
            categories = ["non_retail"] if status == "block" else []
            return rails_module.PolicyDecision(status, "retail_topic", categories)

    evaluator = object.__new__(rails_module.MultimodalSafetyEvaluator)
    evaluator._supported_modalities = {"image"}
    evaluator._image_safety = ImageSafety()

    decision = await evaluator.check(
        "find this look",
        [
            SimpleNamespace(type="image", data="data:image/png;base64,AAAA"),
            SimpleNamespace(type="image", data="data:image/png;base64,BBBB"),
        ],
    )

    assert decision.status == "block"
    assert decision.violated_categories == ["non_retail"]
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_engine_converts_policy_timeout_to_typed_error(rails_module):
    class SlowText:
        async def check(self, *_args):
            await asyncio.sleep(1)
            return rails_module.PolicyDecision("allow", "text")

    engine = rails_module.GuardrailEngine(
        text=SlowText(), multimodal=SimpleNamespace()
    )
    engine._policy_timeout_seconds = 0.01
    decision, _, _, _ = await engine.check(SimpleNamespace(
        stage="input", shopper_text="hello", assistant_text="", attachments=[]
    ))

    assert decision.status == "error"
    assert decision.diagnostic_code == "policy_timeout"


@pytest.mark.asyncio
async def test_engine_reports_actual_guardrail_model_calls(rails_module):
    class Completions:
        def __init__(self, response):
            self.response = response

        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=self.response),
                    )
                ]
            )

    safety = object.__new__(rails_module.NemotronSafetyEvaluator)
    safety._content_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=Completions("User Safety: safe"),
        )
    )
    safety._content_model = "nvidia/nemotron-3.5-content-safety"
    safety._topic_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions("on-topic"))
    )
    safety._topic_model = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    safety._topic_policy = "Only discuss retail shopping."

    class Text:
        async def check(self, _stage, text, conversation):
            return rails_module.aggregate_decisions(
                await asyncio.gather(
                    safety.check_content("input", text),
                    safety.check_topic(text, conversation=conversation),
                )
            )

    engine = rails_module.GuardrailEngine(
        text=Text(),
        multimodal=SimpleNamespace(),
    )
    decision, _, _, model_calls = await engine.check(
        SimpleNamespace(
            stage="input",
            shopper_text="what about the first one?",
            assistant_text="",
            attachments=[],
            conversation=[
                SimpleNamespace(role="user", content="Show me dresses"),
            ],
        )
    )

    assert decision.status == "allow"
    assert model_calls == {"content_safety": 1, "topic_control": 1}
