from pathlib import Path
import sys

import yaml


EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_ROOT))

from src.config import ConfigError, ModelRuntime, load_eval_config
from src.judge import _parse_model_mapping, _resolve_run_path, judge_run


class FakeJudge:
    def judge_scenario(self, *, rules, run, scenario):
        assert "Critical Failures" in rules
        assert run["run_id"] == "testrun"
        assert scenario["turns"][0]["shopper"] == "Find this under $60."
        return {
            "score": 4,
            "pass": True,
            "reason": "The assistant stayed grounded and respected the budget.",
            "criteria": {"goal_completion": 4},
            "critical_failures": [],
        }


def test_judge_run_appends_scores_and_regenerates_report(tmp_path):
    run_dir = tmp_path / "results" / "runs" / "testrun"
    run_dir.mkdir(parents=True)
    run_path = run_dir / "run.yaml"
    run_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "run_id": "testrun",
                "started_at": "2026-06-27T00:00:00+00:00",
                "completed_at": "2026-06-27T00:01:00+00:00",
                "summary": {"scenario_count": 1},
                "scenarios": [
                    {
                        "id": "case_1",
                        "dataset": "text_shopping",
                        "brief": "Budget search.",
                        "shopper_goal": "Find a bag under $60.",
                        "constraints": ["Budget: $60"],
                        "turns": [
                            {
                                "turn": 1,
                                "shopper": "Find this under $60.",
                                "image_sent": False,
                                "target": {
                                    "response": "Here is a grounded option.",
                                    "images": {},
                                },
                            }
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    judged = judge_run(load_eval_config(), run_path, judge=FakeJudge())

    assert judged["judge_summary"]["pass_count"] == 1
    assert judged["scenarios"][0]["judge"]["score"] == 4
    assert (run_dir / "index.html").exists()
    assert (tmp_path / "results" / "latest.txt").read_text(encoding="utf-8") == "testrun"


def test_resolve_run_path_can_use_latest(tmp_path):
    config = load_eval_config()
    config = config.__class__(
        version=config.version,
        root=tmp_path / "evaluation",
        config_path=config.config_path,
        challenger_model=config.challenger_model,
        judge_model=config.judge_model,
        target_agent=config.target_agent,
        run=config.run,
        conversation=config.conversation,
    )
    latest_path = config.root / "results" / "latest.txt"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text("testrun", encoding="utf-8")

    assert _resolve_run_path(config, None, latest=True) == (
        config.root / "results" / "runs" / "testrun" / "run.yaml"
    )


def test_judge_run_requires_enabled_for_live_judge(tmp_path):
    run_path = tmp_path / "results" / "runs" / "testrun" / "run.yaml"
    _write_minimal_run(run_path)

    try:
        judge_run(load_eval_config(), run_path)
    except ConfigError as exc:
        assert "judge_model.enabled is false" in str(exc)
    else:
        raise AssertionError("judge_run should require enabled judge config")


def test_judge_run_can_bypass_enabled_for_one_command(tmp_path, monkeypatch):
    run_path = tmp_path / "results" / "runs" / "testrun" / "run.yaml"
    _write_minimal_run(run_path)

    class StubLiveJudge:
        def __init__(self, runtime):
            assert runtime.model == "judge-model"

        def judge_scenario(self, *, rules, run, scenario):
            return {
                "score": 5,
                "pass": True,
                "reason": "Good.",
                "criteria": {},
                "critical_failures": [],
            }

    monkeypatch.setattr(
        "src.judge.resolve_model_runtime",
        lambda model_config, require=True: ModelRuntime(
            provider="openai_compatible",
            base_url="http://localhost:8000/v1",
            model="judge-model",
            api_key=None,
            disable_thinking=True,
            json_mode=True,
            temperature=0.0,
            max_tokens=768,
        ),
    )
    monkeypatch.setattr("src.judge.OpenAICompatibleJudge", StubLiveJudge)

    judged = judge_run(load_eval_config(), run_path, require_enabled=False)

    assert judged["judge_summary"]["pass_count"] == 1
    assert judged["scenarios"][0]["judge"]["score"] == 5


def test_parse_model_mapping_accepts_first_json_object_with_trailing_text():
    parsed = _parse_model_mapping('{"score": 4, "pass": true}\n{"score": 1}')

    assert parsed == {"score": 4, "pass": True}


def _write_minimal_run(run_path: Path) -> None:
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "run_id": "testrun",
                "started_at": "2026-06-27T00:00:00+00:00",
                "completed_at": "2026-06-27T00:01:00+00:00",
                "summary": {"scenario_count": 1},
                "scenarios": [
                    {
                        "id": "case_1",
                        "dataset": "text_shopping",
                        "brief": "Budget search.",
                        "turns": [
                            {
                                "turn": 1,
                                "shopper": "Find this under $60.",
                                "target": {"response": "No matching products."},
                            }
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
