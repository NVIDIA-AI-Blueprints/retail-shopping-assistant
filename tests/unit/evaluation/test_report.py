from pathlib import Path
import sys

import yaml


EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_ROOT))

from src.report import generate_report


def _write_run(path: Path, run_id: str, *, completed: int, errors: int) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "run_id": run_id,
                "started_at": "2026-06-28T00:00:00+00:00",
                "completed_at": f"2026-06-28T00:{completed:02d}:00+00:00",
                "summary": {
                    "scenario_count": completed + errors,
                    "completed_scenarios": completed,
                    "errored_scenarios": errors,
                },
                "judge_summary": {
                    "scenario_count": completed + errors,
                    "pass_count": completed,
                    "fail_count": errors,
                    "average_score": 4 if errors == 0 else 2,
                },
                "scenarios": [
                    {
                        "id": "text_budget_work_bag",
                        "dataset": "text_shopping",
                        "brief": "Budget work bag.",
                        "target_turns": 8,
                        "turns": [],
                        "error": None,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_latest_report_is_clickable_index_without_redirect(tmp_path, monkeypatch):
    monkeypatch.delenv("JUPYTER_SERVER_ROOT", raising=False)
    monkeypatch.delenv("JUPYTER_SERVER_URL", raising=False)
    results = tmp_path / "results"
    first = results / "runs" / "20260628T000001Z" / "run.yaml"
    second = results / "runs" / "20260628T000002Z" / "run.yaml"
    _write_run(first, "20260628T000001Z", completed=1, errors=0)
    _write_run(second, "20260628T000002Z", completed=2, errors=1)

    generate_report(second)

    latest_html = (results / "latest.html").read_text(encoding="utf-8")
    assert "http-equiv=\"refresh\"" not in latest_html
    assert 'href="runs/20260628T000002Z/index.html"' in latest_html
    assert 'href="runs/20260628T000001Z/index.html"' in latest_html
    assert 'href="runs/20260628T000002Z/run.yaml"' in latest_html
    assert "avg 2; pass 2; fail 1" in latest_html
    assert str((results / "runs" / "20260628T000002Z" / "index.html").resolve()) in latest_html
    assert str((results / "runs" / "20260628T000002Z" / "run.yaml").resolve()) in latest_html


def test_run_report_links_back_to_latest_and_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("JUPYTER_SERVER_ROOT", raising=False)
    monkeypatch.delenv("JUPYTER_SERVER_URL", raising=False)
    run_path = tmp_path / "results" / "runs" / "testrun" / "run.yaml"
    _write_run(run_path, "testrun", completed=1, errors=0)

    generate_report(run_path)

    index_html = (run_path.parent / "index.html").read_text(encoding="utf-8")
    assert 'href="../../latest.html"' in index_html
    assert 'href="run.yaml"' in index_html
    assert str((tmp_path / "results" / "latest.html").resolve()) in index_html
    assert str(run_path.resolve()) in index_html
    assert 'href="#text_budget_work_bag"' in index_html
    assert 'id="text_budget_work_bag"' in index_html
