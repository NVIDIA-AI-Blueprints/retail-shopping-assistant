# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "skills" / "retail-test-runner" / "scripts" / "run_retail_tests.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_retail_tests", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")


def test_archive_integration_run_copies_results_quality_and_timing(monkeypatch, tmp_path):
    runner = load_runner_module()
    integration_dir = tmp_path / "integration"
    conversation_dir = integration_dir / "conversations" / "shopping"
    result_dir = conversation_dir / "results"
    quality_dir = conversation_dir / "quality" / "results"

    write_yaml(
        conversation_dir / "conv_0.yaml",
        {"queries": ["Find a summer outfit"], "answers": ["Reference answer"]},
    )
    write_yaml(
        result_dir / "conv_0.yaml",
        {
            "set_name": "conv_0.yaml",
            "results": [
                {
                    "query": "Find a summer outfit",
                    "response": "Actual answer",
                    "timing": {"total": 2.0, "chatter": 1.5, "first_token": 0.5},
                }
            ],
        },
    )
    (result_dir / "timing_summary.png").write_bytes(b"png")
    write_yaml(
        quality_dir / "conv_0.yaml",
        [{"index": 0, "query": "Find a summer outfit", "score": 4, "justification": "Good"}],
    )
    (quality_dir / "quality_summary.json").write_text(
        '{"overall_average": 4.0, "count": 1}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "INTEGRATION_DIR", integration_dir)

    def fake_git_value(args):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["rev-parse", "--short", "HEAD"]:
            return "abc123"
        if args == ["status", "--porcelain"]:
            return ""
        return ""

    monkeypatch.setattr(runner, "git_value", fake_git_value)

    args = Namespace(
        archive_label="latest",
        archive_root=str(tmp_path / "quality"),
        test_path="shopping",
        result_directory="results",
        host="localhost",
        port=8009,
        uri="query/timing",
        skip_quality=False,
        disable_guardrails=True,
        request_timeout=120,
    )

    run_dir = runner.archive_integration_run(args)

    assert (run_dir / "golden" / "conv_0.yaml").is_file()
    assert (run_dir / "results" / "conv_0.yaml").is_file()
    assert (run_dir / "results" / "timing_summary.png").is_file()
    assert (run_dir / "quality" / "quality_summary.json").is_file()

    timing_summary = runner.read_json(run_dir / "timing_summary.json")
    assert timing_summary["count"] == 1
    assert timing_summary["average_total"] == 2.0
    assert timing_summary["average_ft_total"] == 1.0

    previous_run = tmp_path / "quality" / "runs" / "previous"
    (previous_run / "quality").mkdir(parents=True)
    runner.write_json(
        previous_run / "timing_summary.json",
        {
            "count": 1,
            "average_total": 3.0,
            "per_file": {"conv_0.yaml": {"average_total": 3.0}},
        },
    )
    runner.write_json(
        previous_run / "quality" / "quality_summary.json",
        {"overall_average": 5.0, "count": 1},
    )

    report = runner.write_comparison_report(tmp_path / "quality", previous_run, run_dir)
    report_text = report.read_text(encoding="utf-8")

    assert "Overall quality" in report_text
    assert "+-1.00" not in report_text
    assert "-1.00" in report_text
    assert "Avg total timing" in report_text
    assert "-1.00s" in report_text


def test_default_progress_archive_preserves_previous_latest_and_progress(monkeypatch, tmp_path):
    runner = load_runner_module()
    integration_dir = tmp_path / "integration"
    conversation_dir = integration_dir / "conversations" / "shopping"
    result_dir = conversation_dir / "results"
    quality_dir = conversation_dir / "quality" / "results"

    write_yaml(
        conversation_dir / "conv_0.yaml",
        {"queries": ["Find a summer outfit"], "answers": ["Reference answer"]},
    )
    write_yaml(
        result_dir / "conv_0.yaml",
        {
            "set_name": "conv_0.yaml",
            "results": [
                {
                    "query": "Find a summer outfit",
                    "response": "Previous answer",
                    "timing": {"total": 3.0, "chatter": 2.5, "first_token": 0.5},
                }
            ],
        },
    )
    runner.write_json(
        quality_dir / "quality_summary.json",
        {"overall_average": 3.0, "count": 1},
    )

    monkeypatch.setattr(runner, "INTEGRATION_DIR", integration_dir)
    monkeypatch.setattr(runner, "git_value", lambda args: "abc123" if args[:1] == ["rev-parse"] else "")

    args = Namespace(
        archive_label=None,
        archive_root=None,
        test_path="shopping",
        result_directory="results",
        host="localhost",
        port=8009,
        uri="query/timing",
        skip_quality=False,
        disable_guardrails=False,
        request_timeout=120,
    )

    previous_run = runner.preserve_previous_results(args)

    assert previous_run == conversation_dir / "quality_progress" / "runs" / "previous"
    assert (previous_run / "results" / "conv_0.yaml").is_file()
    assert runner.read_json(previous_run / "timing_summary.json")["average_total"] == 3.0

    write_yaml(
        result_dir / "conv_0.yaml",
        {
            "set_name": "conv_0.yaml",
            "results": [
                {
                    "query": "Find a summer outfit",
                    "response": "Latest answer",
                    "timing": {"total": 2.0, "chatter": 1.5, "first_token": 0.5},
                }
            ],
        },
    )
    runner.write_json(
        quality_dir / "quality_summary.json",
        {"overall_average": 4.0, "count": 1},
    )

    latest_run = runner.archive_integration_run(args)
    comparison = runner.write_comparison_report(runner.resolve_archive_root(args), previous_run, latest_run)
    progress_path = runner.append_progress(runner.resolve_archive_root(args), latest_run, comparison)

    assert latest_run == conversation_dir / "quality_progress" / "runs" / "latest"
    assert comparison == conversation_dir / "quality_progress" / "comparisons" / "previous__to__latest.md"
    assert progress_path == conversation_dir / "quality_progress" / "progress.jsonl"
    assert (conversation_dir / "quality_progress" / "progress.md").is_file()

    comparison_text = comparison.read_text(encoding="utf-8")
    assert "4.00/5" in comparison_text
    assert "+1.00" in comparison_text
    assert "-1.00s" in comparison_text
