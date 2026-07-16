#!/usr/bin/env python3
"""Run Retail Shopping Assistant unit and integration test suites."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "tests"
INTEGRATION_DIR = TESTS_DIR / "integration"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def select_python(explicit: str | None = None) -> str:
    if explicit:
        return explicit

    env_python = os.environ.get("RETAIL_TEST_PYTHON")
    if _is_concrete_env_value(env_python):
        return env_python

    candidates = [
        REPO_ROOT / ".local-run" / "dev-venv" / "bin" / "python",
        REPO_ROOT / ".venv-tests" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def _is_concrete_env_value(value: str | None) -> bool:
    """Reject unexpanded shell-default expressions read from dotenv files."""

    return bool(value and value.strip() and "$" not in value)


def normalize_test_env(env: dict[str, str]) -> None:
    """Give host-side tests real repository paths when dotenv values are symbolic."""

    defaults = {
        "SHARED_ROOT": str(REPO_ROOT / "shared"),
        "SHARED_CONFIG_ROOT": str(REPO_ROOT / "shared" / "configs"),
    }
    for key, default in defaults.items():
        if not _is_concrete_env_value(env.get(key)):
            env[key] = default


def print_command(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}\n  cwd: {cwd}", flush=True)


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print_command(cmd, cwd)
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label.strip())
    return cleaned.strip(".-") or f"run-{utc_timestamp()}"


def git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def resolve_archive_label(args: argparse.Namespace) -> str:
    if args.archive_label:
        return safe_label(args.archive_label)
    if args.result_directory == "results":
        return "latest"
    return safe_label(args.result_directory)


def resolve_archive_root(args: argparse.Namespace) -> Path:
    if args.archive_root:
        return Path(args.archive_root).expanduser()

    configured = os.environ.get("RETAIL_TEST_ARCHIVE_ROOT")
    if configured:
        return Path(configured).expanduser()

    return INTEGRATION_DIR / "conversations" / args.test_path / "quality_progress"


def copy_artifact_files(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.is_dir():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_dir.iterdir()):
        if child.is_file():
            shutil.copy2(child, target_dir / child.name)


def maybe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def average(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_phase_values(phase_values: dict[str, list[float]]) -> dict[str, float]:
    return {
        phase: round(mean(values), 6)
        for phase, values in sorted(phase_values.items())
        if values
    }


def build_timing_summary(results_dir: Path) -> dict[str, object]:
    per_file: dict[str, dict[str, object]] = {}
    overall_phases: dict[str, list[float]] = {}
    total_turns = 0

    for result_file in sorted(results_dir.glob("*.y*ml")):
        with result_file.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        file_phases: dict[str, list[float]] = {}
        file_turns = 0
        for entry in payload.get("results", []):
            timing = entry.get("timing", {})
            if not isinstance(timing, dict):
                continue

            numeric_timing: dict[str, float] = {}
            for phase, value in timing.items():
                numeric_value = maybe_float(value)
                if numeric_value is not None:
                    numeric_timing[phase] = numeric_value

            total = numeric_timing.get("total")
            chatter = numeric_timing.get("chatter")
            first_token = numeric_timing.get("first_token")
            if total is not None and chatter is not None and first_token is not None:
                numeric_timing["ft_total"] = total - (chatter - first_token)

            if not numeric_timing:
                continue

            total_turns += 1
            file_turns += 1
            for phase, value in numeric_timing.items():
                file_phases.setdefault(phase, []).append(value)
                overall_phases.setdefault(phase, []).append(value)

        if file_phases:
            per_file[result_file.name] = {
                "turns": file_turns,
                "average_total": average(file_phases.get("total", [])),
                "average_ft_total": average(file_phases.get("ft_total", [])),
                "phases": summarize_phase_values(file_phases),
            }

    return {
        "count": total_turns,
        "average_total": average(overall_phases.get("total", [])),
        "average_ft_total": average(overall_phases.get("ft_total", [])),
        "phases": summarize_phase_values(overall_phases),
        "per_file": per_file,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def format_score(value: object) -> str:
    return f"{value:.2f}/5" if isinstance(value, (int, float)) else "n/a"


def format_seconds(value: object) -> str:
    return f"{value:.2f}s" if isinstance(value, (int, float)) else "n/a"


def write_run_summary(run_dir: Path, metadata: dict[str, object], timing: dict[str, object]) -> None:
    quality = read_json(run_dir / "quality" / "quality_summary.json")
    lines = [
        f"# Integration Run: {metadata['label']}",
        "",
        f"- Test path: `{metadata['test_path']}`",
        f"- Result directory: `{metadata['result_directory']}`",
        f"- Git commit: `{metadata['git_sha'] or 'unknown'}`",
        f"- Dirty worktree: `{metadata['git_dirty']}`",
        f"- Created at: `{metadata['created_at']}`",
        f"- Golden snapshot: `{run_dir / 'golden'}`",
        f"- Result snapshot: `{run_dir / 'results'}`",
        f"- Timing JSON: `{run_dir / 'timing_summary.json'}`",
    ]
    timing_plot = run_dir / "results" / "timing_summary.png"
    if timing_plot.exists():
        lines.append(f"- Timing plot: `{timing_plot}`")
    if quality:
        lines.append(f"- Quality summary: `{run_dir / 'quality' / 'quality_summary.json'}`")
        lines.append(f"- Overall quality: {format_score(quality.get('overall_average'))}")
    else:
        lines.append("- Overall quality: n/a")
    lines.append(f"- Average total timing: {format_seconds(timing.get('average_total'))}")
    lines.append("")
    lines.append("## Per Scenario Timing")
    lines.append("")
    lines.append("| Scenario | Turns | Avg total | Avg first-token total |")
    lines.append("| --- | ---: | ---: | ---: |")
    per_file = timing.get("per_file", {})
    if isinstance(per_file, dict):
        for filename, summary in sorted(per_file.items()):
            if not isinstance(summary, dict):
                continue
            lines.append(
                f"| `{filename}` | {summary.get('turns', 0)} | "
                f"{format_seconds(summary.get('average_total'))} | "
                f"{format_seconds(summary.get('average_ft_total'))} |"
            )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_integration_run(args: argparse.Namespace, label: str | None = None) -> Path:
    archive_root = resolve_archive_root(args)
    label = safe_label(label) if label else resolve_archive_label(args)
    run_dir = archive_root / "runs" / label
    conversation_dir = INTEGRATION_DIR / "conversations" / args.test_path
    result_dir = conversation_dir / args.result_directory
    quality_dir = conversation_dir / "quality" / args.result_directory

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    golden_dir = run_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(conversation_dir.glob("*.y*ml")):
        shutil.copy2(source, golden_dir / source.name)

    copy_artifact_files(result_dir, run_dir / "results")
    copy_artifact_files(quality_dir, run_dir / "quality")

    timing_summary = build_timing_summary(run_dir / "results")
    write_json(run_dir / "timing_summary.json", timing_summary)

    metadata = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_path": args.test_path,
        "result_directory": args.result_directory,
        "host": args.host,
        "port": args.port,
        "uri": args.uri,
        "skip_quality": args.skip_quality,
        "disable_guardrails": args.disable_guardrails,
        "request_timeout": args.request_timeout,
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_short_sha": git_value(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--porcelain"])),
    }
    write_json(run_dir / "metadata.json", metadata)
    write_run_summary(run_dir, metadata, timing_summary)
    print(f"Archived local quality/timing artifacts to: {run_dir}", flush=True)
    return run_dir


def has_result_outputs(result_dir: Path) -> bool:
    return result_dir.is_dir() and any(result_dir.glob("*.y*ml"))


def preserve_previous_results(args: argparse.Namespace) -> Path | None:
    if args.result_directory != "results" or resolve_archive_label(args) != "latest":
        return None

    archive_root = resolve_archive_root(args)
    latest_dir = archive_root / "runs" / "latest"
    previous_dir = archive_root / "runs" / "previous"

    if latest_dir.is_dir():
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        shutil.copytree(latest_dir, previous_dir)
        print(f"Preserved previous integration archive from latest: {previous_dir}", flush=True)
        return previous_dir

    result_dir = INTEGRATION_DIR / "conversations" / args.test_path / args.result_directory
    if has_result_outputs(result_dir):
        previous_dir = archive_integration_run(args, label="previous")
        print(f"Preserved previous integration archive from existing results: {previous_dir}", flush=True)
        return previous_dir

    return None


def resolve_archived_run(archive_root: Path, label_or_path: str) -> Path:
    candidate = Path(label_or_path).expanduser()
    if candidate.exists():
        return candidate
    return archive_root / "runs" / safe_label(label_or_path)


def quality_average(run_dir: Path) -> object:
    summary = read_json(run_dir / "quality" / "quality_summary.json")
    return summary.get("overall_average") if summary else None


def timing_average(run_dir: Path) -> object:
    summary = read_json(run_dir / "timing_summary.json")
    return summary.get("average_total") if summary else None


def numeric_delta(current: object, previous: object) -> float | None:
    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
        return float(current) - float(previous)
    return None


def write_comparison_report(archive_root: Path, previous_run: Path, current_run: Path) -> Path:
    previous_timing = read_json(previous_run / "timing_summary.json") or {}
    current_timing = read_json(current_run / "timing_summary.json") or {}
    previous_quality = quality_average(previous_run)
    current_quality = quality_average(current_run)
    previous_total = timing_average(previous_run)
    current_total = timing_average(current_run)
    quality_delta = numeric_delta(current_quality, previous_quality)
    timing_delta = numeric_delta(current_total, previous_total)

    comparisons_dir = archive_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    report_path = comparisons_dir / f"{safe_label(previous_run.name)}__to__{safe_label(current_run.name)}.md"

    lines = [
        f"# Integration Comparison: {previous_run.name} to {current_run.name}",
        "",
        f"- Previous run: `{previous_run}`",
        f"- Current run: `{current_run}`",
        f"- Previous timing JSON: `{previous_run / 'timing_summary.json'}`",
        f"- Current timing JSON: `{current_run / 'timing_summary.json'}`",
    ]
    previous_plot = previous_run / "results" / "timing_summary.png"
    current_plot = current_run / "results" / "timing_summary.png"
    if previous_plot.exists():
        lines.append(f"- Previous timing plot: `{previous_plot}`")
    if current_plot.exists():
        lines.append(f"- Current timing plot: `{current_plot}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Previous | Current | Delta |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Overall quality | {format_score(previous_quality)} | "
                f"{format_score(current_quality)} | "
                f"{quality_delta:+.2f} |"
                if quality_delta is not None
                else f"| Overall quality | {format_score(previous_quality)} | {format_score(current_quality)} | n/a |"
            ),
            (
                f"| Avg total timing | {format_seconds(previous_total)} | "
                f"{format_seconds(current_total)} | {timing_delta:+.2f}s |"
                if timing_delta is not None
                else f"| Avg total timing | {format_seconds(previous_total)} | {format_seconds(current_total)} | n/a |"
            ),
            "",
            "## Per Scenario Timing",
            "",
            "| Scenario | Previous avg total | Current avg total | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    previous_files = previous_timing.get("per_file", {})
    current_files = current_timing.get("per_file", {})
    if isinstance(previous_files, dict) and isinstance(current_files, dict):
        for filename in sorted(set(previous_files) | set(current_files)):
            previous_summary = previous_files.get(filename, {})
            current_summary = current_files.get(filename, {})
            previous_avg = (
                previous_summary.get("average_total")
                if isinstance(previous_summary, dict)
                else None
            )
            current_avg = (
                current_summary.get("average_total")
                if isinstance(current_summary, dict)
                else None
            )
            delta = numeric_delta(current_avg, previous_avg)
            if delta is not None:
                lines.append(
                    f"| `{filename}` | {format_seconds(previous_avg)} | "
                    f"{format_seconds(current_avg)} | {delta:+.2f}s |"
                )
            else:
                lines.append(
                    f"| `{filename}` | {format_seconds(previous_avg)} | "
                    f"{format_seconds(current_avg)} | n/a |"
                )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote local quality/timing comparison to: {report_path}", flush=True)
    return report_path


def progress_entry(run_dir: Path, comparison_path: Path | None = None) -> dict[str, object]:
    metadata = read_json(run_dir / "metadata.json") or {}
    timing = read_json(run_dir / "timing_summary.json") or {}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": metadata.get("label", run_dir.name),
        "git_sha": metadata.get("git_sha", ""),
        "git_short_sha": metadata.get("git_short_sha", ""),
        "git_dirty": metadata.get("git_dirty", ""),
        "test_path": metadata.get("test_path", ""),
        "result_directory": metadata.get("result_directory", ""),
        "overall_quality": quality_average(run_dir),
        "average_total_seconds": timing.get("average_total"),
        "average_first_token_total_seconds": timing.get("average_ft_total"),
        "comparison": str(comparison_path) if comparison_path else "",
        "run_dir": str(run_dir),
    }


def read_progress_entries(progress_path: Path) -> list[dict[str, object]]:
    if not progress_path.is_file():
        return []

    entries: list[dict[str, object]] = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def write_progress_markdown(progress_md: Path, entries: list[dict[str, object]]) -> None:
    lines = [
        "# Integration Quality Progress",
        "",
        "| Created | Label | Commit | Quality | Avg total | Comparison |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for entry in entries:
        comparison = entry.get("comparison") or ""
        comparison_text = f"`{comparison}`" if comparison else ""
        lines.append(
            f"| `{entry.get('created_at', '')}` | `{entry.get('label', '')}` | "
            f"`{entry.get('git_short_sha') or entry.get('git_sha') or 'unknown'}` | "
            f"{format_score(entry.get('overall_quality'))} | "
            f"{format_seconds(entry.get('average_total_seconds'))} | "
            f"{comparison_text} |"
        )

    progress_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_progress(archive_root: Path, run_dir: Path, comparison_path: Path | None = None) -> Path:
    progress_path = archive_root / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    entry = progress_entry(run_dir, comparison_path)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    write_progress_markdown(archive_root / "progress.md", read_progress_entries(progress_path))
    return progress_path


def validate_judge_env() -> int:
    required_judge_env = ["JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY_ENV"]
    missing = [name for name in required_judge_env if not os.environ.get(name)]
    if not missing:
        judge_api_key_env = os.environ["JUDGE_API_KEY_ENV"]
        if not os.environ.get(judge_api_key_env):
            missing.append(judge_api_key_env)
    if missing:
        print(
            "response_quality.py requires explicit judge configuration. "
            f"Missing: {', '.join(missing)}. "
            "Set JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY_ENV, and the "
            "referenced API key env var, or rerun with --skip-quality.",
            file=sys.stderr,
        )
        return 2
    return 0


def preflight_integration(args: argparse.Namespace) -> int:
    conversation_dir = INTEGRATION_DIR / "conversations" / args.test_path
    if not conversation_dir.is_dir():
        print(
            f"Missing integration conversation directory: {conversation_dir}\n"
            "Choose an existing directory under tests/integration/conversations/.",
            file=sys.stderr,
        )
        return 2

    url = f"http://{args.host}:{args.port}"
    health_url = f"{url}/health"
    try:
        with urlopen(health_url, timeout=3) as response:
            if response.status >= 500:
                print(f"Service preflight returned HTTP {response.status}: {health_url}", file=sys.stderr)
                return 2
    except HTTPError as exc:
        if exc.code >= 500:
            print(f"Service preflight returned HTTP {exc.code}: {health_url}", file=sys.stderr)
            return 2
    except (TimeoutError, URLError) as exc:
        print(
            f"Could not reach {health_url}: {getattr(exc, 'reason', exc)}\n"
            "Start the app stack before running integration tests, or use --no-preflight.",
            file=sys.stderr,
        )
        return 2

    endpoint_url = f"{url}/{args.uri.lstrip('/')}"
    try:
        with urlopen(endpoint_url, timeout=3) as response:
            if response.status >= 500:
                print(f"Endpoint preflight returned HTTP {response.status}: {endpoint_url}", file=sys.stderr)
                return 2
    except HTTPError as exc:
        if exc.code == 404:
            print(
                f"Endpoint preflight returned HTTP 404: {endpoint_url}\n"
                "For the local runner, use --port 8009 --uri query/timing.",
                file=sys.stderr,
            )
            return 2
        if exc.code >= 500:
            print(f"Endpoint preflight returned HTTP {exc.code}: {endpoint_url}", file=sys.stderr)
            return 2
    except (TimeoutError, URLError) as exc:
        print(f"Could not reach {endpoint_url}: {getattr(exc, 'reason', exc)}", file=sys.stderr)
        return 2

    return 0


def run_unit(args: argparse.Namespace, python_bin: str, env: dict[str, str]) -> int:
    pytest_args = args.pytest_args or ["-q", "unit"]
    return run([python_bin, "-m", "pytest", *pytest_args], cwd=TESTS_DIR, env=env)


def run_integration(args: argparse.Namespace, python_bin: str, env: dict[str, str]) -> int:
    if args.no_local_archive and args.compare_with:
        print("--compare-with requires local archiving; remove --no-local-archive.", file=sys.stderr)
        return 2

    if not args.skip_quality:
        judge_env_status = validate_judge_env()
        if judge_env_status:
            return judge_env_status

    if not args.no_preflight:
        preflight_status = preflight_integration(args)
        if preflight_status:
            return preflight_status

    previous_run = None
    if not args.no_local_archive and not args.compare_with:
        previous_run = preserve_previous_results(args)

    integration_env = env.copy()
    integration_env["TEST_PATH"] = args.test_path
    integration_env["RESULT_DIRECTORY"] = args.result_directory

    stages: list[list[str]] = [
        [
            python_bin,
            "conversation_collector.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--uri",
            args.uri,
            "--result_directory",
            args.result_directory,
            "--request-timeout",
            str(args.request_timeout),
        ]
        + (["--disable-guardrails"] if args.disable_guardrails else []),
        [python_bin, "time_breakdown.py"],
    ]

    if not args.skip_quality:
        stages.insert(1, [python_bin, "response_quality.py"])
        stages.append([python_bin, "quality_plots.py"])

    for stage in stages:
        status = run(stage, cwd=INTEGRATION_DIR, env=integration_env)
        if status:
            return status

    if not args.no_local_archive:
        run_dir = archive_integration_run(args)
        archive_root = resolve_archive_root(args)
        comparison_path = None
        if args.compare_with:
            previous_run = resolve_archived_run(archive_root, args.compare_with)
            if not previous_run.is_dir():
                print(f"Missing comparison baseline: {previous_run}", file=sys.stderr)
                return 2
        if previous_run:
            comparison_path = write_comparison_report(archive_root, previous_run, run_dir)
        append_progress(archive_root, run_dir, comparison_path)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        nargs="?",
        choices=("unit", "integration", "all"),
        default="all",
        help="Test suite to run. Defaults to all.",
    )
    parser.add_argument("--python", dest="python_bin", help="Python executable to use.")
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to pytest for unit runs. Put this option last.",
    )
    parser.add_argument(
        "--test-path",
        default=os.environ.get("TEST_PATH", "shopping"),
        help="Integration conversation directory under tests/integration/conversations/.",
    )
    parser.add_argument("--host", default="localhost", help="Integration target host.")
    parser.add_argument("--port", type=int, default=8009, help="Integration target port.")
    parser.add_argument("--uri", default="query/timing", help="Integration API URI.")
    parser.add_argument("--result-directory", default="results", help="Integration result folder name.")
    parser.add_argument(
        "--archive-label",
        help=(
            "Local archive label under the configured archive root's runs/ directory. "
            "Defaults to 'latest' for the default result directory, otherwise the result directory name."
        ),
    )
    parser.add_argument(
        "--archive-root",
        default=None,
        help=(
            "Local root for quality/timing archives. Defaults to RETAIL_TEST_ARCHIVE_ROOT, "
            "then tests/integration/conversations/<TEST_PATH>/quality_progress/."
        ),
    )
    parser.add_argument(
        "--no-local-archive",
        action="store_true",
        help="Do not copy ignored integration outputs into the local quality/timing archive.",
    )
    parser.add_argument(
        "--compare-with",
        help="Archived run label or path to compare with after the current run is archived.",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip LLM-as-judge response quality and quality plot stages.",
    )
    parser.add_argument(
        "--disable-guardrails",
        action="store_true",
        help="Send guardrails=false during integration conversations.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120,
        help="Seconds to wait for each integration API request.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip integration service and environment checks.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="When running all, continue to integration even if unit tests fail.",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    python_bin = select_python(args.python_bin)
    env = os.environ.copy()
    normalize_test_env(env)

    statuses: list[int] = []
    if args.suite in {"unit", "all"}:
        status = run_unit(args, python_bin, env)
        statuses.append(status)
        if status and args.suite == "all" and not args.keep_going:
            return status

    if args.suite in {"integration", "all"}:
        statuses.append(run_integration(args, python_bin, env))

    return 1 if any(statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
