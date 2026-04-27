#!/usr/bin/env python3
"""Run Retail Shopping Assistant unit and integration test suites."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


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
    if env_python:
        return env_python

    candidates = [
        REPO_ROOT / ".local-run" / "dev-venv" / "bin" / "python",
        REPO_ROOT / ".venv-tests" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def print_command(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}\n  cwd: {cwd}", flush=True)


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print_command(cmd, cwd)
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def preflight_integration(args: argparse.Namespace) -> int:
    conversation_dir = INTEGRATION_DIR / "conversations" / args.test_path
    if not conversation_dir.is_dir():
        print(
            f"Missing integration conversation directory: {conversation_dir}\n"
            "Choose an existing directory under tests/integration/conversations/.",
            file=sys.stderr,
        )
        return 2

    if not args.skip_quality and not os.environ.get("NVIDIA_API_KEY"):
        print(
            "NVIDIA_API_KEY is required for response_quality.py. "
            "Set it or rerun with --skip-quality.",
            file=sys.stderr,
        )
        return 2

    url = f"http://{args.host}:{args.port}"
    try:
        with urlopen(url, timeout=3) as response:
            if response.status >= 500:
                print(f"Service preflight returned HTTP {response.status}: {url}", file=sys.stderr)
                return 2
    except HTTPError as exc:
        if exc.code >= 500:
            print(f"Service preflight returned HTTP {exc.code}: {url}", file=sys.stderr)
            return 2
    except URLError as exc:
        print(
            f"Could not reach {url}: {exc.reason}\n"
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
    except URLError as exc:
        print(f"Could not reach {endpoint_url}: {exc.reason}", file=sys.stderr)
        return 2

    return 0


def run_unit(args: argparse.Namespace, python_bin: str, env: dict[str, str]) -> int:
    pytest_args = args.pytest_args or ["-q", "unit"]
    return run([python_bin, "-m", "pytest", *pytest_args], cwd=TESTS_DIR, env=env)


def run_integration(args: argparse.Namespace, python_bin: str, env: dict[str, str]) -> int:
    if not args.no_preflight:
        preflight_status = preflight_integration(args)
        if preflight_status:
            return preflight_status

    integration_env = env.copy()
    integration_env["TEST_PATH"] = args.test_path

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
        ],
        [python_bin, "time_breakdown.py"],
    ]

    if not args.skip_quality:
        stages.insert(1, [python_bin, "response_quality.py"])
        stages.append([python_bin, "quality_plots.py"])

    for stage in stages:
        status = run(stage, cwd=INTEGRATION_DIR, env=integration_env)
        if status:
            return status

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
        "--skip-quality",
        action="store_true",
        help="Skip LLM-as-judge response quality and quality plot stages.",
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
