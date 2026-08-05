# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "skills"
    / "retail-local-runner"
    / "scripts"
    / "local_runner.py"
)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("retail_local_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "service,expects_weather",
    [
        ("memory-retriever", False),
        ("guardrails", False),
        ("catalog-retriever", False),
        ("chain-server", True),
    ],
)
def test_python_service_weather_environment_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service: str,
    expects_weather: bool,
) -> None:
    runner = load_runner_module()
    captured: dict[str, object] = {}
    source_env = {
        "WEATHER_ENABLED": "true",
        "WEATHER_API_KEY": "provider-secret",
        "UNRELATED_SETTING": "preserved",
    }
    spec = {
        "service_dir": tmp_path,
        "module": "src.main:app",
        "port": 8123,
        "pythonpath": [],
    }

    monkeypatch.setattr(runner, "base_env", lambda: dict(source_env))
    monkeypatch.setattr(
        runner,
        "venv_python",
        lambda _service_dir: Path("/test/venv/bin/python"),
    )

    def capture_start_process(
        _service: str,
        _command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        port: int | None = None,
    ) -> None:
        captured.update(cwd=cwd, env=env, port=port)

    monkeypatch.setattr(runner, "start_process", capture_start_process)

    runner.start_python_service(service, spec, skip_install=True)

    process_env = captured["env"]
    assert isinstance(process_env, dict)
    assert process_env["UNRELATED_SETTING"] == "preserved"
    assert ("WEATHER_ENABLED" in process_env) is expects_weather
    assert ("WEATHER_API_KEY" in process_env) is expects_weather
    if expects_weather:
        assert process_env["WEATHER_ENABLED"] == "true"
        assert process_env["WEATHER_API_KEY"] == "provider-secret"


def test_ui_process_does_not_receive_weather_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    captured: dict[str, object] = {}
    monkeypatch.setenv("WEATHER_ENABLED", "true")
    monkeypatch.setenv("WEATHER_API_KEY", "provider-secret")
    monkeypatch.setenv("UNRELATED_SETTING", "preserved")
    monkeypatch.setattr(runner, "ensure_ui_image_assets", lambda: None)

    def capture_start_process(
        _service: str,
        _command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        port: int | None = None,
    ) -> None:
        captured.update(cwd=cwd, env=env, port=port)

    monkeypatch.setattr(runner, "start_process", capture_start_process)

    runner.start_ui(skip_install=True)

    process_env = captured["env"]
    assert isinstance(process_env, dict)
    assert process_env["UNRELATED_SETTING"] == "preserved"
    # Relative, so one forwarded port serves the app and its API. An
    # absolute chain-server URL is resolved by the browser and breaks the
    # moment the browser is not on the machine running the services.
    assert process_env["REACT_APP_API_BASE_URL"] == "/api"
    assert "WEATHER_ENABLED" not in process_env
    assert "WEATHER_API_KEY" not in process_env
