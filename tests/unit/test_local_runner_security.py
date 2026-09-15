# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "skills" / "retail-local-runner" / "scripts" / "local_runner.py"
)


@pytest.fixture()
def local_runner():
    spec = importlib.util.spec_from_file_location("retail_local_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_service_binds_to_loopback(local_runner, monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(local_runner, "venv_python", lambda _path: Path("/venv/python"))
    monkeypatch.setattr(
        local_runner,
        "start_process",
        lambda name, command, **kwargs: captured.update(
            name=name, command=command, kwargs=kwargs
        ),
    )

    local_runner.start_python_service(
        "example",
        {
            "service_dir": REPO_ROOT,
            "module": "src.main:app",
            "port": 8123,
            "pythonpath": [REPO_ROOT],
        },
        skip_install=True,
    )

    command = captured["command"]
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_ui_binds_to_loopback(local_runner, monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(local_runner, "ensure_ui_image_assets", lambda: None)
    monkeypatch.setattr(
        local_runner,
        "start_process",
        lambda name, command, **kwargs: captured.update(
            name=name, command=command, kwargs=kwargs
        ),
    )

    local_runner.start_ui(skip_install=True)

    env = captured["kwargs"]["env"]
    assert env["HOST"] == "127.0.0.1"
