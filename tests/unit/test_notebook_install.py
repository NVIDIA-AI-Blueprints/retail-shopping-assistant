# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The notebooks install their packages into the kernel, even without pip.

Notebook 3 used `%pip install`. On a Brev machine the kernel ran from a uv
environment with no pip, so the magic printed an error and carried on, and the
notebook failed later at `import nemo_evaluator`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_HELPERS = Path(__file__).resolve().parents[2] / "notebook" / "helpers.py"
_spec = importlib.util.spec_from_file_location("notebook_helpers", _HELPERS)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


@pytest.fixture
def ran(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(helpers.subprocess, "run", run)
    return commands


def _environment(monkeypatch, *, pip: bool, uv: str | None, uv_on_path: bool = True) -> None:
    monkeypatch.setattr(helpers.importlib.util, "find_spec", lambda name: object() if pip else None)
    monkeypatch.setattr(
        helpers.shutil, "which", lambda name, path=None: uv if (path is None) == uv_on_path else None
    )


def test_pip_installs_into_the_kernel(monkeypatch, ran) -> None:
    _environment(monkeypatch, pip=True, uv="/usr/bin/uv")

    helpers.install("nemo-evaluator==0.3.0")

    assert ran == [[sys.executable, "-m", "pip", "install", "--quiet", "nemo-evaluator==0.3.0"]]


def test_without_pip_uv_installs_into_the_kernel(monkeypatch, ran) -> None:
    _environment(monkeypatch, pip=False, uv="/usr/bin/uv")

    helpers.install("requests", "openai")

    assert ran == [
        ["/usr/bin/uv", "pip", "install", "--quiet", "--python", sys.executable, "requests", "openai"]
    ]


def test_uv_is_found_in_local_bin_when_jupyter_started_without_it(monkeypatch, ran) -> None:
    _environment(monkeypatch, pip=False, uv="/home/ubuntu/.local/bin/uv", uv_on_path=False)

    helpers.install("openai")

    assert ran[0][0] == "/home/ubuntu/.local/bin/uv"


def test_without_pip_or_uv_it_says_so(monkeypatch, ran) -> None:
    _environment(monkeypatch, pip=False, uv=None)

    with pytest.raises(RuntimeError, match="has no pip, and uv is not installed"):
        helpers.install("openai")
    assert ran == []


def test_a_failed_install_raises_with_its_error(monkeypatch) -> None:
    _environment(monkeypatch, pip=True, uv=None)
    monkeypatch.setattr(
        helpers.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "No matching distribution"),
    )

    with pytest.raises(RuntimeError, match="No matching distribution"):
        helpers.install("nemo-evaluator==99")
