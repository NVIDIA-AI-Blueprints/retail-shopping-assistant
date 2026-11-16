# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook 5's caching-off baseline changes the server's caching flags and nothing else."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[2] / "notebook" / "stress_replay.py"
_spec = importlib.util.spec_from_file_location("stress_replay", _MODULE)
stress_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stress_replay)

LOCAL_LLM = [
    "--model=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "--tensor-parallel-size=4",
    "--max-num-seqs=128", "--enable-prefix-caching", "--block-size=64", "--prefix-match-unit=64",
    "--mamba-cache-mode=align",
]
DEPLOY_SH = [
    "--model", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "--tensor-parallel-size", "4",
    "--enable-prefix-caching", "--block-size", "64", "--prefix-match-unit", "64",
    "--mamba-cache-mode", "align", "--chat-template-content-format", "string",
]


def test_caching_off_swaps_the_flag_and_drops_what_only_caching_uses() -> None:
    assert stress_replay._without_cache(LOCAL_LLM) == [
        "--model=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "--tensor-parallel-size=4",
        "--max-num-seqs=128", "--no-enable-prefix-caching",
    ]


def test_caching_off_handles_flags_with_separate_values() -> None:
    assert stress_replay._without_cache(DEPLOY_SH) == [
        "--model", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "--tensor-parallel-size", "4",
        "--no-enable-prefix-caching", "--chat-template-content-format", "string",
    ]


def test_flags_read_in_either_form() -> None:
    assert stress_replay._flag(LOCAL_LLM, "--tensor-parallel-size") == "4"
    assert stress_replay._flag(DEPLOY_SH, "--block-size") == "64"
    assert stress_replay._flag(DEPLOY_SH, "--max-num-seqs") is None


def test_cache_setting_names_runs_as_summary_csv_does() -> None:
    assert stress_replay.cache_setting({"enable_prefix_caching": "True", "block_size": "2112"}) == ("on", "2112")
    assert stress_replay.cache_setting({"enable_prefix_caching": "False", "block_size": None}) == ("off", "na")
    assert stress_replay.cache_setting({"enable_prefix_caching": None, "block_size": None}) == ("unknown", "na")


def test_the_api_key_is_masked_in_recorded_flags() -> None:
    assert stress_replay._masked(["--api-key", "sk-secret", "--port", "8000", "--api-key=sk-other"]) == [
        "--api-key", "***", "--port", "8000", "--api-key=***",
    ]


def _container(name, cmd, ports=None, network="bridge"):
    return {"Name": f"/{name}", "Config": {"Cmd": cmd, "Image": "vllm/vllm-openai:v0.32.0"},
            "NetworkSettings": {"Ports": ports or {}}, "HostConfig": {"NetworkMode": network}}


def test_server_launch_finds_the_container_on_the_port(monkeypatch) -> None:
    containers = [
        _container("local-embedding", ["--port=8000"], {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}),
        _container("local-llm", LOCAL_LLM, {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]}),
    ]
    replies = {"ps": "a b", "inspect": json.dumps(containers)}
    monkeypatch.setattr(stress_replay, "_docker", lambda *args: replies[args[0]])

    launch = stress_replay.server_launch("http://localhost:8000")

    assert launch == {"container": "local-llm", "image": "vllm/vllm-openai:v0.32.0", "flags": LOCAL_LLM}


def test_server_launch_finds_a_host_network_server_by_its_port_flag(monkeypatch) -> None:
    containers = [_container("nemotron35", [*DEPLOY_SH, "--port", "8000", "--api-key", "k"], network="host")]
    replies = {"ps": "a", "inspect": json.dumps(containers)}
    monkeypatch.setattr(stress_replay, "_docker", lambda *args: replies[args[0]])

    launch = stress_replay.server_launch("http://localhost:8000")

    assert launch["container"] == "nemotron35"
    assert launch["flags"][-2:] == ["--api-key", "***"]


def test_server_launch_is_none_without_docker(monkeypatch) -> None:
    def no_docker(*args):
        raise OSError("docker: not found")

    monkeypatch.setattr(stress_replay, "_docker", no_docker)

    assert stress_replay.server_launch("http://localhost:8000") is None
