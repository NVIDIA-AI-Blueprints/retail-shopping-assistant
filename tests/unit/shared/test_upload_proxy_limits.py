# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re

import yaml

from tests.conftest import REPO_ROOT


def test_nginx_body_limit_covers_base64_expanded_video_upload() -> None:
    chain_config = yaml.safe_load(
        (REPO_ROOT / "shared/configs/chain_server/config.yaml").read_text()
    )
    nginx_config = (REPO_ROOT / "nginx.conf").read_text()

    raw_video_bytes = chain_config["media_input"]["max_video_bytes"]
    body_limit_bytes = _parse_nginx_size(
        re.search(r"client_max_body_size\s+([^;]+);", nginx_config).group(1)
    )

    base64_video_bytes = math.ceil(raw_video_bytes / 3) * 4
    request_overhead_bytes = 1024 * 1024

    assert body_limit_bytes >= base64_video_bytes + request_overhead_bytes


def test_nginx_api_timeout_allows_slow_media_turns() -> None:
    nginx_config = (REPO_ROOT / "nginx.conf").read_text()

    read_timeout = _parse_nginx_duration(
        re.search(r"proxy_read_timeout\s+([^;]+);", nginx_config).group(1)
    )
    send_timeout = _parse_nginx_duration(
        re.search(r"proxy_send_timeout\s+([^;]+);", nginx_config).group(1)
    )

    assert read_timeout >= 300
    assert send_timeout >= 300


def _parse_nginx_duration(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)([smhSMH]?)\s*", value)
    assert match is not None

    amount = int(match.group(1))
    suffix = match.group(2).lower()
    multipliers = {
        "": 1,
        "s": 1,
        "m": 60,
        "h": 60 * 60,
    }
    return amount * multipliers[suffix]


def _parse_nginx_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)([kKmMgG]?)\s*", value)
    assert match is not None

    amount = int(match.group(1))
    suffix = match.group(2).lower()
    multipliers = {
        "": 1,
        "k": 1024,
        "m": 1024 * 1024,
        "g": 1024 * 1024 * 1024,
    }
    return amount * multipliers[suffix]
