# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook 5's replay checks tell the app's own empty replies from failed calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[2] / "notebook" / "stress_replay.py"
_spec = importlib.util.spec_from_file_location("stress_replay", _MODULE)
stress_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stress_replay)

AGENT = {"temperature": 0.7, "parallel_tool_calls": False}
EDITOR = {"temperature": 0.7}
EMPTY = {"type": "InvalidInferenceResultError", "message": "No responses with actual content"}


def _record(turn: int, *, length: int | None = None, error: dict | None = None) -> dict:
    record = {
        "metadata": {"conversation_id": "s", "turn_index": turn, "request_start_ns": turn},
        "metrics": {"output_sequence_length": {"value": length}} if length else {},
    }
    if error:
        record["error"] = error
    return record


def _gates(tmp_path: Path, trace: list[dict], records: list[dict]) -> dict:
    (tmp_path / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return stress_replay.gates(tmp_path, trace, [])


def test_empty_reply_to_an_agent_call_with_no_tools_left_is_not_a_failure(tmp_path: Path) -> None:
    trace = [
        {"session_id": "s", "extra": AGENT, "tools": [{"type": "function"}], "output_length": 5},
        {"session_id": "s", "extra": AGENT, "output_length": 100},
    ]
    gates = _gates(tmp_path, trace, [_record(0, length=5), _record(1, error=EMPTY)])
    assert gates["complete"] and gates["output_lengths"]
    assert (gates["empty_replies"], gates["other_errors"]) == (1, 0)


def test_other_failed_calls_count_as_errors(tmp_path: Path) -> None:
    trace = [
        {"session_id": "s", "extra": EDITOR, "output_length": 50},
        {"session_id": "s", "extra": AGENT, "output_length": 100},
    ]
    timeout = {"type": "ReadTimeout", "message": "timed out"}
    gates = _gates(tmp_path, trace, [_record(0, error=EMPTY), _record(1, error=timeout)])
    assert (gates["empty_replies"], gates["other_errors"]) == (0, 2)


def test_short_reply_fails_output_lengths(tmp_path: Path) -> None:
    trace = [{"session_id": "s", "extra": EDITOR, "output_length": 50}]
    assert not _gates(tmp_path, trace, [_record(0, length=49)])["output_lengths"]
