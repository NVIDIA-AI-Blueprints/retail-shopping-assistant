#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that AGENTS.md's config-field references still name real fields.

AGENTS.md documents runtime behaviour by naming `ChainServerConfig` fields
(e.g. `` `deepagents_execution_timeout_seconds` ``) instead of hardcoding
their current values, so the doc can't silently drift from a config default
the way a hardcoded number ("45 seconds") could. This script is the guard
for that convention: it extracts every backtick-quoted, config-field-shaped
identifier from AGENTS.md and confirms each one is an actual
`ChainServerConfig` field. It does not (and should not) try to catch every
possible doc/code drift -- just this one, narrow, previously-real case.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD = REPO_ROOT / "AGENTS.md"
CONFIG_PY = REPO_ROOT / "chain_server" / "src" / "config.py"

# Identifiers that look like config fields (snake_case, ends the way our
# knobs do) but are documented as env var names, not ChainServerConfig
# attributes -- e.g. `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS` is checked
# separately since it's upper-cased and not a Python attribute.
_FIELD_PATTERN = re.compile(r"`([a-z][a-z0-9_]*_(?:seconds|turn|chars|tokens))`")


def _chain_server_config_fields() -> set[str]:
    """Field names on ChainServerConfig, read by parsing the source directly.

    Importing chain_server.src.config would pull in its runtime deps
    (pydantic, yaml, shared.model_config), which this check has no other
    reason to require -- CI's lint job doesn't install them. Parsing the AST
    keeps this script dependency-free.
    """

    tree = ast.parse(CONFIG_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ChainServerConfig":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise RuntimeError(f"ChainServerConfig class not found in {CONFIG_PY}")


def main() -> int:
    known_fields = _chain_server_config_fields()
    text = AGENTS_MD.read_text(encoding="utf-8")
    referenced = set(_FIELD_PATTERN.findall(text))
    unknown = sorted(referenced - known_fields)
    if unknown:
        print(
            "AGENTS.md references config fields that no longer exist on "
            f"ChainServerConfig: {unknown}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
