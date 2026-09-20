#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report ruff findings on the lines a branch actually wrote.

The repo carries hundreds of pre-existing ruff findings, so CI lints what a
branch changed rather than the whole tree. Scoping that to changed *files*
worked until a branch moved code: splitting one module into seven marks
every line of all seven as changed, and the branch inherits every latent
finding in code it only relocated. One such split arrived carrying 156
findings it did not write, against 20 it did.

Scoping to changed lines is the same intent, applied to the unit the author
is answerable for. A line this branch added or edited is its responsibility;
a line it moved between files is not, and neither is one it never touched.

Deliberately not caught: a finding whose cause is a changed line but whose
location is not. Deleting the last use of an import reports the unused
import at the import, which this passes over. That is the price of the
scoping, and it is the same price the changed-files version paid for every
file a branch did not open at all.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

#: The hunk header names the lines as they are after the change: `+start,count`,
#: where a missing count means one line. Deletions get count 0 and contribute
#: no lines to blame, which is what we want -- nothing is left to report on.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _changed_lines(base: str) -> dict[str, set[int]]:
    """Which lines of which Python files this branch added or edited."""

    diff = subprocess.run(
        # No context lines, so a hunk covers only what changed and not the
        # untouched lines either side of it.
        ["git", "diff", "-U0", "--diff-filter=ACMR", f"{base}...HEAD", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    lines_by_file: dict[str, set[int]] = {}
    current: set[int] | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = lines_by_file.setdefault(line[len("+++ b/") :], set())
            continue
        if current is None:
            continue
        hunk = _HUNK.match(line)
        if hunk:
            start = int(hunk.group(1))
            count = 1 if hunk.group(2) is None else int(hunk.group(2))
            current.update(range(start, start + count))
    return {path: lines for path, lines in lines_by_file.items() if lines}


def _findings(paths: list[str]) -> list[dict]:
    """Every ruff finding in these files, wherever it sits."""

    result = subprocess.run(
        ["ruff", "check", "--output-format", "json", *paths],
        capture_output=True,
        text=True,
    )
    # Ruff exits non-zero when it finds something, which is not a failure to
    # run. An unparseable stdout is, and says so with ruff's own stderr.
    try:
        return json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise SystemExit(2) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="the commit this branch is measured against")
    base = parser.parse_args().base

    changed = _changed_lines(base)
    if not changed:
        print("No Python lines changed.")
        return 0

    reported = [
        finding
        for finding in _findings(sorted(changed))
        if (finding.get("location") or {}).get("row")
        in changed.get(finding.get("filename", "").removeprefix(f"{_cwd()}/"), set())
    ]
    for finding in reported:
        location = finding["location"]
        print(
            f"{finding['filename']}:{location['row']}:{location['column']}: "
            f"{finding['code']} {finding['message']}"
        )

    print(
        f"\n{len(reported)} finding(s) on changed lines "
        f"across {len(changed)} changed file(s)."
    )
    return 1 if reported else 0


def _cwd() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
