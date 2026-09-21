#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delete a conversation and its cart, keeping a copy of what was deleted.

`POST /conversations/{id}/reset` returns everything it is about to delete and
then deletes it. The service deliberately does not write that snapshot
anywhere: an archive table in the database the reset just cleared is not out of
the database. Putting it on disk is this script's job, and where it goes is the
caller's decision.

    python scripts/reset_a_conversation.py my-conversation-id

Without `--archive-dir` the snapshot goes to `debug_archive/`. Pass
`--no-archive` to throw it away, which is the right call for a scratch session
nobody will ask about.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

DEFAULT_ARCHIVE_DIR = Path("debug_archive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversation_id")
    parser.add_argument("--base", default="http://localhost:8011")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="reset without keeping the snapshot",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    response = requests.post(
        f"{args.base.rstrip('/')}/conversations/{args.conversation_id}/reset",
        timeout=args.timeout,
    )
    response.raise_for_status()
    body = response.json()

    deleted = body["deleted"]
    print(
        f"{args.conversation_id}: deleted {deleted['turns']} turn(s), "
        f"{deleted['events']} event(s), {deleted['cart_lines']} cart line(s)"
        f"{', and the projection' if deleted['projection'] else ''}"
    )

    if args.no_archive:
        return 0
    if not deleted["turns"] and not deleted["cart_lines"]:
        print("nothing was there; no archive written")
        return 0

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = args.archive_dir / f"{args.conversation_id}-{stamp}.json"
    path.write_text(json.dumps(body["archive"], indent=2, default=str))
    print(f"archive: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
