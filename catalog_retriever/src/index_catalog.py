# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the catalog index once, as a deployment step.

Indexing is the one operation here that must happen exactly once. Rebuilding
starts by dropping the collection, so two replicas doing it together is not
merely wasted work: one can drop the collection while the other is part way
through filling it, and what is left is a partial index that nothing detects,
because the fingerprint is written row by row as rows are added.

Serving pods therefore do not index. They check whether the index matches the
catalog they loaded and refuse readiness until it does -- see `index_is_ready`
in main.py. This module is what does the building, run once before the pods that
need it:

    kind: Job                      # or an initContainer on the first rollout
    command: ["python", "-m", "app.index_catalog"]

It is safe to run when the index is already current: it makes the same
fingerprint check first and does nothing. It is not safe to run two of these at
once, which is the whole point -- a Job with parallelism 1 is the contract.

Locally, `docker compose` runs a single replica and CATALOG_INDEX_ON_BOOT
defaults to true, so nothing changes and no separate step is needed.
"""

from __future__ import annotations

import logging
import os
import sys


def index_on_boot() -> bool:
    """Whether a serving pod may build the index itself.

    True suits one replica, which is what docker compose runs, and keeps local
    work a single command. It has to be false for more than one, because
    building starts by dropping the collections.

    Lives here rather than inline in main.py so it can be tested without
    importing main, which builds a retriever and needs a live Milvus.
    """

    return os.environ.get("CATALOG_INDEX_ON_BOOT", "true").strip().lower() not in {
        "false",
        "0",
        "no",
    }


def main() -> int:
    # Imported here rather than at module scope: importing main builds the
    # retriever and loads the snapshot, which is the work this needs, and doing
    # it inside the function keeps `--help`-style imports cheap and the failure
    # attributable to this call.
    from .main import retriever, snapshot

    logging.info(
        "CATALOG INDEXER | %d products, fingerprint %s",
        snapshot.product_count,
        snapshot.fingerprint[:12],
    )
    retriever.sync_snapshot(snapshot, verbose=True)
    logging.info("CATALOG INDEXER | index is current")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
