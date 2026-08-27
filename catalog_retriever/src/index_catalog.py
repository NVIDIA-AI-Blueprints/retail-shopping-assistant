# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the catalog index once, as a deployment step.

Indexing is the one operation here that must happen exactly once. Rebuilding
starts by dropping the collection, so two replicas doing it together is not
merely wasted work: one can drop the collection while the other is part way
through filling it, and what is left is a partial index that nothing detects,
because the fingerprint is written row by row as rows are added.

Serving pods therefore never index -- there is no flag and no second path, so
there is no deployment in which a pod might. They check whether the index matches
the catalog they loaded and refuse readiness until it does, which is how they
wait for this without pretending to serve. See `index_is_ready` in main.py.

This is what builds, run once before the pods that read it:

    kind: Job                      # parallelism: 1 -- that is the contract
    command: ["python", "-m", "app.index_catalog"]

It is safe to run when the index is already current: it makes the same
fingerprint check first and does nothing, so it can sit unconditionally in a
deployment pipeline. It is not safe to run two at once, which is the whole
point, and nothing in this process can prevent that -- a lock across replicas
would need somewhere to live. The deployment is what guarantees it.

Locally there is nothing extra to run: `docker compose up` starts the
`catalog-indexer` service, which runs this and exits, and catalog-retriever
waits for it to succeed.
"""

from __future__ import annotations

import logging
import sys


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
