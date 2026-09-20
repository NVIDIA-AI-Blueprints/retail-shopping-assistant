# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record what one turn puts on a shopper's screen, per surface.

The journey transcript prints one `shown:` line, which reads as "the products
the shopper saw". It is the `products` event, and the browser renders that in
a side panel. A separate `images` event draws the row of pictures inside the
chat, and the `content` event is the prose. Three surfaces, three different
subsets, and the transcript shows one of them.

Run against a deployed stack:

    python scripts/what_the_shopper_sees.py "show me cream sweaters and boots"
"""

from __future__ import annotations

import json
import sys
import time

import requests

_URL = "http://localhost:8009/query/stream"


def probe(query: str) -> dict[str, object]:
    conversation = f"what-the-shopper-sees-{int(time.time())}"
    payload = {
        "user_id": 90_000_123,
        "session_id": conversation,
        "conversation_id": conversation,
        "query": query,
        "guardrails": False,
    }
    surfaces: dict[str, object] = {"products": [], "images": [], "content": ""}
    with requests.post(_URL, json=payload, timeout=180, stream=True) as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            event = json.loads(line[6:])
            kind, body = event.get("type"), event.get("payload")
            if kind == "products" and isinstance(body, list):
                surfaces["products"] = [
                    str(p.get("productName") or p.get("display_name") or "?")
                    for p in body
                    if isinstance(p, dict)
                ]
            elif kind == "images" and isinstance(body, dict):
                surfaces["images"] = list(body.keys())
            elif kind == "content" and body:
                surfaces["content"] = str(body)
    return surfaces


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "show me cream sweaters and boots"
    surfaces = probe(query)
    products = surfaces["products"]
    images = surfaces["images"]
    prose = str(surfaces["content"])

    assert isinstance(products, list) and isinstance(images, list)
    named = [name for name in products if name and name in prose]

    print(f"query: {query}\n")
    print(f"side panel   ({len(products)}): the `products` event")
    for i, name in enumerate(products, 1):
        print(f"   {i}. {name}")
    print(f"\nchat pictures ({len(images)}): the `images` event")
    for i, name in enumerate(images, 1):
        print(f"   {i}. {name}")
    print(f"\nnamed in the prose ({len(named)} of {len(products)}):")
    for name in named:
        print(f"   - {name}")
    unnamed = [name for name in products if name not in named]
    if unnamed:
        print("\non screen but never mentioned:")
        for name in unnamed:
            print(f"   - {name}")
    print(f"\nsame set on both visual surfaces: {set(products) == set(images)}")
    print(f"same order on both visual surfaces: {products == images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
