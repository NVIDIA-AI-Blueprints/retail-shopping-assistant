# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the vision model saw, in a shape a browser can render.

The perception model returns JSON with fixed keys and *unstable types*: the same
key arrives as a string on one turn and a list on the next, and a failed
analysis returns a dict. Observed across three runs of the same prompt:

    constraints_detected   ["color: white", ...]   "None visible; ..."   {}
    occasion               ["bridal shower", ...]  "casual fall outing"
    uncertainties          [two items]             one string

So every field is coerced here rather than trusted, and a reader downstream sees
one shape or nothing at all.

This is a projection for display. It establishes nothing: the analysis is an
observation of the media, never catalog fact, and nothing here is used to filter
or to ground a product claim.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Words too common to prove an item is what a query is chasing.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "for", "in", "of", "or", "the", "to", "with",
        "women", "womens", "woman", "men", "mens", "man", "outfit", "look",
        "style", "similar", "like",
    }
)

#: Enough to show range without turning the panel into a wall.
_MAX_PER_FIELD = 8


def summarize_media_analysis(media_analysis: str) -> dict[str, Any] | None:
    """Return a display projection of one media analysis, or None.

    None rather than an empty dict when there is nothing to show, so a caller
    can tell "no media this turn" from "media analysed and found nothing".
    """

    parsed = _parsed(media_analysis)
    if parsed is None:
        return None

    items = _strings(parsed.get("fashion_items"))
    queries = _strings(parsed.get("search_queries"))
    if not items and not _text(parsed.get("summary")):
        return None

    return {
        "summary": _text(parsed.get("summary")),
        # Everything seen, each carrying how many of the model's own searches
        # chase it. A shopper who says "I like the top" gets a top with several
        # and the jeans beside it with none: the breadth is shown, the focus is
        # evidenced rather than asserted.
        "items": [
            {"label": item, "pursued": _pursued_by(item, queries)}
            for item in items[:_MAX_PER_FIELD]
        ],
        "colors": _strings(parsed.get("colors"))[:_MAX_PER_FIELD],
        "materials": _strings(parsed.get("materials_or_textures"))[:_MAX_PER_FIELD],
        "style": _strings(parsed.get("style_terms"))[:_MAX_PER_FIELD],
        "occasion": _strings(parsed.get("occasion"))[:_MAX_PER_FIELD],
        "queries": queries[:_MAX_PER_FIELD],
    }


def _parsed(media_analysis: str) -> dict[str, Any] | None:
    if not media_analysis:
        return None
    try:
        parsed = json.loads(media_analysis)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _strings(value: Any) -> list[str]:
    """Coerce a field to a list of strings, whatever shape it arrived in."""

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    return []


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _pursued_by(item: str, queries: list[str]) -> int:
    """How many of the model's searches chase this item."""

    words = _significant_words(item)
    if not words:
        return 0
    return sum(1 for query in queries if words & _significant_words(query))


def _significant_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[^\W_]+", text.casefold())
        if word not in _STOPWORDS and len(word) > 2
    }
