#!/usr/bin/env python
"""Render what the model was given when it chose skills for one turn.

Rebuilt from the recorded run rather than described, so the context blocks
below are the output of the same formatters the runtime calls. Reads a
journey's raw result and replays the state a chosen turn opened with: the
turns before it, the products those turns showed, and the skills on offer.

    python scripts/show_what_the_model_saw.py RAW_JSON TURN_INDEX > out.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chain_server.src import conversation_memory as memory_module  # noqa: E402
from chain_server.src import conversation_products as products_module  # noqa: E402
from chain_server.src.conversation_memory import (  # noqa: E402
    RecentConversationTurn,
    build_dialogue_context,
)
from chain_server.src.conversation_products import (  # noqa: E402
    format_historical_product_index,
)
from chain_server.src.tool_policy import load_shopper_skill_registry  # noqa: E402


def index_as_of(turns: list[dict], upto: int, set_ids: dict[int, str]) -> list[dict]:
    """The product index a turn opens with: every showing before it."""

    sets = []
    for turn in turns:
        sequence = turn["index"]
        if sequence >= upto:
            break
        products = [
            product
            for product in (turn.get("products") or [])
            if product.get("product_id") and product.get("display_name")
        ]
        if not products:
            continue
        compact = []
        for product in products:
            entry = {
                "ref": product["product_id"],
                "name": product["display_name"],
                "position": product.get("position"),
            }
            if product.get("group"):
                entry["group"] = product["group"]
            if product.get("category"):
                entry["category"] = product["category"]
            sizes = (product.get("attributes") or {}).get("sizes")
            if isinstance(sizes, list) and sizes:
                entry["sizes"] = [str(size) for size in sizes]
            compact.append(entry)
        sets.append(
            {
                "candidate_set_id": set_ids.get(sequence, f"set-turn-{sequence}"),
                "turn_seq": sequence,
                "products": compact,
            }
        )
    return sets


def main() -> None:
    raw_path = Path(sys.argv[1])
    turn_index = int(sys.argv[2])
    run = json.loads(raw_path.read_text())
    turns = run["turns"]
    this_turn = next(turn for turn in turns if turn["index"] == turn_index)

    # Real set ids where the caller supplies them; otherwise a stable stand-in.
    # The id is opaque to the model, so its value changes nothing it reads.
    set_ids = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    set_ids = {int(key): value for key, value in set_ids.items()}

    earlier = [turn for turn in turns if turn["index"] < turn_index]
    recent = [
        RecentConversationTurn(
            sequence=turn["index"],
            shopper_text=turn["said"],
            assistant_text=turn.get("reply") or None,
            status="completed",
        )
        for turn in earlier
        if turn.get("said")
    ]
    _, dialogue = build_dialogue_context(recent)
    product_index = format_historical_product_index(
        index_as_of(turns, turn_index, set_ids)
    )

    out = [
        f"# What the model saw at turn {turn_index}",
        "",
        f"Journey `{raw_path.stem}`, rebuilt from `{raw_path}` with the same",
        "formatters the runtime calls.",
        "",
        f"**The shopper said:** {this_turn.get('said')!r}",
        "",
        f"**It then called:** {', '.join(this_turn.get('tools') or []) or 'nothing'}",
        "",
        f"**Input tokens for the turn:** "
        f"{(this_turn.get('token_usage') or {}).get('input_tokens', 0):,}",
        "",
        "---",
        "",
        "## Lane 1 of 4 — conversation turns (the dialogue window)",
        "",
        "Bounded to the most recent turns and rendered verbatim. This is where",
        "the numbered list the shopper is counting from actually lives.",
        "",
        "```",
        dialogue or "(empty)",
        "```",
        "",
        "## Lane 2 of 4 — product reference index",
        "",
        "The only lane with a full-conversation horizon, and the only one the",
        "runtime reads. Most recently shown first.",
        "",
        "```",
        product_index or "(empty)",
        "```",
        "",
        "## Lane 3 of 4 — active anchors",
        "",
        "```",
        "(reserved and never written -- always [])",
        "```",
        "",
        "## Lane 4 of 4 — effective preferences",
        "",
        "```",
        "(reserved and never written -- always [])",
        "```",
        "",
        "---",
        "",
        "## How the four lanes are organised",
        "",
        "| lane | horizon | written by | read by the runtime | budget |",
        "|---|---|---|---|---|",
        "| conversation turns | last few turns | memory service, each "
        "finalization | yes, verbatim | "
        f"{memory_module._DEFAULT_CONTEXT_MAX_CHARS:,} chars |",
        "| product reference index | whole conversation | rebuilt from "
        "`candidate_set_presented` events | yes | "
        f"{products_module._DEFAULT_INDEX_MAX_CHARS:,} chars in the prompt, "
        "16,384 stored |",
        "| active anchors | - | nothing | no | 50 entries |",
        "| effective preferences | - | nothing | no | 100 entries |",
        "",
        "Two of the four are reserved and never written, so the model's whole",
        "durable memory is the dialogue window plus the product index. The",
        "index is the only lane that outlives the window.",
        "",
        "Carried beside the lanes, from the same turn-start call rather than",
        "from the projection: the cart, the shopper context, the wearer",
        "audience, and the catalog capabilities.",
        "",
        "---",
        "",
        "## The skills it was choosing between",
        "",
    ]

    registry = load_shopper_skill_registry(Path("chain_server/skills"))
    for name, skill in sorted(registry.items()):
        out += [
            f"### `{name}`  ({skill.role})",
            "",
            f"- exclusive group: `{skill.exclusive_group or '-'}`",
            f"- grants: {', '.join(sorted(skill.tools_granted)) or '-'}",
            "",
            "The description the model chooses from:",
            "",
            "```",
            (skill.description or "").strip() or "(no description)",
            "```",
            "",
        ]

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
