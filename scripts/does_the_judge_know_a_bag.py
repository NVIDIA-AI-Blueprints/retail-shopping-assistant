# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure whether the resolver names a garment the shop stocks, and only that.

"Bag" came back as an empty list on six of seven deployed turns, and an empty
list means the shopper is told the shop does not carry bags -- while six
subcategories of them sit in the catalogue.

It reproduces, and the shape of it is narrow. Asked on its own, "bag" resolves
every time. Asked beside "jeans", which this shop really does not stock, it
resolves every time. Asked as one of the three things a wedding shopper wanted
-- dress, shoes, bag -- it comes back empty. "Jewelry" in that same position
never fails, and the prompt names jewelry as a worked example of an umbrella
word while saying nothing about bags.

So the reading is that the model discharges the prompt's insistence that an
empty list is "the right and useful answer" onto whichever word the examples
left unanchored, when no genuinely absent garment is there to take it.

Two things have to hold at once, which is why both are measured on every
variant: a stocked garment must resolve, and an unstocked one must stay empty.
A prompt that fixes bags by never answering empty has moved the bug.

    set -a && . ./.env.nvidia.super35 && set +a
    python scripts/does_the_judge_know_a_bag.py 4
"""

from __future__ import annotations

import json
import os
import sys

import requests
from openai import OpenAI

_CAPABILITIES = "http://localhost:8010/capabilities"

#: The word a batch is being asked about, and whether this shop stocks it.
#: "bag" is the reported failure; "jeans" and "hat" guard the other direction.
_BATCHES: tuple[tuple[list[str], dict[str, bool]], ...] = (
    (["dress", "shoes", "bag"], {"bag": True}),
    (["dress", "shoes", "jeans"], {"jeans": False}),
    (["coat", "bag", "hat"], {"bag": True, "hat": False}),
    # What J01 turn 1 actually asks for.
    (
        ["dress", "shoes", "jewelry", "bag", "sunglasses"],
        {"bag": True, "jewelry": True, "sunglasses": True},
    ),
)


def _catalogue() -> tuple[list[str], dict[str, list[str]], list[str]]:
    payload = requests.get(_CAPABILITIES, timeout=30).json()
    tree = {
        name: sorted(body["subcategories"])
        for name, body in payload["taxonomy"]["categories"].items()
    }
    colours = sorted(
        (value["value"] if isinstance(value, dict) else value)
        for value in (payload["filters"].get("primary_color") or {}).get("values") or []
    )
    return sorted({s for subs in tree.values() for s in subs}), tree, colours


def _head(subcategories: list[str], colours: list[str]) -> list[str]:
    lines = [
        "You are a catalogue vocabulary checker for a clothing shop.",
        "",
        "The catalogue has exactly these subcategories:",
        ", ".join(subcategories),
    ]
    if colours:
        lines += ["", "and exactly these colours:", ", ".join(colours)]
    return lines


def _as_deployed(words: list[str], subs: list[str], colours: list[str], _tree) -> str:
    """The prompt in production today, verbatim."""

    return "\n".join(
        [
            *_head(subs, colours),
            "",
            "TASK A. For each word a shopper asked for, list the subcategories "
            "from the list above that ARE THAT THING.",
            ' - Exact or near-exact naming: "dress" -> dresses, "pumps" -> '
            'heels, "booties" -> boots, "gown" -> dresses.',
            ' - An umbrella word covers several: "shoes" -> flats, heels, '
            'sandals, boots; "tops" -> blouses, camisoles, sweaters; '
            '"jewelry" -> bracelets, earrings, necklaces.',
            " - If this catalogue sells no such thing, return an EMPTY list. "
            "Do NOT reach for the nearest available garment. Jeans are not "
            "skirts; a belt is not a blouse; a coat is not a camisole. An "
            "empty list is the right and useful answer, because the shopper "
            "will be told plainly that the shop does not carry it.",
            "",
            "TASK A words:",
            *[f'  - "{word}"' for word in words],
            "",
            "Reply with JSON and nothing else:",
            '{"a": {"word": ["subcategory"]}, "b": {"word": ["colour"]}}',
        ]
    )


def _without_the_editorial(words, subs, colours, _tree) -> str:
    """The same, with the praise for the empty answer removed.

    Every other line is the deployed prompt. If a stocked garment resolves here
    and an unstocked one still does not, the sentence recommending the empty
    answer is what was answering the question.
    """

    return "\n".join(
        [
            *_head(subs, colours),
            "",
            "TASK A. For each word a shopper asked for, list the subcategories "
            "from the list above that ARE THAT THING.",
            ' - Exact or near-exact naming: "dress" -> dresses, "pumps" -> '
            'heels, "booties" -> boots, "gown" -> dresses.',
            ' - An umbrella word covers several: "shoes" -> flats, heels, '
            'sandals, boots; "tops" -> blouses, camisoles, sweaters; '
            '"jewelry" -> bracelets, earrings, necklaces.',
            " - Never reach for the nearest available garment: jeans are not "
            "skirts, a belt is not a blouse, a coat is not a camisole. Return "
            "an empty list for a word this catalogue sells no form of.",
            "",
            "TASK A words:",
            *[f'  - "{word}"' for word in words],
            "",
            "Reply with JSON and nothing else:",
            '{"a": {"word": ["subcategory"]}, "b": {"word": ["colour"]}}',
        ]
    )


def _with_the_departments(words, subs, colours, tree) -> str:
    """The deployed prompt, plus how the catalogue files its subcategories.

    The examples carry the umbrella cases, and a word absent from them has
    nothing to go on -- "bag" is not there and the departments would say it,
    since `bags` is one. Added rather than substituted: replacing the examples
    with the tree made every answer unreliable, not just this one.
    """

    return "\n".join(
        [
            *_head(subs, colours),
            "",
            "They are filed under departments, which are often what a shopper "
            "says: a word naming a department means every subcategory in it.",
            *[f"  {name}: {', '.join(kinds)}" for name, kinds in sorted(tree.items())],
            "",
            "TASK A. For each word a shopper asked for, list the subcategories "
            "from the list above that ARE THAT THING.",
            ' - Exact or near-exact naming: "dress" -> dresses, "pumps" -> '
            'heels, "booties" -> boots, "gown" -> dresses.',
            ' - An umbrella word covers several: "shoes" -> flats, heels, '
            'sandals, boots; "tops" -> blouses, camisoles, sweaters; '
            '"jewelry" -> bracelets, earrings, necklaces.',
            " - If this catalogue sells no such thing, return an EMPTY list. "
            "Do NOT reach for the nearest available garment. Jeans are not "
            "skirts; a belt is not a blouse; a coat is not a camisole.",
            "",
            "TASK A words:",
            *[f'  - "{word}"' for word in words],
            "",
            "Reply with JSON and nothing else:",
            '{"a": {"word": ["subcategory"]}, "b": {"word": ["colour"]}}',
        ]
    )


def _one_word_at_a_time(words, subs, colours, _tree) -> str:
    """Kept for the record: asked alone, every word resolved correctly.

    Not a candidate. It multiplies one call by the number of roles in the
    turn, and the batch is what keeps the resolver to a single call.
    """

    return _as_deployed(words, subs, colours, _tree)


_VARIANTS = {
    "as deployed": _as_deployed,
    "no editorial": _without_the_editorial,
    "departments": _with_the_departments,
}


def _ask(client: OpenAI, model: str, prompt: str) -> dict[str, list[str]]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1200,
        timeout=120,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content or ""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        answers = (json.loads(content[start : end + 1]) or {}).get("a") or {}
    except ValueError:
        return {}
    return {
        str(word).lower(): [str(value) for value in values]
        for word, values in answers.items()
        if isinstance(values, list)
    }


def main() -> int:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    subs, tree, colours = _catalogue()
    model = os.environ.get("LLM_MODEL") or ""
    base_url = os.environ.get("LLM_BASE_URL")
    if not model or not base_url:
        print("set LLM_MODEL and LLM_BASE_URL (source .env.nvidia.super35)")
        return 2
    client = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY") or "x")

    for label, build in _VARIANTS.items():
        print(f"\n=== {label} ===")
        right = total = 0
        for words, expectations in _BATCHES:
            prompt = build(words, subs, colours, tree)
            scores = {word: 0 for word in expectations}
            for _ in range(runs):
                answers = _ask(client, model, prompt)
                for word, stocked in expectations.items():
                    resolved = bool(answers.get(word))
                    scores[word] += 1 if resolved == stocked else 0
            for word, stocked in expectations.items():
                want = "resolves" if stocked else "stays empty"
                print(f"  {'+'.join(words):<24} {word:<7} {want:<12} "
                      f"{scores[word]}/{runs}")
                right += scores[word]
                total += runs
        print(f"  ---> {right}/{total} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
