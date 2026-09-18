# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One vocabulary question per turn, asked away from the shopper.

Two questions are asked here and they are the same kind of question: does a word
the shopper used correspond to something this catalogue sells, and if so which
advertised values.

Both were previously answered inside the main agent call, and both were answered
wrong in the same way. Asked to scope a search for "jeans" in a shop with no
jeans, the model filed the role under skirts 106 times across the run archive --
and said so out loud: "Since we don't carry jeans, I searched for skirts in
similar dark blue tones." Asked which advertised colour "cream" could be, it
named beige and left out white, so an off-white sweater the shop does stock never
reached the shopper.

Neither is a reasoning failure. In the main call the model is serving a shopper
and every instinct is to be useful, so a near-enough garment looks like help
rather than a substitution, and a single conservative colour looks like
precision. Asked the same questions here there is no shopper, no schema to
satisfy and nothing to be helpful about -- just words and a catalogue. Measured
against the archive's own cases the isolated form answers 14 of 14 correctly,
refusing jeans->skirts, trousers->skirts, blazer->blouses and belt->blouses while
keeping shoes->[flats, heels, sandals], jewelry->[bracelets, earrings,
necklaces], pumps->heels and booties->boots, and mapping cream to beige *and*
white, teal to green and blue.

Deliberately not cached. The whole turn's scopes go in one request, which caps
this at one call per turn, and at roughly 400 tokens against a 47,753-token
median turn the arithmetic does not justify a cache key, an invalidation rule and
a staleness question. A cached "no" also freezes one sampling of a model into a
permanent verdict: the day this shop starts stocking jeans, a cache would still
be refusing them.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

#: Asked once per turn, so the budget is per turn and not per scope.
_MAX_TOKENS = 1200
_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class ScopeQuestion:
    """One thing the shopper asked for, in the shopper's own word.

    No subcategories. The model used to send its own guess at them and this
    module graded it, which meant a wrong guess had to be caught, reported and
    resubmitted -- and "report and resubmit" is what every retry storm in the
    search path was made of. The judge names the values instead, so there is no
    guess to be wrong and nothing to send back.
    """

    requested_product_type: str


@dataclass
class VocabularyVerdict:
    """Where each requested word lives in this catalogue, and in what colours."""

    #: requested word -> the advertised subcategories to search for it.
    #: Present and empty means judged and not carried, which is an answer the
    #: shopper can be told. Absent means never asked or unreadable, which is
    #: not the same thing and must not be reported as "we do not sell that".
    scopes: dict[str, list[str]] = field(default_factory=dict)
    #: unadvertised colour word -> the advertised colours it may mean
    colours: dict[str, list[str]] = field(default_factory=dict)
    #: True when the call did not happen or could not be read.
    unavailable: bool = False

    def subcategories_for(self, word: str) -> list[str] | None:
        """Where to search for this word, or None when it was not judged.

        None is not an empty list. Empty is a verdict -- this catalogue sells
        no such thing -- and the shopper is told so. None is the absence of a
        verdict, and telling a shopper "we do not carry jeans" because an
        endpoint timed out would be inventing a fact about the shop.
        """

        found = self.scopes.get(_normalised(word))
        return None if found is None else list(found)

    def colours_for(self, word: str) -> list[str]:
        """The advertised colours this word may mean, keyed as it was sent.

        The lookup lives here because the key is this module's. The caller
        normalises product words for taxonomy comparison, which singularises
        and drops punctuation -- "off-white" becomes "off white" there and
        stays "off-white" here, so a caller keying the reply itself would find
        nothing and quietly fall back to ranking.
        """

        return list(self.colours.get(_normalised(word)) or ())


def _normalised(text: str) -> str:
    """Lowercase, single-spaced. The reply is keyed on the word we sent."""

    return " ".join((text or "").lower().split())


class CatalogVocabularyJudge:
    """Small adapter around the configured app LLM."""

    def __init__(self, config: Any) -> None:
        self.model_name = getattr(config, "llm_name", None)
        base_url = getattr(config, "llm_port", None)
        api_key_env = getattr(config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.enabled = bool(self.model_name and base_url)
        self.client = (
            OpenAI(base_url=base_url, api_key=api_key or "not-needed")
            if self.enabled
            else None
        )

    def judge(
        self,
        questions: list[ScopeQuestion],
        colour_words: list[str],
        *,
        subcategories: list[str],
        colours: list[str],
    ) -> VocabularyVerdict:
        """Answer every vocabulary question this turn raised, in one call.

        Synchronous, because `search_catalog` is: the scopes are planned with no
        I/O, then the surviving retrievals fan out through a thread pool. This
        sits with the planning, before any budget is spent.
        """

        questions = [q for q in questions if q.requested_product_type]
        colour_words = [w for w in colour_words if w]
        if not questions and not colour_words:
            return VocabularyVerdict()
        if not self.enabled or self.client is None:
            logger.warning("vocabulary judge is not configured; scopes unjudged")
            return VocabularyVerdict(unavailable=True)

        prompt = _prompt(questions, colour_words, subcategories, colours)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=_MAX_TOKENS,
                timeout=_TIMEOUT_SECONDS,
                # No thinking. The question is a lookup against two word lists,
                # and reasoning traces were most of the completion when this was
                # measured -- 562 tokens of output for fourteen one-word
                # answers.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - the caller has a degraded rule.
            logger.warning("vocabulary judge failed: %s", exc)
            return VocabularyVerdict(unavailable=True)

        verdict = _read(content, questions, subcategories, colours)
        # Logged when it works, not only when it breaks. A scope that reached
        # the catalog when it should have been refused looks identical, after
        # the fact, to one that was never asked about -- and telling those two
        # apart is the whole of diagnosing a substitution that got through.
        logger.info(
            "vocabulary judge: asked=%s scopes=%s colours=%s->%s",
            [q.requested_product_type for q in questions],
            verdict.scopes,
            colour_words,
            verdict.colours,
        )
        return verdict

def _prompt(
    questions: list[ScopeQuestion],
    colour_words: list[str],
    subcategories: list[str],
    colours: list[str],
) -> str:
    """Two word lists, the turn's questions, and no mention of a shopper."""

    lines = [
        "You are a catalogue vocabulary checker for a clothing shop.",
        "",
        "The catalogue has exactly these subcategories:",
        ", ".join(sorted(subcategories)),
    ]
    if colours:
        lines += ["", "and exactly these colours:", ", ".join(sorted(colours))]

    if questions:
        lines += [
            "",
            "TASK A. For each word a shopper asked for, list the subcategories "
            "from the list above that ARE THAT THING.",
            " - Exact or near-exact naming: \"dress\" -> dresses, \"pumps\" -> "
            "heels, \"booties\" -> boots, \"gown\" -> dresses.",
            " - An umbrella word covers several: \"shoes\" -> flats, heels, "
            "sandals, boots; \"tops\" -> blouses, camisoles, sweaters; "
            "\"jewelry\" -> bracelets, earrings, necklaces.",
            " - If this catalogue sells no such thing, return an EMPTY list. "
            "Do NOT reach for the nearest available garment. Jeans are not "
            "skirts; a belt is not a blouse; a coat is not a camisole. An "
            "empty list is the right and useful answer, because the shopper "
            "will be told plainly that the shop does not carry it.",
            "",
            "TASK A words:",
        ]
        lines += [f'  - "{q.requested_product_type}"' for q in questions]

    if colour_words:
        lines += [
            "",
            "TASK B. For each colour word, list EVERY catalogue colour the "
            "word could plausibly mean, including the word itself if it is on "
            "the list. Be generous rather than minimal: a shopper saying one "
            "word should see all the near matches. But do not include a colour "
            "they would call plainly the wrong colour -- cream means beige and "
            "white, not black. Use only colours from the list above. Empty "
            "list if genuinely none fit.",
            "",
            "TASK B words:",
        ]
        lines += [f'  - "{word}"' for word in colour_words]

    lines += [
        "",
        "Reply with JSON and nothing else:",
        '{"a": {"word": ["subcategory"]}, "b": {"word": ["colour"]}}',
    ]
    return "\n".join(lines)


def _read(
    content: str,
    questions: list[ScopeQuestion],
    advertised_subcategories: list[str],
    advertised_colours: list[str],
) -> VocabularyVerdict:
    """Parse the reply, keeping only answers to words that were asked about."""

    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        logger.warning("vocabulary judge returned no JSON object")
        return VocabularyVerdict(unavailable=True)
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("vocabulary judge returned unreadable JSON: %s", exc)
        return VocabularyVerdict(unavailable=True)

    asked = {_normalised(q.requested_product_type) for q in questions}
    scopes = {
        word: values
        for word, values in _kept_to_vocabulary(
            payload.get("a"), advertised_subcategories
        ).items()
        if word in asked
    }
    colours = _kept_to_vocabulary(payload.get("b"), advertised_colours)
    return VocabularyVerdict(scopes=scopes, colours=colours)


def _kept_to_vocabulary(
    mapping: Any,
    advertised: list[str],
) -> dict[str, list[str]]:
    """Read a word -> values mapping, discarding values this catalogue lacks.

    The judge names the values now, so this is the only thing standing between
    an invented word and a filter. Both uses need it and for the same reason: a
    filter naming a value the catalogue does not hold matches nothing, so one
    hallucinated "denim" would turn a real search into an empty result -- and,
    worse, an empty result is indistinguishable from "we do not sell that".

    Filtering rather than rejecting the whole reply, because a judge that gets
    three words right and invents a fourth should still be believed about the
    three. A word left with nothing after filtering stays in the mapping as an
    empty list, which is the same verdict as a deliberate empty list: searching
    for it is pointless either way.
    """

    if not isinstance(mapping, dict):
        return {}
    permitted = {str(value).strip().lower() for value in advertised}
    kept: dict[str, list[str]] = {}
    for word, values in mapping.items():
        if not isinstance(values, list):
            continue
        inside = [
            str(value).strip().lower()
            for value in values
            if str(value).strip().lower() in permitted
        ]
        kept[_normalised(str(word))] = list(dict.fromkeys(inside))
    return kept


