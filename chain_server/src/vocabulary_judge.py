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
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

#: Asked once per turn, so the budget is per turn and not per scope.
_MAX_TOKENS = 1200
_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class ScopeQuestion:
    """One role the model wants to search, as it sent it."""

    requested_product_type: str
    subcategories: tuple[str, ...]


@dataclass
class VocabularyVerdict:
    """What the judge said, and whether it got to say anything at all."""

    #: requested type -> subcategory -> is it a kind of the requested thing
    kinds: dict[str, dict[str, bool]] = field(default_factory=dict)
    #: unadvertised colour word -> the advertised colours it may mean
    colours: dict[str, list[str]] = field(default_factory=dict)
    #: True when the call did not happen or could not be read.
    unavailable: bool = False

    def every_subcategory_is_a_kind(self, question: ScopeQuestion) -> bool | None:
        """True, False, or None when this scope was not judged.

        None is not False. A scope the judge never ruled on has to be decided by
        the caller's degraded rule, and conflating the two would refuse a valid
        umbrella every time the endpoint hiccupped.
        """

        verdicts = self.kinds.get(_normalised(question.requested_product_type))
        if not verdicts:
            return None
        answered = [
            verdicts[sub] for sub in question.subcategories if sub in verdicts
        ]
        if len(answered) != len(question.subcategories):
            return None
        return all(answered)

    def colours_for(self, word: str) -> list[str]:
        """The advertised colours this word may mean, keyed as it was sent.

        The lookup lives here because the key is this module's. The caller
        normalises product words for taxonomy comparison, which singularises
        and drops punctuation -- "off-white" becomes "off white" there and
        stays "off-white" here, so a caller keying the reply itself would find
        nothing and quietly fall back to ranking.
        """

        return list(self.colours.get(_normalised(word)) or ())

    def subcategories_to_drop(self, question: ScopeQuestion) -> tuple[str, ...]:
        """The members that are not kinds of the requested thing.

        A mixed scope keeps what it got right: asked for footwear as flats, heels
        and skirts, the judge returns the skirt alone, so the role searches its
        two real members instead of losing all three.
        """

        verdicts = self.kinds.get(_normalised(question.requested_product_type)) or {}
        return tuple(
            sub
            for sub in question.subcategories
            if sub in verdicts and not verdicts[sub]
        )


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

        questions = [
            q for q in questions if q.requested_product_type and q.subcategories
        ]
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

        verdict = _read(content, questions, colours)
        # Logged when it works, not only when it breaks. A scope that reached
        # the catalog when it should have been refused looks identical, after
        # the fact, to one that was never asked about -- and telling those two
        # apart is the whole of diagnosing a substitution that got through.
        logger.info(
            "vocabulary judge: asked=%s verdicts=%s colours=%s->%s",
            [(q.requested_product_type, list(q.subcategories)) for q in questions],
            verdict.kinds,
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
            "TASK A. For each request, a search was scoped to some "
            "subcategories. For each subcategory, say whether it is genuinely A "
            "KIND OF the thing requested.",
            " - A pump IS a kind of heel. A bootie IS a kind of boot. A gown IS "
            "a dress.",
            " - An umbrella word covers several: \"shoes\" covers flats, heels, "
            "sandals and boots; \"tops\" covers blouses, camisoles and "
            "sweaters; \"bottoms\" covers skirts.",
            " - A skirt is NOT a kind of jeans. A blouse is NOT a kind of belt. "
            "Being the nearest available thing, or a reasonable alternative to "
            "suggest, does NOT make it a kind of the thing. Judge the words "
            "only.",
            "",
            "TASK A requests:",
        ]
        lines += [
            f'  - requested "{q.requested_product_type}" -> scoped to '
            f"{list(q.subcategories)}"
            for q in questions
        ]

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
        '{"a": [{"requested": "...", "verdicts": {"subcategory": true}}], '
        '"b": {"word": ["colour"]}}',
    ]
    return "\n".join(lines)


def _read(
    content: str,
    questions: list[ScopeQuestion],
    advertised: list[str],
) -> VocabularyVerdict:
    """Parse the reply, keeping only answers to questions that were asked.

    Answers are matched on the request word *and* on covering every subcategory
    that word was sent with. An earlier version of this matched on the word
    alone, and when one turn asked about "shoes" twice the second question
    silently collected the first one's verdicts -- and scored itself a pass.
    """

    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        logger.warning("vocabulary judge returned no JSON object")
        return VocabularyVerdict(unavailable=True)
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("vocabulary judge returned unreadable JSON: %s", exc)
        return VocabularyVerdict(unavailable=True)

    wanted: dict[str, set[str]] = {}
    for question in questions:
        wanted.setdefault(_normalised(question.requested_product_type), set()).update(
            question.subcategories
        )

    kinds: dict[str, dict[str, bool]] = {}
    for row in payload.get("a") or []:
        if not isinstance(row, dict):
            continue
        requested = _normalised(str(row.get("requested") or ""))
        verdicts = row.get("verdicts")
        if requested not in wanted or not isinstance(verdicts, dict):
            continue
        merged = kinds.setdefault(requested, {})
        for sub, verdict in verdicts.items():
            if sub in wanted[requested] and isinstance(verdict, bool):
                merged[sub] = verdict

    # Kept to the list the question offered. A mapping becomes a filter, and a
    # filter naming a value this catalog does not hold matches nothing, so one
    # invented word would turn a best-effort widening into an empty result.
    # The prompt says to use only these; this is what makes it true.
    permitted = {str(value).strip().lower() for value in advertised}
    colours: dict[str, list[str]] = {}
    for word, values in (payload.get("b") or {}).items():
        if not isinstance(values, list):
            continue
        mapped = [
            str(value).strip().lower()
            for value in values
            if str(value).strip().lower() in permitted
        ]
        colours[_normalised(str(word))] = list(dict.fromkeys(mapped))

    return VocabularyVerdict(kinds=kinds, colours=colours)


def exact_identity_only(question: ScopeQuestion) -> bool:
    """The degraded rule, for when the judge could not be reached.

    Not fuzzy matching. A subcategory passes only when it *is* the requested
    word once plurals and separators are set aside, so `dress` clears `dresses`
    and `tote bag` clears `tote_bags`, while `jeans` never clears `skirts`.

    Umbrellas are refused while the judge is unreachable, which is the honest
    trade: "shoes" is briefly narrowed to nothing rather than a shopper being
    quietly shown a skirt for jeans during an outage.
    """

    def stem(value: str) -> str:
        return re.sub(r"[^a-z]", "", value.lower()).rstrip("s")

    requested = stem(question.requested_product_type)
    return bool(requested) and all(
        stem(sub) == requested for sub in question.subcategories
    )
