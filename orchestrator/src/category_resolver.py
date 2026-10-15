# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a requested product word lives in this catalog's advertised taxonomy.

The one place that answers it. Two answers exist and both live here:

- The vocabulary judge, one isolated model call per search call: which
  advertised subcategories a shopper's word *is* ("shoes" -> flats, heels,
  sandals, boots; "jeans" -> none) and which advertised colours an
  unadvertised colour word may mean. Asked away from the shopper, so a
  near-enough garment does not look like help. When it answers, its scope is
  the last word.
- Comparison against the closed advertised set, after normalising case,
  punctuation and plurals: is `dresses` the same entry as `dress`. It never
  reads the shopper's sentence; it is handed one value and one catalog. It is
  the degraded path when the judge is unavailable, and the coherence check on
  what the model sent.

Deciding whether the shopper *said* a thing is a different question, answered
by string matching in `lexical_provenance.py`.

The judge is deliberately not cached: one call per search call is cheap
against a turn, and a cached "no" would keep refusing a product after the shop
starts stocking it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from shared.commerce_contracts import CatalogCapabilities, ProductSummary

logger = logging.getLogger(__name__)


def _singularize_product_word(word: str) -> str:
    """Conservatively singularize one normalized product-type word."""

    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


def _normalize_product_text(value: str) -> str:
    """Normalize product-type text for deterministic lexical comparison."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("&", " and ").replace("/", " or ")
    words = re.findall(r"[^\W_]+", normalized.replace("_", " "))
    return " ".join(_singularize_product_word(word) for word in words)


def _has_alternative_connector(value: str) -> bool:
    """Return whether a product phrase joins explicit alternatives."""

    normalized = _normalize_product_text(value)
    return bool(re.search(r"\b(?:and|or)\b", normalized))


def _product_scope_key(value: str | None) -> str:
    """Return the full normalized product phrase preserved across repair."""

    return _normalize_product_text(value or "")


def _agent_selected_scope_is_advertised(
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
) -> bool:
    """Check that an open-role choice names its advertised taxonomy scope."""

    requested = _normalize_product_text(requested_product_type or "")
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    subcategories = {
        _normalize_product_text(value)
        for value in (payload.get("subcategory") or [])
    }
    return len(subcategories) == 1 and requested in subcategories


def _advertised_scope_match(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> tuple[str, str, str, str] | None:
    """Return the longest advertised exact or suffix scope match."""

    raw_requested = requested_product_type or ""
    requested = _normalize_product_text(raw_requested)
    suffix_match_allowed = not _has_alternative_connector(raw_requested)
    matches: list[tuple[str, str, str, str]] = []
    for category_name, category in capabilities.taxonomy.categories.items():
        normalized_category = _normalize_product_text(category_name)
        if requested == normalized_category or (
            suffix_match_allowed
            and requested.endswith(f" {normalized_category}")
        ):
            matches.append(
                ("category", category_name, category_name, normalized_category)
            )
        for subcategory_name in category.subcategories:
            normalized_subcategory = _normalize_product_text(subcategory_name)
            if requested == normalized_subcategory or (
                suffix_match_allowed
                and requested.endswith(f" {normalized_subcategory}")
            ):
                matches.append(
                    (
                        "subcategory",
                        subcategory_name,
                        category_name,
                        normalized_subcategory,
                    )
                )
    return max(
        matches,
        key=lambda match: (len(match[3].split()), len(match[3])),
        default=None,
    )


def _advertised_taxonomy_value(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> str | None:
    """The advertised category or subcategory this value names exactly, if any."""

    match = _advertised_scope_match(requested_product_type, capabilities)
    if match is None or match[3] != _normalize_product_text(requested_product_type or ""):
        return None
    return match[1]


def _advertised_subcategories(
    capabilities: CatalogCapabilities,
    categories: Sequence[str] | None = None,
) -> list[str]:
    """Advertised subcategories, sorted, within `categories` or across all."""

    owned = capabilities.taxonomy.categories
    return sorted(
        {
            subcategory
            for name in (categories or list(owned))
            if (category := owned.get(name)) is not None
            for subcategory in category.subcategories
        }
    )


def _owning_categories(
    subcategories: Sequence[str],
    capabilities: CatalogCapabilities,
    *,
    every: bool = False,
) -> list[str]:
    """Categories owning any (or, with `every`, all) of these subcategories."""

    test = all if every else any
    return [
        name
        for name, category in capabilities.taxonomy.categories.items()
        if subcategories and test(value in category.subcategories for value in subcategories)
    ]


def _scoped_to(
    taxonomy: BaseModel | dict[str, Any],
    subcategories: Sequence[str],
    capabilities: CatalogCapabilities,
) -> dict[str, Any]:
    """A role scoped to these subcategories, with the categories that own them.

    Categories are derived rather than kept from what the model sent: the judge
    can move a role across a category boundary ("shoes" sent as apparel lives
    in footwear), and the old category beside the new subcategories would
    filter to an empty intersection.
    """

    payload = (
        taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else dict(taxonomy or {})
    )
    payload["subcategory"] = list(dict.fromkeys(subcategories))
    payload["category"] = _owning_categories(payload["subcategory"], capabilities)
    return payload


def _selected_advertised_subcategories(
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[str, list[str]] | None:
    """One typed multi-subcategory selection owned by one category."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected = list(dict.fromkeys(payload.get("subcategory") or []))
    if len(selected) < 2:
        return None
    owners = _owning_categories(selected, capabilities, every=True)
    if len(owners) != 1 or set(payload.get("category") or []) not in (set(), {owners[0]}):
        return None
    return owners[0], selected


def _duplicates_unavailable_product_type(
    requirements: Any,
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> bool:
    """Return whether requirements only repeat an unavailable product type."""

    requested = _normalize_product_text(requested_product_type or "")
    return bool(
        requested
        and isinstance(requirements, list)
        and len(requirements) == 1
        and not _has_alternative_connector(requested_product_type or "")
        and _advertised_scope_match(requested_product_type, capabilities) is None
        and all(
            isinstance(requirement, str)
            and _normalize_product_text(requirement) == requested
            for requirement in requirements
        )
    )


def _same_product_scope(
    first: str,
    second: str,
    capabilities: CatalogCapabilities,
) -> bool:
    """Compare full product scopes without conflating advertised siblings."""

    if first == second:
        return True
    if _has_alternative_connector(first):
        return False
    first_advertised = _advertised_taxonomy_value(first, capabilities)
    if first_advertised:
        return False
    advertised_match = _advertised_scope_match(first, capabilities)
    if advertised_match:
        return second == advertised_match[3]
    return first.endswith(f" {second}")


def _products_with_subcategory_coverage(
    products: list[ProductSummary],
    selection: tuple[str, list[str]] | None,
    limit: int,
) -> list[ProductSummary]:
    """Keep rank order while reserving one result per selected subcategory."""

    if selection is None or len(products) <= limit:
        return products
    _, subcategories = selection
    selected_indexes: set[int] = set()
    for subcategory in subcategories:
        normalized_subcategory = _normalize_product_text(subcategory)
        match = next(
            (
                index
                for index, product in enumerate(products)
                if _normalize_product_text(product.category or "")
                == normalized_subcategory
            ),
            None,
        )
        if match is not None:
            selected_indexes.add(match)

    for index in range(len(products)):
        if len(selected_indexes) >= max(limit, len(subcategories)):
            break
        selected_indexes.add(index)
    return [products[index] for index in sorted(selected_indexes)]


def _advertised_taxonomy_scope_issue(
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> str | None:
    """Enforce capability-owned relations for exact advertised scopes."""

    advertised_match = _advertised_scope_match(
        requested_product_type,
        capabilities,
    )
    if advertised_match is None:
        return None
    scope_kind, advertised_name, category_name, matched_text = advertised_match
    # Only a phrase the catalog advertises, read whole, can refuse a search; a
    # phrase placed by its last word is a guess and gets no veto. Nothing this
    # compares is part of a retrieval request.
    #
    # This is the degraded mode, used only when the judge is unreachable (see
    # `_reviewed_provenance`); a named type with no taxonomy is searched on its
    # filters alone.
    if _normalize_product_text(requested_product_type or "") != matched_text:
        return None
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected_categories = payload.get("category") or []
    selected_subcategories = payload.get("subcategory") or []
    normalized_category = _normalize_product_text(category_name)
    normalized_selected_categories = {
        _normalize_product_text(value) for value in selected_categories
    }
    normalized_selected_subcategories = {
        _normalize_product_text(value) for value in selected_subcategories
    }
    if scope_kind == "category":
        category = capabilities.taxonomy.categories[category_name]
        owned_subcategories = {
            _normalize_product_text(value) for value in category.subcategories
        }
        selected_owned_children = (
            bool(normalized_selected_subcategories)
            and normalized_selected_subcategories.issubset(owned_subcategories)
            and normalized_selected_categories in (set(), {normalized_category})
        )
        if taxonomy_status == "exact_requested_type" and selected_owned_children:
            return (
                f"Requested product type '{requested_product_type}' binds to "
                f"advertised category '{advertised_name}', while the selected "
                "taxonomy contains only its advertised children. Keep those "
                "children together for the shopper's umbrella request."
            )
        # Selecting exactly the bound category substitutes nothing, so accept
        # it whether the shopper said the word or the agent derived it, as the
        # refusal below instructs.
        exact_category = (
            taxonomy_status
            in {"exact_requested_type", "agent_selected_type"}
            and not normalized_selected_subcategories
            and normalized_selected_categories == {normalized_category}
        )
        owned_children = (
            taxonomy_status
            in {"member_of_requested_umbrella", "agent_selected_type"}
            and selected_owned_children
        )
        if exact_category or owned_children:
            return None
        return (
            f"Requested product type '{requested_product_type}' binds to advertised "
            f"category '{advertised_name}'. Select that category directly or only "
            "its advertised children; do not substitute another category."
        )
    selected_matches = (
        len(selected_subcategories) == 1
        and _normalize_product_text(selected_subcategories[0])
        == _normalize_product_text(advertised_name)
        and normalized_selected_categories in (set(), {normalized_category})
    )
    if selected_matches and taxonomy_status in {
        "exact_requested_type",
        "agent_selected_type",
    }:
        return None
    return (
        f"Requested product type '{requested_product_type}' binds to advertised "
        f"subcategory '{advertised_name}'. Select only that exact subcategory; "
        "do not substitute an advertised sibling."
    )


def _catalog_execution_taxonomy_status(
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
    semantic_query: str,
    capabilities: CatalogCapabilities,
    *,
    shopper_stated_scope: bool,
) -> str:
    """Derive the legacy execution mode without asking the model to label it."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    categories = payload.get("category") or []
    subcategories = payload.get("subcategory") or []
    if (
        not semantic_query.strip()
        and not requested_product_type
        and not categories
        and not subcategories
    ):
        return "image_only"
    if not shopper_stated_scope:
        return "agent_selected_type"

    advertised_match = _advertised_scope_match(
        requested_product_type,
        capabilities,
    )
    if advertised_match is not None:
        scope_kind = advertised_match[0]
        if scope_kind == "category" and subcategories:
            return "member_of_requested_umbrella"
        return "exact_requested_type"
    if len(categories) == 1 and not subcategories:
        return "parent_category_alternative"
    if subcategories:
        return "member_of_requested_umbrella"
    return "exact_requested_type"


def _exact_taxonomy_issue(
    requested_product_type: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Return a coherence issue for an exact taxonomy claim."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    categories = payload.get("category") or []
    subcategories = payload.get("subcategory") or []
    requested_matches_category = not subcategories and any(
        _normalize_product_text(requested_product_type)
        == _normalize_product_text(category)
        for category in categories
    )
    if requested_matches_category:
        return None

    selected_product_types = subcategories or categories
    if len(selected_product_types) != 1:
        return (
            "The selected taxonomy must faithfully represent one requested type "
            "or every child of a shopper-requested umbrella."
        )
    selected_product_type = selected_product_types[0]
    if _normalize_product_text(requested_product_type) == _normalize_product_text(
        selected_product_type
    ):
        return None
    return (
        "A single taxonomy value must match requested_product_type. If no "
        "advertised value faithfully represents it, ask a clarification instead"
    )


#: Asked once per turn, so the budget is per turn and not per scope.
_MAX_TOKENS = 1200
_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class ScopeQuestion:
    """One thing the shopper asked for, in the shopper's own word.

    No subcategories. The judge names the values itself, so there is no guess
    from the model to grade, report and send back.
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
                # and reasoning traces would be most of the completion.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - the caller has a degraded rule.
            logger.warning("vocabulary judge failed: %s", exc)
            return VocabularyVerdict(unavailable=True)

        verdict = _read(content, questions, subcategories, colours)
        # Logged on success too, so a judged scope can be told apart from one
        # never asked about when diagnosing a substitution.
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
            # Says what not to do and stops. Do not add a sentence recommending
            # the empty answer: the model reads it as a quota and empties
            # words the catalogue does sell.
            " - Never reach for the nearest available garment: jeans are not "
            "skirts, a belt is not a blouse, a coat is not a camisole. Return "
            "an empty list for a word this catalogue sells no form of.",
            # Null is "not mine to rule on", read as no verdict, so the search
            # runs on the subcategories the model chose. Without it, phrases
            # that name no garment get wrongly refused as unsold.
            " - A phrase that names no kind of garment at all -- only a "
            'purpose, occasion or quality, such as "something cosy" or "an '
            'outfit for dinner" -- answer null, not a list. A phrase that '
            "names a garment is never null, whether or not this catalogue "
            "sells it: trousers, a scarf and a raincoat are garments, and "
            '"hiking boots" names boots.',
            # Role words such as "layer" must answer null, not empty, or they
            # are judged an unsold garment and the search is refused.
            " - A word for the part a piece plays in a look -- a layer, a "
            "piece, a cover, an extra -- is not a garment name either: answer "
            "null.",
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
        '{"a": {"word": ["subcategory"] or null}, "b": {"word": ["colour"]}}',
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
