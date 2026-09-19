"""Deciding whether the shopper said a thing, by matching strings. To be removed.

This is the one piece of lexical matching left in the search path, and it is
here in its own file so it is impossible to miss rather than because it is a
layer worth having.

What it does: stems the words of the shopper's turn with a fixed English suffix
table, stems the words of a proposed requirement, and calls the requirement
"shopper-stated" when one set of stems contains the other. `catalog_search.py`
calls it at four sites to decide provenance.

Why it has to go. The suffix table is English, hand-written, and unaware of
meaning, so provenance comes out asymmetric for reasons no one intended: a
shopper who says "jeans" is heard, and one who says "cream" is not, because
one word survives the stemmer into a catalog value and the other does not. It
fails silently and in the shopper's favour only by luck.

What replaces it is a judgment about meaning, which is what the resolver
already makes for taxonomy and colour. Nothing here should be repaired in
place; repairing a stemmer is how it earns its keep and stays.

Moved verbatim out of `catalog_vocabulary.py`, which holds the honest
neighbour: comparing one value against the closed set the catalog advertises.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .agenttypes import DialogueTurn
from .catalog_vocabulary import (
    _normalize_product_text,
    _product_scope_key,
    _singularize_product_word,
)


def _text_mentions_product_type(text: str, product_type: str) -> bool:
    """Return whether text names a normalized product-type form."""

    normalized_text = _normalize_product_text(text)
    padded_text = f" {normalized_text} "
    normalized_product_type = _normalize_product_text(product_type)
    return bool(normalized_product_type) and (
        f" {normalized_product_type} " in padded_text
    )




def _requirement_word_stem(word: str) -> str:
    """Return a conservative stem for literal requirement provenance."""

    for suffix in ("ance", "ence", "ancy", "ency", "ant", "ent", "ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[: -len(suffix)]
    return _singularize_product_word(word)




def _shopper_stated_requirement(query: str, requirement: str) -> bool:
    """Return whether a proposed requirement is grounded in the current turn."""

    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    query_words = {
        _requirement_word_stem(word)
        for word in re.findall(r"[^\W_]+", normalized_query)
    }
    requirement_words = {
        _requirement_word_stem(word)
        for word in re.findall(
            r"[^\W_]+",
            unicodedata.normalize("NFKC", requirement).casefold(),
        )
    }
    return bool(requirement_words) and requirement_words.issubset(query_words)




def a_place_the_shopper_named(
    shopper_statements: Sequence[str],
    quoted: str,
) -> bool:
    """Whether the words offered as naming the place were ever actually said.

    The tool asks the model to quote the words that named the place, and the
    model quoted "Italy" on a turn reading "it's going to snow when we get
    back" -- and, in the next run, "Rome", which the shopper never said in any
    turn. A required field it can fill with anything is a field it will fill
    with anything.

    So the citation is checked against the record, which is the same thing
    `expected_display_name` does for a product name: not what the words mean,
    only whether they were said. Reusing the constraint-provenance reader so a
    quotation is judged the same way everywhere.

    The record is every turn the shopper has spoken, not only the current one.
    Checking the current turn alone was narrower than the defect and cost the
    behaviour it was meant to protect: nine turns into planning one trip to
    one city, "will I need a jacket in the evening" names no place, so the
    call was refused -- and the reply then said no forecast was available for
    Cancun and described a typical Cancun September anyway. Earlier runs had
    fetched that forecast and cited the provider.

    What made Rome wrong is not something this function can see. The shopper
    had stated the conditions, and "when we get back" is home rather than the
    city of the trip; both are judgments about meaning, and both are stated on
    the field the quotation comes from. What a substring check can establish is
    that the words were said by the shopper at all, which is what stopped the
    invented "Rome", and that is all it claims to establish.
    """

    if not quoted.strip():
        return False
    return any(
        _shopper_stated_requirement(statement, quoted)
        for statement in shopper_statements
    )




def _recent_shopper_statements(
    dialogue: Sequence[DialogueTurn],
    *,
    limit: int = 4,
) -> str:
    """Read prior shopper text from the typed lane, never from rendered prose.

    Assistant text is deliberately excluded: dialogue may carry shopper intent,
    but assistant prose is not product, policy, inventory, or cart evidence.
    """

    recent = list(dialogue)[-limit:]
    return "\n".join(turn.shopper_text for turn in recent if turn.shopper_text)




def _shopper_stated_product_scope(
    query: str,
    dialogue: Sequence[DialogueTurn],
    product_scope_key: str,
) -> bool:
    """Return whether current or recent shopper text states a product scope."""

    shopper_text = "\n".join(
        value for value in (query, _recent_shopper_statements(dialogue)) if value
    )
    return _text_mentions_product_type(shopper_text, product_scope_key)


def _resolved_agent_selected_product_type(
    *,
    query: str,
    dialogue: Sequence[DialogueTurn],
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Derive open-role provenance from the agent's single taxonomy choice."""

    if taxonomy_status != "agent_selected_type":
        return requested_product_type
    scope_key = _product_scope_key(requested_product_type)
    if scope_key and _shopper_stated_product_scope(query, dialogue, scope_key):
        return requested_product_type
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    subcategories = payload.get("subcategory") or []
    if len(subcategories) == 1:
        return str(subcategories[0])
    return requested_product_type
