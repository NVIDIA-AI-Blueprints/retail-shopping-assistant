"""A ref names a product; how the model wrote it is not the point.

J07 turn 4 and turn 9: shown `generated:92a114b74aaa39ea` one turn earlier, the
model sent `92a114b74aaa39ea`. The lookup missed, the resolver said the ref did
not match, the model retried the identical value until its budget ran out, and
the shopper was then told the bag was in their cart.
"""

import pytest

from chain_server.src.conversation_products import ProductEvidence
from chain_server.src.turn_support import ProductSummary


def _product(ref: str, name: str) -> ProductSummary:
    return ProductSummary(product_id=ref, display_name=name)


TOTE = _product("generated:92a114b74aaa39ea", "Ombre Canvas Tote Bag")
HEELS = _product("generated:37e5fa77c87d8e20", "Jade Suede Heels")


def test_the_qualified_ref_still_resolves() -> None:
    evidence = ProductEvidence([TOTE, HEELS])
    assert evidence.get("generated:92a114b74aaa39ea") is TOTE


def test_a_ref_with_the_scheme_dropped_resolves_to_the_same_product() -> None:
    evidence = ProductEvidence([TOTE, HEELS])
    assert evidence.get("92a114b74aaa39ea") is TOTE


def test_a_wrapped_ref_still_resolves() -> None:
    evidence = ProductEvidence([TOTE, HEELS])
    assert evidence.get("<generated:92a114b74aaa39ea>") is TOTE


def test_a_bare_ref_matching_two_products_resolves_to_neither() -> None:
    """Tolerance may not become guessing: two matches is a real ambiguity."""

    evidence = ProductEvidence(
        [
            _product("generated:dupe", "First"),
            _product("legacy:dupe", "Second"),
        ]
    )
    assert evidence.get("dupe") is None


def test_two_different_schemes_are_two_different_references() -> None:
    evidence = ProductEvidence([_product("generated:abc", "First")])
    assert evidence.get("legacy:abc") is None


def test_an_unknown_ref_still_misses() -> None:
    evidence = ProductEvidence([TOTE])
    assert evidence.get("nothing-like-this") is None
    assert evidence.get("") is None


@pytest.mark.parametrize("written", ["generated:92a114b74aaa39ea", "92a114b74aaa39ea"])
def test_the_system_recognises_its_own_choice_either_way(written: str) -> None:
    """The choice the record made must survive the model rewriting the ref."""

    evidence = ProductEvidence([TOTE])
    evidence._system_identified.add("generated:92a114b74aaa39ea")
    assert evidence.identified_by_the_system(written) is True
