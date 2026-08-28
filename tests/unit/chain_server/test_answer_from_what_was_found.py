"""A turn that found things must not answer with an apology.

J01 turn 2 fetched a Cancun forecast, had one search refused, had its retry
succeed, then ran out of budget before writing anything and said "I could not
complete that shopping request. Please try again." The work was done and thrown
away, and the shopper was asked to pay for it twice.
"""

from types import SimpleNamespace

from chain_server.src.turn_support import _products_found_receipt


def _state(*products):
    return SimpleNamespace(product_results=list(products))


DRESS = {"display_name": "Coral Silk Maxi Dress",
         "price": {"amount": 99.99, "currency": "USD"}}
HEELS = {"display_name": "Jade Suede Heels",
         "price": {"amount": 189.99, "currency": "USD"}}


def test_what_was_found_is_named_with_its_price() -> None:
    receipt = _products_found_receipt(_state(DRESS, HEELS))
    assert "Coral Silk Maxi Dress" in receipt
    assert "99.99" in receipt
    assert "Jade Suede Heels" in receipt


def test_a_turn_that_found_nothing_gets_no_receipt() -> None:
    """Silence here is what lets the caller fall through to the apology."""

    assert _products_found_receipt(_state()) == ""
    assert _products_found_receipt(SimpleNamespace()) == ""


def test_a_product_with_no_name_is_not_offered() -> None:
    assert _products_found_receipt(_state({"price": {"amount": 1}})) == ""


def test_the_same_product_twice_is_named_once() -> None:
    receipt = _products_found_receipt(_state(DRESS, DRESS, HEELS))
    assert receipt.count("Coral Silk Maxi Dress") == 1


def test_a_price_the_record_lacks_is_not_invented() -> None:
    receipt = _products_found_receipt(_state({"display_name": "Plain Tote"}))
    assert "Plain Tote" in receipt
    assert "None" not in receipt


def test_it_says_what_happened_and_offers_a_way_on() -> None:
    receipt = _products_found_receipt(_state(DRESS))
    assert "ran out of time" in receipt
    assert "search again" in receipt
