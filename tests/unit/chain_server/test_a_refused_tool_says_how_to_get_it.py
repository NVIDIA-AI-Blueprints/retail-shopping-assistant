"""A tool refused for its grant says which skill grants it.

J11 turn 7, "add the Ombre Canvas Tote Bag":

    activate_shopper_skills_tool(["outfit-styling", "budget-shopping"])  ok
    search_catalog_tool          -> finds the bag                        ok
    get_product_details_tool     -> confirms the ref and $49.99          ok
    add_cart_items_tool          -> rejected, skill_tool_not_granted

and then, over an empty cart: "I've added the Ombre Canvas Tote Bag to your
cart."

The message it was refused with ended "continue using only the tools available
for this turn", which forbids the one recovery that works -- activating
cart-management and calling the tool again. The model did as it was told and
wrote the reply it would have written had the add worked.

Which skill grants the tool is known where the refusal is written, so it is
said. And a refused call is not a thing that happened, which is said too.
"""

from __future__ import annotations

from chain_server.src.skill_activation import (
    SKILL_TOOL_NOT_GRANTED,
    _tool_not_granted,
)


def test_the_refusal_names_the_skill_that_grants_the_tool() -> None:
    message = _tool_not_granted(
        "add_cart_items_tool", ["outfit-styling", "budget-shopping"]
    )
    assert "cart-management" in message
    assert "add_cart_items_tool" in message
    # And what this turn actually holds, so the model can see the difference.
    assert "outfit-styling" in message
    assert "budget-shopping" in message


def test_the_refusal_asks_for_a_retry_rather_than_forbidding_one() -> None:
    message = _tool_not_granted("add_cart_items_tool", ["outfit-styling"])
    assert "activate_shopper_skills_tool" in message
    assert "call the tool again" in message
    # The sentence that caused the lie is gone.
    assert "Continue using only the tools available" not in message


def test_the_refusal_says_nothing_has_happened() -> None:
    """The reply is written by the model that reads this."""

    message = _tool_not_granted("add_cart_items_tool", ["outfit-styling"])
    assert "Nothing has been done yet" in message
    assert "do not tell the shopper it has" in message


def test_the_prefix_survives_so_the_readers_still_recognise_it() -> None:
    """Four places identify this outcome by its opening."""

    for tool in (
        "add_cart_items_tool",
        "get_store_policy_tool",
        "search_catalog_tool",
    ):
        assert _tool_not_granted(tool, ["outfit-styling"]).startswith(
            SKILL_TOOL_NOT_GRANTED
        )
