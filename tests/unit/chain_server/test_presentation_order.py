# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cards and the sentences are one list, so they carry one order."""

from __future__ import annotations

from chain_server.src.turn_support import _in_presentation_order


def _p(name: str) -> dict:
    return {"display_name": name}


def test_products_follow_the_order_the_reply_presents_them() -> None:
    """Measured across recorded turns, the two orders disagreed about half the
    time. A shopper counting cards and a shopper counting sentences then mean
    different products by "the second one"."""

    products = [_p("Satin Effect Midi Skirt"), _p("Polished Pearl Skirt"), _p("Jasmine Silk Skirt")]
    reply = "I'd start with the Jasmine Silk Skirt, then the Polished Pearl Skirt, then the Satin Effect Midi Skirt."

    assert [p["display_name"] for p in _in_presentation_order(products, reply)] == [
        "Jasmine Silk Skirt",
        "Polished Pearl Skirt",
        "Satin Effect Midi Skirt",
    ]


def test_a_product_the_reply_never_names_keeps_its_ranking_behind_the_named() -> None:
    """It was still shown, so it is still offered -- just not ahead of the ones
    the shopper was actually told about."""

    products = [_p("Alpha Dress"), _p("Beta Dress"), _p("Gamma Dress")]
    reply = "The Gamma Dress is the closest match."

    assert [p["display_name"] for p in _in_presentation_order(products, reply)] == [
        "Gamma Dress",
        "Alpha Dress",
        "Beta Dress",
    ]


def test_two_products_named_in_the_same_breath_keep_the_catalog_order() -> None:
    """Nothing in the sentence separates them, so ranking is the tiebreak."""

    products = [_p("Alpha Dress"), _p("Beta Dress")]
    reply = "Both the Beta Dress and Alpha Dress work."

    ordered = [p["display_name"] for p in _in_presentation_order(products, reply)]
    assert ordered == ["Beta Dress", "Alpha Dress"]


def test_a_reply_that_names_nothing_leaves_the_ranking_alone() -> None:
    products = [_p("Alpha Dress"), _p("Beta Dress")]

    assert _in_presentation_order(products, "Here are some options.") == products


def test_an_empty_reply_leaves_the_ranking_alone() -> None:
    products = [_p("Alpha Dress")]

    assert _in_presentation_order(products, "") == products


def test_no_products_is_not_an_error() -> None:
    assert _in_presentation_order([], "anything") == []


def test_the_images_follow_the_products_and_lose_nothing() -> None:
    """The cards render from this map, so it must agree with the list beside it."""

    from chain_server.src.turn_support import _images_in_product_order

    images = {"Alpha Dress": "/a.jpg", "Beta Dress": "/b.jpg", "Gamma Dress": "/g.jpg"}
    products = [_p("Gamma Dress"), _p("Alpha Dress"), _p("Beta Dress")]

    assert list(_images_in_product_order(images, products)) == [
        "Gamma Dress",
        "Alpha Dress",
        "Beta Dress",
    ]


def test_an_image_the_products_do_not_name_is_kept_at_the_end() -> None:
    """It was shown. Dropping it would remove a card rather than move one."""

    from chain_server.src.turn_support import _images_in_product_order

    images = {"Alpha Dress": "/a.jpg", "Orphan Dress": "/o.jpg"}
    products = [_p("Alpha Dress")]

    ordered = _images_in_product_order(images, products)

    assert list(ordered) == ["Alpha Dress", "Orphan Dress"]
    assert ordered["Orphan Dress"] == "/o.jpg"
