# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vision model's output, in a shape a browser can render.

Every field here is coerced rather than trusted. Observed across three runs of
the same prompt, the same key came back as a list, as a string, and -- on a
failed analysis -- as a dict. A panel that assumed one of those crashes on the
second upload.
"""

import json

from chain_server.src.media_summary import summarize_media_analysis


def _analysis(**overrides) -> str:
    payload = {
        "summary": (
            "A woman models a cream cable-knit sweater with blue jeans and "
            "brown block-heeled boots."
        ),
        "fashion_items": [
            "cable-knit sweater",
            "blue jeans",
            "brown block-heeled boots",
        ],
        "search_queries": [
            "cable knit sweater women",
            "boho sweater",
            "fall sweater outfit",
            "blue jeans outfit",
        ],
        "colors": ["cream", "off-white", "beige"],
        "materials_or_textures": ["cable knit", "denim"],
        "style_terms": ["boho-chic", "casual"],
        "occasion": "casual fall outing",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_everything_seen_is_shown_and_the_focus_is_evidenced() -> None:
    """The shopper asked about the top; the jeans and boots were still seen.

    Which item the request is about is read off the model's own searches rather
    than asserted: three chase the sweater, one the jeans, none the boots.
    """

    summary = summarize_media_analysis(_analysis())

    labels = [item["label"] for item in summary["items"]]
    assert labels == ["cable-knit sweater", "blue jeans", "brown block-heeled boots"]

    pursued = {item["label"]: item["pursued"] for item in summary["items"]}
    assert pursued["cable-knit sweater"] == 3
    assert pursued["blue jeans"] == 1
    assert pursued["brown block-heeled boots"] == 0


def test_a_field_that_arrives_as_a_string_is_still_a_list() -> None:
    """`occasion` came back as a list on one turn and a string on the next."""

    assert summarize_media_analysis(_analysis())["occasion"] == [
        "casual fall outing"
    ]
    assert summarize_media_analysis(
        _analysis(occasion=["date night", "brunch"])
    )["occasion"] == ["date night", "brunch"]


def test_a_field_that_arrives_as_a_dict_is_dropped_not_rendered() -> None:
    """A failed analysis returns dicts where lists are expected."""

    summary = summarize_media_analysis(_analysis(colors={"unexpected": "shape"}))

    assert summary["colors"] == []


def test_nothing_to_show_returns_none() -> None:
    """None distinguishes "no media" from "media analysed, nothing found"."""

    assert summarize_media_analysis("") is None
    assert summarize_media_analysis("not json") is None
    assert summarize_media_analysis(json.dumps(["a", "list"])) is None
    assert summarize_media_analysis(json.dumps({"summary": "", "fashion_items": []})) is None


def test_a_failed_analysis_still_renders_its_summary() -> None:
    """The shopper is told the VLM could not read their media, not shown blanks."""

    failed = json.dumps(
        {
            "summary": "Media was attached, but VLM media analysis failed.",
            "fashion_items": [],
            "colors": [],
            "constraints_detected": {},
        }
    )

    summary = summarize_media_analysis(failed)

    assert summary is not None
    assert "failed" in summary["summary"]
    assert summary["items"] == []


def test_common_words_do_not_make_an_item_look_pursued() -> None:
    """"women" and "outfit" appear in most queries and prove nothing."""

    # The item and the queries share "women" and "outfit" and nothing else.
    # Without the stopword filter every item looks pursued by every search.
    summary = summarize_media_analysis(
        _analysis(
            fashion_items=["women's handbag"],
            search_queries=["women outfit similar look", "outfit for women"],
        )
    )

    assert summary["items"][0]["pursued"] == 0
