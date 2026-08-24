"""A showing made under a size filter is size-qualified.

"ok, just show me sandals in a 7" then "add the first one" asked which size --
the shopper having said it one turn earlier. Those four sandals came back
*because* they come in a 7, and the set forgot.

The size qualifies the SET, never the shopper. That boundary is the whole
safety of this: a later showing carries its own size or none, so a size stated
for sandals can never reach a dress.
"""

from chain_server.src.conversation_products import _format_reference_set

SANDALS = {
    "candidate_set_id": "set-sandals",
    "turn_seq": 5,
    "products": [
        {"ref": "generated:1", "name": "Gleaming Gold Sandals",
         "position": 1, "category": "sandals"},
    ],
}


def test_a_size_qualified_showing_says_so() -> None:
    line = _format_reference_set({**SANDALS, "shopper_size": "7"})
    assert "[shopper asked for size 7]" in line
    assert "Gleaming Gold Sandals" in line


def test_a_showing_with_no_size_filter_claims_none() -> None:
    line = _format_reference_set(SANDALS)
    assert "shopper asked for size" not in line


def test_the_size_belongs_to_one_showing_only() -> None:
    """The fear this design has to answer: a size stated for sandals must
    never reach a dress shown later."""

    dresses = {
        "candidate_set_id": "set-dresses",
        "turn_seq": 7,
        "products": [
            {"ref": "generated:9", "name": "Black Satin Lace-Up Dress",
             "position": 1, "category": "dresses"},
        ],
    }
    sandal_line = _format_reference_set({**SANDALS, "shopper_size": "7"})
    dress_line = _format_reference_set(dresses)
    assert "size 7" in sandal_line
    assert "size" not in dress_line.replace("Lace-Up", "")


def test_a_malformed_set_still_renders_nothing() -> None:
    assert _format_reference_set({"shopper_size": "7"}) == ""
    assert _format_reference_set({**SANDALS, "products": []}) == ""


def test_only_one_size_qualifies_a_showing() -> None:
    """Two sizes means the shopper was comparing, and neither is the one they
    want. The memory service attaches nothing in that case."""

    import inspect

    from memory_retriever.src import product_references

    source = inspect.getsource(product_references.rebuild_product_reference_index)
    assert "len(sizes) == 1" in source


def test_the_collector_returns_a_size_from_a_real_evidence_record() -> None:
    """The test that was missing, and its absence hid a dead feature.

    `shopper_sizes` returned [] for two days. The renderer worked, the
    set-scoped boundary worked, and both mutations landed on those halves --
    nothing fed the collector a record shaped like the ones it actually
    receives. This is that record, copied from a stored J06 turn.
    """

    from chain_server.src.turn_support import _diagnostic_shopper_sizes

    evidence = [
        {
            "product_ref": "generated:abc",
            "product_name": "Black Satin Lace-Up Dress",
            "source_tool": "search_catalog_tool",
            "evidence_type": "search",
            "facts": {},
            "search_scope": {
                "composed_role": False,
                "confirmed_filters": {"primary_color": ["black"], "sizes": ["2"]},
                "taxonomy": {"category": ["apparel"], "subcategory": ["dresses"]},
            },
        }
    ]

    assert _diagnostic_shopper_sizes(evidence) == ["2"]


def test_a_search_with_no_size_filter_records_no_size() -> None:
    from chain_server.src.turn_support import _diagnostic_shopper_sizes

    assert _diagnostic_shopper_sizes(
        [{"search_scope": {"confirmed_filters": {"primary_color": ["black"]}}}]
    ) == []
    assert _diagnostic_shopper_sizes([]) == []
    assert _diagnostic_shopper_sizes([{"no_scope": True}, "junk"]) == []


def test_two_showings_with_different_sizes_both_record() -> None:
    """The set-scoped boundary decides which one qualifies a showing; the
    collector's job is only to report what the searches confirmed."""

    from chain_server.src.turn_support import _diagnostic_shopper_sizes

    sizes = _diagnostic_shopper_sizes([
        {"search_scope": {"confirmed_filters": {"sizes": ["2"]}}},
        {"search_scope": {"confirmed_filters": {"sizes": ["4"]}}},
        {"search_scope": {"confirmed_filters": {"sizes": ["2"]}}},
    ])
    assert sizes == ["2", "4"]
