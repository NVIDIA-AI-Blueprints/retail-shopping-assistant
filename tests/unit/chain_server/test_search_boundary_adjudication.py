# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool schema advertises the catalog; the search body adjudicates it.

A shopper uploaded a video and asked for something similar. The model composed
three scopes -- a sweater, jeans, and brown heels -- and wrote `tan` for the
heels, a colour this catalog does not advertise. Every scope died with it and
the shopper was told "I couldn't complete a valid catalog search", having been
shown neither the sweaters nor anything else.

`search_catalog` already validates each scope and already reports rejections one
role at a time. Nothing reached it, because `args_schema` answered first and can
only answer for the whole call.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from chain_server.src.turn_support import _search_catalog_scopes_input_model
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogTaxonomySubcategory,
)


def _capabilities() -> CatalogCapabilities:
    return CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=["text"],
        filters={
            "category": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["category"],
                values=["apparel", "footwear"],
            ),
            "subcategory": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["subcategory"],
                values=["sweaters", "heels"],
            ),
            "primary_color": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["primary_color"],
                values=["beige", "brown"],
            ),
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category",
            subcategory_field="subcategory",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=2,
                    subcategories={
                        "sweaters": CatalogTaxonomySubcategory(product_count=2)
                    },
                ),
                "footwear": CatalogTaxonomyCategory(
                    product_count=2,
                    subcategories={
                        "heels": CatalogTaxonomySubcategory(product_count=2)
                    },
                ),
            },
        ),
    )


def _scope(
    product_type: str,
    category: str,
    subcategory: list[str],
    colors: list[str] | None = None,
) -> dict:
    scope: dict = {
        "semantic_query": f"a {product_type}",
        "shopper_guidance": f"Looking for a {product_type}.",
        "requested_product_type": product_type,
        "taxonomy": {"category": [category], "subcategory": subcategory},
        "required_constraints": {},
        "scope_complete": True,
        "search_mode": "text",
    }
    if colors is not None:
        scope["required_constraints"] = {"primary_color": colors}
    return scope


def _model():
    return _search_catalog_scopes_input_model(_capabilities(), max_scopes=4)


def test_an_unadvertised_value_in_one_scope_does_not_cancel_the_others() -> None:
    """The live failure: `tan` on the heels killed the sweaters as well."""

    validated = _model().model_validate(
        {
            "scopes": [
                _scope("sweater", "apparel", ["sweaters"]),
                _scope("heels", "footwear", ["heels"], ["brown", "tan"]),
            ],
            "not_covered": None,
        }
    )

    assert len(validated.scopes) == 2
    # Admitted whole and unaltered, so the body judges each role on its merits
    # and the offending value is still there to be named in the rejection.
    sweater, heels = validated.scopes
    assert sweater["requested_product_type"] == "sweater"
    assert heels["required_constraints"]["primary_color"] == ["brown", "tan"]


def test_a_call_the_catalog_fully_advertises_is_validated_as_before() -> None:
    """Nothing changes for a well-formed call: it still arrives typed."""

    validated = _model().model_validate(
        {"scopes": [_scope("sweater", "apparel", ["sweaters"], ["beige"])]}
    )

    scope = validated.scopes[0]
    assert not isinstance(scope, dict)
    assert scope.requested_product_type == "sweater"


def test_the_schema_still_advertises_every_value_it_no_longer_enforces() -> None:
    """The schema is how the catalog's shape reaches the model.

    Relaxing what it *rejects* must not relax what it *publishes*, or the model
    loses the only list of advertised values it is given.
    """

    schema = json.dumps(_model().model_json_schema())

    for advertised in ("apparel", "footwear", "sweaters", "heels", "beige", "brown"):
        assert f'"{advertised}"' in schema


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"scopes": []}, id="no scopes at all"),
        pytest.param({"scopes": ["not an object"]}, id="a scope that is not an object"),
        pytest.param(
            {"scopes": [_scope("sweater", "apparel", ["sweaters"])] * 5},
            id="more scopes than the call allows",
        ),
        pytest.param(
            {
                "scopes": [_scope("sweater", "apparel", ["sweaters"])],
                "not_covered": [1, 2],
            },
            id="a malformed not_covered",
        ),
    ],
)
def test_a_structurally_malformed_call_is_still_refused_at_the_boundary(
    payload: dict,
) -> None:
    """Only scope *content* is delegated. The call's shape stays the boundary's."""

    with pytest.raises(ValidationError):
        _model().model_validate(payload)


def test_a_verdict_that_follows_another_notice_is_still_found() -> None:
    """The body says what is not carried first, then why the scope was refused.

    Read with `startswith`, that verdict was invisible: no repair was queued and
    the bounded-repair accounting never ran.
    """

    from chain_server.src.tool_loop_control import (
        SEARCH_VALIDATION_ERROR_PREFIX,
        _validation_error_body,
    )

    content = (
        "NOT_CARRIED: this catalog advertises nothing of these kinds: jeans\n\n"
        + SEARCH_VALIDATION_ERROR_PREFIX
        + "The catalog search request does not match current capabilities: []"
    )

    assert _validation_error_body(content).startswith(SEARCH_VALIDATION_ERROR_PREFIX)
    assert _validation_error_body("a search that simply found nothing") == ""


def test_the_repair_feedback_carries_the_verdict_rather_than_its_existence() -> None:
    """What the model was told when it wrote `tan`, and what it is told now.

    Pydantic named the scope, the field, the offending value and every value
    that would have been right. The model received
    "Tool arguments failed schema validation." and repeated the mistake.
    """

    from chain_server.src.tool_loop_control import (
        SEARCH_VALIDATION_ERROR_PREFIX,
        _sanitize_repair_feedback,
    )

    content = (
        "NOT_CARRIED: nothing of these kinds: jeans\n\n"
        + SEARCH_VALIDATION_ERROR_PREFIX
        + "The catalog search request does not match current capabilities: "
        "[{'loc': ['required_constraints', 'primary_color'], "
        "'msg': \"Input should be 'beige' or 'brown'\"}]"
    )

    feedback = _sanitize_repair_feedback(content)

    assert "primary_color" in feedback
    assert "'beige' or 'brown'" in feedback
    assert feedback != "Tool arguments failed schema validation."


def test_a_scoped_error_location_names_the_field_inside_the_scope() -> None:
    """Every scoped location reads `scopes.<n>.<field>`.

    Matching the first segment against the field names therefore matched
    "scopes" every time and nothing else, so the extractor returned an empty
    set on every scoped failure and the caller read that as "nothing to say".
    """

    from chain_server.src.tool_loop_control import (
        SEARCH_VALIDATION_ERROR_PREFIX,
        _native_validation_fields,
    )

    content = (
        SEARCH_VALIDATION_ERROR_PREFIX
        + "{'scopes': []} with error:\n"
        + "scopes.2.required_constraints.primary_color\n"
        + "  Input should be 'beige' or 'brown'\n"
        + "scopes.0.taxonomy.subcategory\n"
        + "  Input should be 'sweaters' or 'heels'\n"
    )

    assert _native_validation_fields(content) == {
        "required_constraints",
        "taxonomy",
    }
