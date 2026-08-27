# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything the model actually reads on one turn, as one string.

Rules about how to fill a search live in four places that reach the model: the
system prompt, the catalog rules inside it, the selected skill body, and the
descriptions on the tool schema. A test that asserts a rule is in one of those
files is really asserting where the words sit, and it fails the moment the rule
moves to a channel that carries it better -- which is exactly what a prompt
diet does, and exactly when you most want the test to still be checking
something.

So the union. `reachable_on_a_turn_using` answers the only question worth
asking: on a turn that selects this skill, does the model read this rule at
all? It survives text moving between channels and still fails when a rule is
deleted outright.

It deliberately does not check *which* channel a rule landed in. That matters
enormously -- a rule in the wrong channel is inert -- but it is a different
property, and the tests that care assert against a single channel on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFilterCapability,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogTaxonomySubcategory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHOPPER_SKILLS = REPO_ROOT / "chain_server" / "skills" / "shopper"


def a_catalog_like_the_real_one() -> CatalogCapabilities:
    """Enough shape that the schema builder emits every field it can.

    The audience filter in particular: its description is only attached when
    the configured `wearer_audience_field` is present in the capabilities, and
    that description carries the audience rule.
    """

    enum = lambda values: CatalogFilterCapability(  # noqa: E731
        type="enum", operators=["in"], source_fields=["x"], values=values
    )
    return CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=["text", "image", "hybrid"],
        filters={
            "category": enum(["apparel", "footwear", "bags", "jewelry"]),
            "subcategory": enum(["dresses", "skirts", "heels", "totes"]),
            "primary_color": enum(["black", "beige", "navy"]),
            "target_audience": enum(["womens", "adult_all_genders"]),
            "price": CatalogFilterCapability(
                type="number",
                operators=["gte", "lte"],
                source_fields=["price"],
                min_value=39.90,
                max_value=269.99,
            ),
        },
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category",
            subcategory_field="subcategory",
            categories={
                "apparel": CatalogTaxonomyCategory(
                    product_count=40,
                    subcategories={
                        "dresses": CatalogTaxonomySubcategory(product_count=20),
                        "skirts": CatalogTaxonomySubcategory(product_count=20),
                    },
                ),
                "footwear": CatalogTaxonomyCategory(
                    product_count=10,
                    subcategories={
                        "heels": CatalogTaxonomySubcategory(product_count=10)
                    },
                ),
            },
        ),
    )


def search_tool_schema(capabilities: CatalogCapabilities | None = None) -> str:
    """The search tool's JSON schema, exactly as the model receives it."""

    from chain_server.src.turn_support import _search_catalog_scopes_input_model

    return json.dumps(
        _search_catalog_scopes_input_model(
            capabilities or a_catalog_like_the_real_one(),
            max_scopes=10,
            wearer_audience_field="target_audience",
        ).model_json_schema()
    )


def shopping_tool_descriptions() -> str:
    """Docstrings of the tools, which are their model-facing descriptions.

    Read from source rather than by building an agent: constructing the tools
    needs a live runtime, and the text is what is under test.
    """

    return (REPO_ROOT / "chain_server" / "src" / "deepagents_runtime.py").read_text()


def skill_body(name: str) -> str:
    path = SHOPPER_SKILLS / name / "SKILL.md"
    return path.read_text()


def reachable_on_a_turn_using(
    runtime: Any,
    *skill_names: str,
    capabilities: CatalogCapabilities | None = None,
) -> str:
    """Whitespace-normalised union of every channel that turn reads."""

    capabilities = capabilities or a_catalog_like_the_real_one()
    parts = [
        runtime._system_prompt(capabilities),
        search_tool_schema(capabilities),
        shopping_tool_descriptions(),
        *(skill_body(name) for name in skill_names),
    ]
    return " ".join(" ".join(parts).split())
