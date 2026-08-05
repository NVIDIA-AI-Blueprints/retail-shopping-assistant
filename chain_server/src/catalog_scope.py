# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-visible catalog search boundary."""

from __future__ import annotations

CATALOG_SEARCH_RULES = """- Call search_catalog_tool when exact advertised
  taxonomy values faithfully represent the shopper's product type.
- If the shopper names a product type that is not separately advertised but the
  model determines that one faithful advertised parent category exists, select
  only that broader category. Keep the shopper's product type in
  `requested_product_type` and `semantic_query`, leave subcategory empty, and
  never put a product type in `unadvertised_requirements`. Returned products are
  closest alternatives under their actual catalog types, not confirmed instances
  of the shopper's unadvertised type.
- If neither a direct advertised type nor one faithful parent category exists,
  ask one concise clarification question directly and wait for the shopper; do
  not call the tool, substitute another product type, or claim catalog absence.
- Different wording is not a reason to ask. When the shopper names a true
  umbrella, search every advertised value that is genuinely a kind of that
  umbrella. For example, skirts can satisfy bottoms; dresses cannot.
- A catalog search carries a list of scopes. **One scope carries exactly one
  advertised category.** A product role the shopper names may need more than one
  scope: when a role spans categories, send one scope per category in the same
  call, not one scope listing several categories.
  For "a summer outfit: dress + shoes + one accessory", send four scopes in one
  call, because an accessory is advertised across two categories:
    1. dress     -> category apparel,  subcategory [dresses]
    2. shoes     -> category footwear, subcategory [sandals, flats]
    3. accessory -> category jewelry,  subcategory [earrings, bracelets, necklaces]
    4. accessory -> category eyewear,  subcategory [sunglasses]
- Include every faithful advertised subtype for a role within its own scope.
- Each scope owns its taxonomy and constraints, so a filter chosen for one scope
  can never exclude another scope's products, and each gets its own share of the
  results rather than competing for one ranking.
- Scopes retrieve together, so several scopes cost one round trip rather than one
  each. Send every scope the shopper's request needs in the same call; use a
  second call only for a scope that depends on what the first call returned.
- `semantic_query` supplies ranking direction only; it cannot change or repair
  the selected taxonomy.
"""
