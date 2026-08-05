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
- A catalog search carries a list of scopes, one per product role the shopper
  asked for. "A dress, shoes and a bag" is one call with three scopes, not three
  calls. Each scope owns its own taxonomy and constraints, so a filter chosen
  for one role can never exclude another role's products, and each role gets its
  own share of the results rather than competing for the same ranking.
- Scopes retrieve together, so several roles cost one round trip rather than one
  each. Send every role the shopper named in the same call; use a second call
  only for a role that depends on what the first call returned.
- Each search owns one complete retrieval scope. `semantic_query` supplies
  ranking direction only; it cannot change or repair the selected taxonomy.
- Each search covers at most one catalog category and one focused product role.
  Include every faithful advertised subtype for that role in the same search.
  Use separate searches for separate roles, up to the configured turn limit.
"""
