# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-visible catalog search boundary."""

from __future__ import annotations

CATALOG_SEARCH_RULES = """- Call search_catalog_tool when exact advertised
  taxonomy values faithfully represent the shopper's product type.
- If the shopper names a product type that is not separately advertised, decide
  it against the subcategories that exist rather than the category word. A
  parent category qualifies only when one of its advertised subcategories
  denotes the same kind of thing: pumps are heels, so footwear qualifies. Every
  garment is apparel, so "is it a kind of this category" can never fail and is
  not the test. When a parent qualifies, select only that category and
  keep the shopper's product type in `requested_product_type` and
  `semantic_query`, leave subcategory empty, and
  never put a product type in `unadvertised_requirements`.
  Returned products are closest alternatives under their actual catalog types,
  not confirmed instances of the shopper's type.
- If no advertised subcategory denotes the requested kind, the catalog does not
  carry it. Name it in `not_covered` beside the roles you can search, and build
  no scope for it. If it is the only thing asked for, do not call the tool at
  all: say plainly that the catalog does not carry it and offer one advertised
  direction. Absence read off the published taxonomy is a fact the shopper
  needs; absence guessed from a search that returned little is the thing never
  to claim.
- Different wording is not a reason to ask. When the shopper names a true
  umbrella, search every advertised value that is genuinely a kind of that
  umbrella. For example, skirts can satisfy bottoms; dresses cannot.
- The advertised categories in Catalog capabilities are the whole store. If what
  the shopper asks for is not a kind of any advertised category, do not search
  for it and do not substitute a neighbour. When the same turn also asks for
  things the catalog does carry, send scopes for those and name the uncovered
  ones in `not_covered`: "a pan, a shoe and a bag" is scopes for the shoe and
  the bag, plus `not_covered: ["pan"]`. When nothing in the turn is covered, do
  not call the tool at all -- say so plainly and name what this catalog does
  cover. Either way that is a fact stated by the capability contract above, not
  a guess about stock. Absence *within* an advertised category is different: it
  is unknown until searched.
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
- **A hard filter takes advertised values only. The shopper's own word is not a
  filter value.** What they said -- typed, or seen by the camera -- decides
  *that* they have a constraint. It never decides which token goes in
  `required_constraints`; that token comes from Catalog capabilities above.
  When their word is not advertised, choose the closest advertised value or
  values, keep their own word in `semantic_query`, and say in the reply it is
  not an exact match. A shopper who asks for a cream sweater is searched as
  beige, and told so.
  An unadvertised value is rejected before the search runs, and it takes every
  other scope in the same call down with it -- one unadvertised colour on a
  sweater costs the shopper their jeans and their boots too.
  Do not drop the constraint, and do not move the word into
  `unadvertised_requirements` when an advertised value is close: that field is
  for qualities the catalog cannot filter on at all, such as "cable-knit". If
  nothing advertised is close, leave that filter out and say so.
  **A shopper who ruled out alternatives is not substituted for.** "only teal",
  "nothing else", "no substitutes" -- when their word is unadvertised and they
  said that, do not search a neighbour and do not show one. Say plainly that the
  catalog does not carry it, name the closest advertised values, and ask whether
  either would do. Substituting here answers a question they explicitly did not
  ask: live, "show me ONLY teal dresses, don't show me anything else" returned
  blue and green ones.
"""
