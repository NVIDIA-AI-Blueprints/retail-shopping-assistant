# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-visible catalog search boundary."""

from __future__ import annotations

CATALOG_SEARCH_RULES = """- Call search_catalog_tool only when exact advertised
  taxonomy values faithfully represent the shopper's product type. If they do
  not, ask one concise clarification question directly and wait for the shopper;
  do not call the tool, substitute another product type, or claim catalog absence.
- Different wording is not a reason to ask. When the shopper names a true
  umbrella, search every advertised value that is genuinely a kind of that
  umbrella. For example, skirts can satisfy bottoms; dresses cannot.
- Each search owns one complete retrieval scope. `semantic_query` supplies
  ranking direction only; it cannot change or repair the selected taxonomy.
- Each search covers at most one catalog category and one focused product role.
  Include every faithful advertised subtype for that role in the same search.
  Use separate searches for separate roles, up to the configured turn limit.
"""
