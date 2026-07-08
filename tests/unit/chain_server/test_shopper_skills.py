# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_PATH = (
    REPO_ROOT
    / "chain_server"
    / "skills"
    / "shopper"
    / "outfit-styling"
    / "SKILL.md"
)


def _read_skill() -> tuple[dict, str]:
    text = SKILL_PATH.read_text()
    assert text.startswith("---\n")
    frontmatter_text, body = text.removeprefix("---\n").split("\n---\n", 1)
    return yaml.safe_load(frontmatter_text), body


def test_outfit_styling_skill_has_valid_frontmatter_and_core_modes() -> None:
    frontmatter, body = _read_skill()

    assert frontmatter["name"] == "outfit-styling"
    assert 0 < len(frontmatter["description"]) <= 1024
    assert len(frontmatter["name"]) <= 64
    assert all(
        char.islower() or char.isdigit() or char == "-"
        for char in frontmatter["name"]
    )

    for heading in (
        "Anchor Product",
        "No Anchor Discovery",
        "Cart Styling",
        "Conversational Mid-Browse",
        "Budget Styling",
        "Fact And Inference Boundaries",
        "Response Style",
    ):
        assert f"## {heading}" in body or f"### {heading}" in body


def test_outfit_styling_skill_documents_tool_boundaries() -> None:
    _, body = _read_skill()

    for tool_name in (
        "search_catalog_tool",
        "get_product_details_tool",
        "get_cart_tool",
        "view_cart_total_tool",
    ):
        assert tool_name in body
    assert "cart mutation tools only after explicit shopper intent" in body
    assert "Selection or approval is not cart intent" in body
    assert "do not add earlier anchor, core outfit, or optional pieces" in body
    assert "ask one concise clarification before calling a cart mutation tool" in body
    assert "Do not expose skill names, tool names" in body


def test_outfit_styling_skill_documents_cart_tool_flow() -> None:
    _, body = _read_skill()

    assert "Read the cart, assess the current lines, and answer" in body
    assert "Do not search for items already named as cart contents" in body
    assert "at most one catalog search after the cart read" in body


def test_outfit_styling_skill_documents_fact_inference_boundary() -> None:
    _, body = _read_skill()

    assert "Shopper wording is context, not catalog truth" in body
    assert "do not show them to the shopper" in body
    assert "Styling rationale may infer from confirmed facts" in body
    assert "Attribute material and comfort claims item by item" in body
    assert "Outfit-wide claims" in body
    assert "every included apparel item, shoe, bag" in body
    assert "Do not collapse different material classes" in body
    assert "Cotton and\n  linen are fibers" in body
    assert "Outdoor-practicality claims need exact support" in body
    assert "stable" in body
    assert "grass or gravel" in body
    assert "water-resistant" in body
    assert "all-day comfortable" in body
    assert "Before writing comparison tables" in body
    assert "get_product_details_tool` for each relevant `PRODUCT_REF`" in body
    assert "Do not convert sole or strap facts into surface guarantees" in body
    assert "For surface or weather concerns" in body
    assert "Avoid superlatives and category-wide rankings" in body
    assert "maximum breathability" in body
    assert "Unsupported Commerce Details" in body
    assert "tax, shipping fees, delivery dates" in body
    assert "real-time\n  inventory/stock status" in body
    assert "In initial recommendations" in body
    assert "Do not enumerate materials" in body


def test_skill_registry_matches_runtime_skill_file() -> None:
    registry = (REPO_ROOT / "docs" / "SHOPPER_AGENT_SKILL_REGISTRY.md").read_text()
    runtime_source = (
        REPO_ROOT / "chain_server" / "src" / "deepagents_runtime.py"
    ).read_text()

    assert "| `outfit-styling` |" in registry
    assert "chain_server/skills/shopper/outfit-styling/SKILL.md" in registry
    assert '_SHOPPER_SKILLS_SOURCE = "/shopper"' in runtime_source
    assert 'skills=[_SHOPPER_SKILLS_SOURCE]' in runtime_source


def test_chain_server_docker_image_includes_skill_files() -> None:
    dockerfile = (REPO_ROOT / "chain_server" / "Dockerfile").read_text()

    assert "COPY ./skills ./skills" in dockerfile
