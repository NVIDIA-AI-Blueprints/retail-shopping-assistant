# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SHOPPER_SKILLS_ROOT = REPO_ROOT / "chain_server" / "skills" / "shopper"
REGISTERED_SKILL_PATHS = {
    name: SHOPPER_SKILLS_ROOT / name / "SKILL.md"
    for name in (
        "budget-shopping",
        "cart-management",
        "outfit-styling",
        "product-discovery",
        "store-policy-answers",
    )
}
SKILL_PATH = REGISTERED_SKILL_PATHS["outfit-styling"]
TRENDS_PATH = SHOPPER_SKILLS_ROOT / "trends-current.md"


def _read_skill_path(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    assert text.startswith("---\n")
    frontmatter_text, body = text.removeprefix("---\n").split("\n---\n", 1)
    return yaml.safe_load(frontmatter_text), body


def _read_skill() -> tuple[dict, str]:
    return _read_skill_path(SKILL_PATH)


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


def test_registered_shopper_skills_have_valid_frontmatter() -> None:
    discovered = {
        path.parent.name: path
        for path in SHOPPER_SKILLS_ROOT.glob("*/SKILL.md")
    }

    assert discovered == REGISTERED_SKILL_PATHS
    for name, path in discovered.items():
        frontmatter, body = _read_skill_path(path)

        assert frontmatter["name"] == name
        assert 0 < len(frontmatter["description"]) <= 1024
        assert 0 < len(name) <= 64
        assert not name.startswith("-")
        assert not name.endswith("-")
        assert "--" not in name
        assert all(
            char.islower() or char.isdigit() or char == "-"
            for char in name
        )
        assert body.strip()


def test_outfit_styling_references_the_shared_trend_snapshot() -> None:
    _, body = _read_skill()
    trend_text = TRENDS_PATH.read_text()

    assert trend_text.strip()
    assert "Not catalog truth" in trend_text
    assert "`/shopper/trends-current.md`" in body
    assert not (SKILL_PATH.parent / "trends-current.md").exists()


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


def test_primary_shopper_skills_are_mutually_exclusive() -> None:
    styling_frontmatter, styling_body = _read_skill()
    discovery_frontmatter, discovery_body = _read_skill_path(
        REGISTERED_SKILL_PATHS["product-discovery"]
    )

    assert "Use instead of product-discovery" in styling_frontmatter["description"]
    assert "do not combine it with product-discovery" in styling_body
    assert "Do not activate alongside outfit-styling" in (
        discovery_frontmatter["description"]
    )
    assert "throughout that active outfit-building thread" in (
        styling_frontmatter["description"]
    )
    assert "Do not use it inside an active outfit-building" in (
        discovery_frontmatter["description"]
    )
    assert "do not combine it with outfit-styling" in discovery_body
    assert "Budget-shopping may accompany it only as a modifier" in styling_body
    assert "Budget-shopping may accompany it only as a modifier" in discovery_body


def test_shopper_skills_keep_taxonomy_selection_semantic_and_catalog_owned() -> None:
    _, styling_body = _read_skill()
    _, discovery_body = _read_skill_path(
        REGISTERED_SKILL_PATHS["product-discovery"]
    )

    assert "Map the shopper's meaning to exact values advertised" in styling_body
    assert "generic fashion concepts, not catalog taxonomy" in styling_body
    assert "choose only exact advertised taxonomy values" in styling_body
    assert "choose only exact advertised taxonomy values" in discovery_body
    assert "Generic product language is not taxonomy evidence" in discovery_body
    assert "use the tool's `no_direct_catalog_match` taxonomy status" in (
        discovery_body
    )
    for body in (styling_body, discovery_body):
        assert "do not broaden to" in body.lower()
        assert "silently substitute" in body
        assert "separate the requested product type from its modifiers" in body
        assert "subjective style stays in the semantic query" in body
        assert "supported alternative branch must still be searched" in body


def test_primary_shopper_skills_cover_search_and_followup_contracts() -> None:
    _, styling_body = _read_skill()
    _, discovery_body = _read_skill_path(
        REGISTERED_SKILL_PATHS["product-discovery"]
    )

    assert "plus outfit intent is enough to begin helping" in styling_body
    assert "do not respond with only a questionnaire" in styling_body
    assert "## Mandatory Turn Boundary" in styling_body
    assert "Never answer only with clarification questions" in styling_body
    assert "Do not translate the vibe into an unadvertised product type" in (
        styling_body
    )
    assert 'such as "a statement piece,"' in styling_body
    assert "Ask one concise category or occasion question" in styling_body
    assert "whole or complete outfit remains incomplete" in styling_body
    assert 'a "rainy day outfit" or "wet-weather outfit" should start' in (
        styling_body
    )
    assert "Each search covers one catalog category" in styling_body
    assert "Each search covers at most one catalog category" in discovery_body
    assert '"Do you have water-resistant bags?"' in discovery_body
    assert "## Mandatory Constraint Boundary" in discovery_body
    assert "an empty object is not faithful" in discovery_body
    assert "Only wording that explicitly makes an attribute optional" in (
        discovery_body
    )
    for body in (styling_body, discovery_body):
        assert "explicitly requested concrete product type" in body
        assert "Never use it for an outfit, occasion, season, weather need" in body
        assert "in `required_constraints`" in body
        assert "multiple advertised enum values" in body
        assert "named antecedent" in body
        assert "shared confirmed constraint" in body


def test_outfit_styling_scopes_incremental_requests_and_substitutions() -> None:
    _, body = _read_skill()

    assert '"Start with X" means solve only X' in body
    assert "direct antecedent" in body
    assert "Search only the requested product scope" in body
    assert "explicitly requests multiple pieces" in body
    assert "one product role gets one inclusive search scope" in body
    assert "shared confirmed constraint" in body
    assert "do not copy beige into the bottoms' hard color filter" in body
    assert "do not require the shopper to select one exact top" in body.lower()
    assert "A dress is not a bottom" in body
    assert "Do not spend unused search budget on adjacent categories" in body
    assert "Do not search the alternative until the shopper accepts it" in body
    assert "Do not force a weak substitute" in body
    assert "Treat styling as multi-intent by default" not in body
    assert "same category first, adjacent category second" not in body


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
    assert "skill_registry = _shopper_skill_registry(skills_root)" in runtime_source
    assert "ShopperSkillActivationMiddleware(" in runtime_source
    assert "skills=[" not in runtime_source


def test_skill_registry_lists_all_registered_skill_files() -> None:
    registry = (REPO_ROOT / "docs" / "SHOPPER_AGENT_SKILL_REGISTRY.md").read_text()

    for name, path in REGISTERED_SKILL_PATHS.items():
        source = path.relative_to(REPO_ROOT).as_posix()
        assert f"| `{name}` | `{source}` | Registered |" in registry


def test_chain_server_docker_image_includes_skill_files() -> None:
    dockerfile = (REPO_ROOT / "chain_server" / "Dockerfile").read_text()

    assert "COPY ./skills ./skills" in dockerfile
