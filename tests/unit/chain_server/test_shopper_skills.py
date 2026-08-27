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
        "catalog-questions",
        "store-policy-answers",
    )
}
SKILL_PATH = REGISTERED_SKILL_PATHS["outfit-styling"]
TRENDS_PATH = SHOPPER_SKILLS_ROOT / "trends-current.md"
EXPECTED_SKILL_POLICY = {
    "budget-shopping": {
        "role": "modifier",
        "exclusive_group": None,
        "tools_granted": [],
    },
    "cart-management": {
        "role": "standalone",
        "exclusive_group": None,
        "tools_granted": [
            "get_cart_tool",
            "add_cart_items_tool",
            "remove_cart_item_tool",
            "update_cart_items_tool",
            "view_cart_total_tool",
            "resolve_conversation_products_tool",
        ],
    },
    "outfit-styling": {
        "role": "primary",
        "exclusive_group": "product_procedure",
        "tools_granted": [
            "search_catalog_tool",
            "get_product_details_tool",
            "check_product_availability_tool",
            "check_active_promotions_tool",
            "resolve_conversation_products_tool",
            "get_weather_forecast_tool",
        ],
    },
    "product-discovery": {
        "role": "primary",
        "exclusive_group": "product_procedure",
        "tools_granted": [
            "describe_catalog_tool",
            "search_catalog_tool",
            "get_product_details_tool",
            "check_product_availability_tool",
            "check_active_promotions_tool",
            "resolve_conversation_products_tool",
            "get_weather_forecast_tool",
        ],
    },
    "catalog-questions": {
        "role": "primary",
        "exclusive_group": "product_procedure",
        "tools_granted": [
            "describe_catalog_tool",
            "search_catalog_tool",
            "get_product_details_tool",
        ],
    },
    "store-policy-answers": {
        "role": "standalone",
        "exclusive_group": None,
        "tools_granted": ["get_store_policy_tool"],
    },
}


def _read_skill_path(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    assert text.startswith("---\n")
    frontmatter_text, body = text.removeprefix("---\n").split("\n---\n", 1)
    return yaml.safe_load(frontmatter_text), body


def _read_skill() -> tuple[dict, str]:
    return _read_skill_path(SKILL_PATH)


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
        assert 0 < len(frontmatter["response_guidance"]) <= 1024
        assert frontmatter["role"] == EXPECTED_SKILL_POLICY[name]["role"]
        assert frontmatter.get("exclusive_group") == (
            EXPECTED_SKILL_POLICY[name]["exclusive_group"]
        )
        assert frontmatter["tools_granted"] == (
            EXPECTED_SKILL_POLICY[name]["tools_granted"]
        )
        assert 0 < len(name) <= 64
        assert not name.startswith("-")
        assert not name.endswith("-")
        assert "--" not in name
        assert all(
            char.islower() or char.isdigit() or char == "-"
            for char in name
        )
        assert body.strip()


def test_the_trend_snapshot_is_inline_and_not_a_dangling_pointer() -> None:
    """It pointed at a file `read_file` could fetch, and never did.

    Zero reads across 37 archived runs. `read_file` is no longer registered --
    the base prompt says there is no filesystem and now that is true -- so a
    pointer would be unreachable rather than merely unused. The guidance that
    earned its place is inline; the 7.7k of tables it came from is not.
    """

    _, body = _read_skill()
    normalized = " ".join(body.lower().split())

    assert not TRENDS_PATH.exists(), "an unreachable file is still on disk"
    assert "/shopper/trends-current.md" not in body, "dangling pointer"
    assert "trends" in normalized
    assert "never catalog truth" in normalized
    assert "fundamentals and the shopper's own preferences outrank it" in normalized
    assert "broadly wearable" in normalized


def test_primary_skill_descriptions_define_the_activation_boundary() -> None:
    styling_frontmatter, _ = _read_skill()
    discovery_frontmatter, _ = _read_skill_path(
        REGISTERED_SKILL_PATHS["product-discovery"]
    )

    assert "Use instead of product-discovery" in styling_frontmatter["description"]
    assert "active outfit-building thread" in styling_frontmatter["description"]
    assert "Do not activate alongside outfit-styling" in (
        discovery_frontmatter["description"]
    )


def test_product_discovery_separates_request_lanes(base_config) -> None:
    """Assert the model reads these rules, not which file holds them.

    They used to be five bullets in the skill saying what the search tool's own
    field descriptions and the catalog rules already said. Asserting the skill
    text made the duplication load-bearing: the test would fail the moment a
    rule moved to the channel that binds the decision, which is the direction
    you want rules to move.
    """

    from chain_server.src import deepagents_runtime as runtime_mod

    from .model_visible import reachable_on_a_turn_using

    normalized = reachable_on_a_turn_using(
        runtime_mod.DeepAgentsRuntime(base_config), "product-discovery"
    )

    assert "ask one concise clarification" in normalized
    assert "do not call the tool at all" in normalized
    # The blanket "never claim absence" contradicted `not_covered`, whose
    # whole purpose is to record a kind the catalog does not carry. Absence
    # read off the published taxonomy is a fact; absence guessed from a thin
    # search is the thing that was meant to be banned.
    assert "the catalog does not carry it" in normalized
    assert "absence guessed from a search that returned little" in normalized
    assert "unadvertised_requirements" in normalized
    assert "keep it only in `semantic_query`" in normalized
    assert "product type never belongs" in normalized.lower()


def test_outfit_styling_owns_domain_judgment_and_clarification() -> None:
    _, body = _read_skill()
    normalized = " ".join(body.lower().split())

    for responsibility in (
        "anchor",
        "conversation continuity",
        "one concise question",
        "color",
        "proportion",
        "silhouette",
        "formality",
        "occasion",
        "texture",
        "styling judgment",
        "response style",
    ):
        assert responsibility in normalized

    assert "what bottoms go with that?" in normalized
    assert "do not invent a product category" in normalized
    assert "do not turn the anchor's" in normalized
    assert "same or a matching value" in normalized
    assert "do not dump an unexplained product list" in normalized


def test_outfit_styling_does_not_own_catalog_transport_or_cart_execution() -> None:
    _, body = _read_skill()

    for transport_field in (
        "requested_product_type",
        "taxonomy_status",
        "scope_complete",
        "unadvertised_requirements",
        "agent_selected_type",
        "search_mode",
    ):
        assert transport_field not in body

    assert "cart reads or mutations" in body
    assert "does not own cart totals" in body


def test_skill_bodies_reference_only_tools_they_grant() -> None:
    shopping_tool_names = {
        tool_name
        for policy in EXPECTED_SKILL_POLICY.values()
        for tool_name in policy["tools_granted"]
    }

    for name, path in REGISTERED_SKILL_PATHS.items():
        frontmatter, body = _read_skill_path(path)
        mentioned_tools = {
            tool_name
            for tool_name in shopping_tool_names
            if tool_name in body
        }

        assert mentioned_tools <= set(frontmatter["tools_granted"]), name


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


def test_role_proposing_skills_name_the_audience_without_asking() -> None:
    """Choosing a dress is a styling call; assuming the range applies is not.

    A live turn returned three dress-led outfits and disclosed only the role it
    had picked -- "you didn't specifically ask for dresses" -- while saying
    nothing about who the shop serves. The disclosure that went unsaid is the
    one deciding whether any of it applies to the shopper.
    """

    root = Path(__file__).resolve().parents[3] / "chain_server/skills/shopper"

    for skill in ("outfit-styling", "product-discovery"):
        raw = (root / skill / "SKILL.md").read_text()
        # Normalised: the guidance is wrapped, so a phrase can span a newline.
        body = " ".join(raw.split())
        assert "who the catalog serves" in body, skill
        assert "audience values in Catalog capabilities" in body, skill
        assert "never as a question" in body, skill
        # The values live in the catalog. Naming one here would survive a
        # catalog swap and keep being stated after it stopped being true.
        for value in ("womens", "adult_all_genders", "womenswear", "menswear"):
            assert value not in body, f"{skill} hardcodes {value}"


def test_no_rule_is_stated_in_two_skills_at_once(base_config) -> None:
    """Cross-skill duplication is what made the prune necessary.

    Before it, outfit-styling and product-discovery ended with three
    byte-identical bullets each -- availability, promotions, tax and delivery --
    and shared twenty-five exact sentences in all. A rule in two skills has two
    places to drift and no way to tell which one the model followed.

    Sentence-level and exact, so it catches copy-paste without objecting to two
    skills happening to use the same short phrase.
    """

    import re

    bodies = {}
    for name, path in REGISTERED_SKILL_PATHS.items():
        _, body = _read_skill_path(path)
        text = " ".join(body.split())
        bodies[name] = {
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?]) ", text)
            if len(sentence.split()) >= 12
        }

    seen: dict[str, str] = {}
    duplicated: list[str] = []
    for name, sentences in sorted(bodies.items()):
        for sentence in sentences:
            if sentence in seen and seen[sentence] != name:
                duplicated.append(f"{seen[sentence]} + {name}: {sentence[:80]}")
            seen.setdefault(sentence, name)

    assert not duplicated, "sentences stated in two skills:\n" + "\n".join(
        duplicated
    )
