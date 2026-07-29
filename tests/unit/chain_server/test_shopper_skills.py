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
        "event-context",
        "outfit-styling",
        "product-discovery",
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
    "event-context": {
        "role": "modifier",
        "exclusive_group": None,
        "tools_granted": ["get_weather_forecast_tool"],
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
        ],
    },
    "product-discovery": {
        "role": "primary",
        "exclusive_group": "product_procedure",
        "tools_granted": [
            "search_catalog_tool",
            "get_product_details_tool",
            "check_product_availability_tool",
            "check_active_promotions_tool",
            "resolve_conversation_products_tool",
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


def test_outfit_styling_references_the_shared_trend_snapshot() -> None:
    _, body = _read_skill()
    trend_text = TRENDS_PATH.read_text()

    assert trend_text.strip()
    assert "Not catalog truth" in trend_text
    assert "`/shopper/trends-current.md`" in body
    assert not (SKILL_PATH.parent / "trends-current.md").exists()


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


def test_product_discovery_separates_request_lanes() -> None:
    _, body = _read_skill_path(REGISTERED_SKILL_PATHS["product-discovery"])
    normalized = " ".join(body.split())

    assert "## Request Lanes" in body
    assert "ask one concise clarification directly" in normalized
    assert "Do not call `search_catalog_tool`" in normalized
    assert "claim the requested type is absent" in normalized
    assert "put only that attribute in `unadvertised_requirements`" in normalized
    assert "keep it only in `semantic_query`" in normalized
    assert "A product type never belongs in `unadvertised_requirements`" in normalized


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
    assert "`event-context` response gate overrides" in normalized
    assert "give one short direction paragraph and ask no further" in normalized
    assert "ask another styling question only if it is material" in normalized
    assert "ordinary shop-now occasion request" in normalized


def test_outfit_styling_compares_prior_products_through_current_evidence() -> None:
    _, body = _read_skill()
    normalized = " ".join(body.lower().split())

    assert "## compare established products" in normalized
    assert "not a request to search for those products again" in normalized
    assert "newest historical-product index" in normalized
    assert "every compared product together" in normalized
    assert "single `resolve_conversation_products_tool` call" in normalized
    assert "never invent a fuzzy alias" in normalized
    assert "ambiguous or not found" in normalized
    assert "once for each uniquely resolved product_ref" in normalized
    assert "separate model steps" in normalized
    assert "compare only item-specific confirmed fields" in normalized
    assert "weather is optional additional evidence" in normalized
    assert "never a substitute for product resolution or detail reads" in normalized


def test_event_context_is_a_narrow_weather_styling_modifier() -> None:
    frontmatter, body = _read_skill_path(
        REGISTERED_SKILL_PATHS["event-context"]
    )
    normalized = " ".join(body.lower().split())

    assert frontmatter["role"] == "modifier"
    assert frontmatter["tools_granted"] == ["get_weather_forecast_tool"]
    assert "only with `outfit-styling`" in body
    assert "an explicit destination overrides saved zip" in normalized
    assert "cancun does not mean beach" in normalized
    assert "ask at most one short question, never a questionnaire" in normalized
    assert "`event_location` only when destination is missing" in normalized
    assert "do not ask more than one in a turn" in normalized
    assert "do not append dress code, time of day, product role" in normalized
    assert "## context authority" in normalized
    assert "## one-question policy" in normalized
    assert "## forecast lookup" in normalized
    assert "## response mode" in normalized
    assert "## evidence boundary" in normalized
    assert "exactly two short sentences with no heading or list" in normalized
    assert "one short paragraph of at most four sentences" in normalized
    assert "begin with one grounded requested or core product role" in normalized
    assert "with saved zip as the only clue" in normalized
    assert "without saved zip, ask the destination directly" in normalized
    assert "do not re-ask established context as a finer variant" in normalized
    assert "invent hypothetical exceptions" in normalized
    assert "context fulfillment, not a new catalog request" in normalized
    assert "preserve prior candidates and do not search again" in normalized
    assert "product tools granted by other selected skills remain available" in (
        normalized
    )
    assert "a weather attempt does not close the tool loop" in normalized
    assert "call `get_weather_forecast_tool` once before answering" in normalized
    assert "a typed success or failure completes the attempt" in normalized
    assert "no current non-weather business-tool activity exists" in normalized
    assert "a current typed weather outcome" in normalized
    assert "separate empty-draft fallback" in normalized
    assert "sand-friendly" in normalized
    assert (
        "use saved zip for a forecast only after the shopper explicitly "
        "confirms"
    ) in normalized
    assert (
        "an explicit destination forbids fallback to saved zip"
    ) in normalized
    assert "valid forecast location authority" in normalized
    assert "keep its shortest sufficient phrase exactly" in normalized
    assert "a separate `location_query`" in normalized
    assert "must preserve that exact phrase as its first component" in normalized
    assert "one or two comma-separated" in normalized
    assert "a common abbreviation such as `nyc` remains unchanged" in normalized
    assert "ambiguous name such as `springfield`" in normalized
    assert "`springfield, tx`" in normalized
    assert "never derive a representative zip" in normalized
    assert "add an unstated numeric component" in normalized
    assert "call `get_weather_forecast_tool` at most once in a turn" in normalized
    assert (
        "use `confirmed_saved_zip` only after explicit usual-area confirmation"
    ) in normalized
    assert (
        "use `shopper_provided_location` with the exact shopper phrase in "
        "`location`"
    ) in normalized
    assert "supply an exact iso event date or complete inclusive range" in normalized
    assert "use `relative_date=next_week`" in normalized
    assert "next calendar monday-through-sunday window" in normalized
    assert "visual crossing resolves the shopper's phrase" in normalized
    assert "not as proof of shopper intent" in normalized
    assert "make no weather claim" in normalized
    assert "live forecast is not available yet" in normalized
    assert "only successful current-turn forecast evidence" in normalized
    assert "weather data provided by visual crossing" in normalized
    assert "forecasts can change" in normalized
    assert "never proves that a catalog item is" in normalized
    assert "never creates an unstated product must-have" in normalized
    response_guidance = frontmatter["response_guidance"].lower()
    assert "explicit location overrides saved zip" in response_guidance
    assert 'never ask "usual area" afterward' in response_guidance
    assert "saved zip is tentative" in response_guidance
    assert '"usual area or elsewhere?"' in response_guidance
    assert "without a candidate, ask destination" in response_guidance
    assert "ask one question maximum" in response_guidance
    assert 'bare "next week" means the full window' in response_guidance
    assert "this helper is additive" in response_guidance
    assert "never suppresses product tools" in response_guidance
    assert "an explicit comparison, refinement, or new-product request" in (
        response_guidance
    )
    assert "occasion-only shop-now: one core-role search" in response_guidance
    assert "not a complete look" in response_guidance
    assert "preserve the canonical forecast" in response_guidance
    assert "visual crossing attribution" in response_guidance
    assert "change warning" in response_guidance
    assert "weather cannot prove product performance" in response_guidance
    assert "create an unstated constraint" in response_guidance
    assert "generic advice is possible" in frontmatter["description"]
    assert "do not use for location-independent styling" in (
        frontmatter["description"].lower()
    )
    assert "do not echo its digits" in normalized
    assert "get_weather_forecast_tool" in body
    assert "run exactly one catalog search for one useful core role" in normalized


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
