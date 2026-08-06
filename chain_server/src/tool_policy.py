# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shopper-skill to tool policy."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml


ToolRisk = Literal["read", "mutating"]


@dataclass(frozen=True)
class ToolPolicy:
    """Execution policy for one registered shopping tool."""

    allowed_skills_any_of: frozenset[str]
    risk: ToolRisk
    required_intent_kind: str | None = None

    def allows_any(self, skill_names: Collection[str]) -> bool:
        """Return whether any active skill grants this tool."""

        return bool(self.allowed_skills_any_of.intersection(skill_names))


@dataclass(frozen=True)
class ShopperSkill:
    """Validated shopper-skill metadata and instructions."""

    path: str
    description: str
    response_guidance: str
    role: str
    exclusive_group: str | None
    tools_granted: frozenset[str]
    content: str


SHOPPING_TOOL_POLICIES: Mapping[str, ToolPolicy] = MappingProxyType(
    {
        "search_catalog_tool": ToolPolicy(
            allowed_skills_any_of=frozenset(
                {"outfit-styling", "product-discovery"}
            ),
            risk="read",
        ),
        "get_product_details_tool": ToolPolicy(
            allowed_skills_any_of=frozenset(
                {"outfit-styling", "product-discovery"}
            ),
            risk="read",
        ),
        "check_product_availability_tool": ToolPolicy(
            allowed_skills_any_of=frozenset(
                {"outfit-styling", "product-discovery"}
            ),
            risk="read",
        ),
        "check_active_promotions_tool": ToolPolicy(
            allowed_skills_any_of=frozenset(
                {"outfit-styling", "product-discovery"}
            ),
            risk="read",
        ),
        "resolve_conversation_products_tool": ToolPolicy(
            allowed_skills_any_of=frozenset(
                {
                    "cart-management",
                    "outfit-styling",
                    "product-discovery",
                }
            ),
            risk="read",
        ),
        "get_cart_tool": ToolPolicy(
            allowed_skills_any_of=frozenset({"cart-management"}),
            risk="read",
        ),
        "view_cart_total_tool": ToolPolicy(
            allowed_skills_any_of=frozenset({"cart-management"}),
            risk="read",
        ),
        "add_cart_items_tool": ToolPolicy(
            allowed_skills_any_of=frozenset({"cart-management"}),
            risk="mutating",
            required_intent_kind="cart_add",
        ),
        "remove_cart_item_tool": ToolPolicy(
            allowed_skills_any_of=frozenset({"cart-management"}),
            risk="mutating",
            required_intent_kind="cart_remove",
        ),
        "update_cart_items_tool": ToolPolicy(
            allowed_skills_any_of=frozenset({"cart-management"}),
            risk="mutating",
            required_intent_kind="cart_update",
        ),
        "get_store_policy_tool": ToolPolicy(
            allowed_skills_any_of=frozenset({"store-policy-answers"}),
            risk="read",
        ),
        # Granted to the styling skills only. A forecast is styling input --
        # it never establishes a product fact, so nothing else has a use for
        # it, and a shopper asking about returns should not be able to reach a
        # paid external service.
        "get_weather_forecast_tool": ToolPolicy(
            allowed_skills_any_of=frozenset(
                {"outfit-styling", "product-discovery"}
            ),
            risk="read",
        ),
    }
)


def load_shopper_skill_registry(
    skills_root: Path | None,
) -> dict[str, ShopperSkill]:
    """Load shopper skills and validate their policy frontmatter."""

    if skills_root is None:
        raise RuntimeError("Shopper skills root is unavailable.")

    registry: dict[str, ShopperSkill] = {}
    for skill_path in sorted((skills_root / "shopper").glob("*/SKILL.md")):
        content = skill_path.read_text(encoding="utf-8")
        if not content.startswith("---\n") or "\n---\n" not in content:
            raise RuntimeError(f"Invalid skill frontmatter: {skill_path.name}")
        frontmatter, body = content.removeprefix("---\n").split("\n---\n", 1)
        metadata = yaml.safe_load(frontmatter)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid shopper skill: {skill_path.parent.name}")
        skill = _shopper_skill_from_metadata(
            skill_path,
            metadata,
            content,
            body,
        )
        registry[skill_path.parent.name] = skill
    if not registry:
        raise RuntimeError("No registered shopper skills were found.")
    validate_skill_tool_grants(
        {
            name: skill.tools_granted
            for name, skill in registry.items()
        }
    )
    return registry


def _shopper_skill_from_metadata(
    skill_path: Path,
    metadata: dict,
    content: str,
    body: str,
) -> ShopperSkill:
    """Validate one shopper skill's frontmatter."""

    name = str(metadata.get("name") or "")
    description = _metadata_text(metadata, "description")
    response_guidance = _metadata_text(metadata, "response_guidance")
    role = str(metadata.get("role") or "")
    exclusive_group = _metadata_text(metadata, "exclusive_group") or None
    raw_tools_granted = metadata.get("tools_granted")
    tools_granted = (
        raw_tools_granted
        if isinstance(raw_tools_granted, list)
        and all(
            isinstance(tool_name, str) and tool_name.strip()
            for tool_name in raw_tools_granted
        )
        else None
    )
    if (
        name != skill_path.parent.name
        or not description
        or not response_guidance
        or role not in {"primary", "modifier", "standalone"}
        or (role == "primary") != bool(exclusive_group)
        or tools_granted is None
        or len(tools_granted) != len(set(tools_granted))
        or not body.strip()
    ):
        raise RuntimeError(f"Invalid shopper skill: {skill_path.parent.name}")
    return ShopperSkill(
        path=f"/shopper/{name}/SKILL.md",
        description=description,
        response_guidance=response_guidance,
        role=role,
        exclusive_group=exclusive_group,
        tools_granted=frozenset(tools_granted),
        content=content,
    )


def _metadata_text(metadata: dict, name: str) -> str:
    value = metadata.get(name)
    return value.strip() if isinstance(value, str) else ""


def validate_registered_tool_names(tool_names: Collection[str]) -> None:
    """Require the runtime and policy registry to name the same tools."""

    registered = frozenset(tool_names)
    managed = frozenset(SHOPPING_TOOL_POLICIES)
    if registered != managed:
        missing = sorted(managed - registered)
        unexpected = sorted(registered - managed)
        raise ValueError(
            "Shopping tool policy does not match registered tools: "
            f"missing={missing}, unexpected={unexpected}"
        )


def validate_skill_tool_grants(
    skill_tool_grants: Mapping[str, Collection[str]],
) -> None:
    """Require skill frontmatter and execution policy to agree exactly."""

    actual = {
        (skill_name, tool_name)
        for skill_name, tool_names in skill_tool_grants.items()
        for tool_name in tool_names
    }
    expected = {
        (skill_name, tool_name)
        for tool_name, policy in SHOPPING_TOOL_POLICIES.items()
        for skill_name in policy.allowed_skills_any_of
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Shopper skill tool grants do not match execution policy: "
            f"missing={missing}, unexpected={unexpected}"
        )


def granted_tools_for_skills(
    selected_skills: Collection[str],
    skill_tool_grants: Mapping[str, Collection[str]],
) -> frozenset[str]:
    """Return the union of tools declared by the selected skills."""

    unknown = sorted(set(selected_skills).difference(skill_tool_grants))
    if unknown:
        raise ValueError(f"Unknown shopper skills: {unknown}")
    return frozenset(
        tool_name
        for skill_name in selected_skills
        for tool_name in skill_tool_grants[skill_name]
    )


def tool_is_granted(
    tool_name: str,
    selected_skills: Collection[str],
    granted_tools: Collection[str],
) -> bool:
    """Apply frontmatter grants and execution policy independently."""

    policy = SHOPPING_TOOL_POLICIES.get(tool_name)
    if policy is None:
        return True
    return tool_name in granted_tools and policy.allows_any(selected_skills)
