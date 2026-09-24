# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool that loads shopper skills, and the gate built around it."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from pydantic import ValidationError
from shared.commerce_contracts import CatalogCapabilities

from ..agenttypes import State
from ..fencing import MEDIA_FENCE
from ..identity import RequestIdentity
from .catalog import catalog_prompt_section
from .loop_control import ToolLoopControlMiddleware
from .policy import SHOPPING_TOOL_POLICIES, load_shopper_skill_registry
from .skill_gate import (
    SKILL_ACTIVATION_COMPLETE,
    ShopperSkillActivationMiddleware,
)
from .skill_input import _skill_activation_input_model
from .weather import forecast_prompt_section


def build_skill_activation(
    state: State,
    identity: RequestIdentity,
    skills_root: Path,
    turn_capabilities: CatalogCapabilities,
    tool_loop_control: ToolLoopControlMiddleware,
):
    """The activation tool and the gate middleware that enforces what it
    granted, built from the skills registered under `skills_root`."""

    from langchain_core.tools import tool

    skill_registry = load_shopper_skill_registry(skills_root)
    skill_activation_input = _skill_activation_input_model(skill_registry)

    def _widen_for_tool(
        tool_name: str,
        selected: Collection[str],
    ) -> tuple[list[str], dict[str, str]] | None:
        """Name a legal selection that grants this tool, or nothing.

        The gate asks this when a turn is refused a tool for the grant.
        The registry and the selection model both live here, so the two
        things the answer needs -- which skills grant the tool, and
        whether adding one is a selection the model would have been
        allowed to make -- are answered in one place.

        `skill_activation_input` is the same model the activation tool
        validates against, so a widening that would seat two primaries
        from one exclusive group, or strand a modifier, is rejected here
        for exactly the reason it would have been rejected there.
        """

        policy = SHOPPING_TOOL_POLICIES.get(tool_name)
        if policy is None:
            return None
        current = list(dict.fromkeys(selected))
        for candidate in sorted(policy.allowed_skills_any_of):
            if candidate in current or candidate not in skill_registry:
                continue
            names = [*current, candidate]
            try:
                skill_activation_input(skill_names=names)
            except ValidationError:
                continue
            return names, {
                skill_registry[name].path: skill_registry[name].content
                for name in names
            }
        return None

    skill_gate = ShopperSkillActivationMiddleware(
        request_id=identity.request_id,
        skill_descriptions={
            name: skill.description
            for name, skill in skill_registry.items()
        },
        skill_tool_grants={
            name: skill.tools_granted
            for name, skill in skill_registry.items()
        },
        previous_selected_skills=state.previous_selected_skill_names,
        granted_tool_context={
            "search_catalog_tool": catalog_prompt_section(
                turn_capabilities
            ),
            "get_weather_forecast_tool": forecast_prompt_section(),
        },
        spent_tool_context=tool_loop_control.spent_tool_context,
        widen_for_tool=_widen_for_tool,
        activation_system_prompt=(
            MEDIA_FENCE.notice if state.media_analysis else ""
        ),
    )

    @tool(args_schema=skill_activation_input, return_direct=False)
    def activate_shopper_skills_tool(
        skill_names: list[str],
    ) -> str:
        """Select and load shopper behavior skills for this turn. This is
        the required first step before answering or calling shopping tools.
        Select the smallest set whose registered descriptions cover the
        complete current intent.

        Which primary to pick is answered by the registered descriptions
        themselves, and by the allowed values on `skill_names`. This
        docstring used to answer it again in its own words, naming two of
        the primaries; a third was registered and the list here did not
        know, so the one skill that could answer a question about the shop
        was never offered as an option. Read the descriptions.

        What they cannot tell you, because it spans two of them: dressing
        for a named place and date needs the conditions there, and
        `outfit-styling` cannot fetch them: select `destination-weather`
        with it whenever
        the turn turns on the weather. "A wedding in Rome in June, what
        should I wear" needs both. It is a standalone skill, neither a
        second primary nor a modifier, so selecting it beside a procedure
        is allowed. Leave it out and the turn has no way to know the
        weather -- and the failure that follows is not a refusal, it is a
        reply describing a climate it never fetched.
        """

        selected_names = list(dict.fromkeys(skill_names))
        try:
            selected_files = {
                skill_registry[name].path: skill_registry[name].content
                for name in selected_names
            }
            activated = skill_gate.activate(selected_files, selected_names)
        except (KeyError, ValueError):
            skill_gate.fail()
            return (
                "SHOPPER_SKILL_ACTIVATION_FAILED: Registered skill "
                "instructions could not be loaded."
            )
        if not activated:
            return "SHOPPER_SKILL_ACTIVATION_ALREADY_COMPLETE"
        return (
            f"{SKILL_ACTIVATION_COMPLETE} "
            + ", ".join(selected_files)
        )

    activate_shopper_skills_tool.handle_validation_error = (
        skill_gate.handle_activation_validation_error
    )

    return activate_shopper_skills_tool, skill_gate
