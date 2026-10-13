# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool that loads shopper skills, and the gate built around it."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from shared.commerce_contracts import CatalogCapabilities

from ..agenttypes import State
from ..fencing import MEDIA_FENCE
from ..runtime.identity import RequestIdentity
from .catalog import catalog_prompt_section
from .loop_control import ToolLoopControlMiddleware
from .policy import SHOPPING_TOOL_POLICIES, load_shopper_skill_registry
from .schemas import _a_list_written_as_json_text
from .skill_gate import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
    SKILL_ACTIVATION_MULTIPLE_PRIMARY,
    InputGuardrailToolGate,
    ShopperSkillActivationMiddleware,
)
from .weather import forecast_prompt_section

# The selection model the skill activation tool validates against.

class _ShopperSkillActivationInput(BaseModel):
    """Shared composition rules for dynamic shopper-skill activation.

    The composition rule itself lives on the subclass `create_model` builds,
    because it depends on which skills are registered and what roles they
    declare. This base carries only what every registry shares.
    """

    model_config = ConfigDict(extra="forbid")

    skill_names: list[str]

    # check_fields, because the real field is declared by the create_model
    # subclass that narrows it to the registered skill names.
    _accept_skill_names_as_text = field_validator(
        "skill_names", mode="before", check_fields=False
    )(_a_list_written_as_json_text)


def primary_skills_by_group(
    skills: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Group the registry's primary skills by the group they are exclusive in.

    Read off `role` and `exclusive_group` in the frontmatter, which
    `_shopper_skill_from_metadata` already requires to agree: a skill is
    primary if and only if it names a group.
    """

    groups: dict[str, list[str]] = {}
    for name, skill in skills.items():
        if getattr(skill, "role", "") != "primary":
            continue
        group = getattr(skill, "exclusive_group", None)
        if group:
            groups.setdefault(str(group), []).append(name)
    return {group: tuple(sorted(names)) for group, names in groups.items()}


def _one_primary_per_group(self: Any) -> Any:
    """Reject two primaries from one exclusive group, or a stranded modifier.

    Groups are read from each skill's declared ``exclusive_group`` rather than
    a list of skill names, so a newly registered primary is checked the same
    way as the existing ones.
    """

    cls = type(self)
    groups: Mapping[str, tuple[str, ...]] = cls._primary_skills_by_group
    selected = set(self.skill_names)
    primaries: list[str] = []
    for _group, names in sorted(groups.items()):
        chosen = sorted(selected.intersection(names))
        if len(chosen) > 1:
            # Told only the rule, the model resent the same selection; given
            # the lists to send, it sent one of them.
            valid = [
                json.dumps(
                    [n for n in self.skill_names if n not in chosen or n == keep]
                )
                for keep in chosen
            ]
            raise PydanticCustomError(
                SKILL_ACTIVATION_MULTIPLE_PRIMARY,
                "select exactly one primary procedure, never more than one. "
                "Send one of: {options}",
                {"options": " or ".join(valid)},
            )
        primaries.extend(chosen)

    modifiers: tuple[str, ...] = cls._modifier_skills
    stranded = sorted(selected.intersection(modifiers))
    if stranded and not primaries:
        raise PydanticCustomError(
            SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
            "{modifier} is a modifier and requires exactly one primary "
            "procedure: {options}",
            {
                "modifier": stranded[0],
                "options": " or ".join(
                    name for names in sorted(groups.values()) for name in names
                ),
            },
        )
    return self


def _skill_activation_input_model(
    skills: Mapping[str, Any],
) -> type[BaseModel]:
    """Create the semantic skill-selection schema from the active registry."""

    skill_names = tuple(skills)
    groups = primary_skills_by_group(skills)
    modifiers = tuple(
        sorted(
            name
            for name, skill in skills.items()
            if getattr(skill, "role", "") == "modifier"
        )
    )
    # Named here rather than in a literal, so a skill added to the registry is
    # described to the model without anyone remembering to edit this string.
    every_primary = [name for names in sorted(groups.values()) for name in names]
    choose_one = (
        " Select exactly one primary procedure -- "
        + ", ".join(every_primary)
        + " -- and never two."
        if every_primary
        else ""
    )
    modifier_rule = (
        " " + ", ".join(modifiers) + " may only accompany a primary, never "
        "stand alone."
        if modifiers
        else ""
    )
    model = create_model(
        "ShopperSkillActivationInput",
        __base__=_ShopperSkillActivationInput,
        __validators__={
            "_one_primary_per_group": model_validator(mode="after")(
                _one_primary_per_group
            ),
        },
        skill_names=(
            list[Literal.__getitem__(skill_names)],
            Field(
                ...,
                min_length=1,
                max_length=len(skill_names),
                description=(
                    "Smallest set of registered shopper skills whose descriptions "
                    "cover the current turn's complete intent." + choose_one
                    + modifier_rule
                    + " Standalone skills may be selected with or without a "
                    "primary."
                ),
            ),
        ),
    )
    model._primary_skills_by_group = groups
    model._modifier_skills = modifiers
    return model


def build_skill_activation(
    state: State,
    identity: RequestIdentity,
    skills_root: Path,
    turn_capabilities: CatalogCapabilities,
    tool_loop_control: ToolLoopControlMiddleware,
    input_guardrail_tool_gate: InputGuardrailToolGate | None = None,
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
        input_guardrail_tool_gate=input_guardrail_tool_gate,
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
