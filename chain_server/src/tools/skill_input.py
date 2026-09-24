# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The selection model the skill activation tool validates against."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .schemas import _a_list_written_as_json_text
from .skill_gate import (
    SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
    SKILL_ACTIVATION_MULTIPLE_PRIMARY,
)


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
