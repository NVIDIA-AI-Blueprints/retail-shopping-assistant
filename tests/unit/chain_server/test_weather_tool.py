# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the forecast tool is registered, and which skills may grant it."""

from __future__ import annotations

from pathlib import Path

from chain_server.src.tool_policy import SHOPPING_TOOL_POLICIES

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_weather_tool_is_registered_on_every_serving_surface() -> None:
    """The inverse of the guard this replaces.

    The tool was built complete and deliberately unregistered, with this test
    holding it dormant across policy, runtime and every skill. Activation has
    to happen on all of those surfaces at once -- the runtime validates that
    registered tool names exactly equal the policy, and skill frontmatter must
    match it -- so the check is kept and turned around rather than deleted.
    """

    assert "get_weather_forecast_tool" in SHOPPING_TOOL_POLICIES

    runtime = (REPO_ROOT / "chain_server/src/deepagents_runtime.py").read_text()
    policy = (REPO_ROOT / "chain_server/src/tool_policy.py").read_text()

    assert "get_weather_forecast_tool" in runtime
    assert "get_weather_forecast_tool" in policy


def test_only_skills_that_use_a_forecast_reach_a_paid_external_service() -> None:
    """Two skills have a use for a forecast, and a shopper asking about
    returns must not be able to spend a provider call.

    The grant is not free: 4,270 characters of schema -- the second largest in
    the system -- on every call a granting skill covers, for a tool called on
    1.8% of turns. product-discovery held it and should not have; a browse is
    the non-styling procedure.

    Narrowing it to outfit-styling alone was wrong in the other direction,
    because the grant decides which turns *can* fetch a forecast and not only
    which turns pay for the schema. "Going to Cancun next week, what's the
    weather like" asks for no outfit, selects no styling procedure, and so
    could not see the tool at all -- the reply said no forecast was available
    and then described the climate from memory. A bare conditions question is
    its own task and holds this one tool, which is also the cheapest place for
    it: such a turn loads a fifth of what the browse procedure costs.
    """

    granted = {
        path.parent.name
        for path in (REPO_ROOT / "chain_server/skills/shopper").glob("*/SKILL.md")
        if "get_weather_forecast_tool" in path.read_text()
    }

    assert granted == {"destination-weather"}
    assert SHOPPING_TOOL_POLICIES["get_weather_forecast_tool"].risk == "read"
