# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "conversations"
    / "product_comparison"
)


def test_product_comparison_dataset_is_one_three_turn_gate() -> None:
    files = sorted(DATASET_ROOT.glob("conv_*.yaml"))
    assert [path.name for path in files] == ["conv_prior_products.yaml"]

    conversation = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert "shopper_profile_id" not in conversation
    assert (
        len(conversation["queries"])
        == len(conversation["answers"])
        == len(conversation["diagnostic_expectations"])
        == 3
    )

    comparison = conversation["diagnostic_expectations"][2]
    assert comparison["required_skills"] == ["outfit-styling"]
    assert comparison["required_tools"] == [
        "resolve_conversation_products_tool",
        "get_product_details_tool",
    ]
    assert comparison["tool_call_counts"] == {
        "resolve_conversation_products_tool": 1,
        "search_catalog_tool": 0,
    }
    assert comparison["required_product_detail_names"] == [
        "Intricate Lace Gown",
        "Wavy Hem Satin Dress",
    ]
    assert "search_catalog_tool" in comparison["forbidden_tools"]
