from pathlib import Path

import yaml


DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "integration"
    / "conversations"
    / "catalog_semantic_authority"
)


def test_catalog_semantic_authority_fixtures_are_isolated_and_structural() -> None:
    fixtures = sorted(DATASET_ROOT.glob("conv_*.yaml"))

    assert len(fixtures) == 4
    for fixture in fixtures:
        payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
        assert len(payload["queries"]) == 1
        assert len(payload["answers"]) == 1
        assert len(payload["diagnostic_expectations"]) == 1
        expectation = payload["diagnostic_expectations"][0]
        assert expectation["tool_call_counts"]["search_catalog_tool"] == 1
        assert len(expectation["tool_call_expectations"]) == 1


def test_catalog_semantic_authority_fixtures_cover_clean_boundaries() -> None:
    expectations = {}
    for fixture in DATASET_ROOT.glob("conv_*.yaml"):
        payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
        expectations[fixture.stem] = payload["diagnostic_expectations"][0][
            "tool_call_expectations"
        ][0]

    assert expectations["conv_direct_alternatives"]["arguments"]["taxonomy"] == {
        "category": ["footwear"],
        "subcategory": ["heels", "flats"],
    }
    assert expectations["conv_category_scope"]["arguments"]["taxonomy"] == {
        "category": ["footwear"],
        "subcategory": [],
    }
    assert expectations["conv_semantic_preference"]["arguments"][
        "required_constraints"
    ] == {}
    assert expectations["conv_unadvertised_requirement"][
        "rejection_reason"
    ] == "unsupported_catalog_constraint"
