import csv
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / "tests" / "evaluation"
STYLE_GUIDE_PATH = EVAL_ROOT / "datasets" / "style_guide" / "scenarios.yaml"
IMAGE_SHOPPING_PATH = EVAL_ROOT / "datasets" / "image_shopping" / "scenarios.yaml"
PRODUCTS_PATH = REPO_ROOT / "shared" / "data" / "products.csv"


def test_style_guide_dataset_covers_styling_entry_modes_and_patterns():
    scenarios = _load_scenarios(STYLE_GUIDE_PATH)

    assert {scenario["entry_mode"] for scenario in scenarios} >= {
        "anchor_product",
        "no_anchor_discovery",
        "cart_styling",
        "mid_browse_styling",
    }
    assert {scenario["secondary_entry_pattern"] for scenario in scenarios} >= {
        "occasion_first",
        "constraint_first",
        "product_page_anchor",
        "cart_completion",
        "conversational_mid_browse",
        "comparison_decision",
        "wardrobe_gap",
        "post_selection_refinement",
    }

    low_coupling = [
        scenario
        for scenario in scenarios
        if scenario["catalog_dependency"]["level"] in {"behavior_only", "category_level"}
    ]
    assert len(low_coupling) >= 6

    for scenario in scenarios:
        assert scenario["skill_focus"]
        assert scenario["catalog_dependency"]["refresh_note"]
        assert scenario["success_criteria"]
        assert scenario["failure_modes"]


def test_low_coupling_style_scenarios_do_not_name_exact_seed_products():
    scenarios = _load_scenarios(STYLE_GUIDE_PATH)
    product_names = _catalog_product_names()

    for scenario in scenarios:
        level = scenario["catalog_dependency"]["level"]
        if level not in {"behavior_only", "category_level"}:
            continue
        scenario_text = yaml.safe_dump(scenario, sort_keys=False)
        exact_mentions = [name for name in product_names if name in scenario_text]
        assert exact_mentions == [], scenario["id"]


def test_cart_styling_scenario_seeds_cart_before_gap_check():
    scenarios = _load_scenarios(STYLE_GUIDE_PATH)
    cart_scenario = next(
        scenario for scenario in scenarios if scenario["id"] == "style_cart_build_then_gap_check"
    )
    sequence = cart_scenario["turn_sequence"]

    assert len(sequence) >= 5
    assert "find" in sequence[0].lower()
    assert "add" in sequence[1].lower()
    assert "find" in sequence[2].lower()
    assert "add" in sequence[3].lower()
    assert "cart as an outfit" in sequence[4].lower()


def test_visual_styling_image_scenarios_mark_catalog_asset_dependency():
    scenarios = {scenario["id"]: scenario for scenario in _load_scenarios(IMAGE_SHOPPING_PATH)}

    for scenario_id in ["image_find_dress_and_style", "image_top_layering_question"]:
        scenario = scenarios[scenario_id]
        assert scenario["entry_mode"] == "visual_anchor"
        assert scenario["catalog_dependency"]["level"] == "visual_seed_asset"
        assert scenario["catalog_dependency"]["stable_under_catalog_refresh"] is False
        assert scenario["success_criteria"]
        assert scenario["failure_modes"]


def _load_scenarios(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    scenarios = data["scenarios"]
    assert isinstance(scenarios, list)
    return scenarios


def _catalog_product_names() -> list[str]:
    with PRODUCTS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row["name"]
            for row in csv.DictReader(handle)
            if row.get("name") and len(row["name"]) > 4
        ]
