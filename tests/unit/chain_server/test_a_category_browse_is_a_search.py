"""Naming a category and no subcategory is a complete request.

J17 t1, the first turn of the journey:

    "I'm buying a gift, show me some jewellery"
    sent      taxonomy {"category": ["jewelry"]}
    rejected  capabilities_schema_mismatch -> "Field required"
    reply     "Could you clarify the product type or requirement you want me to use?"

Nothing was shown and the journey failed from there. The model was right not to
narrow: the shopper named no subcategory, and picking bracelets over necklaces
would have been choosing for them.

Four review notes are this same rejection -- J10 t2, J16 t4, J17 t1, and the
skirts turn earlier in the week.
"""

import json
import urllib.request

import pytest

from chain_server.src.catalog_capabilities import CatalogCapabilities
from chain_server.src.turn_support import _search_catalog_tool_input_model


@pytest.fixture(scope="module")
def model():
    raw = json.load(
        urllib.request.urlopen("http://localhost:8010/capabilities", timeout=20)
    )
    return _search_catalog_tool_input_model(CatalogCapabilities.model_validate(raw))


BASE = {
    "semantic_query": "gift",
    "shopper_guidance": "Here are some jewellery pieces.",
    "requested_product_type": "jewellery",
    "required_constraints": {},
    "scope_complete": True,
    "search_mode": "text",
    "taxonomy_status": "exact_requested_type",
}


def test_a_category_with_no_subcategory_is_accepted(model) -> None:
    parsed = model.model_validate({**BASE, "taxonomy": {"category": ["jewelry"]}})
    assert parsed.taxonomy.category == ["jewelry"]
    assert parsed.taxonomy.subcategory == [], "omitted means the whole category"


def test_a_narrowed_search_still_works(model) -> None:
    parsed = model.model_validate(
        {**BASE, "taxonomy": {"category": ["jewelry"], "subcategory": ["bracelets"]}}
    )
    assert parsed.taxonomy.subcategory == ["bracelets"]


def test_a_search_with_no_taxonomy_at_all_is_still_refused(model) -> None:
    """Relaxing the field must not relax the rule. A text search still needs
    somewhere to look."""

    with pytest.raises(Exception) as caught:
        model.model_validate({**BASE, "taxonomy": {}})
    assert "category or subcategory" in str(caught.value)


def test_neither_role_is_required_by_the_schema(model) -> None:
    taxonomy = model.model_json_schema()["$defs"]["CatalogTaxonomySelection"]
    assert not taxonomy.get("required")


def test_a_price_ceiling_with_no_category_is_a_complete_request(model) -> None:
    """J10 t1, "I'm shopping on a tight budget, nothing over $50".

    The shopper named no category, no subcategory and no product type, so
    clothes, shoes and accessories are all on the table and the price is the
    whole scope. The search was refused twice over -- once for naming no
    taxonomy, once as an "open-role search" -- and the assistant answered
    "could you clarify the product type", asking them to narrow a request that
    was already complete.

    The open-role rule exists so a role the model INVENTED cannot be mapped
    onto subcategories silently. An absent role invents nothing.
    """

    assert model.model_validate(
        {
            **BASE,
            "semantic_query": "anything under fifty dollars",
            "requested_product_type": "items",
            "taxonomy_status": "agent_selected_type",
            "taxonomy": {"category": [], "subcategory": []},
            "required_constraints": {"price": {"max": 50}},
        }
    )


def test_no_taxonomy_and_no_filter_is_still_refused(model) -> None:
    """Nothing to search on at all. The relaxation is for a scoped request."""

    with pytest.raises(Exception):
        model.model_validate(
            {
                **BASE,
                "semantic_query": "show me things",
                "requested_product_type": "items",
                "taxonomy_status": "agent_selected_type",
                "taxonomy": {"category": [], "subcategory": []},
                "required_constraints": {},
            }
        )


def test_an_unenforceable_requirement_does_not_scope_a_search(model) -> None:
    """unadvertised_requirements is the field for what the catalog CANNOT
    enforce, so it narrows nothing and must not license an unscoped search."""

    with pytest.raises(Exception):
        model.model_validate(
            {
                **BASE,
                "semantic_query": "something waterproof",
                "requested_product_type": "items",
                "taxonomy_status": "agent_selected_type",
                "taxonomy": {"category": [], "subcategory": []},
                "required_constraints": {
                    "unadvertised_requirements": ["waterproof"]
                },
            }
        )
