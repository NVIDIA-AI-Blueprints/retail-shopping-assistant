---
name: product-discovery
description: General product search, filter-driven browsing, and category exploration as the primary procedure for non-styling requests. Use for standalone search and browse requests. Do not use it inside an active outfit-building or styling thread, even when the current turn asks for one product type. Do not activate alongside outfit-styling; budget-shopping may accompany it only as a modifier.
response_guidance: These candidates match the searched product role and any confirmed constraints shown below. Check details before relying on material, fit, care, or other product-specific attributes.
role: primary
exclusive_group: product_procedure
tools_granted:
  - search_catalog_tool
  - get_product_details_tool
  - check_product_availability_tool
---

# Product Discovery

Use for search, browse, and filter requests. Do not expose skill names or tool names in responses.

## Mandatory Constraint Boundary

- Before every search, split the product noun from its modifiers.
  `requested_product_type` contains only the shortest shopper-named product noun
  or umbrella: "formal tops" and "relaxed-fit tops" both use `tops`; color,
  material, fit, occasion, weather, and style words stay in constraints or the
  semantic query.
- Build hard constraints only from the current turn's target product. Earlier
  conversation context may guide semantic relevance but does not create a new
  hard requirement unless the shopper repeats it for the current target.
- Set `shopper_guidance` to one concise, product-agnostic sentence that restates
  how this search scope serves the shopper's request. Write it before results
  are known; never name or describe candidate products or product types outside
  the selected advertised scope in it.
- Copy every attribute that defines the requested products into `required_constraints`. Use only properties present in its current schema; put a defining requirement the catalog does not advertise into `unadvertised_requirements`. Never weaken it into semantic text just to make a search run.
- Immediately before calling the search tool, compare every target-product
  modifier with the advertised filter schema. Copy each exact advertised value
  into `required_constraints`; do not send an empty object when one applies.
- "Any denim skirts available?" requires `unadvertised_requirements: ["denim"]` when composition is not a hard filter. "Do you have water-resistant bags?" likewise requires `unadvertised_requirements: ["water resistance"]`; an empty object is not faithful to either request. Subjective recommendation adjectives such as comfortable, relaxed, soft, breathable, lightweight, casual, dressy, bold, bright, vibrant, or sporty always remain semantic ranking preferences, never objective hard filters.
- Subjective style or vibe words remain semantic direction unless the shopper explicitly makes them hard requirements. Objective product attributes such as material, weather performance, sale status, or a specific shade remain must-haves when they define the requested product.
- For alternatives joined by "or", preserve every named branch. Search a supported branch with `scope_complete=false` when another branch is unavailable, then report the unavailable branch with one `no_direct_catalog_match` call. Do not reject a supported branch because another alternative is unavailable, and do not put the supported taxonomy value in `unadvertised_requirements`.

## Operating Principles

- Use this as the single primary procedure for non-styling discovery; do not combine it with outfit-styling. Budget-shopping may accompany it only as a modifier.
- Use `exact_requested_type` when the selected taxonomy directly represents the requested focused role. For a single selected value, the requested type must name that value; the semantic query may focus on soft ranking direction. Prefer `member_of_requested_umbrella` when a true shopper-named umbrella spans multiple advertised children, and include only values that are genuinely kinds of that umbrella.
- On `no_direct_catalog_match`, leave required constraints empty and never copy the requested product type into `unadvertised_requirements`.
- An objective attribute that defines the requested products is a must-have even when the shopper does not say the words "must have." For example, "Do you have water-resistant bags?" makes water resistance required. Subjective recommendation adjectives always remain semantic ranking direction.
- Start with one focused `search_catalog_tool` call. Use `exact_requested_type` only when every selected value means the same product type the shopper named. Use `member_of_requested_umbrella` when the shopper names a true umbrella or explicit alternatives, and include every faithful advertised child in the same call; do not reverse that relationship because the requested type belongs to a broader selected category. Apply one direction test to every value: "is this a kind of the product scope the shopper asked for?" A skirt is a kind of bottom; a flat or sandal is not a kind of sneaker even though all are footwear. If a broad browse request names no product type, use `agent_selected_type` with exactly one advertised subcategory as the focused starting role. Evaluate every role in the current turn independently: if the shopper names a product type for that role, including as an alternative, confirmation, comparison, or follow-up, `agent_selected_type` is forbidden. Before using `no_direct_catalog_match`, separate the requested product type from its modifiers: an unavailable attribute does not erase an advertised type, subjective style stays in the semantic query, and a supported alternative branch must still be searched. Use no-direct only when an explicitly requested concrete product type has no direct match, with empty taxonomy arrays and no required constraints so no retrieval runs. Never use it for an outfit, occasion, season, weather need, style/vibe, or product attribute. Do not fan out with synonym queries for the same scope.
- Set `requested_product_type` to the shortest product noun or umbrella. Exclude color, material, fit, occasion, weather, and style modifiers. For `agent_selected_type`, use the chosen advertised role noun. Use null only for image-only search.
- For one requested product role, make one inclusive search using only faithful advertised types for that role. Do not spend unused search budget on adjacent categories or one-piece substitutes.
- Each search covers at most one catalog category. Include all faithful advertised subtypes for the requested role in that call, but do not mix unrelated categories in one retrieval.
- Set `scope_complete` to true only when the search plus existing turn evidence is enough to answer the shopper's complete current request. Judge the current turn, not a broader multi-turn shopping project: a one-role browse request is complete after its one inclusive search. Set it false while another role explicitly requested in this turn, a product-detail verification, an availability check, or a cart action still must run. Never set it false merely to search alternatives. A requested type with no faithful advertised taxonomy match does not make a one-role scope partial; report that gap without another search.
- If the first search returns zero results, offer to relax one constraint before retrying. State which constraint you would relax and why.
- If the request scope exceeds the per-turn search cap, cover the highest-priority scope first and tell the shopper you can search remaining categories next.

## Fact Boundary

- Product names are display names, not attribute evidence. Do not infer material, fit, length, or construction from a name such as "Canvas Tote", "Linen Dress", or "Sport Sneaker" unless product details confirm the attribute.
- Search results support: name, price, category, and image availability. Nothing else.
- Label unverified attributes as "worth checking" or "likely" — not as facts.
- If the shopper asks for a detail (material, care, size range) not visible in search results, call `get_product_details_tool` for the specific product ref. Do not guess.

## Filter Discipline

- Interpret the shopper's meaning against the catalog capabilities and choose only exact advertised taxonomy values that faithfully match. Generic product language is not taxonomy evidence.
- If an explicitly requested product type has no direct advertised match, use the tool's `no_direct_catalog_match` taxonomy status and report that gap. Do not broaden to a parent category, omit the type, or silently substitute another type.
- Use only advertised filter values from the catalog capabilities.
- Subjective style or vibe language belongs in the semantic query, not `unadvertised_requirements`. That field is only for directly stated objective product requirements the catalog cannot enforce.
- Preserve every explicit must-have. Use its exact advertised property in `required_constraints`, or `unadvertised_requirements` when the property is absent; let deterministic validation report what the catalog cannot enforce rather than omitting or weakening it.
- When one shopper constraint maps to multiple advertised enum values, include all faithful values in one list and one search rather than retrying one value at a time.
- If the shopper names a preference the catalog cannot enforce as a hard filter, place it in the semantic query and tell the shopper it is a preference signal, not a guaranteed filter.
- If the shopper states an unsupported constraint as a must-have, do not weaken it into a preference or search as though it were guaranteed. Say that the catalog cannot enforce it and ask whether the shopper wants a preference-only search.
- Never silently weaken a must-have. If the catalog cannot enforce it, say so.

## Availability

- Do not claim products are in stock. Catalog results are not inventory signals.
- If the shopper asks about availability, use `check_product_availability_tool`.

## Response Style

- Lead with the products, then give one-line context for why each fits the request.
- Format: product name, price, category, one reason it matches.
- Do not enumerate attributes from search results. Attributes require product detail reads.
- For a follow-up search, explicitly connect the candidates to the named antecedent or to the shared confirmed constraint of the referenced candidate group.
- If no results match, say so plainly and offer one path forward (relax a constraint, try a different category, or describe what you searched).
