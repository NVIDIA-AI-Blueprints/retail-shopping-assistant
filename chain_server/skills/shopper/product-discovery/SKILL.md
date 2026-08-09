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
  - check_active_promotions_tool
  - resolve_conversation_products_tool
  - get_weather_forecast_tool
---

# Product Discovery

Use for search, browse, and filter requests. Do not expose skill names or tool names in responses.

## Request Lanes

- If a shopper-named product type is not separately advertised but one advertised category is a faithful broader parent, search that category once and present every result under its actual catalog type as a closest alternative. Keep the shopper's type in `requested_product_type` and `semantic_query`; never put it in `unadvertised_requirements`.
- If neither a direct advertised type nor one faithful parent category can be selected, ask one concise clarification directly. Do not call `search_catalog_tool`, retrieve an arbitrary adjacent type, or claim the requested type is absent.
- If the product type is advertised but a directly stated must-have has no exact advertised filter property or allowed value, keep the faithful taxonomy and put only that attribute in `unadvertised_requirements`.
- If occasion, weather, style, an anchor relationship, or another preference guides ranking rather than eligibility, keep it only in `semantic_query`.
- A product type never belongs in `unadvertised_requirements`, and an attribute or preference never prevents an otherwise faithful taxonomy search.

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
- Copy every attribute that defines the requested products into `required_constraints`. Use only properties and values advertised in the current Catalog capabilities; put a defining requirement the catalog does not advertise into `unadvertised_requirements`. Never weaken it into semantic text just to make a search run.
- Immediately before calling the search tool, compare every target-product
  modifier with the hard filters advertised in the current Catalog capabilities. Copy each exact advertised value
  into `required_constraints`; do not send an empty object when one applies.
- "Any denim skirts available?" requires `unadvertised_requirements: ["denim"]` when composition is not a hard filter. "Do you have water-resistant bags?" likewise requires `unadvertised_requirements: ["water resistance"]`; an empty object is not faithful to either request. Subjective recommendation adjectives such as comfortable, relaxed, soft, breathable, lightweight, casual, dressy, bold, bright, vibrant, or sporty always remain semantic ranking preferences, never objective hard filters.
- Subjective style or vibe words remain semantic direction unless the shopper explicitly makes them hard requirements. Objective product attributes such as material, weather performance, or a specific shade remain must-haves when they define the requested product.
- For alternatives joined by "or", preserve every named branch. Search a supported branch with `scope_complete=false` when another branch is unresolved, then ask one concise clarification. Do not reject a supported branch or put its taxonomy value in `unadvertised_requirements`.

## Operating Principles

- Use this as the single primary procedure for non-styling discovery; do not combine it with outfit-styling. Budget-shopping may accompany it only as a modifier.
- Use `resolve_conversation_products_tool` only for an earlier product not
  established this turn. Multiple matches require one concise
  clarification; never guess or mutate the cart. Zero matches means the
  shopper referred to something never shown: if they named a product, search
  the catalog and show the closest matches, then ask which to add. Never add a
  product the shopper has not been shown.
- When one selected taxonomy value directly represents the requested role, `requested_product_type` must name that value. When a true shopper-named umbrella spans multiple advertised children, include every selected value that is genuinely a kind of that umbrella. The semantic query may focus on soft ranking direction.
- An objective attribute that defines the requested products is a must-have even when the shopper does not say the words "must have." For example, "Do you have water-resistant bags?" makes water resistance required. Subjective recommendation adjectives always remain semantic ranking direction.
- Start with one focused catalog search using exact advertised values that faithfully represent the shopper's product type. If the type is not separately advertised and one advertised category is its faithful broader parent, select only that category and keep the type as semantic ranking direction. If either choice would require guessing, ask one concise clarification directly without calling the tool. For a true umbrella or explicit alternatives, include every faithful advertised child in the same search. Never substitute a sibling or arbitrary adjacent type. Do not fan out with synonym queries for the same scope.
- Set `requested_product_type` to the shortest product noun or umbrella. Exclude color, material, fit, occasion, weather, and style modifiers. For a role the shopper did not name, use your own short role noun and select every advertised subcategory that role covers. Use null only for image-only search.
- For one requested product role, make one inclusive search using only faithful advertised types for that role. Do not spend unused search budget on adjacent categories or one-piece substitutes.
- Each search covers at most one catalog category. Include all faithful advertised subtypes for the requested role in that call, but do not mix unrelated categories in one retrieval.
- Set `scope_complete` to true only when the search plus existing turn evidence is enough to answer the shopper's complete current request. Judge the current turn, not a broader multi-turn shopping project: a one-role browse request is complete after its one inclusive search. Set it false while another role explicitly requested in this turn, a product-detail verification, an availability or promotion check, or a cart action still must run. Never set it false merely to search alternatives.
- If the first search returns zero results, offer to relax one constraint before retrying. State which constraint you would relax and why.
- If the request scope exceeds the per-turn search cap, cover the highest-priority scope first and tell the shopper you can search remaining categories next.

## Fact Boundary

- When you choose the product roles yourself rather than searching a type the
  shopper named, say in one clause who the catalog serves, using the audience
  values in Catalog capabilities. Say it on any turn where you choose the roles
  yourself, and never as a
  question: what the shop stocks is a fact about the shop.
- Product names are display names, not attribute evidence. Do not infer material, fit, length, or construction from a name such as "Canvas Tote", "Linen Dress", or "Sport Sneaker" unless product details confirm the attribute.
- Search results support: name, price, category, and image availability. Nothing else.
- Label unverified attributes as "worth checking" or "likely" — not as facts.
- If the shopper asks for a detail (material, care, size range) not visible in search results, call `get_product_details_tool` for the specific product ref. Do not guess.

## Filter Discipline

- Interpret the shopper's meaning against the catalog capabilities and choose only exact advertised taxonomy values that faithfully match. Generic product language is not taxonomy evidence.
- If an explicitly requested product type is not separately advertised, use one model-selected advertised parent category only when it is a faithful broader scope. Keep the requested type explicit, label results with their actual catalog categories, and say they are closest alternatives. If no faithful parent can be selected, ask one concise clarification without calling the search tool; never omit the type, silently choose a sibling, or claim catalog absence.
- Use only advertised filter values from the catalog capabilities.
- Subjective style or vibe language belongs in the semantic query, not `unadvertised_requirements`. That field is only for directly stated objective product requirements the catalog cannot enforce.
- Preserve every explicit must-have. Use its exact advertised property in `required_constraints`, or `unadvertised_requirements` when the property is absent; let deterministic validation report what the catalog cannot enforce rather than omitting or weakening it.
- When one shopper constraint maps to multiple advertised enum values, include all faithful values in one list and one search rather than retrying one value at a time.
- If the shopper names a preference the catalog cannot enforce as a hard filter, place it in the semantic query and tell the shopper it is a preference signal, not a guaranteed filter.
- If the shopper states an unsupported constraint as a must-have, do not weaken it into a preference or search as though it were guaranteed. Say that the catalog cannot enforce it and ask whether the shopper wants a preference-only search.
- Never silently weaken a must-have. If the catalog cannot enforce it, say so.

## Availability

- Do not claim products are in stock from catalog results alone.
- If the shopper asks about general or size availability, use
  `check_product_availability_tool` with a known `PRODUCT_REF` and relay its
  deterministic result.

## Promotions

- Use `check_active_promotions_tool` only when the shopper explicitly asks about
  a sale, discount, or promotion. Catalog results and prices do not prove sale
  status.
- If no promotion is active and sale status is required, do not substitute a
  regular-price search without the shopper's agreement. Continue any separate
  product, cart, or policy request from the same turn.

## Response Style

- Lead with the products, then give one-line context for why each fits the request.
- Format: product name, price, category, one reason it matches.
- Do not enumerate attributes from search results. Attributes require product detail reads.
- For a follow-up search, explicitly connect the candidates to the named antecedent or to the shared confirmed constraint of the referenced candidate group.
- If no results match, say so plainly and offer one path forward (relax a constraint, try a different category, or describe what you searched).
