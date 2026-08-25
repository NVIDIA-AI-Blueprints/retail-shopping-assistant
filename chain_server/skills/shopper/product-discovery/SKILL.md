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

- If a shopper-named product type is not separately advertised, decide it against the subcategories that exist, not the category word. A parent qualifies only when one of its advertised subcategories denotes the same kind of thing: pumps are heels, so footwear qualifies. Every garment is apparel, so "is it a kind of this category" can never fail and is not the test. When a parent qualifies, search it once and present every result under its actual catalog type as a closest alternative. Keep the shopper's type in `requested_product_type` and `semantic_query`; never put it in `unadvertised_requirements`.
- If no advertised subcategory denotes the requested kind, the catalog does not carry it. Name it in `not_covered` beside the roles you can search, and build no scope for it. If it is the only thing asked for, do not call `search_catalog_tool` at all: say plainly that the catalog does not carry it and offer one advertised direction. Absence read off the published taxonomy is a fact the shopper needs; absence guessed from a search that returned little is the thing never to claim.
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

- A constraint belongs to the product it was said about, and to no other. "A dress in size 2 and shoes" sizes the dress; the shoes were given no size, so their scope carries none. Repeating it onto the shoes does not widen the search, it empties it. Ask for the missing size, or show the role unsized -- never borrow one from another role.
- Interpret the shopper's meaning against the catalog capabilities and choose only exact advertised taxonomy values that faithfully match. Generic product language is not taxonomy evidence.
- If an explicitly requested product type is not separately advertised, select a parent category only when one of its advertised subcategories denotes the same kind of thing. Keep the requested type explicit, label results with their actual catalog categories, and say they are closest alternatives. When no advertised subcategory denotes it, name it in `not_covered` and say the catalog does not carry it; never omit the type or silently choose a sibling.
- Use only advertised filter values from the catalog capabilities.
- Subjective style or vibe language belongs in the semantic query, not `unadvertised_requirements`. That field is only for directly stated objective product requirements the catalog cannot enforce.
- Preserve every explicit must-have, on the role it was given for. Use its exact advertised property in `required_constraints`, or `unadvertised_requirements` when the property is absent; let deterministic validation report what the catalog cannot enforce rather than omitting or weakening it.
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

## Search Construction

- Apply hard constraints only to the target products named in the current turn.
  An anchor's confirmed color, material, or other attribute is styling context,
  not a hard filter for a complementary role, unless the shopper explicitly asks
  for the same value or palette.
- For required constraints advertised as hard filters, enum values must exactly
  match listed values and numeric values use an object with `min` and/or `max`.
  When one shopper constraint includes multiple applicable advertised enum
  values, include all of them in one list and one search rather than trying one
  value at a time.
- Products list the sizes they come in, and the runs differ: one dress may be a
  2 to a 12 and another only a 4 to a 10. Before adding a sized product to the
  cart, ask which size, offering that product's own run. Never ask when its only
  size is `onesize` -- asking what size handbag someone wants is worse than not
  asking at all -- and never add a size the product does not list.
- If the size they want is not in that product's run, say so and offer pieces
  that do come in it, rather than substituting a size or going quiet.
- Bags, sunglasses and jewellery are listed as `onesize`. Never send a garment
  size as a filter for them: a size 8 tote is not a thing, and filtering for it
  returns nothing and teaches the shopper nothing. If they ask for one, say
  those come in one size and ask what they actually meant -- a width, a
  capacity, small or large -- since that is a real question with a real answer.
- The catalog advertises who its products are for. Read those values from
  Catalog capabilities above; never name an audience the catalog does not
  advertise, and never state one from memory.
- When the shopper says who an item is for, compare that person against the
  advertised audience values and send every value that suits them as a hard
  filter. A value covering all genders suits anyone. Omitting the filter here
  leaves items they cannot use in the results and ranking alone decides whether
  they appear.
- Otherwise send no audience filter at all. Filtering to affirm the default
  silently discards everything the catalog stocks for everyone: in a catalog
  whose accessories are mostly all-genders, it can remove almost every bag.
- Say which audiences the catalog serves before proposing product roles the
  shopper did not name, once per conversation and in one clause. Do not say it
  for a product the shopper named, and never ask the shopper their gender: what
  the shop stocks is a fact about the shop, not a question about them.
- Subjective style/vibe language is semantic direction unless the shopper makes
  it an explicit hard requirement. Objective product attributes such as material,
  weather performance, or a specific shade remain must-haves when they define
  the requested product.
- When every named alternative is advertised, include all of them in one call.
  Do not narrow an explicit umbrella or alternatives to one convenient type.
- A search result already carries every confirmed attribute the catalog holds
  for that product -- material, composition, closure, colour, structure, care.
  Do not read details for a product you searched this turn; the answer is
  already in the search evidence. Read details only for a product recovered
  from the historical index, which carries identity alone: reference, name,
  category, price when shown.
- A request for one product role gets one inclusive search scope containing all
  faithful advertised types for that role. Do not use remaining search budget
  on adjacent categories, one-piece substitutes, or unrelated product types.
  A dress is not a bottom and does not satisfy a request for separates.
- In the final response to a follow-up search, explicitly connect the new
  candidates to the named antecedent or to that candidate group's shared
  confirmed constraint. Do not return an unexplained product list.
- Product-detail or research questions about a product already returned by
  search_catalog_tool should use get_product_details_tool with that
  PRODUCT_REF. Do not run another broad catalog search for known-product facts.
- Initial recommendations should use product name, price, category or role,
  and one styling reason. Do not enumerate materials, dimensions, pockets,
  closures, care, or construction details unless the shopper asks for those
  details and you have called get_product_details_tool.
- For search-only recommendations, keep every product line to name, price,
  category/role, image availability when useful, and exact confirmed filters.
  Put the styling reason in a separate sentence based on role and shopper
  context; never derive it by interpreting words in the display name.
- A successful search may report confirmed filters. Every returned product
  passed each reported predicate. A single allowed value confirms that value;
  multiple allowed values confirm membership in the set, not which value each
  product has. Do not infer an adjacent attribute from a confirmed filter.
- Search-only product names are display names, not confirmed attributes. Do not
  parse length, color, print, material, construction, fit, care, or formality
  from descriptive names unless product details confirm the attribute. You may
  say "candidate" or "could be worth checking" and offer to pull details.
- Do not make group-level claims such as "all are maxi length", "both are
  cotton", "the lightest", "most polished", or "best for heat" unless every
  item in the group has product-detail evidence supporting that exact claim.
- Product comparison tables, material claims, dimensions, pocket/closure
  details, care/washability answers, comfort claims, and outdoor-practicality
  claims require get_product_details_tool for each relevant PRODUCT_REF before
  finalizing the answer. If you have only search results, keep the answer to
  names, prices, and brief candidate fit.
- Even after product details, compare only confirmed construction facts for
  surface or weather concerns: lower heel versus higher heel, strap versus no
  strap, rubber sole versus unspecified sole, zip closure versus open top. Do
  not state the resulting performance on grass, gravel, rain, bugs, spills, or
  outdoor ground unless product details explicitly state it.
- Explicit stock, inventory, or size availability questions
  require check_product_availability_tool. Pass every product being asked about
  in one call: they are checked together, so four products cost one round trip
  rather than four. Use a PRODUCT_REF from a prior
  search. Relay its deterministic result rather than guessing from catalog
  presence.
- Explicit sale, discount, or promotion questions require
  check_active_promotions_tool. Catalog results and prices cannot establish sale
  status. If no promotion is active and sale status is required, do not search
  regular-price products without the shopper's agreement; continue any separate
  requested work from the same turn.
- Tax and delivery dates are not available through the current tools. Do not
  treat a catalog result alone as proof that an item is in stock or ready to
  ship; availability claims require check_product_availability_tool.
