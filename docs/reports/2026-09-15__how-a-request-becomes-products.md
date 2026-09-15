# How a shopper request becomes products

**Written** 2026-09-15 · **Branch** `fix/a-relaxation-keeps-the-garment` (`c7aa66d`)
**Purpose** A reading of the actual algorithm, layer by layer, so the bug classes we
keep re-fixing can be argued about from the code rather than from symptoms.

Every claim here was checked against the code on this branch, and the behavioural
claims against live traces or live service probes. Where something is inferred
rather than verified, it says so. Optimisation opportunities are recorded in
[Part 8](#part-8--optimisations-noticed-not-done) and deliberately not acted on.

### Which code this describes

This branch is `origin/staging` (`51c3a76`) plus five commits, so most of what
follows is true of staging too. Three mechanisms are not, and a reader on
staging should discount them:

| Mechanism | staging | this branch | deployed container |
| --- | --- | --- | --- |
| `_garment_with_no_advertised_value`, the unconditional carried-check | present | present | **absent** |
| `roles_not_advertised` | present | present | **absent** |
| `repeat_lock`, per-turn tool-call dedupe | present | present | **absent** |
| `_relaxed_alternatives` / `_NEVER_RELAXED`, speculative relaxation | **present** | deleted | **present** |

The last row is the one that misleads: this report describes the speculative
relaxation as deleted, and that deletion exists only here. It is still in
staging and still in the running container, which is why substituting a skirt
for an unstocked pair of jeans can still be observed against a live server.

The deployed container is older than staging — a 12 September build, verified by
grepping the markers above inside the running image rather than inferred from a
tag. Nothing in this report should be used to predict what the live UI does
until the server is rebuilt.

---

## Contents

1. [The foundation: what the catalog is](#part-1--the-foundation-what-the-catalog-is)
2. [How the catalog is leveraged: capabilities become the tool schema](#part-2--how-the-catalog-is-leveraged)
3. [A turn, end to end](#part-3--a-turn-end-to-end)
4. [The search pipeline, step by step](#part-4--the-search-pipeline)
5. [Every "not found", with examples](#part-5--every-not-found)
6. [Resolving a reference to something shown earlier](#part-6--resolving-a-reference)
7. [Why the UI, the reply and the transcript disagree](#part-7--why-the-ui-the-reply-and-the-transcript-disagree)
8. [Optimisations noticed, not done](#part-8--optimisations-noticed-not-done)
9. [Open bugs, ranked](#part-9--open-bugs-ranked)

---

## Part 1 — The foundation: what the catalog is

### 1.1 Storage

Products live in **Milvus**. The code provides for two collections
(`catalog_retriever/src/retriever.py`), but only one is built here:

| Collection | Holds | Metric | Live? |
|---|---|---|---|
| `text_collection` | text embeddings of product prose | `COSINE` | **yes** — `shopping_advisor_text_db`, 215 rows |
| `image_collection` | image embeddings | `COSINE` | **no** — `self.image_db = None` unless `image_enabled` |

Verified against the running database: `list_collections()` returns
`shopping_advisor_text_db` alone, and `/capabilities` reports
`retrieval_modes: ['text']` with `image_search_enabled: False`. So image search is
built but switched off, and every retrieval in this deployment is text.

Both are created with `index_params={"metric_type": "COSINE"}`, and the relevance
score follows `langchain-milvus`'s COSINE contract (there is a comment at
`retriever.py:323` pinning this deliberately). A search is a vector query with an
optional Milvus **boolean filter expression** (`expr=...`) applied as a hard
pre-filter, so metadata filters are exact set membership, not ranking hints.

### 1.1a Where the metadata actually lives

The declared schema has **three columns only** — `pk`, `text` (the embedded prose)
and `vector` (2048 dims). Every product attribute is a key in Milvus's **dynamic
field**, the collection being created with `enable_dynamic_field: True`. That is why
`subcategory in ["dresses"]` is a legal expression against a column no schema
declares. One real dress, read back from the live collection:

| Key | Value | Shape |
|---|---|---|
| `category`, `subcategory` | `apparel`, `dresses` | the taxonomy filters |
| `primary_color` | `pink` | **one scalar per product**, not a list |
| `sizes` | `["2","4","6","8","10","12"]` | list → `json_contains_any` |
| `price` | `"119.99"` | **string**, so it cannot push down |
| `pattern`, `silhouette`, `neckline`, `sleeve_length`, `closure`, `composition`, `target_audience` | `solid`, `a_line`, `v_neck`, `long`, `button`, `cotton`, `womens` | filterable attributes |
| `description`, `enriched_description`, `text` | prose | `text` is the embedded one |
| `image`, `url` | `/images/Pastel_Pink_Peasant_Dress.jpg` | |
| `pk`, `record_id`, `source_row`, `catalog_fingerprint` | | index bookkeeping |

Two consequences worth carrying into the colour discussion in 4.1. `price` being
text is exactly why numbers are decided in Python rather than by the database. And
`primary_color` being a single scalar means a garment is filed under **one**
advertised colour — so `primary_color in ["beige","white"]` is membership against
that one value, and a wrong mapping does not rank the product lower, it excludes it.

### 1.2 There is no hand-authored taxonomy

This is the most important fact in the whole system and the source of several
bugs. The taxonomy is **derived from the product rows at index time**, by grouping:

```
catalog_retriever/src/capabilities.py :: _taxonomy_capabilities
    category_field    = schema.taxonomy.fields[0]
    subcategory_field = schema.taxonomy.fields[1]
    for category in _group_products(products, category_field):
        for subcategory in _group_products(category_products, subcategory_field):
            ...
```

So a "category" is nothing more than *a distinct value of the category column*.
Nobody decided the shop sells "footwear"; 4 groups of shoes happened to carry that
string. Live, as of today:

| Category | Subcategories |
|---|---|
| `apparel` | blouses, camisoles, dresses, jumpsuits, skirts, sweaters |
| `bags` | clutches, crossbody_bags, satchels, shoulder_bags, tote_bags, travel_bags |
| `eyewear` | sunglasses |
| `footwear` | boots, flats, heels, sandals |
| `jewelry` | bracelets, earrings, necklaces |

Two consequences we keep paying for:

- **There is no hierarchy above "category".** No node says *footwear is what people
  call shoes*, or *skirts are a kind of bottom*. Umbrella words shoppers actually
  use — "tops", "bottoms", "shoes" — resolve to **nothing**, verified live:

  | word | `_advertised_scope_match` |
  |---|---|
  | `shoes` | **None** (the column says `footwear`) |
  | `top` / `tops` | **None** |
  | `bottoms` | **None** |
  | `boots` | `boots` |
  | `footwear` | `footwear` |

- **The categories are wildly uneven.** `apparel` is 39 skirts, 33 dresses, 18
  sweaters, 9 blouses, 2 camisoles and 1 jumpsuit. So "apparel" is a department,
  not a family — which is exactly why ranking `"dark blue straight leg jeans"`
  across apparel returns skirts. This reasoning is written out in the docstring of
  `_reconciled_with_what_is_advertised`, and it is correct.

### 1.3 Filters are advertised value sets

For each category and subcategory, `_scoped_fields` computes, per field, whether it
is `filterable`, `searchable`, and its coverage. A field with `coverage.present > 0`
and `filterable` becomes a **filter with a finite advertised value list** —
`primary_color: [beige, black, blue, ...]`, `sizes: [2, 4, ...]`, and so on.

That finite list is the whole basis of validation downstream. "Is cream a colour
this shop can filter on?" is a set membership test, answered locally, with no model
call.

---

## Part 2 — How the catalog is leveraged

### 2.1 Capabilities become the tool schema, per turn

`chain_server` fetches `/capabilities` and builds the `search_catalog_tool` input
model **from the live taxonomy** (`_search_catalog_tool_input_model`). The
`subcategory` field is a **Literal enum of the real subcategories**, so an invalid
value is rejected by Pydantic before any of our code runs. That is why the trace
for a jeans scope reads:

```
literal_error — Input should be 'blouses', 'boots', 'bracelets', 'camisoles',
'clutches', 'crossbody_bags', 'dresses', 'earrings', 'flats', 'heels',
'jumpsuits', 'necklaces', 'sandals', 'satchels', 'shoulder_bags', 'skirts',
'sunglasses', 'sweaters', 'tote_bags' or 'travel_bags'
```

Worth knowing: fields redefined with live enums **replace the whole `FieldInfo`,
description included**. There is a comment on `taxonomy` and `required_constraints`
in `turn_support.py` recording that prose written on the static model never reaches
the model — it sat there for months saying things the live schema did not say.

### 2.2 The shape of a search request

The model sends `scopes`, a list. Each scope is one **role**:

```jsonc
{
  "semantic_query": "cream cable-knit sweater",   // ranked, never filtered
  "shopper_guidance": "...",                      // shopper-facing sentence
  "requested_product_type": "sweater",            // provenance, not taxonomy
  "taxonomy": {"category": ["apparel"], "subcategory": ["sweaters"]},
  "required_constraints": {"primary_color": ["beige", "white"]},
  "scope_complete": true
}
```

Multi-role requests ("shop this look") arrive as several scopes in **one** call,
each searched independently and merged. This matters for bug reading: one tool call
can produce one accepted scope and one refused scope, and the refusal text for
scope 2 sits inside the same tool result as products for scope 1.

---

## Part 3 — A turn, end to end

```
shopper text (+ optional image/video)
        │
        ▼
 ┌─ MEDIA PERCEPTION ─────────────────────────────────────────┐
 │ media_perception.py :: MediaPerceptionClient.analyze()     │
 │  · skipped entirely when state.media is empty              │
 │  · one VLM call → JSON:                                    │
 │      summary, fashion_items, colors,                       │
 │      materials_or_textures, search_queries, uncertainties   │
 │  · emitted to the UI immediately (media_analysis SSE)       │
 │  · video + unavailable analysis + media-dependent query     │
 │      → SHORT CIRCUIT, ask for text, turn ends               │
 └────────────────────────────────────────────────────────────┘
        │
        ▼  wrapped in MEDIA_FENCE — "sight, never a catalog fact"
 ┌─ PROMPT ASSEMBLY ──────────────────────────────────────────┐
 │  base prompt + granted skills + catalog capabilities        │
 │  + MEDIA ANALYSIS (fenced) + CURRENT CART                   │
 │  + HISTORICAL PRODUCT INDEX (most recent first)             │
 │  + RECENT DISCUSSION                                        │
 └────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─ AGENT LOOP (deep agents) ─────────────────────────────────┐
 │  recursion_limit = 24        (config default)              │
 │  execution timeout = 45s     (config default)              │
 │  step 1 is always activate_shopper_skills_tool             │
 │     · max 3 activations per turn (_MAX_TURN_ACTIVATIONS)   │
 │     · skills grant tools via SHOPPING_TOOL_POLICIES        │
 │  then: search / details / availability / promotions /       │
 │        resolve-references / cart / weather                  │
 └────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─ COMPOSITION ──────────────────────────────────────────────┐
 │  draft reply → grounding editor (fail-closed rewrite)      │
 │  products emitted on the `products` SSE event              │
 └────────────────────────────────────────────────────────────┘
```

### 3.1 With and without media

| Input | What differs |
|---|---|
| **Text only** | No VLM call. `media_analysis` is `""`. `_stated_shopper_text` == the typed query. |
| **Image or video + text** | One VLM call. Its JSON is fenced into the prompt. `fashion_items`, `colors`, `materials_or_textures` are treated as **shopper-stated** by `stated_media_terms()`. |
| **Media only, no text** | Query becomes the sentinel `MEDIA_ONLY_QUERY` = *"The user submitted visual media without additional text."* The VLM is told to study the media neutrally. |
| **Video whose analysis failed** | If the query depends on the media, the turn short-circuits with an apology asking for a text description. No search runs. |

### 3.2 The provenance asymmetry (root cause of the video/jeans class)

There is a helper written for exactly this problem:

```
catalog_search.py :: _stated_shopper_text(state)
    """What the shopper said this turn, including what they showed.

    Provenance was computed from the typed query alone, so every attribute a
    shopper conveyed by attaching a photo or video read as model-invented and
    was refused. An image is a statement; this makes the gate able to hear it.
    """
```

**It is applied to attributes and not to product types.** Verified by grep on this
branch:

| Gate | Helper used | Call sites |
|---|---|---|
| *Attribute* provenance (colour, material) | `_stated_shopper_text(ctx.state)` | 2 — lines 577, 1186 |
| *Product type* provenance | `_shopper_stated_product_scope(ctx.state.query, ...)` | **6** — lines 446, 533, 887, 971, 1191, 1289 |

All six product-type call sites pass `ctx.state.query`, the raw typed text. So when
a shopper uploads a video and types *"I want to shop this look"*:

- "cream" (a colour the video showed) **is** heard as shopper-stated.
- "jeans" (a garment the video showed) **is not**.

`composed_role` is then computed as *the model chose this type and the shopper never
said it* — which is **true by construction for every garment in every video** — and
a composed role is the trusted path: it is allowed to span many subcategories,
because that is what "a top" or "shoes" legitimately means.

That is the whole mechanism. The fix is to pass `_stated_shopper_text(ctx.state)`
at those six sites; the helper and its intent already exist.

---

## Part 4 — The search pipeline

`search_catalog(ctx, scopes)` runs each scope through `_PLAN_STEPS` in order. Every
step either returns text (**ending that scope**) or records findings on the attempt
and returns `None`. Nothing touches the network until step 8.

```
 1  _reconciled_with_what_is_advertised   set aside filter values we cannot honour
 2  _admit_search                         may this scope run at all?
 3  _classify_requirements                which requirements did the shopper state?
 4  _validated_request                    validate against live capabilities
 5  _reviewed_provenance                  hold a repair to what it is repairing
 6  _no_direct_match_outcome              report an unadvertised type as a gap
 7  _planned_search                       build the retrieval plan
    ───────────────────────────────────── no I/O above this line
 8  _reserved_search_slot                 claim turn budget under the lock
 9  _executed_search                      the one network call
10  _rendered_evidence                    render what the model may speak from
```

### 4.1 Step 1 — unadvertised filter values (the colour path)

Values outside the advertised set are **set aside, not translated**, and the search
runs anyway. The words stay in `semantic_query`, so the index still ranks on them,
and the disclosure says the value was ranked rather than filtered.

Partial keep is the rule:

| Asked | Advertised? | Kept filter | Effect |
|---|---|---|---|
| `["cream"]` | no | *field deleted* | no colour filter; ranked on "cream" only |
| `["cream","white"]` | white only | `["white"]` | filters to white |
| `["beige","white"]` | both | `["beige","white"]` | filters to both |

Two failure modes are documented in the code and both are real:

- Deleting the **whole field** when one value is bad was tried and is worse — asked
  for a cream sweater as `["cream","beige"]` the search ranked on "cable-knit"
  alone and returned **a red sweater**.
- Keeping half can be **narrower than the model meant** — `["cream","white"]`
  filters to white in a shop whose cream is beige. This is why the scope prompt asks
  the model to name *every* advertised value the shopper's word could be, and why
  it is a prompt instruction rather than a mapping table.

**Product types are deliberately excluded from step 1.** Setting aside a type
leaves the department, and a department is not a family (Part 1.2). A type this shop
does not sell stays a schema error, which the code calls "the earliest and plainest
place to say so."

### 4.2 Step 2 — `_admit_search`, the gate that matters

For a **composed role** (model chose the type, shopper never said it) this asks two
questions:

```
advertised_request  = _advertised_scope_match(requested_product_type, caps)
uncarried_garment   = _garment_with_no_advertised_value(type + query, caps)

if uncarried_garment is not None
   or (len(selected_subcategories) == 1 and advertised_request is None):
       → refuse, name it in not_covered
```

`_garment_with_no_advertised_value` walks a 17-word list
(`_GARMENTS_A_SHOP_MAY_NOT_STOCK`: jeans, pants, trousers, shorts, leggings,
jackets, coats, blazers, hoodies, sweatpants, socks, hats, scarves, gloves,
swimsuits, suits, vests) and returns the first one that **is mentioned and resolves
to nothing advertised**. Each word is re-checked against the live taxonomy, so the
day a jeans product is indexed the entry goes inert on its own.

The list is a denylist and we want it gone. It currently earns its place by being
the only thing that separates two cases the taxonomy cannot distinguish:

| type | resolves? | in denylist? | correct answer |
|---|---|---|---|
| `shoes` → [boots, flats, heels, sandals] | no | no | **search** — umbrella over 4 real things |
| `tops` → [blouses, camisoles, sweaters] | no | no | **search** — umbrella over 3 real things |
| `jeans` → [skirts, dresses, ...] | no | **yes** | **refuse** |

Delete the list without first giving the catalog an umbrella vocabulary and `shoes`
and `tops` start being refused. See [8.6](#86-umbrella-vocabulary-as-data).

Repair locks also live in this step: a role already refused this turn returns
`STOP_TOOL_USE` rather than a second explanation, keyed on the **semantic query**
rather than the declared type (the declared type is the part the model changes when
it tries again).

### 4.3 Step 8 — the three hard stops

`_reserved_search_slot` reserves **before** executing, under `catalog_lock`, so the
cap holds when tool calls overlap. Three ways a scope dies here, all
`STOP_TOOL_USE`:

| Stop | Trigger |
|---|---|
| `DUPLICATE_SHOPPER_SCOPE` | this shopper-requested scope was already searched this turn |
| `DUPLICATE_CATALOG_SCOPE` | this exact taxonomy + constraints was already searched |
| `CATALOG_SEARCH_LIMIT` | `catalog_searches >= max_catalog_searches_per_turn` |

Duplicate detection is judged against **earlier calls**, never against a sibling in
the same call — two roles of one request are not retries of each other. A scope that
returned **zero** has its shopper-scope key withdrawn, so a relaxed retry is still
allowed; refusing it left *"no green dress in a 2"* with nothing to show but a menu.

### 4.4 The limits, in one place

| Limit | Value | Source |
|---|---|---|
| `deepagents_recursion_limit` | **24** | `config.py:100` |
| `max_catalog_searches_per_turn` | **10** | `config.py:120` |
| `search_products_per_call` | **36** | `config.py:128` |
| `deepagents_execution_timeout_seconds` | **45.0** | `config.py:112` |
| `_MAX_TURN_ACTIVATIONS` (skills) | **3** | `skill_activation.py:48` |

So "endless" is bounded by construction: at most 10 searches and 24 agent steps per
turn, whatever the model tries. The observed 19- and 30-call turns were **not**
unbounded search — they were repeated *stub* tool calls (availability, promotions)
which have their own per-turn dedupe, added in `#232`.

---

## Part 5 — Every "not found"

Five genuinely different situations. Conflating them is how we got skirts for jeans
*and* a question instead of tote bags.

### 5.1 A filter value the catalog cannot honour

*"a cream sweater"*, and cream is not an advertised colour.

Set aside, search proceeds, ranked not filtered, disclosed
(`SEARCH_WORDS_RANKED_NOT_FILTERED`). **Products come back.** The shopper may see a
beige one beside an off-white one; that is the accepted cost of ranking over
refusing.

### 5.2 A product type that is not advertised, narrowly filed

*"jeans"*, declared as `subcategory: ["skirts"]`.

Caught by `_admit_search`'s second arm (`len == 1 and advertised_request is None`).
Refused with instructions to search the garment actually asked for, or name it in
`not_covered`. **No products.**

### 5.3 A product type this shop has no word for, at any width

*"jeans"*, declared as all six apparel subcategories.

This is the shape that kept escaping. On `staging` (`51c3a76`) both arms of the
check miss: `uncarried_garment` was computed **only if `advertised_request is not
None`**, and "jeans" resolves to nothing, so the check was skipped; the other arm
wanted one subcategory and got six. The search ran, and a live trace shows what
came back for *"dark wash straight leg jeans"*:

```
RELATION: {"relation":"model_composed_role","requested_product_type":"jeans",
           "role_advertised_types":["blouses","camisoles","dresses",
                                    "jumpsuits","skirts","sweaters"]}
→ navy collared blouse · Avelina Satin Sheath Dress · Trendy Turtleneck Sweater
```

No `NOT_CARRIED`, no `NOT_COVERED`. On this branch, `c7aa66d` removes the gate so
the check runs at any width and reads the declared type as well as the query, and
the enum rejection now says "not carried" instead of offering the list of six.

**How the model learned to widen:** the enum rejection used to reply *"select every
advertised subcategory that role covers. Choose from these: [...]"* — added in
`#200` (Aug 26). The model complied literally. The refusal message was the attack
vector.

### 5.4 A valid search that returns zero rows

*"a green dress in a size 2"*, and there is no green dress in a 2.

**This is where the branch and staging differ, and where a regression lives.**

| | staging `51c3a76` | this branch |
|---|---|---|
| mechanism | `_relaxed_alternatives` runs a looser search server-side | deleted; the model is told to retry itself |
| J01 "green dress in a 2" | 4 dresses + junk of other types | **4 dresses, clean** ✅ |
| J01 "tote bag in a size 8" | 4 tote bags + 4 phantom shoes | **0 products, a question** ❌ |

The branch's instruction says *drop one optional requirement, keep the product type
and keep the size*. For the tote there **was** no optional requirement — the size was
the whole request — so the model correctly concluded nothing was droppable and
replied *"Would you like me to show you the tote bags we do have?"* That is the
menu-instead-of-products failure the deleted retry existed to prevent. `88e1cc2`
adds the missing case (when the size is the only requirement, drop the size and say
what sizes do exist); **unverified against a live run.**

A second lesson: the journey check `products_within: [tote_bags]` **passed** this,
because with zero products there are no offenders. A ceiling with no floor. Both
J01 turns now assert `products_min: 1` *and* `products_within`.

### 5.5 A reference to a product that was never shown

Covered in Part 6 — and it is not really a "not found" at all.

---

## Part 6 — Resolving a reference

*"do you have the first sweater in size 2"*, *"add the black one"*, *"the second
one"*.

### 6.1 Where the answer lives

Not in process memory. `ConversationProductsClient` POSTs to a **durable event
store**:

```
POST {memory_retriever}/conversations/{conversation_id}/products/resolve
```

Products shown in a turn are recorded as events with **presentation coordinates** —
`candidate_set_id`, `turn_sequence`, `position` — and the same data is rendered into
the prompt as the `HISTORICAL PRODUCT INDEX`, **most recently shown first** (a
comment records that opening with turn 1 made the model reach back fourteen turns
for a navy dress when four black dresses had been shown one turn earlier).

### 6.2 The selectors

| Selector | Meaning |
|---|---|
| `product_ref` | exact `PRODUCT_REF` from the index |
| `display_name` | exact product name |
| `turn_sequence` / `candidate_set_id` | which showing |
| `ordinal` | **1-based position within that showing** |
| `category` | the bracketed value, e.g. `dresses` |
| `attributes` | a dict of extra field values |

The schema is emphatic that a positional reference must use `ordinal`, not a guessed
ref: *"asked to add 'the first one' of four sandals, a guessed ref picked the
fourth, the name did not match it, nothing was added, and the shopper was re-shown
the same four in a different order."*

### 6.3 The bug: the size being asked about is used to find the product

Verified live against the real event store, for the real conversation
`conversation-8db4d0d9-...`:

| Request | `status` | `blocking_field` | matches |
|---|---|---|---|
| `product_ref` alone | `resolved` | — | 1 |
| `product_ref` + `attributes {sizes: "2"}` | **`not_found`** | **`None`** | 0 |
| `product_ref` + `attributes {sizes: "6"}` | `resolved` | — | 1 |

The first sweater shown was **Polished Peplum Pullover Sweater, sizes 4 · 6 · 8 ·
10**. The model passed its correct ref *and* `sizes: "2"` — the size it was asking
*about*. Because `attributes` is a **match selector**, the lookup fails exactly when
the honest answer is "no".

> **A correct "no, not in your size" is indistinguishable from "that product was
> never shown."**

And `blocking_field` comes back `None`, so the precise near-miss message —
*"every other field you supplied matched one earlier product; 'sizes' did not"* —
cannot fire. The generic branch fires instead.

### 6.4 What the generic branch then does

```
REFERENCE first_sweater: NOT FOUND. No product shown in this conversation
carries that PRODUCT_REF or that exact name ...
CATALOG NAME LOOKUP "first_sweater": not shown earlier in this conversation.
  1. Jade Serenity Top Blouse Sweater  $179.99
  2. Polished Peplum Pullover Sweater  $89.99
  3. Qute Cashmere Sweater             $139.99
  4. Green Meadow Sweater Top          $...
```

Three separate faults in that block:

1. **It searches the catalog for `"first_sweater"`** — the model's own internal
   label. `deepagents_runtime.py:1826` takes `display_name or reference_id`, so a
   made-up id becomes a semantic query.
2. **The results are registered as shown products** — `scope.product_evidence.add`
   and `_append_product_results` at lines 1861–1862, with the comment "registered
   exactly as a search result is". They reach the UI. **This is the source of the
   brand-new sweaters, including the red one.**
3. **The model then asked availability for a product never shown**, and the
   availability **stub answered "Yes, available in size 2"** for the 4–10 product.

The fallback is not unreasonable in principle — a shopper naming a product that was
never shown *was* making a search request, and the code comment explains that
ending the turn with an unspent search budget was worse. The faults are the
*trigger* (a false not-found) and the *query* (a reference id).

### 6.5 The path when it works

```
"the first sweater"
   │
   ├─ model sends ordinal:1 (+ set or turn)      ← what the schema asks for
   │     └─ event store returns position 1 → resolved, exact product
   │
   └─ model sends a remembered product_ref       ← what it did today
         ├─ ref only            → resolved
         └─ ref + asked-about size → not_found → name-search on reference_id
                                              → new products injected
```

---

## Part 7 — Why the UI, the reply and the transcript disagree

The question "the UI shows N products but the text mentions M" has **four**
independent causes, and they compound. There is no single number called "the
products this turn showed" — there are three different accountings, fed by
different code, and they are all reporting honestly about different things.

### 7.1 The three accountings

| What you read | Fed by | Scope |
|---|---|---|
| **UI "Recent results" panel** | `products` SSE **and** `images` SSE, merged | **cumulative across the whole conversation** |
| **Transcript / telemetry `N products`** | `diagnostics["product_evidence"]` | one turn |
| **The reply prose** | whatever the model chose to write | one turn, a subset |

`products_shown` is literally `len(diagnostics.get("product_evidence"))`
(`deepagents_runtime.py:316`). The UI is fed by `output.product_results`
(`deepagents_runtime.py:1173-1177`). **Two different collections.** Nothing keeps
them in step.

### 7.2 Cause 1 — the UI panel never clears

`ui/src/components/chatbox/chatbox.tsx :: mergeProductResults`:

```js
productOrderRef.current = [
  ...arriving,
  ...productOrderRef.current.filter((key) => !arriving.includes(key)),
];
```

Arriving products are prepended; **everything previously shown is kept**. The
backing `productsByNameRef` Map only accumulates. The panel is cleared only by
switching shopper profile or resetting the session (`setProducts([])` in
`App.tsx`). So by turn 10 the panel holds most of what the conversation ever
returned, while the reply describes only the current turn. The panel is titled
"Recent results", which is doing a lot of work.

### 7.3 Cause 2 — the panel is fed by two events, not one

Both of these call `mergeProductResults`:

- `type === "products"` → the turn's product payload (line 676)
- `type === "images"` → `productsFromImagePayload(...)` (line 732)

So an image row adds entries to the product panel. A turn that emits zero
products can still put products on screen if it emitted images.

### 7.4 Cause 3 — keyed by name, deduped by id

The UI keys by **product name** (`productKey(product.productName)`), while the
server dedupes by **`product_id`** (`_append_product_results`, `turn_support.py:3180-3192`).
Two products sharing a display name collapse into one in the panel but count as two
on the server. The counts can differ without either being wrong.

### 7.5 Cause 4 — selection happens after registration

Within a single turn, a search registers its **entire** result set (up to
`search_products_per_call = 36` per scope) and the model then names a subset in the
prose. Measured across all 53 scenarios of the `2026-09-14__no-stand-in` run:
**240 of 979 emitted products (25%) are never named in the reply.** So even with a
per-turn panel and no image rows, UI ≥ text is the normal case, not an anomaly.

### 7.6 The turn-2 case: "why did it show new random sweaters?"

Traced from the UI session `conversation-8db4d0d9-...`, turn 2, *"do you have the
first sweater in size 2"*. This is the sharpest example of the two accountings
disagreeing, because they disagree **completely**.

What the trace shows, verified:

- `metadata.products_shown` = **0**
- `diagnostics["product_evidence"]` = **empty list**
- the reply prose names exactly one product, the correct one:
  *"Yes — the Polished Peplum Pullover Sweater (the first sweater I showed) is
  available in size 2."*
- and yet the `resolve_conversation_products_tool` result contained **four catalog
  products presented as never shown**:

  ```
  CATALOG NAME LOOKUP "first_sweater": not shown earlier in this conversation.
  The catalog was searched by that name; these are the closest matches in rank
  order, none previously shown.
    1. Jade Serenity Top Blouse Sweater  $179.99  [generated:e329429505c7d5ca]
    2. Polished Peplum Pullover Sweater   $89.99  [generated:8a785791aa953d49]
    3. Qute Cashmere Sweater             $139.99  [generated:c3ec6564bbaf69d6]
    4. Green Meadow Sweater Top           $49.99  [generated:c8c9645ee9be58c0]
  ```

  **"None previously shown" is false.** Entries 2 and 3 were both shown in turn 1 —
  entry 2 *is* the product being asked about. Only Jade Serenity and Green Meadow
  are genuinely new. So the fallback text asserts a fact about the conversation it
  never checked, and hands the model a list mixing already-shown items with new
  ones under a blanket "new" label. A model told this can re-present a product the
  shopper is already looking at as a fresh find, and has been given a reason to
  distrust the index it holds.

The chain, in order:

1. The model sent the **correct** ref plus `attributes {sizes: "2"}` — the size it
   was asking about. Because `attributes` is a match selector, resolution returned
   `not_found` (see [6.3](#63-the-bug-the-size-being-asked-about-is-used-to-find-the-product)).
2. The generic not-found branch ran a **catalog name search on `"first_sweater"`** —
   the model's own internal label (`deepagents_runtime.py:1826` takes
   `display_name or reference_id`).
3. That path calls **both** `scope.product_evidence.add(found.products)` **and**
   `_append_product_results(state, found.products)` (lines 1861-1862), with the
   comment *"registered exactly as a search result is"*. The second of those is
   exactly what feeds the UI's `products` event.
4. The telemetry counter read 0 anyway, so the transcript and my own reports said
   the turn showed nothing.

So: **the injected sweaters went to the screen and not to the counter.** Every
analysis I have done from `products_shown` — including the 25% figure in 8.2 and the
`> shown:` line I added to the transcripts — is blind to products that arrive by
this path. That is why I kept telling you J01 looked clean.

Not fully explained: why `product_evidence` came back empty when line 1861 appears
to add to it. Either the add did not take on that scope object or the diagnostics
snapshot is assembled from a different one. Worth a breakpoint before fixing
anything here, because the answer changes whether 7.6 is one bug or two.

---

## Part 8 — Optimisations noticed, not done

Recorded while reading. **None applied.**

### 8.1 No prefill caching at all
Every LLM span on every traced turn reports
`llm.token_count.prompt_details.cache_read = 0`. With 4–9 calls per turn at
7k–23k prompt each, the same prefix is paid for every time. Largest single ISL
lever available and it is a serving-side change, not an application one.

### 8.2 A quarter of emitted products are never named
Across all 53 scenarios of the `2026-09-14__no-stand-in` run, **240 of 979** emitted
products (25%) never appear in the reply text. Cause is structural: a search
registers its **entire** result set, then the model names a subset. Selection
happens after registration.

### 8.3 Availability and promotions are stubs
`check_product_availability` always returns in-stock; `check_active_promotions`
always returns none. Every call is pure cost, and worse than free — 6.3 shows the
stub actively producing a wrong answer. Sizes are already present in search
evidence and in the event store, so a size question is answerable with **no tool
call**.

### 8.4 The widening round trip
Pre-`c7aa66d`, an uncarried garment cost: enum rejection → model widens → search
runs → wrong products. Answering at the rejection saves a full model call at full
prompt. Already fixed on this branch; noted because the same pattern (a refusal that
invites a retry it could have pre-empted) may exist elsewhere.

### 8.5 `search_products_per_call = 36`
36 products of rendered attributes per scope is a large share of the prompt on a
multi-scope turn. Worth measuring what a smaller number costs in answer quality,
especially given 8.2.

### 8.6 Umbrella vocabulary as data
If the catalog advertised its umbrella words — `footwear ← shoes`,
`{blouses,camisoles,sweaters} ← tops`, `{skirts} ← bottoms` — then
`_advertised_scope_match` would resolve them, "resolves to nothing" would genuinely
mean "not carried", and **both** the 17-word denylist **and** the subcategory-width
rules could be deleted. It also makes the covering the *shop's*, not the model's,
which structurally prevents jeans→dresses. This is the clean endpoint; it is data,
not another condition.

### 8.7 Provenance helper applied at six sites
Passing `_stated_shopper_text(ctx.state)` instead of `ctx.state.query` at the six
product-type gates (3.2) closes the video/jeans class at its source. Small, but it
changes behaviour on the trusted path, so it needs a journey run.

---

## Part 9 — Open bugs, ranked

| # | Bug | Evidence | Status |
|---|---|---|---|
| 1 | Size asked about is used as a match selector, so "no" is unrepresentable; falls back to name-searching the `reference_id` and injects new products | live probes, 6.3–6.4 | **open, root cause known** |
| 2 | Video-named product types classed as model-composed, bypassing the carried check | trace + grep, 3.2 | partially fixed (`c7aa66d`); six call sites remain |
| 3 | Availability stub answers "yes" for sizes a product does not have | trace, 6.4 | **open** |
| 4 | Zero-result turn can show nothing at all on this branch | J01 turn 11, 5.4 | fix written (`88e1cc2`), **unverified live** |
| 5 | `blocking_field` not populated when an attribute blocks a ref match, so the precise message cannot fire | live probe, 6.3 | **open** |
| 6 | 25% of emitted products never named | full-suite scan, 8.2 | **open, structural** |
| 7 | Paraphrase evades the garment check (`"dark blue bottoms"` names no listed garment) | reasoning, not observed | open; 8.6 dissolves it |
| 8 | Products reach the UI without being counted, so telemetry and transcripts under-report what the shopper saw | trace, 7.6 | **open** — invalidates metrics, fix before measuring anything |
| 9 | UI panel is cumulative and also fed by the `images` event, so it cannot be compared to a turn's prose | UI code, 7.2–7.3 | **open**, arguably by design |
| 10 | The name-lookup fallback claims "none previously shown" without checking; 2 of 4 had been shown, including the product being asked about | trace, 7.6 | **open**, one-line check against the index |

### What I got wrong, for the record

- I reported the jeans guard fixed in `#234`. The same commit contained the gate
  that made it inert. It never fired for jeans.
- I deleted the zero-result retry as redundant. It is not: it is the only guarantee
  that an empty search still shows the right kind of product.
- I wrote `products_within` with no floor, so a turn showing nothing scored green,
  and I read that green as J01 being healthy.
- I reported turn counts from `products_shown` without checking that it and the UI
  read different collections. When I said a turn "showed 0 products", the shopper
  may have been looking at four.

---

## Appendix — reproducing the probes

```bash
# advertised taxonomy as the shop actually publishes it
curl -s localhost:8010/capabilities | python -m json.tool | less

# does an umbrella word resolve?
python - <<'PY'
from chain_server.src.turn_support import _advertised_scope_match
# ... load capabilities, then:
#   _advertised_scope_match("shoes", caps) -> None
PY

# the reference-resolution bug, against the live event store
curl -s -X POST \
  "http://127.0.0.1:8011/conversations/<conversation-id>/products/resolve" \
  -H 'Content-Type: application/json' \
  -d '{"references":[{"reference_id":"p","product_ref":"<ref>","attributes":{"sizes":"2"}}]}'
```

Phoenix spans for a session:
`http://localhost:6006/v1/projects/default/spans?limit=500`, paged with
`next_cursor`, filtered on `session.id`.
