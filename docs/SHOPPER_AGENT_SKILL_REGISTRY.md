# Shopper Agent Skill Registry

This registry documents the Deep Agents skills loaded by the shopper-serving
assistant. Skill names and paths are internal implementation details. They are
for engineers, evaluators, and agent instructions, not shopper-facing UI copy.

## Current Runtime Boundary

The runtime source of truth is
`chain_server/src/deepagents_runtime.py::DeepAgentsRuntime._create_agent`.
The assistant passes `skills=["/shopper"]` to `create_deep_agent` and uses a
`FilesystemBackend` rooted at `chain_server/skills` in virtual mode. In the
container image, `chain_server/Dockerfile` copies that directory to
`/app/skills`.

Deep Agents `read_file` is available so the model can read skill instructions
from the skill backend. Write, edit, list, grep, glob, shell, todo, and
general-purpose subagent tools remain disabled for the shopper harness.
Customer profile, cart, catalog, price, inventory, order, and payment truth
must stay in application services, not skill files.

After the Deep Agent drafts a response from tool calls, the runtime can run a
configurable grounding editor over the final shopper-facing text. This editor is
not a planner and does not call shopping tools; it rewrites only against the
current turn's tool evidence and cart snapshot to remove unsupported product
claims, surface guarantees, and internal refs. It is enabled by default and can
be disabled with `GROUNDING_REWRITE_ENABLED=false`; the evidence window is
controlled by `GROUNDING_REWRITE_MAX_EVIDENCE_CHARS`.

## Registered Skills

| Skill | Source | Status | Primary entry modes |
| --- | --- | --- | --- |
| `outfit-styling` | `chain_server/skills/shopper/outfit-styling/SKILL.md` | Registered | Anchor product, no-anchor discovery, cart styling, conversational mid-browse |

## `outfit-styling`

Purpose: customer-facing fashion styling that can build, complete, validate,
compare, refine, or budget outfits from product context, cart context, uploaded
images, or mid-browse questions.

Tool boundary:

- Uses catalog search for grounded product recommendations and substitutions.
- Uses product details only for known `PRODUCT_REF` values.
- Product-detail reads are capped per turn; once the cap is reached the agent
  should stop tool calling and answer from evidence already collected.
- Uses cart read and cart total tools for cart styling and budget checks.
- Cart outfit checks should read cart state and avoid unnecessary catalog search
  for items already in the cart.
- Uses cart mutation tools only when the shopper explicitly asks to add or
  remove items and the normal cart tool preconditions are satisfied.
- Multi-item outfit adds use the batched add tool with selected `PRODUCT_REF`
  values, then report exact added and failed items.
- Styling approval does not imply cart consent; the add scope must match the
  items explicitly included in the shopper's add request.
- Ambiguous add scope should produce one concise clarification, not a guessed
  cart mutation.
- Does not own catalog facts, pricing, inventory, cart state, checkout, or
  profile persistence.

Behavior boundary:

- Supports multi-intent requests such as styling plus budget, styling plus
  cart review, image styling plus search, or comparison plus value judgment.
- Separates catalog facts from styling inference. Shopper wording is context,
  not product evidence, and material or comfort claims should be attributed
  item by item unless the whole outfit is supported by catalog evidence.
- Treats product names as display names, not attribute evidence. Length, color,
  print, material, fit, care, construction, and group-level claims require
  product-detail evidence.
- Keeps internal `PRODUCT_REF` and `CART_LINE_ID` values out of shopper-facing
  responses.
- Avoids grouping leather, rubber, metal, or generic canvas under natural-fiber
  claims; material summaries should stay item-specific.
- Uses product-detail reads before detailed comparison tables or claims about
  material, dimensions, pockets, closures, care, comfort, or outdoor
  practicality.
- Does not use product-detail reads just to enrich the first no-anchor outfit
  recommendation. Initial outfit building should search by item role and
  synthesize from names, prices, categories, images, and styling fundamentals.
- Keeps initial recommendations lightweight: product name, price, role, and one
  styling reason. Detailed specs require product-detail reads and a shopper need
  for that detail.
- Does not claim tax, shipping, delivery, or real-time stock status.
- Keeps outdoor-practicality language modest unless the catalog explicitly
  supports stronger claims such as grass/gravel stability, water resistance,
  all-day comfort, or weather safety.
- Avoids category-wide superlatives such as maximum breathability or best grip
  unless all relevant catalog options have been checked with product details.
- Asks at most one concise clarifying question when no anchor, occasion, vibe,
  category, budget, image, or cart context exists.
- Keeps shopper-owned wardrobe items separate from catalog products.
- Does not expose skill names, tool names, entry-mode names, or internal
  routing language in shopper responses.
- Final responses pass through the grounding editor when tool evidence exists;
  the editor sees a customer-safe evidence summary rather than raw catalog
  marketing copy. It may soften or remove unsupported material, comfort,
  outdoor, or internal-reference language but should preserve tool-confirmed
  cart actions.

## Tuning Loop

The intended tuning surface is the markdown file:
`chain_server/skills/shopper/outfit-styling/SKILL.md`.

When changing the skill:

1. Keep the frontmatter `name` stable unless changing runtime behavior on
   purpose.
2. Prefer catalog-agnostic behavior rules over hard-coded product names.
3. Validate the skill file with the skill validator.
4. Run unit tests that assert the skill is present and loaded by the runtime.
5. Restart or redeploy the chain server so the container/process sees the
   current file.
6. Start a fresh chat thread when testing frontmatter or description changes;
   Deep Agents caches skill metadata per thread.
7. Run the `style_guide` Challenger/Judge evaluation before treating behavior
   changes as ready.

If a new deployment uses a materially different catalog, regenerate or adjust
catalog-dependent style evaluation fixtures before judging. The skill should
usually remain stable; the eval scenarios and catalog expectations are the
pieces that may need refresh.

## Minimal First, Subagent Later

`outfit-styling` is currently a file-backed Deep Agents skill, not a separate
subagent. That is intentional for the first production slice: the main Deep
Agent already has the shopping tools and can perform multi-step tool use while
the skill guides decision boundaries and response style.

Promote styling to a dedicated subagent only if evaluation shows repeated
failures that require private multi-step planning beyond the main agent loop,
or if styling needs its own tool budget, memory policy, or response schema.
