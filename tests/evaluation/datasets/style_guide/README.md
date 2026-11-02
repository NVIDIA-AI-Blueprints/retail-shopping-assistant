# Style Guide Evaluation Dataset

This dataset exercises the customer-facing styling skill behavior with live
Challenger/Judge conversations. It is intentionally less coupled to the seed
catalog than product-search regression tests.

## Catalog Coupling Levels

Every scenario has `catalog_dependency` metadata:

- `behavior_only`: tests conversation flow, clarification, and decision
  boundaries. No named catalog product is required.
- `category_level`: requires broad categories such as dresses, shoes, bags, or
  accessories, but does not require exact product names.
- `seed_anchor`: intentionally uses a current seed-catalog product as an anchor
  product-page style case.
- `cart_state_seed`: intentionally uses current seed-catalog products to create
  cart state before asking for cart styling.
- `visual_seed_asset`: uses committed image-shopping assets and should be
  refreshed if the visual catalog or asset sidecars change.

Prefer `behavior_only` and `category_level` for durable style behavior tests.
Use `seed_anchor`, `cart_state_seed`, and `visual_seed_asset` only when the
entry point cannot be tested well without a concrete product or image.

Cart-state scenarios can include `turn_sequence`. Challenger must follow that
sequence in order so the cart is actually seeded before the shopper asks for a
cart styling assessment. Do not replace this with a single shopper claim such
as "I added these already" unless the scenario is explicitly testing empty-cart
recovery.

## Refreshing For A New Catalog

When a deployment uses a materially different catalog:

1. Run the style guide dry run:
   ```bash
   PYTHONPATH=tests/evaluation python -m src.challenger --dry-run --all-scenarios --dataset style_guide
   ```
2. Review scenarios with `catalog_dependency.level` equal to `seed_anchor` or
   `cart_state_seed` and replace product names with equivalent available items.
3. Keep `behavior_only` scenarios unless the new deployment no longer supports
   apparel, footwear, bags, or accessory styling.
4. Re-map visual styling coverage in `datasets/image_shopping/scenarios.yaml`
   when image assets or sidecar ids change.
5. Update each scenario's `refresh_note` when the replacement anchor or cart
   setup changes.

The Judge should score the assistant on grounded styling behavior, not on
matching a hard-coded outfit from this file. Product names, prices, materials,
availability, and cart claims still must be supported by the live transcript.
