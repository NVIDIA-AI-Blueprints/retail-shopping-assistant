# val — replaying fixed conversations, checking state

Twenty-five conversations whose words never change, with assertions answered by
the cart the service holds, the products it returned, and the tools it called.

Nothing here reads the assistant's wording. That is the point: a run once
reported "no failed turns" while a size 6 nobody asked for sat in the cart, and
again while a dress had gone missing entirely.

## Running

```bash
# every script, one conversation at a time
python -m tests.evaluation.src.replay --label 2026-08-14

# one script, while fixing it
python -m tests.evaluation.src.replay --only cart_size_change

# a turn that fails one run in three, eight times
python -m tests.evaluation.src.replay --only reference_ambiguous_colour --repeat 8

# when wall clock matters more than comparable timings
python -m tests.evaluation.src.replay --label nightly --concurrency 4
```

Serial by default, so a failure is not competing with five other conversations
for the model and the timings mean something. Roughly 90 minutes for all
twenty-five; under 30 at `--concurrency 4`.

Needs the stack up (chain server on 8009, memory on 8011) and
`EXPOSE_AGENT_DIAGNOSTICS=true` for the `tools_used` assertions.

Results land in `tests/evaluation/results/val/<label>/`: `report.md` to read,
`raw/<scenario>.json` for every turn, reply, product, tool and cart.

## Writing a script

```yaml
id: cart_size_required
covers: [cart-management, sizes]
why: >
  Why this matters, and what went wrong when it did not hold. The next person
  to see this fail deserves to know whether it is a real rule.
turns:
  - say: show me some heels
    expect:
      products_min: 2
  - say: add the Polished Pearl Pumps to my cart
    expect:
      cart_unchanged: true          # must ask the size, not choose one
  - say: size 7
    expect:
      cart:
        - {name: Polished Pearl Pumps, size: "7", qty: 1}
```

| Assertion | Answered by |
|---|---|
| `cart:` | the cart service — name, size, qty, and the line count must match |
| `cart_unchanged: true` | the cart before and after this turn |
| `products_min` / `products_max` | the products event |
| `every_product: {field: value}` | the catalog, joined by display name |
| `no_product_named: [...]` | the products event |
| `tools_used` / `tools_not_used` | turn diagnostics |

Two rules for new scripts:

**Assert state, never wording.** "It should ask rather than act" is
`cart_unchanged: true`, not a search for a question mark.

**Do not assert that a tool went uncalled when a refusal is acceptable.** A
model may call `add_cart_items_tool` and be refused by the size gate; the cart
is what must not change. That mistake was made in the first draft of
`cart_two_sizes_are_two_lines`.

## What is here

| Group | Scripts |
|---|---|
| **Discovery and filters** | `only_means_only`, `not_carried`, `unadvertised_requirement`, `no_results_recovery` |
| **References** | `reference_by_full_name`, `reference_partial_name`, `reference_pronoun`, `reference_ordinal`, `reference_across_turns`, `reference_ambiguous_colour` |
| **Cart** | `cart_size_required`, `cart_size_change`, `cart_two_sizes_are_two_lines`, `cart_onesize`, `cart_quantity`, `cart_readback` |
| **Styling, weather, budget** | `weather_dated_destination`, `outfit_multi_role`, `budget_under_150` |
| **Media** | `video_look_then_reference`, `image_find_similar`, `image_out_of_catalog` |
| **Honesty and memory** | `store_policy`, `assumed_audience_disclosed`, `price_when_shown` |

Assets in `assets/`, committed as the image-shopping dataset's are.

## How this stands beside the challenger

Neither replaces the other, and the loop between them is the point.

| | Challenger | Replay |
|---|---|---|
| The shopper's words | generated fresh each run | frozen |
| Judged by | a model reading the reply | assertions reading the cart |
| Answers | what is broken that nobody considered | did this build keep its promises |
| Run | nightly, and after a large change | before a merge |

The challenger invented aprons on one run and bath towels on the next, which is
what makes it good at finding a bug and useless for confirming one. When it
finds something, freeze those turns here with an assertion and it cannot come
back unnoticed.

## Known failure

`cart_two_sizes_are_two_lines` fails on the size carried out of a search
filter: asked for "black dresses in a size 2" and then "add the Black Satin
Lace-Up Dress", the assistant adds it in a size 2 that was never a purchase
decision. The rule — a size used to narrow a search is not a size the shopper
chose — lives in the cart skill as prose, and prose leaks. The script asserts
the correct behaviour and will pass when that rule is enforced.
