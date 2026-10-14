# val — replaying fixed conversations, checking state

Forty-five conversations whose words never change, with assertions answered by
the cart the service holds, the products it returned, and the tools it called.

## Running

```bash
# every script, one conversation at a time
python -m tests.evaluation.src.replay --label 2026-08-14

# one script, while fixing it
python -m tests.evaluation.src.replay --only J06

# three of them, in one run
python -m tests.evaluation.src.replay --label three --only J01,J02,J13

# a turn that fails one run in three, eight times
python -m tests.evaluation.src.replay --only J04 --repeat 8

# all at once, when wall clock matters more than comparable timings
python -m tests.evaluation.src.replay --label three --only J01,J02,J13 --parallel

# or an exact number
python -m tests.evaluation.src.replay --label nightly --concurrency 4
```

`--sequential` is the default, so a failure is not competing with five other
conversations for the model and the timings mean something. `--parallel` runs
up to six at once; beyond that nothing finishes sooner, because the bottleneck
is the model endpoint.

Forty-five scenarios -- twenty journeys and twenty-five probes, 231 turns --
take roughly two hours one at a time, or about forty minutes with `--parallel`.
A single journey is four to ten minutes.

Needs the stack up (chain server on 8009, memory on 8011) and
`EXPOSE_AGENT_DIAGNOSTICS=true` for the `tools_used` assertions.

Results land in `tests/evaluation/results/val/<label>/`, which is gitignored:

```
report.md                 one row per scenario, failures expanded
transcripts/<id>-<n>.md   the conversation, with the cart after every turn
raw/<id>-<n>.json         everything, for a tool to read
```

**The transcript is the artifact.** Judge from it -- by eye, or by handing it to
a model -- and note that the cart is printed after every turn. A transcript of
replies alone is judged on prose, and prose is what reported "no failed turns"
over a size nobody asked for and again over a dress that had gone missing. Here
the words and the cart sit next to each other:

```markdown
## 2. add the Black Satin Lace-Up Dress

Added it in a size 2.

> **Cart: 1 x Black Satin Lace-Up Dress (size 2)**
> 18.0s · 0 products · tools ['add_cart_items_tool']
> **FAILED** `cart_unchanged` — cart went from [] to [...]
```

Each transcript records the build it was run against, because several of ours
could not say, and were misread for it.

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

## Two kinds of script

**Journeys** (`scripts/journeys/`) are continuous conversations of six to twenty
turns, like the demo scripts they came from. They are where context accumulates,
and accumulated context is what produced most of this week's failures: a navy
dress chosen at turn nineteen, a size carried out of a turn-one search filter
into a turn-five purchase. Nothing shorter would have caught either.

| # | journey | turns | asserted | covers |
|---|---|---|---|---|
| J01 | `wedding_abroad` | 20 | 13 | weather, outfit-styling, references, cart-management, sizes, honesty |
| J02 | `video_look_full` | 13 | 10 | media, references, sizes, cart-management, weather, budget-shopping |
| J03 | `indecisive_shopper` | 11 | 11 | references, cart-management, sizes, memory |
| J04 | `colour_ambiguity` | 7 | 6 | references, cart-management, discovery |
| J05 | `long_memory` | 10 | 9 | memory, references, cart-management, sizes |
| J06 | `size_explorer` | 10 | 7 | cart-management, sizes, references |
| J07 | `cart_manager` | 10 | 5 | cart-management, sizes, references |
| J08 | `abandon_and_restart` | 8 | 4 | cart-management, references, memory |
| J09 | `one_size_conversation` | 9 | 6 | sizes, cart-management, discovery |
| J10 | `budget_journey` | 9 | 6 | budget-shopping, filters, cart-management, honesty |
| J11 | `work_capsule` | 9 | 8 | outfit-styling, budget-shopping, cart-management, sizes |
| J12 | `beach_holiday` | 10 | 8 | weather, outfit-styling, discovery, cart-management, sizes |
| J13 | `comparison_shopper` | 9 | 9 | discovery, memory, honesty, cart-management |
| J14 | `changing_mind` | 8 | 6 | discovery, filters, references, cart-management |
| J15 | `impossible_requirements` | 7 | 2 | discovery, filters, honesty |
| J16 | `not_carried_pivot` | 7 | 4 | discovery, honesty, cart-management |
| J17 | `gift_shopping` | 7 | 7 | cart-management, sizes, discovery, references |
| J18 | `policy_and_purchase` | 8 | 5 | store-policy-answers, honesty, cart-management |
| J19 | `photo_then_build` | 7 | 6 | media, discovery, outfit-styling, cart-management |
| J20 | `menswear_pivot` | 6 | 5 | media, honesty, discovery, cart-management |

Run one by its number: `--only J13`. A prefix runs a group: `--only J1`
runs J10 through J19.

**Probes** (`scripts/probes/`) are one to four turns and check a single
behaviour. They are for narrowing down what a journey found, and for keeping a
fixed bug fixed.

## What the probes cover


| script | covers | what it proves | turns | assertions | media |
|---|---|---|---|---|---|
| `assumed_audience_disclosed` | discovery, honesty | An unscoped ask returns womenswear; the shopper is told nobody chose that | 1 | `products_min` | — |
| `budget_under_150` | budget-shopping, filters | A ceiling that genuinely excludes, in a $39.90-$269.99 catalog | 1 | `products_min` | — |
| `cart_onesize` | cart-management, sizes | A one-size bag is added without ever being asked its size | 2 | `cart, products_min` | — |
| `cart_quantity` | cart-management | Two means two; a number the shopper did not choose is not theirs | 2 | `cart, products_min` | — |
| `cart_readback` | cart-management | What it says is in the cart is what the cart holds | 3 | `cart, products_min` | — |
| `cart_size_change` | cart-management, sizes | A size change adds the new line before removing the old | 3 | `cart, products_min` | — |
| `cart_size_required` | cart-management, sizes | No size, no add -- and the size comes from the shopper | 3 | `cart, cart_unchanged, products_min` | — |
| `cart_two_sizes_are_two_lines` | cart-management, sizes | Two sizes are two lines; a search filter is not a purchase decision | 4 | `cart, cart_unchanged, every_product, products_min` | — |
| `image_find_similar` | media, discovery | A photo becomes search terms, not catalog facts | 1 | `products_min` | Black_dress.jpeg |
| `image_out_of_catalog` | media, honesty | No menswear exists; it says so rather than offering womenswear | 1 | `cart_unchanged` | male_look.jpeg |
| `no_results_recovery` | discovery, filters | A combination with nothing behind it relaxes and says what it relaxed | 1 | `cart_unchanged` | — |
| `not_carried` | discovery, honesty | Aprons are not carried; that is an answer, not a failure | 1 | `cart_unchanged, products_max` | — |
| `only_means_only` | discovery, filters | Only is a constraint, not a preference | 1 | `every_product, products_min` | — |
| `outfit_multi_role` | outfit-styling, discovery | Three roles in one sentence, each with its own filters | 1 | `products_min` | — |
| `price_when_shown` | memory, honesty | A price shown earlier is history, not a current claim | 2 | `cart_unchanged, products_min` | — |
| `reference_across_turns` | references, memory | A product named ten turns ago is still resolvable | 4 | `cart, products_min` | — |
| `reference_ambiguous_colour` | references, cart-management | Several black dresses: ask which, never pick one | 3 | `cart_unchanged, every_product, products_min, tools_not_used` | — |
| `reference_by_full_name` | references, cart-management | The article in 'the Southwest Bracelet' must not defeat naming | 2 | `cart, every_product, products_min` | — |
| `reference_ordinal` | references | 'That first one' is answerable only from stored position | 2 | `cart_unchanged, every_product, products_min` | — |
| `reference_partial_name` | references, cart-management | Shoppers shorten names and still mean one product | 2 | `cart, every_product, products_min` | — |
| `reference_pronoun` | references, cart-management | 'Add it' with one product in play is unambiguous | 2 | `cart, products_min` | — |
| `store_policy` | store-policy-answers, honesty | Returns and shipping answered, never invented | 1 | `cart_unchanged, products_max` | — |
| `unadvertised_requirement` | discovery, filters, honesty | Waterproof is not an attribute here: disclose, do not veto | 1 | `every_product, products_min` | — |
| `video_look_then_reference` | media, references, sizes | The camera path, then an ordinal reference across it | 2 | `cart_unchanged, products_min` | casual_lady_fall.mp4 |
| `weather_dated_destination` | weather, outfit-styling | A forecast is not an answer on its own -- show clothes too | 1 | `products_min` | — |

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
