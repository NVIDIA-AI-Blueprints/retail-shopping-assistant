# Regression baseline — frozen shopper turns

Three conversations, 24 shopper turns, **frozen verbatim** from the eval run of
2026-08-11 (`tests/evaluation/results/runs/20260811T180255Z`). The words never
change, so any two runs are directly comparable.

## Why this exists

The eval challenger regenerates the shopper's words every run. It invented
aprons on one run, bath towels and headphones on the next. That makes it good at
**finding** bugs and useless for **confirming** them — and three separate fixes
were reported as done after being validated against the single example each fix
came from. The apron fix scored 4/4 on apron wording and failed on towels an
hour later.

**A change is not fixed until a number moves in this table and nothing else
regresses.**

## How to run

```bash
# chain server must be up on :8009
python3 ~/exec-briefs/retail-shopping-assistant/regression/run.py <label>
```

~15 minutes. Writes `result-<label>.json` with the full text of every turn and
reply, and prints a per-turn pass/fail table.

A turn counts as failed when the reply is empty or contains: "could not
complete", "please try again", "couldn't complete a valid catalog search",
"encountered an error".

## Results

| case | 2026-08-11 baseline | `2026-08-12-pr171` | `2026-08-12-not-carried` |
|---|---|---|---|
| `text_compare_value_between_similar_items` | 2, 3, 4, 5, 6 | 2, 3 | **none** |
| `text_cart_pronoun_resolution` | 2, 7, 8 | 2 | **none** |
| `text_accessory_for_existing_outfit` | 8 | none | **none** |
| **total failing turns** | **9** | **3** | **0** |

No turn that passed at baseline fails now.

### What moved, and what fixed it

- **accessory turn 8** — the cart add that ended a seven-turn conversation with
  "I could not complete that shopping request". Fixed by the cart validating
  against the conversation product index instead of a per-turn cache (#171).
- **cart_pronoun turns 7, 8** — `remove_cart_item` `agent_error` pair. Fixed by
  the same change plus refusing a destructive `quantity: 0` (#171).
- **compare turns 4, 5, 6** — the tail of the "not carried" loop. Shortened by
  the `not_covered` work (#170).

### What is still failing

**Turn 2 of both remaining cases.** The shopper names a product this catalog
does not carry — bath towels, headphones — and the search is refused *before*
`not_covered` is ever considered. #170 fixed the half where the model sends
`not_covered` and the runtime discards it; these turns never reach that half.

One target, precisely located. Not a class, not a guess.

## Tool for tool

| tool | use for | do not use for |
|---|---|---|
| this suite | did my fix work | finding new bugs — the turns never change |
| eval challenger (`tests/evaluation`) | finding bugs, adversarial pressure | confirming fixes |
| 13-turn demo script (`DEMO_SCRIPT_*.txt`) | is the demo recordable | anything else |
| unit tests | did I break a code path | what the model does |

## Adding a case

Take the shopper turns from a real failure, paste them verbatim into
`cases.json`, record the failing turns here as a new baseline column. Never
paraphrase a turn — the point is that it does not move.
