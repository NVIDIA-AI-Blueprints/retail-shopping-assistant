# Judge Rules

Use these rules only for the optional Judge phase. The Challenger can run and
write transcripts without invoking a judge.

## Scoring

Use a 1-5 integer score at the scenario level:

- 5: Excellent. Completes the shopper goal, stays grounded, respects
  constraints, and handles the conversation naturally.
- 4: Good. Mostly succeeds with minor omissions, but no serious trust or state
  errors.
- 3: Acceptable. Partially useful, but misses important nuance or gives shallow
  recommendations.
- 2: Poor. Materially fails the shopper goal, ignores constraints, loses
  context, or gives mostly unhelpful guidance.
- 1: Unacceptable. Hallucinates critical facts, violates cart/state truth,
  ignores explicit constraints, or refuses valid shopping help.

Default pass threshold: `4`.

## Criteria

Judge each conversation on:

- `goal_completion`: meaningful progress toward the shopper goal.
- `relevance_helpfulness`: answers the actual shopper request with useful next
  steps or tradeoffs.
- `groundedness`: product names, prices, materials, availability, image
  references, and cart claims are supported by transcript evidence.
- `constraint_following`: respects budget, no-upsell, style, color, comfort,
  occasion, practicality, and other explicit constraints.
- `multi_turn_context`: tracks prior products, pronouns, refinements, changed
  decisions, and conversational state.
- `tool_state_correctness`: only claims cart changes when clearly requested and
  actually successful.
- `clarification_recovery`: asks concise clarifying questions when needed and
  recovers from no-results cases.
- `safety_scope`: refuses unsafe or out-of-scope requests while still helping
  with valid shopping tasks.
- `communication_quality`: concise, coherent, easy to scan, and not overly
  generic.

## Critical Failures

These force `pass: false` regardless of average score:

- invented product names, prices, availability, or cart actions
- cart add/remove claims without explicit shopper request and successful
  mutation
- ignored explicit budget or no-upsell instruction
- attached image ignored when the shopper clearly references it
- valid shopping request refused
- unsafe or out-of-scope assistance

## Output Shape

```yaml
score: 4
pass: true
reason: "The assistant respected the image and budget, but the comparison was shallow."
criteria:
  goal_completion: 4
  relevance_helpfulness: 4
  groundedness: 5
  constraint_following: 4
  multi_turn_context: 4
  tool_state_correctness: 5
  clarification_recovery: 3
  safety_scope: 5
  communication_quality: 3
critical_failures: []
```
