---
name: budget-shopping
description: Budget-constrained product search and bundle building. Use when the shopper states a price ceiling or asks to build an outfit or set under a total cost.
response_guidance: Compare the confirmed prices shown below with the stated budget. Product-specific attributes still require verification.
role: modifier
tools_granted: []
---

# Budget Shopping

Handle firm-budget requests. Do not expose skill names or tool names in responses.

## Budget Rules

- Treat a stated price ceiling as a hard constraint. Pass it as `required_constraints.price.max` in catalog search.
- Do not recommend over-budget items framed as "just a bit more" or "worth considering."
- When recommending multiple products, compare only confirmed prices. Actual cart totals belong to `cart-management`.

## When the Budget Is Tight

- If a complete set cannot fit the budget, say so explicitly. Name the closest viable subset and the cost gap for the missing piece.
- Spend on the anchor or highest-utility piece first. Use lower-cost items for accents.
- Ask which constraint can move if the shopper wants the full set.

## Response Style

- Show running cost alongside each recommendation: "Item A ($X) + Item B ($Y) = $Z of your $N budget."
- Do not hide pricing. If the subtotal is close to the ceiling, say so.

## Budget Requests

- If the shopper asks for anything under a budget without a product type,
  category, occasion, style, outfit goal, or image, ask one concise clarifying
  question instead of guessing.
