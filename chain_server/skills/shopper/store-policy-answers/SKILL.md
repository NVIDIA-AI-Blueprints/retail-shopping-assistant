---
name: store-policy-answers
description: Store policy questions about returns, shipping, sizing, payment, price matching, and gift cards. Use when the shopper asks about policies, not products.
---

# Store Policy Answers

Answer policy questions from `get_store_policy_tool` only. Do not expose tool names in responses.

## Rules

- Always call `get_store_policy_tool` for policy questions. Never answer from model knowledge.
- If the tool returns a not-found error, relay the message directly: the policy is not available through the assistant; direct the shopper to the retailer's help center.
- Do not blend a policy answer with a product recommendation in the same sentence.
- Do not speculate on policies not covered by the tool (e.g., international shipping, loyalty programs, promotional pricing).

## Supported Topics

`returns` · `shipping` · `sizing` · `payment` · `price_match` · `gift_cards`

Any other policy topic should be acknowledged honestly as not available through the assistant.
