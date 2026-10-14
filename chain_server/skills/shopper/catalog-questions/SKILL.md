---
name: catalog-questions
description: Questions about the shop rather than about a product — the most or least expensive thing, whether anything falls in a price range, what departments exist and how much they hold. Use when the shopper is asking what the catalog contains, not asking to see a kind of product.
response_guidance: Facts about the shop come from the published shape. A superlative names the actual item and shows it; a range the shop does not reach is answered by saying where prices start.
role: primary
exclusive_group: product_procedure
tools_granted:
  - describe_catalog_tool
  - search_catalog_tool
  - get_product_details_tool
---

# Questions About The Shop

Some asks are not searches. "What's the most expensive thing you have", "do you
have anything under $10", "what do you sell" are about the catalog itself, and
`describe_catalog_tool` answers them from what the catalog publishes: how many
products, which categories, the price range of each.

## The One Rule

**A fact about the shop comes from the published shape, never from the results
of one search.** The dearest item a search returned is that search's maximum,
not the catalog's. Answering "the most expensive item in the catalog is the
Quintessence Zippered Crossbody Bag at $199.99" in a shop that reaches $269.99
is how this goes wrong, and it went wrong that way three times in five.

## A Superlative Is Two Steps, And Ends With The Product

1. `describe_catalog_tool` says which category reaches the ceiling or the floor.
2. Search that category at that bound to get the item.

Then answer with both: what the extreme is, and which product it is. A number
on its own is not an answer — the shopper asked what the thing is, so show it
to them.

## A Range The Shop Does Not Reach

Read the range and say so. "Nothing here runs below $39.90 — the least
expensive things we have start there" is an answer the shopper can rely on.

Do not search first. A search returning nothing is a fact about that query; the
published floor is a fact about the shop, and only the second supports the
claim. Offer what does exist at the nearest real price.

## A Range The Shop Does Reach

Say which departments hold anything in it, then show some. The counts come from
the shape; the products come from a search.

## Where This Ends

The moment the shopper narrows to a kind of product — "show me the dresses in
that range" — that is product-discovery's work, not this. Hand over rather than
carrying on with a catalog-level answer to a product-level question.
