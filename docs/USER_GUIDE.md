# 👤 User Guide

Using the assistant once it is deployed. For deploying it, see
[Deployment](DEPLOYMENT.md).

Open `http://localhost:3000`. There is no account and no login.

## What it can do

Ask for products in your own words, refine across turns, and manage a cart by
description rather than by clicking. Upload an image to find similar products.
Optional safety checks can be switched on per session — those have their own
page, [Guardrails](GUARDRAILS.md).

Available filters are not fixed in the UI or the application code; they come
from the loaded catalog and its schema sidecar. To see what the current
catalog advertises:

```bash
curl -s http://localhost:8010/capabilities   # the catalog's live snapshot
curl -s http://localhost:8009/capabilities   # the contract the assistant is using
```

The bundled catalog advertises taxonomy, price, colour, pattern, and
category-specific facets. A different catalog exposes different fields — see
[Catalog Architecture](CATALOG_ARCHITECTURE.md).

## Talking to it

Type, press Enter, wait. Specific requests work better than broad ones, and
context carries across turns:

```text
You: Show me red summer dresses under $80
You: What accessories would go with the first one?
You: Add the second dress to my cart
You: What's my total?
```

Cart actions need to be explicit. "I like the black one" selects nothing —
"add the black one to my cart" does. You can remove items, change quantities,
and change a size the same way.

### What it remembers

Within one conversation the assistant gets the recent turns plus an index of
the products it actually showed you as cards. That is how "the second dress"
or "the bag you showed me earlier" resolves, and it survives a server restart
because it is stored by the memory service.

It works only inside the same conversation, and only when the reference is
specific enough to identify one product. If it matches nothing, or matches
several, you get a clarifying question rather than a guess — the assistant
does not pick one and hope. It does not search across conversations, and it
does not infer preferences or sentiment from what you said earlier.

## Image search

Click the camera icon and choose a file. JPEG and PNG, up to 10MB.

Clear, well-lit shots of a single product on an uncluttered background work
best. Group shots, heavy filtering, and screenshots work poorly, because the
search matches the whole image rather than isolating a product within it.

Image and hybrid search appear only when the deployment has image embeddings
built; otherwise the modes are absent from `/capabilities` and the assistant
will say so.

## 🛠️ Troubleshooting

**The assistant says it specializes in apparel instead of answering.** A safety
check found the request off-topic or unsafe. Open the **Safety** row under the
reply to see which check refused it. See [Guardrails](GUARDRAILS.md).

**The Safety row says a check is Unavailable.** The safety model could not be
reached. That is not a judgement about what you sent. Check your cart before
retrying, because a cart change may already have gone through.

**No products match.** Try broader wording or a different category. The
assistant will not invent a product that the catalog does not hold, and a
no-match answer applies only to what it actually searched for.

**An image will not upload.** Check it is JPEG or PNG and under 10MB.

**Replies are slow.** A turn makes several model calls, and complex requests
make more. If it is consistently slow, that is a deployment concern —
see [Performance](PERFORMANCE.md).

## ❓ FAQ

**What can I search for?**
Whatever the loaded catalog holds. `http://localhost:8010/capabilities` lists
the active fields and their allowed values.

**How accurate are the results?**
The assistant interprets your request against the catalog's advertised
capabilities. The catalog itself runs no language model — it embeds text and
images, applies exact filters, and fuses results deterministically. More
specific descriptions usually match better.

**Does it remember me between sessions?**
No. Preferences are not extracted or saved. Conversation text is retained as
part of the durable transcript so a conversation can be replayed, but it is
not turned into a reusable profile.

**Is my data private?**
Shopper and assistant text is stored in the operator-controlled memory-service
database so turns can be replayed and recent context loaded. Raw uploaded
media is not stored in that transcript. Access, backup, retention, and
deletion are the operator's responsibility.

**Does my cart persist?**
Refreshing the same browser tab keeps its cart. Closing the tab or starting a
new browser session creates a new identity, so the old cart is not reopened
even though its row remains until an operator clears it.

**Can I actually buy anything?**
No. This is a demonstration application — there is no checkout, no payment,
and no order. Prices come from the demonstration dataset.

**Can I see more product detail?**
Yes. Ask about an item after a search and the assistant reads the catalog's
structured details, such as material or care. It states when the catalog does
not carry a fact rather than filling the gap.

---

Report issues through the [main README](../README.md).
