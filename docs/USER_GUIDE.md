# 👤 User Guide

## 📋 Table of Contents

- [Getting Started](#-getting-started)
- [Features Overview](#-features-overview)
- [Using the Chat Interface](#-using-the-chat-interface)
- [Product Search](#-product-search)
- [Shopping Cart Management](#-shopping-cart-management)
- [Image Upload Feature](#%EF%B8%8F-image-upload-feature)
- [Best Practices](#-best-practices)
- [Troubleshooting](#%EF%B8%8F-troubleshooting)
- [FAQ](#-faq)

## 🚀 Getting Started

### Accessing the Application

1. **Open your web browser**
2. **Navigate to**: `http://localhost:3000`
3. **Wait for the page to load** (may take a few seconds on first visit)

### First Time Setup

The application should be ready to use immediately. If you encounter any issues:

1. **Check the status**: Look for any error messages on the page
2. **Refresh the page**: Press `F5` or `Ctrl+R`
3. **Contact support**: If problems persist, check the troubleshooting section

## 🎯 Features Overview

### Core Features

- **🤖 Intelligent Product Search**: Find products using natural language
- **🛒 Shopping Cart Management**: Add, remove, and manage items
- **🖼️ Visual Search**: Upload images to find similar products
- **💬 Conversational AI**: Natural language interactions
- **📱 Responsive Design**: Works on desktop and mobile devices

### Available Product Filters

Catalog filters and allowed values are determined by the loaded catalog and
its schema sidecar. They are not fixed in the UI, chain server, or application
code.

To see the current filter fields and values:

```bash
curl -s http://localhost:8010/capabilities
curl -s http://localhost:8009/capabilities
```

Port `8010` shows the catalog service's live snapshot. Port `8009` shows the
contract cached for the chain-server process lifetime. After a catalog change,
operators restart the catalog, wait for it to become healthy, and then restart
the chain server so these responses match.

The bundled catalog advertises taxonomy, price, color, pattern, and observed
category-specific facets. A different catalog can expose different fields by
declaring their roles in its adjacent schema sidecar. Actual values and
category scopes always come from the ingested rows.

## 💬 Using the Chat Interface

### Representative Shoppers

The navigation bar includes a required **Shop as** dropdown with Guest mode plus
five fictional representative shoppers derived from the live-evaluation
behavior set: Alex, Casey, Jordan, Morgan, and Riley. A new browser-tab session
shows a short selection screen and does not start chat until you explicitly
choose one of those six modes. Only a named profile's ID is retained in the
current browser tab and sent with a chat request; profile contents are resolved
by the server. An explicit Guest choice is retained as a mode but sends no
profile ID. After a named profile is selected, a compact strip beneath the
navigation shows its display name, shopper type, saved ZIP, and behavior. Guest
mode does not show a profile strip.

The selected behavior may guide interaction and styling, while your explicit
request always wins. A profile never supplies an unstated budget, size, color,
material requirement, cart action, product fact, or weather fact. Saved ZIP is
not proof of where an event will occur. For occasion-led styling, the assistant
may treat it as a tentative local-event candidate and naturally ask whether the
event uses that saved area or is elsewhere when the answer would materially
change the guidance. It does not echo the ZIP digits. For weather, the server
accepts that saved area only when you explicitly say the event is in `my` or
`the` usual/home area, or when you answer affirmatively immediately after the
assistant asks about that area. If it then asks for the date, an immediate
date-only reply preserves that confirmation. A new explicit place, address,
postal code, question, negation, uncertainty, or location override rejects
saved-area mode. A modal lowercase `may be` is treated as uncertainty, while a
calendar date such as `May 5` remains valid. A shopper-stated destination or
venue always wins and prevents fallback to the saved ZIP. Guest sends no
profile ID and has no saved-location fallback.

Location and venue are styling context, not weather or product facts. For
example, “wedding in Cancun” does not imply a beach; the assistant may ask one
short destination-or-venue question if it matters. If you explicitly ask to
plan before seeing products and context is missing, the assistant uses exactly
two short sentences: one conditional direction, then that question. With
context complete, it uses one short paragraph and asks no further event-context
question. Ceremony and reception both stated as on the sand in Cancun are
complete for this helper. If you ask to see products now, the assistant starts
with one grounded requested or core product role and, if location is still
missing and materially changes the next recommendation, asks only event
location beside the results; venue is deferred.

Weather provider calls are disabled by default. When an operator has enabled
them, the assistant gets one forecast attempt in a turn only after it has an
accepted saved-area confirmation or one exact place, address, or postal-code
phrase you supplied, plus an exact event date, complete date range, or the
exact phrase `next week`. It keeps your place phrase as the authority. If that
phrase is an abbreviation such as `NYC` or an ambiguous name such as
`Springfield`, the provider query must keep that exact phrase first and may add
only one or two region/country qualifiers. It sends `NYC` directly or uses
`NYC, NY`; it does not rewrite the abbreviation. `Springfield, TX` is one
possible explicit regional assumption, but it never invents a ZIP or numeric
component you did not state. An invalidly formed call consumes that attempt
rather than being retried. The provider resolution is shown so its model-owned
place assumption is correctable.
`Next week` is
resolved server-side from one captured UTC date to the next
Monday-through-Sunday range; a current negation or different date overrides an
earlier `next week`. An unambiguous single-day phrase such as
`tomorrow` is resolved against that same date into an exact day; other
ambiguous or unresolved relative dates prompt one exact-date question. There
is no separate weather screen or API response type.

Successful weather guidance contains one server-authored canonical forecast
block. When you said `next week`, it first shows the
Monday-through-Sunday dates used. It then includes every validated day's date,
condition, available low/high temperature, precipitation probability/types,
[Weather Data Provided by Visual Crossing](https://www.visualcrossing.com/),
and the warning that forecasts can change. The block appears exactly once and
cannot be shortened by model-written prose. Only evidence fetched in the
current turn supports those weather facts. When you give a place, the response
also states the place the provider resolved so the geographic assumption is
visible and reversible; that resolved place is omitted when the confirmed saved
ZIP is used.
Weather tool inputs/output are redacted from diagnostics and failed-turn
partial output; saved profile ZIP is also scrubbed from diagnostic string keys
and values. The final assistant summary remains part of the durable
conversation and may be exactly replayed, but prior forecast summaries are
redacted from later graph and grounding-editor discussion, and prior weather
tool output is not reused as evidence.
Grounding-editor sentences containing weather-domain fact language or
fact-shaped dates/values are removed while ordinary grounded styling language
remains. If none remains, deterministic catalog rendering plus the canonical
weather block is used.
When a context-only reply applies weather to options already shown, that
accepted reuse path bypasses the grounding editor. On success, the server
renders the exact names from the newest candidate set, one bounded styling
direction derived from the structured forecast, and the canonical forecast
block. If the provider fails, it keeps those prior names and shows a typed safe
weather-failure message.

A forecast may guide general styling, but it does not prove that a product is
warm, waterproof, breathable, comfortable, safe, surface-suitable, or otherwise
performance-ready, and it cannot silently create a catalog must-have. Before
enabling shopper traffic, the operator must confirm that the selected Visual
Crossing plan permits the intended attribution, display, storage, and sharing,
including durable assistant summaries and downstream app-model/output-guardrail
processing.

Changing the dropdown starts a clean visible session: chat, product cards,
selected product, attachments, inference activity, and metrics are cleared, and
fresh session, conversation, and cart identities are created. A conversation
cannot switch between Guest and a profile or between two profiles. The Reset
button keeps the explicitly selected shopper mode while starting another clean
conversation. If the profile service cannot load, Guest mode remains available.
The UI makes one delayed automatic retry, then retries when the browser returns
online or the tab regains focus. It does not poll for the lifetime of the app,
and a recovered picker does not interrupt an active Guest conversation.

### Basic Interaction

1. **Type your question** in the chat input box at the bottom
2. **Press Enter** or click the send button
3. **Wait for the response** - the AI will process your request
4. **View results** - products and information will appear

### Chat Examples

#### Product Discovery
```
You: "Show me summer dresses"
AI: "I found several summer dresses that might interest you..."

You: "Find black shoes under $50"
AI: "Here are some black shoes within your budget..."
```

#### Shopping Cart
```
You: "Add the first dress to my cart"
AI: "I've added the Black Polka-Dotted Slip Dress to your cart"

You: "What's in my cart?"
AI: "Your cart contains: 1x Black Polka-Dotted Slip Dress ($59.90)"
```

#### General Questions
```
You: "What accessories go with a red dress?"
AI: "For a red dress, I'd recommend..."

You: "Help me build an outfit for a wedding"
AI: "For a wedding, I suggest starting with..."
```

### Conversation Continuity

Within the active conversation, the assistant receives a bounded set of
finalized prior shopper/assistant turns plus a compact index of products that
were actually shown as cards. If an earlier product is needed, the applicable
discovery, styling, or cart skill can resolve exact product, turn, candidate-set,
or position details against that same conversation. One match can support a
detail, availability, or cart action; no match or multiple matches causes a
clarifying question instead of a guess. This continuity survives a chain-server
restart because it is stored by the memory service.

The resolver does not search across conversations or infer preferences,
sentiment, or fuzzy descriptions. A request such as "show me the bag from last
week" works only if it refers to a product presented in this same conversation
and is specific enough to resolve uniquely. Catalog replacements can still
require a fresh product search.

## 🔍 Product Search

### Text-Based Search

#### Search by Category
- "Show me dresses"
- "Find bags"
- "I need shoes"

#### Search by Style
- "Summer dresses"
- "Formal wear"
- "Casual outfits"

#### Search by Price
- "Dresses under $100"
- "Shoes under $50"
- "Affordable bags"

#### Search by Color
- "Red dresses"
- "Black shoes"
- "Blue accessories"

#### Search by Occasion
- "Wedding dress"
- "Work outfit"
- "Beach wear"

### Advanced Search Tips

#### Combine Multiple Criteria
```
"Show me red summer dresses under $80"
"Find black formal shoes for work"
"Casual bags under $40"
```

#### Use Descriptive Language
```
"Elegant evening dress"
"Comfortable walking shoes"
"Stylish handbag for work"
```

#### Ask for Recommendations
```
"What would go well with a black dress?"
"Help me choose shoes for this outfit"
"Suggest accessories for a wedding"
```

## 🛒 Shopping Cart Management

### Adding Items

#### By Description
```
"Add the black polka dot dress to my cart"
"Put the red shoes in my cart"
"Add the first dress to my cart"
```

#### By Position
```
"Add the second item to my cart"
"Add item number 3 to my cart"
```

### Viewing Cart

#### Check Contents
```
"What's in my cart?"
"Show me my cart"
"Cart contents"
```

#### Get Total
```
"How much is in my cart?"
"What's my total?"
"Cart total"
```

### Managing Items

#### Remove Items
```
"Remove the red shoes from my cart"
"Take out the first dress"
"Remove item number 2"
```

#### Update Quantities
```
"Change the dress quantity to 2"
"Update the shoes to 3"
```

#### Clear Cart
```
"Clear my cart"
"Empty my cart"
"Remove everything from my cart"
```

## 🖼️ Image Upload Feature

### How to Use Image Search

1. **Click the camera icon** in the chat interface
2. **Select an image** from your device
3. **Wait for upload** (progress bar will show)
4. **The AI will analyze** the image and find similar products
5. **View results** - similar products will be displayed

### Supported Image Types

- **Formats**: JPEG, PNG
- **Size**: Up to 10MB
- **Resolution**: Any resolution (higher quality = better results)

### Best Practices for Image Search

#### Product Images
- **Clear, well-lit photos** work best
- **Front-facing product shots** are ideal
- **Avoid cluttered backgrounds**
- **Ensure good contrast**

#### What to Upload
- **Individual products** (not group shots)
- **Clear product details** visible
- **Similar style to what you want**

#### What to Avoid
- **Blurry or low-quality images**
- **Multiple products in one image**
- **Heavily edited or filtered photos**
- **Screenshots or poor lighting**

### Example Image Searches

```
Upload: A red dress photo
AI: "I found several similar red dresses..."

Upload: A black shoe image
AI: "Here are some black shoes that match your image..."
```

## 📝 Best Practices

### Writing Effective Queries

#### Be Specific
```
❌ "Show me clothes"
✅ "Show me summer dresses under $100"

❌ "I need shoes"
✅ "I need black formal shoes for work"
```

#### Use Natural Language
```
❌ "dress red cheap"
✅ "Show me affordable red dresses"

❌ "bag work"
✅ "I need a professional handbag for work"
```

#### Ask Follow-up Questions
```
"Show me summer dresses"
"What accessories would go with the first dress?"
"Add the second dress to my cart"
```

### Getting Better Results

#### Provide Context
```
"I'm going to a summer wedding, show me appropriate dresses"
"I need comfortable shoes for walking around the city"
"I'm looking for a professional outfit for job interviews"
```

#### Use Descriptive Terms
```
"Elegant evening dress"
"Comfortable casual shoes"
"Stylish work bag"
"Trendy summer outfit"
```

#### Ask for Recommendations
```
"What would you recommend for a beach vacation?"
"Help me build an outfit for a first date"
"What's trending this season?"
```

### Shopping Cart Tips

#### Check Before Adding
```
"What's the price of the first dress?"
"Show me more details about the red shoes"
"Tell me about the material of this bag"
```

#### Manage Quantities
```
"Add 2 of the black dresses to my cart"
"Change the shoe quantity to 3"
"Remove one of the bags from my cart"
```

#### Review Regularly
```
"What's in my cart right now?"
"How much is my total?"
"Show me a summary of my cart"
```

## 🛠️ Troubleshooting

### Common Issues

#### Page Won't Load

**Problem**: The application doesn't load or shows an error

**Solutions**:
1. **Refresh the page** (F5 or Ctrl+R)
2. **Check your internet connection**
3. **Try a different browser**
4. **Clear browser cache and cookies**
5. **Contact support if the issue persists**

#### Chat Not Responding

**Problem**: The AI doesn't respond to your messages

**Solutions**:
1. **Wait a few seconds** - responses can take time
2. **Check if the page is still loading**
3. **Refresh the page and try again**
4. **Try a simpler query**
5. **Check your internet connection**

#### Image Upload Fails

**Problem**: Images won't upload or process

**Solutions**:
1. **Check file size** (should be under 10MB)
2. **Verify file format** (JPEG, PNG only)
3. **Try a different image**
4. **Check your internet connection**
5. **Refresh the page and try again**

#### Products Not Found

**Problem**: No products match your search

**Solutions**:
1. **Try different keywords**
2. **Be more specific** in your search
3. **Check spelling** of product names
4. **Try broader categories**
5. **Ask for recommendations instead**

#### Cart Issues

**Problem**: Items not adding to cart or cart not updating

**Solutions**:
1. **Refresh the page**
2. **Try adding items again**
3. **Check if the item exists** in the results
4. **Use different wording** to add items
5. **Contact support if persistent**

### Performance Issues

#### Slow Responses

**Problem**: The AI takes a long time to respond

**Solutions**:
1. **Wait patiently** - complex queries take time
2. **Try simpler queries**
3. **Check your internet speed**
4. **Close other browser tabs**
5. **Try during off-peak hours**

#### Page Freezes

**Problem**: The page becomes unresponsive

**Solutions**:
1. **Wait a few minutes** for processing
2. **Refresh the page**
3. **Close and reopen the browser**
4. **Clear browser cache**
5. **Try a different browser**

## ❓ FAQ

### General Questions

**Q: What products can I search for?**
A: Searchable products come from the currently loaded catalog. Check
`http://localhost:8010/capabilities` for the active hard-filter fields and
allowed enum values. The chain-server view at
`http://localhost:8009/capabilities` updates after the chain server restarts.

**Q: How accurate are the search results?**
A: The assistant model interprets your request using the catalog's advertised
capabilities. The catalog itself makes no generative-LLM call: it creates
text/image embeddings, applies exact filters, and deterministically fuses
retrieval results. More specific descriptions usually improve semantic
matching.

**Q: Can I save my preferences?**
A: Typed preferences are not extracted or saved between sessions. Conversation
text may be retained as part of the durable turn transcript, but this release
does not convert it into a reusable preference profile. The five shoppers in
the **Shop as** picker are fixed evaluation-derived examples, not learned
customer profiles.

**Q: Is my data private?**
A: The deployment processes data within its configured services. Shopper and
assistant text is stored in the operator-controlled memory-service SQLite
database so turns can be replayed and recent conversation can be loaded. Raw
uploaded media is not stored in that transcript. Operators are responsible for
database access, backup, retention, and deletion policy.

### Shopping Cart

**Q: How many items can I add to my cart?**
A: There's no limit to the number of items you can add to your cart.

**Q: Can I change quantities in my cart?**
A: Yes, you can ask to change quantities or remove items from your cart.

**Q: Does the cart persist between sessions?**
A: The cart is stored by the memory service. Refreshing the same browser tab
keeps its browser-scoped identity and cart. Closing that tab or starting a new
browser session creates a new bundled-UI identity, so the old cart is not
automatically reopened even though its database row may remain until operator
cleanup.

**Q: Can I see the total price?**
A: Yes, you can ask "What's my total?" or "How much is in my cart?" to see the total price.

### Image Search

**Q: What types of images work best?**
A: Clear, well-lit photos of individual products work best. Avoid group shots or cluttered backgrounds.

**Q: How large can my images be?**
A: Images should be under 10MB. Supported formats are JPEG and PNG.

**Q: Can I search for multiple products in one image?**
A: It's best to upload images with single products for more accurate results.

**Q: What if no similar products are found?**
A: Try uploading a different image or use text-based search instead.

### Technical Issues

**Q: What browsers are supported?**
A: Modern browsers like Chrome, Firefox, Safari, and Edge are supported.

**Q: Does it work on mobile devices?**
A: Yes, the interface is responsive and works on mobile phones and tablets.

**Q: Do I need to create an account?**
A: No, the application works without requiring an account or login.

**Q: Is there a mobile app?**
A: Currently, the application is web-based only. You can access it through your mobile browser.

### Product Information

**Q: Where do the products come from?**
A: The products are from a curated dataset of clothing and accessories.

**Q: Can I actually purchase these items?**
A: This is a demonstration application. The products shown are for display purposes only.

**Q: Are the prices accurate?**
A: The prices shown are from the demonstration dataset and may not reflect current market prices.

**Q: Can I see more product details?**
A: Yes. After a product search, ask about that item. The assistant reads
catalog-provided structured details such as material or care when present and
states when the catalog does not provide a requested fact.

---

For technical support or to report issues, please refer to the [main README](../README.md) or contact the development team. 
