<a id="top"></a>
# 🛍️ NVIDIA AI Blueprint: Retail Shopping Assistant

<div align="center">

![NVIDIA Logo](https://avatars.githubusercontent.com/u/178940881?s=200&v=4)

**AI-powered retail shopping assistant with Deep Agents SDK orchestration**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://www.docker.com/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-AI-Blueprints/retail-shopping-assistant?style=social)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/NVIDIA-AI-Blueprints/retail-shopping-assistant)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/NVIDIA-AI-Blueprints/retail-shopping-assistant)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/commits)
[![Contributors](https://img.shields.io/github/contributors/NVIDIA-AI-Blueprints/retail-shopping-assistant)](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/graphs/contributors)

</div>

## 📋 Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [Architecture](#architecture)
- [Get Started](#get-started)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contribution Guidelines](#contribution-guidelines)
- [Community](#community)
- [References](#references)
- [License](#license)

## Overview

The Retail Shopping Assistant is an AI-powered blueprint that provides a comprehensive interface for an intelligent retail shopping advisor. The chain server uses the Deep Agents SDK as the assistant harness over deterministic shopping tools, with SSE-framed responses, image-based search, optional VLM media perception, and intelligent shopping cart management.

### Key Features

- 🤖 **Intelligent Product Search**: The assistant translates natural language
  into catalog queries and advertised filters; the catalog performs only
  deterministic embedding retrieval and ranking
- 🛒 **Deterministic Cart Management**: Read, add, remove, update quantities,
  and compute subtotals through typed tools
- 🧠 **Durable Turn Transcript**: A single memory-service SQLite replica
  starts every turn before agent work, finalizes its terminal outcome, and
  exactly replays finalized requests from ordered shopper/assistant records;
  rotating attempt tokens reject late finalizers after interrupted-turn
  recovery. A bounded tools-disabled compactor folds only memory's oldest exact
  raw-turn prefix into the versioned rolling summary after a successful
  response, so the result affects only the next request. Turn start uses an
  opt-in response contract so older chain instances receive their exact legacy
  shape and full bounded raw tail during rolling deployment or rollback. The
  chain accepts staging-era v1 abandoned/null-assistant rows and filters them
  before model context, while v2+ keeps strict memory-owned eligible-tail
  semantics
- 💭 **Durable Product Continuity**: Finalized product-card output becomes
  ordered `candidate_set_presented` evidence in SQLite; a typed resolver can
  recover one exact earlier product or require clarification without another
  catalog search or model call
- 👤 **Representative Shopper Picker**: Five immutable, database-backed
  shoppers mirror the committed live-evaluation behavior profiles; a new UI
  session requires an explicit dropdown choice of Guest mode or one of those
  five shoppers before chat starts. The selected ID is bound to the durable
  conversation and resolved into compact soft guidance
- 📍 **Grounded Event Context**: A styling modifier uses explicit event
  destination and venue context, or naturally confirms whether a selected
  profile's saved area applies. It never assumes that a destination determines
  the venue
- 🌦️ **Guarded Event Forecasts**: A registered read-only weather tool is
  granted only by event context beside outfit styling. It accepts a confirmed
  saved ZIP or an exact shopper-stated place, plus an exact event date/range or
  bounded `next week` phrase, gets at most one model-visible call on an eligible
  turn, and appends a canonical attributed forecast block. A successful
  normalized result may become one short-lived, exact-scope
  `weather_forecast.v1` receipt so a later comparison can reuse the event
  direction without another provider call or a repeated forecast block. An
  atomic durable scope and typed pending-question binding prevent one subject's
  location or date from silently completing another. Provider calls remain
  disabled by default
- 📚 **Enforced Shopper Skills**: Every turn first semantically selects and
  fully loads the smallest applicable skill set; each selected `SKILL.md`
  declares its role and tool grants, only their grant union becomes
  model-visible, and dispatch rechecks the grant before execution
- 🖼️ **Visual Search**: Upload images to find similar products
- 🎥 **Optional VLM Media Perception**: Enable a VLM role to analyze image and video uploads in shopping context
- 💬 **Conversational AI**: Natural language interactions
- 🔒 **Configurable Content Safety**: Built-in moderation and safety checks are on by default and can be disabled per request or config
- ⚡ **SSE Response Stream**: Event-stream response framing for chat clients; token-level Deep Agents streaming is a follow-up after the harness migration
- 📊 **Inference Visibility**: Model names, call counts, and token usage, with
  detailed ordered agent/tool diagnostics available only when explicitly
  enabled for a trusted operator or evaluation deployment
- 📱 **Responsive UI**: Modern, mobile-friendly interface

### Architecture

![Shopper Deep Agent architecture](docs/images/shopper-agent-architecture.svg)

[Open the architecture diagram at full size](docs/images/shopper-agent-architecture.svg).

The application follows a microservices architecture:
- **Chain Server**: Deep Agents SDK orchestration with six registered shopper
  skills, a required per-turn activation phase, a twelve-tool registry with
  deterministic per-skill binding, capability-derived search schemas, bounded
  search-schema repair, a category-aware no-I/O availability stub for known
  product refs, a no-I/O active-promotions stub, typed same-conversation product
  resolution, grounded response assembly, a configurable Deep Agents execution
  deadline, a request-scoped process-local checkpointer, and a registered
  scope-bound provider-neutral weather boundary whose model-visible execution
  tool accepts no location/date arguments
- **Catalog Retriever**: Generative-LLM-free text/image embedding search, hard
  filtering, normalized COSINE relevance scores, and deterministic result
  ranking
- **Memory Retriever**: Ordered durable turns with start/finalize and exact
  replay, a versioned rolling summary, separate newest raw-turn tail and
  oldest compaction prefix, a bounded versioned projection of short-lived typed
  weather receipts, one versioned current weather-planning scope core with
  per-component source turns plus a separately stored, revision-matched pending
  location/date binding, typed
  prior-skill continuity, presented-product
  events and a compact reference index, stable cart-line IDs, atomically idempotent
  add/remove/quantity mutations, an immutable five-row representative shopper
  registry, atomic conversation/profile binding, and request-scoped database
  sessions. Additive summary, receipt, and current-weather-scope lanes negotiate
  the highest response contract both services support. Contract 4 advertises
  atomic finalize-time scope resolution while contract 3 remains supported for
  rolling deployment. Contract v1 remains a legacy downstream-filtered raw
  lane; v2 and later require memory-owned context eligibility. Migration 11
  extracts any complete pre-split pending
  binding into a defaulted separate lane, drops incomplete unsourceable WIP
  pending fields, keeps the rollback-readable scope JSON free of v4 keys, and
  ignores pending state whose stored revision no longer matches the
  core; standard Compose
  exposes its host port on loopback only
- **Guardrails**: Content safety and moderation
- **UI**: React-based frontend interface with Guest/representative-shopper
  dropdown selection required before a new chat session starts

The UI initially gates chat on an explicit **Shop as** dropdown choice. Guest
omits the profile ID; a named selection sends only its server-owned ID. Durable
turn start resolves the row and prevents one conversation from switching
between Guest and another shopper or between two shoppers. The model receives
one small current-turn block with `shopper_type`, exact `behavior`, and
`saved_zipcode`. This is soft interaction/style guidance only: explicit shopper
instructions win, and a profile cannot invent a budget, product requirement,
cart action, product fact, skill choice, or tool permission. Changing the
selection clears visible chat/product state and rotates the browser-scoped
session, conversation, and cart identities. Reset keeps the explicit shopper
mode while rotating the conversation identity.

The `event-context` modifier accompanies only `outfit-styling` and is the sole
skill that grants `get_weather_forecast_tool`. Semantic activation selects it
only when physical context is part of the current styling subject: the shopper
supplies or changes destination/date/venue/weather context, directly requests
weather-aware guidance, answers its pending question, or explicitly continues
that established event, trip, or weather-planning subject. Hypothetical weather
relevance does not activate it for otherwise location-independent styling. An
explicit event destination or venue wins over
the selected profile's saved ZIP. When the ZIP is the only location clue and
location would materially change the recommendation, the assistant keeps any
starting direction conditional and asks at most one natural confirmation
around whether the event is in the shopper's usual area or elsewhere, without
echoing the ZIP; it must not discard that candidate and ask a bare destination
question. A destination such as Cancun does not establish that the event is on
a beach, outdoors, indoors, or on any terrain; Guest has no saved-location
fallback. On an explicit
plan-before-products turn, missing material context produces exactly two short
sentences and may include only the activation-selected location, venue, or date
question. With context complete, the answer stays to one short paragraph and asks no
further event-context question. An occasion-only shop-now request that does not
ask for a complete look or name multiple roles runs one catalog search for one
grounded core role. If location is still missing and materially changes the
next recommendation, activation selects only event location alongside the
results. Once destination is established, a materially missing venue/setting
can instead select the one venue question; geography never implies beach,
outdoor/indoor setting, or terrain. The response boundary receives only whether
a saved-ZIP candidate exists, not its digits, and restores deterministic
grounded candidates if a successful-search rewrite drops every returned
product. Current-turn non-weather business-tool evidence always uses ordinary
grounding and, after successful weather, prevents the response postprocessor
from restoring unrelated names from the historical-product index. The
protected decision renderer is selected structurally only when event context is
active, there is no current non-weather business-tool activity, and either a
current typed weather outcome or the one explicitly bound durable receipt
supplies event evidence. Missing location/venue or an empty draft skips its
decision editor. A separate prior-candidate fallback uses deterministic event
assembly only when the draft is empty. A comparison with current product
resolution/detail activity stays on ordinary grounding: a bound receipt may
guide the styling judgment silently, but it does not repeat the prior canonical
forecast facts. Other protected weather-evidence turns with a nonempty draft
give a narrow tools-disabled editor only bounded shopper-authored event text and
the server-owned deterministic weather styling direction. It accepts only exact
JSON containing an exact shopper-authored venue quote and one or two distinct
allowlisted adjustment codes; malformed or ungrounded output falls back. The
server renders fixed phrases and assembles exact prior names when present, the
deterministic weather direction, only the accepted location/venue/date question,
and a current typed failure or current canonical forecast block.

An explicit comparison of established candidates remains part of
`outfit-styling`; it does not create a comparison skill, deterministic intent
router, or rediscovery search. When either product is absent from current-turn
evidence, the model submits every compared prior product in the one batched
historical-resolution call, then reads details once per uniquely resolved ref in
separate model steps. The default two-read cap fits one pair. A missing or
ambiguous required product produces one concise clarification with no substitute
search. Weather is optional additional event evidence and never replaces the
product procedure or proves product performance. This sequence is model-owned
semantic procedure; deterministic handlers enforce exact refs, per-tool limits,
and evidence boundaries rather than classifying comparison intent.

The same Deep Agent owns that semantic procedure from skill activation through
the candidate answer. There is no post-answer semantic completion reviewer or
second correction trajectory that can discard the answer and reopen tools.
When live weather is enabled and material, a scope update has supplied valid
location/date authority or the shopper explicitly requests a refresh of an
unchanged complete scope, the selected skill tells that same agent to make its
one weather call before answering.
Deterministic layers remain limited to tool authorization, typed validation,
evidence accounting, and factual grounding.

Weather provider calls remain disabled by default. When an operator enables
`WEATHER_ENABLED` and supplies `WEATHER_API_KEY` to the chain server,
`event-context` supports event and non-event weather-aware styling beside
`outfit-styling`. When a prior weather scope exists, a request-local
tools-disabled semantic resolver compares the current query with the exact
shopper turns named by the stored location, date, and pending-question source
sequences. It makes one
forced typed control call and is neither a business tool nor a subagent.
The resolver returns only the semantic relation; it does not duplicate
location/date extraction. Normal activation is the sole producer of one atomic
`weather_scope` selection: it copies the scope revision and chooses `retain`,
`set`, or `clear` independently for location and date. Only current-turn
shopper authority can enter a `set`. The server applies the relation to that
selection: `new_subject` clears every retain, `same_subject` may retain ordinary
same-subject authority, and invalid, unavailable, or unclear output clears
proposed retains and blocks prior-dependent weather. A validated current-turn
`set`/`set` replacement remains independent authority and may require a fresh
forecast. Completing a typed pending component is a
narrower boundary: only exact-handle `answers_pending` may retain its stored
counterpart. That relation means the reply answers only the pending question;
a reply that also changes or withdraws the opposite component is
`same_subject`, whose explicit `set` or `clear` remains authoritative.

When activation asks for a missing location or date, that question is stored in
the singleton as a typed pending binding stamped with the originating turn ID
and sequence, even when the location/date authority values otherwise remain
unchanged. `answers_pending` is accepted only when the relation call echoes the
exact opaque pending handle and activation sets the named missing component.
The opaque handle is resolver-only and is never part of the normal activation
scope. The runtime carries it to memory only as the server-authored
`complete_pending_source_turn_id`; memory independently verifies the exact live
binding and canonical completion shape before committing. An existing opposite
component is retained, a current-turn replacement remains `set`, and an absent
opposite component becomes the newly bound pending question. If the same
unanswered question remains pending during an ordinary
same-subject update, the runtime may preserve its source only by supplying that
exact server-owned handle to memory; without it memory stamps the current
finalized turn. If semantic
resolution is unavailable or unclear, every proposed cross-turn retain is
cleared and receipt/refresh reuse is rejected. Weather remains blocked only
when the effective scope still depends on prior authority; a complete
current-turn `set`/`set` replacement may proceed without importing an older
subject.
The pending binding records that the question was already asked, so an
intervening product turn is instructed not to repeat it. This preserves a new
conference's stated date while asking its location, then safely combines the
location-only reply with that date. The durable singleton—not the rolling
summary or recent prose—is the only cross-turn weather authority.

The forecast tool is model-visible but has an empty argument schema. After
activation, the runtime derives the provider location and exact date window
solely from the effective typed scope. An incomplete scope or an accepted
location/venue/date question hides and execution-blocks only weather. Thus a
new Seattle subject with no date cannot inherit an older NYC date or reach the
provider. The same scope also handles “What should I wear in Denver next week?”
without creating an event or asking for a venue.
A scope update that produces a complete effective scope requires the
zero-argument weather call before prose. For an unchanged complete scope,
activation sets `weather_refresh=true` only when the shopper explicitly asks
for a fresh forecast; comparisons and other turns do not auto-refresh.

Activation may optionally bind one listed `weather_receipt_id` only with
`event-context`, `event_context_next_question=none`, no scope update, no refresh
request, and deterministic equality to the effective location/date scope. A
bound receipt blocks a redundant weather call. The accepted next-question
value remains the model's semantic one-question decision: location when
material and missing, venue only for an occasion, date when live weather is
material and the typed scope lacks a bounded window, or none. This is not an
intent classifier or event-anchor registry.
An explicitly shopper-stated outdoor patio, beach, garden, rooftop, or open-air
setting makes enabled live weather material; with destination and that setting
but no bounded date, activation selects `event_date`. Skill selection,
location, venue, materiality, and intent remain semantic model judgment. The
dynamic enum is typed argument consistency, not an intent router or keyword
routing layer.
Every event-context control is capability-scoped. Before nested weather
validation, the activation boundary deterministically removes
`event_context_next_question`, `weather_scope`, `weather_refresh`, and
`weather_receipt_id` when `event-context` was not selected. Those ungranted
values cannot mutate weather state, grant the forecast tool, or reject an
otherwise valid shopping activation.
The server does not infer one from enabled weather or missing context. Accepted
`event_location` or `event_venue` hides and execution-blocks weather. Event
context is additive and may gate only weather; every non-weather tool in the
selected grant union remains available for normal product, comparison, cart,
and policy work. Consuming the one forecast attempt does not close those
business tools. The successful event-context activation result, the
model-visible catalog-search description, and the outfit-styling procedure
repeat one semantic boundary: a reply that only supplies the destination,
venue, or date requested in the prior response fulfills context, so the model
retains established candidates without repeating non-weather product work. If
that same reply also asks to compare, refine, replace, search, check, manage the
cart, or answer policy, the normal selected-skill procedure still applies. This
is procedural model guidance, not a deterministic intent router or tool gate,
and it changes neither grants nor dispatch authorization. A date question may
appear only when activation selected `event_date`. Scope compilation accepts
either a confirmed saved area or a shopper-provided current location. Saved ZIP
remains behind the narrow current-turn confirmation gate and is never
model-visible. A shopper location must be copied from the current turn; an
optional provider qualifier preserves that phrase and appends only
region/country context, such as `NYC` → `NYC, NY`. No alias table,
representative ZIP, or separate geocoder is used. Visual Crossing resolves the
named place in the same forecast request and the returned place remains a
transparent, correctable assumption.

An exact ISO event date or complete date range remains the normal date
contract. The exact shopper phrase `next week` is the sole server-owned
relative-date shortcut. For exact `<weekday> next week`, the model supplies the
matching lowercase `weekday`; the server validates that shopper-authored phrase
and resolves it to one exact day inside the next Monday-through-Sunday window.
For bare `next week`, `weekday` is omitted and the server resolves the full
window. Missing, invented, mismatched, or mixed weekdays fail closed. A current
negation, standalone weekday correction, or different date supersedes an
earlier relative date. Without a complete effective location/date scope, the
runtime hides and execution-blocks weather for that turn. Prior raw-turn dates
can inform source-bound semantic resolution but cannot flow directly into the
adapter. The scoped tool accepts no date or location
arguments from the post-activation model.
The model may resolve an unambiguous
single-day phrase such as
`tomorrow` against that same prompt-visible UTC anchor and send the exact ISO
date. Using server UTC rather than caller/shopper local time is an explicit
current limitation; a genuinely ambiguous or unresolved relative date gets one concise
clarification only under that same enabled-and-material rule. Within a
scope-valid call, the adapter may make one additional provider
attempt only after a timeout or HTTP 5xx response. HTTP 400 remains a generic
invalid-request outcome; it is not proof that the shopper's location was
unresolved.
Provider-resolved place is omitted in saved-ZIP mode. For an explicit shopper
location, it is included in bounded current-turn evidence and the final
forecast block as a transparent, reversible provider assumption, not proof
that the event is there. Prior durable assistant forecast summaries are
replaced with a refresh placeholder in both graph and grounding-editor recent
discussion, and prior weather tool messages are excluded from prior evidence.
A successful current-turn tool call and its same-ID successful tool result may
instead be atomically promoted at successful finalization into one typed
`weather_forecast.v1` receipt. Failures, raw provider request/response data,
prepared provider endpoint URLs, keys, and exceptions are never promoted; the
pinned public attribution URL remains part of validated evidence. Memory prunes expired receipts, replaces an
older receipt for the same exact location/date scope, and retains at most four.
The active receipt lane is separate from the rolling summary, raw transcript,
and product ledger.
Memory evaluates receipt freshness atomically at durable turn start. That
accepted set is the validity snapshot for the request; it is not checked again
against the wall clock mid-turn. Before skill activation, the model sees only
receipt ID/type, shopper location/date scope, and `valid_until`—never forecast
conditions or other normalized evidence. Full evidence stays server-side and
can enter grounding only after activation explicitly binds the receipt.
Weather arguments/output are redacted from diagnostics and failed-turn partial
capture. Diagnostics retain only categorical weather call metadata—date shape,
location-source kind, provider-input kind, and typed outcome—with no location,
ZIP, date, resolved place, URL, body, or exception.
Activation diagnostics preserve the model-submitted next question and expose
the separately accepted server boundary when normalization changes it.
Saved profile ZIP is also scrubbed from diagnostic string keys and values, and
the complete grounding-editor prompt replaces those saved digits before the
editor call. Final rendering appends one exact canonical block with the resolved
exact date for `<weekday> next week` or the Monday-through-Sunday range for bare
`next week`, every validated daily date, condition, available temperature,
precipitation fact, the supplied Visual Crossing attribution, and the
forecast-change warning.
Only a current successful weather result produces that canonical block.
Unbound receipts are never grounding evidence. When activation binds one
still-valid exact-scope receipt, current successful weather takes precedence if
present; otherwise the receipt may guide styling. Product comparison uses that
guidance silently and strips exact forecast facts instead of repeating the
earlier block. Receipt diagnostics are categorical only, such as promotion
prepared or receipt bound; receipt identifiers, scopes, locations, dates, and
evidence are not exposed.
Optional receipt-promotion conflicts keep their typed error code across the
HTTP client boundary, so finalization retries once without that optional
promotion and does not rerun the model. A scope revision, resolution, or status
conflict is authoritative instead: the draft and unsent products are discarded
and the durable turn is terminalized as failed without applying the disputed
scope update.
For ordinary-grounding weather paths, grounding-editor sentences containing
weather-domain fact language or fact-shaped dates/values are removed while
ordinary grounded styling language remains. The structurally selected protected
weather-outcome decision path never accepts free-form editor prose: the exact
JSON decision contains only a shopper-grounded venue quote and one or two
allowlisted adjustments. Invalid, malformed, extra-key, non-shopper-quote, or
unknown/duplicate-code output falls back. The prior-candidate-only empty-draft
branch bypasses that editor and assembles deterministically. The server maps
valid codes to fixed phrases and deterministically assembles exact newest names
when present, its weather styling direction, the accepted question, and the
typed weather failure or canonical success block. A provider failure never
becomes a demand for a finer location.
Forecast
conditions cannot prove
product warmth, waterproofing,
breathability, comfort, safety, or another catalog attribute or create an
unstated hard constraint. This uses the existing query/SSE/UI response shapes
and requires no MCP server.

Before enabling weather for shopper traffic, the operator must confirm that the
selected Visual Crossing plan permits the intended attribution, display,
storage, and sharing. Include durable final assistant summaries and processing
by the downstream app model and output guardrails in that review. The key is
kept in the named server-side environment variable and is never stored in YAML
or an image.

Every turn still makes a fresh semantic skill-selection decision. The previous
turn's selected skill names are persisted with its durable output and supplied
to the next activation model step only as a read-only continuity signal; they
do not force routing or authorize tools.
When `event-context` is selected, that same mandatory activation step—not a
second classifier or the weather call—owns the typed
`event_context_next_question` boundary described above.
If the model selects an invalid skill composition, it receives the typed reason
and one correction attempt. A second invalid selection returns a deterministic
clarifying question without running catalog or commerce tools. Multiple
activation calls in one response execute none and clarify immediately.
Conversation context still matters: a terse item-only follow-up inside an
active outfit-building or style-led single-piece thread remains an
`outfit-styling` task.
`search_catalog_tool` exposes one flat, capability-derived executable search
schema. The model cannot submit a clarification or catalog-absence result
through that tool. The active skills and tool descriptions instruct the model
to author `requested_product_type`, select faithful advertised taxonomy, use a
category-only scope only when it judges that category to be a faithful parent,
or ask one concise clarification directly without a tool call. Runtime does not
parse current or recent shopper prose, suffix-match product phrases, classify
shopper-named versus open roles, or validate a semantic relationship between
`requested_product_type` and taxonomy. A category-only search records the
requested role and searched category separately; grounded output presents
category-scoped candidates under their actual catalog categories without
asserting a parent relationship or catalog absence.

At most one structural catalog repair is available for the entire turn. The
isolated repair receives the capability-derived typed `search_catalog_tool`,
compact server-generated Catalog capabilities, the current shopper message,
bounded sanitized validator feedback, and active shopper-skill context. Only
that search tool is available, parallel calls are disabled, and the repair may
either submit one corrected search or signal that clarification is needed by
returning no tool call. The server discards that model prose and emits the fixed
clarification `Could you clarify the product type or requirement you want me to
use?`. If another requested search scope already succeeded, its deterministic
grounded products are kept before that clarification. If another shopping tool
already completed, the existing grounding editor preserves that evidence with
the fixed clarification. The base runtime prompt, invalid AI/tool history, and
prior conversation history are absent.
Native validation feedback contains only rejected top-level field names; raw
Pydantic `input_value` metadata and free-form `requested_product_type` text are
never copied into the authoritative repair message. After activation, the
server rejects a model response containing more than one shopping tool call,
in addition to requesting `parallel_tool_calls=false`.
When request validation fails while finite structural fields remain valid,
middleware preserves `required_constraints`, `scope_complete`, and
`search_mode`; their names may appear in bounded `restored_fields` diagnostics,
but their values do not. It never derives or locks `requested_product_type` or
taxonomy from shopper prose. Any nonempty `unadvertised_requirements` lane fails
closed without retrieval or repair. Another validation failure after the one
repair closes to synthesis; a later distinct valid search may still run within
the configured search cap.

The resolved chain-server agent stack remains `deepagents==0.6.12`,
`langchain==1.3.11`, `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`.
`orjson==3.11.5` is pinned in every service requirement set that resolves it as
the last upstream release limited to the project's Apache-2.0/MIT license
policy. Redis checkpoint packages remain absent; the runtime supports only
process-local `CHECKPOINT_STORE=memory`. Each graph thread is request-scoped
with a collision-safe pair of conversation ID and request ID, deleted after
successful durable finalization, and retained only when finalization fails.
Deep Agents model-stage execution defaults to one 45-second deadline shared by
the graph and grounding editor. A graph timeout is captured as `agent_timeout`,
clears unsent products, finalizes the durable turn as failed, releases the
durable conversation turn, and then deletes its request checkpoint. The
grounding editor receives only the remaining time. Its timeout is finalized as
failed with `grounding_timeout`: search-only turns use the existing deterministic
catalog renderer, the protected context-only path uses deterministic event
assembly, and turns with current product-detail evidence deterministically retain
only those verified names, prices, categories, and listed detail fields, followed
by the typed weather outcome when present. Only a current tool-role result named
`get_product_details_tool` that begins with the server's canonical successful-
detail marker can enter that fallback. Other non-search turns return a fixed
retry/cart-check response instead of the unverified draft. Outside the protected
context-only path, editor errors and empty or whitespace-only output follow the
same evidence-preserving rule with `grounding_error`; invalid protected decisions
fall back deterministically. The verified-detail fallback does not invent a
comparative judgment.

For the serving-agent flow, see
[Shopper Agent Architecture](docs/SHOPPER_AGENT_ARCHITECTURE.md). The
[Documentation Hub](docs/README.md) links the detailed contracts and operations
guides.

### Catalog lifecycle and capability publishing

1. At startup, the catalog service loads `enriched_products.jsonl` and its
   field-role sidecar into one validated snapshot.
2. That snapshot supplies embedding documents, product details, filters, and
   the live contract at `http://localhost:8010/capabilities`.
3. On its first successful fetch, the chain server caches one process-wide
   contract shared by all sessions. Its aggregate endpoint at
   `http://localhost:8009/capabilities` returns the cached catalog contract with
   the other runtime capabilities.
4. Cached capabilities generate `search_catalog_tool`'s flat schema:
   `semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
   `required_constraints`, `scope_complete`, and optional `search_mode`.
   Taxonomy values, hard-filter properties and enum values, typed numeric range
   shape, and search-mode values come from the active contract. This typed schema
   deliberately omits cross-field validators; the handler applies a separate
   structural capability model to the same payload. Invalid individual values
   fail at the tool boundary, while cross-field failures reach capability-aware
   handler validation and can receive one bounded repair. The agent semantically selects
   exact advertised values; deterministic chain code validates and maps the
   selection against the capability-owned
   exact category/subcategory relationships and returns corrective feedback for
   incoherent combinations. Each call covers at most one category. Every text
   search carries
   `requested_product_type`: the shortest product noun or true umbrella from
   the shopper's current turn or direct antecedent. It excludes color,
   material, fit, occasion, weather, and style modifiers. It is provenance, not
   taxonomy or ranking text, and is `null` only for image-only search. The model
   owns that provenance plus all alternative, comparison, ordering, negation,
   and faithful-parent semantics. Runtime does not derive or suffix-match those
   meanings from shopper prose and does not validate
   `requested_product_type` against taxonomy. When the model submits multiple
   advertised subcategories from one category through the typed
   taxonomy field, the valid request remains one catalog execution; its
   candidate window expands for that selection, then rank-preserving selection
   keeps one returned candidate per selected subcategory when available before
   trimming to the configured result count. The runtime does not derive that
   selection from the shopper's raw text. Each search also requires
   `shopper_guidance`: one nonempty, product-agnostic
   sentence authored before retrieval under the active skill to connect the
   selected role to the shopper's goal or direct antecedent. Empty guidance is
   valid only for image-only search.
5. If the model chooses a category-only scope, the result records the
   model-authored requested role and advertised category as separate facts. The
   response discloses that category scope and keeps each result's actual
   category; runtime does not certify that the category is a parent or that the
   requested type is absent. A directly stated must-have missing from the
   generated schema is placed in `unadvertised_requirements`, while preference,
   styling, occasion, weather, and anchor context remain in the semantic query.
   A product type never belongs in `unadvertised_requirements`. Any nonempty
   unadvertised-requirement lane fails closed before retrieval and is not
   repaired; runtime does not classify it as stated versus inferred by parsing
   shopper prose. One structural schema/capability repair is available for the
   whole turn, preserving independently valid `required_constraints`,
   `scope_complete`, and `search_mode`. A successful partial search may advance
   to a new valid role, but the turn receives no second repair. The configured
   turn cap remains three searches. When a successful or zero-result search
   consumes the final configured slot, its result records
   `SEARCH_BUDGET_EXHAUSTED`; the next model step omits only
   `search_catalog_tool`. This prevents a fourth search while preserving product
   details, availability, cart work, and honest partial synthesis.
6. The catalog validates executable requests again, generates embeddings,
   applies hard filters, and ranks results. It performs no shopper-language
   interpretation or chat/completion call.

Deterministic validation does not compare `requested_product_type` with the
shopper's prose or selected taxonomy. The semantic query remains independent
soft ranking direction and need not repeat the taxonomy noun. Successful search
evidence preserves it as a private ranking preference.
For a completed successful search-only turn, the runtime allows one final
tools-disabled synthesis under the active skill and then grounds that draft
against tool-role evidence. The pre-retrieval `shopper_guidance` and active
skill's static `response_guidance` support deterministic fallback when synthesis
or editing cannot produce an answer. If the shopper's goal depends on a
material, fit, comfort, durability, care, weather, or other functional property
that the evidence does not confirm, final grounding states that gap and presents
the candidates as the closest catalog or styling direction rather than as
proven suitable. Deterministic fallback ends with the same generic disclosure.
Before fallback guidance is serialized, a
narrow runtime scrub replaces documented unsupported outdoor/weather guarantee
language with neutral guidance for the selected role. This changes only response
framing; the semantic query, taxonomy, constraints, and executed search remain
unchanged. The scrub includes outdoor-surface or outdoor-walking claims and
constructions such as "handle rain," "work well for outdoor surfaces," or
"stay secure for outdoor walking," plus `wet conditions` and "works well in wet
weather/conditions." Candidate results, filters, and the assistant draft are not
rewritten into guidance after retrieval. Deterministic code
separately lists every returned candidate with its name, price, category,
and only the confirmed filters from that candidate's search. For multi-role
results, each guidance sentence is grouped with the products from the search
that produced it. Candidate groups deduplicate by `product_ref`, not display
name: the same catalog product appears once, while distinct products that share
a name remain distinct. Mixed-outcome turns retain every successful product
group when a later scope has an unsupported requirement and append the honest
gap. The fixed unsupported-requirement response is used only when that rejection
is the sole current-turn business-tool outcome;
otherwise the other outcome remains available for rendering or synthesis. If
successful search evidence remains incomplete, the renderer adds a neutral
offer to continue with the next requested piece or search scope.
Separate searches are never flattened into one global filter claim. Zero-result
evidence retains its exact taxonomy and filter scope and cannot support a claim
about a different product type or the whole catalog.
Operator diagnostics include bounded `catalog_scope_outcomes` for zero-result
scopes. The grounding boundary keeps current-turn and prior-turn
tool-role evidence separate, so earlier results can resolve references but
cannot prove that a new search or cart mutation ran. If every current-turn
business call is a rejected catalog search and no current product evidence
exists, the runtime returns a fixed retry response before model-based response
editing; prior evidence cannot be presented as results from the rejected search.

Final-response extraction ignores tool messages, assistant tool-call messages,
and internal skill-activation markers. If a completed graph contains no
shopper-facing answer, the runtime returns a safe retry response and records the
termination reason as `incomplete_agent_response` rather than exposing internal
content.

At turn start, the memory service returns the durable semantic summary, a
bounded newest raw-turn tail strictly after its watermark, and a separate
bounded oldest unsummarized prefix for compaction, plus the authoritative cart
and a service-issued attempt token. The summary, exact raw discussion, and
historical product index remain separate prompt/state lanes: summary prose is
continuity only and cannot become exact shopper wording, product/cart/tool
evidence, location/date authority, policy, availability, or current weather.
Blocked and abandoned turns remain durable and exactly replayable but are
excluded from both raw lanes; only completed or failed turns with assistant
text are eligible. After a successfully guarded response, default configuration
triggers one tools-disabled compactor call at six unsummarized turns, keeps at
least two raw, and folds the largest fitting contiguous part of memory's oldest
prefix. If its first turn alone exceeds the input budget, only the compactor
receives marked head-and-tail excerpts of that one turn; durable and replay
text remain exact.
The compactor receives no current query, profile/ZIP, cart, product ledger,
media, tool transcript, diagnostics, or request identity. Invalid, timed-out,
or failed compaction leaves the old summary/watermark and all raw turns intact.
Configuration reserves input headroom beyond the maximum summary output. An
accepted summary update commits atomically with turn
finalization and is visible only to the next request; a summary-only conflict
gets one finalization retry without the optional update and never reruns the
model. Only the latest abandoned turn can reopen; reopening retains its request
identity but rotates the attempt token, so a late finalize cannot overwrite the
retry. The memory service also returns a compact index of products actually
presented as ordered cards on earlier turns. One typed batch resolution call
can restore an exact product as request-local evidence; zero or multiple
matches require clarification and never authorize a guess.

LangGraph `MemorySaver` now holds only one request's working graph state under a
collision-safe pair of conversation ID and request ID. It is deleted only after
durable finalization succeeds; a finalize failure preserves that checkpoint.
The compact historical-product index is capped at 16,384 characters, and its
typed batch resolver can run at most once per turn. Caller-supplied persona data
is not accepted as turn context. The fixed representative shoppers use a typed,
bounded server-owned registry and an atomic turn-start binding; only the
resolved three-field snapshot enters the current model input. Guest turns carry
neither that snapshot nor profile-specific prompt rules.

Catalog values are never copied into agent or catalog code. After replacing the
JSONL or sidecar, restart and verify the catalog service first, then restart and
verify the chain server so its process-lifetime cache matches the new snapshot.
See [Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md) for the complete flow
and [Catalog Schema and Filters](docs/CATALOG_FILTERS.md) for the sidecar rules.
The exact published response is documented in
[Catalog Retriever capabilities](docs/API.md#catalog-retriever-get-capabilities).

## Get Started

### Prerequisites

- **Docker**: Version 20.10+ with Docker Compose plugin
- **Python**: Host Python for deployment helpers. From the cloned repo, install
  deploy-helper dependencies with:
  ```bash
  python -m pip install --user -r requirements-deploy.txt
  ```
- **NVIDIA NGC Account**: For API access ([Get API Key](https://ngc.nvidia.com/))
- **Hardware**: 4x H100 GPUs (preferred) or 4x A100 GPUs (minimum) for local deployment, or cloud access

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
   cd retail-shopping-assistant
   ```

2. **Authenticate with NVIDIA Container Registry**:
   ```bash
   docker login nvcr.io
   ```
   Use `$oauthtoken` as the username and your NGC API key as the password.

3. **Install host deploy-helper dependencies**:
   ```bash
   python -m pip install --user -r requirements-deploy.txt
   ```

4. **Create and source an environment profile**:
   ```bash
   cp .env.example .env
   $EDITOR .env
   source .env
   ```

   Set `NVIDIA_API_KEY` in the file. The env file is a sourceable shell file;
   sourcing it also sets `COMPOSE_DISABLE_ENV_FILE=1` so Docker Compose uses
   the exported shell environment instead of auto-parsing repo-root `.env`.
   `CHECKPOINT_STORE=memory` is the only supported graph-checkpoint
   configuration. Graph checkpoints disappear on chain-server restart and are
   not shared across replicas. Separately, Compose stores the single-replica
   memory-service SQLite database at `/data/context.db` on the `memory-data`
   named volume. A production shared graph backend and multi-replica memory
   design remain open decisions described in the
   [Deployment Guide](docs/DEPLOYMENT.md).

   Weather remains disabled by default. Leave `WEATHER_ENABLED=false` unless
   the operator has confirmed that the selected Visual Crossing plan permits
   the intended shopper-facing attribution, display, storage, and sharing,
   including durable assistant summaries and downstream app-model/guardrail
   processing. When approved, set `WEATHER_API_KEY` only in the ignored `.env`,
   process environment, or deployment secret store.

5. **Validate and deploy**:
   ```bash
   python scripts/model_config.py show --validate
   python scripts/model_config.py deploy --build
   ```

   The helper prints resolved endpoints without printing API keys. By default,
   `shared/configs/models.yaml` uses NVIDIA Build hosted endpoints for the
   app LLM, text embeddings, image embeddings, and guardrails, and starts no
   local NIM containers. The `vlm` role uses a hosted endpoint by default for
   image/video media understanding in addition to image embedding search; set it
   to `disabled` in `models.yaml` when that capability should be off.

   For local NIMs, edit the desired model roles in
   `shared/configs/models.yaml` to `source: local_nim`, then run:
   ```bash
   # Set LOCAL_NIM_CACHE in the sourced env profile first.
   mkdir -p "$LOCAL_NIM_CACHE" && chmod a+w "$LOCAL_NIM_CACHE"
   python scripts/model_config.py show --validate
   python scripts/model_config.py deploy --build
   ```

   Model routing lives in `shared/configs/models.yaml`.

6. **Access the application**: Open your browser to `http://localhost:3000`

7. **Stop the containers**:

   **Application services**:
   ```bash
   docker compose -f docker-compose.yaml down
   ```

   **Local NIM services, if `models.yaml` started any**:
   ```bash
   docker compose -f docker-compose-nim-local.yaml down
   ```

For detailed installation instructions, see [Deployment Guide](docs/DEPLOYMENT.md).

## Deploy on NVIDIA Brev

For a streamlined cloud deployment experience, you can deploy the Retail Shopping Assistant on **NVIDIA Brev** using GPU Environment Templates (Launchables):

**[NVIDIA Brev Deployment Guide](docs/BREV.md)** - Complete step-by-step instructions for deploying on Brev

### Why Choose NVIDIA Brev?

- **One-Click Deployment**: Pre-configured GPU environments with automatic setup
- **Managed Infrastructure**: No need to manage servers or GPU clusters
- **Secure Access**: Built-in secure tunneling for web interface access  
- **Flexible Resources**: Choose from H100, A100, and other GPU configurations
- **Cost-Effective**: Pay only for actual usage time

The Brev deployment guide walks you through the entire process from creating a Launchable to accessing your fully functional retail shopping assistant.

## Documentation

- **[Project Status](STATUS.md)**: Current implementation, verification, quality qualification, and remaining risks
- **[User Guide](docs/USER_GUIDE.md)**: How to use the application
- **[API Documentation](docs/API.md)**: Complete API reference
- **[Catalog Schema and Filters](docs/CATALOG_FILTERS.md)**: JSONL field roles and data-derived filter capabilities
- **[Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md)**: Start here for JSONL ingest, lifecycle-cached capabilities, compact agent discovery, validation, and retrieval
- **[Commerce Contracts](docs/COMMERCE_CONTRACTS.md)**: Internal product, cart, and commerce tool contracts
- **[Shopper Agent Architecture](docs/SHOPPER_AGENT_ARCHITECTURE.md)**: Clean map of the published catalog, turn flow, skills, tools, and memory boundaries
- **[Shopper Deep Agent Architecture — 2026-07-29](docs/SHOPPER_DEEP_AGENT_ARCHITECTURE_2026-07-29.md)**: Source-audited serving flow, exact three-turn live gate, observed result, and the built durable rolling-summary and selective weather-receipt boundaries agreed on 2026-07-30
- **[Shopper Agent Leadership Note](docs/SHOPPER_AGENT_LEADERSHIP_NOTE.md)**: Concise request flow, memory ownership, worked styling example, and prioritized next steps
- **[Shopper Agent Tool Registry](docs/SHOPPER_AGENT_TOOL_REGISTRY.md)**: Registered Deep Agents tools for the shopper-serving agent
- **[Shopper Agent Skill Registry](docs/SHOPPER_AGENT_SKILL_REGISTRY.md)**: Registered Deep Agents skills and markdown tuning loop
- **[Deep Agents Migration Plan](docs/DEEP_AGENTS_MIGRATION_PLAN.md)**: SDK migration, session isolation, tools, skills, and scaling notes
- **[Deep Agents Cart Tool Goal](docs/DEEP_AGENTS_CART_TOOL_GOAL.md)**: Minimal cart-tool smoke gate and constraints
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Installation and setup instructions
- **[Testing and Evaluation](tests/README.md)**: Unit, integration, and
  Challenger/Judge workflows; multi-turn judging uses the actual generated
  conversation plus bounded current-turn catalog evidence from successful
  search and detail tools
- **[Documentation Hub](docs/README.md)**: Complete documentation index

## Contribution Guidelines

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup and environment configuration
- Coding standards and best practices
- Testing guidelines and examples
- Pull request process and code review guidelines

## Community

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues)
- **Documentation**: [Comprehensive guides and references](docs/README.md)

## References

### NVIDIA AI Blueprints
- [NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints): Collection of AI application blueprints
- [NVIDIA NIM](https://catalog.ngc.nvidia.com/orgs/nim): Containerized AI models
- [NVIDIA NGC](https://ngc.nvidia.com/): AI platform and container registry

### Technologies Used
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview): Agent harness for tool and skill orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph): Runtime used underneath Deep Agents
- [FastAPI](https://fastapi.tiangolo.com/): Modern Python web framework
- [React](https://reactjs.org/): JavaScript library for building user interfaces
- [Milvus](https://milvus.io/): Vector database for similarity search

### Related Projects
- [NVIDIA Retrieval QA](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nv-embedqa-e5-v5): Embedding model for semantic search
- [NV-CLIP](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nvclip): Visual understanding model for image retrieval
- [Nemotron 3 Super](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3-super-120b-a12b): Large language model

## License

GOVERNING TERMS: Use of the blueprint software and materials and NIM containers are governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-specific Terms for AI products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/);  and the use of models is governed by the [NVIDIA Community Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/).
 
ADDITIONAL INFORMATION: [Llama 3.1 Community License Agreement](https://www.llama.com/llama3_1/license/) for Llama 3.1 70B Instruct NIM, Llama 3.1 NemoGuard 8B - Content Safety and Llama 3.1 NemoGuard 8B - Topic Control models, built with Llama, (ii) MIT license for NV-EmbedQA-E5-v5.
 
This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use, found in [License-3rd-party.txt](/LICENSE-3rd-party.txt).
 
Use of the product catalog data in the retail shopping assistant is governed by the terms of the [NVIDIA Data License for Retail Shopping Assistant](/LICENSE-assets.txt) (15Aug2025).

---

<div align="center">

[Back to Top](#top)

</div>
