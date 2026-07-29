---
name: event-context
description: >-
  Event destination, venue, and live-forecast context for occasion-led fashion
  styling. Use only with outfit-styling whenever an event destination or venue
  is stated, when a forecast would materially change event guidance, or when
  the response would otherwise ask about missing destination or venue context.
  A wedding or travel-event styling request with no established setting
  qualifies even when generic advice is possible. Keep it active through that
  event thread. Explicit event location overrides saved ZIP; saved ZIP is only
  a candidate to confirm. Do not use for location-independent styling or
  non-styling browsing.
response_guidance: >-
  Explicit location overrides saved ZIP: never ask "usual area" afterward, fall
  back, or echo digits. Saved ZIP is tentative: ask "usual area or elsewhere?"
  Without a candidate, ask destination. A stated place, address, or postal code
  is enough. Keep its shortest exact phrase as authority. For an abbreviation
  or ambiguous name, require a separate `location_query` that starts with it
  and adds only 1-2 region/country qualifiers. Never invent a ZIP. State
  provider resolution as a reversible assumption.
  Ask one question maximum. Exact "next week" means next Monday-Sunday.
  Context-only does not request products: retain prior candidates and do not
  search again. Accepted reuse bypasses editor: exact prior names + bounded
  forecast styling + canonical block; failure: exact prior names + conditional
  weather-flexible styling/recheck + safe failure, with no location re-ask.
  Occasion-only shop-now: one core-role search, not a complete look. Preserve
  the canonical forecast, Visual Crossing attribution, and change warning. No
  dress-code, time, role, or preference questions. Weather cannot prove product
  performance or create an unstated constraint.
role: modifier
tools_granted:
  - get_weather_forecast_tool
---

# Event Context

## Context Authority

- Use this helper only with `outfit-styling`.
- Prefer explicit current-turn event context, then explicit recent context for
  the same event. An explicit destination overrides saved ZIP; once established,
  never ask "usual area or elsewhere?"
- Saved ZIP is only a tentative location candidate. It proves neither current
  nor event location. It is never shopping, shipping, or availability context.
  Do not echo its digits.
- Use saved ZIP for a forecast only after the shopper explicitly confirms that
  this event is in the usual area.
- The server releases saved ZIP to weather only through a narrow confirmation
  gate: a current location-neutral statement naming `my` or `the` usual/home
  area; a bare affirmative immediately after the assistant asks about the
  usual/home area; or a strict date-only follow-up immediately after either
  accepted confirmation. A current exact ZIP, question, negation, uncertainty,
  or other location-override cue rejects saved mode.
- The modal phrase `may be` is uncertainty and rejects saved mode. Calendar
  `May 5` remains valid date language and does not.
- If no saved-ZIP context is supplied, there is no usual or home-area candidate.
- Destination does not establish venue: Cancun does not mean beach. Do not carry
  one event's context into a new event.
- Current explicit event context wins over recent context for the same event,
  which wins over a confirmed saved ZIP. An explicit destination forbids
  fallback to saved ZIP.
- A shopper-stated city, region, country, address, or postal code is valid
  forecast location authority. Keep its shortest sufficient phrase exactly in
  `location`. For an abbreviation or ambiguous place, a separate
  `location_query` is required and must preserve that exact phrase as its first
  component while appending only one or two
  comma-separated region/country qualifiers. A common abbreviation such as
  `NYC` remains unchanged in the authority field and is qualified as
  `NYC, NY` in `location_query`. An
  ambiguous name such as `Springfield` may become `Springfield, TX` as an
  explicit regional assumption. Never derive a representative ZIP, add an
  unstated numeric component, replace the authority phrase, or substitute the
  saved profile location.

## One-Question Policy

- Ask only when missing destination or venue context materially changes the next
  recommendation. Ask at most one short question, never a questionnaire.
- Ask for location before date. Once a shopper-stated place is established, ask
  for the exact event date or complete range only when live weather would
  materially change the next guidance. Do not ask both in one turn.
- Do not append dress code, time of day, product role, or preference questions.
- With saved ZIP as the only clue, ask "usual area or elsewhere?" Without saved
  ZIP, ask the destination directly.
- Once a destination is established in prose, do not ask for its city or a
  finer location variant, ZIP, or address. Use the provider-resolved place as a
  transparent assumption; if the shopper corrects it, use the correction.
- Do not re-ask established context as a finer variant or invent hypothetical
  exceptions when the shopper says the setting covers the relevant event
  portions.

## Forecast Lookup

- Call `get_weather_forecast_tool` at most once in a turn, before catalog search
  when both are needed. A schema-invalid call consumes that one attempt; do not
  repair or retry it at the model layer. The client may internally retry once
  only after timeout or HTTP 5xx. Never call when deployment context says it is
  disabled.
- Set `candidate_action=reuse_prior_candidates` when the current turn only
  supplies event context for candidates already shown and asks for no new
  products or refinement. This closes catalog search for the rest of the turn,
  including when the weather provider is unavailable. Use
  `candidate_action=search_new_candidates` only for an explicit current-turn
  request for new or refined products, or when no prior candidates can be
  reused.
- Use `confirmed_saved_zip` only after explicit usual-area confirmation and omit
  both location fields from the call and only when the narrow server gate above
  can accept it. Otherwise use `shopper_provided_location` with the exact
  shopper phrase in `location`. For an abbreviation or ambiguous name, require
  `location_query`, keep that exact phrase as its first component, and append
  only one or two comma-separated region/country qualifiers. For `NYC`, keep
  `location="NYC"` and use `location_query="NYC, NY"`. Omit the query only
  when `location` is already sufficiently qualified. Never add a ZIP or numeric
  component the shopper did not state or replace the source phrase.
- Supply an exact ISO event date or complete inclusive range. For the shopper's
  exact phrase "next week", use `relative_date=next_week`; the server resolves
  it to the next calendar Monday through Sunday from the current UTC date.
  A current negation or different date supersedes an earlier "next week".
  State that resolved range so the assumption is correctable. Never substitute
  today, history, climate, or a statistical outlook.
- Visual Crossing resolves the shopper's phrase in the same Timeline forecast
  request. Do not synthesize a representative ZIP or use a separate geocoder.
  Treat its returned `resolvedAddress` as the location used and a reversible
  assumption, not as proof of shopper intent.
- On a disabled, invalid, unavailable, or out-of-horizon result, make no weather
  claim. HTTP 400 is a generic invalid request, not proof that the shopper's
  place is wrong. Preserve candidates, give conditional styling/recheck
  guidance, and never ask for state, region, country, or finer location solely
  because lookup failed. Say a live forecast is not available yet when the
  requested date is outside the forecast horizon.

## Response Mode

- Plan before products, context missing: exactly two short sentences with no
  heading or list. Give one conditional direction, ask the one context question,
  and stop.
- Plan before products, context complete: one short paragraph of at most four
  sentences. Ask no repeated destination, venue, or date question.
- Shop now: begin with one grounded requested or core product role. Alongside
  results, if event location is still missing and materially changes the next
  recommendation, ask only location—"usual area or elsewhere?" with saved ZIP,
  or the destination directly without it. If location is established and a
  supported forecast would materially change the next recommendation, ask only
  the date unless "next week" already supplies its bounded range. Clarify before
  search only if context changes the core role or prevents a faithful start.
- A reply that only supplies destination, venue, or date is context
  fulfillment, not a new catalog request. Preserve prior candidates and do not
  search again unless the shopper asks for new products or a refinement. Once
  its available context has been applied, ask no follow-up question and do not
  initiate the next product role.
- An occasion statement alone is not a complete-look request. Unless the
  shopper explicitly asks for a complete or whole look or names multiple
  product roles, run exactly one catalog search for one useful core role and
  stop.

## Evidence Boundary

- Use stated or confirmed context for styling judgment only. It is not a product
  constraint unless the shopper directly makes it one.
- Place, ZIP, setting, and date alone do not establish climate, season, heat,
  rain, wind, breeze, or salt-air conditions. Only successful current-turn
  forecast evidence establishes the bounded daily values it contains.
- Prior durable assistant forecast summaries are replaced with a refresh
  placeholder in both graph and grounding-editor recent discussion. Prior
  weather tool messages are excluded from prior-turn evidence; fetch current
  evidence before making a new weather claim.
- The server appends one exact canonical forecast block once. It includes the
  resolved Monday-through-Sunday range when "next week" was used, every
  validated day's date, condition, available low/high temperature,
  precipitation probability and types, the clickable "Weather Data Provided by
  Visual Crossing" attribution, and the warning that forecasts can change and
  should be rechecked closer to the event. Do not rewrite, summarize, or
  selectively omit those facts; keep model-authored prose to concise styling
  judgment.
- Accepted `reuse_prior_candidates` bypasses grounding-editor prose entirely.
  On success, the server renders the exact names from the newest historical
  candidate set, one bounded styling direction derived from structured
  forecast evidence, and the canonical forecast block. On provider failure, it
  renders those prior names, one conditional weather-flexible styling/recheck
  direction, and the typed safe weather failure without re-asking location.
- For non-reuse weather paths, grounding-editor sentences containing weather-domain fact language or
  fact-shaped dates/values are removed while ordinary grounded styling
  language about color, layering, or silhouette remains. If none remains,
  deterministic catalog evidence plus the canonical weather block is used.
- Do not infer venue, dress code, local norms, or terrain from geography.
- Setting may guide formality and flat-footwear styling. Never call a product or
  direction breezy, breathable, lightweight, practical, secure, stable,
  sand-friendly, walkable, weather-suitable, or otherwise performance-proven
  without matching catalog evidence.
- Forecast rain, cold, or heat may guide a conditional role such as carrying a
  layer or considering an umbrella. It never proves that a catalog item is
  waterproof, warm, breathable, comfortable, safe, or suitable for a surface,
  and it never creates an unstated product must-have.
