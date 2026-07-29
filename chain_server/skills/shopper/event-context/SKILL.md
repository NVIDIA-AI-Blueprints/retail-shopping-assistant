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
  back, or echo digits. Only for accepted `event_location`, treat saved ZIP as
  tentative and ask "usual area or elsewhere?"; without a candidate, ask
  destination. For accepted `event_venue`, ask setting without inferring it
  from destination. A stated place, address, or postal code is enough. Keep its
  shortest exact phrase as authority. For an abbreviation or ambiguous name,
  require a separate `location_query` that starts with it and adds only 1-2
  region/country qualifiers. Never invent a ZIP. State provider resolution as
  a reversible assumption.
  Ask one question maximum. Exact "<weekday> next week" means that one day in
  the next Monday-Sunday window; bare "next week" means the full window.
  Ask only the activation-selected event-location, event-venue, or event-date
  question; ask none when activation selected none. Do not infer a venue from
  geography or a date question from enabled weather or a missing date.
  Context-only does not request products: retain prior candidates and do not
  search again. Missing location/venue reuse skips the reuse editor. Other
  eligible reuse accepts only a grounded venue quote plus allowlisted
  adjustment codes; the server renders fixed phrases, exact prior names,
  deterministic weather direction, only the accepted question, and any typed
  weather failure or canonical block.
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
- Destination does not establish venue: Cancun does not mean beach, outdoors,
  indoors, or any terrain. Do not carry one event's context into a new event.
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

- Activation has already selected the only permitted event-context follow-up in
  `event_context_next_question` from the current and recent shopper
  conversation. Follow that accepted value exactly: `event_location` only when
  destination is missing and material; `event_venue` only after destination is
  established when venue or setting is missing and material; `event_date` only
  after destination and any material venue are established, live weather is
  enabled and material, and a bounded date is neither established nor
  explicitly unavailable; `none` otherwise. Do not classify the follow-up again
  or infer one from weather configuration or missing context.
- An explicitly stated outdoor patio, beach, garden, rooftop, or open-air
  setting makes enabled live weather material. Once the destination and that
  setting are established but no bounded date is available, activation selects
  `event_date`, not `none`. This is semantic activation guidance, not a server
  keyword or venue alias table; never infer one of these settings from a
  destination alone.
- Ask only the accepted typed question when its missing context materially
  changes the next recommendation. Ask at most one short question, never a
  questionnaire.
- When activation selected `event_location`, ask location. When it selected
  `event_venue`, ask the venue or setting. When it selected `event_date`, ask
  for the exact event date or complete range. Do not ask more than one in a
  turn. When it selected `none`, ask no event-context follow-up.
- Do not append dress code, time of day, product role, or preference questions.
- With saved ZIP as the only clue, ask "usual area or elsewhere?" Without saved
  ZIP, ask the destination directly.
- Once a destination is established in prose, do not ask for its city or a
  finer location variant, ZIP, or address. Use the provider-resolved place as a
  transparent assumption; if the shopper corrects it, use the correction.
- Never infer beach, outdoor/indoor setting, or terrain from a destination. If
  that setting is material and not established, only accepted `event_venue`
  may ask for it.
- Example: after the shopper establishes `Cancun` but no setting, select
  `event_venue`. After they answer `on the beach`, select `event_date` only
  when live weather is enabled and material and no bounded date is established
  or explicitly unavailable; otherwise select `none`.
- Do not re-ask established context as a finer variant or invent hypothetical
  exceptions when the shopper says the setting covers the relevant event
  portions.

## Forecast Lookup

- Shopper-skill activation has already bound this turn's
  `event_context_action` and `event_context_next_question`. Treat those accepted
  activation values as authoritative; do not classify reuse/new product work or
  the next question again in this skill. With established candidates,
  activation could admit `search_new_candidates` only with a trimmed
  1–240-character `event_context_product_work_quote` whose case-insensitive
  text is an exact current-message substring. The activation model owns the
  semantic assertion that it is the shortest span explicitly requesting
  product, cart, or policy work; event context alone never supplies that
  provenance.
- Call `get_weather_forecast_tool` at most once in a turn, before catalog search
  when both are needed. A schema-invalid call consumes that one attempt; do not
  repair or retry it at the model layer. The client may internally retry once
  only after timeout or HTTP 5xx. Never call when deployment context says it is
  disabled. Never call weather to discover or prompt for missing context. When
  the accepted next question is `event_location` or `event_venue`, the runtime
  hides and execution-blocks weather. On reuse it also closes the tool loop
  immediately; on search-new it leaves normal product work open.
  When the shopper has not supplied an exact date, complete range, or exact "next
  week" phrase, the runtime hides and execution-blocks weather for that turn;
  ask a direct date question only when the accepted activation selected
  `event_date`. Otherwise continue without collecting weather-only context.
  Never invent `next_week` as a placeholder.
- Copy the activation-bound `event_context_action` into the weather call's
  `candidate_action` exactly. A mismatch fails before provider I/O.
  `reuse_prior_candidates` has already hidden and execution-blocked catalog
  search, product details, historical-product resolution, availability, and
  promotions. If date authority is missing, weather is also hidden and the
  tool loop is already closed for a tools-disabled response that may ask only
  the activation-selected location, venue, or date question. If date
  authority exists, consuming the one weather attempt closes the remaining
  loop, including on a validation or provider failure.
  `search_new_candidates` leaves normal granted product work available for an
  initial search or explicit new/refined-product request.
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
  exact phrase "<weekday> next week", use `relative_date=next_week` plus the
  matching lowercase `weekday`; the server validates the exact phrase and
  resolves that one day inside the next calendar Monday-through-Sunday window
  from the current UTC date. Never omit or change a stated weekday. For bare
  "next week", use `relative_date=next_week` without `weekday`; the server
  resolves the full window. A missing, invented, mismatched, mixed, or
  standalone corrected weekday fails closed. A current negation or different
  date supersedes an earlier relative date. State the resolved exact date or
  range so the interpretation is correctable. Never substitute today, history,
  climate, or a statistical outlook.
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
  heading or list. Give one conditional direction, ask the one context question
  selected by activation, and stop. If activation selected `none`, give the
  direction without inventing a follow-up.
- Plan before products, context complete: one short paragraph of at most four
  sentences. Ask no repeated destination, venue, or date question.
- Shop now: begin with one grounded requested or core product role. Alongside
  results, render only the question authorized by
  `event_context_next_question`: for `event_location`, ask "usual area or
  elsewhere?" with saved ZIP or the destination directly without it; for
  `event_venue`, ask the venue or setting without assuming beach,
  outdoor/indoor context, or terrain; for `event_date`, ask the date; for
  `none`, ask no event-context follow-up.
  Clarify before search only if context changes the core role or prevents a
  faithful start.
- A reply that only supplies destination, venue, or date is context
  fulfillment, not a new catalog request. Preserve prior candidates and do not
  search again unless the shopper asks for new products or a refinement. Ask
  only the activation-selected event-context question; for `none`, ask no
  follow-up and do not initiate the next product role.
- Missing-location or missing-venue context-only reuse skips the structured
  reuse editor. Other eligible reuse with a nonempty draft sends only bounded
  shopper-authored event text and the server-owned deterministic weather
  styling direction to that editor. No candidate, price, free-form draft, or
  full recent discussion is editor input.
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
  resolved exact date for "<weekday> next week" or the
  Monday-through-Sunday range for bare "next week", every validated day's date,
  condition, available low/high temperature,
  precipitation probability and types, the clickable "Weather Data Provided by
  Visual Crossing" attribution, and the warning that forecasts can change and
  should be rechecked closer to the event. Do not rewrite, summarize, or
  selectively omit those facts; keep model-authored prose to concise styling
  judgment.
- Eligible activation-time `reuse_prior_candidates` accepts no free-form editor
  prose. The structured editor must return exactly one JSON object with only
  `venue_quote` and `adjustments`. `venue_quote` must be a trimmed,
  single-line, 1–80-character exact case-insensitive substring of bounded
  shopper-authored event text that explicitly names the setting.
  `adjustments` must contain one or two distinct values from
  `streamlined_accessories`, `lower_profile_footwear`,
  `polished_unfussy_finish`, and `adaptable_finishing`. A null/missing or
  non-shopper quote, malformed JSON, extra key, wrong cardinality, duplicate,
  or unknown code falls back.
- The server maps valid codes to fixed phrases, escapes the canonical shopper
  quote, and assembles exact names from the newest historical candidate set,
  that fixed venue sentence when valid, its deterministic weather direction,
  only the accepted next question, and the current typed weather failure or
  canonical success block. The editor never authors shopper-facing prose,
  ranks candidates, or claims product performance.
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
