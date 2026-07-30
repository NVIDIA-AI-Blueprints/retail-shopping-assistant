# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date as CalendarDate
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, AsyncIterator, Literal, Sequence
import unicodedata
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
import requests

from .agenttypes import Cart, ShopperContext, State
from .catalog_capabilities import (
    CatalogCapabilitiesClient,
    effective_filter_capabilities,
    format_catalog_capabilities_for_prompt,
)
from .catalog_execution import execute_catalog_search
from .catalog_request import (
    CatalogSearchIntent,
    CatalogSearchPlan,
    build_catalog_search_plan,
)
from .catalog_scope import CATALOG_SEARCH_RULES
from .commerce_tools import (
    add_cart_item,
    check_active_promotions,
    check_product_availability,
    get_cart,
    get_product_details,
    get_store_policy,
    remove_cart_item,
    update_cart_item,
)
from .conversation_memory import (
    ConversationMemoryClient,
    ConversationMemoryError,
    ConversationSummaryAdvance,
    FinalTurnStatus,
    TurnReplayOutput,
    TurnStartResult,
    format_conversation_context,
)
from .conversation_summary import (
    CONVERSATION_SUMMARY_SYSTEM_PROMPT,
    build_conversation_summary_work,
    parse_conversation_summary_output,
)
from .conversation_products import (
    ConversationProductsClient,
    ConversationProductsError,
    ProductEvidence,
    ProductReferenceDescriptor,
    ResolveConversationProductsRequest,
    format_historical_product_index,
    format_product_resolution,
)
from .media_perception import MediaPerceptionClient
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_EVENT_CONTEXT_INVALID,
    SKILL_ACTIVATION_EVENT_CONTEXT_REQUIRES_STYLING,
    SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
    SKILL_ACTIVATION_MULTIPLE_PRIMARY,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_ACTIVATION_WEATHER_RECEIPT_INVALID,
    SKILL_TOOL_NOT_GRANTED,
    ShopperSkillActivationError,
    ShopperSkillActivationMiddleware,
    selected_skill_names_for_turn,
)
from .tool_policy import (
    load_shopper_skill_registry as _shopper_skill_registry,
    validate_registered_tool_names,
)
from .tool_loop_control import (
    CONSTRAINT_REVIEW_PREFIX,
    SEARCH_BUDGET_EXHAUSTED_PREFIX,
    SEARCH_SCOPE_COMPLETE_PREFIX,
    SEARCH_VALIDATION_ERROR_PREFIX,
    SERVER_CATALOG_CLARIFICATION,
    SERVER_RESTORED_TOOL_CALL_FIELDS,
    ToolLoopControlMiddleware,
    _SERVER_REJECTED_TOOL_CALLS,
)
from .weather import (
    VISUAL_CROSSING_ATTRIBUTION_LABEL,
    VISUAL_CROSSING_ATTRIBUTION_URL,
    WeatherFailure,
    build_weather_client,
    weather_failure,
)
from .weather_tool import (
    EventWeatherRequest,
    WeatherForecastEvidence,
    WEATHER_FORECAST_EVIDENCE_PREFIX,
    WEATHER_FORECAST_FAILURE_PREFIX,
    get_event_weather_forecast_tool,
    parse_weather_tool_evidence,
    weather_date_context_available,
)
from shared.commerce_contracts import (
    AddCartItemInput,
    CatalogCapabilities,
    Cart as CommerceCart,
    CartMutationResult,
    CheckActivePromotionsResult,
    CheckProductAvailabilityInput,
    CheckProductAvailabilityResult,
    CommerceError,
    GetCartInput,
    GetProductDetailsInput,
    GetStorePolicyInput,
    GetStorePolicyResult,
    ProductDetail,
    ProductSummary,
    RemoveCartItemInput,
    UpdateCartItemInput,
)
from shared.weather_receipts import (
    SavedAreaWeatherScope,
    ShopperLocationWeatherScope,
    WeatherForecastReceipt,
    WeatherReceiptEvidence,
    WeatherReceiptPromotion,
)


logger = logging.getLogger(__name__)
_SHOPPER_SKILLS_ENV = "SHOPPER_SKILLS_ROOT"
_SHARED_CONFIG_ROOT_ENV = "SHARED_CONFIG_ROOT"
_STORE_POLICIES_RELATIVE_PATH = Path("chain_server/store_policies.yaml")


def _build_checkpointer():
    """Return the process-local LangGraph checkpointer."""

    store = os.environ.get("CHECKPOINT_STORE", "memory").strip().lower()
    if store != "memory":
        raise ValueError(
            "CHECKPOINT_STORE currently supports only 'memory'. "
            f"Received: {store!r}."
        )
    return MemorySaver()


def _store_policies_path() -> Path:
    """Resolve controlled policy content outside the agent-readable skill root."""

    configured_root = os.environ.get(_SHARED_CONFIG_ROOT_ENV, "").strip()
    if configured_root:
        return Path(configured_root) / _STORE_POLICIES_RELATIVE_PATH

    deployed_path = Path("/app/shared/configs") / _STORE_POLICIES_RELATIVE_PATH
    if deployed_path.is_file():
        return deployed_path

    return (
        Path(__file__).resolve().parents[2]
        / "shared"
        / "configs"
        / _STORE_POLICIES_RELATIVE_PATH
    )


try:
    from deepagents.backends import FilesystemBackend as _FilesystemBackend
except Exception:  # pragma: no cover - dependency import is validated at runtime.
    _FilesystemBackend = None

_SEARCH_RESULT_GROUNDING_NOTE = (
    "SEARCH_RESULT_GROUNDING_NOTE: Use search results for candidate names, prices, "
    "categories, image availability, confirmed filters listed in "
    "SEARCH_FILTER_EVIDENCE, advertised taxonomy listed in "
    "SEARCH_TAXONOMY_EVIDENCE, and modest styling fit only. Treat product names as "
    "display names, not attribute evidence. Do not infer or group-claim "
    "length, color, print, material, care, construction, fit, comfort, weather, "
    "grass, gravel, or best-in-category performance from names or search snippets. "
    "Do not override a confirmed filter based on words in a display name."
)
_SEARCH_NO_MATCH_GROUNDING_NOTE = (
    "SEARCH_NO_MATCH_GROUNDING_NOTE: Zero products matched this exact "
    "advertised taxonomy and filter scope. This result does not establish "
    "whether products exist in a different, unsearched, or unadvertised "
    "product type."
)
_SEARCH_FILTER_EVIDENCE_PREFIX = "SEARCH_FILTER_EVIDENCE:"
_SEARCH_TAXONOMY_EVIDENCE_PREFIX = "SEARCH_TAXONOMY_EVIDENCE:"
_SEARCH_DIRECTION_EVIDENCE_PREFIX = "SEARCH_DIRECTION_EVIDENCE:"
_SEARCH_GUIDANCE_EVIDENCE_PREFIX = "SEARCH_GUIDANCE_EVIDENCE:"
_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX = "SEARCH_SCOPE_RELATION_EVIDENCE:"
_CATALOG_SCOPE_OUTCOME_PREFIX = "CATALOG_SCOPE_OUTCOME:"
_WEATHER_TOOL_NAME = "get_weather_forecast_tool"
EventContextNextQuestion = Literal[
    "event_location",
    "event_venue",
    "event_date",
    "none",
]
_EVENT_CONTEXT_DATE_REQUIRED_WEATHER_BLOCK = (
    "STOP_TOOL_USE: No shopper-authored event date is established for this "
    "turn. Do not call weather or invent a date argument."
)
_EVENT_CONTEXT_LOCATION_REQUIRED_WEATHER_BLOCK = (
    "STOP_TOOL_USE: Event location is still the activation-owned next "
    "question, so weather lookup cannot run this turn."
)
_EVENT_CONTEXT_VENUE_REQUIRED_WEATHER_BLOCK = (
    "STOP_TOOL_USE: Event venue or setting is still the activation-owned next "
    "question, so weather lookup cannot run this turn."
)
_EVENT_CONTEXT_RECEIPT_BOUND_WEATHER_BLOCK = (
    "STOP_TOOL_USE: A valid durable forecast receipt is bound for this exact "
    "event scope. Use that receipt and do not refresh weather this turn."
)
_EVENT_CONTEXT_RECEIPT_BOUND_REMINDER = (
    "DURABLE_WEATHER_RECEIPT_BOUND: The exact-scope typed forecast receipt is "
    "authoritative for weather-aware styling on this turn. Do not repeat its "
    "forecast facts during a product comparison."
)
_EVENT_CONTEXT_ADDITIVE_ACTIVATION_REMINDER = (
    "EVENT_CONTEXT_ADDITIVE_BOUNDARY: Tool availability is not a product "
    "request. A shopper reply that only supplies the destination, venue, or "
    "date requested in the prior response is context fulfillment, even when "
    "it changes styling advice: preserve established candidates and do not "
    "call a non-weather business tool. A same-turn comparison, refinement, "
    "replacement, search, check, cart request, or policy request continues "
    "its selected skill's normal tool procedure."
)
_WEATHER_DIAGNOSTIC_REDACTION = {"redacted": True}
_WEATHER_PARTIAL_OUTPUT_REDACTION = "WEATHER TOOL OUTPUT REDACTED"
_PRIOR_WEATHER_CONTEXT_REDACTION = (
    "(prior forecast prose omitted; use only current evidence or an explicitly "
    "bound durable receipt)"
)
_WEATHER_UNCERTAINTY_TEXT = (
    "Forecasts can change, so recheck closer to the event."
)
_WEATHER_FACT_LANGUAGE_RE = re.compile(
    r"(?:"
    r"\b(?:forecasts?|weather|conditions?|outlook|sk(?:y|ies)|"
    r"sun(?:ny|shine)?|cloud(?:s|y|iness)|overcast|"
    r"rain(?:s|ed|ing|y|fall)?|shower(?:s|ed|ing)?|"
    r"drizzl(?:e|es|ed|ing)|downpours?|snow(?:s|ed|ing|y|fall)?|"
    r"ice|icy|sleet(?:s|ed|ing)?|hail(?:s|ed|ing)?|"
    r"storm(?:s|ed|ing|y)?|thunder(?:s|ed|ing|storms?)?|lightning|"
    r"fog(?:s|ged|ging|gy)?|misty?|"
    r"wind|winds|windy|breez(?:e|es|y)|gust(?:s|y)?|"
    r"muggy|chilly|balmy|humidity|humid|temperatures?|degrees?|"
    r"precipitation|flurries|frost|freezing|blizzards?|tornado(?:es)?|"
    r"hurricanes?|cyclones?|typhoons?|monsoons?|heatwaves?)\b"
    r"|\b(?:cloud\s+cover|wind\s+chill|heat\s+index|air\s+quality|"
    r"dew\s*point|uv\s+index|barometric\s+pressure|visibility)\b"
    r"|\b(?:it(?:['’]s| is| will be)|conditions?\s+(?:are|will be)|"
    r"day\s+(?:is|will be))\s+"
    r"(?:clear|sunny|cloudy|rainy|snowy|icy|stormy|foggy)\b"
    r"|\b(?:clear|sunny|cloudy|rainy|snowy|icy|stormy|foggy)\s+"
    r"(?:skies|conditions?|forecast|weather|day)\b"
    r"|\b(?:breezy|hot|cold|warm|cool|chilly|muggy|humid|dry|wet)\s+"
    r"(?:day|afternoon|morning|evening|night|week|weather|conditions?)\b"
    r"|\bexpect(?:ed)?\s+(?:clear|sunny|cloudy|rain|showers?|snow|ice|"
    r"storm|fog|sun|sunshine)\b"
    r"|\bit(?:['’]ll| will| won['’]t| will not| should| may| might| could)\s+"
    r"(?:(?:be|stay|remain|turn|get)\s+)?"
    r"(?:clear|sunny|cloudy|rain|rainy|snow|snowy|ice|icy|"
    r"storm|stormy|fog|foggy|windy|breezy|hot|cold|humid|dry|wet)\b"
    r"|\bthere\s+(?:will|won['’]t|will not|should|may|might|could)\s+"
    r"(?:not\s+)?(?:be\s+)?(?:no\s+)?(?:rain|snow|ice|storm|fog)\b"
    r"|\b(?:rain|snow|ice|storm|fog)\s+"
    r"(?:will|won['’]t|will not|should|may|might|could)\s+"
    r"(?:not\s+)?(?:hold\s+off|continue|start|stop|clear|arrive|be)\b"
    r"|\b(?:it\s+)?looks?\s+"
    r"(?:clear|sunny|cloudy|rainy|snowy|icy|stormy|foggy|windy|"
    r"breezy|hot|cold|humid|dry)\b"
    r"|\b(?:plan|prepare|dress)\s+for\s+"
    r"(?:sun|sunshine|rain|showers?|snow|ice|storms?|fog|wind|heat|cold)\b"
    r"|\b(?:rain|snow|ice|storm|fog)\s+(?:is|are|will be)\b"
    r"|\b(?:high|low)\s+(?:of\s+)?[-+]?[0-9]+(?:\.[0-9]+)?\b"
    r"|[-+]?[0-9]+(?:\.[0-9]+)?"
    r"(?:\s*°\s*[FfCc]|\s+[FfCc]\b|\s*%)"
    r"|\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b"
    r"|\b[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"[0-9]{1,2}(?:st|nd|rd|th)?\b"
    r")",
    flags=re.IGNORECASE | re.ASCII,
)
_SAVED_AREA_CONFIRMATION_RE = re.compile(
    r"\b(?:my|the)\s+(?:usual|home)\s+area\b",
    flags=re.IGNORECASE | re.ASCII,
)
_SAVED_AREA_QUESTION_RE = re.compile(
    r"\b(?:your|the)\s+(?:usual|home)\s+area\b",
    flags=re.IGNORECASE | re.ASCII,
)
_SAVED_AREA_OVERRIDE_RE = re.compile(
    r"\b(?:not|no\s+longer|elsewhere|somewhere\s+else|outside|away|instead|"
    r"actually|moved|changed|different|maybe|might|could|unsure|unknown|"
    r"don't|do\s+not|if)\b",
    flags=re.IGNORECASE | re.ASCII,
)
_SAVED_AREA_MODAL_UNCERTAINTY_RE = re.compile(
    r"\bmay(?:\s+(?:still|possibly|perhaps|well))?\s+"
    r"(?:be|use|take|happen|occur)\b",
    flags=re.IGNORECASE | re.ASCII,
)
_RECENT_CONVERSATION_TURN_RE = re.compile(
    r"(?m)^\[turn [0-9]+\]\nUser: ([^\n]*)\nAssistant: ([^\n]*)$",
    flags=re.ASCII,
)
_SAVED_AREA_BARE_CONFIRMATIONS = frozenset(
    {
        "yes",
        "yes please",
        "yes that's right",
        "that's right",
        "that is right",
        "correct",
        "here",
        "here please",
        "home area",
        "my home area",
        "my usual area",
        "the home area",
        "the usual area",
        "usual area",
    }
)
_DATE_ONLY_CONTEXT_WORDS = frozenset(
    {
        "a",
        "and",
        "at",
        "aware",
        "before",
        "ceremony",
        "date",
        "dates",
        "day",
        "direction",
        "event",
        "exact",
        "for",
        "forecast",
        "from",
        "give",
        "help",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "next",
        "of",
        "occasion",
        "on",
        "outfit",
        "plan",
        "please",
        "reception",
        "same",
        "shop",
        "shopping",
        "still",
        "style",
        "styling",
        "the",
        "this",
        "through",
        "thru",
        "to",
        "today",
        "tomorrow",
        "until",
        "update",
        "use",
        "will",
        "be",
        "we",
        "weather",
        "wedding",
        "week",
        "weekend",
        "s",
        "st",
        "nd",
        "rd",
        "th",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)
_SAVED_AREA_CONFIRMATION_CONTEXT_WORDS = (
    _DATE_ONLY_CONTEXT_WORDS
    | {
        "area",
        "around",
        "confirm",
        "confirmed",
        "home",
        "right",
        "that",
        "usual",
        "yes",
    }
)
_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE = 24
_MAX_DIAGNOSTIC_PRODUCT_FACTS = 40
_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS = 500
_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE_CHARS = 32_000
_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS = 1.0
_SEARCH_SCOPE_COMPLETE_NOTE = (
    "SEARCH_SCOPE_COMPLETE: The shopper's current request can now be answered "
    "from this search and existing turn evidence. Answer now. Do not search an "
    "adjacent category or substitute merely because search budget remains. Use "
    "the direct antecedent from recent discussion as the styling anchor; an item "
    "does not need to be in the cart to receive styling advice."
)
_SEARCH_BUDGET_EXHAUSTED_NOTE = (
    f"{SEARCH_BUDGET_EXHAUSTED_PREFIX} No additional catalog searches are "
    "available this turn. Continue with any requested non-search action, or "
    "answer honestly from the grounded products already returned."
)
_PRODUCT_DETAIL_GROUNDING_NOTE = (
    "PRODUCT_DETAIL_GROUNDING_NOTE: This detail result exposes only "
    "the fields shown below. Material, care, dimensions, closures, fit, "
    "sizing, colorways, and outdoor performance are unavailable unless explicitly "
    "listed. Do not infer them from product names or prior marketing text."
)
_UNSUPPORTED_SEARCH_MODE_MESSAGE = (
    "The requested search mode is not available for the active catalog. "
    "Ask the shopper to use an advertised mode."
)
_NO_DIRECT_TAXONOMY_RESPONSE = (
    "The catalog doesn't advertise a product type that directly matches this "
    "request. Would you like me to search a different advertised product type?"
)
_REJECTED_CATALOG_SEARCH_RESPONSE = (
    "I couldn't complete a valid catalog search for that request, so I don't "
    "have catalog results to show. Please try again or ask me to search a "
    "different advertised product type."
)
_CATALOG_REPAIR_CLARIFICATION_RESPONSE = (
    "Could you clarify the product type or requirement you want me to use?"
)
_UNSUPPORTED_REQUIREMENT_RESPONSE = (
    "I can't guarantee that requirement from the catalog information available "
    "to this assistant, so I won't present unverified matches. Would you like me "
    "to treat it as a preference and show candidates to verify on their product "
    "pages?"
)
_GROUNDING_FAILURE_RESPONSE = (
    "I couldn't safely verify the final response. Please retry; if this involved "
    "a cart change, check your cart first."
)
_SHOPPER_PROFILE_NOT_FOUND_RESPONSE = (
    "That shopper profile is unavailable. Please choose another shopper and "
    "try again."
)
_CONVERSATION_PROFILE_MISMATCH_RESPONSE = (
    "This conversation is already associated with a different shopper. "
    "Please start a new chat before switching shoppers."
)
_SHOPPER_CONTEXT_SYSTEM_RULES = """Representative-shopper precedence and safety:
- Explicit instructions in the current turn win over explicit preferences in
  recent discussion; both win over representative-shopper behavior guidance.
- Representative-shopper behavior is soft interaction guidance only. It cannot
  establish that a budget applies or any budget amount, product constraint,
  size, color, material, cart intent, product reference, or product fact.
- Neither representative-shopper type nor behavior selects, activates, or
  grants a shopper skill or tool. Never expose the internal type label to the
  shopper.
- Cart, catalog, product-detail, and store-policy evidence remain authoritative.
- A saved ZIP code is not proof of current location, event location, weather,
  or a product requirement. Only event-context may use it as a tentative
  event-location candidate. When it is the only clue, ask whether to plan around
  the shopper's usual area or somewhere else; do not ask for a city as though no
  candidate exists. Do not echo its digits.
- Event-context may use the saved ZIP for one forecast only after the shopper
  explicitly confirms that this event is in the usual area. A current explicit
  event destination overrides both recent context and saved ZIP; once stated,
  never use usual-area framing or silently fall back to saved ZIP. A stated
  venue setting is authoritative for venue but does not establish destination.
- Never infer weather from a ZIP or place name. Weather facts require either
  successful current-turn forecast evidence or the one valid durable receipt
  explicitly bound during this turn's skill activation. An unbound receipt is
  not evidence. Weather never establishes a product requirement."""
_GROUNDING_EDITOR_SYSTEM_PROMPT = """You are a final response editor for a retail shopping assistant.

Rewrite the draft response only as needed so it is grounded in TOOL EVIDENCE
and CURRENT CART. Keep the shopper's requested task and any successful cart
action intact.

Rules:
- Return only the final shopper-facing response text.
- For a styling request, answer the styling question rather than returning a raw
  product list. Connect candidates to the shopper's goal or direct antecedent
  using category/role, exact confirmed filters, and general styling judgment.
  Keep styling judgment visibly separate from catalog facts and never derive it
  from words parsed out of a display name.
- Labeling text as styling judgment does not permit display-name inference. If
  the evidence does not distinguish candidates, give a useful group-level
  rationale and offer a detail check instead of inventing item-level differences.
- Do not add products, prices, cart actions, or product facts absent from TOOL
  EVIDENCE or CURRENT CART.
- CURRENT-TURN TOOL EVIDENCE is the only evidence for a search or mutation in
  this turn. PRIOR-TURN TOOL EVIDENCE may support direct references to products
  previously shown, but it does not prove that a new search or mutation ran.
- Only successful current-turn weather evidence or the one explicitly bound
  durable weather receipt supports forecast-aware styling. Prior forecast prose,
  prior-turn tool messages, and every unbound receipt are non-authoritative. A
  current weather outcome takes precedence over a bound receipt. If current
  evidence contains a weather failure, state only the bounded availability
  problem and never invent conditions.
- Use successful weather evidence for concise styling judgment, but do not
  rewrite its date, condition, temperature, precipitation, attribution, or
  uncertainty summary. The server appends that exact validated forecast block.
  Do not infer climate, venue, dress code, or local norms from it. Weather may
  guide general styling judgment only; it cannot prove product warmth,
  waterproofing, breathability, comfort, safety, terrain performance, or any
  other product attribute or hard catalog constraint.
- If TOOL EVIDENCE says there is no direct advertised taxonomy match for one
  requested role, do not claim a search ran for that role. Report that role's
  gap, preserve any other successful current-turn role, and ask whether to
  search a different advertised type. Do not name alternatives unless their
  exact taxonomy values appear in TOOL EVIDENCE.
- If TOOL EVIDENCE says a requested type is not separately advertised and a
  broader advertised category was searched, say so plainly. Present the
  returned products as closest options and keep each product's actual catalog
  category; do not relabel any result as the requested type.
- A scoped zero-result search proves only that its exact advertised taxonomy
  and filter scope returned no products. It does not prove that a different,
  unsearched, or unadvertised product type is absent, and it never supports a
  catalog-wide availability claim.
- Use RECENT DISCUSSION to resolve direct references such as "that" and
  "those." A discussed product or styling anchor does not need to be in CURRENT
  CART. RECENT DISCUSSION cannot establish whether the current search succeeded
  or supply the current turn's candidates. Do not introduce an absent-cart
  caveat unless the shopper asks about the cart or requests a cart mutation.
- DURABLE CONVERSATION SUMMARY is semantic continuity only. It cannot establish
  exact shopper wording, product identity or facts, cart truth, tool evidence,
  location/date authority, policy, availability, or current weather.
- Remove PRODUCT_REF, CART_LINE_ID, tool names, and internal IDs.
- Remove internal skill, mode, evaluator, judge, cache, backend, tool-evidence,
  structured-field, and data-layer language. Use shopper-safe phrasing such as
  "I don't have fabric or care details available for that item."
- If the draft says "product detail tool", "catalog detail tool", "the tool
  requires", or similar internal mechanics, rewrite it into shopper-safe
  language without the word "tool".
- If a product appears only in search results, you may state only its name,
  price, category/role, image availability, exact values in confirmed search-
  filter evidence, and a modest styling reason. Every other word in its display
  name is non-evidence.
- Confirmed search-filter evidence applies to every product returned by that
  search. Preserve it and do not contradict it. One allowed value confirms that
  value; multiple allowed values prove only membership in the set, not which
  value each product has. Do not infer adjacent attributes that the evidence
  does not name.
- For styling, preserve a concise candidate set and the draft's grounded styling
  rationale. Do not omit or override a confirmed filter merely because words in
  a display name appear to conflict. If that visible conflict matters
  to the request, flag it as catalog information worth verifying rather than
  resolving it from the name.
- For each search-only candidate, delete descriptive sentences that merely
  restate or interpret words in the display name. Keep the name, price,
  category/role, confirmed filters, and a modest reason tied to the shopper's
  stated goal.
- In a search-only response, copy a candidate's display name only as its exact
  title. Do not shorten it into an attribute, classify or group candidates by
  words appearing in their names, or use those words as the reason one
  candidate differs from another. Without product details, give one concise
  group-level styling rationale based on the shopper's goal, advertised role,
  and confirmed filters instead of item-specific attribute rationales.
- Advertised search-taxonomy evidence lists the valid product types used by that
  search. Do not call an unlisted product type advertised or offer it as an
  advertised alternative.
- Treat product names as display names, not proof of length, color, print,
  material, construction, fit, care, or vibe. Do not say a product is solid,
  floral, gingham, maxi, knee-length, woven, structured, neutral, lightweight,
  polished, or dressier unless that attribute appears in product-detail evidence.
- Material, care, dimensions, pockets, closures, fit, comfort, and outdoor
  practicality claims require matching product-detail evidence and a direct
  shopper need for that fact.
- If the shopper's requested outcome depends on a material, fit, comfort,
  durability, care, weather, or other functional property that TOOL EVIDENCE
  does not confirm, say that property is not confirmed. Frame the candidates as
  the closest catalog or styling direction, not as complete, suitable, ready,
  or proven for that outcome. Keep any missing functional element explicit
  without inventing a product.
- Group claims such as "all are maxi length", "both are cotton", "the lightest",
  "most polished", or "best for heat" require product-detail evidence for every
  item included in that claim. Remove the claim if any item lacks that support.
- Do not say an item is stable on grass or gravel, water-resistant, bug-safe,
  all-day comfortable, maximally breathable, or best-in-category unless the
  evidence explicitly says that exact claim.
- Do not convert indirect evidence into outdoor surface performance. If the
  evidence says flat sole, ankle strap, linen, cotton, or elastic waistband,
  state only that fact when needed; do not add grass, gravel, outdoor-surface,
  heat, or all-evening performance claims. Avoid phrases such as "works well
  for outdoor surfaces"; use "a flat shoe option" or "fits the practical
  direction" instead.
- The shopper and Judge see only the final answer, not hidden tool output. Avoid
  long product-spec dumps; keep catalog facts item-specific and visibly modest.
- If image evidence is available, do not say images are unavailable or that you
  cannot show them. Say the product image should appear with the result, or
  simply answer the comparison.
- Preserve exact cart totals and cart contents when they are present in CURRENT
  CART or tool evidence.
- If the draft is already compliant, return it unchanged.
"""
_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {"write_todos", "ls", "write_file", "edit_file", "glob", "grep", "execute"}
)
_PRODUCT_NAME_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to", "with"})
_INTERNAL_SHOPPER_REPLACEMENTS = (
    ("The product detail tool doesn't return", "I don't have"),
    ("the product detail tool doesn't return", "I don't have"),
    ("The catalog detail tool doesn't return", "I don't have"),
    ("the catalog detail tool doesn't return", "I don't have"),
    ("The product detail tool does not return", "I don't have"),
    ("the product detail tool does not return", "I don't have"),
    ("The catalog detail tool does not return", "I don't have"),
    ("the catalog detail tool does not return", "I don't have"),
    ("the product detail tool", "the product details I can access"),
    ("the catalog detail tool", "the product details I can access"),
    ("because the tool requires", "because I need"),
    ("The tool requires", "I need"),
    ("the tool requires", "I need"),
)


class CatalogTaxonomyToolInput(BaseModel):
    """Catalog-derived taxonomy roles used by the agent search tool."""

    model_config = ConfigDict(extra="forbid")

    category: list[str] = Field(
        ...,
        description=(
            "Exact advertised category values required by the shopper. Use an "
            "empty list only when subcategory supplies the text-search scope or "
            "the search is image-only."
        ),
    )
    subcategory: list[str] = Field(
        ...,
        description=(
            "Exact advertised subcategory values required by the shopper. Use an "
            "empty list when category supplies the text-search scope or the search "
            "is image-only."
        ),
    )

    @field_validator("category", "subcategory", mode="before")
    @classmethod
    def deduplicate_values(cls, value: Any) -> Any:
        """Normalize repeated taxonomy values before cardinality validation."""

        if isinstance(value, str):
            return [value]
        return list(dict.fromkeys(value)) if isinstance(value, list) else value


def _singularize_product_word(word: str) -> str:
    """Conservatively singularize one normalized product-type word."""

    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


def _normalize_product_text(value: str) -> str:
    """Normalize product-type text for deterministic lexical comparison."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("&", " and ").replace("/", " or ")
    words = re.findall(r"[^\W_]+", normalized.replace("_", " "))
    return " ".join(_singularize_product_word(word) for word in words)


def _has_alternative_connector(value: str) -> bool:
    """Return whether a product phrase joins explicit alternatives."""

    normalized = _normalize_product_text(value)
    return bool(re.search(r"\b(?:and|or)\b", normalized))


def _text_mentions_product_type(text: str, product_type: str) -> bool:
    """Return whether text names a normalized product-type form."""

    normalized_text = _normalize_product_text(text)
    padded_text = f" {normalized_text} "
    normalized_product_type = _normalize_product_text(product_type)
    return bool(normalized_product_type) and (
        f" {normalized_product_type} " in padded_text
    )


def _requirement_word_stem(word: str) -> str:
    """Return a conservative stem for literal requirement provenance."""

    for suffix in ("ance", "ence", "ancy", "ency", "ant", "ent", "ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[: -len(suffix)]
    return _singularize_product_word(word)


def _shopper_stated_requirement(query: str, requirement: str) -> bool:
    """Return whether a proposed requirement is grounded in the current turn."""

    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    query_words = {
        _requirement_word_stem(word)
        for word in re.findall(r"[^\W_]+", normalized_query)
    }
    requirement_words = {
        _requirement_word_stem(word)
        for word in re.findall(
            r"[^\W_]+",
            unicodedata.normalize("NFKC", requirement).casefold(),
        )
    }
    return bool(requirement_words) and requirement_words.issubset(query_words)


def _product_scope_key(value: str | None) -> str:
    """Return the full normalized product phrase preserved across repair."""

    return _normalize_product_text(value or "")


def _generic_shopper_guidance(requested_product_type: str | None) -> str:
    """Return safe guidance after an inferred attribute is removed."""

    product_type = (requested_product_type or "products").replace("_", " ")
    return f"Finding {product_type} for the shopper's request."


_UNSUPPORTED_GUIDANCE_PATTERN = re.compile(
    r"\b(?:waterproof|water[ -]?resistant|weather[ -]?safe|bug[ -]?safe|"
    r"wet (?:surfaces?|grounds?|conditions?)|grass|gravel|all[ -]?day|best[ -]?in[ -]?category|"
    r"maximally|outdoor (?:surfaces?|walking)|"
    r"(?:handles?|suitable for|works? well (?:for|in)|secure for|stay secure for)"
    r"[^.]{0,32}(?:rain|wet weather|outdoor))\b",
    flags=re.IGNORECASE,
)

_CONTEXT_ONLY_EVENT_ADJUSTMENT_PHRASES = {
    "streamlined_accessories": "keep accessories streamlined",
    "lower_profile_footwear": "favor lower-profile footwear",
    "polished_unfussy_finish": (
        "keep the overall finish polished and unfussy"
    ),
    "adaptable_finishing": (
        "keep the finishing details simple and adaptable"
    ),
}


def _safe_shopper_guidance(
    shopper_guidance: str,
    requested_product_type: str | None,
) -> str:
    """Remove unsupported performance language from pre-search guidance."""

    if _UNSUPPORTED_GUIDANCE_PATTERN.search(shopper_guidance):
        return _generic_shopper_guidance(requested_product_type)
    return shopper_guidance


def _unsupported_requirement_message(requirements: list[str]) -> str:
    """Return the shopper-facing failure for unsupported hard requirements."""

    return (
        "The requested catalog requirement cannot be enforced: "
        + ", ".join(
            f"'{requirement}' is not an advertised hard filter"
            for requirement in requirements
        )
        + ". Ask the shopper whether to treat it as a preference."
    )


def _recent_shopper_statements(context: str, *, limit: int = 4) -> str:
    """Extract prior shopper text without exposing assistant responses."""

    statements = re.findall(
        r"(?:^|\n)User:\s*(.*?)(?=\nAssistant:|\nUser:|\Z)",
        context or "",
        flags=re.DOTALL,
    )
    compact = [" ".join(statement.split()) for statement in statements]
    return "\n".join(statement for statement in compact[-limit:] if statement)


def _shopper_stated_product_scope(
    query: str,
    context: str,
    product_scope_key: str,
) -> bool:
    """Return whether current or recent shopper text states a product scope."""

    shopper_text = "\n".join(
        value for value in (query, _recent_shopper_statements(context)) if value
    )
    return _text_mentions_product_type(shopper_text, product_scope_key)


def _resolved_agent_selected_product_type(
    *,
    query: str,
    context: str,
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Derive open-role provenance from the agent's single taxonomy choice."""

    if taxonomy_status != "agent_selected_type":
        return requested_product_type
    scope_key = _product_scope_key(requested_product_type)
    if scope_key and _shopper_stated_product_scope(query, context, scope_key):
        return requested_product_type
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    subcategories = payload.get("subcategory") or []
    if len(subcategories) == 1:
        return str(subcategories[0])
    return requested_product_type


def _agent_selected_scope_is_advertised(
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
) -> bool:
    """Check that an open-role choice names its advertised taxonomy scope."""

    requested = _normalize_product_text(requested_product_type or "")
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    subcategories = {
        _normalize_product_text(value)
        for value in (payload.get("subcategory") or [])
    }
    return len(subcategories) == 1 and requested in subcategories


def _advertised_subcategories_for_selection(
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> list[str]:
    """Return advertised role choices within the selected category."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected_categories = payload.get("category") or []
    categories = capabilities.taxonomy.categories
    category_names = selected_categories or list(categories)
    return sorted(
        {
            subcategory
            for category_name in category_names
            if (category := categories.get(category_name)) is not None
            for subcategory in category.subcategories
        }
    )


def _advertised_taxonomy_value(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> str | None:
    """Return the matching advertised taxonomy value, if one exists."""

    requested = _normalize_product_text(requested_product_type or "")
    for category_name, category in capabilities.taxonomy.categories.items():
        if requested == _normalize_product_text(category_name):
            return category_name
        for subcategory_name in category.subcategories:
            if requested == _normalize_product_text(subcategory_name):
                return subcategory_name
    return None


def _advertised_scope_match(
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> tuple[str, str, str, str] | None:
    """Return the longest advertised exact or suffix scope match."""

    raw_requested = requested_product_type or ""
    requested = _normalize_product_text(raw_requested)
    suffix_match_allowed = not _has_alternative_connector(raw_requested)
    matches: list[tuple[str, str, str, str]] = []
    for category_name, category in capabilities.taxonomy.categories.items():
        normalized_category = _normalize_product_text(category_name)
        if requested == normalized_category or (
            suffix_match_allowed
            and requested.endswith(f" {normalized_category}")
        ):
            matches.append(
                ("category", category_name, category_name, normalized_category)
            )
        for subcategory_name in category.subcategories:
            normalized_subcategory = _normalize_product_text(subcategory_name)
            if requested == normalized_subcategory or (
                suffix_match_allowed
                and requested.endswith(f" {normalized_subcategory}")
            ):
                matches.append(
                    (
                        "subcategory",
                        subcategory_name,
                        category_name,
                        normalized_subcategory,
                    )
                )
    return max(
        matches,
        key=lambda match: (len(match[3].split()), len(match[3])),
        default=None,
    )


def _duplicates_unavailable_product_type(
    requirements: Any,
    requested_product_type: str | None,
    capabilities: CatalogCapabilities,
) -> bool:
    """Return whether requirements only repeat an unavailable product type."""

    requested = _normalize_product_text(requested_product_type or "")
    return bool(
        requested
        and isinstance(requirements, list)
        and len(requirements) == 1
        and not _has_alternative_connector(requested_product_type or "")
        and _advertised_scope_match(requested_product_type, capabilities) is None
        and all(
            isinstance(requirement, str)
            and _normalize_product_text(requirement) == requested
            for requirement in requirements
        )
    )


def _same_product_scope(
    first: str,
    second: str,
    capabilities: CatalogCapabilities,
) -> bool:
    """Compare full product scopes without conflating advertised siblings."""

    if first == second:
        return True
    if _has_alternative_connector(first):
        return False
    first_advertised = _advertised_taxonomy_value(first, capabilities)
    if first_advertised:
        return False
    advertised_match = _advertised_scope_match(first, capabilities)
    if advertised_match:
        return second == advertised_match[3]
    return first.endswith(f" {second}")


def _selected_advertised_subcategories(
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[str, list[str]] | None:
    """Return one typed multi-subcategory selection owned by one category."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected = list(dict.fromkeys(payload.get("subcategory") or []))
    if len(selected) < 2:
        return None
    owners = [
        category_name
        for category_name, category in capabilities.taxonomy.categories.items()
        if all(value in category.subcategories for value in selected)
    ]
    selected_categories = set(payload.get("category") or [])
    if len(owners) != 1 or selected_categories not in (set(), {owners[0]}):
        return None
    return owners[0], selected


def _multi_subcategory_candidate_limit(
    selection: tuple[str, list[str]] | None,
    capabilities: CatalogCapabilities,
    default: int,
) -> int:
    """Fetch enough ranked candidates for a typed multi-subcategory selection."""

    if selection is None:
        return default
    category_name, subcategories = selection
    category = capabilities.taxonomy.categories[category_name]
    product_count = sum(
        category.subcategories[name].product_count for name in subcategories
    )
    return min(50, max(default, len(subcategories), product_count))


def _products_with_subcategory_coverage(
    products: list[ProductSummary],
    selection: tuple[str, list[str]] | None,
    limit: int,
) -> list[ProductSummary]:
    """Keep rank order while reserving one result per selected subcategory."""

    if selection is None or len(products) <= limit:
        return products
    _, subcategories = selection
    selected_indexes: set[int] = set()
    for subcategory in subcategories:
        normalized_subcategory = _normalize_product_text(subcategory)
        match = next(
            (
                index
                for index, product in enumerate(products)
                if _normalize_product_text(product.category or "")
                == normalized_subcategory
            ),
            None,
        )
        if match is not None:
            selected_indexes.add(match)

    for index in range(len(products)):
        if len(selected_indexes) >= max(limit, len(subcategories)):
            break
        selected_indexes.add(index)
    return [products[index] for index in sorted(selected_indexes)]


def _advertised_taxonomy_scope_issue(
    requested_product_type: str | None,
    taxonomy_status: str,
    taxonomy: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> str | None:
    """Enforce capability-owned relations for exact advertised scopes."""

    advertised_match = _advertised_scope_match(
        requested_product_type,
        capabilities,
    )
    if advertised_match is None:
        return None
    scope_kind, advertised_name, category_name, _ = advertised_match
    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    selected_categories = payload.get("category") or []
    selected_subcategories = payload.get("subcategory") or []
    normalized_category = _normalize_product_text(category_name)
    normalized_selected_categories = {
        _normalize_product_text(value) for value in selected_categories
    }
    normalized_selected_subcategories = {
        _normalize_product_text(value) for value in selected_subcategories
    }
    if scope_kind == "category":
        category = capabilities.taxonomy.categories[category_name]
        owned_subcategories = {
            _normalize_product_text(value) for value in category.subcategories
        }
        selected_owned_children = (
            bool(normalized_selected_subcategories)
            and normalized_selected_subcategories.issubset(owned_subcategories)
            and normalized_selected_categories in (set(), {normalized_category})
        )
        if taxonomy_status == "exact_requested_type" and selected_owned_children:
            return (
                f"Requested product type '{requested_product_type}' binds to "
                f"advertised category '{advertised_name}', while the selected "
                "taxonomy contains only its advertised children. Keep those "
                "children together for the shopper's umbrella request."
            )
        exact_category = (
            taxonomy_status == "exact_requested_type"
            and not normalized_selected_subcategories
            and normalized_selected_categories == {normalized_category}
        )
        owned_children = (
            taxonomy_status
            in {"member_of_requested_umbrella", "agent_selected_type"}
            and selected_owned_children
        )
        if exact_category or owned_children:
            return None
        return (
            f"Requested product type '{requested_product_type}' binds to advertised "
            f"category '{advertised_name}'. Select that category directly or only "
            "its advertised children; do not substitute another category."
        )
    selected_matches = (
        len(selected_subcategories) == 1
        and _normalize_product_text(selected_subcategories[0])
        == _normalize_product_text(advertised_name)
        and normalized_selected_categories in (set(), {normalized_category})
    )
    if selected_matches and taxonomy_status in {
        "exact_requested_type",
        "agent_selected_type",
    }:
        return None
    return (
        f"Requested product type '{requested_product_type}' binds to advertised "
        f"subcategory '{advertised_name}'. Select only that exact subcategory; "
        "do not substitute an advertised sibling."
    )


def _catalog_execution_taxonomy_status(
    requested_product_type: str | None,
    taxonomy: BaseModel | dict[str, Any],
    semantic_query: str,
    capabilities: CatalogCapabilities,
    *,
    shopper_stated_scope: bool,
) -> str:
    """Derive the legacy execution mode without asking the model to label it."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    categories = payload.get("category") or []
    subcategories = payload.get("subcategory") or []
    if (
        not semantic_query.strip()
        and not requested_product_type
        and not categories
        and not subcategories
    ):
        return "image_only"
    if not shopper_stated_scope:
        return "agent_selected_type"

    advertised_match = _advertised_scope_match(
        requested_product_type,
        capabilities,
    )
    if advertised_match is not None:
        scope_kind = advertised_match[0]
        if scope_kind == "category" and subcategories:
            return "member_of_requested_umbrella"
        return "exact_requested_type"
    if len(categories) == 1 and not subcategories:
        return "parent_category_alternative"
    if subcategories:
        return "member_of_requested_umbrella"
    return "exact_requested_type"


class SearchCatalogToolArguments(BaseModel):
    """Stable agent-facing catalog-search arguments."""

    model_config = ConfigDict(extra="forbid")

    semantic_query: str = Field(
        ...,
        description=(
            "One soft or descriptive product search string for semantic ranking. "
            "Product type may be repeated for relevance, but taxonomy and other "
            "must-haves are enforced only by the structured fields below. Use an "
            "empty string only for an image-only search."
        ),
    )
    shopper_guidance: str = Field(
        ...,
        max_length=400,
        description=(
            "One concise, product-agnostic shopper-facing sentence written under "
            "the active skill before search results are known. Connect this "
            "product role to the shopper's stated goal or direct antecedent. Do "
            "not name unselected or unavailable product types, name or describe "
            "candidate products, assert product attributes, or mention tools, "
            "schemas, filters, evidence, or identifiers. Use an empty string only "
            "for image-only search."
        ),
    )
    requested_product_type: str | None = Field(
        ...,
        description=(
            "Shortest product noun or umbrella phrase for this focused role, "
            "resolved from the shopper's current turn or direct antecedent. "
            "Exclude color, material, fit, occasion, weather, and style modifiers: "
            "'formal tops' and 'relaxed-fit tops' both use 'tops'. For a genuinely "
            "open role selected by the agent, use the chosen advertised role noun. "
            "If this type is not separately advertised and you select one faithful "
            "advertised parent category, keep this shopper-named type unchanged. "
            "This is provenance, not catalog taxonomy or a ranking query. Use null "
            "only for image-only search."
        ),
    )
    taxonomy: CatalogTaxonomyToolInput = Field(
        ...,
        description=(
            "Required catalog-derived taxonomy selection. Allowed category and "
            "subcategory values come from the active catalog capabilities. Every "
            "selected value must be the requested product type or a child of an "
            "umbrella the shopper actually named. For a genuinely open request, "
            "select one advertised subcategory as the focused role. Never select "
            "a parent or sibling as a substitute for an advertised type. If a "
            "shopper-named type is not separately advertised but one advertised "
            "category is its faithful broader parent, select only that category "
            "and leave subcategory empty; results remain alternatives under their "
            "actual catalog types. For example, skirts may satisfy bottoms; "
            "dresses may not."
        ),
    )
    required_constraints: dict[str, Any] = Field(
        ...,
        description=(
            "Every non-taxonomy must-have stated for the target products in the "
            "current shopper turn, as a structured field and value. Facts about "
            "an antecedent or anchor guide semantic styling judgment; do not copy "
            "them onto a complementary product unless the shopper explicitly asks "
            "for the same value. "
            "Use exact hard-filter properties and advertised enum values from "
            "current Catalog capabilities. Put a directly stated requirement "
            "those capabilities do not advertise in unadvertised_requirements. "
            "Season, weather, occasion, "
            "and subjective style/vibe "
            "context remain semantic direction unless the shopper directly "
            "requires an objective product attribute. A defining material is a "
            "must-have: for 'Any denim skirts available?', put 'denim' in "
            "unadvertised_requirements when composition is not a hard filter. "
            "Product types never belong in unadvertised_requirements."
        ),
    )
    scope_complete: bool = Field(
        ...,
        description=(
            "True only when this search plus existing turn evidence is enough to "
            "answer the shopper's complete current request. For a recommendation-"
            "only one-role request this is true. Set false when an explicitly "
            "requested product role, product-detail verification, availability "
            "check, or cart action still must run after this search. Do not set "
            "false merely to search alternatives or adjacent types, or because a "
            "broader multi-turn outfit project remains unfinished. 'Start with a "
            "beige top' and 'What bottoms go with that?' are each complete after "
            "their own one-role search."
        ),
    )
    search_mode: str | None = Field(
        default=None,
        description="Optional search mode from Catalog capabilities.",
    )


class SearchCatalogToolInput(SearchCatalogToolArguments):
    """Runtime-validated catalog search request."""

    taxonomy_status: Literal[
        "exact_requested_type",
        "member_of_requested_umbrella",
        "parent_category_alternative",
        "agent_selected_type",
        "no_direct_catalog_match",
        "image_only",
    ] = Field(..., description="Server-derived catalog execution mode.")

    @model_validator(mode="after")
    def text_search_has_taxonomy_scope(self) -> "SearchCatalogToolInput":
        if len(set(self.taxonomy.category)) > 1:
            raise ValueError("catalog search accepts at most one category")
        has_taxonomy = bool(
            self.taxonomy.category or self.taxonomy.subcategory
        )
        has_query = bool(self.semantic_query.strip())
        has_shopper_guidance = bool(self.shopper_guidance.strip())
        requested_product_type = (
            self.requested_product_type.strip()
            if isinstance(self.requested_product_type, str)
            else ""
        )
        if self.taxonomy_status in {
            "image_only",
            "no_direct_catalog_match",
        }:
            if has_shopper_guidance:
                raise ValueError(
                    "A non-text retrieval path requires empty shopper_guidance"
                )
        elif not has_shopper_guidance:
            raise ValueError(
                "catalog retrieval requires non-empty shopper_guidance"
            )
        if self.taxonomy_status != "image_only" and not requested_product_type:
            raise ValueError(
                "text catalog search requires requested_product_type"
            )
        if self.taxonomy_status == "image_only" and requested_product_type:
            raise ValueError(
                "image-only search requires requested_product_type=null"
            )
        if self.taxonomy_status == "no_direct_catalog_match":
            if has_taxonomy:
                raise ValueError(
                    "a non-retrieval result requires empty taxonomy arrays"
                )
            if not has_query:
                raise ValueError(
                    "a non-retrieval result requires a requested product type"
                )
            constraints = (
                self.required_constraints.model_dump(exclude_none=True)
                if isinstance(self.required_constraints, BaseModel)
                else self.required_constraints
            )
            has_constraint = any(
                value not in (None, "", [], {})
                for value in constraints.values()
            )
            if has_constraint:
                raise ValueError(
                    "a non-retrieval result cannot include required constraints"
                )
            return self
        if self.taxonomy_status == "image_only":
            if has_query or has_taxonomy:
                raise ValueError(
                    "image-only search requires an empty semantic query and taxonomy"
                )
            return self
        if self.taxonomy_status == "member_of_requested_umbrella" and not (
            self.taxonomy.subcategory
        ):
            raise ValueError(
                "an umbrella search requires an advertised subcategory"
            )
        if self.taxonomy_status == "agent_selected_type" and not (
            self.taxonomy.subcategory
        ):
            raise ValueError(
                "an open-role search requires an advertised subcategory"
            )
        if self.taxonomy_status == "parent_category_alternative" and not (
            len(self.taxonomy.category) == 1
            and not self.taxonomy.subcategory
        ):
            raise ValueError(
                "a parent-category alternative requires one advertised category "
                "and no subcategory"
            )
        if has_query and not has_taxonomy:
            raise ValueError(
                "text catalog search requires an advertised category or subcategory"
            )
        if not has_query:
            raise ValueError("text catalog search requires a semantic query")
        return self


def _exact_taxonomy_issue(
    requested_product_type: str,
    taxonomy: BaseModel | dict[str, Any],
) -> str | None:
    """Return a coherence issue for an exact taxonomy claim."""

    payload = taxonomy.model_dump() if isinstance(taxonomy, BaseModel) else taxonomy
    categories = payload.get("category") or []
    subcategories = payload.get("subcategory") or []
    requested_matches_category = not subcategories and any(
        _normalize_product_text(requested_product_type)
        == _normalize_product_text(category)
        for category in categories
    )
    if requested_matches_category:
        return None

    selected_product_types = subcategories or categories
    if len(selected_product_types) != 1:
        return (
            "The selected taxonomy must faithfully represent one requested type "
            "or every child of a shopper-requested umbrella."
        )
    selected_product_type = selected_product_types[0]
    if _normalize_product_text(requested_product_type) == _normalize_product_text(
        selected_product_type
    ):
        return None
    return (
        "A single taxonomy value must match requested_product_type. If no "
        "advertised value faithfully represents it, ask a clarification instead"
    )


class _CatalogNumberConstraint(BaseModel):
    """Inclusive numeric range accepted by catalog hard filters."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def has_a_bound(self) -> "_CatalogNumberConstraint":
        if self.min is None and self.max is None:
            raise ValueError("numeric constraint requires min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("numeric constraint min cannot exceed max")
        return self


def _search_catalog_tool_input_model(
    capabilities: CatalogCapabilities,
    *,
    validate_scope: bool = True,
) -> type[SearchCatalogToolArguments]:
    """Create one tool schema whose enums come from the active catalog."""

    taxonomy = capabilities.taxonomy
    advertised_search_modes = tuple(dict.fromkeys(capabilities.retrieval_modes))
    search_mode_type = (
        Literal.__getitem__(advertised_search_modes)
        if advertised_search_modes
        else str
    )
    category_values = (
        sorted(taxonomy.categories, key=str.casefold)
        if taxonomy.category_field
        else []
    )
    subcategory_values = (
        sorted(
            {
                name
                for category in taxonomy.categories.values()
                for name in category.subcategories
            },
            key=str.casefold,
        )
        if taxonomy.subcategory_field
        else []
    )

    category_type, category_field = _taxonomy_list_field(
        category_values,
        role="category",
        advertised_field=taxonomy.category_field,
    )
    subcategory_type, subcategory_field = _taxonomy_list_field(
        subcategory_values,
        role="subcategory",
        advertised_field=taxonomy.subcategory_field,
    )
    taxonomy_model = create_model(
        "CatalogTaxonomySelection",
        __base__=CatalogTaxonomyToolInput,
        category=(category_type, category_field),
        subcategory=(subcategory_type, subcategory_field),
    )
    required_constraints_model = _required_constraints_input_model(
        capabilities,
    )
    return create_model(
        "CatalogSearchInput" if validate_scope else "CatalogSearchToolArguments",
        __base__=(
            SearchCatalogToolInput
            if validate_scope
            else SearchCatalogToolArguments
        ),
        taxonomy=(
            taxonomy_model,
            Field(
                ...,
                description=(
                    "Required taxonomy selection generated from the active catalog. "
                    "Use exact enum values. A text search requires at least one "
                    "category or subcategory; both arrays may be empty only for "
                    "an image-only search."
                    " Every value must be a kind of the requested scope: skirts "
                    "may satisfy bottoms; dresses may not. For a broad request "
                    "that names no product type, choose one exact advertised "
                    "subcategory as the focused starting role. If a shopper-named "
                    "type is not separately advertised but one faithful broader "
                    "advertised parent category exists, select only that category "
                    "and leave subcategory empty."
                ),
            ),
        ),
        required_constraints=(
            required_constraints_model,
            Field(
                ...,
                description=(
                    "Catalog hard filters and any defining requirement the active "
                    "catalog cannot enforce. Apply constraints only when the "
                    "current turn states them for the target products; an anchor's "
                    "attributes belong in semantic styling context unless the "
                    "shopper explicitly requests the same value. Use only the "
                    "advertised properties "
                    "in this object; put unsupported must-haves in "
                    "unadvertised_requirements instead of weakening them. For "
                    "'Do you have water-resistant bags?', use "
                    "{'unadvertised_requirements': ['water resistance']}. Do not "
                    "put broad season, weather, occasion, or subjective style/vibe "
                    "context here unless the shopper directly requires a product "
                    "attribute. 'Rainy day outfit' does not require water "
                    "resistance; 'water-resistant bags' does. Recommendation "
                    "adjectives such as comfortable, relaxed, soft, breathable, "
                    "lightweight, casual, dressy, bold, bright, vibrant, or "
                    "sporty are always semantic ranking preferences, not "
                    "objective hard filters. Before calling the tool, compare "
                    "every target-product modifier with this advertised schema "
                    "and include every exact matching filter value. A product type "
                    "never belongs in unadvertised_requirements. Use an empty "
                    "object for image-only search."
                ),
            ),
        ),
        search_mode=(
            search_mode_type | None,
            Field(
                default=None,
                description="Optional search mode advertised by the active catalog.",
            ),
        ),
    )


def _taxonomy_list_field(
    values: list[str],
    *,
    role: str,
    advertised_field: str | None,
) -> tuple[Any, Any]:
    field_name = advertised_field or "not advertised"
    description = (
        f"Exact {role} values advertised through catalog field '{field_name}'. "
        "Use an empty list when the other taxonomy role supplies the text scope or "
        "the search is image-only."
    )
    if role == "category":
        description += " Select at most one category per catalog search."
    if not values:
        return list[str], Field(..., max_length=0, description=description)

    literal_type = Literal.__getitem__(tuple(values))
    max_length = 1 if role == "category" else None
    return list[literal_type], Field(
        ...,
        max_length=max_length,
        description=description,
    )


def _required_constraints_input_model(
    capabilities: CatalogCapabilities,
) -> type[BaseModel]:
    """Create a hard-constraint schema from advertised catalog filters."""

    taxonomy_fields = {
        field_name
        for field_name in (
            capabilities.taxonomy.category_field,
            capabilities.taxonomy.subcategory_field,
        )
        if field_name
    }
    fields: dict[str, tuple[Any, Any]] = {}
    for name, capability in sorted(effective_filter_capabilities(capabilities).items()):
        if name in taxonomy_fields:
            continue
        if capability.type in {"enum", "enum_list"} and capability.values:
            value_type = Literal.__getitem__(tuple(capability.values))
            field_type = value_type | list[value_type] | None
        elif capability.type == "number":
            field_type = _CatalogNumberConstraint | None
        else:
            field_type = str | list[str] | None
        fields[name] = (
            field_type,
            Field(
                default=None,
                description=f"Advertised hard filter '{name}'.",
            ),
        )
    fields["unadvertised_requirements"] = (
        list[str],
        Field(
            default_factory=list,
            description=(
                "Only objective product requirements directly stated in the "
                "current shopper turn and absent from this schema. Never infer a "
                "requirement from season, weather, occasion, or subjective "
                "style/vibe context: 'rainy day outfit' produces an empty list. "
                "Use ['water resistance'] for 'water-resistant bags' because that "
                "turn directly states the product requirement. Never omit or "
                "soften a directly stated objective requirement. Subjective "
                "recommendation adjectives such as comfortable, relaxed, soft, "
                "breathable, lightweight, casual, dressy, bold, bright, vibrant, "
                "or sporty always stay semantic; never put them in this field. "
                "Before calling the tool, include every exact advertised filter "
                "value that modifies the target product; do not leave this object "
                "empty when one applies. "
                "In an 'A or B' request, do not put the "
                "supported advertised branch here. A product type never belongs "
                "here."
            ),
        ),
    )
    return create_model(
        "CatalogRequiredConstraints",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class _ShopperSkillActivationInput(BaseModel):
    """Shared composition rules for dynamic shopper-skill activation."""

    model_config = ConfigDict(extra="forbid")

    skill_names: list[str]
    event_context_next_question: EventContextNextQuestion | None = None
    weather_receipt_id: str | None = None

    @model_validator(mode="after")
    def primary_procedures_are_exclusive(self) -> "_ShopperSkillActivationInput":
        selected = set(self.skill_names)
        primary = selected.intersection(
            {"outfit-styling", "product-discovery"}
        )
        if len(primary) > 1:
            raise PydanticCustomError(
                SKILL_ACTIVATION_MULTIPLE_PRIMARY,
                "select exactly one primary procedure: outfit-styling or "
                "product-discovery, never both",
            )
        if "budget-shopping" in selected and len(primary) != 1:
            raise PydanticCustomError(
                SKILL_ACTIVATION_MODIFIER_REQUIRES_PRIMARY,
                "budget-shopping requires exactly one primary procedure: "
                "outfit-styling or product-discovery",
            )
        if (
            "event-context" in selected
            and "outfit-styling" not in selected
        ):
            raise PydanticCustomError(
                SKILL_ACTIVATION_EVENT_CONTEXT_REQUIRES_STYLING,
                "event-context requires outfit-styling",
            )
        event_context_selected = "event-context" in selected
        if event_context_selected != (
            self.event_context_next_question is not None
        ):
            raise PydanticCustomError(
                SKILL_ACTIVATION_EVENT_CONTEXT_INVALID,
                "event_context_next_question is required exactly when "
                "event-context is selected",
            )
        if self.weather_receipt_id is not None and (
            not event_context_selected
            or self.event_context_next_question != "none"
        ):
            raise PydanticCustomError(
                SKILL_ACTIVATION_WEATHER_RECEIPT_INVALID,
                "weather receipt binding requires event-context with no "
                "event-context follow-up question",
            )
        return self


def _skill_activation_input_model(
    skill_names: tuple[str, ...],
    weather_receipt_ids: tuple[str, ...] = (),
    *,
    weather_date_available: bool = False,
) -> type[BaseModel]:
    """Create the semantic skill-selection schema from the active registry."""

    skill_name_type = Literal.__getitem__(skill_names)
    event_context_next_question_type: Any = (
        Literal["event_location", "event_venue", "none"]
        if weather_date_available
        else EventContextNextQuestion
    )
    event_context_next_question_description = (
        "Required exactly when event-context is selected; omit otherwise. "
        "Choose event_location only when the event destination is still "
        "missing and materially changes the next guidance. Choose event_venue "
        "only after destination is established, when venue/setting is still "
        "missing and material. Choose none otherwise. This is the only "
        "event-context follow-up question the response may ask. A bounded "
        "weather date is already established for this request, so asking for "
        "another date is not allowed."
        if weather_date_available
        else (
            "Required exactly when event-context is selected; omit otherwise. "
            "Choose event_location only when the event destination is still "
            "missing and materially changes the next guidance. Choose "
            "event_venue only after destination is established, when "
            "venue/setting is still missing and material. Choose event_date "
            "only after destination and any material venue are established, "
            "live weather is enabled and material, and no bounded date is "
            "established or explicitly unavailable. Choose none otherwise. "
            "This is the only event-context follow-up question the response "
            "may ask. An explicitly stated outdoor patio, beach, garden, "
            "rooftop, or open-air setting makes enabled live weather material: "
            "with the destination and setting established but no date, choose "
            "event_date, not none. Example: after Cancun alone choose "
            "event_venue when setting matters; after the shopper confirms a "
            "beach setting choose event_date when live weather is enabled and "
            "material."
        )
    )
    weather_receipt_id_type: Any = (
        Literal.__getitem__(weather_receipt_ids)
        if weather_receipt_ids
        else type(None)
    )
    activation_model = create_model(
        "ShopperSkillActivationInput",
        __base__=_ShopperSkillActivationInput,
        skill_names=(
            list[skill_name_type],
            Field(
                ...,
                min_length=1,
                max_length=len(skill_names),
                description=(
                    "Smallest set of registered shopper skills whose descriptions "
                    "cover the current turn's complete intent. For product search "
                    "or styling, select exactly one primary procedure: outfit-"
                    "styling or product-discovery, never both. Cart and policy "
                    "intents may select their standalone skill without a primary. "
                    "Select event-context only with outfit-styling, and select "
                    "it whenever an event destination or venue is stated or the "
                    "response would otherwise ask about or branch on missing "
                    "destination or venue context. Generic occasion advice is "
                    "not a reason to omit it."
                ),
            ),
        ),
        event_context_next_question=(
            event_context_next_question_type | None,
            Field(
                default=None,
                description=event_context_next_question_description,
            ),
        ),
        weather_receipt_id=(
            weather_receipt_id_type | None,
            Field(
                default=None,
                description=(
                    "Optional currently valid durable weather receipt for the "
                    "exact same event location and date scope. Select one only "
                    "with event-context and event_context_next_question=none. "
                    "Omit it after a current location/date correction or when "
                    "the shopper explicitly asks for a fresh forecast."
                ),
            ),
        ),
    )
    return activation_model


def _taxonomy_hard_constraints(
    selection: BaseModel | dict[str, Any],
    capabilities: CatalogCapabilities,
) -> tuple[dict[str, list[str]], list[str]]:
    """Map generic taxonomy roles to catalog-owned hard-filter field names."""

    raw = selection.model_dump() if isinstance(selection, BaseModel) else selection
    categories = sorted(set(raw.get("category", [])), key=str.casefold)
    subcategories = sorted(set(raw.get("subcategory", [])), key=str.casefold)
    taxonomy = capabilities.taxonomy
    issues: list[str] = []

    for category in categories:
        if category not in taxonomy.categories:
            issues.append(f"category '{category}' is not advertised")

    owners: dict[str, list[str]] = {
        subcategory: sorted(
            [
                category_name
                for category_name, category in taxonomy.categories.items()
                if subcategory in category.subcategories
            ],
            key=str.casefold,
        )
        for subcategory in subcategories
    }
    for subcategory, subcategory_owners in owners.items():
        if not subcategory_owners:
            issues.append(f"subcategory '{subcategory}' is not advertised")
        elif categories and not set(categories).intersection(subcategory_owners):
            issues.append(
                f"subcategory '{subcategory}' is not available in selected categories"
            )

    if categories and subcategories:
        for category in categories:
            advertised = taxonomy.categories.get(category)
            if advertised is None:
                continue
            if not set(subcategories).intersection(advertised.subcategories):
                issues.append(
                    f"category '{category}' has no selected subcategory"
                )

    effective_categories = categories
    if subcategories and not categories:
        inferred_categories = sorted(
            {
                owner
                for subcategory_owners in owners.values()
                for owner in subcategory_owners
            },
            key=str.casefold,
        )
        if len(inferred_categories) > 1:
            issues.append(
                "subcategory selection has multiple owning categories; select "
                "exactly one advertised category"
            )
            effective_categories = []
        else:
            effective_categories = inferred_categories

    constraints: dict[str, list[str]] = {}
    if effective_categories:
        if taxonomy.category_field:
            constraints[taxonomy.category_field] = effective_categories
        else:
            issues.append("the catalog does not advertise a category filter field")
    if subcategories:
        if taxonomy.subcategory_field:
            constraints[taxonomy.subcategory_field] = subcategories
        else:
            issues.append("the catalog does not advertise a subcategory filter field")
    return constraints, issues


def _catalog_search_scope(
    taxonomy: dict[str, list[str]],
    required_constraints: dict[str, Any],
) -> dict[str, Any]:
    """Return the normalized hard-filter identity for one catalog search."""

    return {
        "taxonomy": _normalized_scope_value(taxonomy),
        "required_constraints": _normalized_scope_value(required_constraints),
    }


def _normalized_scope_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized_scope_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        normalized = [_normalized_scope_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return value


class AddCartItemsToolItemInput(BaseModel):
    product_ref: str = Field(
        ...,
        min_length=1,
        description="PRODUCT_REF returned by search_catalog_tool in this conversation.",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="Quantity of this product to add.",
    )
    expected_display_name: str | None = Field(
        default=None,
        description=(
            "Shopper-facing product name the agent intends to add. When the "
            "shopper explicitly names the product, copy that exact product name."
        ),
    )


class AddCartItemsToolInput(BaseModel):
    items: list[AddCartItemsToolItemInput] = Field(
        ...,
        min_length=1,
        description=(
            "One or more products to add. Each must use a PRODUCT_REF "
            "established by current-turn search or historical-product resolution."
        ),
    )


class _UpdateCartItemsInput(BaseModel):
    cart_line_id: str = Field(
        description="CART_LINE_ID from get_cart_tool. Not the product name."
    )
    quantity: int = Field(
        ge=0,
        description="New total quantity. Set to 0 to remove the line.",
    )


class _GetStorePolicyInput(BaseModel):
    topic: Literal[
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    ] = Field(description="Policy topic to look up.")


class _CheckAvailabilityInput(BaseModel):
    product_ref: str = Field(
        description=(
            "PRODUCT_REF established by current-turn search or historical-product "
            "resolution."
        )
    )
    variant_hint: str | None = Field(
        default=None,
        description="Requested size wording, such as 'size 8'.",
    )


@dataclass(frozen=True)
class RequestIdentity:
    """Server-owned identity used to scope one assistant turn."""

    session_id: str
    conversation_id: str
    cart_id: str
    context_user_id: int
    cart_user_id: int
    request_id: str
    shopper_profile_id: str | None = None

    @property
    def legacy_user_id(self) -> int:
        return self.context_user_id

    @property
    def checkpoint_thread_id(self) -> str:
        return json.dumps(
            [self.conversation_id, self.request_id],
            separators=(",", ":"),
        )


def _conversation_turn_status(termination_reason: str) -> FinalTurnStatus:
    if termination_reason in {
        "input_guardrail_blocked",
        "output_guardrail_blocked",
    }:
        return "blocked"
    if termination_reason == "completed":
        return "completed"
    return "failed"


class DeepAgentsRuntime:
    """Small adapter around the Deep Agents SDK.

    This class intentionally keeps commerce truth in existing services. Deep
    Agents gets scoped context and deterministic tools; it does not own carts,
    profiles, prices, inventory, or session identity.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._checkpointer = _build_checkpointer()
        self._profile_registered = False
        self._media_perception = MediaPerceptionClient(config)
        self._catalog_capabilities = CatalogCapabilitiesClient(
            config.retriever_port,
            timeout_seconds=config.catalog_search_timeout_seconds,
        )
        self._conversation_memory = ConversationMemoryClient(config.memory_port)
        self._conversation_products = ConversationProductsClient(config.memory_port)
        self._weather_client = build_weather_client(config.weather)

    def catalog_capabilities(self) -> CatalogCapabilities:
        """Return the process-lifecycle catalog capability contract."""

        return self._catalog_capabilities.get()

    def _exposed_agent_diagnostics(self, output: State) -> dict[str, Any]:
        if not getattr(self.config, "expose_agent_diagnostics", False):
            return {}
        return output.agent_diagnostics

    async def astream(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> AsyncIterator[str]:
        output = await self._run_turn(state, identity)
        products = output.product_results or []
        if products:
            yield json.dumps(
                {"type": "products", "payload": products, "timestamp": time.time()}
            )
        images = output.retrieved or {}
        yield json.dumps({"type": "images", "payload": images, "timestamp": time.time()})
        if output.response:
            yield json.dumps(
                {"type": "content", "payload": output.response, "timestamp": time.time()}
            )
        yield json.dumps(
            {
                "type": "metrics",
                "payload": {
                    "timings": output.timings,
                    "total_seconds": sum(output.timings.values()),
                    "token_usage": _normalized_token_usage(output.token_usage),
                    "model_usage": output.model_usage,
                    "agent_diagnostics": self._exposed_agent_diagnostics(output),
                },
                "timestamp": time.time(),
            }
        )

    async def ainvoke(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> dict[str, Any]:
        output = await self._run_turn(state, identity)
        return {
            "response": output.response,
            "images": output.retrieved or {},
            "cart": output.cart.model_dump(mode="json"),
            "timings": output.timings,
            "token_usage": _normalized_token_usage(output.token_usage),
            "model_usage": output.model_usage,
            "agent_diagnostics": self._exposed_agent_diagnostics(output),
        }

    async def _run_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> State:
        state.user_id = identity.context_user_id
        state.agent_diagnostics = _empty_agent_diagnostics("not_started")
        state.previous_selected_skill_names = []
        state.selected_skill_names = []
        state.shopper_profile_id = identity.shopper_profile_id
        state.shopper_context = None
        state.conversation_summary = ""
        state.historical_product_context = ""
        state.conversation_projection_version = 0
        state.active_weather_receipts = []
        state.selected_weather_receipt_id = None
        state.weather_receipt_promotion = None
        turn = self._start_conversation_turn(state, identity)
        if turn is not None and turn.replayed:
            await self._delete_turn_checkpoint(identity)
            return self._restore_replayed_turn(state, turn)
        if turn is None and state.response:
            return state

        summary_advance = None
        try:
            output = await self._execute_turn(state, identity)
            if (
                turn is not None
                and state.agent_diagnostics.get("final_termination_reason")
                == "completed"
            ):
                summary_advance = await self._prepare_conversation_summary(
                    state,
                    turn,
                )
        except asyncio.CancelledError:
            if turn is not None:
                if not state.response:
                    state.response = (
                        "This request was interrupted. Please check your cart "
                        "before retrying."
                    )
                finalized = self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="request_cancelled",
                    present_products=False,
                )
                if finalized:
                    await self._delete_turn_checkpoint(identity)
            raise
        except Exception:
            if turn is not None:
                finalized = self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="unexpected_runtime_error",
                    present_products=False,
                )
                if finalized:
                    await self._delete_turn_checkpoint(identity)
            raise

        if turn is not None:
            finalized = self._finalize_conversation_turn(
                state,
                identity,
                turn,
                summary_advance=summary_advance,
            )
            if finalized:
                await self._delete_turn_checkpoint(identity)
        return output

    async def _execute_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> State:
        start = time.monotonic()

        if state.guardrails:
            safety_start = time.monotonic()
            input_safe, input_check_ok = self._check_safety(
                "input",
                identity.context_user_id,
                state.query,
            )
            state.timings["safety_input"] = time.monotonic() - safety_start
            _record_safety_model_usage(state, "input", ok=input_check_ok)
            if not input_safe:
                state.response = self.config.unsafe_message
                state.timings["deepagents"] = time.monotonic() - start
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "input_guardrail_blocked"
                )
                return state

        media_start = time.monotonic()
        state.media_analysis = await self._media_perception.analyze(state)
        if state.media:
            state.timings["media_perception"] = time.monotonic() - media_start
            _record_media_model_usage(state, self.config)
        if _should_short_circuit_media_failure(state):
            state.response = _media_failure_response(state.media_analysis)
            state.timings["deepagents"] = time.monotonic() - start
            state.agent_diagnostics = _empty_agent_diagnostics("media_failure")
            return state

        turn_capabilities = await asyncio.to_thread(self._catalog_capabilities.get)
        invoke_config = {
            "configurable": {"thread_id": identity.checkpoint_thread_id},
            "recursion_limit": self.config.deepagents_recursion_limit,
        }
        agent = None
        try:
            execution_deadline = (
                time.monotonic()
                + self.config.deepagents_execution_timeout_seconds
            )
            agent = self._create_agent(
                state,
                identity,
                turn_capabilities,
            )
            input_message = self._build_user_message(state, identity)
            agent_timeout = max(0.0, execution_deadline - time.monotonic())
            if agent_timeout <= 0:
                raise TimeoutError
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": input_message}]},
                    config=invoke_config,
                ),
                timeout=agent_timeout,
            )
            result_messages = _result_messages(result)
            state.selected_skill_names = list(
                selected_skill_names_for_turn(
                    result_messages,
                    identity.request_id,
                )
            )
            draft_response = _extract_final_text(result)
            state.token_usage = _collect_token_usage(result)
            state.agent_diagnostics = _safe_collect_agent_diagnostics(
                result_messages,
                request_id=identity.request_id,
                final_termination_reason="completed",
                saved_zipcode=_saved_zipcode(state.shopper_context),
            )
            state.weather_receipt_promotion = (
                _current_weather_receipt_promotion(
                    state,
                    result,
                    request_id=identity.request_id,
                    ttl_seconds=self.config.weather.receipt_ttl_seconds,
                )
            )
            if state.weather_receipt_promotion is not None:
                state.agent_diagnostics["weather_receipt_status"] = (
                    "promotion_prepared"
                )
            elif _bound_weather_receipt(state) is not None:
                state.agent_diagnostics["weather_receipt_status"] = "bound"
            state.response = (
                _no_direct_taxonomy_response(
                    result,
                    request_id=identity.request_id,
                )
                or _unsupported_requirement_response(
                    result,
                    request_id=identity.request_id,
                )
            )
            if not state.response:
                remaining_seconds = max(
                    0.0,
                    execution_deadline - time.monotonic(),
                )
                state.response = await self._rewrite_response_for_grounding(
                    state,
                    result,
                    draft_response,
                    request_id=identity.request_id,
                    timeout_seconds=remaining_seconds,
                )
            if not state.response:
                state.response = (
                    "I could not complete that shopping request. Please try again."
                )
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = "incomplete_agent_response"
        except Exception as exc:  # noqa: BLE001 - keep endpoint resilient.
            partial_messages, capture_error = await _partial_graph_messages(
                agent,
                invoke_config,
            )
            state.selected_skill_names = list(
                selected_skill_names_for_turn(
                    partial_messages,
                    identity.request_id,
                )
            )
            if isinstance(exc, GraphRecursionError):
                termination_reason = "recursion_limit"
            elif isinstance(exc, ShopperSkillActivationError):
                termination_reason = "skill_activation_failed"
            elif isinstance(exc, TimeoutError):
                termination_reason = "agent_timeout"
            else:
                termination_reason = "agent_error"
            state.agent_diagnostics = _safe_collect_agent_diagnostics(
                partial_messages,
                request_id=identity.request_id,
                final_termination_reason=termination_reason,
                preserve_partial_messages=True,
                saved_zipcode=_saved_zipcode(state.shopper_context),
            )
            if capture_error:
                state.agent_diagnostics["partial_graph_capture_error"] = capture_error
            logger.exception("DeepAgentsRuntime failed")
            if termination_reason == "agent_timeout":
                state.product_results = []
                state.retrieved = {}
                state.response = (
                    "This request took too long to complete. Please retry. If it "
                    "involved a cart change, check your cart first."
                )
            else:
                fallback_response = _partial_product_results_response(state)
                state.response = fallback_response or (
                    "I encountered an error while helping with your shopping request. "
                    "Please try again."
                )
            _record_language_model_failure(state)
            state.timings["deepagents_error"] = time.monotonic() - start
            return state

        final_weather_outcome = _current_weather_outcome(
            result,
            request_id=identity.request_id,
        )
        if isinstance(final_weather_outcome, WeatherForecastEvidence):
            state.response = _ensure_weather_forecast_evidence(
                state.response,
                final_weather_outcome,
            )
        state.response = _scrub_saved_zip_from_response(
            state.response,
            state.shopper_context,
        )
        state.response = _scrub_weather_receipt_ids(state.response, state)
        if state.guardrails:
            safety_start = time.monotonic()
            output_safe, output_check_ok = self._check_safety(
                "output",
                identity.context_user_id,
                state.response,
            )
            state.timings["safety_output"] = time.monotonic() - safety_start
            _record_safety_model_usage(state, "output", ok=output_check_ok)
            if not output_safe:
                state.response = self.config.unsafe_message
                state.weather_receipt_promotion = None
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = "output_guardrail_blocked"

        state.timings["deepagents"] = time.monotonic() - start
        return state

    def _create_agent(
        self,
        state: State,
        identity: RequestIdentity,
        turn_capabilities: CatalogCapabilities | None = None,
    ):
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from langchain_core.tools import tool

        # One cached lifecycle contract is authoritative for prompt construction
        # and deterministic validation. Catalog requests are revalidated by the
        # active catalog service before execution.
        if turn_capabilities is None:
            turn_capabilities = self._catalog_capabilities.get()
        current_utc_date = CalendarDate.fromisoformat(
            time.strftime("%Y-%m-%d", time.gmtime())
        )

        if not self._profile_registered:
            register_harness_profile(
                f"openai:{self.config.llm_name}",
                HarnessProfile(
                    excluded_tools=_EXCLUDED_DEEP_AGENT_TOOLS,
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )
            self._profile_registered = True

        skills_root = self._shopper_skills_root()
        skills_backend = self._create_skills_backend()
        if skills_backend is None:
            raise RuntimeError("Shopper skill backend is unavailable.")
        skill_registry = _shopper_skill_registry(skills_root)
        shopper_authored_texts = _shopper_authored_texts(state)
        weather_date_available = weather_date_context_available(
            shopper_authored_texts
        )
        skill_activation_input = _skill_activation_input_model(
            tuple(skill_registry),
            tuple(
                receipt.receipt_id for receipt in state.active_weather_receipts
            ),
            weather_date_available=_current_turn_weather_date_available(state),
        )
        retrieved: dict[str, str] = {}
        state.retrieved = retrieved
        search_input_model = _search_catalog_tool_input_model(turn_capabilities)
        search_tool_arguments_model = _search_catalog_tool_input_model(
            turn_capabilities,
            validate_scope=False,
        )
        constraint_input_model = search_input_model.model_fields[
            "required_constraints"
        ].annotation
        catalog_searches_this_turn = 0
        searched_catalog_scopes: list[dict[str, Any]] = []
        searched_shopper_scopes: set[tuple[str, str]] = set()
        catalog_tool_lock = Lock()
        product_detail_reads_this_turn = 0
        failed_repair_scope_key: str | None = None
        failed_agent_selected_scope = False
        failed_constraint_scope_key: str | None = None
        constraint_reviewed_scopes: set[str] = set()
        pending_constraint_reviews: dict[str, dict[str, Any]] = {}
        pending_taxonomy_constraints: dict[str, Any] | None = None
        pending_no_direct_constraint_clear = False
        pending_schema_requirements: list[str] = []
        product_evidence = ProductEvidence()
        product_resolution_used = False
        product_resolution_lock = Lock()

        def _lock_taxonomy_constraint_values(
            scope_key: str | None,
            constraints: dict[str, Any],
            *,
            allow_no_direct_clear: bool = False,
        ) -> str:
            """Store canonical hard constraints for one taxonomy repair."""

            nonlocal pending_taxonomy_constraints
            nonlocal pending_no_direct_constraint_clear
            constraints = _normalized_scope_value(constraints)
            constraints.pop("unadvertised_requirements", None)
            pending_taxonomy_constraints = constraints
            pending_no_direct_constraint_clear = allow_no_direct_clear
            serialized_constraints = json.dumps(
                constraints,
                ensure_ascii=False,
                sort_keys=True,
            )
            if allow_no_direct_clear:
                return (
                    " Clear advertised required_constraints if the corrected "
                    "request does not retrieve. Otherwise preserve these "
                    "capability-validated advertised required_constraints "
                    f"exactly: {serialized_constraints}."
                )
            if not constraints:
                return (
                    " The rejected call had no advertised required_constraints. "
                    "Keep advertised required_constraints empty on repair. "
                    "Change only taxonomy or an explicitly identified ungrounded "
                    "product scope."
                )
            return (
                " Preserve these capability-validated advertised "
                "required_constraints exactly on repair: "
                f"{serialized_constraints}. Change only taxonomy or an "
                "explicitly identified ungrounded product scope."
            )

        def _lock_taxonomy_constraints(
            scope_key: str | None,
            request: SearchCatalogToolArguments,
        ) -> str:
            """Preserve validated hard constraints across one taxonomy repair."""

            return _lock_taxonomy_constraint_values(
                scope_key,
                request.required_constraints.model_dump(exclude_none=True),
            )

        @tool(args_schema=search_tool_arguments_model, return_direct=False)
        def search_catalog_tool(
            semantic_query: str,
            requested_product_type: str | None,
            taxonomy: BaseModel | dict[str, Any],
            required_constraints: BaseModel | dict[str, Any],
            shopper_guidance: str,
            scope_complete: bool = True,
            search_mode: str | None = None,
        ) -> str:
            """Find products by description, advertised taxonomy, or constraints.

            Use for browse, search, and recommendation requests after product
            discovery or outfit styling is active. Select exact values from the
            current Catalog capabilities. Do not use for a product already
            established in this conversation, and do not repeat a completed hard-
            filter scope with different semantic wording. Tool availability is
            not a search request. Do not call when the current shopper message
            only supplies the event destination, venue, or date requested in the
            prior response; preserve established candidates. If that same message
            also asks for a comparison, refinement, replacement, or new product,
            follow the active primary skill's normal procedure.
            """

            nonlocal catalog_searches_this_turn
            nonlocal failed_agent_selected_scope
            nonlocal failed_constraint_scope_key
            nonlocal failed_repair_scope_key
            nonlocal pending_no_direct_constraint_clear
            nonlocal pending_taxonomy_constraints
            nonlocal pending_schema_requirements
            taxonomy = taxonomy or {"category": [], "subcategory": []}
            required_constraints = required_constraints or {}
            capabilities = turn_capabilities
            if capabilities.catalog_id == "unavailable" and not capabilities.filters:
                return "Catalog search is unavailable. Please try again."
            initial_scope_key = _product_scope_key(requested_product_type)
            shopper_stated_requested_scope = bool(
                initial_scope_key
                and _shopper_stated_product_scope(
                    state.query,
                    state.context,
                    initial_scope_key,
                )
            )
            taxonomy_status = _catalog_execution_taxonomy_status(
                requested_product_type,
                taxonomy,
                semantic_query,
                capabilities,
                shopper_stated_scope=shopper_stated_requested_scope,
            )

            requested_product_type = _resolved_agent_selected_product_type(
                query=state.query,
                context=state.context,
                requested_product_type=requested_product_type,
                taxonomy_status=taxonomy_status,
                taxonomy=taxonomy,
            )
            candidate_scope_key = _product_scope_key(requested_product_type)
            locked_repair_scope = (
                failed_constraint_scope_key or failed_repair_scope_key
            )
            repairing_same_scope = bool(
                locked_repair_scope
                and (
                    candidate_scope_key == failed_constraint_scope_key
                    if failed_constraint_scope_key
                    else _same_product_scope(
                        locked_repair_scope,
                        candidate_scope_key,
                        capabilities,
                    )
                )
            )
            if (
                locked_repair_scope
                and not repairing_same_scope
            ):
                expected_scope_key = locked_repair_scope
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "A catalog search repair cannot replace product scope "
                    f"'{expected_scope_key}' "
                    f"with '{candidate_scope_key or 'none'}'. Preserve the "
                    "requested_product_type and repair taxonomy instead."
                )
            if (
                failed_agent_selected_scope
                and taxonomy_status != "agent_selected_type"
            ):
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "This open-role repair must choose exactly one advertised "
                    "subcategory for the role."
                )

            taxonomy_payload = (
                taxonomy.model_dump()
                if isinstance(taxonomy, BaseModel)
                else taxonomy
            )
            constraint_payload = (
                required_constraints.model_dump()
                if isinstance(required_constraints, BaseModel)
                else required_constraints
            )
            shopper_stated_scope = bool(
                candidate_scope_key
                and _shopper_stated_product_scope(
                    state.query,
                    state.context,
                    candidate_scope_key,
                )
            )
            raw_unadvertised_requirements = (
                constraint_payload.get("unadvertised_requirements", [])
                if isinstance(constraint_payload, dict)
                else []
            )
            duplicated_product_type = bool(
                shopper_stated_scope
                and _duplicates_unavailable_product_type(
                    raw_unadvertised_requirements,
                    requested_product_type,
                    capabilities,
                )
            )
            if duplicated_product_type:
                failed_repair_scope_key = candidate_scope_key
                failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "Product types never belong in "
                    "unadvertised_requirements. Preserve the shopper-stated "
                    "requested_product_type, remove it from requirements, and "
                    "preserve the selected advertised parent category when it is "
                    "a faithful broader scope. Otherwise ask one concise "
                    "clarification."
                )
            stated_unadvertised_requirements = (
                [
                    requirement
                    for requirement in raw_unadvertised_requirements
                    if isinstance(requirement, str)
                    and _shopper_stated_requirement(state.query, requirement)
                ]
                if isinstance(raw_unadvertised_requirements, list)
                else []
            )
            if (
                isinstance(raw_unadvertised_requirements, list)
                and raw_unadvertised_requirements
                and (shopper_stated_scope or stated_unadvertised_requirements)
            ):
                return _unsupported_requirement_message(
                    raw_unadvertised_requirements
                )
            try:
                request = search_input_model.model_validate(
                    {
                        "semantic_query": semantic_query,
                        "shopper_guidance": shopper_guidance,
                        "requested_product_type": requested_product_type,
                        "taxonomy_status": taxonomy_status,
                        "taxonomy": taxonomy_payload,
                        "required_constraints": constraint_payload,
                        "scope_complete": scope_complete,
                        "search_mode": search_mode,
                    }
                )
            except ValidationError as exc:
                validation_errors = [
                    {
                        "loc": list(error.get("loc") or ()),
                        "type": str(error.get("type") or "validation_error"),
                        "msg": str(error.get("msg") or "Invalid value"),
                    }
                    for error in exc.errors(include_url=False)
                ]
                taxonomy_error = any(
                    any(
                        marker in (
                            str(error.get("loc", ""))
                            + " "
                            + str(error.get("msg", ""))
                        )
                        .replace("_", " ")
                        .casefold()
                        for marker in (
                            "taxonomy",
                            "category",
                            "subcategory",
                            "requested product type",
                        )
                    )
                    for error in validation_errors
                )
                repair_guidance = ""
                canonical_agent_selected_scope = bool(
                    candidate_scope_key
                    and taxonomy_status == "agent_selected_type"
                    and _agent_selected_scope_is_advertised(
                        requested_product_type,
                        taxonomy,
                    )
                )
                if shopper_stated_scope or canonical_agent_selected_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                elif taxonomy_error and taxonomy_status == "agent_selected_type":
                    failed_agent_selected_scope = True
                if candidate_scope_key and taxonomy_error:
                    if (
                        taxonomy_status == "agent_selected_type"
                        and not shopper_stated_scope
                    ):
                        advertised_choices = _advertised_subcategories_for_selection(
                            taxonomy,
                            capabilities,
                        )
                        repair_guidance = (
                            " For an open-role search, choose "
                            "exactly one advertised subcategory and copy it into "
                            "requested_product_type. Choose exactly one of these "
                            "currently advertised subcategories: "
                            + json.dumps(advertised_choices, ensure_ascii=False)
                            + "."
                        )
                        constraints = (
                            required_constraints.model_dump(exclude_none=True)
                            if isinstance(required_constraints, BaseModel)
                            else required_constraints
                        )
                        proposed_requirements = constraints.get(
                            "unadvertised_requirements",
                            [],
                        )
                        if proposed_requirements:
                            pending_schema_requirements = list(
                                proposed_requirements
                            )
                            repair_guidance += (
                                " The rejected call proposed "
                                "unadvertised_requirements "
                                + json.dumps(
                                    proposed_requirements,
                                    ensure_ascii=False,
                                )
                                + ". Preserve only an objective product attribute "
                                "directly stated for the selected role; remove one "
                                "inferred from season, weather, occasion, or style."
                            )
                constraint_lock = ""
                try:
                    validated_constraints = constraint_input_model.model_validate(
                        constraint_payload
                    )
                except ValidationError:
                    pass
                else:
                    constraint_lock = _lock_taxonomy_constraint_values(
                        candidate_scope_key,
                        validated_constraints.model_dump(exclude_none=True),
                        allow_no_direct_clear=(
                            taxonomy_status == "no_direct_catalog_match"
                        ),
                    )
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "The catalog search request does not match current "
                    f"capabilities: {validation_errors}"
                    + repair_guidance
                    + constraint_lock
                )

            all_constraints = request.required_constraints.model_dump(
                exclude_none=True,
            )
            normalized_advertised_constraints = _normalized_scope_value(
                all_constraints
            )
            normalized_advertised_constraints.pop(
                "unadvertised_requirements",
                None,
            )
            if (
                pending_taxonomy_constraints is not None
                and not (
                    pending_no_direct_constraint_clear
                    and request.taxonomy_status == "no_direct_catalog_match"
                )
                and normalized_advertised_constraints
                != pending_taxonomy_constraints
            ):
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "A taxonomy repair must preserve previously validated "
                    "advertised required_constraints exactly. Change only "
                    "taxonomy or an explicitly identified "
                    "ungrounded product scope."
                )
            if pending_taxonomy_constraints is not None:
                pending_taxonomy_constraints = None
                pending_no_direct_constraint_clear = False
            normalized_constraints = dict(all_constraints)
            unadvertised_requirements = normalized_constraints.pop(
                "unadvertised_requirements",
                [],
            )
            shopper_stated_scope = bool(
                candidate_scope_key
                and _shopper_stated_product_scope(
                    state.query,
                    state.context,
                    candidate_scope_key,
                )
            )
            if unadvertised_requirements and shopper_stated_scope:
                return _unsupported_requirement_message(
                    unadvertised_requirements
                )

            advertised_taxonomy_issue = _advertised_taxonomy_scope_issue(
                request.requested_product_type,
                request.taxonomy_status,
                request.taxonomy,
                capabilities,
            )
            if advertised_taxonomy_issue:
                if shopper_stated_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + advertised_taxonomy_issue
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                    + (
                        " Preserve the shopper-stated requested_product_type."
                        if shopper_stated_scope
                        else " The rejected requested_product_type was not "
                        "shopper-stated. Re-read the current shopper request "
                        "and correct it rather than preserving this scope."
                    )
                )

            if pending_schema_requirements and not unadvertised_requirements:
                request = request.model_copy(
                    update={
                        "shopper_guidance": _generic_shopper_guidance(
                            request.requested_product_type
                        )
                    }
                )
                pending_schema_requirements = []
            pending_constraint_review = pending_constraint_reviews.get(
                candidate_scope_key
            )
            if pending_constraint_review and (
                request.taxonomy.model_dump()
                != pending_constraint_review["taxonomy"]
                or request.scope_complete
                != pending_constraint_review["scope_complete"]
                or request.search_mode != pending_constraint_review["search_mode"]
                or normalized_constraints
                != pending_constraint_review["required_constraints"]
            ):
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + "A constraint-provenance repair must preserve "
                    "requested_product_type, taxonomy, scope_complete, "
                    "search_mode, and all "
                    "advertised required constraints exactly. Change only the "
                    "reviewed unadvertised requirement wording or remove an "
                    "inferred requirement; the soft semantic query may be "
                    "corrected within the preserved product scope."
                )
            agent_selected_issue: str | None = None
            agent_selected_shopper_scope = False
            if (
                request.taxonomy_status == "agent_selected_type"
                and (
                    _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                    or not _agent_selected_scope_is_advertised(
                        request.requested_product_type,
                        request.taxonomy,
                    )
                )
            ):
                agent_selected_shopper_scope = bool(
                    candidate_scope_key
                    and _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                )
                if agent_selected_shopper_scope:
                    payload = request.taxonomy.model_dump()
                    selected_values = (
                        payload.get("subcategory")
                        or payload.get("category")
                        or []
                    )
                    advertised_match = _advertised_scope_match(
                        request.requested_product_type,
                        capabilities,
                    )
                    exact_selected_scope = bool(
                        advertised_match
                        and len(selected_values) == 1
                        and _normalize_product_text(selected_values[0])
                        == _normalize_product_text(advertised_match[1])
                    )
                    repair_status = (
                        "Keep that exact advertised taxonomy selection."
                        if exact_selected_scope
                        else (
                            "Select only advertised values that are kinds of the "
                            "named scope; do not narrow to one convenient child."
                        )
                    )
                    agent_selected_issue = (
                        "The shopper named requested_product_type "
                        f"'{request.requested_product_type}', so "
                        "preserve that requested_product_type and these advertised "
                        "values on repair: "
                        + json.dumps(selected_values, sort_keys=True)
                        + ". "
                        + repair_status
                    )
                else:
                    advertised_choices = _advertised_subcategories_for_selection(
                        request.taxonomy,
                        capabilities,
                    )
                    agent_selected_issue = (
                        "An open-role search must select exactly one advertised "
                        "subcategory and copy that exact "
                        "subcategory into requested_product_type. Do not use a "
                        "parent category or an unadvertised role. Choose exactly "
                        "one of these currently advertised subcategories: "
                        + json.dumps(advertised_choices, ensure_ascii=False)
                        + ". Rewrite "
                        "shopper_guidance for that selected role without "
                        "asserting an inferred product attribute."
                    )
            if agent_selected_issue:
                if agent_selected_shopper_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                elif candidate_scope_key:
                    failed_agent_selected_scope = True
                constraint_issue = ""
                if unadvertised_requirements:
                    constraint_issue = (
                        " The same rejected call proposed "
                        "unadvertised_requirements "
                        + json.dumps(
                            unadvertised_requirements,
                            ensure_ascii=False,
                        )
                        + ". Preserve only an objective product attribute "
                        "directly stated for the selected role. If it was "
                        "inferred from season, weather, occasion, or style, "
                        "send an empty list and remove its promise from "
                        "shopper_guidance."
                    )
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + agent_selected_issue
                    + constraint_issue
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                )
            failed_agent_selected_scope = False

            if (
                request.taxonomy_status != "no_direct_catalog_match"
                and unadvertised_requirements
            ):
                stated_requirements = [
                    requirement
                    for requirement in unadvertised_requirements
                    if _shopper_stated_requirement(state.query, requirement)
                ]
                shopper_stated_scope = bool(
                    candidate_scope_key
                    and _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                )
                if stated_requirements or shopper_stated_scope:
                    return _unsupported_requirement_message(
                        unadvertised_requirements
                    )
                review_scope = candidate_scope_key or "__unknown__"
                if review_scope in constraint_reviewed_scopes:
                    return (
                        "The requested catalog requirement cannot be enforced: "
                        "its current-turn provenance could not be established. "
                        "Ask the shopper to state the exact required attribute "
                        "or allow it to be treated as a preference."
                    )
                constraint_reviewed_scopes.add(review_scope)
                pending_constraint_reviews[review_scope] = {
                    "requirements": list(unadvertised_requirements),
                    "taxonomy": request.taxonomy.model_dump(),
                    "scope_complete": request.scope_complete,
                    "search_mode": request.search_mode,
                    "required_constraints": dict(normalized_constraints),
                }
                failed_constraint_scope_key = review_scope
                return (
                    CONSTRAINT_REVIEW_PREFIX
                    + "These proposed unadvertised requirements do not match "
                    "the current shopper turn's normalized wording: "
                    + json.dumps(unadvertised_requirements, ensure_ascii=False)
                    + ". Preserve requested_product_type "
                    + json.dumps(request.requested_product_type)
                    + ", taxonomy "
                    + json.dumps(request.taxonomy.model_dump(), sort_keys=True)
                    + ", and scope_complete "
                    + json.dumps(request.scope_complete)
                    + ", search_mode "
                    + json.dumps(request.search_mode)
                    + ", and advertised required constraints "
                    + json.dumps(normalized_constraints, sort_keys=True)
                    + ". Keep semantic_query within that same product scope; "
                    "you may remove inferred attribute wording from it"
                    + ". If the shopper explicitly stated the same objective "
                    "requirement using different words, replace each value with "
                    "the shopper's shortest exact wording. Otherwise the model "
                    "inferred it: remove it from required_constraints and remove "
                    "the attribute claim from shopper_guidance. Implied weather, "
                    "occasion, or style goals are not explicit requirements."
                )

            reviewed_constraint = pending_constraint_reviews.pop(
                candidate_scope_key,
                None,
            )
            if reviewed_constraint:
                request = request.model_copy(
                    update={
                        "shopper_guidance": _generic_shopper_guidance(
                            request.requested_product_type
                        )
                    }
                )

            exact_taxonomy_issue = (
                _exact_taxonomy_issue(
                    request.requested_product_type or "",
                    request.taxonomy,
                )
                if (
                    request.taxonomy_status == "exact_requested_type"
                    and _advertised_scope_match(
                        request.requested_product_type,
                        capabilities,
                    )
                    is None
                )
                else None
            )
            if exact_taxonomy_issue:
                shopper_stated_scope = bool(
                    candidate_scope_key
                    and _shopper_stated_product_scope(
                        state.query,
                        state.context,
                        candidate_scope_key,
                    )
                )
                if shopper_stated_scope:
                    failed_repair_scope_key = candidate_scope_key
                    failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + exact_taxonomy_issue
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                    + (
                        ". Preserve the shopper-stated requested_product_type "
                        f"{json.dumps(request.requested_product_type)}."
                        if shopper_stated_scope
                        else ". Re-read the current shopper request and correct "
                        "requested_product_type if the rejected value was not "
                        "shopper-stated."
                    )
                    + " Choose only advertised taxonomy values that faithfully "
                    "represent that scope. If none does, ask one concise "
                    "clarifying question instead of searching an adjacent type."
                )

            advertised_match = (
                _advertised_taxonomy_value(
                    request.requested_product_type,
                    capabilities,
                )
                if request.taxonomy_status == "no_direct_catalog_match"
                else None
            )
            if advertised_match:
                failed_repair_scope_key = candidate_scope_key
                failed_agent_selected_scope = False
                return (
                    SEARCH_VALIDATION_ERROR_PREFIX
                    + f"The requested product type '{request.requested_product_type}' "
                    f"matches advertised taxonomy value '{advertised_match}'. "
                    "Select that advertised value instead of reporting a gap."
                    + _lock_taxonomy_constraints(candidate_scope_key, request)
                )

            failed_repair_scope_key = None
            failed_constraint_scope_key = None
            failed_agent_selected_scope = False

            shopper_scope_key = (
                (_normalize_product_text(state.query), candidate_scope_key)
                if candidate_scope_key
                and _text_mentions_product_type(
                    state.query,
                    candidate_scope_key,
                )
                else None
            )

            if request.taxonomy_status == "no_direct_catalog_match":
                with catalog_tool_lock:
                    if (
                        shopper_scope_key is not None
                        and shopper_scope_key in searched_shopper_scopes
                    ):
                        return (
                            "STOP_TOOL_USE: This shopper-requested product scope "
                            "was already searched in this turn. Do not search an "
                            "adjacent taxonomy or report the requested scope as "
                            "unavailable. Use the result already returned.\n\n"
                            + _SEARCH_SCOPE_COMPLETE_NOTE
                        )
                lines = [
                    "STOP_TOOL_USE: No faithful advertised catalog taxonomy "
                    "matches the requested product type "
                    f"'{request.requested_product_type}'. "
                    "Do not search adjacent product types. Tell the shopper the "
                    "requested type is not advertised and ask before offering an "
                    "alternative.",
                    _format_catalog_scope_outcome(
                        {
                            "outcome": "no_direct_catalog_match",
                            "requested_product_type": request.requested_product_type,
                        }
                    ),
                ]
                if request.scope_complete:
                    lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
                return "\n\n".join(lines)

            taxonomy_constraints, taxonomy_issues = _taxonomy_hard_constraints(
                request.taxonomy,
                capabilities,
            )
            taxonomy_fields = {
                field_name
                for field_name in (
                    capabilities.taxonomy.category_field,
                    capabilities.taxonomy.subcategory_field,
                )
                if field_name
            }
            overlapping_fields = sorted(
                taxonomy_fields.intersection(normalized_constraints)
            )
            if overlapping_fields:
                taxonomy_issues.append(
                    "taxonomy fields must use the taxonomy selection, not "
                    "required_constraints: " + ", ".join(overlapping_fields)
                )
            if taxonomy_issues:
                return (
                    "The requested catalog taxonomy cannot be enforced: "
                    + "; ".join(taxonomy_issues)
                    + ". Ask the shopper to choose an advertised product type."
                )

            normalized_search_mode = _tool_search_mode(request.search_mode)
            if request.search_mode is not None and (
                normalized_search_mode is None
                or request.search_mode not in capabilities.retrieval_modes
            ):
                return _UNSUPPORTED_SEARCH_MODE_MESSAGE

            intent = CatalogSearchIntent(
                semantic_query=request.semantic_query,
                required_constraints={
                    **normalized_constraints,
                    **taxonomy_constraints,
                },
                search_mode=normalized_search_mode,
            )
            selected_subcategories = _selected_advertised_subcategories(
                request.taxonomy,
                capabilities,
            )
            plan = build_catalog_search_plan(
                intent,
                capabilities,
                has_image=bool(state.image),
                top_k=_multi_subcategory_candidate_limit(
                    selected_subcategories,
                    capabilities,
                    self.config.top_k_retrieve,
                ),
            )
            if not plan.should_search:
                if plan.constraint_issues:
                    return (
                        "The requested catalog requirement cannot be enforced: "
                        + "; ".join(plan.constraint_issues)
                        + ". Ask the shopper to relax it or use an advertised filter."
                    )
                if plan.no_search_reason == "image_search_unavailable":
                    return (
                        "Image search is not available for the active catalog. "
                        "Ask the shopper to describe what they want to find."
                    )
                if plan.no_search_reason == "unsupported_search_mode":
                    return _UNSUPPORTED_SEARCH_MODE_MESSAGE
                if plan.no_search_reason == "missing_image_for_search_mode":
                    return (
                        "That search mode requires an attached image. Ask the shopper "
                        "to attach one or use text search."
                    )
                return "Catalog search requires a query or image."

            with catalog_tool_lock:
                search_scope = _catalog_search_scope(
                    taxonomy_constraints,
                    normalized_constraints,
                )
                if (
                    shopper_scope_key is not None
                    and shopper_scope_key in searched_shopper_scopes
                ):
                    return (
                        "STOP_TOOL_USE: This shopper-requested product scope "
                        "was already searched in this turn. Do not search an "
                        "adjacent taxonomy. Use the result already returned.\n\n"
                        + _SEARCH_SCOPE_COMPLETE_NOTE
                    )
                if search_scope in searched_catalog_scopes:
                    return (
                        "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already "
                        "searched in this turn. Do not retry it "
                        "with a paraphrase or query expansion. Use the products "
                        "already returned, or ask one concise clarifying question."
                    )
                if catalog_searches_this_turn >= self.config.max_catalog_searches_per_turn:
                    return (
                        "STOP_TOOL_USE: Catalog search limit reached for this turn. "
                        "Do not call more tools this turn. Use the products already "
                        "returned in this turn to answer concisely, or ask one concise "
                        "clarifying question if the available products are not enough."
                    )
                searched_catalog_scopes.append(search_scope)
                if shopper_scope_key is not None:
                    searched_shopper_scopes.add(shopper_scope_key)
                catalog_searches_this_turn += 1
                search_budget_exhausted = (
                    catalog_searches_this_turn
                    >= self.config.max_catalog_searches_per_turn
                )

            search_start = time.monotonic()
            execution = execute_catalog_search(
                plan,
                self.config.retriever_port,
                image_base64=state.image,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            catalog_elapsed = time.monotonic() - search_start
            result = execution.result
            if result.ok:
                result = result.model_copy(
                    update={
                        "products": _products_with_subcategory_coverage(
                            result.products,
                            selected_subcategories,
                            self.config.top_k_retrieve,
                        )
                    }
                )
            with catalog_tool_lock:
                state.timings["catalog_search"] = max(
                    state.timings.get("catalog_search", 0.0),
                    catalog_elapsed,
                )
                _record_catalog_model_usage(
                    state,
                    plan,
                    result.ok,
                    fallback_attempted=execution.fallback_attempted,
                )
                if result.ok and result.products:
                    product_evidence.add(result.products)
                    _append_product_results(state, result.products)
                    for product in result.products:
                        if product.image_url:
                            retrieved[product.display_name] = product.image_url
            if not result.ok:
                return result.error.message if result.error else "Catalog search failed."

            confirmed_filters = {
                name: value
                for name, value in plan.hard_filters.items()
                if name not in taxonomy_fields
            }
            scope_relation_evidence = (
                _format_search_scope_relation_evidence(
                    requested_product_type=request.requested_product_type or "",
                    advertised_category=request.taxonomy.category[0],
                )
                if request.taxonomy_status == "parent_category_alternative"
                else ""
            )
            if not result.products:
                lines = [
                    _SEARCH_NO_MATCH_GROUNDING_NOTE,
                    _format_search_taxonomy_evidence(taxonomy_constraints),
                ]
                if scope_relation_evidence:
                    lines.append(scope_relation_evidence)
                if confirmed_filters:
                    lines.append(_format_search_filter_evidence(confirmed_filters))
                lines.append(
                    _format_catalog_scope_outcome(
                        {
                            "outcome": "zero_results",
                            "requested_product_type": request.requested_product_type,
                            "taxonomy": taxonomy_constraints,
                            "confirmed_filters": confirmed_filters,
                        }
                    )
                )
                if request.scope_complete:
                    lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
                elif search_budget_exhausted:
                    lines.append(_SEARCH_BUDGET_EXHAUSTED_NOTE)
                return "\n\n".join(lines)

            lines = [
                _SEARCH_RESULT_GROUNDING_NOTE,
                _format_search_direction_evidence(request.semantic_query),
                _format_search_guidance_evidence(
                    _safe_shopper_guidance(
                        request.shopper_guidance,
                        request.requested_product_type,
                    )
                ),
                _format_search_taxonomy_evidence(taxonomy_constraints),
            ]
            if scope_relation_evidence:
                lines.append(scope_relation_evidence)
            if confirmed_filters:
                lines.append(_format_search_filter_evidence(confirmed_filters))
            if request.scope_complete:
                lines.append(_SEARCH_SCOPE_COMPLETE_NOTE)
            elif search_budget_exhausted:
                lines.append(_SEARCH_BUDGET_EXHAUSTED_NOTE)
            for product in result.products:
                lines.append(_format_product(product))
            prefix = (
                "Image similarity returned no matches; text fallback results:\n\n"
                if execution.fallback_used
                else ""
            )
            return prefix + "\n\n".join(lines)

        @tool(return_direct=False)
        def get_cart_tool() -> str:
            """Read the current cart. Use before cart mutations to get
            CART_LINE_ID values, or when the shopper asks what is in their cart.
            Do NOT call again if the cart was already read this turn and no
            mutation has occurred since.
            """

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_product_images(
                retrieved,
                cart,
                product_evidence.values(),
            )
            return _format_cart(cart)

        @tool(return_direct=False)
        def get_product_details_tool(product_ref: str) -> str:
            """Get detailed facts (material, care, dimensions, closures) for a
            product established in this turn by search or historical-product
            resolution. Requires a PRODUCT_REF — not a product name. Do NOT call
            for initial recommendations. For an explicit comparison, call once
            for each compared PRODUCT_REF in separate model steps before
            answering. Stop immediately if STOP_TOOL_USE is returned.
            """

            nonlocal product_detail_reads_this_turn
            if (
                product_detail_reads_this_turn
                >= self.config.max_product_detail_reads_per_turn
            ):
                return (
                    "STOP_TOOL_USE: Product-detail read limit reached for this "
                    "turn. Do not call more tools this turn. Answer now from the "
                    "details already read and keep any other products to names, "
                    "prices, categories, image availability, and styling role."
                )
            product_detail_reads_this_turn += 1

            cached_product = product_evidence.get(product_ref)
            if cached_product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search this turn or resolve the earlier product first."
                )
            detail_result = get_product_details(
                GetProductDetailsInput(product_id=cached_product.product_id),
                self.config.retriever_port,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            if not detail_result.ok or detail_result.product is None:
                return _product_detail_failure_message(
                    detail_result.error,
                    cart_validation=False,
                )
            product = detail_result.product
            if not _same_product_display_name(
                product.display_name,
                cached_product.display_name,
            ):
                return (
                    "That product reference now resolves to a different item. "
                    "Search the catalog again before using its details."
                )
            if product.image_url:
                retrieved[product.display_name] = product.image_url
            return _format_product_details(product)

        @tool(args_schema=ResolveConversationProductsRequest, return_direct=False)
        def resolve_conversation_products_tool(
            references: list[ProductReferenceDescriptor],
        ) -> str:
            """Resolve products the shopper refers to from earlier in this
            conversation. Use only when a needed product was not established
            in the current turn. For a comparison, submit every compared prior
            product together in this one batched call. Use exact descriptors
            from the historical product index. If a reference is missing or
            ambiguous, ask one concise clarification; do not guess or search
            for a substitute.
            """

            nonlocal product_resolution_used
            with product_resolution_lock:
                if product_resolution_used:
                    return (
                        "STOP_TOOL_USE: Historical product resolution limit "
                        "reached for this turn. Use the first resolution result "
                        "and ask one concise clarification if needed."
                    )
                product_resolution_used = True

            try:
                result = self._conversation_products.resolve(
                    identity.conversation_id,
                    references,
                )
            except (ConversationProductsError, ValidationError):
                return (
                    "REFERENCE RESOLUTION UNAVAILABLE: Ask which earlier product "
                    "the shopper means; do not guess or search for a substitute."
                )
            product_evidence.add_resolutions(result.results)
            for resolution in result.results:
                if resolution.status != "resolved":
                    continue
                product = resolution.matches[0].product
                if product.image_url:
                    retrieved[product.display_name] = product.image_url
            return format_product_resolution(result)

        @tool(args_schema=AddCartItemsToolInput, return_direct=False)
        def add_cart_items_tool(items: list[AddCartItemsToolItemInput]) -> str:
            """Add products to the cart. Use ONLY on explicit shopper intent to
            add, buy, or put items in the cart. Requires PRODUCT_REF values from
            current-turn search or historical-product resolution — not names.
            Call once with all items, not once per item.
            """

            try:
                requested_items = _normalize_cart_add_tool_items(items)
            except ValueError as exc:
                return f"Cart add failed: {exc}"
            if not requested_items:
                return "Cart add failed: provide at least one PRODUCT_REF to add."

            resolved: list[tuple[str, ProductSummary, int]] = []
            failed: list[str] = []
            blocked: list[str] = []
            for product_ref, request in requested_items.items():
                product = product_evidence.get(product_ref)
                if product is None:
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': Search this turn or "
                        "resolve the earlier product first."
                    )
                    continue
                expected_name = request.get("expected_display_name") or ""
                if expected_name and not _same_product_display_name(
                    expected_name,
                    product.display_name,
                ):
                    blocked.append(
                        f"- PRODUCT_REF '{product_ref}': expected "
                        f"'{expected_name}', but that ref resolves to "
                        f"'{product.display_name}'. Use the matching PRODUCT_REF "
                        "for the intended product before adding."
                    )
                    continue
                active_detail = get_product_details(
                    GetProductDetailsInput(product_id=product.product_id),
                    self.config.retriever_port,
                    timeout_seconds=self.config.catalog_search_timeout_seconds,
                )
                if not active_detail.ok or active_detail.product is None:
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': "
                        + _product_detail_failure_message(
                            active_detail.error,
                            cart_validation=True,
                        )
                    )
                    continue
                if not _same_product_display_name(
                    active_detail.product.display_name,
                    product.display_name,
                ):
                    blocked.append(
                        f"- PRODUCT_REF '{product_ref}': That reference now "
                        "resolves to a different product. Search again and use "
                        "the new PRODUCT_REF before adding it."
                    )
                    continue
                resolved.append(
                    (product_ref, active_detail.product, int(request["quantity"]))
                )

            blocked.extend(
                _cart_add_scope_failures(
                    state.query,
                    [(product_ref, product) for product_ref, product, _ in resolved],
                    product_evidence.values(),
                )
            )
            if blocked:
                state.cart = self._read_cart(identity.cart_user_id)
                self._append_product_images(
                    retrieved,
                    state.cart,
                    product_evidence.values(),
                )
                return _format_cart_add_result([], failed + blocked, state.cart)

            added: list[str] = []
            for product_ref, product, quantity in resolved:
                result = add_cart_item(
                    AddCartItemInput(
                        user_id=str(identity.cart_user_id),
                        product_id=product.product_id,
                        display_name=product.display_name,
                        quantity=quantity,
                        unit_price=product.price,
                        image_url=product.image_url,
                        idempotency_key=(
                            f"{identity.request_id}:add:{product.product_id}:{quantity}"
                        ),
                    ),
                    self.config.memory_port,
                )
                if result.ok:
                    added.append(
                        f"- {quantity} x {product.display_name} "
                        f"(PRODUCT_REF: {product.product_id})"
                    )
                else:
                    message = (
                        result.error.message if result.error else "Cart add failed."
                    )
                    failed.append(f"- PRODUCT_REF '{product_ref}': {message}")

            state.cart = self._read_cart(identity.cart_user_id)
            self._append_product_images(
                retrieved,
                state.cart,
                product_evidence.values(),
            )
            return _format_cart_add_result(added, failed, state.cart)

        @tool(return_direct=False)
        def remove_cart_item_tool(cart_line_id: str, quantity: int = 1) -> str:
            """Remove a cart line. Use ONLY on explicit shopper intent to remove
            an item. Requires CART_LINE_ID from get_cart_tool — do not guess.
            Use update_cart_items_tool to change quantity instead of removing
            and re-adding.
            """

            quantity = max(1, int(quantity or 1))
            cart = self._read_cart(identity.cart_user_id)
            line = _cart_line_by_id(cart_line_id, cart)
            if line is None:
                return f"No cart line with CART_LINE_ID '{cart_line_id}' could be found."
            result = remove_cart_item(
                RemoveCartItemInput(
                    user_id=str(identity.cart_user_id),
                    cart_line_id=line["cart_line_id"],
                    product_id=line.get("product_id"),
                    display_name=line["item"],
                    quantity=quantity,
                    idempotency_key=f"{identity.request_id}:remove:{line['cart_line_id']}:{quantity}",
                ),
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            self._append_product_images(
                retrieved,
                state.cart,
                product_evidence.values(),
            )
            return _format_cart_remove_result(
                result,
                fallback=f"Removed {quantity} {line['item']} from cart.",
            )

        @tool(args_schema=_UpdateCartItemsInput, return_direct=False)
        def update_cart_items_tool(cart_line_id: str, quantity: int) -> str:
            """Change the quantity of an item already in the cart, or remove it
            by setting quantity to 0. Use ONLY when the shopper explicitly asks
            to change a quantity or remove by quantity. Do NOT use for initial
            adds — use add_cart_items_tool. Do NOT guess the CART_LINE_ID; call
            get_cart_tool first if you do not have one.
            """

            result = update_cart_item(
                UpdateCartItemInput(
                    user_id=str(identity.cart_user_id),
                    cart_line_id=cart_line_id,
                    quantity=quantity,
                    idempotency_key=(
                        f"{identity.request_id}:update:{cart_line_id}:{quantity}"
                    ),
                ),
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            self._append_product_images(
                retrieved,
                state.cart,
                product_evidence.values(),
            )
            return _format_update_cart_result(result, state.cart)

        @tool(args_schema=_GetStorePolicyInput, return_direct=False)
        def get_store_policy_tool(
            topic: Literal[
                "returns",
                "shipping",
                "sizing",
                "payment",
                "price_match",
                "gift_cards",
            ],
        ) -> str:
            """Look up store policy for: returns, shipping, sizing, payment,
            price_match, or gift_cards. Use ONLY for these policy topics. Do
            NOT use for product facts, prices, or availability. If this tool
            returns a not-found error, relay the message to the shopper and
            direct them to the retailer's help center. Do NOT substitute model
            knowledge for a missing policy.
            """

            result = get_store_policy(
                GetStorePolicyInput(topic=topic),
                _store_policies_path(),
            )
            return _format_policy_result(result)

        @tool(args_schema=_CheckAvailabilityInput, return_direct=False)
        def check_product_availability_tool(
            product_ref: str,
            variant_hint: str | None = None,
        ) -> str:
            """Check whether a specific product is available or in stock. Use
            ONLY when the shopper explicitly asks about availability, stock,
            or a specific size. Requires a PRODUCT_REF established by search or
            historical-product resolution. Do NOT use for browsing. The
            deterministic stub reports general availability, sized availability
            for apparel and footwear, and one-size availability for other
            product categories.
            """

            product = product_evidence.get(product_ref)
            if product is None:
                return (
                    f"PRODUCT_REF '{product_ref}' is unknown in this conversation. "
                    "Search this turn or resolve the earlier product first."
                )
            result = check_product_availability(
                CheckProductAvailabilityInput(
                    product_ref=product_ref,
                    variant_hint=variant_hint,
                ),
                product,
            )
            return _format_availability_result(result)

        @tool(return_direct=False)
        def check_active_promotions_tool() -> str:
            """Check whether a sale, discount, or promotion is currently active.
            Use ONLY when the shopper explicitly asks about promotion status. Do
            NOT use for ordinary affordable browsing, a price ceiling, price
            matching, or product availability. Catalog search does not establish
            sale status.
            """

            return _format_promotions_result(check_active_promotions())

        @tool(return_direct=False)
        def view_cart_total_tool() -> str:
            """Compute the cart subtotal. Use for budget checks or when the
            shopper asks for the total. Does not include tax or shipping. Use
            get_cart_tool for line contents.
            """

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_product_images(
                retrieved,
                cart,
                product_evidence.values(),
            )
            return _format_cart_total(cart)

        tool_loop_control = ToolLoopControlMiddleware(
            catalog_context=format_catalog_capabilities_for_prompt(
                turn_capabilities
            )
        )
        skill_gate = ShopperSkillActivationMiddleware(
            request_id=identity.request_id,
            skill_descriptions={
                name: skill.description
                for name, skill in skill_registry.items()
            },
            skill_tool_grants={
                name: skill.tools_granted
                for name, skill in skill_registry.items()
            },
            previous_selected_skills=state.previous_selected_skill_names,
        )

        weather_forecast_tool = get_event_weather_forecast_tool(
            self._weather_client,
            saved_zipcode=_saved_zipcode(state.shopper_context),
            saved_zip_authorized=_saved_zip_authorized_for_weather(state),
            shopper_provided_texts=shopper_authored_texts,
            current_date=current_utc_date,
        )
        if not weather_date_available:
            skill_gate.deny_tool_for_turn(
                _WEATHER_TOOL_NAME,
                _EVENT_CONTEXT_DATE_REQUIRED_WEATHER_BLOCK,
            )
        shopping_tools = [
            search_catalog_tool,
            get_product_details_tool,
            resolve_conversation_products_tool,
            get_cart_tool,
            add_cart_items_tool,
            remove_cart_item_tool,
            update_cart_items_tool,
            view_cart_total_tool,
            get_store_policy_tool,
            check_product_availability_tool,
            check_active_promotions_tool,
            weather_forecast_tool,
        ]
        validate_registered_tool_names(
            {
                str(
                    getattr(candidate, "name", None)
                    or getattr(candidate, "__name__", "")
                )
                for candidate in shopping_tools
            }
        )
        @tool(args_schema=skill_activation_input, return_direct=False)
        def activate_shopper_skills_tool(
            skill_names: list[str],
            event_context_next_question: EventContextNextQuestion | None = None,
            weather_receipt_id: str | None = None,
        ) -> str:
            """Select and load shopper behavior skills for this turn. This is
            the required first step before answering or calling shopping tools.
            Select the smallest set whose registered descriptions cover the
            complete current intent. Use outfit-styling for outfit building,
            completion, refinement, or mid-browse styling questions; style-led
            single-piece selection, including a statement or balancing piece,
            also uses outfit-styling. Use product-discovery for search and browse
            without styling intent. These are alternative primary procedures,
            never a pair. Add budget-shopping only as a modifier when the shopper
            states a budget. Add event-context only beside outfit-styling, and add
                it whenever an event destination or venue is stated, a supported
                forecast would materially change event guidance, or the guidance
                would otherwise ask about or branch on missing destination or venue
                context. Generic occasion advice is not a reason to omit it. Keep
            outfit-styling as the primary skill throughout an
            active outfit-building thread, including its single-piece follow-up
            searches.
            When event-context is selected, set exactly one
            event_context_next_question: event_location only when destination
            is still missing and material; event_venue only after destination
            is established when venue/setting is missing and material;
            event_date only after destination and any material venue are
            established, weather is enabled and material, and no bounded date
            is established or explicitly unavailable; otherwise none.
            This is the only event-context follow-up the response may ask.
            An explicitly stated outdoor patio, beach, garden, rooftop, or
            open-air setting makes enabled live weather material: with the
            destination and setting established but no bounded date, choose
            event_date, not none. For Cancun alone, choose event_venue when
            setting matters; after
            the shopper confirms a beach setting, choose event_date when live
            weather is enabled and material. Omit it when event-context is not
            selected. When a valid durable weather receipt is listed in the
            current request, bind its weather_receipt_id only when this turn
            continues the exact same event location and date scope, the next
            question is none, and the shopper did not ask for a refresh. A
            location/date correction means omit the receipt. Event context is
            additive: it never replaces the primary skill's product work or
            tool grants.

            """

            try:
                validated_activation = skill_activation_input.model_validate(
                    {
                        "skill_names": skill_names,
                        "event_context_next_question": (
                            event_context_next_question
                        ),
                        "weather_receipt_id": weather_receipt_id,
                    }
                )
            except ValidationError as exc:
                return skill_gate.handle_activation_validation_error(exc)
            selected_names = list(
                dict.fromkeys(validated_activation.skill_names)
            )
            validated_next_question = (
                validated_activation.event_context_next_question
            )
            validated_receipt_id = validated_activation.weather_receipt_id
            try:
                selected_files = {
                    skill_registry[name].path: skill_registry[name].content
                    for name in selected_names
                }
                activated = skill_gate.activate(selected_files, selected_names)
            except (KeyError, ValueError):
                skill_gate.fail()
                return (
                    "SHOPPER_SKILL_ACTIVATION_FAILED: Registered skill "
                    "instructions could not be loaded."
                )
            if not activated:
                return "SHOPPER_SKILL_ACTIVATION_ALREADY_COMPLETE"
            if validated_next_question == "event_location":
                skill_gate.deny_tool_for_turn(
                    _WEATHER_TOOL_NAME,
                    _EVENT_CONTEXT_LOCATION_REQUIRED_WEATHER_BLOCK,
                )
            elif validated_next_question == "event_venue":
                skill_gate.deny_tool_for_turn(
                    _WEATHER_TOOL_NAME,
                    _EVENT_CONTEXT_VENUE_REQUIRED_WEATHER_BLOCK,
                )
            if validated_receipt_id is not None:
                state.selected_weather_receipt_id = validated_receipt_id
                skill_gate.deny_tool_for_turn(
                    _WEATHER_TOOL_NAME,
                    _EVENT_CONTEXT_RECEIPT_BOUND_WEATHER_BLOCK,
                )
            activation_result = (
                f"{SKILL_ACTIVATION_COMPLETE} "
                + ", ".join(selected_files)
            )
            if "event-context" in selected_names:
                reminders = [
                    f"{activation_result}\n"
                    f"{_EVENT_CONTEXT_ADDITIVE_ACTIVATION_REMINDER}"
                ]
                if validated_receipt_id is not None:
                    reminders.append(_EVENT_CONTEXT_RECEIPT_BOUND_REMINDER)
                return "\n".join(reminders)
            return activation_result

        activate_shopper_skills_tool.handle_validation_error = (
            skill_gate.handle_activation_validation_error
        )

        return create_deep_agent(
            model=self._create_chat_model(),
            tools=[activate_shopper_skills_tool, *shopping_tools],
            system_prompt=self._system_prompt(
                turn_capabilities,
                shopper_context=state.shopper_context,
                current_utc_date=current_utc_date,
            ),
            middleware=[tool_loop_control, skill_gate],
            backend=skills_backend,
            checkpointer=self._checkpointer,
        )

    async def _delete_turn_checkpoint(self, identity: RequestIdentity) -> None:
        try:
            delete_thread = getattr(self._checkpointer, "adelete_thread", None)
            if delete_thread is not None:
                await delete_thread(identity.checkpoint_thread_id)
            else:
                self._checkpointer.delete_thread(identity.checkpoint_thread_id)
        except Exception as exc:  # pragma: no cover - cleanup is best effort.
            logger.warning("Could not delete Deep Agents turn checkpoint: %s", exc)

    def _create_chat_model(self):
        from langchain_openai import ChatOpenAI

        api_key_env = getattr(self.config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        return ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _prepare_conversation_summary(
        self,
        state: State,
        turn: TurnStartResult,
    ) -> ConversationSummaryAdvance | None:
        """Compact only memory's oldest prior raw prefix for the next turn."""

        settings = self.config.conversation_summary
        if not settings.enabled:
            return None
        source = turn.summary_compaction_source
        if source is not None:
            source = source.model_copy(
                update={
                    "turns": [
                        item.model_copy(
                            update={
                                "assistant_text": (
                                    _summary_safe_assistant_text(
                                        item.assistant_text
                                    )
                                )
                            }
                        )
                        for item in source.turns
                    ]
                }
            )
        work = build_conversation_summary_work(
            turn.projection,
            source,
            unsummarized_turn_count=turn.unsummarized_turn_count,
            trigger_raw_turns=settings.trigger_raw_turns,
            retain_raw_turns=settings.retain_raw_turns,
            max_input_chars=max(1000, int(self.config.memory_length)),
        )
        if work is None:
            return None

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._create_chat_model().ainvoke(
                    [
                        {
                            "role": "system",
                            "content": CONVERSATION_SUMMARY_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": work.prompt},
                    ]
                ),
                timeout=settings.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            state.timings["conversation_summary"] = time.monotonic() - start
            state.agent_diagnostics["conversation_summary_compaction"] = "timeout"
            _add_model_usage(
                state,
                "app_llm_conversation_summary",
                status="failed",
                calls=1,
                detail="Durable conversation summary timed out; raw turns retained",
            )
            return None
        except Exception:  # noqa: BLE001 - compaction must fail open.
            state.timings["conversation_summary"] = time.monotonic() - start
            state.agent_diagnostics["conversation_summary_compaction"] = "error"
            _add_model_usage(
                state,
                "app_llm_conversation_summary",
                status="failed",
                calls=1,
                detail="Durable conversation summary failed; raw turns retained",
            )
            logger.exception("Durable conversation summary compaction failed")
            return None

        state.timings["conversation_summary"] = time.monotonic() - start
        state.token_usage = _merge_token_usage(
            state.token_usage,
            _collect_token_usage(result),
        )
        content = _content_to_text(_value(result, "content"))
        if not content:
            content = _content_to_text(result)
        summary_text = parse_conversation_summary_output(
            content,
            max_output_chars=settings.max_output_chars,
        )
        if summary_text is None:
            state.agent_diagnostics["conversation_summary_compaction"] = (
                "invalid_output"
            )
            _add_model_usage(
                state,
                "app_llm_conversation_summary",
                status="failed",
                calls=1,
                detail=(
                    "Durable conversation summary returned invalid output; "
                    "raw turns retained"
                ),
            )
            return None

        state.agent_diagnostics["conversation_summary_compaction"] = "prepared"
        _add_model_usage(
            state,
            "app_llm_conversation_summary",
            status="used",
            calls=1,
            detail="Durable semantic conversation summary",
        )
        return ConversationSummaryAdvance(
            expected_projection_version=work.expected_projection_version,
            summary_text=summary_text,
            summary_through_sequence=work.through_sequence,
        )

    async def _rewrite_response_for_grounding(
        self,
        state: State,
        result: Any,
        draft_response: str,
        *,
        request_id: str,
        timeout_seconds: float | None = None,
    ) -> str:
        clarification = _catalog_repair_clarification_response(
            result,
            request_id=request_id,
        )
        if clarification:
            if _has_search_only_tool_evidence(
                result,
                request_id=request_id,
            ):
                grounded_response = self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
                return _scrub_internal_shopper_language(
                    f"{grounded_response}\n\n{clarification}"
                )
            if not _has_successful_non_search_tool_evidence(
                result,
                request_id=request_id,
            ):
                return _scrub_internal_shopper_language(clarification)
            draft_response = clarification
        rejected_search_response = _rejected_catalog_search_response(
            result,
            request_id=request_id,
        )
        if rejected_search_response:
            return rejected_search_response

        max_evidence_chars = getattr(
            self.config,
            "grounding_rewrite_max_evidence_chars",
            12000,
        )
        current_evidence = _collect_tool_grounding_evidence(
            result,
            max_chars=max_evidence_chars,
            request_id=request_id,
        )
        prior_messages = [
            message
            for message in _prior_turn_messages(
                _result_messages(result),
                request_id,
            )
            if not _message_contains_weather_evidence(message)
        ]
        prior_evidence = _collect_message_grounding_evidence(
            prior_messages,
            max_chars=max_evidence_chars,
        )
        search_only = bool(current_evidence) and _has_search_only_tool_evidence(
            result,
            request_id=request_id,
        )
        weather_outcome, reused_weather_receipt = _effective_weather_outcome(
            state,
            result,
            request_id=request_id,
        )
        event_context_next_question = _current_event_context_next_question(
            result,
            request_id=request_id,
        )
        enforce_event_context = self._event_context_response_guidance_active(
            state
        )
        has_current_non_weather_business_activity = (
            _has_current_non_weather_business_tool_activity(
                result,
                request_id=request_id,
            )
        )
        fallback_weather_outcome = (
            None
            if (
                reused_weather_receipt
                and has_current_non_weather_business_activity
            )
            else weather_outcome
        )
        protected_event_response = (
            enforce_event_context
            and not has_current_non_weather_business_activity
            and (
                weather_outcome is not None
                or (
                    not draft_response.strip()
                    and bool(
                        _historical_product_names(
                            state.historical_product_context
                        )
                    )
                )
            )
        )
        if (
            protected_event_response
            and event_context_next_question
            in {"event_location", "event_venue"}
        ):
            return _compose_context_only_event_response(
                state,
                styling="",
                next_question=event_context_next_question,
                weather_outcome=weather_outcome,
                include_weather_facts=not reused_weather_receipt,
            )
        if protected_event_response and not draft_response.strip():
            return _compose_context_only_event_response(
                state,
                styling="",
                next_question=event_context_next_question,
                weather_outcome=weather_outcome,
                include_weather_facts=not reused_weather_receipt,
            )
        current_search_groups = _search_result_groups(
            result,
            request_id=request_id,
        )
        has_current_search_candidates = bool(current_search_groups)
        if not getattr(self.config, "grounding_rewrite_enabled", True):
            if protected_event_response:
                return _compose_context_only_event_response(
                    state,
                    styling="",
                    next_question=event_context_next_question,
                    weather_outcome=weather_outcome,
                    include_weather_facts=not reused_weather_receipt,
                )
            if weather_outcome is not None:
                return self._grounding_failure_fallback(
                    state,
                    result,
                    request_id=request_id,
                    weather_outcome=fallback_weather_outcome,
                )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            return _scrub_internal_shopper_language(draft_response)
        if not draft_response:
            if not search_only and weather_outcome is None:
                return draft_response
            return self._grounding_failure_fallback(
                state,
                result,
                request_id=request_id,
                weather_outcome=fallback_weather_outcome,
            )
        if (
            not current_evidence
            and not prior_evidence
            and not enforce_event_context
        ):
            return _scrub_internal_shopper_language(draft_response)

        start = time.monotonic()
        active_skill_guidance = (
            self._active_skill_response_guidance(state)
            if enforce_event_context
            else ""
        )
        shopper_location_context = (
            self._grounding_shopper_location_context(state)
            if enforce_event_context
            else ""
        )
        event_context_prompt = ""
        if enforce_event_context:
            question_boundary = _event_context_question_boundary(
                event_context_next_question
            )
            event_context_prompt = (
                "SHOPPER LOCATION CANDIDATE:\n"
                f"{shopper_location_context}\n\n"
                "SERVER-ACCEPTED EVENT-CONTEXT QUESTION BOUNDARY:\n"
                f"{question_boundary}\n\n"
                "ACTIVE SKILL RESPONSE GUIDANCE:\n"
                f"{active_skill_guidance}\n\n"
            )
        editor_system_prompt = _GROUNDING_EDITOR_SYSTEM_PROMPT
        skill_guidance_only = (
            enforce_event_context
            and not current_evidence
            and not prior_evidence
        )
        if protected_event_response:
            editor_system_prompt = (
                "You select a tiny event-setting styling decision. Return only "
                "one JSON object with exactly these keys:\n"
                '- "venue_quote": the shortest exact substring of one '
                "SHOPPER-AUTHORED EVENT TEXT line that explicitly names a "
                "venue or setting, or null when none does;\n"
                '- "adjustments": an array containing one or two distinct '
                "values from: streamlined_accessories, "
                "lower_profile_footwear, polished_unfussy_finish, "
                "adaptable_finishing.\n\n"
                "A destination alone is not a venue. Never infer beach, "
                "outdoor, indoor, terrain, climate, or local norms from a "
                "destination. Do not return prose, Markdown, extra keys, "
                "products, prices, materials, questions, or forecast facts. "
                "The server validates the exact shopper quote and renders all "
                "shopper-facing text."
            )
        elif skill_guidance_only:
            editor_system_prompt = (
                "You are the final event-context response editor. Return only "
                "the shopper-facing response.\n\n"
                "Apply this response boundary exactly:\n"
                f"{active_skill_guidance}\n\n"
                "Shopper location-candidate status:\n"
                f"{shopper_location_context}\n\n"
                "Server-accepted event-context question boundary:\n"
                f"{question_boundary}\n\n"
                "Use only USER QUERY and RECENT DISCUSSION for event facts. "
                "Saved ZIP is only a tentative event-location candidate, never "
                "where products will be bought, shipped, or available. Remove "
                "any draft question that repeats an established destination or "
                "venue. Preserve the useful styling direction, but add no "
                "products, location facts, weather, or performance claims."
            )
        elif enforce_event_context and has_current_search_candidates:
            editor_system_prompt = (
                "Highest-priority event-context response boundary:\n"
                f"{active_skill_guidance}\n\n"
                "Shopper location-candidate status:\n"
                f"{shopper_location_context}\n\n"
                "Server-accepted event-context question boundary:\n"
                f"{question_boundary}\n\n"
                "The current catalog search already succeeded. The final "
                "response must name at least one returned candidate exactly as "
                "shown in CURRENT-TURN TOOL EVIDENCE. Do not replace candidates "
                "with a promise to pull, find, or search for products. If event "
                "location is still missing and material, put its single "
                "confirmation question after the candidate set. Do not ask that "
                "question after an explicit destination, open with a location "
                "request, say location is required first, or ask for location "
                "twice. "
                "Saved ZIP is only a tentative event-location candidate, never "
                "shopping, shipping, or availability context. "
                "Apply that boundary to the final text. Remove any draft "
                "question that repeats an established event destination or "
                "venue setting.\n\n"
                f"{_GROUNDING_EDITOR_SYSTEM_PROMPT}"
            )
        elif enforce_event_context:
            editor_system_prompt = (
                "Highest-priority event-context response boundary:\n"
                f"{active_skill_guidance}\n\n"
                "Shopper location-candidate status:\n"
                f"{shopper_location_context}\n\n"
                "Server-accepted event-context question boundary:\n"
                f"{question_boundary}\n\n"
                "Apply that boundary to the final text using only USER QUERY, "
                "RECENT DISCUSSION, and the supplied tool evidence. Do not claim "
                "that current evidence returned a product candidate unless one "
                "is shown, and do not invent a candidate. Remove any draft "
                "question that repeats an established event destination or "
                "venue setting.\n\n"
                f"{_GROUNDING_EDITOR_SYSTEM_PROMPT}"
            )
        if isinstance(weather_outcome, WeatherForecastEvidence) and not (
            protected_event_response
        ):
            if reused_weather_receipt:
                weather_editor_rule = (
                    "Highest-priority bound-weather output rule:\n"
                    "- Use the server-owned durable weather styling direction "
                    "silently while answering the current product task.\n"
                    "- Do not repeat or summarize any forecast place, date, "
                    "condition, temperature, precipitation, attribution, or "
                    "uncertainty fact. No forecast block is appended on this "
                    "comparison turn.\n"
                    "- Ask no repeated location, venue, or date question.\n\n"
                )
            else:
                weather_editor_rule = (
                    "Highest-priority successful-weather output rule:\n"
                    "- Return one or two concise styling sentences before the "
                    "server-authored forecast block.\n"
                    "- Use the supplied forecast evidence silently for styling "
                    "judgment. Do not restate its resolved place, dates, "
                    "condition words, temperatures, precipitation, attribution, "
                    "or uncertainty; the server appends those exact facts.\n"
                    "- When the shopper's turn only supplies event context, "
                    "retain relevant exact candidate names from RECENT "
                    "DISCUSSION or the HISTORICAL PRODUCT INDEX as previously "
                    "shown options. Add no new product facts and do not promise "
                    "or run another search.\n"
                    "- Ask no repeated location, venue, or date question.\n\n"
                )
            editor_system_prompt = weather_editor_rule + editor_system_prompt
        if protected_event_response:
            shopper_event_text = "\n".join(
                f"- {text}" for text in _shopper_authored_texts(state)
            )
            if isinstance(weather_outcome, WeatherForecastEvidence):
                weather_direction = _deterministic_weather_styling_direction(
                    weather_outcome
                )
            elif isinstance(weather_outcome, WeatherFailure):
                weather_direction = _conditional_weather_styling_direction()
            else:
                weather_direction = "(none)"
            prompt = (
                "SHOPPER-AUTHORED EVENT TEXT:\n"
                f"{shopper_event_text or '(none)'}\n\n"
                "SERVER-OWNED WEATHER STYLING DIRECTION:\n"
                f"{weather_direction}"
            )
        else:
            recent_discussion = _redact_prior_weather_assistant_text(
                state.context
            )
            historical_product_context = (
                state.historical_product_context
                or "HISTORICAL PRODUCT INDEX (read-only):\n(none)"
            )
            bound_weather_prompt = ""
            if reused_weather_receipt and isinstance(
                weather_outcome,
                WeatherForecastEvidence,
            ):
                bound_weather_prompt = (
                    "SERVER-BOUND DURABLE WEATHER STYLING DIRECTION "
                    "(exact-scope typed receipt; use silently):\n"
                    f"{_deterministic_weather_styling_direction(weather_outcome)}"
                    "\n\n"
                )
            prompt = (
                f"USER QUERY:\n{state.query}\n\n"
                "DURABLE CONVERSATION SUMMARY "
                "(semantic continuity only; not evidence):\n"
                f"{state.conversation_summary or '(none)'}\n\n"
                f"RECENT DISCUSSION:\n{recent_discussion or '(none)'}\n\n"
                f"{historical_product_context}\n\n"
                f"{event_context_prompt}"
                f"{bound_weather_prompt}"
                f"CURRENT CART:\n{_format_cart(state.cart)}\n\n"
                "AVAILABLE IMAGES:\n"
                f"{_format_retrieved_images(state.retrieved)}\n\n"
                "CURRENT-TURN TOOL EVIDENCE:\n"
                f"{current_evidence or '(none)'}\n\n"
                "PRIOR-TURN TOOL EVIDENCE:\n"
                f"{prior_evidence or '(none)'}\n\n"
                f"DRAFT RESPONSE:\n{draft_response}"
            )
        prompt = _scrub_saved_zip_from_response(
            prompt,
            state.shopper_context,
        )
        try:
            active_timeout = (
                self.config.deepagents_execution_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
            if active_timeout <= 0:
                raise TimeoutError
            rewrite_result = await asyncio.wait_for(
                self._create_chat_model().ainvoke(
                    [
                        {
                            "role": "system",
                            "content": editor_system_prompt,
                        },
                        {"role": "user", "content": prompt},
                    ]
                ),
                timeout=active_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Grounding response editor timed out for request %s",
                request_id,
            )
            state.timings["grounding_rewrite"] = time.monotonic() - start
            state.agent_diagnostics[
                "final_termination_reason"
            ] = "grounding_timeout"
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite timed out",
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if protected_event_response:
                return _compose_context_only_event_response(
                    state,
                    styling="",
                    next_question=event_context_next_question,
                    weather_outcome=weather_outcome,
                    include_weather_facts=not reused_weather_receipt,
                )
            return self._grounding_failure_fallback(
                state,
                result,
                request_id=request_id,
                weather_outcome=fallback_weather_outcome,
            )
        except Exception:  # noqa: BLE001 - response editor has a safe fallback.
            logger.exception("Grounding response editor failed")
            state.timings["grounding_rewrite"] = time.monotonic() - start
            state.agent_diagnostics["final_termination_reason"] = "grounding_error"
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite failed closed",
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if protected_event_response:
                return _compose_context_only_event_response(
                    state,
                    styling="",
                    next_question=event_context_next_question,
                    weather_outcome=weather_outcome,
                    include_weather_facts=not reused_weather_receipt,
                )
            return self._grounding_failure_fallback(
                state,
                result,
                request_id=request_id,
                weather_outcome=fallback_weather_outcome,
            )

        state.timings["grounding_rewrite"] = time.monotonic() - start
        state.token_usage = _merge_token_usage(
            state.token_usage,
            _collect_token_usage(rewrite_result),
        )

        rewritten = _content_to_text(_value(rewrite_result, "content"))
        if not rewritten:
            rewritten = _content_to_text(rewrite_result)
        if not rewritten.strip():
            state.agent_diagnostics["final_termination_reason"] = "grounding_error"
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite returned empty output",
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if protected_event_response:
                return _compose_context_only_event_response(
                    state,
                    styling="",
                    next_question=event_context_next_question,
                    weather_outcome=weather_outcome,
                    include_weather_facts=not reused_weather_receipt,
                )
            return self._grounding_failure_fallback(
                state,
                result,
                request_id=request_id,
                weather_outcome=fallback_weather_outcome,
            )
        _add_model_usage(
            state,
            "app_llm_grounding_editor",
            status="used",
            calls=1,
            detail="Final response grounding rewrite",
        )
        if protected_event_response:
            return _compose_context_only_event_response(
                state,
                styling=_render_context_only_event_styling_decision(
                    rewritten,
                    shopper_texts=_shopper_authored_texts(state),
                ),
                next_question=event_context_next_question,
                weather_outcome=weather_outcome,
                include_weather_facts=not reused_weather_receipt,
            )
        if isinstance(weather_outcome, WeatherFailure):
            rewritten = _strip_weather_fact_sentences(
                _scrub_internal_shopper_language(rewritten)
            )
            if not rewritten.strip():
                return self._grounding_failure_fallback(
                    state,
                    result,
                    request_id=request_id,
                    weather_outcome=fallback_weather_outcome,
                )
            return "\n\n".join(
                (
                    rewritten,
                    _conditional_weather_styling_direction(),
                    _format_weather_outcome(weather_outcome),
                )
            )
        if enforce_event_context and has_current_search_candidates and not (
            _response_mentions_search_candidate(
                rewritten,
                result,
                request_id=request_id,
            )
        ):
            candidate_fallback = self._rewrite_search_only_response(
                state,
                result,
                request_id=request_id,
            )
            if search_only:
                return candidate_fallback
            rewritten = _scrub_internal_shopper_language(
                f"{candidate_fallback}\n\n{rewritten.strip()}"
            )
        rewritten = _scrub_internal_shopper_language(rewritten)
        if isinstance(weather_outcome, WeatherForecastEvidence):
            rewritten = _strip_weather_fact_sentences(rewritten)
            if not has_current_non_weather_business_activity:
                rewritten = _ensure_prior_weather_candidates(
                    rewritten,
                    state.historical_product_context,
                )
            if not rewritten.strip():
                return self._grounding_failure_fallback(
                    state,
                    result,
                    request_id=request_id,
                    weather_outcome=fallback_weather_outcome,
                )
            if not reused_weather_receipt:
                rewritten = _ensure_weather_forecast_evidence(
                    rewritten,
                    weather_outcome,
                )
        return rewritten

    def _rewrite_search_only_response(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
    ) -> str:
        """Return the deterministic fallback for a search-only turn."""

        shopper_guidance = _search_guidance_evidence(
            result,
            request_id=request_id,
        )
        return self._build_search_only_response(
            state,
            result,
            request_id=request_id,
            intro=(
                "\n".join(shopper_guidance)
                or self._active_skill_response_guidance(state)
            ),
        )

    def _grounding_failure_fallback(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
        weather_outcome: WeatherForecastEvidence | WeatherFailure | None,
    ) -> str:
        """Compose deterministic catalog and weather evidence on editor failure."""

        parts: list[str] = []
        detail_fallback = _format_product_detail_fallback(
            result,
            request_id=request_id,
        )
        if detail_fallback:
            parts.append(detail_fallback)
        elif _search_result_groups(result, request_id=request_id):
            parts.append(
                self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            )
        elif weather_outcome is not None:
            prior_candidates = _historical_product_names(
                state.historical_product_context
            )
            if prior_candidates:
                parts.append(_prior_weather_candidates_text(prior_candidates))
        if weather_outcome is not None:
            if isinstance(weather_outcome, WeatherFailure):
                parts.append(_conditional_weather_styling_direction())
            parts.append(_format_weather_outcome(weather_outcome))
        return (
            _scrub_internal_shopper_language("\n\n".join(parts))
            if parts
            else _GROUNDING_FAILURE_RESPONSE
        )

    def _active_skill_response_guidance(self, state: State) -> str:
        active_paths = set(
            state.agent_diagnostics.get("skill_files_read", [])
        )
        registry = _shopper_skill_registry(self._shopper_skills_root())
        guidance = [
            skill.response_guidance
            for skill in registry.values()
            if skill.path in active_paths
        ]
        return "\n".join(guidance)

    @staticmethod
    def _event_context_response_guidance_active(state: State) -> bool:
        return "/shopper/event-context/SKILL.md" in set(
            state.agent_diagnostics.get("skill_files_read", [])
        )

    @staticmethod
    def _grounding_shopper_location_context(state: State) -> str:
        shopper_context = state.shopper_context
        if shopper_context is not None and shopper_context.zipcode:
            return (
                "A saved ZIP candidate is present. Do not expose its digits. "
                'Event-context may use "usual area or elsewhere" framing only '
                "to confirm event location, never shopping or availability."
            )
        return (
            "No saved ZIP candidate is present. Ask the event destination "
            'directly; never imply a saved, home, or "usual" area.'
        )

    def _build_search_only_response(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
        intro: str = "",
    ) -> str:
        """Return a deterministic facts-only fallback for search evidence."""

        return _format_search_only_response(
            state,
            result,
            request_id=request_id,
            intro=intro,
        )

    def _create_skills_backend(self):
        if _FilesystemBackend is None:
            logger.warning("Deep Agents filesystem backend unavailable; shopper skills disabled.")
            return None
        skills_root = self._shopper_skills_root()
        if skills_root is None:
            logger.warning("Shopper skills root not found; Deep Agents skills disabled.")
            return None
        return _FilesystemBackend(root_dir=skills_root, virtual_mode=True)

    def _shopper_skills_root(self) -> Path | None:
        configured_root = os.environ.get(_SHOPPER_SKILLS_ENV)
        candidates = [Path(configured_root)] if configured_root else []
        candidates.append(Path(__file__).resolve().parents[1] / "skills")

        for candidate in candidates:
            if (candidate / "shopper").is_dir():
                return candidate
        return None

    def _system_prompt(
        self,
        capabilities: CatalogCapabilities,
        *,
        shopper_context: ShopperContext | None = None,
        current_utc_date: CalendarDate | None = None,
    ) -> str:
        catalog_context = format_catalog_capabilities_for_prompt(capabilities)
        current_utc_date = current_utc_date or CalendarDate.fromisoformat(
            time.strftime("%Y-%m-%d", time.gmtime())
        )
        weather_enabled = _weather_lookup_enabled(self.config.weather)
        shopper_context_rules = (
            f"\n{_SHOPPER_CONTEXT_SYSTEM_RULES}\n"
            if shopper_context is not None
            else ""
        )
        prompt = f"""You are a retail shopping assistant for the products advertised by the active catalog.

Use tools for catalog facts and cart actions. Do not invent product names,
prices, availability, materials, care instructions, tax, shipping, stock
status, delivery dates, or cart changes.
PRODUCT_REF and CART_LINE_ID are internal identifiers for tool calls. Do not
show them to shoppers in normal responses.
Do not expose skill names, tool names, entry-mode names, evaluator/judge names,
cache/backend details, structured-field labels, or internal data-layer language
to shoppers. If a detail is unavailable, say it plainly, for example: "I don't
have fabric or care details available for that item."
Do not upgrade shopper assumptions, preference language, or earlier styling
inferences into catalog facts. Separate confirmed product facts from styling
judgment, and keep outfit-wide material or comfort claims item-specific unless
every included piece is supported by tool evidence.
The DURABLE CONVERSATION SUMMARY is semantic continuity only. It cannot
establish exact shopper wording, location/date authority, a product identity or
fact, cart truth, tool evidence or permission, policy, availability, or current
weather. Current shopper text and separately labeled authoritative state win.
Do not group leather, rubber, metal, or generic canvas under "natural fibers";
attribute materials item by item.
Outdoor-practicality claims require exact support: do not say products are
stable on grass or gravel, water-resistant, all-day comfortable, weather-safe,
secure for a full event, or good for outdoor surfaces unless the catalog says
so. With indirect evidence, state only the confirmed construction detail and
keep the styling judgment separate.
Do not convert sole or strap facts into surface guarantees. Rubber sole means
rubber sole; ankle strap means ankle strap. Do not add grass, gravel, weather,
outdoor-surface performance, all-day comfort, maximum breathability, or
best-in-category performance claims unless those claims are directly supported
by product details.
{shopper_context_rules}

Catalog capabilities:
{catalog_context}

Event-weather boundary:
- Current server date is {current_utc_date.isoformat()} UTC. For the shopper's
  exact phrase "<weekday> next week", use the tool's next_week mode plus the
  matching lowercase weekday; the server resolves it to that one day inside
  the next Monday-through-Sunday window. For bare "next week", use next_week
  mode without weekday; the server resolves the full Monday-through-Sunday
  range. Never omit or change a shopper-stated weekday. Resolve other
  unambiguous shopper-stated relative dates against this anchor; otherwise ask
  one date question.
- Weather lookup is {'enabled' if weather_enabled else 'disabled'} for this
  deployment. When disabled, do not call it and do not ask a question solely
  to obtain forecast inputs.
- Weather is an event-context helper only. Call it at most once in a turn and,
  when both weather and product search are needed, call weather first.
- Event context is additive. It may grant weather beside outfit styling, but it
  never removes product tools granted by the primary skill. Use only the tools
  needed for the shopper's current request.
- Tool availability is not a request to use it. When the current shopper turn
  only supplies an event destination, venue, or date requested in the prior
  response, preserve the established candidates and do not call a non-weather
  business tool. If that same turn also asks to compare, refine, replace,
  search, check, or change something, continue the primary or standalone
  skill's normal tool procedure.
- A successful lookup requires a shopper-stated place plus an exact date,
  complete inclusive range, or the bounded next_week mode, including its
  required weekday when the shopper supplied one. Keep the shortest sufficient
  shopper-authored place phrase in `location`. If it is ambiguous, or is an
  abbreviation, `location_query` is required: preserve that exact phrase as its
  first component and append only one or two comma-separated region/country
  qualifiers. For NYC, send `location="NYC"` and
  `location_query="NYC, NY"`. Omit `location_query` only when `location` is
  already sufficiently qualified. Never rewrite the authority phrase. Never
  add a ZIP or numeric component the shopper did not state or substitute the
  saved profile location. The provider resolves the query; the final response
  states the resolved place as a transparent, reversible assumption.
- Never call weather to discover or prompt for missing context. If no
  shopper-authored exact date, complete range, or exact "next week" phrase is
  established, do not invent next_week or any date argument; weather is
  unavailable for that turn.
- Ask only the single next missing fact when it materially changes the next
  recommendation: establish location before date. Do not ask a date question
  for generic advice or when weather would not change the guidance.

Rules:
- Every turn begins with shopper-skill activation. Select the smallest set of
  registered skills that covers the complete current intent, then follow the
  injected full instructions before constructing shopping-tool arguments or a
  final response. Never combine activation and shopping calls in one response.
- Outfit styling and product discovery are alternative primary procedures;
  never activate both. Budget shopping may accompany either only as a modifier
  when the shopper states a budget.
- Event context may accompany outfit styling only. Select it whenever an event
  destination or venue is stated, or when the response would otherwise ask
  about or branch on missing destination or venue context. Generic occasion
  advice is not a reason to omit it. Never use it with product discovery or by
  itself.
- "Usual area," "home area," or saved-location framing is allowed only when the
  current turn includes a server-resolved saved-ZIP block. Without that
  block there is no saved-location candidate: ask the event destination
  directly and never imply that a saved, home, or usual area exists.
- Style-led fashion selection belongs to outfit styling even when the shopper
  asks for one statement or balancing piece. Product discovery is for browse
  and filter intent without styling judgment.
- Keep the primary skill aligned with the active conversation task. An outfit-
  building or styling thread continues to use outfit styling for piece-by-piece
  searches until the shopper changes tasks; do not reclassify a follow-up as
  product discovery merely because it asks for one product type.
{CATALOG_SEARCH_RULES}
- Put every non-taxonomy shopper must-have in `required_constraints`, including
  requirements whose fields are semantic/detail-only or absent from Catalog
  capabilities. Do not omit an unsupported must-have or rely on semantic search
  to enforce it; deterministic validation must refuse that search instead of
  weakening it. Preserve performance requirements such as water resistance or
  machine washability here as well. An attribute that defines the requested
  products is required even without the words "must have." Semantic relevance cannot guarantee
  a must-have requirement. Recommendation adjectives such as comfortable,
  relaxed, soft, breathable, lightweight, casual, dressy, bold, bright,
  vibrant, or sporty always remain semantic preferences, never objective hard
  filters. Before every search, compare each modifier on the target product
  with the hard filters advertised in Catalog capabilities and copy every exact matching value into
  `required_constraints`; never leave the object empty when one applies. Use only advertised
  filter properties directly. For "Do you have
  water-resistant bags?", include
  `{{"unadvertised_requirements": ["water resistance"]}}`; do not send an
  empty object.
- Apply hard constraints only to the target products named in the current turn.
  An anchor's confirmed color, material, or other attribute is styling context,
  not a hard filter for a complementary role, unless the shopper explicitly asks
  for the same value or palette.
- For required constraints advertised as hard filters, enum values must exactly
  match listed values and numeric values use an object with `min` and/or `max`.
  When one shopper constraint includes multiple applicable advertised enum
  values, include all of them in one list and one search rather than trying one
  value at a time.
- Media-only or descriptive media requests such as "what's in this look",
  "describe this outfit", "what am I wearing", or "what colors are here" must
  be answered from MEDIA ANALYSIS. Do not call search_catalog_tool and do not
  show catalog products unless the shopper explicitly asks to find, shop,
  recommend, compare, price-check, check availability, or add an item.
- Use at most {self.config.max_catalog_searches_per_turn} catalog search calls per
  user turn. One normalized taxonomy-and-required-constraint scope can execute
  only once in a turn; do not retry the same hard-filter scope with different
  semantic wording. For outfit requests
  with multiple required item types, run one focused search per distinct
  taxonomy scope, then stop and synthesize from those results.
- An outfit request with a season, weather need, occasion, or style/vibe already
  has enough direction to begin with a grounded partial outfit. Do not answer
  only with a questionnaire; search the most useful core role first and ask at
  most one concise follow-up while presenting the grounded result. If the
  shopper explicitly asks to plan before seeing products, do not search yet.
  When event-context identifies a material missing destination or venue in that
  plan-first turn, its compact response gate overrides the general styling
  response: exactly two short sentences, no heading or list, ending with its
  single destination-or-venue confirmation question. When the needed event
  context is already established, follow its one-short-paragraph boundary and
  ask no further destination-or-venue question. Once the shopper states an
  explicit event destination, it overrides saved ZIP and usual-area framing is
  forbidden. Treat an explicitly established venue setting that covers the
  relevant event portions as complete. Do not re-ask it as a finer variant or
  invent hypothetical exceptions. Do not add geographic condition language
  such as terrain performance, salt-air, breeze, heat, or weather suitability.
  For an ordinary shop-now event request, show the grounded core role first and
  ask only one location question alongside it when event location remains
  missing and material; defer venue setting until the next recommendation
  needs it.
- An unspecified request for one style-led piece, such as a statement piece,
  does not identify a product role. Ask one concise category or occasion question
  before searching. This does not apply to an outfit or complete-look request,
  where the named vibe, occasion, season, or weather need is enough to begin.
- A whole or complete outfit remains incomplete until current or directly
  referenced prior evidence covers multiple complementary roles, or the search
  cap is reached and the missing role is disclosed. A one-piece dress may be the
  clothing core, but does not by itself complete an outfit request.
- For a broad style/vibe request, select a useful core role from exact taxonomy
  values currently advertised by the catalog. Do not invent an unadvertised
  product type from the vibe or copy a generic styling example into taxonomy.
- Treat broad weather or occasion context as styling direction, not automatically
  as a product-attribute guarantee. A "rainy day outfit" or "wet-weather outfit"
  should search practical
  advertised roles without adding an unadvertised requirement; add water
  resistance to `unadvertised_requirements` only when the shopper directly
  requires it for a product. "Rainy day outfit" does not imply water resistance;
  "water-resistant bags" directly requires it.
- Subjective style/vibe language is semantic direction unless the shopper makes
  it an explicit hard requirement. Objective product attributes such as material,
  weather performance, or a specific shade remain must-haves when they define
  the requested product.
- For alternatives joined by "or", search the faithful advertised branch and
  preserve every named branch. If another branch cannot be mapped faithfully,
  present the supported result and ask one concise clarification before any
  adjacent search. Do not reject a supported branch merely because another
  alternative is unresolved, and never list the supported taxonomy value in
  `unadvertised_requirements`.
- When every named alternative is advertised, include all of them in one call.
  Do not narrow an explicit umbrella or alternatives to one convenient type.
- Use at most {self.config.max_product_detail_reads_per_turn} product-detail
  reads per user turn. Product details are for direct product fact questions,
  cart or comparison follow-ups, or already-shortlisted items; they are not
  required for an initial no-anchor outfit recommendation.
- A comparison of established products is part of the active primary procedure,
  not a new catalog search. If any compared product is only in the historical
  index, resolve every compared product together in the turn's single batched
  resolver call. Then call product details once per uniquely resolved ref, in
  separate model steps, before comparing. A missing or ambiguous required
  product needs one concise clarification; do not guess, substitute, or search
  again. Weather may add current event evidence but never replaces product
  resolution or detail reads.
- If a tool returns STOP_TOOL_USE, stop tool calling immediately and produce the
  best concise shopper-facing answer from the evidence already available.
- A tool result is enough to produce a final answer. Once you have at least one
  plausible product for each required item type, answer from those results. Do
  not keep searching for alternatives unless the shopper explicitly rejects the
  current result.
- A request for one product role gets one inclusive search scope containing all
  faithful advertised types for that role. Do not use remaining search budget
  on adjacent categories, one-piece substitutes, or unrelated product types.
  A dress is not a bottom and does not satisfy a request for separates.
- In an active styling thread, a group of recent candidates can be the direct
  antecedent. If they share a confirmed constraint that is sufficient for the
  next request, use it as the provisional anchor and search the requested role.
  Do not require one exact product selection or an occasion when the shopper is
  asking for generally compatible options and the shared anchor is sufficient.
- In the final response to a follow-up search, explicitly connect the new
  candidates to the named antecedent or to that candidate group's shared
  confirmed constraint. Do not return an unexplained product list.
- Product-detail or research questions about a product already returned by
  search_catalog_tool should use get_product_details_tool with that
  PRODUCT_REF. Do not run another broad catalog search for known-product facts.
- Initial recommendations should use product name, price, category or role,
  and one styling reason. Do not enumerate materials, dimensions, pockets,
  closures, care, or construction details unless the shopper asks for those
  details and you have called get_product_details_tool.
- For search-only recommendations, keep every product line to name, price,
  category/role, image availability when useful, and exact confirmed filters.
  Put the styling reason in a separate sentence based on role and shopper
  context; never derive it by interpreting words in the display name.
- A successful search may report confirmed filters. Every returned product
  passed each reported predicate. A single allowed value confirms that value;
  multiple allowed values confirm membership in the set, not which value each
  product has. Do not infer an adjacent attribute from a confirmed filter.
- Search-only product names are display names, not confirmed attributes. Do not
  parse length, color, print, material, construction, fit, care, or formality
  from descriptive names unless product details confirm the attribute. You may
  say "candidate" or "could be worth checking" and offer to pull details.
- Do not make group-level claims such as "all are maxi length", "both are
  cotton", "the lightest", "most polished", or "best for heat" unless every
  item in the group has product-detail evidence supporting that exact claim.
- For no-anchor outfit building, do not call product details just to make the
  outfit sound richer. Search by the needed item roles, choose a coherent set,
  and keep the rationale to color, proportion, formality, silhouette, and
  shopper goal.
- If the shopper mentions outdoor practicality in a broad outfit request,
  prefer searched categories that naturally fit the situation, such as flat
  shoes or a light layer, but do not state product material, breathability,
  ground stability, outdoor-surface performance, heat performance, or
  all-evening comfort unless the shopper asks a direct product-specific
  question and details support it.
- Product comparison tables, material claims, dimensions, pocket/closure
  details, care/washability answers, comfort claims, and outdoor-practicality
  claims require get_product_details_tool for each relevant PRODUCT_REF before
  finalizing the answer. If you have only search results, keep the answer to
  names, prices, and brief candidate fit.
- Even after product details, compare only confirmed construction facts for
  surface or weather concerns: lower heel versus higher heel, strap versus no
  strap, rubber sole versus unspecified sole, zip closure versus open top. Do
  not state the resulting performance on grass, gravel, rain, bugs, spills, or
  outdoor ground unless product details explicitly state it.
- Shopper wording is not product evidence. If the shopper mentions an
  unverified attribute such as heel shape, material, colorway, fit, or care,
  verify it with tools or refer to it as the shopper's preference, not as a
  catalog fact.
- When the shopper asks to add an item that has not already been searched in
  this conversation, call search_catalog_tool first, then call
  add_cart_items_tool with a one-item list containing the selected PRODUCT_REF.
- If an image is attached, the current image is already available to
  search_catalog_tool. Use that tool for "this", "similar", and image-price
  refinement requests.
- If MEDIA ANALYSIS is present, use it as the visual/video understanding of
  the attached media. It can guide search_catalog_tool queries and follow-up
  pronoun resolution, but catalog results remain the source of truth for
  product names and prices. Catalog results are not inventory evidence.
- If MEDIA ANALYSIS says media analysis failed, VLM authentication failed, the
  VLM is unavailable, or video understanding is not configured, say so plainly.
  Do not infer video-similar products from the media; ask the shopper for a
  text description or search only from explicit text in the shopper request.
  If an image is attached, image embedding search through search_catalog_tool is
  still available even when MEDIA ANALYSIS is unavailable.
- Cart reads require get_cart_tool. Cart totals require view_cart_total_tool.
- Use recent discussion, not CURRENT CART, to resolve ordinary product and
  styling references such as "that" and "those." A discussed anchor does not
  need to be in the cart for styling advice. Mention that an item is absent from
  the cart only when the shopper asks about cart contents or a cart mutation.
- Cart mutations require explicit shopper intent and must use
  add_cart_items_tool, remove_cart_item_tool, or update_cart_items_tool. Never
  claim a cart mutation unless the tool reports success.
- Cart mutation scope must match the shopper's explicit add or remove request.
  Selection, approval, or styling preference is not cart intent by itself.
  If the shopper asks to "add those", add only the items named in that add
  request or its direct antecedent. Do not add earlier anchor, core outfit, or
  optional pieces unless the shopper explicitly includes them in the cart
  request.
- For an explicit cart swap, finish the whole swap before the final response:
  remove the rejected cart line, add the selected replacement when a valid
  PRODUCT_REF is already available, then summarize the updated cart. If the
  replacement is from an earlier turn, resolve it first. Search only for a new
  replacement that has not already been presented.
- If cart mutation scope is ambiguous, ask one concise clarification before
  calling any cart mutation tool. Example: "Do you want me to add just the bag,
  layer, and earrings, or the full outfit including the dress and sandals?"
- For cart styling requests, inspect CURRENT CART or call get_cart_tool first.
  Do not search for products already named as cart contents just to verify them.
  If the cart is empty but the shopper names items, say you do not see those
  items in the cart yet, then give provisional styling advice from the named
  items without claiming cart truth. Search at most once for a missing piece
  only after identifying the gap.
- Use PRODUCT_REF established by current-turn search or
  resolve_conversation_products_tool when adding items. Do not pass display
  names as product_ref values to add_cart_items_tool. Include
  expected_display_name for each item so the tool can verify that the selected
  PRODUCT_REF resolves to the shopper-facing product name you intend to add.
- When the shopper asks to add multiple selected products, call
  add_cart_items_tool once with an item list. The tool may report partial
  success; the final answer must clearly distinguish added items from failures.
- Use PRODUCT_REF established by current-turn search or historical-product
  resolution when requesting product details. Do not pass display names to
  get_product_details_tool.
- Use internal identifiers only in tool calls. Do not expose PRODUCT_REF or
  CART_LINE_ID in customer-facing responses.
- Use CART_LINE_ID from CURRENT CART or get_cart_tool when removing an item. Do
  not guess cart line IDs from product names.
- Use update_cart_items_tool for quantity changes. Set quantity to zero only
  when the shopper explicitly asks to remove that line.
- Store policy questions about returns, shipping, sizing, payment, price
  matching, or gift cards require get_store_policy_tool. Never substitute model
  knowledge for policy content that the tool does not return.
- Explicit stock, inventory, or size availability questions
  require check_product_availability_tool with a PRODUCT_REF from a prior
  search. Relay its deterministic result rather than guessing from catalog
  presence.
- Explicit sale, discount, or promotion questions require
  check_active_promotions_tool. Catalog results and prices cannot establish sale
  status. If no promotion is active and sale status is required, do not search
  regular-price products without the shopper's agreement; continue any separate
  requested work from the same turn.
- If the shopper asks for anything under a budget without a product type,
  category, occasion, style, outfit goal, or image, ask one concise clarifying
  question instead of guessing.
- Tax and delivery dates are not available through the current tools. Do not
  treat a catalog result alone as proof that an item is in stock or ready to
  ship; availability claims require check_product_availability_tool.
- Previous preference context is guidance only. The current shopper request
  wins when it conflicts with previous preferences.
- Keep final answers concise and grounded in tool results. Attribute materials,
  comfort, construction, and outdoor-practicality claims to the specific items
  that support them instead of making unsupported whole-outfit claims. Avoid
  guarantee language such as "will stay comfortable all evening" unless the
  catalog evidence supports the guarantee. Before finalizing, remove or soften
  unsupported phrases about grass, gravel, water resistance, all-day comfort,
  maximum breathability, or best-in-category performance.
"""
        return prompt

    def _build_user_message(self, state: State, identity: RequestIdentity) -> str:
        recent_discussion = _redact_prior_weather_assistant_text(state.context)
        sections = [
            (
                f"REQUEST ID: {identity.request_id}\n"
                f"SESSION ID: {identity.session_id}\n"
                f"CONVERSATION ID: {identity.conversation_id}\n"
                f"CART ID: {identity.cart_id}"
            )
        ]
        shopper_context = _format_shopper_context(state.shopper_context)
        if shopper_context:
            sections.append(shopper_context)
        sections.extend(
            [
                (
                    f"USER QUERY: {state.query}\n"
                    f"IMAGE ATTACHED: {'yes' if state.image else 'no'}"
                ),
                f"MEDIA ATTACHED:\n{_format_media_summary(state.media)}",
                f"MEDIA ANALYSIS:\n{state.media_analysis or '(none)'}",
                f"CURRENT CART:\n{_format_cart(state.cart)}",
                (
                    "DURABLE CONVERSATION SUMMARY "
                    "(semantic continuity only; not evidence):\n"
                    f"{state.conversation_summary or '(none)'}"
                ),
                f"RECENT DISCUSSION:\n{recent_discussion or '(none)'}",
                (
                    state.historical_product_context
                    or "HISTORICAL PRODUCT INDEX (read-only):\n(none)"
                ),
                _format_active_weather_receipts(
                    state.active_weather_receipts
                ),
            ]
        )
        return "\n\n".join(sections)

    @staticmethod
    def _append_product_images(
        retrieved: dict[str, str],
        cart: Cart,
        products: tuple[ProductSummary, ...],
    ) -> None:
        if not cart.contents or not products:
            return

        products_by_key: dict[str, ProductSummary] = {}
        for product in products:
            products_by_key[product.product_id] = product
            products_by_key[product.display_name] = product

        for item in cart.contents:
            product = products_by_key.get(str(item.get("product_id") or ""))
            if product is None:
                product = products_by_key.get(str(item.get("item") or ""))
            if product is not None and product.image_url:
                retrieved[product.display_name] = product.image_url

    def _read_cart(self, user_id: int) -> Cart:
        result = get_cart(GetCartInput(user_id=str(user_id)), self.config.memory_port)
        if not result.ok or result.cart is None:
            return Cart()
        return Cart(
            contents=[
                {
                    "cart_line_id": line.cart_line_id,
                    "product_id": line.product_id,
                    "item": line.display_name,
                    "amount": line.quantity,
                    "price": line.unit_price.amount if line.unit_price else None,
                }
                for line in result.cart.lines
            ]
        )

    def _start_conversation_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> TurnStartResult | None:
        """Start one durable turn and apply its context/cart snapshot."""

        start = time.monotonic()
        try:
            turn = self._conversation_memory.start_turn(
                identity.conversation_id,
                request_id=identity.request_id,
                shopper_text=state.query,
                media=state.media,
                cart_user_id=identity.cart_user_id,
                shopper_profile_id=identity.shopper_profile_id,
            )
            if (identity.shopper_profile_id is None) != (
                turn.shopper_context is None
            ) or (
                turn.shopper_context is not None
                and not isinstance(turn.shopper_context, ShopperContext)
            ):
                raise ConversationMemoryError(
                    "shopper_context_invalid",
                    "Conversation memory returned mismatched shopper context.",
                )
        except (ConversationMemoryError, ValidationError) as exc:
            logger.error("Failed to start durable conversation turn: %s", exc)
            state.conversation_summary = ""
            state.context = ""
            state.historical_product_context = ""
            state.conversation_projection_version = 0
            state.active_weather_receipts = []
            state.selected_weather_receipt_id = None
            state.weather_receipt_promotion = None
            state.cart = Cart()
            state.shopper_context = None
            error_code = getattr(exc, "code", "memory_start_payload_invalid")
            if error_code == "shopper_profile_not_found":
                state.response = _SHOPPER_PROFILE_NOT_FOUND_RESPONSE
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            elif error_code == "conversation_profile_mismatch":
                state.response = _CONVERSATION_PROFILE_MISMATCH_RESPONSE
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            elif getattr(exc, "status_code", None) == 409:
                if error_code in {"turn_in_progress", "conversation_turn_in_progress"}:
                    state.response = (
                        "This conversation is still processing another request. "
                        "Please retry shortly."
                    )
                elif error_code == "turn_abandoned":
                    state.response = (
                        "That earlier request was interrupted. Please retry with "
                        "a new request."
                    )
                elif error_code == "turn_superseded":
                    state.response = (
                        "That interrupted request was superseded by a newer turn. "
                        "Please continue from the latest conversation state."
                    )
                else:
                    state.response = (
                        "That request identifier was already used for different "
                        "input. Please retry with a new request."
                    )
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            else:
                state.response = (
                    "I cannot safely load this conversation right now. "
                    "Please retry shortly."
                )
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "memory_start_failed"
                )
                state.agent_diagnostics["memory_start_error"] = getattr(
                    exc,
                    "code",
                    "memory_start_payload_invalid",
                )
            return None
        finally:
            state.timings["memory"] = time.monotonic() - start

        state.conversation_summary = turn.projection.summary_text
        state.context = format_conversation_context(
            turn.recent_turns,
            max_chars=max(1000, int(self.config.memory_length)),
        )
        state.previous_selected_skill_names = list(
            turn.previous_selected_skill_names
        )
        state.shopper_context = turn.shopper_context
        historical_products = format_historical_product_index(
            turn.projection.product_reference_index
        )
        state.historical_product_context = historical_products
        state.conversation_projection_version = turn.projection.version
        state.active_weather_receipts = list(turn.projection.active_receipts)
        state.cart = Cart(
            contents=[
                item.model_dump(mode="json", exclude_none=True) for item in turn.cart
            ]
        )
        return turn

    def _restore_replayed_turn(
        self,
        state: State,
        turn: TurnStartResult,
    ) -> State:
        """Restore a finalized turn without repeating model or tool work."""

        state.response = turn.assistant_text or (
            "That earlier request did not complete. Please retry with a new request."
        )
        state.response = _scrub_saved_zip_from_response(
            state.response,
            state.shopper_context,
        )
        state.response = _scrub_weather_receipt_ids(state.response, state)
        if turn.output is not None:
            state.product_results = [
                product.model_dump(mode="json")
                for product in turn.output.product_results
            ]
            state.retrieved = dict(turn.output.retrieved)
            state.agent_diagnostics = dict(turn.output.agent_diagnostics)
            state.selected_skill_names = list(turn.output.selected_skill_names)
        else:
            state.agent_diagnostics = _empty_agent_diagnostics(
                turn.termination_reason or "durable_turn_replayed"
            )
        return state

    def _finalize_conversation_turn(
        self,
        state: State,
        identity: RequestIdentity,
        turn: TurnStartResult,
        *,
        status: FinalTurnStatus | None = None,
        termination_reason: str | None = None,
        present_products: bool = True,
        summary_advance: ConversationSummaryAdvance | None = None,
    ) -> bool:
        """Persist one terminal turn without changing its shopper response."""

        state.response = _scrub_saved_zip_from_response(
            state.response,
            state.shopper_context,
        )
        reason = termination_reason or str(
            state.agent_diagnostics.get("final_termination_reason") or "completed"
        )
        final_status = status or _conversation_turn_status(reason)
        state.agent_diagnostics["final_termination_reason"] = reason
        receipt_promotion = (
            state.weather_receipt_promotion
            if final_status == "completed" and reason == "completed"
            else None
        )
        start = time.monotonic()
        finalized = False

        def finalize_once(
            advance: ConversationSummaryAdvance | None,
            promotion: WeatherReceiptPromotion | None,
        ) -> None:
            output = TurnReplayOutput(
                product_results=(state.product_results if present_products else []),
                retrieved=(state.retrieved if present_products else {}),
                agent_diagnostics=state.agent_diagnostics,
                selected_skill_names=state.selected_skill_names,
            )
            self._conversation_memory.finalize_turn(
                identity.conversation_id,
                turn.turn_id,
                request_id=identity.request_id,
                attempt_id=turn.attempt_id,
                assistant_text=state.response,
                status=final_status,
                termination_reason=reason,
                output=output,
                summary_advance=advance,
                weather_receipt_promotion=promotion,
            )

        try:
            finalize_once(summary_advance, receipt_promotion)
            finalized = True
        except (ConversationMemoryError, ValidationError) as exc:
            error_code = getattr(
                exc,
                "code",
                "memory_finalize_payload_invalid",
            )
            if (
                summary_advance is not None or receipt_promotion is not None
            ) and error_code in {
                "projection_version_conflict",
                "summary_boundary_conflict",
                "weather_receipt_status_conflict",
                "weather_receipt_stale",
            }:
                if summary_advance is not None:
                    state.agent_diagnostics[
                        "conversation_summary_compaction"
                    ] = "conflict_raw_retained"
                if receipt_promotion is not None:
                    state.agent_diagnostics["weather_receipt_status"] = (
                        "promotion_dropped"
                    )
                try:
                    finalize_once(None, None)
                    finalized = True
                except (ConversationMemoryError, ValidationError) as retry_exc:
                    exc = retry_exc
                    error_code = getattr(
                        retry_exc,
                        "code",
                        "memory_finalize_payload_invalid",
                    )
            if not finalized:
                logger.error(
                    "Failed to finalize durable conversation turn: %s",
                    exc,
                )
                state.agent_diagnostics["memory_finalize_error"] = error_code
                if error_code == "turn_attempt_superseded":
                    state.response = (
                        "This request was superseded by a newer attempt. "
                        "Please use the latest response."
                    )
                    state.product_results = []
                    state.retrieved = {}
                    state.agent_diagnostics["final_termination_reason"] = error_code
        finally:
            state.timings["memory"] = state.timings.get("memory", 0.0) + (
                time.monotonic() - start
            )
        return finalized

    def _check_safety(self, mode: str, user_id: int, text: str) -> tuple[bool, bool]:
        endpoint = "input" if mode == "input" else "output"
        try:
            response = requests.post(
                f"{self.config.rails_port}/rail/{endpoint}/check",
                json={"user_id": user_id, "query": text},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Guardrails %s check failed: %s", mode, exc)
            return True, False

        responses = payload.get("response") or []
        if not responses:
            return True, True
        return responses[0].get("content") == text, True


def create_request_identity(
    *,
    legacy_user_id: int,
    session_id: str | None = None,
    conversation_id: str | None = None,
    cart_id: str | None = None,
    request_id: str | None = None,
    shopper_profile_id: str | None = None,
) -> RequestIdentity:
    """Create scoped request identity while preserving legacy user_id behavior."""

    session = session_id or f"legacy-session-{legacy_user_id}"
    conversation = conversation_id or f"legacy-conversation-{legacy_user_id}"
    cart = cart_id or f"legacy-cart-{legacy_user_id}"
    return RequestIdentity(
        session_id=session,
        conversation_id=conversation,
        cart_id=cart,
        context_user_id=(
            _stable_numeric_id("conversation", conversation_id)
            if conversation_id
            else legacy_user_id
        ),
        cart_user_id=_stable_numeric_id("cart", cart_id) if cart_id else legacy_user_id,
        request_id=request_id or str(uuid.uuid4()),
        shopper_profile_id=shopper_profile_id,
    )


def _stable_numeric_id(namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _empty_agent_diagnostics(final_termination_reason: str) -> dict[str, Any]:
    return {
        "skill_files_read": [],
        "tool_calls": [],
        "rejected_tool_calls": [],
        "duplicate_tool_calls": [],
        "product_evidence": [],
        "product_evidence_truncated": False,
        "catalog_scope_outcomes": [],
        "final_termination_reason": final_termination_reason,
        "partial_graph_messages": [],
    }


async def _partial_graph_messages(
    agent: Any,
    invoke_config: dict[str, Any],
) -> tuple[list[Any], str | None]:
    """Read the last graph state before its failed checkpoint is deleted."""

    get_state = getattr(agent, "aget_state", None)
    if get_state is None:
        return [], "state_snapshot_unavailable"
    try:
        snapshot = await asyncio.wait_for(
            get_state(invoke_config),
            timeout=_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("Timed out snapshotting Deep Agents state before cleanup")
        return [], "state_snapshot_timeout"
    except Exception as exc:  # noqa: BLE001 - diagnostics cannot block cleanup.
        error_type = type(exc).__name__
        logger.warning(
            "Could not snapshot Deep Agents state before cleanup: %s",
            error_type,
        )
        return [], error_type

    values = _value(snapshot, "values")
    messages = _value(values, "messages")
    return (messages if isinstance(messages, list) else []), None


def _collect_agent_diagnostics(
    messages: list[Any],
    *,
    request_id: str,
    final_termination_reason: str,
    preserve_partial_messages: bool = False,
    saved_zipcode: str | None = None,
) -> dict[str, Any]:
    """Collect current-turn skill, tool, and termination diagnostics."""

    diagnostics = _empty_agent_diagnostics(final_termination_reason)
    turn_messages = _current_turn_messages(messages, request_id)
    tool_results = _tool_results_by_call_id(turn_messages)
    skill_files_read: list[str] = []
    successful_product_tool_calls: dict[str, str] = {}

    for message in turn_messages:
        if _message_type(message) != "ai":
            continue
        calls = [
            (raw_call, None)
            for raw_call in (_value(message, "tool_calls") or [])
        ]
        additional_kwargs = _value(message, "additional_kwargs") or {}
        restored_fields_by_call: dict[str, list[str]] = {}
        if isinstance(additional_kwargs, dict):
            restored_calls = additional_kwargs.get(
                SERVER_RESTORED_TOOL_CALL_FIELDS
            )
            if isinstance(restored_calls, list):
                for restored_call in restored_calls[:8]:
                    tool_call_id = str(
                        _value(restored_call, "tool_call_id") or ""
                    )
                    fields = _value(restored_call, "fields")
                    if not tool_call_id or not isinstance(fields, list):
                        continue
                    restored_fields_by_call[tool_call_id] = [
                        str(field)[:64] for field in fields[:8]
                    ]
            calls.extend(
                (raw_call, str(_value(raw_call, "rejection_reason") or "rejected"))
                for raw_call in (
                    additional_kwargs.get(_SERVER_REJECTED_TOOL_CALLS) or []
                )
            )
        for raw_call, forced_rejection_reason in calls:
            call = _normalized_tool_call(raw_call)
            sequence = len(diagnostics["tool_calls"]) + 1
            result_message = tool_results.get(call["tool_call_id"])
            if forced_rejection_reason:
                status, rejection_reason = "rejected", forced_rejection_reason
            else:
                status, rejection_reason = _tool_call_status(
                    call["tool_name"],
                    result_message,
                )
            entry = {
                "sequence": sequence,
                "tool_name": call["tool_name"],
                "arguments": _diagnostic_tool_arguments(
                    call["tool_name"],
                    call["arguments"],
                ),
                "status": status,
            }
            if call["tool_name"] == _WEATHER_TOOL_NAME:
                entry["weather"] = _weather_diagnostic_summary(
                    call["arguments"],
                    result_message,
                )
            if rejection_reason:
                entry["rejection_reason"] = rejection_reason
            restored_fields = restored_fields_by_call.get(call["tool_call_id"])
            if restored_fields:
                entry["restored_fields"] = restored_fields
            if rejection_reason == "duplicate_catalog_scope":
                entry["duplicate"] = True
                diagnostics["duplicate_tool_calls"].append(sequence)
            if status == "rejected":
                diagnostics["rejected_tool_calls"].append(sequence)
            diagnostics["tool_calls"].append(entry)

            if (
                status == "completed"
                and call["tool_call_id"]
                and call["tool_name"]
                in {"search_catalog_tool", "get_product_details_tool"}
            ):
                successful_product_tool_calls[call["tool_call_id"]] = call[
                    "tool_name"
                ]

            for skill_path in _skill_file_paths(call, status):
                if skill_path not in skill_files_read:
                    skill_files_read.append(skill_path)

    diagnostics["skill_files_read"] = skill_files_read
    product_evidence, product_evidence_truncated = _diagnostic_product_evidence(
        turn_messages,
        successful_product_tool_calls,
    )
    diagnostics["product_evidence"] = product_evidence
    diagnostics["product_evidence_truncated"] = product_evidence_truncated
    diagnostics["catalog_scope_outcomes"] = _diagnostic_catalog_scope_outcomes(
        turn_messages
    )
    if preserve_partial_messages:
        partial, truncated = _serialize_partial_graph_messages(turn_messages)
        diagnostics["partial_graph_messages"] = partial
        if truncated:
            diagnostics["partial_graph_messages_truncated"] = True
    return _redact_saved_zip_from_diagnostics(diagnostics, saved_zipcode)


def _safe_collect_agent_diagnostics(
    messages: list[Any],
    *,
    request_id: str,
    final_termination_reason: str,
    preserve_partial_messages: bool = False,
    saved_zipcode: str | None = None,
) -> dict[str, Any]:
    """Collect diagnostics without allowing tracing to change turn behavior."""

    try:
        return _collect_agent_diagnostics(
            messages,
            request_id=request_id,
            final_termination_reason=final_termination_reason,
            preserve_partial_messages=preserve_partial_messages,
            saved_zipcode=saved_zipcode,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must fail independently.
        error_type = type(exc).__name__
        logger.warning("Could not collect Deep Agents diagnostics: %s", error_type)
        diagnostics = _empty_agent_diagnostics(final_termination_reason)
        diagnostics["diagnostic_collection_error"] = error_type
        return diagnostics


def _current_turn_messages(messages: list[Any], request_id: str) -> list[Any]:
    marker = f"REQUEST ID: {request_id}"
    start: int | None = None
    for index, message in enumerate(messages):
        if _message_type(message) != "human":
            continue
        if marker in _content_to_text(_value(message, "content")):
            start = index + 1
    return [] if start is None else messages[start:]


def _prior_turn_messages(messages: list[Any], request_id: str) -> list[Any]:
    """Return messages before the current server-owned request marker."""

    marker = f"REQUEST ID: {request_id}"
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if _message_type(message) != "human":
            continue
        if marker in _content_to_text(_value(message, "content")):
            return messages[:index]
    return []


def _no_direct_taxonomy_response(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Return the fixed shopper response for a current-turn no-match result."""

    outcomes = _business_tool_result_contents(
        _current_turn_messages(_result_messages(result), request_id)
    )
    no_direct_outcomes = [
        content
        for content in outcomes
        if content.startswith(
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy"
        )
    ]
    repair_or_no_direct = all(
        content.startswith(
            (
                SEARCH_VALIDATION_ERROR_PREFIX,
                "STOP_TOOL_USE: No faithful advertised catalog taxonomy",
            )
        )
        for content in outcomes
    )
    if no_direct_outcomes and repair_or_no_direct:
        return _NO_DIRECT_TAXONOMY_RESPONSE
    return None


def _rejected_catalog_search_response(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Fail closed when every current-turn business call is a rejected search."""

    messages = _current_turn_messages(_result_messages(result), request_id)
    tool_results = _tool_results_by_call_id(messages)
    business_calls: list[tuple[str, str]] = []
    for message in messages:
        if _message_type(message) != "ai":
            continue
        calls = [
            (raw_call, False)
            for raw_call in (_value(message, "tool_calls") or [])
        ]
        additional_kwargs = _value(message, "additional_kwargs") or {}
        if isinstance(additional_kwargs, dict):
            calls.extend(
                (raw_call, True)
                for raw_call in (
                    additional_kwargs.get(_SERVER_REJECTED_TOOL_CALLS) or []
                )
            )
        for raw_call, server_rejected in calls:
            call = _normalized_tool_call(raw_call)
            tool_name = call["tool_name"]
            if tool_name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}:
                continue
            status = (
                "rejected"
                if server_rejected
                else _tool_call_status(
                    tool_name,
                    tool_results.get(call["tool_call_id"]),
                )[0]
            )
            business_calls.append((tool_name, status))

    if not business_calls:
        return None
    if _catalog_repair_clarification_response(
        result,
        request_id=request_id,
    ):
        return None
    if all(
        tool_name == "search_catalog_tool" and status == "rejected"
        for tool_name, status in business_calls
    ):
        return _REJECTED_CATALOG_SEARCH_RESPONSE
    return None


def _catalog_repair_clarification_response(
    result: Any,
    *,
    request_id: str,
) -> str:
    """Return a fixed response for a server-marked no-tool repair branch."""

    messages = _current_turn_messages(_result_messages(result), request_id)
    for message in reversed(messages):
        if _message_type(message) != "ai" or _value(message, "tool_calls"):
            continue
        additional_kwargs = _value(message, "additional_kwargs") or {}
        if not isinstance(additional_kwargs, dict) or not additional_kwargs.get(
            SERVER_CATALOG_CLARIFICATION
        ):
            continue
        return _CATALOG_REPAIR_CLARIFICATION_RESPONSE
    return ""


def _unsupported_requirement_response(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Return the fixed shopper response for an unenforceable must-have."""

    outcomes = _business_tool_result_contents(
        _current_turn_messages(_result_messages(result), request_id)
    )
    unsupported_outcomes = [
        content
        for content in outcomes
        if content.startswith(
            "The requested catalog requirement cannot be enforced:"
        )
    ]
    if unsupported_outcomes and len(unsupported_outcomes) == len(outcomes):
        return _UNSUPPORTED_REQUIREMENT_RESPONSE
    return None


def _business_tool_result_contents(messages: list[Any]) -> list[str]:
    """Return non-activation tool outcomes in graph order."""

    outcomes: list[str] = []
    for message in messages:
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"} or content.startswith(
            (
                SKILL_ACTIVATION_COMPLETE,
                SKILL_ACTIVATION_REQUIRED,
                SKILL_TOOL_NOT_GRANTED,
                "SHOPPER_SKILL_ACTIVATION_FAILED:",
            )
        ):
            continue
        outcomes.append(content)
    return outcomes


def _has_unsupported_requirement_outcome(
    result: Any,
    *,
    request_id: str,
) -> bool:
    """Return whether the current turn contains an unenforceable requirement."""

    return any(
        _message_type(message) == "tool"
        and _content_to_text(_value(message, "content")).startswith(
            "The requested catalog requirement cannot be enforced:"
        )
        for message in _current_turn_messages(
            _result_messages(result),
            request_id,
        )
    )


def _tool_results_by_call_id(messages: list[Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in messages:
        if _message_type(message) != "tool":
            continue
        tool_call_id = str(_value(message, "tool_call_id") or "")
        if tool_call_id:
            results[tool_call_id] = message
    return results


def _weather_diagnostic_summary(
    arguments: dict[str, Any],
    result_message: Any | None,
) -> dict[str, str]:
    """Return non-sensitive weather request shape and typed outcome."""

    if arguments.get("relative_date") == "next_week":
        weekday = arguments.get("weekday")
        if weekday in {
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        }:
            request_shape = "relative_exact_date"
        elif weekday is None:
            request_shape = "relative_range"
        else:
            request_shape = "invalid"
    elif arguments.get("date"):
        request_shape = "exact_date"
    elif arguments.get("start_date") and arguments.get("end_date"):
        request_shape = "date_range"
    else:
        request_shape = "invalid"

    location_source = arguments.get("location_source")
    if location_source not in {
        "confirmed_saved_zip",
        "shopper_provided_location",
    }:
        location_source = "invalid"
    if location_source == "confirmed_saved_zip":
        provider_input = "saved_zip"
    elif arguments.get("location_query"):
        provider_input = "location_query"
    elif arguments.get("location"):
        provider_input = "location"
    else:
        provider_input = "invalid"

    outcome = "not_executed"
    if result_message is not None:
        parsed = parse_weather_tool_evidence(
            _content_to_text(_value(result_message, "content"))
        )
        if isinstance(parsed, WeatherForecastEvidence):
            outcome = "success"
        elif isinstance(parsed, WeatherFailure):
            outcome = parsed.code
        else:
            outcome = "weather_response_invalid"
    return {
        "request_shape": request_shape,
        "location_source": str(location_source),
        "provider_input": provider_input,
        "outcome": outcome,
    }


def _normalized_tool_call(raw_call: Any) -> dict[str, Any]:
    arguments = _value(raw_call, "args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"value": arguments}
    return {
        "tool_call_id": str(_value(raw_call, "id") or ""),
        "tool_name": str(_value(raw_call, "name") or "unknown"),
        "arguments": _diagnostic_json_value(arguments),
    }


def _diagnostic_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Remove weather request and durable receipt identities from diagnostics."""

    if tool_name == _WEATHER_TOOL_NAME:
        return dict(_WEATHER_DIAGNOSTIC_REDACTION)
    safe_arguments = dict(arguments)
    if tool_name == SKILL_ACTIVATION_TOOL_NAME:
        safe_arguments.pop("weather_receipt_id", None)
    return safe_arguments


def _tool_call_status(
    tool_name: str,
    result_message: Any | None,
) -> tuple[str, str | None]:
    if result_message is None:
        return "pending", None
    content = _content_to_text(_value(result_message, "content"))
    rejection_reason = _tool_rejection_reason(content)
    if rejection_reason:
        return "rejected", rejection_reason
    if _value(result_message, "status") == "error":
        return "error", None
    if tool_name == "read_file" and content.lower().startswith(
        ("error", "file not found")
    ):
        return "error", None
    return "completed", None


def _tool_rejection_reason(content: str) -> str | None:
    markers = (
        (SKILL_ACTIVATION_REQUIRED, "skill_activation_required"),
        (SKILL_TOOL_NOT_GRANTED, "skill_tool_not_granted"),
        ("SHOPPER_SKILL_ACTIVATION_FAILED:", "skill_activation_failed"),
        (
            "STOP_TOOL_USE: This catalog taxonomy and constraint scope was already searched",
            "duplicate_catalog_scope",
        ),
        ("STOP_TOOL_USE: Catalog search limit reached", "catalog_search_limit"),
        (
            "STOP_TOOL_USE: No faithful advertised catalog taxonomy",
            "no_advertised_taxonomy_match",
        ),
        (
            "STOP_TOOL_USE: Product-detail read limit reached",
            "product_detail_read_limit",
        ),
        (
            "The catalog search request does not match current capabilities:",
            "invalid_catalog_request",
        ),
        (SEARCH_VALIDATION_ERROR_PREFIX, "invalid_catalog_request"),
        (CONSTRAINT_REVIEW_PREFIX, "constraint_review_required"),
        (
            "The requested catalog taxonomy cannot be enforced:",
            "unsupported_catalog_taxonomy",
        ),
        (
            "The requested catalog requirement cannot be enforced:",
            "unsupported_catalog_constraint",
        ),
        (_UNSUPPORTED_SEARCH_MODE_MESSAGE, "unsupported_search_mode"),
    )
    for marker, reason in markers:
        if content.startswith(marker):
            return reason
    if content.startswith("STOP_TOOL_USE:"):
        return "stop_tool_use"
    return None


def _skill_file_paths(call: dict[str, Any], status: str) -> list[str]:
    if status != "completed":
        return []
    arguments = call["arguments"]
    if call["tool_name"] == SKILL_ACTIVATION_TOOL_NAME:
        names = arguments.get("skill_names") or []
        if not isinstance(names, list):
            return []
        return [
            f"/shopper/{name}/SKILL.md"
            for name in names
            if isinstance(name, str) and name.strip()
        ]
    if call["tool_name"] != "read_file":
        return []
    path = str(arguments.get("file_path") or arguments.get("path") or "")
    normalized = path.replace("\\", "/")
    if normalized.startswith("/shopper/") and normalized.endswith("/SKILL.md"):
        return [normalized]
    return []


def _serialize_partial_graph_messages(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], bool]:
    relevant = [
        message for message in messages if _message_type(message) in {"ai", "tool"}
    ]
    contains_weather = any(
        _message_contains_weather_evidence(message)
        for message in relevant
    ) or any(
        _EVENT_CONTEXT_RECEIPT_BOUND_REMINDER
        in _content_to_text(_value(message, "content"))
        for message in relevant
    )
    truncated = len(relevant) > 24
    serialized: list[dict[str, Any]] = []
    for message in relevant[-24:]:
        content = _content_to_text(_value(message, "content"))
        weather_tool_message = _message_contains_weather_evidence(message)
        if weather_tool_message:
            content = _WEATHER_PARTIAL_OUTPUT_REDACTION
        elif contains_weather and _message_type(message) == "ai" and content:
            content = "WEATHER-DERIVED ASSISTANT CONTENT REDACTED"
        content_truncated = len(content) > 2000
        payload: dict[str, Any] = {
            "type": _message_type(message),
            "content": content[:2000],
        }
        for field in ("name", "tool_call_id"):
            value = _value(message, field)
            if value:
                payload[field] = str(value)
        tool_calls = _value(message, "tool_calls")
        if tool_calls:
            normalized_calls = []
            for raw_call in tool_calls:
                call = _normalized_tool_call(raw_call)
                call["arguments"] = _diagnostic_tool_arguments(
                    call["tool_name"],
                    call["arguments"],
                )
                normalized_calls.append(call)
            payload["tool_calls"] = normalized_calls
        if content_truncated:
            payload["truncated"] = True
            truncated = True
        serialized.append(payload)
    return serialized, truncated


def _message_contains_weather_evidence(message: Any) -> bool:
    content = _content_to_text(_value(message, "content"))
    if content.startswith(
        (
            SKILL_ACTIVATION_REQUIRED,
            SKILL_TOOL_NOT_GRANTED,
        )
    ):
        return False
    if str(_value(message, "name") or "") == _WEATHER_TOOL_NAME:
        return True
    if content.startswith(
        (WEATHER_FORECAST_EVIDENCE_PREFIX, WEATHER_FORECAST_FAILURE_PREFIX)
    ):
        return True
    return any(
        str(_value(call, "name") or "") == _WEATHER_TOOL_NAME
        for call in (_value(message, "tool_calls") or [])
    )


def _message_type(message: Any) -> str:
    message_type = str(_value(message, "type") or _value(message, "role") or "")
    return {"assistant": "ai", "user": "human"}.get(message_type, message_type)


def _diagnostic_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): _diagnostic_json_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _redact_saved_zip_from_diagnostics(
    value: dict[str, Any],
    saved_zipcode: str | None,
) -> dict[str, Any]:
    """Remove the server-resolved profile ZIP from every diagnostic field."""

    if not saved_zipcode:
        return value
    pattern = re.compile(
        rf"(?<![0-9]){re.escape(saved_zipcode)}(?![0-9])"
    )

    def scrub(candidate: Any) -> Any:
        if isinstance(candidate, str):
            return pattern.sub("SAVED ZIP REDACTED", candidate)
        if isinstance(candidate, dict):
            return {
                (
                    scrub(key)
                    if isinstance(key, str)
                    else key
                ): scrub(item)
                for key, item in candidate.items()
            }
        if isinstance(candidate, (list, tuple)):
            return [scrub(item) for item in candidate]
        return candidate

    return scrub(value)


def _diagnostic_product_evidence(
    messages: list[Any],
    successful_tool_calls: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    """Return bounded product facts from successful current-turn tool results."""

    evidence: list[dict[str, Any]] = []
    truncated = False
    aggregate_limit_reached = False
    for message in messages:
        if _message_type(message) != "tool":
            continue

        tool_call_id = str(_value(message, "tool_call_id") or "")
        source_tool = successful_tool_calls.get(tool_call_id)
        content = _content_to_text(_value(message, "content"))
        if source_tool == "search_catalog_tool":
            if "SEARCH_RESULT_GROUNDING_NOTE" not in content:
                continue
            evidence_type = "search_result"
            search_scope = {
                "taxonomy": _bounded_product_evidence_value(
                    _search_taxonomy_evidence(content)
                ),
                "confirmed_filters": _bounded_product_evidence_value(
                    _search_filter_evidence(content)
                ),
            }
        elif source_tool == "get_product_details_tool":
            if "PRODUCT_DETAIL_GROUNDING_NOTE" not in content:
                continue
            evidence_type = "product_detail"
            search_scope = None
        else:
            continue

        for product in _product_evidence_records(content):
            product_ref = product.get("product_ref")
            product_name = product.get("name")
            if not product_ref or not product_name:
                continue
            record = {
                "product_ref": _bounded_product_evidence_value(product_ref),
                "product_name": _bounded_product_evidence_value(product_name),
                "source_tool": source_tool,
                "evidence_type": evidence_type,
                "facts": _diagnostic_product_facts(product),
            }
            if search_scope is not None:
                record["search_scope"] = search_scope
            if (
                len(evidence) >= _MAX_DIAGNOSTIC_PRODUCT_EVIDENCE
                or aggregate_limit_reached
            ):
                truncated = True
                continue
            candidate = [*evidence, record]
            if len(json.dumps(candidate, sort_keys=True, default=str)) > (
                _MAX_DIAGNOSTIC_PRODUCT_EVIDENCE_CHARS
            ):
                truncated = True
                aggregate_limit_reached = True
                continue
            evidence.append(record)
    return evidence, truncated


def _diagnostic_catalog_scope_outcomes(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Return bounded server-authored outcomes that contain no products."""

    outcomes: list[dict[str, Any]] = []
    allowed_fields = {
        "outcome",
        "requested_product_type",
        "taxonomy",
        "confirmed_filters",
    }
    for message in messages:
        if _message_type(message) != "tool":
            continue
        outcome = _search_json_evidence(
            _content_to_text(_value(message, "content")),
            _CATALOG_SCOPE_OUTCOME_PREFIX,
        )
        if (
            not outcome
            or not set(outcome).issubset(allowed_fields)
            or outcome.get("outcome")
            not in {"no_direct_catalog_match", "zero_results"}
        ):
            continue
        bounded = _bounded_product_evidence_value(outcome)
        if isinstance(bounded, dict) and bounded not in outcomes:
            outcomes.append(bounded)
        if len(outcomes) >= 8:
            break
    return outcomes


def _diagnostic_product_facts(product: dict[str, Any]) -> dict[str, Any]:
    """Extract bounded, structured facts from one parsed product record."""

    facts: dict[str, Any] = {}
    for key in ("category", "brand", "price"):
        value = product.get(key)
        if value:
            facts[key] = _bounded_product_evidence_value(value)
    facts["image_available"] = bool(product.get("image_url"))

    for raw_detail in product.get("details") or []:
        if len(facts) >= _MAX_DIAGNOSTIC_PRODUCT_FACTS:
            break
        if not isinstance(raw_detail, str) or ":" not in raw_detail:
            continue
        name, value = raw_detail.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        bounded_name = str(_bounded_product_evidence_value(name))
        if bounded_name not in facts:
            facts[bounded_name] = _bounded_product_evidence_value(value)
    return facts


def _bounded_product_evidence_value(value: Any, *, depth: int = 0) -> Any:
    """Bound strings and collections copied into product evidence diagnostics."""

    if isinstance(value, str):
        return value[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if depth >= 4:
        return str(value)[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]
    if isinstance(value, dict):
        return {
            str(key)[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]: (
                _bounded_product_evidence_value(item, depth=depth + 1)
            )
            for key, item in list(value.items())[:_MAX_DIAGNOSTIC_PRODUCT_FACTS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_product_evidence_value(item, depth=depth + 1)
            for item in value[:_MAX_DIAGNOSTIC_PRODUCT_FACTS]
        ]
    return str(value)[:_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS]


def _extract_final_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if _message_type(message) in {"human", "system", "tool"}:
                    continue
                if _value(message, "tool_calls"):
                    continue
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                text = _content_to_text(content)
                if text and not text.startswith(
                    (
                        SKILL_ACTIVATION_COMPLETE,
                        SKILL_ACTIVATION_REQUIRED,
                        SKILL_TOOL_NOT_GRANTED,
                        "SHOPPER_SKILL_ACTIVATION_FAILED:",
                    )
                ):
                    return text
        return _content_to_text(result.get("response")) or ""
    return _content_to_text(getattr(result, "content", result)) or ""


def _append_product_results(state: State, products: list[ProductSummary]) -> None:
    existing_ids = {
        str(product.get("product_id") or "")
        for product in state.product_results
        if isinstance(product, dict)
    }
    for product in products:
        payload = product.model_dump(mode="json")
        product_id = str(payload.get("product_id") or "")
        if product_id and product_id in existing_ids:
            continue
        state.product_results.append(payload)
        if product_id:
            existing_ids.add(product_id)


def _partial_product_results_response(state: State) -> str:
    products = [
        product for product in state.product_results if isinstance(product, dict)
    ]
    if not products:
        return ""

    lines = [
        "I found these grounded catalog options so far:",
        "",
    ]
    seen: set[str] = set()
    for product in products[:8]:
        name = str(product.get("display_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parts = [f"**{name}**"]
        category = str(product.get("category") or "").strip()
        if category:
            parts.append(category)
        price = _product_result_price(product)
        if price:
            parts.append(price)
        lines.append("- " + " — ".join(parts))

    if len(lines) <= 2:
        return ""
    lines.extend(
        [
            "",
            (
                "I do not want to overstate outdoor performance, material, care, "
                "or fit details without a completed detail check. I can continue "
                "from these options or narrow one piece at a time."
            ),
        ]
    )
    return "\n".join(lines)


def _has_search_only_tool_evidence(result: Any, *, request_id: str) -> bool:
    """Return whether current-turn commerce evidence contains only searches."""

    tool_names: list[str] = []
    has_search_result = False
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if not name and "SEARCH_RESULT_GROUNDING_NOTE" in content:
            name = "search_catalog_tool"
        if (
            name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}
            or content.startswith(
                (
                    SKILL_ACTIVATION_COMPLETE,
                    SKILL_ACTIVATION_REQUIRED,
                    SKILL_TOOL_NOT_GRANTED,
                    "SHOPPER_SKILL_ACTIVATION_FAILED:",
                )
            )
        ):
            continue
        tool_names.append(name)
        if name == "search_catalog_tool" and "SEARCH_RESULT_GROUNDING_NOTE" in (
            content
        ):
            has_search_result = True
    return (
        has_search_result
        and set(tool_names) == {"search_catalog_tool"}
    )


def _has_current_non_weather_business_tool_activity(
    result: Any,
    *,
    request_id: str,
) -> bool:
    """Return whether this turn attempted any non-weather business capability."""

    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if name in {SKILL_ACTIVATION_TOOL_NAME, "read_file", _WEATHER_TOOL_NAME}:
            continue
        if content.startswith(
            (
                SKILL_ACTIVATION_COMPLETE,
                SKILL_ACTIVATION_REQUIRED,
                SKILL_TOOL_NOT_GRANTED,
                "SHOPPER_SKILL_ACTIVATION_FAILED:",
                WEATHER_FORECAST_EVIDENCE_PREFIX,
                WEATHER_FORECAST_FAILURE_PREFIX,
            )
        ):
            continue
        return True
    return False


def _current_event_context_next_question(
    result: Any,
    *,
    request_id: str,
) -> str | None:
    """Return the server-accepted event-context question boundary."""

    messages = _current_turn_messages(_result_messages(result), request_id)
    activation_by_call_id: dict[str, str | None] = {}
    for message in messages:
        if _message_type(message) != "ai":
            continue
        for raw_call in _value(message, "tool_calls") or []:
            call = _normalized_tool_call(raw_call)
            if (
                call["tool_name"] != SKILL_ACTIVATION_TOOL_NAME
                or not call["tool_call_id"]
            ):
                continue
            next_question = call["arguments"].get(
                "event_context_next_question"
            )
            accepted_next_question = (
                str(next_question)
                if next_question in {
                    "event_location",
                    "event_venue",
                    "event_date",
                    "none",
                }
                else None
            )
            activation_by_call_id[call["tool_call_id"]] = accepted_next_question

    for message in messages:
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if (
            name == SKILL_ACTIVATION_TOOL_NAME
            and content.startswith(SKILL_ACTIVATION_COMPLETE)
        ):
            return activation_by_call_id.get(
                str(_value(message, "tool_call_id") or ""),
            )
    return None


def _bound_weather_receipt(state: State) -> WeatherForecastReceipt | None:
    """Return only the receipt explicitly selected during this activation."""

    selected_id = state.selected_weather_receipt_id
    if selected_id is None:
        return None
    return next(
        (
            receipt
            for receipt in state.active_weather_receipts
            if receipt.receipt_id == selected_id
        ),
        None,
    )


def _weather_forecast_from_receipt(
    receipt: WeatherForecastReceipt | None,
) -> WeatherForecastEvidence | None:
    """Project a validated durable receipt into the serving evidence type."""

    if receipt is None:
        return None
    try:
        return WeatherForecastEvidence.model_validate(
            receipt.evidence.model_dump(mode="python", exclude_none=True)
        )
    except ValidationError:
        return None


def _effective_weather_outcome(
    state: State,
    result: Any,
    *,
    request_id: str,
) -> tuple[WeatherForecastEvidence | WeatherFailure | None, bool]:
    """Prefer current evidence, otherwise use only the explicitly bound receipt."""

    current = _current_weather_outcome(result, request_id=request_id)
    if current is not None:
        return current, False
    bound = _weather_forecast_from_receipt(_bound_weather_receipt(state))
    return bound, bound is not None


def _current_weather_receipt_promotion(
    state: State,
    result: Any,
    *,
    request_id: str,
    ttl_seconds: int,
) -> WeatherReceiptPromotion | None:
    """Pair one successful current weather call/result for atomic promotion."""

    messages = _current_turn_messages(_result_messages(result), request_id)
    requests_by_call_id: dict[str, EventWeatherRequest] = {}
    for message in messages:
        if _message_type(message) != "ai":
            continue
        for raw_call in _value(message, "tool_calls") or []:
            call = _normalized_tool_call(raw_call)
            call_id = call["tool_call_id"]
            if call["tool_name"] != _WEATHER_TOOL_NAME or not call_id:
                continue
            try:
                requests_by_call_id[call_id] = (
                    EventWeatherRequest.model_validate(call["arguments"])
                )
            except ValidationError:
                continue

    for message in messages:
        if _message_type(message) != "tool":
            continue
        call_id = str(_value(message, "tool_call_id") or "")
        weather_request = requests_by_call_id.get(call_id)
        if weather_request is None:
            continue
        outcome = parse_weather_tool_evidence(
            _content_to_text(_value(message, "content"))
        )
        if not isinstance(outcome, WeatherForecastEvidence):
            continue
        if weather_request.location_source == "confirmed_saved_zip":
            location_scope = SavedAreaWeatherScope()
        else:
            if weather_request.location is None:
                continue
            location_scope = ShopperLocationWeatherScope(
                location=weather_request.location,
                location_query=weather_request.location_query,
            )
        try:
            evidence = WeatherReceiptEvidence.model_validate(
                outcome.model_dump(mode="python", exclude_none=True)
            )
            return WeatherReceiptPromotion(
                expected_projection_version=(
                    state.conversation_projection_version
                ),
                source_tool_call_id=call_id,
                location_scope=location_scope,
                evidence=evidence,
                ttl_seconds=ttl_seconds,
            )
        except ValidationError:
            return None
    return None


def _current_weather_outcome(
    result: Any,
    *,
    request_id: str,
) -> WeatherForecastEvidence | WeatherFailure | None:
    """Return the one validated current-turn weather outcome, if present."""

    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        content = _content_to_text(_value(message, "content"))
        if content.startswith(
            (
                SKILL_ACTIVATION_REQUIRED,
                SKILL_TOOL_NOT_GRANTED,
            )
        ):
            continue
        if name != _WEATHER_TOOL_NAME and not content.startswith(
            (WEATHER_FORECAST_EVIDENCE_PREFIX, WEATHER_FORECAST_FAILURE_PREFIX)
        ):
            continue
        parsed = parse_weather_tool_evidence(content)
        if isinstance(parsed, (WeatherForecastEvidence, WeatherFailure)):
            return parsed
        return weather_failure("weather_response_invalid")
    return None


def _escape_markdown_inline(value: str) -> str:
    """Escape provider-owned text before placing it in Markdown."""

    return re.sub(r"([\\`*_\[\]\(\)<>])", r"\\\1", value)


def _format_weather_outcome(
    outcome: WeatherForecastEvidence | WeatherFailure,
) -> str:
    """Render a deterministic shopper-safe weather success or failure."""

    if isinstance(outcome, WeatherFailure):
        if outcome.code == "weather_outside_forecast_horizon":
            return (
                "A live forecast is not available for those event dates yet, "
                "so I won't assume the conditions."
            )
        if outcome.code in {
            "weather_timeout",
            "weather_rate_limited",
            "weather_unavailable",
        }:
            return (
                "I couldn't retrieve the live forecast right now, so I won't "
                "assume the conditions."
            )
        if outcome.code in {
            "weather_disabled",
            "weather_config_invalid",
            "weather_auth_failed",
        }:
            return (
                "Live weather guidance is unavailable right now, so I won't "
                "assume the event conditions. I can still plan from the stated "
                "destination and venue."
            )
        return (
            "I couldn't establish a valid live forecast for that event place "
            "and date, so I won't assume the conditions."
        )

    lines: list[str] = []
    if outcome.resolved_location:
        lines.append(
            "Forecast location used: "
            f"{_escape_markdown_inline(outcome.resolved_location)}."
        )
    if outcome.relative_date == "next_week":
        start_label = outcome.requested_window.start_date.strftime(
            "%b %d, %Y"
        ).replace(" 0", " ")
        if outcome.weekday is not None:
            lines.append(
                f'Interpreting "{outcome.weekday.title()} next week" as '
                f"{start_label}."
            )
        else:
            end_label = outcome.requested_window.end_date.strftime(
                "%b %d, %Y"
            ).replace(" 0", " ")
            lines.append(
                f'Interpreting "next week" as {start_label} through '
                f"{end_label}."
            )
    if len(outcome.days) == 1:
        lines.append("Live forecast:")
    else:
        lines.append("Live forecast for the event window:")
    for day in outcome.days:
        details = [day.condition.replace("_", " ")]
        if (
            day.temperature_low_f is not None
            and day.temperature_high_f is not None
        ):
            details.append(
                f"{day.temperature_low_f:g}–{day.temperature_high_f:g}°F"
            )
        elif day.temperature_high_f is not None:
            details.append(f"high {day.temperature_high_f:g}°F")
        elif day.temperature_low_f is not None:
            details.append(f"low {day.temperature_low_f:g}°F")
        details.append(
            f"{day.precipitation_probability_pct:g}% precipitation chance"
        )
        if day.precipitation_types:
            details.append(
                "possible "
                + ", ".join(
                    value.replace("_", " ")
                    for value in day.precipitation_types
                )
            )
        date_label = day.date.strftime("%b %d, %Y").replace(" 0", " ")
        lines.append(f"- {date_label}: " + "; ".join(details) + ".")
    lines.append(
        f"[{VISUAL_CROSSING_ATTRIBUTION_LABEL}]"
        f"({VISUAL_CROSSING_ATTRIBUTION_URL}). "
        f"{_WEATHER_UNCERTAINTY_TEXT}"
    )
    return "\n".join(lines)


def _conditional_weather_styling_direction() -> str:
    """Return useful styling guidance without asserting forecast facts."""

    return (
        "Keep the look weather-flexible with a removable layer and, if the "
        "event is outdoors, a footwear backup for wet or uneven ground; "
        "recheck the forecast closer to the event."
    )


def _ensure_weather_forecast_evidence(
    response: str,
    outcome: WeatherForecastEvidence,
) -> str:
    """Append the exact validated forecast block once."""

    canonical = _format_weather_outcome(outcome)
    if canonical in response:
        return response
    if not response.strip():
        return canonical
    return response.rstrip() + "\n\n" + canonical


def _strip_weather_fact_sentences(response: str) -> str:
    """Retain editor styling prose while removing restated forecast facts."""

    retained_lines: list[str] = []
    for raw_line in response.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if retained_lines and retained_lines[-1]:
                retained_lines.append("")
            continue
        if (
            "{" in stripped
            or "}" in stripped
            or re.match(r'^"[^"]{1,64}"\s*:', stripped) is not None
        ):
            continue
        sentences = re.split(r"(?<=[!?])\s+|(?<=\.)\s+(?![0-9])", stripped)
        retained: list[str] = []
        for sentence in sentences:
            if not sentence:
                continue
            if _WEATHER_FACT_LANGUAGE_RE.search(sentence) is None:
                retained.append(sentence)
                continue
            causal_parts = re.split(
                r"\s+(?:because|since|as)\s+",
                sentence,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            if (
                len(causal_parts) == 2
                and causal_parts[0].strip()
                and _WEATHER_FACT_LANGUAGE_RE.search(causal_parts[0]) is None
            ):
                retained.append(causal_parts[0].rstrip(" ,;:") + ".")
                continue
            for clause in re.split(r"[;,]\s+", sentence):
                safe_clause = clause.strip()
                if (
                    safe_clause
                    and _WEATHER_FACT_LANGUAGE_RE.search(safe_clause) is None
                    and re.match(
                        r"^(?:so|therefore|thus|hence)\b",
                        safe_clause,
                        flags=re.IGNORECASE,
                    )
                    is None
                ):
                    retained.append(safe_clause.rstrip(" ,;:.") + ".")
        if retained:
            retained_lines.append(" ".join(retained))
    while retained_lines and not retained_lines[-1]:
        retained_lines.pop()
    return "\n".join(retained_lines).strip()


def _deterministic_weather_styling_direction(
    outcome: WeatherForecastEvidence,
) -> str:
    precipitation_types = {
        precipitation_type
        for day in outcome.days
        for precipitation_type in day.precipitation_types
    }
    conditions = {day.condition for day in outcome.days}
    precipitation_chance = any(
        day.precipitation_probability_pct is not None
        and day.precipitation_probability_pct >= 30
        for day in outcome.days
    )
    low_temperature = min(
        (
            day.temperature_low_f
            for day in outcome.days
            if day.temperature_low_f is not None
        ),
        default=None,
    )
    high_temperature = max(
        (
            day.temperature_high_f
            for day in outcome.days
            if day.temperature_high_f is not None
        ),
        default=None,
    )
    if precipitation_types.intersection({"snow", "freezing_rain", "ice"}) or (
        "snow" in conditions
    ):
        return (
            "Styling direction: keep the established options polished and "
            "flexible with streamlined accessories, an optional outer layer, "
            "and a compact weather backup."
        )
    if "rain" in precipitation_types or conditions.intersection(
        {"rain", "storm"}
    ):
        return (
            "Styling direction: keep the look polished and flexible with "
            "streamlined accessories, a removable layer, and a compact rain "
            "backup."
        )
    if precipitation_chance:
        return (
            "Styling direction: keep the established options polished and "
            "flexible with streamlined accessories and a compact weather backup."
        )
    if low_temperature is not None and low_temperature <= 60:
        return (
            "Styling direction: keep the established options polished and "
            "flexible with streamlined accessories and a removable layer."
        )
    if high_temperature is not None and high_temperature >= 80:
        return (
            "Styling direction: keep the established options polished with "
            "streamlined accessories and make any optional layer easy to remove."
        )
    return (
        "Styling direction: keep the established options polished and flexible "
        "with streamlined accessories and adaptable finishing pieces."
    )


def _historical_product_names(
    context: str,
    *,
    max_names: int = 6,
) -> tuple[str, ...]:
    """Read exact names from the newest server-formatted historical set."""

    candidate_lines = [
        line for line in context.splitlines() if line.startswith("- set=")
    ]
    if not candidate_lines:
        return ()
    _prefix, separator, records = candidate_lines[-1].partition(": ")
    if not separator:
        return ()
    names: list[str] = []
    for record in records.split("; "):
        position, colon, remainder = record.partition(":")
        if not colon or not position.isdigit():
            continue
        name_and_category, ref_separator, product_ref = remainder.rpartition(
            " <"
        )
        if (
            not ref_separator
            or not product_ref.endswith(">")
            or not product_ref[:-1]
        ):
            continue
        name = re.sub(r"\s+\[[^\]\n]*\]$", "", name_and_category).strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= max_names:
            break
    return tuple(names)


def _prior_weather_candidates_text(names: tuple[str, ...]) -> str:
    return "Previously shown options still in play: " + ", ".join(names) + "."


def _ensure_prior_weather_candidates(response: str, context: str) -> str:
    """Keep a context-only forecast follow-up attached to grounded options."""

    names = _historical_product_names(context)
    missing = tuple(
        name
        for name in names
        if name.casefold() not in response.casefold()
    )
    if not missing:
        return response
    prefix = _prior_weather_candidates_text(missing)
    return f"{prefix}\n\n{response.strip()}" if response.strip() else prefix


def _render_context_only_event_styling_decision(
    response: str,
    *,
    shopper_texts: tuple[str, ...],
) -> str:
    """Render only an exact shopper venue quote plus allowlisted adjustments."""

    try:
        decision = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(decision, dict) or set(decision) != {
        "venue_quote",
        "adjustments",
    }:
        return ""

    venue_quote = decision.get("venue_quote")
    adjustments = decision.get("adjustments")
    if (
        not isinstance(venue_quote, str)
        or venue_quote != venue_quote.strip()
        or not 1 <= len(venue_quote) <= 80
        or "\n" in venue_quote
        or not isinstance(adjustments, list)
        or not 1 <= len(adjustments) <= 2
        or any(not isinstance(value, str) for value in adjustments)
        or len(set(adjustments)) != len(adjustments)
        or any(
            value not in _CONTEXT_ONLY_EVENT_ADJUSTMENT_PHRASES
            for value in adjustments
        )
    ):
        return ""

    canonical_quote = ""
    for shopper_text in shopper_texts:
        match = re.search(
            re.escape(venue_quote),
            shopper_text,
            flags=re.IGNORECASE,
        )
        if match:
            canonical_quote = shopper_text[match.start() : match.end()]
            break
    if not canonical_quote:
        return ""

    phrases = [
        _CONTEXT_ONLY_EVENT_ADJUSTMENT_PHRASES[value]
        for value in adjustments
    ]
    if len(phrases) == 1:
        direction = phrases[0]
    else:
        direction = f"{phrases[0]} and {phrases[1]}"
    return (
        "Based on your venue detail "
        f"(“{_escape_markdown_inline(canonical_quote)}”), "
        f"{direction}."
    )


def _event_context_followup_question(
    state: State,
    next_question: str | None,
) -> str:
    """Render only the server-accepted event-context follow-up question."""

    if next_question == "event_date":
        return "What date or date range is the event?"
    if next_question == "event_venue":
        return "What kind of venue or setting is planned for the event?"
    if next_question != "event_location":
        return ""
    if (
        _saved_zipcode(state.shopper_context)
        and not _saved_area_override_present(state.query)
        and not _contains_exact_zip(state.query)
    ):
        return "Is the event in your usual area or elsewhere?"
    return "Where is the event taking place?"


def _compose_context_only_event_response(
    state: State,
    *,
    styling: str,
    next_question: str | None,
    weather_outcome: WeatherForecastEvidence | WeatherFailure | None,
    include_weather_facts: bool = True,
) -> str:
    """Compose a no-business-tool event turn from bounded, grounded context."""

    names = _historical_product_names(state.historical_product_context)
    parts: list[str] = []
    if names:
        parts.append(_prior_weather_candidates_text(names))

    styling = styling.strip()
    if styling:
        parts.append(styling)
    if isinstance(weather_outcome, WeatherForecastEvidence):
        parts.append(_deterministic_weather_styling_direction(weather_outcome))
    elif isinstance(weather_outcome, WeatherFailure):
        parts.append(_conditional_weather_styling_direction())
    elif not styling:
        if next_question in {
            "event_location",
            "event_venue",
            "event_date",
        }:
            styling = "I’ll apply the event setting before refining the guidance."
        else:
            styling = (
                "I’ll keep the options already shown and apply the event "
                "setting you provided."
            )
        parts.append(styling)

    followup = _event_context_followup_question(state, next_question)
    if followup:
        parts.append(followup)

    if weather_outcome is not None and include_weather_facts:
        parts.append(_format_weather_outcome(weather_outcome))
    return _scrub_internal_shopper_language(
        "\n\n".join(part for part in parts if part)
    )


def _has_successful_non_search_tool_evidence(
    result: Any,
    *,
    request_id: str,
) -> bool:
    """Return whether another current-turn shopping tool completed."""

    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        tool_name = str(_value(message, "name") or "")
        if tool_name in {
            "",
            SKILL_ACTIVATION_TOOL_NAME,
            "read_file",
            "search_catalog_tool",
        }:
            continue
        if (
            tool_name == _WEATHER_TOOL_NAME
            and not isinstance(
                parse_weather_tool_evidence(
                    _content_to_text(_value(message, "content"))
                ),
                WeatherForecastEvidence,
            )
        ):
            continue
        if _tool_call_status(tool_name, message)[0] == "completed":
            return True
    return False


def _format_search_only_response(
    state: State,
    result: Any,
    *,
    request_id: str,
    products: list[dict[str, Any]] | None = None,
    intro: str = "",
    heading: str = "Catalog candidates (verified name, price, and category):",
) -> str:
    """Render grounded search facts without interpreting display-name words."""

    search_groups = _search_result_groups(result, request_id=request_id)
    grouped_lines, grouped_names = _grouped_search_response_lines(search_groups)
    if len(grouped_lines) > 0:
        lines = grouped_lines
        displayed_names = grouped_names
    else:
        displayed_products = (
            products
            if products is not None
            else [
                product
                for product in state.product_results
                if isinstance(product, dict)
            ]
        )
        candidate_count = len(displayed_products)
        if intro.strip():
            lines = [
                "General guidance (not product-specific facts):",
                intro.strip(),
                "",
            ]
        else:
            noun = "candidate" if candidate_count == 1 else "candidates"
            lines = [f"I found {candidate_count} catalog {noun}.", ""]
        lines.extend((heading, ""))
        displayed_names = set()
        for product in displayed_products:
            name = str(product.get("display_name") or "").strip()
            if not name:
                continue
            displayed_names.add(name)
            parts = [f"**{name}**"]
            price = _product_result_price(product)
            if price:
                parts.append(price)
            category = str(product.get("category") or "").strip()
            if category:
                parts.append(category.replace("_", " "))
            lines.append("- " + " — ".join(parts))

    parent_relations = []
    for group in search_groups:
        relation = group.get("scope_relation") or {}
        requested_type = str(
            relation.get("requested_product_type") or ""
        ).strip()
        category = str(relation.get("advertised_category") or "").strip()
        normalized = {
            "requested_product_type": requested_type,
            "advertised_category": category,
        }
        if (
            relation.get("relation") == "model_selected_parent_category"
            and requested_type
            and category
            and normalized not in parent_relations
        ):
            parent_relations.append(normalized)
    if parent_relations:
        relation_lines = [
            (
                f"The catalog does not advertise **{relation['requested_product_type']}** "
                "as a separate product type. I searched the broader "
                f"**{relation['advertised_category']}** category for the closest "
                "options; each result keeps its actual catalog category."
            )
            for relation in parent_relations
        ]
        lines = relation_lines + [""] + lines

    filter_groups = _confirmed_search_filter_groups(
        result,
        request_id=request_id,
        displayed_names=displayed_names,
    )
    if filter_groups:
        lines.extend(("", "Catalog-confirmed filters by search:"))
        for group in filter_groups:
            product_names = group["product_names"]
            scope = (
                ", ".join(f"**{name}**" for name in product_names)
                if product_names
                else "Search candidates"
            )
            lines.append(f"- {scope}: {'; '.join(group['statements'])}.")
    unavailable_types = _no_direct_search_types(
        result,
        request_id=request_id,
    )
    if unavailable_types:
        lines.extend(("", "Unavailable requested catalog types:"))
        lines.extend(
            f"- **{product_type}** is not advertised; I did not substitute "
            "an adjacent product type."
            for product_type in unavailable_types
        )
    if _has_unsupported_requirement_outcome(
        result,
        request_id=request_id,
    ):
        lines.extend(("", _UNSUPPORTED_REQUIREMENT_RESPONSE))
    if not _search_scope_is_complete(result, request_id=request_id):
        lines.extend(
            (
                "",
                (
                    "This is a partial result set. I can continue with the next "
                    "requested piece or search scope."
                ),
            )
        )
    lines.extend(
        (
            "",
            (
                "These candidates were ranked toward your requested direction. "
                "Product-specific material, construction, length, fit, comfort, "
                "care, or weather performance remains unverified unless listed "
                "above as a catalog-confirmed filter."
            ),
        )
    )
    return "\n".join(lines)


def _format_product_detail_fallback(
    result: Any,
    *,
    request_id: str,
) -> str:
    """Render verified current-turn detail facts when final editing fails."""

    records: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        if str(_value(message, "name") or "") != "get_product_details_tool":
            continue
        content = _content_to_text(_value(message, "content"))
        if not content.startswith(_PRODUCT_DETAIL_GROUNDING_NOTE):
            continue
        for record in _product_evidence_records(content):
            product_ref = str(record.get("product_ref") or "").strip()
            if not product_ref or product_ref in seen_refs:
                continue
            seen_refs.add(product_ref)
            records.append(record)

    if not records:
        return ""

    lines = ["Verified catalog details:"]
    for record in records:
        name = _escape_markdown_inline(str(record.get("name") or "").strip())
        if not name:
            continue
        summary = [f"**{name}**"]
        price = str(record.get("price") or "").strip()
        if price:
            summary.append(_escape_markdown_inline(price))
        category = str(record.get("category") or "").strip()
        if category:
            summary.append(
                _escape_markdown_inline(category.replace("_", " "))
            )
        lines.append("- " + " — ".join(summary))
        for detail in record.get("details") or []:
            label, separator, value = str(detail).partition(":")
            if not separator or not label.strip() or not value.strip():
                continue
            lines.append(
                "  - "
                f"{_escape_markdown_inline(label.strip())}: "
                f"{_escape_markdown_inline(value.strip())}"
            )
    if len(lines) == 1:
        return ""
    lines.extend(
        (
            "",
            (
                "Only the listed fields are confirmed. Any missing material, "
                "construction, fit, care, comfort, or performance detail "
                "remains unavailable."
            ),
        )
    )
    return "\n".join(lines)


def _search_result_groups(result: Any, *, request_id: str) -> list[dict[str, Any]]:
    """Return successful search evidence grouped by the call that produced it."""

    groups: list[dict[str, Any]] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        content = _content_to_text(_value(message, "content"))
        if "SEARCH_RESULT_GROUNDING_NOTE" not in content:
            continue
        guidance = _search_json_evidence(
            content,
            _SEARCH_GUIDANCE_EVIDENCE_PREFIX,
        )
        groups.append(
            {
                "guidance": _scrub_internal_shopper_language(
                    str(guidance.get("text") or "")
                ).strip(),
                "products": _product_evidence_records(content),
                "taxonomy": _search_taxonomy_evidence(content),
                "scope_relation": _search_scope_relation_evidence(content),
            }
        )
    return groups


def _response_mentions_search_candidate(
    response: str,
    result: Any,
    *,
    request_id: str,
) -> bool:
    """Return whether final text preserves at least one current search candidate."""

    normalized_response = response.casefold()
    return any(
        str(product.get("name") or "").casefold() in normalized_response
        for group in _search_result_groups(result, request_id=request_id)
        for product in group.get("products") or []
        if str(product.get("name") or "").strip()
    )


def _grouped_search_response_lines(
    groups: list[dict[str, Any]],
) -> tuple[list[str], set[str]]:
    """Render multi-search candidates without losing their guidance scope."""

    if len(groups) < 2 or not all(group.get("guidance") for group in groups):
        return [], set()
    lines: list[str] = []
    displayed_names: set[str] = set()
    displayed_product_refs: set[str] = set()
    for index, group in enumerate(groups, start=1):
        products = [
            product
            for product in group.get("products") or []
            if str(
                product.get("product_ref")
                or f"name:{product.get('name') or ''}"
            )
            not in displayed_product_refs
        ]
        if not products:
            continue
        lines.extend(_format_search_group(group, products, index=index))
        displayed_product_refs.update(
            str(
                product.get("product_ref")
                or f"name:{product.get('name') or ''}"
            )
            for product in products
        )
        displayed_names.update(str(product["name"]) for product in products)
    if lines and not lines[-1]:
        lines.pop()
    return lines, displayed_names


def _format_search_group(
    group: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    index: int,
) -> list[str]:
    """Format one search group's bounded guidance and verified products."""

    taxonomy = group.get("taxonomy") or {}
    values = taxonomy.get("subcategory") or taxonomy.get("category") or []
    label = " / ".join(str(value).replace("_", " ") for value in values)
    title = label.title() if label else f"Product group {index}"
    lines = [f"**{title}**", "", "General guidance (not product-specific facts):"]
    lines.extend((str(group["guidance"]), "", "Catalog candidates:", ""))
    for product in products:
        parts = [f"**{product['name']}**"]
        if product.get("price"):
            parts.append(str(product["price"]))
        if product.get("category"):
            parts.append(str(product["category"]).replace("_", " "))
        lines.append("- " + " — ".join(parts))
    lines.append("")
    return lines


def _no_direct_search_types(result: Any, *, request_id: str) -> list[str]:
    """Return current-turn product types with a server-authored no-direct outcome."""

    product_types: list[str] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        outcome = _search_json_evidence(
            _content_to_text(_value(message, "content")),
            _CATALOG_SCOPE_OUTCOME_PREFIX,
        )
        if not outcome or outcome.get("outcome") != "no_direct_catalog_match":
            continue
        product_type = str(outcome.get("requested_product_type") or "").strip()
        if product_type and product_type not in product_types:
            product_types.append(product_type)
    return product_types


def _search_scope_is_complete(result: Any, *, request_id: str) -> bool:
    """Return whether a current-turn search declared its requested scope complete."""

    return any(
        _message_type(message) == "tool"
        and SEARCH_SCOPE_COMPLETE_PREFIX
        in _content_to_text(_value(message, "content"))
        for message in _current_turn_messages(_result_messages(result), request_id)
    )


def _search_guidance_evidence(result: Any, *, request_id: str) -> list[str]:
    """Return bounded pre-search shopper guidance from successful searches."""

    guidance: list[str] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        content = _content_to_text(_value(message, "content"))
        if "SEARCH_RESULT_GROUNDING_NOTE" not in content:
            continue
        payload = _search_json_evidence(
            content,
            _SEARCH_GUIDANCE_EVIDENCE_PREFIX,
        )
        text = _scrub_internal_shopper_language(
            str((payload or {}).get("text") or "")
        ).strip()
        if text and text not in guidance:
            guidance.append(text)
    return guidance


def _confirmed_search_filter_groups(
    result: Any,
    *,
    request_id: str,
    displayed_names: set[str] | None = None,
) -> list[dict[str, list[str]]]:
    """Keep canonical filters scoped to products from the same search."""

    groups: list[dict[str, list[str]]] = []
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        content = _content_to_text(_value(message, "content"))
        filters = _search_filter_evidence(content)
        statements: list[str] = []
        for name, value in filters.items():
            statement = _format_filter_statement(name, value)
            if statement:
                statements.append(statement)
        if not statements:
            continue
        product_names = [
            product["name"]
            for product in _product_evidence_records(content)
            if product.get("name")
        ]
        if displayed_names is not None:
            product_names = [
                name for name in product_names if name in displayed_names
            ]
            if not product_names:
                continue
        groups.append(
            {"product_names": product_names, "statements": statements}
        )
    return groups


def _format_filter_statement(name: str, value: Any) -> str:
    label = name.replace("_", " ")
    if isinstance(value, list):
        values = [str(item).replace("_", " ") for item in value]
        if len(values) == 1:
            return f"{label} is {values[0]}"
        if values:
            return f"{label} is one of {', '.join(values)}"
        return ""
    if isinstance(value, dict):
        bounds = []
        if value.get("min") is not None:
            bounds.append(f"minimum {value['min']}")
        if value.get("max") is not None:
            bounds.append(f"maximum {value['max']}")
        return f"{label} {' and '.join(bounds)}" if bounds else ""
    return f"{label} is {value}"


def _product_result_price(product: dict[str, Any]) -> str:
    price = product.get("price")
    if isinstance(price, dict):
        amount = price.get("amount")
        currency = str(price.get("currency") or "USD")
    else:
        amount = price
        currency = "USD"
    if not isinstance(amount, (int, float)):
        return ""
    return f"${float(amount):.2f} {currency}"


def _should_short_circuit_media_failure(state: State) -> bool:
    if not state.media or not state.media_analysis:
        return False
    if not any(str(item.get("type") or "").lower() == "video" for item in state.media):
        return False
    if not _media_analysis_unavailable(state.media_analysis):
        return False
    return _query_depends_on_media(state.query)


def _record_media_model_usage(state: State, config: Any) -> None:
    if not getattr(config, "vlm_enabled", False) or not getattr(config, "vlm_name", None):
        _add_model_usage(state, "vlm", status="disabled", calls=0)
        return

    status = "failed" if _media_analysis_unavailable(state.media_analysis) else "used"
    _add_model_usage(
        state,
        "vlm",
        status=status,
        calls=1,
        detail="Attached media understanding",
    )


def _record_catalog_model_usage(
    state: State,
    plan: CatalogSearchPlan,
    ok: bool,
    *,
    fallback_attempted: bool = False,
) -> None:
    status = "used" if ok else "failed"
    uses_image_endpoint = bool(state.image) and plan.search_mode in {"image", "hybrid"}
    text_embedding_calls = 1 if plan.semantic_queries else 0
    if uses_image_endpoint and text_embedding_calls == 0:
        # Image retrieval currently includes one deterministic text-side query
        # alongside the image embedding, even when the shopper supplied no text.
        text_embedding_calls = 1
    if fallback_attempted:
        text_embedding_calls += 1 if plan.semantic_queries else 0

    if text_embedding_calls:
        _add_model_usage(
            state,
            "text_embedding",
            status=status,
            calls=text_embedding_calls,
            detail="Catalog text/vector retrieval",
        )
    if uses_image_endpoint:
        _add_model_usage(
            state,
            "image_embedding",
            status=status,
            calls=1,
            detail="Catalog image similarity retrieval",
        )


def _record_language_model_failure(state: State) -> None:
    _add_model_usage(
        state,
        "app_llm",
        status="failed",
        calls=1,
        detail="Planning, tool use, and response generation failed",
    )


def _record_safety_model_usage(state: State, mode: str, *, ok: bool = True) -> None:
    status = "used" if ok else "failed"
    detail = "Input and output safety checks" if ok else "Guardrails check failed open"
    if mode == "input":
        _add_model_usage(
            state,
            "content_safety",
            status=status,
            calls=1,
            detail=detail,
        )
        _add_model_usage(
            state,
            "topic_control",
            status=status,
            calls=1,
            detail="Input topic check" if ok else "Guardrails topic check failed open",
        )
        return

    _add_model_usage(
        state,
        "content_safety",
        status=status,
        calls=1,
        detail=detail,
    )


def _add_model_usage(
    state: State,
    role: str,
    *,
    status: str,
    calls: int,
    detail: str = "",
) -> None:
    existing = state.model_usage.get(role, {})
    existing_calls = _safe_int(existing.get("calls"))
    existing_status = str(existing.get("status") or "")
    merged_status = _merged_model_usage_status(existing_status, status)
    existing_detail = str(existing.get("detail") or "")
    next_detail = (
        existing_detail
        if existing_status == merged_status and existing_detail
        else detail or existing_detail
    )
    state.model_usage[role] = {
        "status": merged_status,
        "calls": existing_calls + max(0, calls),
        "detail": next_detail,
    }


def _merged_model_usage_status(existing: str, current: str) -> str:
    priority = {"failed": 4, "used": 3, "disabled": 2, "not_used": 1, "": 0}
    return existing if priority.get(existing, 0) > priority.get(current, 0) else current


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _media_analysis_unavailable(media_analysis: str) -> bool:
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        text = media_analysis
    else:
        if not isinstance(parsed, dict):
            text = str(parsed)
        else:
            parts = [str(parsed.get("summary") or "")]
            uncertainties = parsed.get("uncertainties")
            if isinstance(uncertainties, list):
                parts.extend(str(item) for item in uncertainties)
            text = " ".join(parts)

    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "not configured",
            "unavailable",
            "authentication failed",
            "could not authenticate",
            "not allowed to access",
            "media analysis failed",
            "vlm returned no analysis",
        )
    )


def _query_depends_on_media(query: str) -> bool:
    lowered = (query or "").strip().lower()
    if not lowered or lowered == "the user submitted visual media without additional text.":
        return True
    return any(
        phrase in lowered
        for phrase in (
            "this video",
            "the video",
            "in video",
            "from video",
            "attached video",
            "she is wearing",
            "he is wearing",
            "they are wearing",
            "person is wearing",
            "like that",
            "like this",
            "similar to that",
            "similar to this",
            "what is in this",
            "what's in this",
            "what am i wearing",
            "what are they wearing",
            "describe this",
        )
    )


def _media_failure_response(media_analysis: str) -> str:
    detail = "video/image understanding is unavailable for this turn"
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
        if summary:
            detail = _clean_media_failure_detail(summary)

    return (
        f"I could not analyze the attached media because {detail}. "
        "Please describe the item in text, such as color, silhouette, material, "
        "and any visible details, and I can search the catalog from that description."
    )


def _clean_media_failure_detail(summary: str) -> str:
    detail = summary.strip()
    for prefix in (
        "Media was attached, but ",
        "Video was attached, but ",
        "Image was attached, but ",
    ):
        if detail.startswith(prefix):
            detail = detail[len(prefix):]
            break
    if detail:
        detail = detail[0].lower() + detail[1:]
    return detail.rstrip(". ") or "video/image understanding is unavailable for this turn"


def _collect_token_usage(result: Any) -> dict[str, int]:
    """Collect normalized token usage from LangChain/OpenAI message metadata."""

    usage = _empty_token_usage()
    for message in _result_messages(result):
        record = _message_token_usage_record(message)
        if not record:
            continue

        input_tokens = _token_int(record, ("input_tokens", "prompt_tokens", "input"))
        output_tokens = _token_int(
            record,
            ("output_tokens", "completion_tokens", "output"),
        )
        total_tokens = _token_int(record, ("total_tokens", "total"))
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue

        usage["input_tokens"] += int(input_tokens or 0)
        usage["output_tokens"] += int(output_tokens or 0)
        usage["total_tokens"] += int(total_tokens or 0)
        usage["model_calls"] += 1
    return usage


def _merge_token_usage(
    base: dict[str, Any] | None,
    additional: dict[str, Any] | None,
) -> dict[str, int]:
    merged = _normalized_token_usage(base)
    extra = _normalized_token_usage(additional)
    for key in merged:
        merged[key] += extra[key]
    return merged


def _collect_tool_grounding_evidence(
    result: Any,
    *,
    max_chars: int,
    request_id: str | None = None,
) -> str:
    messages = _result_messages(result)
    if request_id is not None:
        messages = _current_turn_messages(messages, request_id)
    return _collect_message_grounding_evidence(messages, max_chars=max_chars)


def _collect_message_grounding_evidence(
    messages: list[Any],
    *,
    max_chars: int,
) -> str:
    """Collect customer-safe evidence from actual tool-role messages."""

    parts: list[str] = []
    for message in messages:
        content = _content_to_text(_value(message, "content"))
        if not content:
            continue
        if not _is_tool_evidence_message(message, content):
            continue
        parts.append(_customer_safe_tool_evidence(content))

    evidence = "\n\n---\n\n".join(parts).strip()
    if len(evidence) <= max_chars:
        return evidence
    return evidence[-max_chars:]


def _customer_safe_tool_evidence(content: str) -> str:
    if content.startswith(
        (WEATHER_FORECAST_EVIDENCE_PREFIX, WEATHER_FORECAST_FAILURE_PREFIX)
    ):
        return _customer_safe_weather_evidence(content)
    if content.startswith(SEARCH_VALIDATION_ERROR_PREFIX):
        return (
            "CUSTOMER_SAFE_INVALID_SEARCH_EVIDENCE: No valid catalog search "
            "scope was established and no retrieval ran. This does not support "
            "a product-availability or catalog-absence claim."
        )
    if content.startswith(
        "STOP_TOOL_USE: No faithful advertised catalog taxonomy"
    ):
        return (
            "CUSTOMER_SAFE_NO_MATCH_EVIDENCE: The active catalog has no direct "
            "advertised taxonomy match for this requested product role. "
            "No retrieval ran and no alternative product type was selected. Say "
            "that plainly, preserve any successful evidence for other roles, and "
            "ask permission before searching a different advertised type. Do not "
            "name alternatives."
        )
    if "SEARCH_NO_MATCH_GROUNDING_NOTE" in content:
        taxonomy = _search_taxonomy_evidence(content)
        confirmed_filters = _search_filter_evidence(content)
        lines = [
            (
                "CUSTOMER_SAFE_SCOPED_NO_MATCH_EVIDENCE: Zero products matched "
                "only the exact advertised search scope below. This does not "
                "establish that a different, unsearched, or unadvertised product "
                "type is absent, and it does not support a catalog-wide "
                "availability claim."
            )
        ]
        if taxonomy:
            lines.append(
                "ADVERTISED_SEARCH_TAXONOMY: "
                + json.dumps(taxonomy, sort_keys=True)
            )
        if confirmed_filters:
            lines.append(
                "CONFIRMED_SEARCH_FILTERS: "
                + json.dumps(confirmed_filters, sort_keys=True)
            )
        scope_relation = _customer_safe_scope_relation(
            content,
            has_products=False,
        )
        if scope_relation:
            lines.append(scope_relation)
        return "\n".join(lines)
    if "SEARCH_RESULT_GROUNDING_NOTE" in content:
        summary = _summarize_product_evidence(
            content,
            heading="CUSTOMER_SAFE_SEARCH_EVIDENCE",
            note=(
                "Search results support only product names, prices, categories, "
                "image availability, confirmed search filters, and a modest "
                "styling role. Beyond an exact confirmed filter, they do not "
                "support length, color, print, materials, care, construction, "
                "fit, comfort, weather, grass, gravel, heat, or best-in-category "
                "claims. Treat names as display names, not attribute evidence; "
                "group claims require product-detail evidence for every item."
            ),
            confirmed_filters=_search_filter_evidence(content),
            taxonomy_scope=_search_taxonomy_evidence(content),
        )
        scope_relation = _customer_safe_scope_relation(
            content,
            has_products=True,
        )
        return "\n".join(
            part for part in (summary, scope_relation) if part
        )
    if "PRODUCT_DETAIL_GROUNDING_NOTE" in content:
        return _summarize_product_evidence(
            content,
            heading="CUSTOMER_SAFE_PRODUCT_DETAIL_EVIDENCE",
            note=(
                "Product details were read for these products, but the available "
                "detail data contains only the listed facts. Do "
                "not state material, care, dimensions, closures, fit, sizing, "
                "colorways, or outdoor performance unless the field appears in "
                "this evidence summary."
            ),
        )
    return _summarize_cart_evidence(content)


def _customer_safe_weather_evidence(content: str) -> str:
    parsed = parse_weather_tool_evidence(content)
    if isinstance(parsed, WeatherForecastEvidence):
        lines = [
            (
                "CUSTOMER_SAFE_WEATHER_FORECAST_EVIDENCE: This is bounded "
                "current-turn live forecast evidence. Call it a forecast and "
                "do not infer climate, venue, product attributes, performance, "
                "comfort, or safety from it."
            ),
            "FORECAST_FETCHED_AT: " + parsed.fetched_at.isoformat(),
            (
                "FORECAST_WINDOW: "
                + json.dumps(
                    parsed.requested_window.model_dump(mode="json"),
                    sort_keys=True,
                )
            ),
        ]
        if parsed.resolved_location:
            lines.append(
                "FORECAST_LOCATION_USED: " + parsed.resolved_location
            )
        if parsed.relative_date:
            lines.append(
                "FORECAST_RELATIVE_DATE: " + parsed.relative_date
            )
        if parsed.weekday:
            lines.append("FORECAST_RELATIVE_WEEKDAY: " + parsed.weekday)
        lines.extend(
            "FORECAST_DAY: "
            + json.dumps(day.model_dump(mode="json"), sort_keys=True)
            for day in parsed.days
        )
        lines.extend(
            [
                (
                    "FORECAST_ATTRIBUTION: "
                    f"[{parsed.attribution.label}]"
                    f"({parsed.attribution.url})"
                ),
                "FORECAST_UNCERTAINTY: " + _WEATHER_UNCERTAINTY_TEXT,
            ]
        )
        return "\n".join(lines)
    if isinstance(parsed, WeatherFailure):
        return (
            "CUSTOMER_SAFE_WEATHER_FAILURE_EVIDENCE: No forecast facts were "
            "established. Do not claim conditions. Stable outcome: "
            + json.dumps(parsed.model_dump(mode="json"), sort_keys=True)
        )
    return (
        "CUSTOMER_SAFE_WEATHER_FAILURE_EVIDENCE: Weather evidence was invalid. "
        "No forecast facts were established; do not claim conditions."
    )


def _summarize_product_evidence(
    content: str,
    *,
    heading: str,
    note: str,
    confirmed_filters: dict[str, Any] | None = None,
    taxonomy_scope: dict[str, Any] | None = None,
) -> str:
    products = _product_evidence_records(content)
    lines = [f"{heading}: {note}"]
    if confirmed_filters:
        lines.append(
            "CONFIRMED_SEARCH_FILTERS: Every product below passed each filter "
            "predicate. A one-value list confirms that value; a multi-value list "
            "confirms only membership in the set, not which value matched: "
            + json.dumps(confirmed_filters, sort_keys=True, default=str)
        )
    if taxonomy_scope:
        lines.append(
            "ADVERTISED_SEARCH_TAXONOMY: This search used only these advertised "
            "taxonomy values. Lists are inclusive scopes; they do not mean every "
            "product has every value. Do not describe an unlisted product type "
            "as advertised: "
            + json.dumps(taxonomy_scope, sort_keys=True, default=str)
        )
    if not products:
        return "\n".join(lines + [_strip_internal_ids_from_evidence_line(content)])
    for product in products:
        summary_parts = [product["name"]]
        if product.get("category"):
            summary_parts.append(f"category: {product['category']}")
        if product.get("price"):
            summary_parts.append(f"price: {product['price']}")
        if product.get("image_url"):
            summary_parts.append("image: available")
        if product.get("details"):
            summary_parts.append("details: " + "; ".join(product["details"]))
        lines.append("- " + " | ".join(summary_parts))
    return "\n".join(lines)


def _search_filter_evidence(content: str) -> dict[str, Any]:
    """Read the canonical hard-filter marker from one search result."""

    return _search_json_evidence(content, _SEARCH_FILTER_EVIDENCE_PREFIX)


def _search_taxonomy_evidence(content: str) -> dict[str, Any]:
    """Read the advertised taxonomy marker from one search result."""

    return _search_json_evidence(content, _SEARCH_TAXONOMY_EVIDENCE_PREFIX)


def _search_scope_relation_evidence(content: str) -> dict[str, Any]:
    """Read the requested-to-advertised scope relation from one search result."""

    return _search_json_evidence(
        content,
        _SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX,
    )


def _customer_safe_scope_relation(
    content: str,
    *,
    has_products: bool,
) -> str:
    """Describe one model-selected parent search without asserting subtype identity."""

    relation = _search_scope_relation_evidence(content)
    if relation.get("relation") != "model_selected_parent_category":
        return ""
    requested_type = str(relation.get("requested_product_type") or "").strip()
    category = str(relation.get("advertised_category") or "").strip()
    if not requested_type or not category:
        return ""
    if not has_products:
        return (
            f"REQUESTED_SCOPE_RELATION: {requested_type} is not separately "
            f"advertised. The broader advertised category {category} returned "
            "zero products for this search, so do not claim that the requested "
            "type is absent from the whole catalog."
        )
    return (
        f"REQUESTED_SCOPE_RELATION: {requested_type} is not separately "
        f"advertised. The search used the broader advertised category {category}. "
        "Present these as closest options and keep every returned product's "
        "actual catalog category; do not relabel them as the requested type."
    )


def _search_json_evidence(content: str, prefix: str) -> dict[str, Any]:
    """Read one JSON evidence marker from a tool result."""

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        payload = line.removeprefix(prefix).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _product_evidence_records(content: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    reading_details = False
    key_map = {
        "NAME:": "name",
        "CATEGORY:": "category",
        "BRAND:": "brand",
        "PRICE:": "price",
        "IMAGE_URL:": "image_url",
    }
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("PRODUCT_REF:"):
            if current.get("name"):
                records.append(current)
            current = {"product_ref": line.removeprefix("PRODUCT_REF:").strip()}
            reading_details = False
            continue
        if line == "DETAILS:":
            reading_details = True
            continue
        if reading_details and line.startswith("- ") and ":" in line:
            label, value = line[2:].split(":", 1)
            if label.strip() and value.strip():
                current.setdefault("details", []).append(
                    f"{label.strip()}: {value.strip()}"
                )
            continue
        for prefix, key in key_map.items():
            if line.startswith(prefix):
                current[key] = line[len(prefix) :].strip()
                break
    if current.get("name"):
        records.append(current)
    return records


def _summarize_cart_evidence(content: str) -> str:
    lines = ["CUSTOMER_SAFE_CART_EVIDENCE:"]
    for raw_line in content.splitlines():
        cleaned = _strip_internal_ids_from_evidence_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _strip_internal_ids_from_evidence_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("PRODUCT_REF:") or stripped.startswith("CART_LINE_ID:"):
        return ""
    if stripped.startswith("- CART_LINE_ID:") and "|" in stripped:
        return "- " + stripped.split("|", 1)[1].strip()

    marker = "(PRODUCT_REF:"
    while marker in stripped:
        start = stripped.find(marker)
        end = stripped.find(")", start)
        if end == -1:
            stripped = stripped[:start].rstrip()
            break
        stripped = (stripped[:start] + stripped[end + 1 :]).strip()
    return stripped


def _is_tool_evidence_message(message: Any, content: str) -> bool:
    message_type = str(_value(message, "type") or "").lower()
    role = str(_value(message, "role") or "").lower()
    tool_name = str(_value(message, "name") or "")
    if tool_name in {SKILL_ACTIVATION_TOOL_NAME, "read_file"}:
        return False
    if content.startswith(
        (
            SKILL_ACTIVATION_COMPLETE,
            SKILL_ACTIVATION_REQUIRED,
            SKILL_TOOL_NOT_GRANTED,
            "SHOPPER_SKILL_ACTIVATION_FAILED:",
        )
    ):
        return False
    return message_type == "tool" or role == "tool"


def _normalized_token_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    usage = _empty_token_usage()
    if not isinstance(raw, dict):
        return usage
    for key in usage:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            usage[key] = max(0, int(value))
    return usage


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
    }


def _result_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            return messages
        return [result]

    messages = getattr(result, "messages", None)
    if isinstance(messages, list):
        return messages
    return [result]


def _message_token_usage_record(message: Any) -> Any:
    usage_metadata = _value(message, "usage_metadata")
    if usage_metadata:
        return usage_metadata

    response_metadata = _value(message, "response_metadata")
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage:
            return token_usage

    for key in ("token_usage", "usage"):
        record = _value(message, key)
        if record:
            return record
    return None


def _value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _token_int(record: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _value(record, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None


def _format_search_filter_evidence(filters: dict[str, Any]) -> str:
    """Format canonical hard filters proven by a successful search."""

    return (
        f"{_SEARCH_FILTER_EVIDENCE_PREFIX} "
        + json.dumps(filters, sort_keys=True, default=str)
    )


def _format_search_direction_evidence(semantic_query: str) -> str:
    """Record the model-authored preference used for successful ranking."""

    return (
        f"{_SEARCH_DIRECTION_EVIDENCE_PREFIX} "
        + json.dumps(semantic_query, ensure_ascii=False)
    )


def _format_search_guidance_evidence(shopper_guidance: str) -> str:
    """Record bounded product-agnostic guidance authored before retrieval."""

    return (
        f"{_SEARCH_GUIDANCE_EVIDENCE_PREFIX} "
        + json.dumps({"text": shopper_guidance.strip()}, ensure_ascii=False)
    )


def _format_search_taxonomy_evidence(taxonomy: dict[str, Any]) -> str:
    """Format the advertised taxonomy scope used by a successful search."""

    return (
        f"{_SEARCH_TAXONOMY_EVIDENCE_PREFIX} "
        + json.dumps(taxonomy, sort_keys=True, default=str)
    )


def _format_search_scope_relation_evidence(
    *,
    requested_product_type: str,
    advertised_category: str,
) -> str:
    """Record a model-selected advertised parent for honest response framing."""

    return (
        f"{_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX} "
        + json.dumps(
            {
                "relation": "model_selected_parent_category",
                "requested_product_type": requested_product_type,
                "advertised_category": advertised_category,
            },
            sort_keys=True,
        )
    )


def _format_catalog_scope_outcome(outcome: dict[str, Any]) -> str:
    """Format one bounded non-product catalog outcome for diagnostics."""

    return (
        f"{_CATALOG_SCOPE_OUTCOME_PREFIX} "
        + json.dumps(outcome, sort_keys=True, default=str)
    )


def _format_product(product: Any) -> str:
    lines = [
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if getattr(product, "category", None):
        lines.append(f"CATEGORY: {product.category}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    lines.append(
        "DETAILS: Call get_product_details_tool with this PRODUCT_REF before "
        "stating materials, dimensions, pockets, closures, care, comfort, or "
        "outdoor-practicality claims."
    )
    return "\n".join(lines)


def _format_product_details(product: ProductDetail) -> str:
    lines = [
        _PRODUCT_DETAIL_GROUNDING_NOTE,
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if product.category:
        lines.append(f"CATEGORY: {product.category}")
    if product.brand:
        lines.append(f"BRAND: {product.brand}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    if product.attributes:
        lines.append("DETAILS:")
        for name, value in sorted(product.attributes.items()):
            lines.append(
                f"- {name.replace('_', ' ')}: {_format_detail_value(value)}"
            )
    else:
        lines.append("NO_ADDITIONAL_STRUCTURED_DETAILS")
    return "\n".join(lines)


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(
            f"{key}={value[key]}" for key in sorted(value)
        )
    return str(value)


def _normalize_cart_add_tool_items(
    items: list[AddCartItemsToolItemInput] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for item in items or []:
        try:
            parsed = (
                item
                if isinstance(item, AddCartItemsToolItemInput)
                else AddCartItemsToolItemInput.model_validate(item)
            )
            quantity = max(1, int(parsed.quantity or 1))
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError("each item must include a PRODUCT_REF and quantity") from exc
        entry = normalized.setdefault(
            parsed.product_ref,
            {
                "quantity": 0,
                "expected_display_name": (
                    parsed.expected_display_name.strip()
                    if parsed.expected_display_name
                    else ""
                ),
            },
        )
        entry["quantity"] += quantity
        if not entry["expected_display_name"] and parsed.expected_display_name:
            entry["expected_display_name"] = parsed.expected_display_name.strip()
    return normalized


def _cart_add_scope_failures(
    user_query: str,
    requested_products: list[tuple[str, ProductSummary]],
    available_products: Any,
) -> list[str]:
    explicitly_named = _explicitly_named_products(user_query, available_products)
    if not explicitly_named:
        return []

    explicit_names = {
        _normalize_product_name(product.display_name) for product in explicitly_named
    }
    failures = []
    for product_ref, product in requested_products:
        if _normalize_product_name(product.display_name) in explicit_names:
            continue
        failures.append(
            f"- PRODUCT_REF '{product_ref}': selected '{product.display_name}' is "
            "outside the current explicit add request. The current request names: "
            f"{_format_product_refs(explicitly_named)}. Retry with matching "
            "PRODUCT_REF values only, or ask a clarification."
        )
    return failures


def _explicitly_named_products(
    text: str,
    available_products: Any,
) -> list[ProductSummary]:
    normalized_text = _normalize_product_name(text)
    if not normalized_text:
        return []

    padded_text = f" {normalized_text} "
    matches: list[ProductSummary] = []
    seen: set[str] = set()
    products = list(available_products)
    for product in products:
        normalized_name = _normalize_product_name(product.display_name)
        if not normalized_name:
            continue
        if f" {normalized_name} " not in padded_text:
            continue
        key = product.product_id or product.display_name
        if key in seen:
            continue
        seen.add(key)
        matches.append(product)

    query_tokens = set(_product_name_tokens(text))
    for product in products:
        key = product.product_id or product.display_name
        if key in seen:
            continue
        product_tokens = _product_name_tokens(product.display_name)
        required_overlap = 3 if len(product_tokens) > 3 and matches else 2
        if not _product_name_tokens_match(
            query_tokens,
            product_tokens,
            required_overlap=required_overlap,
        ):
            continue
        seen.add(key)
        matches.append(product)
    return matches


def _same_product_display_name(expected: str, actual: str) -> bool:
    return _normalize_product_name(expected) == _normalize_product_name(actual)


def _product_detail_failure_message(
    error: CommerceError | None,
    *,
    cart_validation: bool,
) -> str:
    if error is not None and error.code == "product_not_found":
        return (
            "The product is no longer present in the active catalog. "
            "Search again before adding it."
            if cart_validation
            else (
                "That product is no longer available in the active catalog. "
                "Search the catalog again before using its details."
            )
        )
    if error is not None and error.retryable:
        return (
            "The catalog is temporarily unavailable, so the cart was not changed. "
            "Please try again."
            if cart_validation
            else "Product details are temporarily unavailable. Please try again."
        )
    return (
        "The product could not be verified, so the cart was not changed. "
        "Search again before adding it."
        if cart_validation
        else "Product details could not be verified. Search the catalog again."
    )


def _normalize_product_name(value: str) -> str:
    chars = []
    for char in str(value or "").casefold():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def _product_name_tokens(value: str) -> list[str]:
    return [
        token
        for token in _normalize_product_name(value).split()
        if token not in _PRODUCT_NAME_STOPWORDS
    ]


def _product_name_tokens_match(
    query_tokens: set[str],
    product_tokens: list[str],
    *,
    required_overlap: int,
) -> bool:
    if len(product_tokens) < 2:
        return False
    overlap = query_tokens.intersection(product_tokens)
    required = min(required_overlap, len(set(product_tokens)))
    return len(overlap) >= required


def _format_product_refs(products: list[ProductSummary]) -> str:
    return ", ".join(
        f"{product.display_name} (PRODUCT_REF: {product.product_id})"
        for product in products
    )


def _scrub_internal_shopper_language(text: str) -> str:
    scrubbed = text or ""
    for internal, replacement in _INTERNAL_SHOPPER_REPLACEMENTS:
        scrubbed = scrubbed.replace(internal, replacement)
    return scrubbed


def _saved_zipcode(shopper_context: ShopperContext | None) -> str | None:
    return (
        shopper_context.zipcode
        if shopper_context is not None
        else None
    )


def _weather_lookup_enabled(weather_config: Any) -> bool:
    """Return whether configured weather lookup can make provider requests."""

    api_key_env = str(getattr(weather_config, "api_key_env", ""))
    weather_key = os.environ.get(api_key_env, "") if api_key_env else ""
    return bool(
        getattr(weather_config, "enabled", False)
        and weather_key
        and weather_key == weather_key.strip()
    )


def _event_context_question_boundary(next_question: str | None) -> str:
    """Render the activation-owned event-context follow-up boundary."""

    if next_question == "event_location":
        return (
            "At most one event-location question is permitted. Do not ask for "
            "the event date, venue details, dress code, or preferences."
        )
    if next_question == "event_date":
        return (
            "At most one event date or complete date-range question is "
            "permitted. The destination is already established: do not ask "
            "about location, usual/home area, venue, dress code, or "
            "preferences."
        )
    if next_question == "event_venue":
        return (
            "At most one event venue or setting question is permitted. The "
            "destination is already established: do not ask about location, "
            "usual/home area, date, weather, dress code, or preferences, and "
            "do not infer a beach, outdoor, indoor, or terrain setting from "
            "the destination."
        )
    return (
        "Ask no event-location, venue, date, weather, dress-code, or preference "
        "follow-up question."
    )


def _scrub_saved_zip_from_response(
    text: str,
    shopper_context: ShopperContext | None,
) -> str:
    """Keep the selected profile's saved ZIP out of shopper-facing text."""

    if shopper_context is None or not shopper_context.zipcode:
        return text
    scrubbed = re.sub(
        rf"(?<![0-9]){re.escape(shopper_context.zipcode)}(?![0-9])",
        "your usual area",
        text,
    )
    return scrubbed.replace(
        "ZIP code your usual area",
        "your usual area",
    ).replace(
        "ZIP your usual area",
        "your usual area",
    )


def _scrub_weather_receipt_ids(text: str, state: State) -> str:
    """Keep internal durable receipt identities out of shopper-facing text."""

    receipt_ids = {
        receipt.receipt_id for receipt in state.active_weather_receipts
    }
    if state.selected_weather_receipt_id is not None:
        receipt_ids.add(state.selected_weather_receipt_id)
    scrubbed = text
    for receipt_id in receipt_ids:
        scrubbed = scrubbed.replace(receipt_id, "the validated forecast")
    return scrubbed


def _format_cart_add_result(added: list[str], failed: list[str], cart: Cart) -> str:
    lines = ["CART_ADD_RESULT"]
    if added:
        lines.append("Added:")
        lines.extend(added)
    if failed:
        lines.append("Failed:")
        lines.extend(failed)
    lines.append("Current cart:")
    lines.append(_format_cart_lines(cart))
    lines.append("Cart total:")
    lines.append(_format_cart_total(cart))
    return "\n".join(lines)


def _format_cart_lines(cart: Cart | CommerceCart) -> str:
    if isinstance(cart, CommerceCart):
        if not cart.lines:
            return "  (cart is empty)"
        lines = [
            f"  {line.cart_line_id} | {line.display_name} | qty {line.quantity}"
            + (
                f" | {line.unit_price.currency} {line.unit_price.amount:.2f}"
                if line.unit_price
                else ""
            )
            for line in cart.lines
        ]
        if cart.subtotal:
            lines.append(
                f"  SUBTOTAL: {cart.subtotal.currency} {cart.subtotal.amount:.2f}"
            )
        return "\n".join(lines)

    if not cart.contents:
        return "(empty)"
    lines = []
    for item in cart.contents:
        price = item.get("price")
        suffix = ""
        if price is not None:
            try:
                suffix = f" @ ${float(price):.2f}"
            except (TypeError, ValueError):
                suffix = ""
        cart_line_id = item.get("cart_line_id") or item.get("item", "")
        lines.append(
            f"- CART_LINE_ID: {cart_line_id} | "
            f"{item.get('amount', 1)} x {item.get('item', '')}{suffix}"
        )
    return "\n".join(lines)


def _format_cart(cart: Cart) -> str:
    return _format_cart_lines(cart)


def _format_cart_remove_result(
    result: CartMutationResult,
    *,
    fallback: str,
) -> str:
    if not result.ok:
        return result.error.message if result.error else "Cart remove failed."
    message = result.message or fallback
    if result.cart is not None:
        return "\n".join(
            [message, "Current cart:", _format_cart_lines(result.cart)]
        )
    return message


def _format_update_cart_result(
    result: CartMutationResult,
    cart: Cart | CommerceCart | None = None,
) -> str:
    if not result.ok:
        message = result.error.message if result.error else "unknown error"
        return f"CART UPDATE FAILED: {message}"
    lines = ["CART UPDATED"]
    if result.changed_line:
        lines.append(
            f"  {result.changed_line.display_name} → "
            f"qty {result.changed_line.quantity}"
        )
    active_cart = cart if cart is not None else result.cart
    if active_cart is not None:
        lines.append(_format_cart_lines(active_cart))
    return "\n".join(lines)


def _format_policy_result(result: GetStorePolicyResult) -> str:
    if not result.ok or result.policy is None:
        message = result.error.message if result.error else "unknown error"
        return f"POLICY NOT AVAILABLE: {message}"
    policy = result.policy
    return f"STORE POLICY — {policy.title}\n{policy.body}"


def _format_availability_result(result: CheckProductAvailabilityResult) -> str:
    return f"AVAILABILITY ({result.product_ref}): {result.message}"


def _format_promotions_result(result: CheckActivePromotionsResult) -> str:
    status = "YES" if result.active else "NO"
    return f"ACTIVE PROMOTIONS: {status}\n{result.message}"


def _format_cart_total(cart: Cart) -> str:
    if not cart.contents:
        return "Your cart is empty, so the total is $0.00."
    subtotal = 0.0
    missing = []
    lines = []
    for item in cart.contents:
        name = item.get("item", "")
        amount = int(item.get("amount") or 0)
        price = item.get("price")
        if price is None:
            missing.append(name)
            lines.append(f"- {amount} x {name}: price unavailable")
            continue
        line_total = float(price) * amount
        subtotal += line_total
        lines.append(f"- {amount} x {name} @ ${float(price):.2f} = ${line_total:.2f}")
    total = f"Cart total: ${subtotal:.2f}"
    if missing:
        total += f" excluding items without cached prices: {', '.join(missing)}"
    return "\n".join(lines + [total])


def _format_shopper_context(context: ShopperContext | None) -> str:
    if context is None:
        return ""
    return (
        "SHOPPER CONTEXT (server-resolved; soft guidance only):\n"
        f"shopper_type: {context.shopper_type}\n"
        f"behavior: {context.behavior}\n"
        f"saved_zipcode: {context.zipcode}\n"
        "END SHOPPER CONTEXT"
    )


def _format_active_weather_receipts(
    receipts: Sequence[WeatherForecastReceipt],
) -> str:
    """Render activation scope only; forecast evidence stays server-side."""

    if not receipts:
        return "VALID DURABLE WEATHER RECEIPTS (activation scopes):\n(none)"
    payload = [
        {
            "receipt_id": receipt.receipt_id,
            "receipt_type": receipt.receipt_type,
            "location_scope": receipt.location_scope.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "requested_window": receipt.evidence.requested_window.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "valid_until": receipt.valid_until.isoformat(),
        }
        for receipt in receipts
    ]
    return (
        "VALID DURABLE WEATHER RECEIPTS "
        "(activation scopes only; forecast facts remain server-side; bind "
        "exactly one during activation only for the unchanged event "
        "location/date scope; unbound receipts are not evidence):\n"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def _shopper_authored_texts(state: State) -> tuple[str, ...]:
    """Return bounded shopper text for exact event-location provenance."""

    return tuple(
        dict.fromkeys(
            text.strip()
            for text in [state.query, *_recent_shopper_texts(state.context)]
            if text.strip()
        )
    )


def _current_turn_weather_date_available(state: State) -> bool:
    """Narrow activation only from current-turn bounded date authority."""

    return weather_date_context_available((state.query,))


def _saved_zip_authorized_for_weather(state: State) -> bool:
    """Recognize a narrow, recent confirmation before releasing saved ZIP."""

    if state.shopper_context is None:
        return False

    current = state.query.strip()
    if _saved_area_override_present(current) or _contains_exact_zip(current):
        return False
    recent_turns = _recent_conversation_turns(state.context)
    if _saved_area_confirmation_present(current):
        return True
    if (
        _bare_saved_area_confirmation(current)
        and recent_turns
        and _assistant_asked_about_saved_area(recent_turns[-1][1])
    ):
        return True

    if not _date_only_followup(current) or not recent_turns:
        return False
    prior_shopper = recent_turns[-1][0]
    if (
        _saved_area_override_present(prior_shopper)
        or _contains_exact_zip(prior_shopper)
    ):
        return False
    if _saved_area_confirmation_present(prior_shopper):
        return True
    return (
        _bare_saved_area_confirmation(prior_shopper)
        and len(recent_turns) >= 2
        and _assistant_asked_about_saved_area(recent_turns[-2][1])
    )


def _recent_conversation_turns(
    context: str,
) -> list[tuple[str, str]]:
    return [
        (match.group(1).strip(), match.group(2).strip())
        for match in _RECENT_CONVERSATION_TURN_RE.finditer(context)
    ]


def _recent_shopper_texts(context: str) -> list[str]:
    turns = _recent_conversation_turns(context)
    if turns:
        return [shopper for shopper, _assistant in turns]
    return [
        line.removeprefix("User:").strip()
        for line in context.splitlines()
        if line.startswith("User:")
    ]


def _redact_prior_weather_assistant_text(context: str) -> str:
    """Remove stale forecast facts while retaining prior styling discussion."""

    if not context:
        return context
    redacted: list[str] = []
    weather_markers = (
        "Forecast location used:",
        'Interpreting "next week" as ',
        *(
            f'Interpreting "{weekday} next week" as '
            for weekday in (
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            )
        ),
        "Live forecast:",
        "Live forecast for the event window:",
        VISUAL_CROSSING_ATTRIBUTION_LABEL,
        VISUAL_CROSSING_ATTRIBUTION_URL,
    )
    for line in context.splitlines():
        marker_positions = [
            position
            for marker in weather_markers
            if (position := line.find(marker)) >= 0
        ]
        if line.startswith("Assistant: ") and marker_positions:
            forecast_start = min(marker_positions)
            if forecast_start >= 0:
                styling_prefix = line[:forecast_start].rstrip()
                if styling_prefix != "Assistant:":
                    redacted.append(
                        f"{styling_prefix} {_PRIOR_WEATHER_CONTEXT_REDACTION}"
                    )
                    continue
            redacted.append("Assistant: " + _PRIOR_WEATHER_CONTEXT_REDACTION)
        else:
            redacted.append(line)
    return "\n".join(redacted)


def _summary_safe_assistant_text(text: str) -> str:
    """Remove canonical stale forecast facts before semantic compaction."""

    inline = " ".join(text.split())
    redacted = _redact_prior_weather_assistant_text(f"Assistant: {inline}")
    return redacted.removeprefix("Assistant: ").strip()


def _saved_area_confirmation_present(text: str) -> bool:
    if (
        "?" in text
        or _saved_area_override_present(text)
        or _contains_exact_zip(text)
        or _SAVED_AREA_CONFIRMATION_RE.search(text) is None
    ):
        return False
    remaining = _SAVED_AREA_CONFIRMATION_RE.sub(" ", text)
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z]+", remaining, flags=re.ASCII)
    }
    return words.issubset(_SAVED_AREA_CONFIRMATION_CONTEXT_WORDS)


def _saved_area_override_present(text: str) -> bool:
    return (
        _SAVED_AREA_OVERRIDE_RE.search(text) is not None
        or _SAVED_AREA_MODAL_UNCERTAINTY_RE.search(text) is not None
    )


def _contains_exact_zip(text: str) -> bool:
    return (
        re.search(
            r"(?<![0-9])[0-9]{5}(?![0-9])",
            text,
            flags=re.ASCII,
        )
        is not None
    )


def _bare_saved_area_confirmation(text: str) -> bool:
    normalized = re.sub(
        r"[^a-z0-9']+",
        " ",
        text.replace("’", "'").lower(),
        flags=re.ASCII,
    ).strip()
    return normalized in _SAVED_AREA_BARE_CONFIRMATIONS


def _assistant_asked_about_saved_area(text: str) -> bool:
    return (
        "?" in text
        and _SAVED_AREA_QUESTION_RE.search(text) is not None
    )


def _date_only_followup(text: str) -> bool:
    if (
        not text.strip()
        or _contains_exact_zip(text)
        or _saved_area_override_present(text)
    ):
        return False
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z]+", text, flags=re.ASCII)
    }
    return words.issubset(_DATE_ONLY_CONTEXT_WORDS)


def _format_retrieved_images(retrieved: dict[str, str] | None) -> str:
    if not retrieved:
        return "(none)"
    lines = []
    for name, image_url in retrieved.items():
        lines.append(f"- {name}: image available")
    return "\n".join(lines)


def _format_media_summary(media: list[dict[str, Any]]) -> str:
    if not media:
        return "(none)"
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown")
        counts[media_type] = counts.get(media_type, 0) + 1
    return ", ".join(f"{count} {media_type}(s)" for media_type, count in sorted(counts.items()))


def _cart_line_by_id(cart_line_id: str, cart: Cart) -> dict[str, Any] | None:
    if not cart.contents:
        return None
    target = (cart_line_id or "").strip()
    if not target:
        return None
    for item in cart.contents:
        if str(item.get("cart_line_id") or "").strip() == target:
            return item
    return None
