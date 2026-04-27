# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .agenttypes import Cart, State
from .functions import (
    add_to_cart_function,
    bulk_add_to_cart_function,
    bulk_remove_from_cart_function,
    remove_from_cart_function,
    view_cart_function,
    view_cart_total_function,
    parse_tool_call_fallback,
)
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
import json
import logging
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
import time
from typing import Any, Optional
from langgraph.config import get_stream_writer


_PRICE_PATTERN = re.compile(r"PRICE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# Matches **Product Name** spans emitted by the chatter in its user-facing
# responses. We post-filter the captures against
# ``_looks_like_product_name`` to reject bolded price/heading spans like
# "**Price: $69.99**".
_BOLD_NAME_RE = re.compile(r"\*\*([^*\n]+?)\*\*")

# Matches the catalog-retriever's "NAME | description | category" row format.
# We only need the name (before the first pipe) to harvest candidates.
_CATALOG_ROW_RE = re.compile(r"^([^|\n]+?)\s+\|\s+", re.MULTILINE)

# A product-name-shaped string: at least two tokens, each token made of
# letters, digits, hyphens, or apostrophes. Excludes currency, colons, and
# other punctuation that shows up in incidental bold spans.
_PRODUCT_NAME_SHAPE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9'\-]*(?:\s+[A-Za-z0-9'\-]+){1,}$"
)

# Words/phrases the user can use to refer to a product without naming it.
# Kept deliberately narrow: adding ambiguous tokens like "the dress" would
# fire the override for legitimate new-product queries.
_PRONOUN_REFERENCE_RE = re.compile(
    r"\b(it|this|that|one|them|these|those|both)\b",
    re.IGNORECASE,
)


def _extract_price(catalog_text: Optional[str]) -> Optional[float]:
    """Pull the ``PRICE: X.XX`` token embedded in a catalog description.

    Used to cache a unit price per cart line so cart totals are computed
    deterministically rather than asking the LLM to do arithmetic.
    """
    if not catalog_text:
        return None
    match = _PRICE_PATTERN.search(catalog_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for name compare.

    Used purely for string comparison; no catalog-specific assumptions.
    """
    if not name:
        return ""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _resolve_catalog_match(
    query_name: str,
    catalog_names: list,
    similarities: list,
    min_similarity: float = 0.5,
    min_token_overlap: float = 0.5,
) -> Optional[int]:
    """Pick the best index in ``catalog_names`` for ``query_name``.

    Name-based matching is preferred over raw embedding similarity because
    distinctive product names are systematically underrated by similarity
    against full descriptions. Resolution order:
        1. Exact normalized name equality.
        2. Normalized substring containment either way.
        3. Highest Jaccard token overlap above ``min_token_overlap``.
        4. Fallback to the top embedding hit if it clears ``min_similarity``.

    Returns the chosen index, or None if nothing plausibly matches.
    """
    if not catalog_names:
        return None

    q_norm = _normalize_name(query_name)
    if not q_norm:
        if similarities and similarities[0] >= min_similarity:
            return 0
        return None

    q_tokens = set(q_norm.split())
    best_overlap = 0.0
    best_idx: Optional[int] = None
    for idx, candidate in enumerate(catalog_names):
        c_norm = _normalize_name(candidate)
        if not c_norm:
            continue
        if q_norm == c_norm:
            return idx
        if q_norm in c_norm or c_norm in q_norm:
            return idx
        c_tokens = set(c_norm.split())
        if not c_tokens or not q_tokens:
            continue
        overlap = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is not None and best_overlap >= min_token_overlap:
        return best_idx

    if similarities and similarities[0] >= min_similarity:
        return 0
    return None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

# Configuration will be loaded by the main application

class CartAgent():
    """
    CartAgent is an agent which manages a user's cart.
    It can perform a few actions:
    - add_to_cart : Adds a product to the cart.
    - remove_from_cart : Removes a product from the cart.
    - view_cart : Reports what is in the cart.
    """
    def __init__(self,
        config,
    ) -> None:
        logging.info(f"CartAgent.__init__() | Initializing with llm_name={config.llm_name}, llm_port={config.llm_port}")
        self.llm_name = config.llm_name
        self.llm_port = config.llm_port
        
        # Store configuration
        self.memory_retriever_url = config.memory_port
        self.model = OpenAI(base_url=config.llm_port, api_key=os.environ["LLM_API_KEY"])
        self.catalog_retriever_port = config.retriever_port
        self.categories = config.categories
        self.retry_strategy = Retry(
                total=3,                    
                status_forcelist=[422, 429, 500, 502, 503, 504],  
                allowed_methods=["POST"],   
                backoff_factor=1            
            )
        logging.info(f"CartAgent.__init__() | Initialization complete")
        
    def _get_cart(self, user_id: int) -> Cart:
        response = requests.get(f"{self.memory_retriever_url}/user/{user_id}/cart")
        logging.info(f"CartAgent._get_cart() | Response text: {response.text}.")
        if response.status_code == 200:
            cart_data = json.loads(response.text)["cart"]
            return Cart(contents=cart_data)
        return Cart(contents=[])

    _CATALOG_LOOKUP_K = 5

    def _lookup_in_catalog(self, item_name: str) -> Optional[dict]:
        """Look up a product in the catalog by name.

        Returns ``{"name", "text", "similarity"}`` for the best match per
        ``_resolve_catalog_match`` or None. ``k`` is widened so the right
        record is present even when embedding similarity ranks it below
        the top hit.
        """
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        logging.info(f"CartAgent._lookup_in_catalog() | /query/text -- query: {item_name}")
        ret_response = session.post(
            f"{self.catalog_retriever_port}/query/text",
            json={
                "text": [item_name],
                "categories": self.categories,
                "k": self._CATALOG_LOOKUP_K,
            },
        )
        ret_response.raise_for_status()
        res_json = ret_response.json()
        names = res_json.get("names") or []
        similarities = res_json.get("similarities") or []
        texts = res_json.get("texts") or []

        match_idx = _resolve_catalog_match(item_name, names, similarities)
        if match_idx is None:
            logging.info(
                f"CartAgent._lookup_in_catalog() | no match for '{item_name}' "
                f"(candidates={names[:self._CATALOG_LOOKUP_K]}, sims={similarities[:self._CATALOG_LOOKUP_K]})"
            )
            return None

        similarity = similarities[match_idx] if match_idx < len(similarities) else 0.0
        text = texts[match_idx] if match_idx < len(texts) else None
        logging.info(
            f"CartAgent._lookup_in_catalog() | query='{item_name}' -> "
            f"matched='{names[match_idx]}' sim={similarity}"
        )
        return {"name": names[match_idx], "text": text, "similarity": similarity}

    def _add_to_cart(self, user_id: int, item_name: str, quantity: int) -> str:
        match = self._lookup_in_catalog(item_name)
        if match is None:
            return f"No such item ({item_name}) could be found in the catalog."

        catalog_item_name = match["name"]
        price = _extract_price(match.get("text"))
        payload = {"item": catalog_item_name, "amount": quantity}
        if price is not None:
            payload["price"] = price
        response = requests.post(
            f"{self.memory_retriever_url}/user/{user_id}/cart/add",
            json=payload,
        )
        if response.status_code == 200:
            return response.json()["message"]
        return f"Failed to add {quantity} {catalog_item_name} to cart."

    def _view_cart_total(self, user_id: int) -> str:
        """Compute the cart total deterministically from cached prices.

        LLMs are unreliable for arithmetic, so we sum line totals server-side
        using the per-line price stored when the item was added. Missing prices
        are reported explicitly rather than silently dropped.
        """
        cart = self._get_cart(user_id)
        if not cart.contents:
            return "Your cart is empty, so the total is $0.00."

        lines = []
        subtotal = 0.0
        missing_price: list[str] = []
        for entry in cart.contents:
            item_name = entry.get("item", "")
            amount = int(entry.get("amount", 0) or 0)
            price = entry.get("price")
            if price is None:
                missing_price.append(item_name)
                lines.append(f"- {amount} x {item_name}: price unavailable")
                continue
            line_total = float(price) * amount
            subtotal += line_total
            lines.append(
                f"- {amount} x {item_name} @ ${float(price):.2f} = ${line_total:.2f}"
            )

        summary = "\n".join(lines)
        total_line = f"Cart total: ${subtotal:.2f}"
        if missing_price:
            names = ", ".join(missing_price)
            total_line += (
                f" (excluding items without a cached price: {names}. "
                "Re-add them to include their price in the total.)"
            )
        return f"{summary}\n{total_line}"

    def _remove_from_cart(self, user_id: int, item_name: str, quantity: int) -> str:
        match = self._lookup_in_catalog(item_name)
        if match is None:
            return f"No such item ({item_name}) could be found in the catalog."

        catalog_item_name = match["name"]
        response = requests.post(
            f"{self.memory_retriever_url}/user/{user_id}/cart/remove",
            json={"item": catalog_item_name, "amount": quantity},
        )
        if response.status_code == 200:
            return response.json()["message"]
        return f"Failed to remove {quantity} {catalog_item_name} from cart."

    @staticmethod
    def _looks_like_product_name(candidate: str) -> bool:
        """Reject non-name bold spans (``**Price: $69.99**``, ``**Tip:**``).

        Product names are at least two tokens of letters/digits, with no
        punctuation beyond hyphens or apostrophes.
        """
        return bool(_PRODUCT_NAME_SHAPE_RE.match(candidate or ""))

    @staticmethod
    def _collect_known_products(state: State) -> list[str]:
        """Gather product names the user could plausibly be referring to.

        Sources (deduped case-insensitively, preserving first-seen casing):
          1. Cart contents (trusted; added unconditionally).
          2. Catalog rows ``Name | description | category`` in ``state.context``.
          3. ``**Name**`` spans in ``state.context``.

        Context-derived sources are filtered through
        ``_looks_like_product_name`` because the running context also
        contains prose with stray ``" | "`` and bolded price/heading spans.
        """
        seen: dict[str, str] = {}

        def _add(candidate: str, trusted: bool = False) -> None:
            name = (candidate or "").strip()
            if not name:
                return
            if not trusted and not CartAgent._looks_like_product_name(name):
                return
            key = _normalize_name(name)
            if key and key not in seen:
                seen[key] = name

        if state.cart and state.cart.contents:
            for entry in state.cart.contents:
                _add(entry.get("item", ""), trusted=True)

        ctx = state.context or ""
        for match in _CATALOG_ROW_RE.findall(ctx):
            _add(match)
        for match in _BOLD_NAME_RE.findall(ctx):
            _add(match)

        return list(seen.values())

    @staticmethod
    def _is_pronoun_reference(query: str) -> bool:
        """True if the query contains a pronoun-style reference.

        A pronoun alone isn't sufficient (a query can both pronoun and
        name a product). Combined with the product-name scan, it's the
        trigger for the focus-item fallback.
        """
        return bool(_PRONOUN_REFERENCE_RE.search(query or ""))

    @staticmethod
    def _find_named_product(query: str, known: list[str]) -> Optional[str]:
        """Return the known product the query most specifically names.

        Users abbreviate catalog names, so strict substring matching is
        not enough. Each candidate is scored with a symmetric token
        overlap::

            score = max(|q ∩ n| / |q|, |q ∩ n| / |n|)

        which keeps long and short queries on equal footing. A candidate
        is chosen only if it (1) scores >= 0.5, (2) beats the runner-up
        by >= 0.2, and (3) shares a signal token with the query.

        Signal tokens are derived from the catalog itself: only tokens
        that appear in at least one ``known`` product name count. Any
        query word outside that vocabulary ("please", "add", "the",
        and their equivalents in any other language) is treated as
        filler. This avoids a curated stopword list and stays correct
        as the catalog (or its language) evolves.
        """
        q_norm = _normalize_name(query)
        if not q_norm:
            return None

        candidates: list[tuple[str, str, set[str]]] = []
        catalog_vocab: set[str] = set()
        for name in known:
            n_norm = _normalize_name(name)
            if not n_norm:
                continue
            n_tokens = set(n_norm.split())
            if not n_tokens:
                continue
            candidates.append((name, n_norm, n_tokens))
            catalog_vocab |= n_tokens

        if not candidates:
            return None

        q_tokens = set(q_norm.split()) & catalog_vocab
        if not q_tokens:
            return None

        scored: list[tuple[float, str]] = []
        for name, n_norm, n_tokens in candidates:
            if n_norm in q_norm:
                return name
            shared = q_tokens & n_tokens
            if not shared:
                continue
            q_cov = len(shared) / len(q_tokens)
            n_cov = len(shared) / len(n_tokens)
            scored.append((max(q_cov, n_cov), name))

        if not scored:
            return None

        scored.sort(key=lambda entry: entry[0], reverse=True)
        best_score, best_name = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < 0.5:
            return None
        if best_score - runner_up < 0.2:
            return None
        return best_name

    @staticmethod
    def _last_mentioned_product(known: list[str], context: str) -> Optional[str]:
        """Return the product most recently mentioned in ``context``.

        "Most recent" is the rightmost occurrence across all known names;
        used as the deterministic anchor for pronoun-only cart requests.
        """
        if not known or not context:
            return None
        ctx_lower = context.lower()
        best_name: Optional[str] = None
        best_pos = -1
        for name in known:
            needle = name.lower()
            if not needle:
                continue
            pos = ctx_lower.rfind(needle)
            if pos > best_pos:
                best_pos = pos
                best_name = name
        return best_name

    def _resolve_target_item(self, state: State) -> Optional[str]:
        """Deterministically pick the product a cart request is targeting.

        Resolution order:
          1. Named product found in the query -> use it.
          2. Pronoun-style query -> fall back to the last-mentioned product.
          3. Otherwise return None (defer to the LLM's choice).

        The catalog lookup still runs after this, so a bad anchor can
        only shift which real product gets picked, not invent one.
        """
        query = state.query or ""
        known = self._collect_known_products(state)
        if not known:
            return None

        named = self._find_named_product(query, known)
        if named:
            return named

        if self._is_pronoun_reference(query):
            return self._last_mentioned_product(known, state.context or "")

        return None

    def _override_bulk_item_names(
        self, items: list, state: State
    ) -> list:
        """Apply the deterministic resolver to each entry in a bulk tool call.

        For ``bulk_add_to_cart`` / ``bulk_remove_from_cart`` the LLM provides
        per-item names, so pronoun resolution (which operates over the full
        query) is not useful. Instead we re-anchor each name against the set
        of products actually present in cart + context. This catches the same
        class of mistake the single-item override does (LLM paraphrasing a
        catalog name) without affecting cases where the LLM got it right.

        Mutates and returns ``items`` for convenience.
        """
        if not items:
            return items
        known = self._collect_known_products(state)
        if not known:
            return items

        for entry in items:
            if not isinstance(entry, dict):
                continue
            llm_pick = entry.get("item_name") or ""
            resolved = self._find_named_product(llm_pick, known)
            if (
                resolved
                and _normalize_name(llm_pick) != _normalize_name(resolved)
            ):
                logging.warning(
                    f"CartAgent.invoke() | overriding bulk item_name "
                    f"llm={llm_pick!r} -> deterministic={resolved!r}"
                )
                entry["item_name"] = resolved
        return items

    @staticmethod
    def _coerce_bulk_items(raw: Any) -> list:
        """Accept the ``items`` argument in its various possible shapes.

        Nemotron's XML fallback surfaces ``items`` as a string repr that has
        already been parsed to a list by ``_coerce_value`` in almost all
        cases. A defensive re-parse here makes the code tolerant of any
        edge-case model output without special casing it upstream.
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return []
            try:
                import ast
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(stripped)
                except (ValueError, TypeError):
                    return []
            return parsed if isinstance(parsed, list) else []
        return []

    @staticmethod
    def _extract_recent_discussion(context: str, max_chars: int = 2000) -> str:
        """Return the tail of the conversation context to focus pronoun resolution.

        The running context string grows unboundedly per turn. When resolving
        pronouns like "it" in cart requests, only the most recent exchange is
        relevant; older product mentions and prior cart actions create noise
        that can cause the LLM to misidentify the target item.
        """
        if not context:
            return "(no prior discussion)"
        trimmed = context.strip()
        if len(trimmed) <= max_chars:
            return trimmed
        tail = trimmed[-max_chars:]
        newline_idx = tail.find("\n")
        if 0 < newline_idx < max_chars // 4:
            tail = tail[newline_idx + 1:]
        return "...\n" + tail

    def _update_context(self, user_id: int, context: str) -> None:
        response = requests.post(
            f"{self.memory_retriever_url}/user/{user_id}/context/add",
            json={"new_context": context}
        )
        if response.status_code != 200:
            logging.error(f"Failed to update context: {response.text}")

    def invoke(
        self,
        state: State,
        verbose : bool = True
    ) -> State:
        """
        Determines which function to perform and does that function using NVIDIA NIM.
        """
        start = time.monotonic()
        logging.info(f"CartAgent.invoke() | Starting with query: {state.query}")
        tools = [
            add_to_cart_function,
            remove_from_cart_function,
            bulk_add_to_cart_function,
            bulk_remove_from_cart_function,
            view_cart_function,
            view_cart_total_function,
        ]

        system_prompt = (
            "You are a retail cart manager. Your ONLY job is to execute exactly one cart "
            "tool call that fulfils the user's CURRENT QUERY. Do not return plain text.\n\n"
            "TOOL SELECTION (choose exactly one):\n"
            "- add_to_cart: user wants to put ONE item IN the cart. Triggers: 'add', "
            "'put in cart', \"I'll take\", 'buy', 'get me', 'include'.\n"
            "- bulk_add_to_cart: user wants to put TWO OR MORE distinct items IN the cart "
            "in the same request. Prefer this over multiple add_to_cart calls whenever the "
            "user enumerates several products (separated by commas, 'and', 'also', 'plus', "
            "'as well as', etc.). Populate 'items' with one entry per named product.\n"
            "- remove_from_cart: user wants to take ONE item OUT. Triggers: 'remove', "
            "'take out', 'delete', 'drop'.\n"
            "- bulk_remove_from_cart: user wants to take TWO OR MORE distinct items OUT in "
            "the same request. Prefer this over multiple remove_from_cart calls whenever "
            "the user enumerates several products to remove.\n"
            "- view_cart: user wants to SEE cart contents. Triggers: \"what's in my cart\", "
            "'show my cart', 'view cart', 'check my cart'. Do NOT use view_cart when the "
            "user is asking to add or remove an item.\n"
            "- view_cart_total: user wants the MONETARY TOTAL of the cart. Triggers: "
            "\"what's my total\", 'how much is my cart', 'cart subtotal', \"how much do I owe\", "
            "'total cost', 'sum of my cart'. Prefer this over view_cart whenever the user "
            "asks about a price, sum, or total. Never attempt arithmetic yourself.\n\n"
            "REFERENCE RESOLUTION:\n"
            "When the user says 'it', 'this', 'that', 'the one', 'another', etc., resolve "
            "the pronoun to the MOST RECENT specific product in RECENT DISCUSSION. Give the "
            "MOST RECENT ASSISTANT MESSAGE the highest weight, then the user's last query, "
            "then older context. Do NOT default to items already in the cart.\n\n"
            "ITEM NAME RULES (apply to item_name for single tools AND to every items[].item_name "
            "for bulk tools):\n"
            "- Copy the full product name VERBATIM from RECENT DISCUSSION. Do not "
            "shorten it, do not substitute a category word, do not paraphrase.\n"
            "- Examples of the same rule applied across product types (these names "
            "are illustrative only; use the actual name from RECENT DISCUSSION):\n"
            "    * Discussed: 'Alpine Waterproof Hiking Boot'. User says 'add it to "
            "my cart' -> item_name = 'Alpine Waterproof Hiking Boot'. NOT 'boot' "
            "or 'hiking boot'.\n"
            "    * Discussed: 'Pearl Drop Stud Earrings'. User says 'buy the "
            "pearls' -> item_name = 'Pearl Drop Stud Earrings'. NOT 'pearls' or "
            "'earrings'.\n"
            "    * Discussed: 'Midnight Velvet Blazer'. User says 'add the blue "
            "one' -> item_name = 'Midnight Velvet Blazer'. NOT 'blue one' or "
            "'blazer'.\n"
            "    * Discussed: 'Bamboo Slim-Fit Chinos'. User says 'add those' -> "
            "item_name = 'Bamboo Slim-Fit Chinos'. NOT 'those' or 'chinos'.\n"
            "    * Discussed: 'Honey Floral Print Midi Skirt', 'Lace and Silk Blouse', "
            "'Pearl Bracelet'. User says 'please add the Honey Floral Print Midi Skirt, "
            "the Lace and Silk Blouse, and the Pearl Bracelet to my cart' -> call "
            "bulk_add_to_cart with items=[{item_name: 'Honey Floral Print Midi Skirt', "
            "quantity: 1}, {item_name: 'Lace and Silk Blouse', quantity: 1}, "
            "{item_name: 'Pearl Bracelet', quantity: 1}].\n"
            "- If the user specifies a quantity use it; otherwise default to 1.\n"
            "- Ignore minor typos in the user's query.\n"
        )

        recent_discussion = self._extract_recent_discussion(state.context)
        cart_contents = (
            [f"{c.get('amount', 1)} x {c.get('item', '')}" for c in state.cart.contents]
            if state.cart and state.cart.contents else []
        )

        user_content = (
            f"CURRENT QUERY: {state.query}\n\n"
            f"CURRENT CART: {', '.join(cart_contents) if cart_contents else 'empty'}\n\n"
            f"RECENT DISCUSSION (most recent first, use this to resolve pronouns):\n"
            f"{recent_discussion}"
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Create the request parameters
        response = self.model.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            tools=tools,
            tool_choice="auto",
            stream=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        message = response.choices[0].message
        tool_name = None
        tool_args = {}

        if message.tool_calls:
            called_tool = message.tool_calls[0]
            tool_name = called_tool.function.name
            tool_args = json.loads(called_tool.function.arguments)
        else:
            logging.warning(f"CartAgent.invoke() | No structured tool_calls returned. Content: {message.content}")
            tool_name, tool_args = parse_tool_call_fallback(message.content)

        if not tool_name:
            logging.error("CartAgent.invoke() | Could not determine tool call from model response.")
            output_state = state
            output_state.response = "I couldn't process that cart action. Could you please rephrase your request?"
            end = time.monotonic()
            output_state.context = output_state.context + f"\nAgent Response: {output_state.response}"
            output_state.timings["cart"] = end - start
            return output_state

        logging.info(f"CartAgent.invoke() | Tool name: {tool_name}")

        # Override ``item_name`` for add/remove with a deterministic
        # resolver. Coreference over a multi-KB context is where the
        # model silently picks the wrong product; anchored signals from
        # cart + context fix that.
        if tool_name in ("add_to_cart", "remove_from_cart"):
            resolved = self._resolve_target_item(state)
            if resolved:
                llm_pick = tool_args.get("item_name")
                if not llm_pick or _normalize_name(llm_pick) != _normalize_name(resolved):
                    logging.warning(
                        f"CartAgent.invoke() | overriding item_name "
                        f"llm={llm_pick!r} -> deterministic={resolved!r}"
                    )
                    tool_args["item_name"] = resolved
        elif tool_name in ("bulk_add_to_cart", "bulk_remove_from_cart"):
            # Normalize ``items`` shape and fix up any per-entry names the
            # model paraphrased. Quantity defaults live here too so the
            # dispatch branches don't need to repeat the coercion.
            items = self._coerce_bulk_items(tool_args.get("items"))
            items = self._override_bulk_item_names(items, state)
            tool_args["items"] = items

        output_state = state 
        if verbose:
            logging.info(f"CartAgent.invoke() | tool_name: {tool_name}\n\t| tool_args: {tool_args}")

        # Perform our associated action.
        if tool_name == "add_to_cart":
            logging.info(f"CartAgent.invoke() | Adding to cart")
            item_name = tool_args["item_name"]
            quantity = tool_args["quantity"]
            output_state.response = self._add_to_cart(state.user_id, item_name, quantity)
            output_state.cart = self._get_cart(state.user_id)
            
        elif tool_name == "remove_from_cart":
            logging.info(f"CartAgent.invoke() | Removing from cart")
            item_name = tool_args["item_name"]
            quantity = tool_args["quantity"]    
            output_state.response = self._remove_from_cart(state.user_id, item_name, quantity)
            output_state.cart = self._get_cart(state.user_id)

        elif tool_name == "bulk_add_to_cart":
            items = tool_args.get("items") or []
            logging.info(
                f"CartAgent.invoke() | Bulk adding {len(items)} item(s) to cart"
            )
            lines = []
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                name = (entry.get("item_name") or "").strip()
                if not name:
                    continue
                try:
                    quantity = int(entry.get("quantity", 1) or 1)
                except (TypeError, ValueError):
                    quantity = 1
                lines.append(self._add_to_cart(state.user_id, name, quantity))
            if lines:
                output_state.response = "\n".join(lines)
            else:
                output_state.response = (
                    "No items were specified to add to the cart."
                )
            output_state.cart = self._get_cart(state.user_id)

        elif tool_name == "bulk_remove_from_cart":
            items = tool_args.get("items") or []
            logging.info(
                f"CartAgent.invoke() | Bulk removing {len(items)} item(s) from cart"
            )
            lines = []
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                name = (entry.get("item_name") or "").strip()
                if not name:
                    continue
                try:
                    quantity = int(entry.get("quantity", 1) or 1)
                except (TypeError, ValueError):
                    quantity = 1
                lines.append(self._remove_from_cart(state.user_id, name, quantity))
            if lines:
                output_state.response = "\n".join(lines)
            else:
                output_state.response = (
                    "No items were specified to remove from the cart."
                )
            output_state.cart = self._get_cart(state.user_id)

        elif tool_name == "view_cart":
            cart = self._get_cart(state.user_id)
            logging.info(f"CartAgent.invoke() | Viewing cart.\n\t| Cart: {cart}")
            if len(cart.contents) == 0:
                output_state.response = "Your cart is empty."
            else:
                contents = cart.contents
                items = [f"The user has ({contents[ind]['amount']} {contents[ind]['item']}) in their cart" for ind in range(len(contents))]
                items_str = ". ".join(items)
                logging.info(f"CartAgent.invoke() | item list retrieved: {items_str}")
                output_state.response = f"{items_str}"
            output_state.cart = cart

        elif tool_name == "view_cart_total":
            logging.info("CartAgent.invoke() | Computing cart total.")
            output_state.response = self._view_cart_total(state.user_id)
            output_state.cart = self._get_cart(state.user_id)

        # Update our context and return our state.
        if verbose:
            logging.info(f"CartAgent.invoke() | output_state: {output_state}")
        
        #self._update_context(state.user_id, f"USER QUERY:{output_state.query}\nRESPONSE:{output_state.response}")
        end = time.monotonic()
        output_state.context = output_state.context + f"\nAgent Response: {output_state.response}"
        output_state.timings["cart"] = end - start
        logging.info(f"CartAgent.invoke() | Returning final state with response: {output_state.response}")

        return output_state


