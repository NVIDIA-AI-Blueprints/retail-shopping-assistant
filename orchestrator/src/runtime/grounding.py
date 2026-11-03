# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The grounding editor's rewrite of a draft reply, and the search-only fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..agenttypes import State
from ..cart_format import _format_cart, format_cart_change
from ..conversation_products import format_historical_product_index
from ..fencing import MEDIA_FENCE
from ..tools.policy import load_shopper_skill_registry as _shopper_skill_registry
from .grounding_evidence import _collect_tool_grounding_evidence
from .message_shape import _content_to_text, _value
from .model_usage import _add_model_usage, _collect_token_usage, _merge_token_usage
from .prompts import _GROUNDING_EDITOR_SYSTEM_PROMPT, _format_retrieved_images
from .replies import (
    _GROUNDING_FAILURE_RESPONSE,
    _format_search_only_response,
    _has_grounding_authority,
    _has_search_only_tool_evidence,
    _scrub_internal_shopper_language,
    _search_guidance_evidence,
)
from .turn_diagnostics import (
    _catalog_repair_clarification_response,
    _has_successful_non_search_tool_evidence,
    _rejected_catalog_search_response,
)

logger = logging.getLogger(__name__)


class GroundingMixin:
    """The `DeepAgentsRuntime` methods that ground a draft reply in the turn's evidence."""

    def _grounding_reserve(self) -> float:
        return min(
            max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "grounding_editor_reserve_seconds",
                        0.0,
                    )
                ),
            ),
            max(
                0.0,
                float(self.config.deepagents_execution_timeout_seconds),
            )
            / 2,
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
        search_only = bool(current_evidence) and _has_search_only_tool_evidence(
            result,
            request_id=request_id,
        )
        if not getattr(self.config, "grounding_rewrite_enabled", True):
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            return _scrub_internal_shopper_language(draft_response)
        if not draft_response:
            if not search_only:
                return draft_response
            return self._rewrite_search_only_response(
                state,
                result,
                request_id=request_id,
            )
        # No evidence is a reason to run the editor, not to skip it: those are
        # the turns where the model improvises about its own machinery.

        # Fail-closed guards unsupported product facts. A turn with no
        # authority has none, so an editor failure degrades to the draft.
        has_grounding_authority = _has_grounding_authority(state, current_evidence)
        termination_reason_before_editor = state.agent_diagnostics.get(
            "final_termination_reason"
        )

        start = time.monotonic()
        # Separated authority lanes. Each is labelled with what it may be used
        # for, because a lane that mixes intent with identity gives the editor
        # no way to tell which half can support a factual claim.
        prompt = (
            f"USER QUERY:\n{state.query}\n\n"
            "CURRENT-TURN TOOL EVIDENCE (facts established this turn):\n"
            f"{current_evidence or '(none)'}\n\n"
            "PRODUCTS SHOWN EARLIER (identity only — not current facts):\n"
            f"{format_historical_product_index(state.historical_product_sets) or '(none)'}\n\n"
            f"CURRENT CART (authoritative):\n{_format_cart(state.cart)}\n\n"
            "WHAT THIS TURN DID TO THE CART (authoritative, computed):\n"
            f"{format_cart_change(state.cart_at_turn_start, state.cart)}\n\n"
            "CONVERSATION (shopper intent only — never a product fact):\n"
            f"{state.dialogue_context or '(none)'}\n\n"
            # Without this lane the editor cuts descriptions of the shopper's
            # media, since it removes what no lane supports.
            # Fenced: this text describes an untrusted upload, and without a
            # boundary it could read like an instruction we set.
            "WHAT THE SHOPPER'S MEDIA SHOWED (sight, never a catalog fact):\n"
            f"{MEDIA_FENCE.wrap(state.media_analysis) or '(none)'}\n\n"
            f"AVAILABLE IMAGES:\n{_format_retrieved_images(state.retrieved)}\n\n"
            f"DRAFT RESPONSE:\n{draft_response}"
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
                self._create_chat_model(
                    max_tokens=self.config.grounding_editor_max_output_tokens
                ).ainvoke(
                    [
                        {
                            "role": "system",
                            "content": _GROUNDING_EDITOR_SYSTEM_PROMPT,
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
            if not has_grounding_authority:
                # The turn itself completed; only the optional tidy-up failed,
                # and that is already recorded in model usage. Leaving an error
                # termination reason here would mark a good turn as failed.
                if termination_reason_before_editor is None:
                    state.agent_diagnostics.pop("final_termination_reason", None)
                else:
                    state.agent_diagnostics["final_termination_reason"] = (
                        termination_reason_before_editor
                    )
                return _scrub_internal_shopper_language(draft_response)
            return _GROUNDING_FAILURE_RESPONSE
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
            if not has_grounding_authority:
                # The turn itself completed; only the optional tidy-up failed,
                # and that is already recorded in model usage. Leaving an error
                # termination reason here would mark a good turn as failed.
                if termination_reason_before_editor is None:
                    state.agent_diagnostics.pop("final_termination_reason", None)
                else:
                    state.agent_diagnostics["final_termination_reason"] = (
                        termination_reason_before_editor
                    )
                return _scrub_internal_shopper_language(draft_response)
            return _GROUNDING_FAILURE_RESPONSE

        state.timings["grounding_rewrite"] = time.monotonic() - start
        rewrite_usage = _collect_token_usage(rewrite_result)
        state.token_usage = _merge_token_usage(state.token_usage, rewrite_usage)
        rewrite_tokens = int(rewrite_usage.get("total_tokens") or 0)

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
                tokens=rewrite_tokens,
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if not has_grounding_authority:
                # The turn itself completed; only the optional tidy-up failed,
                # and that is already recorded in model usage. Leaving an error
                # termination reason here would mark a good turn as failed.
                if termination_reason_before_editor is None:
                    state.agent_diagnostics.pop("final_termination_reason", None)
                else:
                    state.agent_diagnostics["final_termination_reason"] = (
                        termination_reason_before_editor
                    )
                return _scrub_internal_shopper_language(draft_response)
            return _GROUNDING_FAILURE_RESPONSE
        _add_model_usage(
            state,
            "app_llm_grounding_editor",
            status="used",
            calls=1,
            detail="Final response grounding rewrite",
            tokens=rewrite_tokens,
        )
        return _scrub_internal_shopper_language(rewritten)

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
