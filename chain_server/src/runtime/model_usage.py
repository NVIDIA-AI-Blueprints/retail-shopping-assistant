# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each model call cost, and what it was for."""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import json
from typing import Any

from ..agenttypes import State
from ..catalog_request import (
    CatalogSearchPlan,
)
from .message_shape import (
    _result_messages,
    _value,
)


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
    # One embed_query call per semantic query, not one per search. A
    # multi-scope search carries several roles in one round trip and embeds
    # each of them, so counting the search under-reports the work.
    text_embedding_calls = len(plan.semantic_queries)
    if uses_image_endpoint and text_embedding_calls == 0:
        # Image retrieval currently includes one deterministic text-side query
        # alongside the image embedding, even when the shopper supplied no text.
        text_embedding_calls = 1
    if fallback_attempted:
        text_embedding_calls += len(plan.semantic_queries)

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
    tokens: int = 0,
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
    entry: dict[str, Any] = {
        "status": merged_status,
        "calls": existing_calls + max(0, calls),
        "detail": next_detail,
    }
    # Tokens are additive across calls, and only chat models report them:
    # embeddings and the guardrails checks have none to report, and showing a
    # zero there would read as "this model used no tokens" rather than "tokens
    # are not a thing this model has".
    merged_tokens = _safe_int(existing.get("tokens")) + max(0, tokens)
    if merged_tokens > 0:
        entry["tokens"] = merged_tokens
    state.model_usage[role] = entry



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



def _token_int(record: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _value(record, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


