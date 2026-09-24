"""What a turn did, recorded for reading afterwards.

Lifted out of `turn_support.py` unchanged.
"""


from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import (
    BaseModel,
)

from .control_signals import (
    not_carried_of,
    rejections_of,
)
from .message_shape import (
    _content_to_text,
    _current_turn_messages,
    _message_type,
    _result_messages,
    _tool_results_by_call_id,
    _value,
)
from .search_input import _UNSUPPORTED_SEARCH_MODE_MESSAGE
from .tools.evidence import (
    detail_evidence_of,
    evidence_of,
)
from .tools.loop_control import (
    _SERVER_REJECTED_TOOL_CALLS,
    SEARCH_VALIDATION_ERROR_PREFIX,
    SERVER_CATALOG_CLARIFICATION,
    SERVER_RESTORED_TOOL_CALL_FIELDS,
    STOP_TOOL_USE_PREFIX,
    UNSUPPORTED_CONSTRAINT_PREFIX,
    UNSUPPORTED_TAXONOMY_PREFIX,
)
from .tools.skill_gate import (
    SKILL_ACTIVATION_REQUIRED,
    SKILL_ACTIVATION_TOOL_NAME,
    SKILL_TOOL_NOT_GRANTED,
)

logger = logging.getLogger(__name__)


_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE = 24



_MAX_DIAGNOSTIC_PRODUCT_FACTS = 40



_MAX_DIAGNOSTIC_PRODUCT_STRING_CHARS = 500



_MAX_DIAGNOSTIC_PRODUCT_EVIDENCE_CHARS = 32_000



_REJECTED_CATALOG_SEARCH_RESPONSE = (
    "I couldn't complete a valid catalog search for that request, so I don't "
    "have catalog results to show. Please try again or ask me to search a "
    "different advertised product type."
)



_CATALOG_REPAIR_CLARIFICATION_RESPONSE = (
    "Could you clarify the product type or requirement you want me to use?"
)



def _empty_agent_diagnostics(final_termination_reason: str) -> dict[str, Any]:
    return {
        "skill_files_read": [],
        "tool_calls": [],
        "rejected_tool_calls": [],
        "duplicate_tool_calls": [],
        "product_evidence": [],
        "product_evidence_truncated": False,
        "catalog_scope_outcomes": [],
        "shopper_sizes": [],
        "final_termination_reason": final_termination_reason,
        "partial_graph_messages": [],
    }



def _collect_agent_diagnostics(
    messages: list[Any],
    *,
    request_id: str,
    final_termination_reason: str,
    preserve_partial_messages: bool = False,
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
                "arguments": call["arguments"],
                "status": status,
            }
            if rejection_reason:
                entry["rejection_reason"] = rejection_reason
            scope_rejections = rejections_of(result_message)
            if len(scope_rejections) > 1 and any(scope_rejections):
                # A call that searched several roles and was refused only some
                # of them is not a rejected call, so its refusals would
                # otherwise be counted nowhere at all.
                entry["scope_rejections"] = scope_rejections
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
    diagnostics["shopper_sizes"] = _diagnostic_shopper_sizes(turn_messages)
    if preserve_partial_messages:
        partial, truncated = _serialize_partial_graph_messages(turn_messages)
        diagnostics["partial_graph_messages"] = partial
        if truncated:
            diagnostics["partial_graph_messages_truncated"] = True
    return diagnostics



def _safe_collect_agent_diagnostics(
    messages: list[Any],
    *,
    request_id: str,
    final_termination_reason: str,
    preserve_partial_messages: bool = False,
) -> dict[str, Any]:
    """Collect diagnostics without allowing tracing to change turn behavior."""

    try:
        return _collect_agent_diagnostics(
            messages,
            request_id=request_id,
            final_termination_reason=final_termination_reason,
            preserve_partial_messages=preserve_partial_messages,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must fail independently.
        error_type = type(exc).__name__
        logger.warning("Could not collect Deep Agents diagnostics: %s", error_type)
        diagnostics = _empty_agent_diagnostics(final_termination_reason)
        diagnostics["diagnostic_collection_error"] = error_type
        return diagnostics



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
            if tool_name == SKILL_ACTIVATION_TOOL_NAME:
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
        if _search_reported_not_covered(messages):
            # The scopes were invalid, but the call also named what this catalog
            # does not carry, and NOT_COVERED evidence went back to the model.
            # That is a fact for the model to speak to, not a reason to seize
            # the turn: asked to compare two aprons in a catalog with none, the
            # model correctly sent not_covered and had its answer replaced by
            # "I couldn't complete a valid catalog search", five turns running.
            return None
        if _search_reported_not_carried(messages):
            # The same fact, established by the tool rather than volunteered by
            # the model. Leaving it to the model meant leaving it to chance:
            # replayed five times, it filled `not_covered` unprompted in four
            # and in the fifth wrote the right answer -- "aprons aren't a
            # product type this store carries" -- only to have it replaced by
            # the refusal. Nothing about a catalog that carries no aprons
            # differed between those runs.
            return None
        return _REJECTED_CATALOG_SEARCH_RESPONSE
    return None



def _search_reported_not_covered(messages: Any) -> bool:
    """Whether a search this turn named product kinds the catalog lacks."""

    for message in messages:
        if _message_type(message) != "ai":
            continue
        for raw_call in (_value(message, "tool_calls") or []):
            call = raw_call if isinstance(raw_call, dict) else {}
            if (call.get("name") or "") != "search_catalog_tool":
                continue
            args = call.get("args")
            if isinstance(args, dict) and args.get("not_covered"):
                return True
    return False



def _search_reported_not_carried(messages: Any) -> bool:
    """Whether the search tool established a product type as not carried."""

    for message in messages:
        if _message_type(message) != "tool":
            continue
        if (_value(message, "name") or "") != "search_catalog_tool":
            continue
        if not_carried_of(message):
            return True
    return False



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



def _tool_call_status(
    tool_name: str,
    result_message: Any | None,
) -> tuple[str, str | None]:
    if result_message is None:
        return "pending", None
    content = _content_to_text(_value(result_message, "content"))
    rejection_reason = _tool_rejection_reason(
        content,
        rejections_of(result_message),
    )
    if rejection_reason:
        return "rejected", rejection_reason
    if _value(result_message, "status") == "error":
        return "error", None
    return "completed", None



def _tool_rejection_reason(
    content: str,
    scope_rejections: list[Any] | None = None,
) -> str | None:
    """Name what refused this call, preferring what the gate itself recorded.

    Nine catalog-search gates render one model-visible prefix, so the text can
    only ever say ``invalid_catalog_request``; a gate that recorded its own code
    names itself instead. The code is believed only when every searched scope
    carries one, because a call that refused one role and answered another is
    not a refused call. A result carrying no codes -- an older checkpoint, or
    one of the paths that deliberately records none -- falls back to matching
    the text, so a missed path degrades to today's answer rather than losing
    its reason.
    """

    codes = list(scope_rejections or ())
    if codes and all(codes):
        return str(codes[0])
    markers = (
        (SKILL_ACTIVATION_REQUIRED, "skill_activation_required"),
        (SKILL_TOOL_NOT_GRANTED, "skill_tool_not_granted"),
        ("SHOPPER_SKILL_ACTIVATION_FAILED:", "skill_activation_failed"),
        (
            f"{STOP_TOOL_USE_PREFIX} This catalog taxonomy and constraint scope was already searched",
            "duplicate_catalog_scope",
        ),
        (f"{STOP_TOOL_USE_PREFIX} Catalog search limit reached", "catalog_search_limit"),
        (
            f"{STOP_TOOL_USE_PREFIX} Product-detail read limit reached",
            "product_detail_read_limit",
        ),
        (
            "The catalog search request does not match current capabilities:",
            "invalid_catalog_request",
        ),
        (SEARCH_VALIDATION_ERROR_PREFIX, "invalid_catalog_request"),
        (
            UNSUPPORTED_TAXONOMY_PREFIX,
            "unsupported_catalog_taxonomy",
        ),
        (
            UNSUPPORTED_CONSTRAINT_PREFIX,
            "unsupported_catalog_constraint",
        ),
        (_UNSUPPORTED_SEARCH_MODE_MESSAGE, "unsupported_search_mode"),
    )
    for marker, reason in markers:
        if content.startswith(marker):
            return reason
    if content.startswith(STOP_TOOL_USE_PREFIX):
        return "stop_tool_use"
    return None



def _skill_file_paths(call: dict[str, Any], status: str) -> list[str]:
    if status != "completed":
        return []
    if call["tool_name"] != SKILL_ACTIVATION_TOOL_NAME:
        return []
    names = call["arguments"].get("skill_names") or []
    if not isinstance(names, list):
        return []
    return [
        f"/shopper/{name}/SKILL.md"
        for name in names
        if isinstance(name, str) and name.strip()
    ]



def _serialize_partial_graph_messages(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], bool]:
    relevant = [
        message for message in messages if _message_type(message) in {"ai", "tool"}
    ]
    truncated = len(relevant) > 24
    serialized: list[dict[str, Any]] = []
    for message in relevant[-24:]:
        content = _content_to_text(_value(message, "content"))
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
            payload["tool_calls"] = [
                _normalized_tool_call(call) for call in tool_calls
            ]
        if content_truncated:
            payload["truncated"] = True
            truncated = True
        serialized.append(payload)
    return serialized, truncated



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
        payload = evidence_of(message)
        if source_tool == "search_catalog_tool":
            if not payload or payload.get("outcome") != "results":
                continue
            evidence_type = "search_result"
            search_scope = {
                "taxonomy": _bounded_product_evidence_value(
                    payload.get("taxonomy") or {}
                ),
                "confirmed_filters": _bounded_product_evidence_value(
                    payload.get("confirmed_filters") or {}
                ),
                "composed_role": bool(payload.get("composed_role")),
            }
            records = payload.get("products") or []
        elif source_tool == "get_product_details_tool":
            detail = detail_evidence_of(message)
            if not detail:
                continue
            evidence_type = "product_detail"
            search_scope = None
            records = detail.get("products") or []
        else:
            continue

        for product in records:
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
            # A multi-role call's payload carries the union of every role's
            # taxonomy and filters. The product knows which role retrieved it;
            # prefer that over the union, or the trace records a product as
            # confirmed under a filter it was never searched with.
            product_scope = _product_search_scope(product)
            if product_scope is not None:
                record["search_scope"] = product_scope
            elif search_scope is not None:
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



def _diagnostic_shopper_sizes(messages: list[Any]) -> list[str]:
    """The sizes the shopper's own searches were filtered by this turn.

    A showing made under a size filter is a size-qualified showing: those four
    sandals came back *because* they come in a 7. That fact belonged to the
    turn and was thrown away at the end of it, so "ok, just show me sandals in
    a 7" followed by "add the first one" asked which size -- the shopper having
    said it one turn earlier.

    Recorded per turn rather than per shopper. Nothing here says the shopper
    is a 7; it says this showing was. A later showing carries its own sizes or
    none, and the two never merge.
    """

    sizes: list[str] = []
    for message in messages:
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message) or {}
        outcome = payload.get("scope_outcome") or {}
        if outcome.get("outcome") not in {None, "results"}:
            continue
        for scope in (payload.get("products") or []):
            if not isinstance(scope, dict):
                continue
            confirmed = (scope.get("search_scope") or {}).get("confirmed_filters")
            for value in ((confirmed or {}).get("sizes") or []):
                text = str(value).strip()
                if text and text not in sizes:
                    sizes.append(text)
    return sizes[:4]



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
        "composed_role",
    }
    for message in messages:
        if _message_type(message) != "tool":
            continue
        payload = evidence_of(message)
        outcome = (payload or {}).get("scope_outcome") or {}
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



def _product_search_scope(product: Any) -> dict[str, Any] | None:
    """Return the scope that actually retrieved one product, if it carries one.

    Only a multi-role call stamps this, because only then is the call-level
    scope a union of several roles rather than a description of this product.
    """

    if not isinstance(product, dict):
        return None
    scope = product.get("search_scope")
    if not isinstance(scope, dict):
        return None
    return {
        "taxonomy": _bounded_product_evidence_value(scope.get("taxonomy") or {}),
        "confirmed_filters": _bounded_product_evidence_value(
            scope.get("confirmed_filters") or {}
        ),
        "composed_role": bool(scope.get("composed_role")),
    }



def _diagnostic_product_facts(product: dict[str, Any]) -> dict[str, Any]:
    """Extract bounded, structured facts from one parsed product record."""

    facts: dict[str, Any] = {}
    for key in ("category", "brand", "price"):
        value = product.get(key)
        if value:
            facts[key] = _bounded_product_evidence_value(value)
    facts["image_available"] = bool(product.get("image_url"))

    # Attributes the catalog confirmed on a search result. The composer is
    # allowed to state these, so the evidence trace has to carry them: a fact an
    # observer cannot see the support for is indistinguishable from an invented
    # one, and gets judged as invention however accurate it is.
    attributes = product.get("attributes")
    if isinstance(attributes, dict):
        for name, value in attributes.items():
            if len(facts) >= _MAX_DIAGNOSTIC_PRODUCT_FACTS:
                break
            bounded_name = str(_bounded_product_evidence_value(name))
            if bounded_name and bounded_name not in facts:
                facts[bounded_name] = _bounded_product_evidence_value(value)

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
            "search_catalog_tool",
        }:
            continue
        if _tool_call_status(tool_name, message)[0] == "completed":
            return True
    return False


