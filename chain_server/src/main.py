# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main FastAPI application for the Shopping Assistant API.

This module provides the main API endpoints for the shopping assistant,
including query processing and streaming responses.
"""
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal, Any
import base64
import logging
import os
import sys
import time
import json
import re

from .agenttypes import SHOPPER_PROFILE_ID_PATTERN, State, Cart
from .config import load_config
from .deepagents_runtime import DeepAgentsRuntime
from .turn_support import create_request_identity
from .media_perception import MEDIA_ONLY_QUERY
from .shopper_profiles import (
    ShopperProfile,
    ShopperProfilesClient,
    ShopperProfilesError,
)
from .commerce_tools import get_cart, update_cart_item
from shared.commerce_contracts import GetCartInput, UpdateCartItemInput
from shared.model_config import resolve_model_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def _configure_tracing() -> None:
    """Export OpenTelemetry spans when an OTLP endpoint is configured.

    Configured entirely through the standard ``OTEL_*`` environment variables so
    no chain-server setting has to exist for it, and so the app never names a
    backend: spans go to a collector, and the collector decides where they land.

    Absent ``OTEL_EXPORTER_OTLP_ENDPOINT`` this does nothing at all -- no
    provider, no exporter, no instrumentation -- which is the default and the
    same idiom as ``EXPOSE_AGENT_DIAGNOSTICS``.
    """

    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": os.environ.get("OTEL_SERVICE_NAME", "chain-server")}
            )
        )
        # Batched, never synchronous: the turn budget is 90s and the graph runs
        # under asyncio.wait_for, so a blocking export sits on the shopper's path.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception as exc:  # noqa: BLE001 - tracing must never break startup.
        logger.warning("Could not configure tracing: %s", type(exc).__name__)
    else:
        logger.info("Tracing enabled")


_configure_tracing()

# Load configuration and initialize runtime.
try:
    config = load_config()  # Load and validate configuration
    assistant_runtime = DeepAgentsRuntime(config)
    shopper_profiles_client = ShopperProfilesClient(config.memory_port)
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Shopping Assistant API",
    description="AI-powered shopping assistant with Deep Agents SDK orchestration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class MediaItem(BaseModel):
    """Request model for attached visual media."""

    type: Literal["image", "video"]
    data: str
    mime_type: str = ""
    filename: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for shopping queries."""
    user_id: int
    query: str
    image: str = ""
    media: List[MediaItem] = Field(default_factory=list)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    cart_id: Optional[str] = None
    shopper_profile_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=SHOPPER_PROFILE_ID_PATTERN,
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    context: Optional[str] = ""
    cart: Optional[Cart] = None
    retrieved: Optional[Dict[str, str]] = {}
    guardrails: Optional[bool] = None
    image_bool: bool = False


class QueryResponse(BaseModel):
    """Response model for shopping queries."""
    response: str
    images: Dict[str, str] = {}
    cart: Cart = Field(default_factory=Cart)
    timings: Dict[str, float] = {}
    token_usage: Dict[str, int] = Field(default_factory=dict)
    model_usage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agent_diagnostics: Dict[str, Any] = Field(default_factory=dict)


_MODEL_LABELS = {
    "app_llm": "Language reasoning",
    "vlm": "Vision-language inference",
    "text_embedding": "Text embedding",
    "image_embedding": "Image embedding",
    "content_safety": "Content safety",
    "topic_control": "Topic control",
}


def create_initial_state(request: QueryRequest) -> State:
    """Create initial state from request."""
    media = _normalized_media(request)
    first_image = next(
        (item["data"] for item in media if item.get("type") == "image"),
        request.image,
    )
    return State(
        user_id=request.user_id,
        query=request.query,
        shopper_profile_id=request.shopper_profile_id,
        image=first_image,
        media=media,
        context=request.context or "",
        cart=request.cart or Cart(),
        guardrails=config.guardrails_enabled if request.guardrails is None else request.guardrails,
    )

@app.post("/query/stream")
async def process_query_stream(request: QueryRequest):
    """
    Stream responses to user queries in real-time.
    
    This endpoint provides streaming responses for responsive UIs
    and chat-like experiences.
    """
    try:
        logger.info(f"chain-server | /query/stream | Processing streaming query for user {request.user_id}: {request.query}")
        
        media = _normalized_media(request)
        _validate_media(media)

        # Handle media-only queries
        if media and not request.query:
            request.query = MEDIA_ONLY_QUERY
        
        # Create initial state
        state = create_initial_state(request)
        identity = create_request_identity(
            legacy_user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            cart_id=request.cart_id,
            request_id=request.request_id,
            shopper_profile_id=request.shopper_profile_id,
        )
        
        async def send_updates():
            """Generator function for streaming updates."""
            try:
                async for chunk in assistant_runtime.astream(
                    state,
                    identity,
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'type': 'error', 'payload': str(e)})}\n\n"

        return StreamingResponse(send_updates(), media_type="text/event-stream")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing streaming query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/timing", response_model=QueryResponse)
async def process_query_timing(request: QueryRequest):
    """
    Process a query and return detailed timing information.
    
    This endpoint is useful for performance analysis and debugging.
    """
    try:
        logger.info(f"chain-server | /query/timing | Processing timing query for user {request.user_id}: {request.query}")
        
        media = _normalized_media(request)
        _validate_media(media)
        if media and not request.query:
            request.query = MEDIA_ONLY_QUERY

        # Create initial state
        state = create_initial_state(request)
        identity = create_request_identity(
            legacy_user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            cart_id=request.cart_id,
            request_id=request.request_id,
            shopper_profile_id=request.shopper_profile_id,
        )
        
        # Process query and collect timing data
        start_time = time.monotonic()
        out_state_dict = await assistant_runtime.ainvoke(
            state,
            identity,
        )
        end_time = time.monotonic()
        
        logger.info(f"chain-server | /query/timing | Collected state: {out_state_dict}")

        total_time = end_time - start_time

        # Create response with timing information
        response = QueryResponse(
            response=out_state_dict["response"],
            images=out_state_dict.get("images", {}),
            cart=out_state_dict.get("cart", Cart()),
            timings=out_state_dict["timings"],
            token_usage=out_state_dict.get("token_usage", {}),
            model_usage=out_state_dict.get("model_usage", {}),
            agent_diagnostics=out_state_dict.get("agent_diagnostics", {}),
        )
        response.timings["total"] = total_time

        logger.info(f"chain-server | /query | Successfully processed timing query in {total_time:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing timing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def capabilities():
    """Return runtime capabilities that the UI should enforce."""
    media_config = config.media_input
    catalog = await asyncio.to_thread(assistant_runtime.catalog_capabilities)
    return {
        "media_input": {
            "enabled": media_config.enabled,
            "allow_mixed_media": media_config.allow_mixed_media,
            "max_images_per_turn": media_config.max_images_per_turn,
            "max_videos_per_turn": media_config.max_videos_per_turn,
            "image_mime_types": media_config.image_mime_types,
            "video_mime_types": media_config.video_mime_types,
            "max_image_bytes": media_config.max_image_bytes,
            "max_video_bytes": media_config.max_video_bytes,
            "max_video_duration_seconds": media_config.max_video_duration_seconds,
            "vlm_enabled": config.vlm_enabled,
        },
        "models": _model_capabilities(),
        "catalog": catalog.model_dump(),
    }


@app.get(
    "/shopper-profiles",
    response_model=list[ShopperProfile],
)
async def list_shopper_profiles() -> list[ShopperProfile]:
    """Return the reviewed representative shoppers without changing chat context."""

    try:
        return await asyncio.to_thread(shopper_profiles_client.list_profiles)
    except ShopperProfilesError as exc:
        raise HTTPException(
            status_code=_shopper_profiles_status(exc),
            detail=str(exc),
        ) from exc


class CartLineResponse(BaseModel):
    """One cart line, as a shopper's browser needs to see it."""

    cart_line_id: str
    product_id: str
    display_name: str
    quantity: int
    size: Optional[str] = None
    unit_price: Optional[float] = None


class CartReadResponse(BaseModel):
    lines: List[CartLineResponse] = Field(default_factory=list)
    subtotal: Optional[float] = None


class CartQuantityRequest(BaseModel):
    """An absolute quantity for one line. Zero removes it.

    Absolute rather than a delta, and one verb rather than separate update and
    delete routes, because that is exactly `update_cart_items_tool`'s contract.
    The shopper and the assistant then cannot hold different ideas of what a
    cart mutation is.
    """

    quantity: int = Field(ge=0)
    #: Minted by the caller per intent, and reused only to retry that same
    #: request. Deriving it from the target quantity is tempting and wrong:
    #: a replay returns the stored response without mutating, so 2 -> 3 -> 2
    #: would report success and leave the cart at 3.
    idempotency_key: str = Field(min_length=1, max_length=128)


def _cart_identity(cart_id: str):
    """Resolve the opaque cart handle the browser holds.

    The memory service keys carts on an integer derived from this string. The
    derivation stays on the server: an endpoint taking that integer directly
    would let any caller read or mutate any cart, since the memory service has
    no authentication at all.
    """

    if not cart_id or not cart_id.strip():
        raise HTTPException(status_code=422, detail="cart_id is required")
    return create_request_identity(legacy_user_id=0, cart_id=cart_id.strip())


def _cart_response(cart) -> CartReadResponse:
    return CartReadResponse(
        lines=[
            CartLineResponse(
                cart_line_id=line.cart_line_id,
                product_id=line.product_id,
                display_name=line.display_name,
                quantity=line.quantity,
                size=line.size,
                unit_price=line.unit_price.amount if line.unit_price else None,
            )
            for line in (cart.lines if cart else [])
        ],
        subtotal=cart.subtotal.amount if cart and cart.subtotal else None,
    )


@app.get("/cart", response_model=CartReadResponse)
async def read_cart(cart_id: str) -> CartReadResponse:
    """Read the authoritative cart for a browser-held cart handle."""

    identity = _cart_identity(cart_id)
    result = await asyncio.to_thread(
        get_cart,
        GetCartInput(user_id=str(identity.cart_user_id)),
        config.memory_port,
    )
    if not result.ok:
        raise HTTPException(
            status_code=502,
            detail=result.error.message if result.error else "Could not read cart",
        )
    return _cart_response(result.cart)


@app.patch("/cart/lines/{cart_line_id}", response_model=CartReadResponse)
async def set_cart_line_quantity(
    cart_line_id: str,
    request: CartQuantityRequest,
    cart_id: str,
) -> CartReadResponse:
    """Set one line to an absolute quantity. Zero removes the line."""

    identity = _cart_identity(cart_id)
    result = await asyncio.to_thread(
        update_cart_item,
        UpdateCartItemInput(
            user_id=str(identity.cart_user_id),
            cart_line_id=cart_line_id,
            quantity=request.quantity,
            idempotency_key=request.idempotency_key,
        ),
        config.memory_port,
    )
    if not result.ok:
        code = result.error.code if result.error else ""
        message = result.error.message if result.error else "Cart update failed"
        details = (result.error.details if result.error else None) or {}
        upstream = details.get("status_code")
        if code == "cart_line_not_found":
            # A line the shopper no longer has is their answer, not a fault.
            status = 404
        elif isinstance(upstream, int) and 400 <= upstream < 500:
            # Pass a client error through as one. Reusing an idempotency key
            # for a different quantity is a 409, and a caller that sees 502
            # will retry a request that can only fail the same way.
            status = upstream
            if upstream == 409:
                message = (
                    "That idempotency key was already used for a different "
                    "cart change. Use a new key for a new change."
                )
        else:
            status = 502
        raise HTTPException(status_code=status, detail=message)

    # The mutation returns only the changed line; the panel needs the whole
    # cart, and reading it back is also what proves the change landed.
    read_back = await asyncio.to_thread(
        get_cart,
        GetCartInput(user_id=str(identity.cart_user_id)),
        config.memory_port,
    )
    if not read_back.ok:
        raise HTTPException(status_code=502, detail="Cart updated but could not be read")
    return _cart_response(read_back.cart)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Shopping Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "stream": "/query/stream",
            "timing": "/query/timing",
            "capabilities": "/capabilities",
            "shopper_profiles": "/shopper-profiles",
            "health": "/health",
            "docs": "/docs"
        }
    }


def _shopper_profiles_status(error: ShopperProfilesError) -> int:
    if error.code == "shopper_profiles_response_invalid":
        return 502
    return 503


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)


def _normalized_media(request: QueryRequest) -> List[Dict[str, str]]:
    """Normalize legacy image and media[] into one internal media list."""
    normalized: List[Dict[str, str]] = []
    if request.image.strip():
        image_data = request.image.strip()
        image_mime_type = _mime_from_data_url(image_data) or "image/jpeg"
        if not _mime_from_data_url(image_data):
            image_data = f"data:{image_mime_type};base64,{image_data}"
        normalized.append(
            {
                "type": "image",
                "data": image_data,
                "mime_type": image_mime_type,
            }
        )

    seen_data = {item["data"] for item in normalized}
    for item in request.media:
        data = item.data.strip()
        if not data:
            continue
        mime_type = (item.mime_type or _mime_from_data_url(data) or "").strip()
        if mime_type and not _mime_from_data_url(data):
            data = f"data:{mime_type};base64,{data}"
        if data in seen_data:
            continue
        normalized.append(
            {
                "type": item.type,
                "data": data,
                "mime_type": mime_type,
                "filename": item.filename or "",
            }
        )
        seen_data.add(data)
    return normalized


def _model_capabilities() -> Dict[str, Dict[str, Any]]:
    """Return non-secret model metadata suitable for UI display."""

    models: Dict[str, Dict[str, Any]] = {
        "app_llm": {
            "label": _MODEL_LABELS["app_llm"],
            "model": config.llm_name,
            "source": "configured",
            "enabled": True,
        },
        "vlm": {
            "label": _MODEL_LABELS["vlm"],
            "model": config.vlm_name,
            "source": "configured",
            "enabled": bool(config.vlm_enabled and config.vlm_name),
        },
    }

    try:
        resolved = resolve_model_config()
    except Exception as exc:  # noqa: BLE001 - capabilities should stay best-effort.
        logger.warning("Failed to resolve model capability metadata: %s", exc)
        return models

    for role, endpoint in resolved.models.items():
        models[role] = {
            "label": _MODEL_LABELS.get(role, role.replace("_", " ").title()),
            "model": endpoint.model,
            "source": endpoint.source,
            "enabled": not endpoint.disabled and bool(endpoint.model),
        }

    models["app_llm"].update(
        {
            "model": config.llm_name,
            "enabled": True,
        }
    )
    models["vlm"].update(
        {
            "model": config.vlm_name,
            "enabled": bool(config.vlm_enabled and config.vlm_name),
        }
    )
    return models


def _validate_media(media: List[Dict[str, str]]) -> None:
    if not media:
        return

    media_config = config.media_input
    if not media_config.enabled:
        raise HTTPException(status_code=400, detail="Media uploads are disabled.")

    images = [item for item in media if item.get("type") == "image"]
    videos = [item for item in media if item.get("type") == "video"]
    if len(images) > media_config.max_images_per_turn:
        raise HTTPException(
            status_code=400,
            detail=f"At most {media_config.max_images_per_turn} image(s) are allowed per turn.",
        )
    if len(videos) > media_config.max_videos_per_turn:
        raise HTTPException(
            status_code=400,
            detail=f"At most {media_config.max_videos_per_turn} video(s) are allowed per turn.",
        )
    if not media_config.allow_mixed_media and images and videos:
        raise HTTPException(
            status_code=400,
            detail="Mixed image and video uploads are not enabled.",
        )

    for item in media:
        mime_type = item.get("mime_type") or _mime_from_data_url(item.get("data", ""))
        if item.get("type") == "image":
            _validate_media_item(
                item,
                allowed_mime_types=media_config.image_mime_types,
                max_bytes=media_config.max_image_bytes,
                fallback_label="image",
            )
        elif item.get("type") == "video":
            _validate_media_item(
                item,
                allowed_mime_types=media_config.video_mime_types,
                max_bytes=media_config.max_video_bytes,
                fallback_label="video",
            )
        elif mime_type:
            raise HTTPException(status_code=400, detail="Unsupported media type.")


def _validate_media_item(
    item: Dict[str, str],
    *,
    allowed_mime_types: List[str],
    max_bytes: int,
    fallback_label: str,
) -> None:
    data = item.get("data", "")
    mime_type = item.get("mime_type") or _mime_from_data_url(data)
    if mime_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {fallback_label} MIME type. Allowed: "
                f"{', '.join(allowed_mime_types)}."
            ),
        )

    try:
        byte_count = _data_url_byte_count(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if byte_count > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{fallback_label.title()} upload exceeds the configured size limit.",
        )


def _mime_from_data_url(data: str) -> str:
    match = _DATA_URL_RE.match((data or "").strip())
    return match.group(1).lower() if match else ""


def _data_url_byte_count(data: str) -> int:
    match = _DATA_URL_RE.match((data or "").strip())
    if not match:
        encoded = (data or "").strip()
    else:
        encoded = match.group(2).strip()
    encoded += "=" * (-len(encoded) % 4)
    try:
        return len(base64.b64decode(encoded, validate=True))
    except Exception as exc:  # noqa: BLE001 - normalize validation errors.
        raise ValueError("Media is not valid base64.") from exc
