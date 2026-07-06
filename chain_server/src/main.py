# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main FastAPI application for the Shopping Assistant API.

This module provides the main API endpoints for the shopping assistant,
including query processing and streaming responses.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal, Any
import base64
import logging
import sys
import time
import json
import re

from .agenttypes import State, Cart
from .config import load_config
from .deepagents_runtime import DeepAgentsRuntime, create_request_identity
from .media_perception import MEDIA_ONLY_QUERY
from shared.model_config import resolve_model_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load configuration and initialize runtime.
try:
    config = load_config()  # Load and validate configuration
    assistant_runtime = DeepAgentsRuntime(config)
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
    context: Optional[str] = ""
    cart: Optional[Cart] = None
    retrieved: Optional[Dict[str, str]] = {}
    guardrails: Optional[bool] = True
    image_bool: bool = False


class QueryResponse(BaseModel):
    """Response model for shopping queries."""
    response: str
    images: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    token_usage: Dict[str, int] = Field(default_factory=dict)


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
        image=first_image,
        media=media,
        context=request.context or "",
        cart=request.cart or Cart(),
        guardrails=request.guardrails,
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
        )
        
        async def send_updates():
            """Generator function for streaming updates."""
            try:
                async for chunk in assistant_runtime.astream(state, identity):
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
        )
        
        # Process query and collect timing data
        start_time = time.monotonic()
        out_state_dict = await assistant_runtime.ainvoke(state, identity)
        end_time = time.monotonic()
        
        logger.info(f"chain-server | /query/timing | Collected state: {out_state_dict}")

        total_time = end_time - start_time

        # Create response with timing information
        response = QueryResponse(
            response=out_state_dict["response"],
            images=out_state_dict.get("images", {}),
            timings=out_state_dict["timings"],
            token_usage=out_state_dict.get("token_usage", {}),
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
    catalog = assistant_runtime.catalog_capabilities()
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
            "health": "/health",
            "docs": "/docs"
        }
    }


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
