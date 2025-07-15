"""
Main FastAPI application for the Shopping Assistant API.

This module provides the main API endpoints for the shopping assistant,
including query processing and streaming responses.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import sys
import time
import json

from .agenttypes import State, Cart
from .planner import PlannerAgent
from .retriever import RetrieverAgent
from .cart import CartAgent
from .chatter import ChatterAgent
from .summarizer import SummaryAgent
from .graph import create_graph
from .config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def initialize_agents(config) -> tuple:
    """Initialize all agent instances."""
    agents = {
        'planner': PlannerAgent(config=config),
        'retriever': RetrieverAgent(config=config),
        'cart': CartAgent(config=config),
        'chatter': ChatterAgent(config=config),
        'summary': SummaryAgent(config=config)
    }
    
    return agents


# Load configuration and initialize agents
try:
    config = load_config()  # Load and validate configuration
    agents = initialize_agents(config)
    graph = create_graph(
        cart_agent=agents['cart'],
        retriever_agent=agents['retriever'],
        planner_agent=agents['planner'],
        chatter_agent=agents['chatter'],
        summary_agent=agents['summary'],
        config=config
    )
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Shopping Assistant API",
    description="AI-powered shopping assistant with multi-agent architecture",
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
class QueryRequest(BaseModel):
    """Request model for shopping queries."""
    user_id: int
    query: str
    image: str = ""
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


def create_initial_state(request: QueryRequest) -> State:
    """Create initial state from request."""
    return State(
        user_id=request.user_id,
        query=request.query,
        image=request.image,
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
        
        # Handle image-only queries
        if request.image and not request.query:
            request.query = "The user has submitted an image, and is looking for items from the catalog that appear similar."
        
        # Create initial state
        state = create_initial_state(request)
        
        async def send_updates():
            """Generator function for streaming updates."""
            try:
                async for chunk in graph.astream(state, stream_mode="custom"):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'type': 'error', 'payload': str(e)})}\n\n"

        return StreamingResponse(send_updates(), media_type="text/event-stream")
        
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
        
        # Create initial state
        state = create_initial_state(request)
        
        # Process query and collect timing data
        start_time = time.monotonic()
        out_state_dict = await graph.ainvoke(state)
        end_time = time.monotonic()
        
        total_time = end_time - start_time

        # Create response with timing information
        response = QueryResponse(
            response=out_state_dict["response"],
            images=out_state_dict.get("image", {}),
            timings=out_state_dict["timings"]
        )
        response.timings["total"] = total_time

        logger.info(f"chain-server | /query | Successfully processed timing query in {total_time:.2f}s")
        return response

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
            "health": "/health",
            "docs": "/docs"
        }
    } 