# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from rails import Rails
from pydantic import BaseModel
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the ContextUpdate data class
class QueryRequest(BaseModel):
    user_id: int
    query: str

# Create the FastAPI app
app = FastAPI()

rails = Rails().getGuardRails()


def _add_timings(response: Any, elapsed: float) -> dict[str, Any]:
    """Return a mutable JSON response with endpoint timing metadata."""
    payload = jsonable_encoder(response)
    if not isinstance(payload, dict):
        raise TypeError("Guardrails response must serialize to a JSON object")
    payload["timings"] = [{"rails": elapsed}, {"total": elapsed}]
    return payload

@app.post("/rail/input/check")
async def check_input(request: QueryRequest):
    return await rails.call_input_content_rails(request.query)

@app.post("/rail/input/timing")
async def timing_input(request: QueryRequest):
    start = time.monotonic()
    response = await check_input(request)
    end = time.monotonic()
    logging.info(f"Guardrails | check_input | Time: {end - start}")
    return _add_timings(response, end - start)

@app.post("/rail/output/check")
async def check_output(request: QueryRequest):
    return await rails.call_output_content_rails(request.query)

@app.post("/rail/output/timing")
async def timing_output(request: QueryRequest):
    start = time.monotonic()
    response = await check_output(request)
    end = time.monotonic()
    logging.info(f"Guardrails | check_output | Time: {end - start}")
    return _add_timings(response, end - start)
