from fastapi import FastAPI, HTTPException
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

@app.post("/rail/input/check")
async def check_input(request: QueryRequest):
    return await rails.call_input_content_rails(request.query)

@app.post("/rail/input/timing")
async def timing_input(request: QueryRequest):
    start = time.monotonic()
    response = await check_input(request)
    end = time.monotonic()
    logging.info(f"Guardrails | check_input | Time: {end - start}")
    response["timings"] = [{"rails": end - start}, {"total": end - start}]
    return response

@app.post("/rail/output/check")
async def check_output(request: QueryRequest):
    return await rails.call_output_content_rails(request.query)

@app.post("/rail/output/timing")
async def timing_output(request: QueryRequest):
    start = time.monotonic()
    response = await check_output(request)
    end = time.monotonic()
    logging.info(f"Guardrails | check_output | Time: {end - start}")
    response["timings"] = [{"rails": end - start}, {"total": end - start}]
    return response
