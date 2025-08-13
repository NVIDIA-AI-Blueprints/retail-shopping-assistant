# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and config
COPY ./src ./app

# Set environment variables
ENV PYTHONPATH=/app
ENV LLM_API_KEY=${LLM_API_KEY}
ENV CATALOG_RETRIEVER_URL=http://catalog-retriever:8010
ENV MEMORY_RETRIEVER_URL=http://memory-retriever:8011
ENV RAILS_URL=http://rails:8012

# Expose the port the app runs on
EXPOSE 8009

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8009", "--reload"]