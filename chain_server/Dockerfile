# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NeMo Relay tracing, absent from the default image on purpose: it requires
# langgraph>=1.2.9 while requirements.txt pins 1.2.7, so installing it moves the
# agent runtime. Keeping it behind a build arg means the image that ships is the
# image that was tested, and tracing can be built and tried without changing it.
#   docker compose build --build-arg INSTALL_RELAY=true chain-server
ARG INSTALL_RELAY=false
COPY requirements-relay.txt .
RUN if [ "$INSTALL_RELAY" = "true" ]; then \
        pip install --no-cache-dir -r requirements-relay.txt; \
    fi

# Copy the application code and config
COPY ./src ./app
COPY ./skills ./skills

# Set environment variables
ENV PYTHONPATH=/app
ENV LLM_API_KEY=${LLM_API_KEY}
ENV CATALOG_RETRIEVER_URL=http://catalog-retriever:8010
ENV MEMORY_RETRIEVER_URL=http://memory-retriever:8011
ENV RAILS_URL=http://rails:8012

# Expose the port the app runs on
EXPOSE 8009

# Command to run the application
# --reload is a development flag: it runs a file-watching parent process, so
# the server is a child and never receives the signal that should drain it.
# CHAIN_SERVER_RELOAD=true brings it back for local work.
#
# --timeout-graceful-shutdown must clear a whole turn. A turn is allowed
# DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS (150 in this deployment), and a rollout
# that waits less than that kills shoppers mid-answer. Kubernetes must be told
# the same thing: terminationGracePeriodSeconds has to exceed this, or the
# kubelet sends SIGKILL while uvicorn is still draining.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8009 \
  --timeout-graceful-shutdown ${SHUTDOWN_GRACE_SECONDS:-160} \
  $([ \"${CHAIN_SERVER_RELOAD:-false}\" = true ] && echo --reload)"]
