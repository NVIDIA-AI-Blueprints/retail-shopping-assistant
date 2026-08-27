# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use an official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./src ./app

# Expose port
EXPOSE 8011

# Run the app. --limit-concurrency is the third of the three limits that have
# to agree with MEMORY_MAX_CONCURRENT_REQUESTS; it is what turns overload into
# an immediate 503 that a caller can retry, instead of a queue of requests
# waiting on connections that are all taken.
#
# `exec` because the shell form is only here to expand the variable. Without it
# the shell stays as PID 1, SIGTERM stops at the shell, and every deploy kills
# uvicorn mid-request ten seconds later instead of draining it.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8011 --limit-concurrency ${MEMORY_MAX_CONCURRENT_REQUESTS:-64} \
  --timeout-graceful-shutdown ${SHUTDOWN_GRACE_SECONDS:-30}"]
