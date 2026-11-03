# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY ./src ./app

# Default port
EXPOSE 8010

# Run FastAPI
# Searches are short, so this window only has to clear one in-flight request.
# `exec` so SIGTERM reaches uvicorn rather than stopping at the shell.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8010 \
  --timeout-graceful-shutdown ${SHUTDOWN_GRACE_SECONDS:-30}"]
