# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use an official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./src/requirements.txt .
RUN apt-get update && apt-get install -y gcc g++ python3-dev
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-deps --upgrade nemoguardrails==0.19.0

# Copy app code
COPY ./src .

ENV NVIDIA_API_KEY=${NVIDIA_API_KEY}

# Expose port
EXPOSE 8012

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
