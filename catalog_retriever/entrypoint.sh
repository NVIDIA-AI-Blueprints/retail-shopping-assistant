#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Entrypoint script that initializes shared data if not present

set -e

echo "CATALOG RETRIEVER | entrypoint | Starting initialization..."

# Check if shared data directory exists and has the required CSV
if [ ! -f "/app/shared/data/products_extended.csv" ]; then
    echo "CATALOG RETRIEVER | entrypoint | Product data not found. Copying from bundled data..."
    
    # Create directories if they don't exist
    mkdir -p /app/shared/data
    mkdir -p /app/shared/images
    
    # Copy bundled data to shared volume
    if [ -d "/app/bundled_data/data" ]; then
        cp -r /app/bundled_data/data/* /app/shared/data/ 2>/dev/null || true
        echo "CATALOG RETRIEVER | entrypoint | Copied product data to shared volume."
    fi
    
    if [ -d "/app/bundled_data/images" ]; then
        cp -r /app/bundled_data/images/* /app/shared/images/ 2>/dev/null || true
        echo "CATALOG RETRIEVER | entrypoint | Copied product images to shared volume."
    fi
else
    echo "CATALOG RETRIEVER | entrypoint | Product data already exists in shared volume."
fi

echo "CATALOG RETRIEVER | entrypoint | Initialization complete. Starting application..."

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8010
