# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY catalog_retriever/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY catalog_retriever/src ./app

# Copy entrypoint script
COPY catalog_retriever/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Bundle product data and images (for automatic initialization)
COPY shared/data ./bundled_data/data/
COPY shared/images ./bundled_data/images/

# Default port
EXPOSE 8010

# Run via entrypoint script (handles data initialization)
CMD ["/app/entrypoint.sh"]
