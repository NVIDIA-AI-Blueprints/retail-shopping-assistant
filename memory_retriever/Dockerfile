# Use an official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./src ./app

# Expose port
EXPOSE 8011

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]
