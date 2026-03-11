FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY static/ static/

# Render sets PORT dynamically; default to 8000 for local use
ENV PORT=8000
EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
