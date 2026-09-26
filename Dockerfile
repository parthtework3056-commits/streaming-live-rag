FROM python:3.11-slim

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . /workspace

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    SLRAG_CORPUS_ISOLATION=true

EXPOSE 8000

# Default command launches FastAPI SSE Gateway
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
