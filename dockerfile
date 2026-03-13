# Base image that matches your local Python exactly
FROM python:3.10.11-slim

# Prevents Python from writing .pyc files & enables clean logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside container
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install uv and use it to install dependencies (faster, lockfile-friendly)
RUN pip install uv \
    && uv pip install --system -r requirements.txt

# Copy full project
COPY . .

# Expose Flask port (8080: universal on Windows/macOS/Linux)
EXPOSE 8080

# Start server with Uvicorn (FastAPI)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

