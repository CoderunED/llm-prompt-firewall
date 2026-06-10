# ── Stage 1: dependency cache ─────────────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /app

# System libs needed by sentence-transformers (tokenizers builds from Rust)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY app/        ./app/
COPY dashboard.py .

# Persistent volume mount point for logs (SQLite + JSONL)
RUN mkdir -p logs && touch logs/.gitkeep

# Non-root user — good practice for container security (relevant for your portfolio)
RUN useradd --no-create-home --shell /bin/false firewall
RUN chown -R firewall:firewall /app
USER firewall

EXPOSE 8000
EXPOSE 8501

# Default: run the FastAPI service
# Docker Compose overrides this per-service
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
