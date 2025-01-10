FROM python:3.12.3-slim as builder

WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install and configure Poetry
RUN pip install poetry && \
    poetry config virtualenvs.create false

# Copy dependency files and install
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-root --without dev

# Runtime environment
FROM python:3.12.3-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Security: Create non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["poetry", "run", "python", "app.py"]