FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy only dependency files first for better caching
COPY pyproject.toml poetry.lock ./

# Install dependencies only
RUN poetry install --no-root --no-interaction --no-ansi --without dev

# Copy the rest of the application
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY utils/ ./utils/
COPY app.py ./

# Install the project itself
RUN poetry install --only-root

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Security: Create non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["poetry", "run", "python", "app.py"]