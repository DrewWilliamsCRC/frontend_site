ARG PYTHON_VERSION=3.10-slim

# Build stage
FROM python:${PYTHON_VERSION} as builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Create and set working directory
WORKDIR /build

# Install build dependencies - separate this to cache better
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install basic build dependencies - we don't need the extensive AI dependencies anymore
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools first - this changes less frequently
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first for better layer caching
COPY requirements-frontend.txt ./requirements.txt
COPY build-helpers/ ./build-helpers/

# Install critical packages first - these rarely change
RUN pip install Flask flask-caching Flask-WTF gunicorn python-dotenv Flask-Limiter requests psycopg2-binary click urllib3 zipp

# Install remaining dependencies
RUN chmod +x ./build-helpers/install-deps.sh && \
    ./build-helpers/install-deps.sh && \
    # Save the result of pip freeze for runtime
    pip freeze > /build/requirements-freeze.txt

# Final stage
FROM python:${PYTHON_VERSION}

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PATH="/usr/local/bin:$PATH"

# Create and set working directory
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage only what's needed
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Install critical Python packages directly in the final stage too
# This ensures they're available even in case of any issues with the copied site-packages
RUN pip install python-dotenv flask flask-caching flask-wtf flask-limiter gunicorn requests psycopg2-binary click urllib3

# Create non-root user early for better layer caching
RUN useradd -m -u 1000 appuser

# Copy and set entrypoint script first (changes less frequently)
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application code (changes most frequently, keep at end)
COPY app.py wsgi.py ./
COPY static/ ./static/
COPY templates/ ./templates/
COPY src/ ./src/

# Set permissions after copying all files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set entrypoint and default command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app:app"]