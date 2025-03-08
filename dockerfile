ARG PYTHON_VERSION=3.10-alpine

# Build stage
FROM python:${PYTHON_VERSION} as builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Create and set working directory
WORKDIR /build

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    make \
    postgresql-dev \
    musl-dev \
    linux-headers \
    build-base \
    python3-dev \
    openblas-dev \
    lapack-dev \
    freetype-dev \
    libpng-dev \
    libjpeg-turbo-dev \
    pkgconfig \
    gfortran \
    cmake

# Install critical packages first
RUN pip install --upgrade pip setuptools wheel && \
    pip install Flask flask-caching Flask-WTF gunicorn python-dotenv Flask-Limiter requests psycopg2-binary click urllib3 zipp

# Copy requirements and installation scripts
COPY requirements.txt ./
COPY build-helpers/ ./build-helpers/

# Install dependencies
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
RUN apk add --no-cache \
    postgresql-client \
    curl \
    libstdc++

# Copy from builder stage only what's needed
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Install critical Python packages directly in the final stage too
# This ensures they're available even in case of any issues with the copied site-packages
RUN pip install python-dotenv flask flask-caching flask-wtf flask-limiter gunicorn requests psycopg2-binary click urllib3 pandas numpy

# Copy application code
COPY . .

# Copy and set entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create non-root user
RUN adduser -D appuser && chown -R appuser:appuser /app
USER appuser

# Set entrypoint and default command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app:app"]