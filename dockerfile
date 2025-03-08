FROM python:3.10-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

# Create and set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    make \
    postgresql-client \
    postgresql-dev \
    musl-dev \
    linux-headers \
    build-base \
    python3-dev \
    openblas-dev \
    pkgconfig \
    curl

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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