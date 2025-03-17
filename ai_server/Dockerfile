# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages with optimizations
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a minimal healthcheck script
RUN echo '#!/bin/sh\necho "Checking AI server health..."\ncurl -fs http://localhost:5002/health >/dev/null || {\n  echo "Health check failed"\n  exit 1\n}\necho "Health check passed"\n' > /usr/local/bin/healthcheck.sh && \
    chmod +x /usr/local/bin/healthcheck.sh

# Final stage
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 aiuser

# Copy only necessary files from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/healthcheck.sh /usr/local/bin/
COPY --chown=aiuser:aiuser . /app
COPY --chown=aiuser:aiuser ai_entrypoint.sh /usr/local/bin/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=ai_server.py \
    FLASK_ENV=testing

# Set working directory
WORKDIR /app

# Make scripts executable
RUN chmod +x /usr/local/bin/ai_entrypoint.sh /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER aiuser

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/healthcheck.sh"]

# Start the application
ENTRYPOINT ["/usr/local/bin/ai_entrypoint.sh"] 