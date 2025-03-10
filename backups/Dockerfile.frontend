FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Create and set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    python3-dev \
    postgresql-client \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools
RUN pip install --upgrade pip setuptools wheel

# Create a requirements file directly in the Dockerfile
RUN echo "Flask \n\
Werkzeug==3.1.3 \n\
requests==2.32.2 \n\
flask-caching \n\
Flask-WTF \n\
Flask-Limiter \n\
python-dotenv~=0.19.0 \n\
psycopg2-binary \n\
gunicorn \n\
click \n\
zipp==3.21.0 \n\
urllib3==2.2.2 \n\
certifi>=2023.7.22 \n\
charset-normalizer~=2.0.0 \n\
idna>=2.5 \n\
dnspython==2.6.1 \n\
feedparser>=6.0.10 \n\
pandas>=2.0.0 \n\
numpy>=1.24.0" > /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app

# Make entrypoint script executable first
COPY frontend_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/frontend_entrypoint.sh

# Switch to non-root user
USER appuser

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/frontend_entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app:app"] 