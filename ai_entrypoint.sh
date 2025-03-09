#!/bin/sh
set -e

# Create log file if it doesn't exist
mkdir -p /app/logs
touch /app/logs/ai_server.log
chmod 777 /app/logs/ai_server.log

# Print environment for debugging (excluding secrets)
echo "AI Server starting with environment:"
env | grep -v "SECRET\|KEY\|PASSWORD" | sort

# Create all required directories with proper permissions
mkdir -p /app/ai_experiments/data
mkdir -p /app/ai_experiments/models
mkdir -p /tmp/ai_cache
chmod -R 777 /app/ai_experiments/data /app/ai_experiments/models /tmp/ai_cache

# Verify database connection is available
echo "Verifying database connection..."
MAX_RETRIES=15
RETRY_INTERVAL=5
RETRIES=0

while [ $RETRIES -lt $MAX_RETRIES ]; do
    if pg_isready -h db -U $POSTGRES_USER -d $POSTGRES_DB; then
        echo "Database connection successful"
        break
    fi
    
    RETRIES=$((RETRIES+1))
    echo "Waiting for database connection... (Attempt $RETRIES/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
    echo "Could not connect to database after $MAX_RETRIES attempts. Starting anyway but services may fail."
fi

# Initialize any required data
echo "Initializing AI server data..."
python -c "
import sys
sys.path.append('/app')
from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI
print('Initializing Alpha Vantage API...')
api = AlphaVantageAPI()
print('Alpha Vantage API initialized')
"

# Run the command passed to docker
echo "Starting AI server: $@"
exec "$@" 