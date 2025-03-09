#!/bin/sh
set -e

# Create log file if it doesn't exist
mkdir -p /app/logs
touch /app/logs/ai_server.log
chmod 777 /app/logs/ai_server.log

# Print environment for debugging (excluding secrets)
echo "AI Server starting with environment:"
env | grep -v "SECRET\|KEY\|PASSWORD" | sort

# Check if running in CI mode
CI_MODE=${CI_BUILD:-false}
echo "CI Mode: $CI_MODE"

# Create all required directories with proper permissions
mkdir -p /app/ai_experiments/data
mkdir -p /app/ai_experiments/models
mkdir -p /tmp/ai_cache
chmod -R 777 /app/ai_experiments/data /app/ai_experiments/models /tmp/ai_cache

# Verify database connection only if not in CI mode
if [ "$CI_MODE" != "true" ]; then
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
    try:
        from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI
        print('Initializing Alpha Vantage API...')
        api = AlphaVantageAPI()
        print('Alpha Vantage API initialized')
    except Exception as e:
        print(f'Error initializing Alpha Vantage API: {str(e)}')
        # Continue anyway
    "
else
    echo "Running in CI mode - skipping database check and initialization"
fi

# Run the command passed to docker
echo "Starting AI server: $@"
exec "$@" 