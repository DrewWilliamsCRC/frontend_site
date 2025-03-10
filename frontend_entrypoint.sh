#!/bin/sh
set -e

# Print environment for debugging (excluding secrets)
echo "Frontend starting with environment:"
env | grep -v "SECRET\|KEY\|PASSWORD" | sort

# Wait for database to be ready
echo "Verifying database connection..."
for i in $(seq 1 15); do
    if PGPASSWORD=$POSTGRES_PASSWORD psql -h db -U $POSTGRES_USER -d $POSTGRES_DB -c '\q' >/dev/null 2>&1; then
        echo "Database connection verified!"
        break
    fi
    echo "Waiting for database connection... (Attempt $i/15)"
    sleep 2
done

# Wait for AI server to be ready
echo "Verifying AI server connection..."
for i in $(seq 1 15); do
    if curl -s http://ai_server:5002/health > /dev/null; then
        echo "AI server connection verified!"
        break
    fi
    echo "Waiting for AI server connection... (Attempt $i/15)"
    sleep 2
done

# Start the Flask application
echo "Starting Flask application..."
cd /build
export PYTHONPATH=/build:$PYTHONPATH
exec "$@" 