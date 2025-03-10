#!/bin/sh
set -e

# Print environment for debugging (excluding secrets)
echo "AI Server starting with environment:"
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

# Start the Flask application with gunicorn
echo "Starting Flask application with gunicorn..."
cd /app
export PYTHONPATH=/app:$PYTHONPATH
exec gunicorn --bind 0.0.0.0:5002 --workers 2 --timeout 120 --graceful-timeout 60 ai_server:app