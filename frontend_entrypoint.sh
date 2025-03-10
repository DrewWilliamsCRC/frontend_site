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

# Debug info about the environment
echo "Starting Flask application..."
echo "Current directory: $(pwd)"
echo "Directory contents: $(ls -la)"
echo "Python path: $PYTHONPATH"
echo "Python version: $(python --version)"

# Verify app.py exists
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found in $(pwd)"
    echo "Contents of directory:"
    ls -la
    exit 1
fi

# Check for required Python modules
echo "Checking for Flask..."
python -c "import flask; print(f'Flask version: {flask.__version__}')" || echo "WARNING: Flask import failed"

# Change to app directory
cd /app
export PYTHONPATH=/app:$PYTHONPATH

# Run the command with error handling
echo "Executing: $@"
exec "$@" || {
    echo "ERROR: Command failed with status $?"
    echo "Last 20 lines of log (if available):"
    if [ -f "/app/logs/gunicorn.log" ]; then
        tail -20 /app/logs/gunicorn.log
    fi
    if [ -f "/app/logs/gunicorn-error.log" ]; then
        echo "Error log contents:"
        tail -20 /app/logs/gunicorn-error.log
    fi
    exit 1
} 