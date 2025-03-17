#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        echo "Python version: $PYTHON_VERSION"
    else
        echo "Python3 not found"
        exit 1
    fi
}

# Function to check Flask installation
check_flask() {
    echo "Checking for Flask..."
    if command_exists python3; then
        FLASK_VERSION=$(python3 -c 'import flask; print(flask.__version__)')
        echo "Flask version: $FLASK_VERSION"
    else
        echo "Flask not found"
        exit 1
    fi
}

# Function to check environment variables
check_env_vars() {
    echo "Loading environment variables from .env file..."
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi

    # Check for required environment variables
    local required_vars=("SECRET_KEY" "FLASK_ENV" "DATABASE_URL" "AI_SERVER_URL")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "Warning: $var not found in environment variables"
        fi
    done

    # Check for optional API keys
    if [ -z "$ALPHA_VANTAGE_API_KEY" ]; then
        echo "Warning: ALPHA_VANTAGE_API_KEY not found in environment variables"
        echo "Warning: ALPHA_VANTAGE_API_KEY not set. Stock ticker will display error message."
    fi

    if [ -z "$GUARDIAN_API_KEY" ]; then
        echo "Warning: GUARDIAN_API_KEY not found in environment variables"
    fi
}

# Function to wait for AI server
wait_for_ai_server() {
    echo "Waiting for AI server connection..."
    local max_attempts=15
    local attempt=1
    local wait_time=2

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://ai_server:5001/health > /dev/null; then
            echo "AI server is ready!"
            return 0
        fi
        echo "Waiting for AI server connection... (Attempt $attempt/$max_attempts)"
        sleep $wait_time
        attempt=$((attempt + 1))
    done

    echo "Error: Could not connect to AI server after $max_attempts attempts"
    return 1
}

# Main execution
echo "Starting Flask application..."
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

echo "Python path: $PYTHONPATH"
check_python_version
check_flask
check_env_vars

if ! wait_for_ai_server; then
    echo "Error: Process completed with exit code 1."
    exit 1
fi

# Start the Flask application
echo "Starting Flask application..."
exec gunicorn --bind 0.0.0.0:5000 --workers 1 --log-level debug \
    --error-logfile /app/logs/gunicorn-error.log \
    --access-logfile /app/logs/gunicorn-access.log \
    app:app 