#!/bin/bash
# Production deployment script
# This script handles post-deployment tasks

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration - adjust as needed
COMPOSE_FILE="docker-compose.prod.yml"
FRONTEND_CONTAINER="frontend"
DB_CONTAINER="db"
MAX_WAIT=60  # Maximum wait time in seconds

echo "===== Starting production deployment ====="
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Deployment started at: $timestamp"

# Function to check if containers are running
check_container_status() {
    container_name=$1
    if [ "$(docker compose -f $COMPOSE_FILE ps -q $container_name 2>/dev/null)" ]; then
        if [ "$(docker compose -f $COMPOSE_FILE ps -a --filter "status=running" --format "{{.Name}}" | grep $container_name)" ]; then
            return 0  # Container is running
        fi
    fi
    return 1  # Container is not running
}

# Check if containers are running
if ! check_container_status $FRONTEND_CONTAINER || ! check_container_status $DB_CONTAINER; then
    echo "Starting or restarting containers..."
    docker compose -f $COMPOSE_FILE up -d
    
    # Wait for containers to be ready
    echo "Waiting for containers to be ready..."
    for i in $(seq 1 $MAX_WAIT); do
        if check_container_status $FRONTEND_CONTAINER && check_container_status $DB_CONTAINER; then
            echo "Containers are ready!"
            break
        fi
        if [ $i -eq $MAX_WAIT ]; then
            echo "Error: Containers failed to start within $MAX_WAIT seconds."
            exit 1
        fi
        echo "Waiting for containers... ($i/$MAX_WAIT)"
        sleep 1
    done
else
    echo "Containers are already running."
fi

# Wait for database to be fully initialized
echo "Waiting for database to be ready..."
for i in $(seq 1 $MAX_WAIT); do
    if docker compose -f $COMPOSE_FILE exec -T $DB_CONTAINER pg_isready -U db -d frontend_db > /dev/null 2>&1; then
        echo "Database is ready!"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "Error: Database failed to become ready within $MAX_WAIT seconds."
        exit 1
    fi
    echo "Waiting for database... ($i/$MAX_WAIT)"
    sleep 1
done

# Create admin user
echo "Ensuring admin user exists..."
docker compose -f $COMPOSE_FILE exec -T $FRONTEND_CONTAINER flask create-admin

# Run any pending migrations (if needed)
echo "Running any pending migrations..."
# Add migration commands here if needed

# Verify deployment
echo "Verifying deployment..."
if docker compose -f $COMPOSE_FILE ps | grep -q "Up" && \
   docker compose -f $COMPOSE_FILE exec -T $FRONTEND_CONTAINER python -c "import sys; sys.exit(0)"; then
    echo "Deployment verification successful!"
else
    echo "Warning: Deployment verification failed. Please check container logs."
    docker compose -f $COMPOSE_FILE logs --tail=50
    exit 1
fi

echo "===== Post-deployment tasks completed successfully ====="
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Deployment completed at: $timestamp"
exit 0 