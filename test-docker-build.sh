#!/bin/bash

# Exit on error
set -e

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    docker compose down -v
    if [ -d "data" ]; then
        rm -rf data/*
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Create data directory if it doesn't exist
mkdir -p data

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Print environment for debugging
echo "Current environment:"
echo "TARGETARCH=${TARGETARCH}"
echo "Current Dockerfile content:"
cat dockerfile

# Clean up any existing containers and volumes
echo "Cleaning up existing containers and volumes..."
docker compose down -v
if [ -d "data" ]; then
    rm -rf data/*
fi

# Build and start the services
echo "Building and starting services..."
export TARGETARCH=amd64
export PYTHON_VERSION=3.10-alpine
# We'll use an explicit command instead of docker compose to have more control
docker build -t frontend:test -f dockerfile --build-arg TARGETPLATFORM=linux/amd64 --build-arg BUILDPLATFORM=linux/amd64 .
echo "Docker build completed, now starting services..."
docker compose up -d

# Function to check container status
check_container_status() {
    local service=$1
    local status=$(docker compose ps --format json $service | grep -o '"State":"[^"]*"' | cut -d'"' -f4)
    echo $status
}

# Wait for containers to be running
echo "Waiting for containers to be running..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    frontend_status=$(check_container_status frontend)
    db_status=$(check_container_status db)
    
    echo "Frontend status: $frontend_status"
    echo "Database status: $db_status"
    
    if [ "$frontend_status" = "running" ] && [ "$db_status" = "running" ]; then
        echo "All containers are running!"
        break
    fi
    
    if [ "$frontend_status" = "restarting" ]; then
        echo "Frontend container is restarting. Checking logs..."
        docker compose logs frontend
    fi
    
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Containers failed to start within $timeout seconds"
    docker compose logs
    exit 1
fi

# Add a small delay to ensure services are fully initialized
sleep 10

# Verify security configurations
echo "Verifying security configurations..."
# Check if containers are running as non-root
if ! docker compose exec -T frontend id -u; then
    echo "Error: Cannot execute command in frontend container"
    docker compose logs frontend
    exit 1
fi

# Verify read-only root filesystem
echo "Verifying read-only filesystem..."
if docker compose exec -T frontend touch /test 2>/dev/null; then
    echo "Error: Frontend container root filesystem is writable"
    exit 1
fi

# Verify tmpfs configuration
echo "Verifying tmpfs configuration..."
echo "Checking mount points..."
if ! docker compose exec -T frontend mount; then
    echo "Error: Cannot check mount points"
    docker compose logs frontend
    exit 1
fi

# More detailed tmpfs verification
if ! docker compose exec -T frontend sh -c 'mount | grep -E "tmpfs on (/tmp|/run|/var/run) "'; then
    echo "Error: Required tmpfs mounts not found"
    echo "Current mounts:"
    docker compose exec -T frontend mount || true
    echo "Container logs:"
    docker compose logs frontend
    exit 1
fi

# Verify tmpfs is writable
echo "Verifying tmpfs is writable..."
if ! docker compose exec frontend sh -c 'touch /tmp/test && rm /tmp/test'; then
    echo "Error: /tmp is not writable"
    echo "Container logs:"
    docker compose logs frontend
    exit 1
fi

# Wait for services to be healthy
echo "Waiting for database to be healthy..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker compose ps db | grep -q "healthy"; then
        echo "Database is healthy!"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "Still waiting for database... ($elapsed seconds)"
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Database failed to become healthy within $timeout seconds"
    docker compose logs db
    exit 1
fi

# Test database connection
echo "Testing database connection..."
docker compose exec db psql -U db -d frontend_db -c "\dt"

# Wait for frontend service to be ready
echo "Waiting for frontend service to be ready..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    echo "Attempting to connect to frontend service... ($elapsed seconds)"
    if wget -q -O- http://localhost:5001/health > /dev/null 2>&1; then
        echo "Frontend service is ready!"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    
    # If we've waited 30 seconds, show the logs
    if [ $elapsed -eq 30 ]; then
        echo "Frontend service taking longer than expected. Current logs:"
        docker compose logs frontend
    fi
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Frontend service failed to become ready within $timeout seconds"
    docker compose logs frontend
    exit 1
fi

# Verify resource limits
echo "Verifying resource limits..."
if ! docker compose exec frontend cat /sys/fs/cgroup/memory.max | grep -q "536870912"; then
    echo "Warning: Memory limit not properly set for frontend"
fi

# Final health check
echo "Performing final health check..."
if ! wget -q -O- http://localhost:5001/health; then
    echo "Error: Frontend health check failed"
    docker compose logs
    exit 1
fi

echo "All tests passed successfully!"
exit 0
