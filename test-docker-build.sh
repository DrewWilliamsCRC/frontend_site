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

# Clean up any existing containers and volumes
echo "Cleaning up existing containers and volumes..."
docker compose down -v
if [ -d "data" ]; then
    rm -rf data/*
fi

# Build and start the services
echo "Building and starting services..."
docker compose build
docker compose up -d

# Verify security configurations
echo "Verifying security configurations..."
# Check if containers are running as non-root
if [ "$(docker compose exec frontend id -u)" = "0" ]; then
    echo "Error: Frontend container is running as root"
    exit 1
fi
if [ "$(docker compose exec db id -u)" = "0" ]; then
    echo "Error: Database container is running as root"
    exit 1
fi

# Verify read-only root filesystem
echo "Verifying read-only filesystem..."
if docker compose exec frontend touch /test 2>/dev/null; then
    echo "Error: Frontend container root filesystem is writable"
    exit 1
fi

# Verify tmpfs
echo "Verifying tmpfs configuration..."
if ! docker compose exec frontend mount | grep -q "/tmp.*tmpfs"; then
    echo "Error: tmpfs not properly configured for frontend"
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
    if curl -s -f http://localhost:5001/health > /dev/null 2>&1; then
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
if ! curl -v http://localhost:5001/health; then
    echo "Error: Frontend health check failed"
    docker compose logs
    exit 1
fi

echo "All tests passed successfully!"
exit 0
