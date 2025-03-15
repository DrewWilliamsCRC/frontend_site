#!/bin/bash
# Script to restart containers with .env file mounted and pull latest images

# Stop containers
docker compose -f docker-compose.prod.yml down

# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Start containers with updated compose file
docker compose -f docker-compose.prod.yml up -d

echo "Containers restarted with latest images. The .env file should now be properly mounted."
echo "Check the logs with: docker compose -f docker-compose.prod.yml logs frontend" 