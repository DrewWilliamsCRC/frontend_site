#!/bin/bash
# Script to restart containers with .env file mounted

# Stop containers
docker compose -f docker-compose.prod.yml down

# Start containers with updated compose file
docker compose -f docker-compose.prod.yml up -d

echo "Containers restarted. The .env file should now be properly mounted."
echo "Check the logs with: docker compose -f docker-compose.prod.yml logs frontend" 